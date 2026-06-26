#!/usr/bin/env python3
"""mod_dmcg: Gene-level differential modification calling between two conditions.

Compares modification levels between two experimental conditions at the
gene level using Fisher's exact test.  Takes gene-level site summaries
from ``mod_gene`` as input — no HDF5 or read-level data required.

Two tests are performed per matched gene-position:

* **Weighted** — ``wt_modified`` and ``wt_unmodified`` are rounded to
  the nearest integer before constructing the 2×2 contingency table.
* **Unweighted** — the raw ``n_modified`` and ``n_unmodified`` integer
  counts are used directly.

Matching key: ``(gene_id, chrom, strand, gpos, mod_type)``.
"""

import argparse
import sys
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import fisher_exact

try:
    from isolens._stats import bh_fdr
except ImportError:
    from _stats import bh_fdr  # type: ignore[no-redef]

# ---------- constants ----------

_OUTPUT_COLS = [
    "gene_id",
    "chrom",
    "strand",
    "gpos",
    "mod_type",
    "n_modified_1",
    "n_unmodified_1",
    "n_modified_2",
    "n_unmodified_2",
    "wt_modified_1",
    "wt_unmodified_1",
    "wt_modified_2",
    "wt_unmodified_2",
    "mod_level_1",
    "mod_level_2",
    "wt_mod_level_1",
    "wt_mod_level_2",
    "delta_mod_level",
    "delta_wt_mod_level",
    "log2_or",
    "p_value",
    "q_value",
    "w_log2_or",
    "w_p_value",
    "w_q_value",
]

_TSV_HEADER = "\t".join(_OUTPUT_COLS)

# Columns we read from the gene-level site summary
_SITE_COLS = [
    "gene_id",
    "chrom",
    "strand",
    "gpos",
    "mod_type",
    "n_modified",
    "wt_modified",
    "n_unmodified",
    "wt_unmodified",
    "mod_level",
    "wt_mod_level",
]

_SiteKey = tuple[str, str, str, int, str]
# (gene_id, chrom, strand, gpos, mod_type)


# ---------- CLI ----------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for mod_dmcg."""
    parser = argparse.ArgumentParser(
        description="mod_dmcg: Gene-level differential modification "
        "calling between two conditions"
    )
    parser.add_argument(
        "--sites-1",
        required=True,
        metavar="FILE",
        help="Gene-level site summary for condition 1 "
        "(Parquet or TSV/TSV.GZ from mod_gene)",
    )
    parser.add_argument(
        "--sites-2",
        required=True,
        metavar="FILE",
        help="Gene-level site summary for condition 2 "
        "(Parquet or TSV/TSV.GZ from mod_gene)",
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Output file path"
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=["parquet", "tsv"],
        default="parquet",
        help="Output format: parquet (default) or tsv",
    )
    parser.add_argument(
        "-z",
        "--gzip",
        action="store_true",
        help="Gzip-compress TSV output (ignored for parquet)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print progress to stderr",
    )
    return parser.parse_args()


# ---------- site-summary reader ----------


def _read_sites_parquet(path: str) -> dict[_SiteKey, dict[str, Any]]:
    """Read a Parquet gene-level site summary into a flat dict."""
    table = pq.read_table(path, columns=_SITE_COLS)
    sites: dict[_SiteKey, dict[str, Any]] = {}
    col_data = {c: table.column(c) for c in _SITE_COLS}
    for i in range(len(table)):
        gene_id = col_data["gene_id"][i].as_py()
        chrom = col_data["chrom"][i].as_py()
        strand = col_data["strand"][i].as_py()
        gpos = col_data["gpos"][i].as_py()
        mod_type = col_data["mod_type"][i].as_py()
        key: _SiteKey = (gene_id, chrom, strand, gpos, mod_type)
        row: dict[str, Any] = {}
        for c in _SITE_COLS:
            row[c] = col_data[c][i].as_py()
        sites[key] = row
    return sites


def _read_sites_tsv(path: str) -> dict[_SiteKey, dict[str, Any]]:
    """Read a TSV/TSV.GZ gene-level site summary into a flat dict."""
    import gzip

    open_func = gzip.open if path.endswith(".gz") else open
    mode = "rt" if path.endswith(".gz") else "r"
    sites: dict[_SiteKey, dict[str, Any]] = {}
    with open_func(path, mode, encoding="utf-8") as fh:
        header = fh.readline().strip().split("\t")
        col_idx = {c: header.index(c) for c in _SITE_COLS if c in header}
        for line in fh:
            parts = line.strip().split("\t")
            if not parts:
                continue
            gene_id = parts[col_idx["gene_id"]]
            chrom = parts[col_idx["chrom"]]
            strand = parts[col_idx["strand"]]
            gpos = int(parts[col_idx["gpos"]])
            mod_type = parts[col_idx["mod_type"]]
            key: _SiteKey = (gene_id, chrom, strand, gpos, mod_type)
            row: dict[str, Any] = {}
            for c in _SITE_COLS:
                if c not in col_idx:
                    row[c] = None
                    continue
                val = parts[col_idx[c]]
                if val == "NA":
                    row[c] = None
                elif c in ("n_modified", "n_unmodified", "gpos"):
                    row[c] = int(val)
                elif c in (
                    "wt_modified", "wt_unmodified",
                    "mod_level", "wt_mod_level",
                ):
                    row[c] = float(val)
                else:
                    row[c] = val
            sites[key] = row
    return sites


def read_gene_summary(path: str) -> dict[_SiteKey, dict[str, Any]]:
    """Read a gene-level site summary file (Parquet or TSV/TSV.GZ).

    Returns a dict keyed by ``(gene_id, chrom, strand, gpos, mod_type)``.
    """
    if path.endswith(".parquet"):
        return _read_sites_parquet(path)
    return _read_sites_tsv(path)


# ---------- statistical testing ----------


def _fisher_test(
    n1_mod: int, n1_unmod: int,
    n2_mod: int, n2_unmod: int,
) -> dict[str, float]:
    """Run Fisher's exact test on a 2×2 contingency table.

    Returns ``{"log2_or": float, "p_value": float}``.
    Returns ``{"log2_or": nan, "p_value": nan}`` when the table is
    degenerate (all row or column marginals are zero).
    """
    total = n1_mod + n1_unmod + n2_mod + n2_unmod
    if total == 0:
        return {"log2_or": float("nan"), "p_value": float("nan")}
    row1 = n1_mod + n1_unmod
    row2 = n2_mod + n2_unmod
    col1 = n1_mod + n2_mod
    col2 = n1_unmod + n2_unmod
    if row1 == 0 or row2 == 0 or col1 == 0 or col2 == 0:
        return {"log2_or": float("nan"), "p_value": float("nan")}
    odds, p_val = fisher_exact(
        [[n1_mod, n1_unmod], [n2_mod, n2_unmod]]
    )
    log2_or = float(np.log2(odds)) if odds > 0 else float("-inf")
    return {"log2_or": log2_or, "p_value": float(p_val)}


# ---------- per-matched-site processing ----------


def process_matched_sites(
    sites_1: dict[_SiteKey, dict[str, Any]],
    sites_2: dict[_SiteKey, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compare matched gene-level sites between two conditions.

    Parameters
    ----------
    sites_1, sites_2 : dict
        Gene-level site summaries keyed by
        ``(gene_id, chrom, strand, gpos, mod_type)``.

    Returns
    -------
    list of dict
        One dict per matched site with all ``_OUTPUT_COLS`` fields.
        ``q_value`` and ``w_q_value`` are set to 0.0 — the caller
        must apply BH FDR correction afterward.
    """
    common_keys = sorted(sites_1.keys() & sites_2.keys())
    rows: list[dict[str, Any]] = []

    for key in common_keys:
        s1 = sites_1[key]
        s2 = sites_2[key]

        n_mod_1 = s1.get("n_modified", 0)
        n_unmod_1 = s1.get("n_unmodified", 0)
        n_mod_2 = s2.get("n_modified", 0)
        n_unmod_2 = s2.get("n_unmodified", 0)

        wt_mod_1 = s1.get("wt_modified", 0.0)
        wt_unmod_1 = s1.get("wt_unmodified", 0.0)
        wt_mod_2 = s2.get("wt_modified", 0.0)
        wt_unmod_2 = s2.get("wt_unmodified", 0.0)

        # Skip if both conditions have zero total counts for this site
        if (n_mod_1 + n_unmod_1 == 0) or (n_mod_2 + n_unmod_2 == 0):
            continue

        # Unweighted Fisher test
        unweighted = _fisher_test(
            int(n_mod_1), int(n_unmod_1),
            int(n_mod_2), int(n_unmod_2),
        )

        # Weighted Fisher test (round to nearest integer)
        r_mod_1 = int(round(wt_mod_1))
        r_unmod_1 = int(round(wt_unmod_1))
        r_mod_2 = int(round(wt_mod_2))
        r_unmod_2 = int(round(wt_unmod_2))

        weighted = _fisher_test(
            r_mod_1, r_unmod_1,
            r_mod_2, r_unmod_2,
        )

        # Effect sizes from site summary
        ml1 = s1.get("mod_level")
        ml2 = s2.get("mod_level")
        wml1 = s1.get("wt_mod_level")
        wml2 = s2.get("wt_mod_level")

        rows.append({
            "gene_id": key[0],
            "chrom": key[1],
            "strand": key[2],
            "gpos": key[3],
            "mod_type": key[4],
            "n_modified_1": int(n_mod_1),
            "n_unmodified_1": int(n_unmod_1),
            "n_modified_2": int(n_mod_2),
            "n_unmodified_2": int(n_unmod_2),
            "wt_modified_1": wt_mod_1,
            "wt_unmodified_1": wt_unmod_1,
            "wt_modified_2": wt_mod_2,
            "wt_unmodified_2": wt_unmod_2,
            "mod_level_1": ml1,
            "mod_level_2": ml2,
            "wt_mod_level_1": wml1,
            "wt_mod_level_2": wml2,
            "delta_mod_level": (
                round(ml2 - ml1, 6)
                if ml1 is not None and ml2 is not None
                else None
            ),
            "delta_wt_mod_level": (
                round(wml2 - wml1, 6)
                if wml1 is not None and wml2 is not None
                else None
            ),
            "log2_or": unweighted["log2_or"],
            "p_value": unweighted["p_value"],
            "q_value": 0.0,  # filled after BH correction
            "w_log2_or": weighted["log2_or"],
            "w_p_value": weighted["p_value"],
            "w_q_value": 0.0,  # filled after BH correction
        })

    return rows


# ---------- output writers ----------


def _write_tsv(
    all_rows: list[dict[str, Any]], path: str, use_gzip: bool
) -> None:
    """Write rows as tab-separated values."""
    import gzip

    open_func = gzip.open if use_gzip else open
    mode = "wt" if use_gzip else "w"
    with open_func(path, mode, encoding="utf-8") as fh:
        fh.write(_TSV_HEADER + "\n")
        for row in all_rows:
            fh.write(
                "\t".join(
                    "NA" if row[c] is None else str(row[c])
                    for c in _OUTPUT_COLS
                )
                + "\n"
            )


def _write_parquet(all_rows: list[dict[str, Any]], path: str) -> None:
    """Write rows as a Parquet file via pyarrow."""
    if not all_rows:
        schema = pa.schema([
            ("gene_id", pa.string()),
            ("chrom", pa.string()),
            ("strand", pa.string()),
            ("gpos", pa.int32()),
            ("mod_type", pa.string()),
            ("n_modified_1", pa.int32()),
            ("n_unmodified_1", pa.int32()),
            ("n_modified_2", pa.int32()),
            ("n_unmodified_2", pa.int32()),
            ("wt_modified_1", pa.float64()),
            ("wt_unmodified_1", pa.float64()),
            ("wt_modified_2", pa.float64()),
            ("wt_unmodified_2", pa.float64()),
            ("mod_level_1", pa.float64()),
            ("mod_level_2", pa.float64()),
            ("wt_mod_level_1", pa.float64()),
            ("wt_mod_level_2", pa.float64()),
            ("delta_mod_level", pa.float64()),
            ("delta_wt_mod_level", pa.float64()),
            ("log2_or", pa.float64()),
            ("p_value", pa.float64()),
            ("q_value", pa.float64()),
            ("w_log2_or", pa.float64()),
            ("w_p_value", pa.float64()),
            ("w_q_value", pa.float64()),
        ])
        with pq.ParquetWriter(path, schema) as w:
            w.write_table(
                pa.table({
                    k: pa.array([], type=schema.field(k).type)
                    for k in schema.names
                })
            )
        return

    arrays: dict[str, pa.Array] = {}
    for col in _OUTPUT_COLS:
        values = [r[col] for r in all_rows]
        if col in ("gene_id", "chrom", "strand", "mod_type"):
            arrays[col] = pa.array(values)
        elif col == "gpos":
            arrays[col] = pa.array(values, type=pa.int32())
        elif col.startswith("n_"):
            arrays[col] = pa.array(values, type=pa.int32())
        else:
            arrays[col] = pa.array(values, type=pa.float64())
    pq.write_table(pa.table(arrays), path)


# ---------- main ----------


def main(args: argparse.Namespace | None = None) -> None:
    """Gene-level differential modification calling between two conditions.

    Reads gene-level site summaries for two conditions from ``mod_gene``,
    matches sites by ``(gene_id, chrom, strand, gpos, mod_type)``, runs
    Fisher's exact test (weighted and unweighted) per site, and writes
    results with global BH FDR correction.
    """
    if args is None:
        args = parse_args()

    # ---- 1. Read gene-level site summaries ----
    if args.verbose:
        print("[mod_dmcg] Reading gene summaries...", file=sys.stderr)
    sites_1 = read_gene_summary(args.sites_1)
    sites_2 = read_gene_summary(args.sites_2)

    if args.verbose:
        print(
            f"[mod_dmcg] Condition 1: {len(sites_1)} gene-sites",
            file=sys.stderr,
        )
        print(
            f"[mod_dmcg] Condition 2: {len(sites_2)} gene-sites",
            file=sys.stderr,
        )

    # ---- 2. Match sites and test ----
    common = set(sites_1.keys()) & set(sites_2.keys())
    only_1 = len(sites_1) - len(common)
    only_2 = len(sites_2) - len(common)

    if args.verbose:
        print(
            f"[mod_dmcg] {len(common)} gene-sites in common "
            f"(cond1 only: {only_1}, cond2 only: {only_2})",
            file=sys.stderr,
        )

    all_rows = process_matched_sites(sites_1, sites_2)

    # ---- 3. Global BH FDR correction ----
    if all_rows:
        # Unweighted p-values
        p_vals = [r["p_value"] for r in all_rows]
        q_vals = bh_fdr(p_vals)
        for r, qv in zip(all_rows, q_vals):
            r["q_value"] = round(qv, 6)

        # Weighted p-values
        w_p_vals = [r["w_p_value"] for r in all_rows]
        w_q_vals = bh_fdr(w_p_vals)
        for r, qv in zip(all_rows, w_q_vals):
            r["w_q_value"] = round(qv, 6)

    if args.verbose:
        n_tested = len(all_rows)
        n_sig_u = sum(
            1 for r in all_rows
            if not np.isnan(r["q_value"]) and r["q_value"] < 0.05
        )
        n_sig_w = sum(
            1 for r in all_rows
            if not np.isnan(r["w_q_value"]) and r["w_q_value"] < 0.05
        )
        print(
            f"[mod_dmcg] {n_tested} gene-sites tested "
            f"({n_sig_u} significant unweighted, "
            f"{n_sig_w} significant weighted at FDR<0.05)",
            file=sys.stderr,
        )

    # ---- 4. Write output ----
    if args.format == "tsv":
        _write_tsv(all_rows, args.output, args.gzip)
    else:
        _write_parquet(all_rows, args.output)

    if args.verbose:
        print(
            f"[mod_dmcg] Done. Output written to {args.output}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
