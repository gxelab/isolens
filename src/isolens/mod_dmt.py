#!/usr/bin/env python3
"""mod_dmt: Differential modification testing between transcript isoforms.

Compares modification levels between transcript isoforms that share a
genomic locus, using read-level weighted logistic regression.
Transcripts are grouped by ``(gene_id, chrom, gpos, strand, mod_type)``
and all isoform pairs within each group are tested.

Site grouping key: ``(gene_id, chrom, gpos, strand, mod_type)``.
"""

import argparse
import itertools
import sys
from contextlib import ExitStack
from typing import Any

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from isolens._stats import bh_fdr, weighted_logistic_test
    from isolens.mod_scan import (
        CODE_DELETION,
        CODE_FAIL,
        CODE_MISMATCH,
        CODE_UNCOVERED,
    )
except ImportError:
    from _stats import bh_fdr, weighted_logistic_test  # type: ignore[no-redef]
    from mod_scan import (  # type: ignore[no-redef]
        CODE_DELETION,
        CODE_FAIL,
        CODE_MISMATCH,
        CODE_UNCOVERED,
    )

# ---------- constants ----------

_OUTPUT_COLS = [
    "gene_id",
    "chrom",
    "gpos",
    "strand",
    "mod_type",
    "transcript_id_1",
    "transcript_id_2",
    "position_1",
    "position_2",
    "mod_level_1",
    "mod_level_2",
    "wt_mod_level_1",
    "wt_mod_level_2",
    "delta_mod_level",
    "delta_wt_mod_level",
    "n_modified_1",
    "n_unmodified_1",
    "n_modified_2",
    "n_unmodified_2",
    "wt_modified_1",
    "wt_unmodified_1",
    "wt_modified_2",
    "wt_unmodified_2",
    "log2_or",
    "p_value",
    "q_value",
]

_TSV_HEADER = "\t".join(_OUTPUT_COLS)

# Columns needed from the site summary
_SITE_COLS = [
    "transcript_id",
    "position",
    "mod_type",
    "n_modified",
    "wt_modified",
    "n_unmodified",
    "wt_unmodified",
    "mod_level",
    "wt_mod_level",
    "gene_id",
    "chrom",
    "strand",
    "gpos",
]

# Key for grouping by genomic locus
_LocusKey = tuple[str, str, int, str, str]
# (gene_id, chrom, gpos, strand, mod_type)


# ---------- CLI ----------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for mod_dmt."""
    parser = argparse.ArgumentParser(
        description="mod_dmt: Differential modification testing between "
        "transcript isoforms"
    )
    parser.add_argument(
        "-i",
        "--h5",
        required=True,
        nargs="+",
        metavar="H5",
        help="Input HDF5 file(s) from mod_scan.  When multiple files "
        "are provided, reads for the same transcript are pooled.",
    )
    parser.add_argument(
        "-s",
        "--sites",
        required=True,
        metavar="FILE",
        help="Pooled site summary from mod_sites (Parquet or TSV/TSV.GZ).  "
        "Must include genomic coordinate columns (run mod_sites with --gtf).",
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
        "-p",
        "--min-asp",
        type=float,
        default=0.0,
        help="Minimum Oarfish assignment probability for a read to be "
        "included [default: 0.0 (no filter)]",
    )
    parser.add_argument(
        "-x",
        "--transcripts",
        nargs="+",
        default=None,
        metavar="TX",
        help="Only consider the specified transcript ID(s).  "
        "[default: all transcripts in the HDF5]",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print progress to stderr",
    )
    return parser.parse_args()


# ---------- site-summary reader ----------


def _read_sites_parquet(path: str) -> list[dict[str, Any]]:
    """Read a Parquet site summary into a list of row dicts."""
    table = pq.read_table(path, columns=_SITE_COLS)
    rows: list[dict[str, Any]] = []
    col_data = {c: table.column(c) for c in _SITE_COLS}
    for i in range(len(table)):
        row: dict[str, Any] = {}
        for c in _SITE_COLS:
            row[c] = col_data[c][i].as_py()
        rows.append(row)
    return rows


def _read_sites_tsv(path: str) -> list[dict[str, Any]]:
    """Read a TSV/TSV.GZ site summary into a list of row dicts."""
    import gzip

    open_func = gzip.open if path.endswith(".gz") else open
    mode = "rt" if path.endswith(".gz") else "r"
    rows: list[dict[str, Any]] = []
    with open_func(path, mode, encoding="utf-8") as fh:
        header = fh.readline().strip().split("\t")
        col_idx = {c: header.index(c) for c in _SITE_COLS if c in header}
        for line in fh:
            parts = line.strip().split("\t")
            if not parts:
                continue
            row: dict[str, Any] = {}
            for c in _SITE_COLS:
                if c not in col_idx:
                    row[c] = None
                    continue
                val = parts[col_idx[c]]
                if val == "NA":
                    row[c] = None
                elif c in ("n_modified", "n_unmodified", "position"):
                    row[c] = int(val)
                elif c in (
                    "wt_modified", "wt_unmodified",
                    "mod_level", "wt_mod_level",
                ):
                    row[c] = float(val)
                elif c == "gpos":
                    row[c] = int(val) if val != "NA" else None
                else:
                    row[c] = val
            rows.append(row)
    return rows


def read_sites_grouped_by_locus(
    path: str,
) -> dict[_LocusKey, list[dict[str, Any]]]:
    """Read a site summary and group rows by genomic locus.

    Rows with a null ``gene_id`` or ``gpos`` are dropped (no GTF data).
    Only groups with at least two distinct transcripts are kept.

    Returns
    -------
    dict
        ``{(gene_id, chrom, gpos, strand, mod_type): [row_dict, ...]}``
    """
    if path.endswith(".parquet"):
        rows = _read_sites_parquet(path)
    else:
        rows = _read_sites_tsv(path)

    # Drop rows without genomic coordinates
    rows = [
        r for r in rows
        if r.get("gene_id") is not None and r.get("gpos") is not None
    ]

    # Group by locus key
    groups: dict[_LocusKey, list[dict[str, Any]]] = {}
    for row in rows:
        key: _LocusKey = (
            row["gene_id"],
            row["chrom"],
            row["gpos"],
            row["strand"],
            row["mod_type"],
        )
        groups.setdefault(key, []).append(row)

    # Keep only groups with ≥ 2 distinct transcripts
    filtered: dict[_LocusKey, list[dict[str, Any]]] = {}
    for key, group_rows in groups.items():
        unique_tx = {r["transcript_id"] for r in group_rows}
        if len(unique_tx) >= 2:
            filtered[key] = group_rows

    return filtered


def validate_input(groups: dict[_LocusKey, list[dict[str, Any]]]) -> None:
    """Exit with an error if the input contains no usable locus groups.

    This typically means the site summary was generated without ``--gtf``.
    """
    if not groups:
        print(
            "[mod_dmt] Error: No locus groups with ≥ 2 transcripts found. "
            "Ensure the site summary was generated with mod_sites --gtf "
            "so that gene_id, chrom, gpos, and strand columns are present.",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------- per-site read extraction ----------


def _extract_site_reads(
    matrix: np.ndarray,
    weights: np.ndarray,
    position_1b: int,
    mod_code: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract binary modified/unmodified vector and weights for a site.

    Filters out uncovered, mismatch, deletion, failed, and other-mod
    reads so that only *modified* (this mod type) and *unmodified*
    (canonical) reads remain.
    """
    col = matrix[:, position_1b - 1]
    valid = (
        (col != CODE_UNCOVERED)
        & (col != CODE_MISMATCH)
        & (col != CODE_DELETION)
        & (col != CODE_FAIL)
    )
    other_mod = (col >= 4) & (col != mod_code) & (col != CODE_FAIL)
    valid = valid & (~other_mod)
    if valid.sum() == 0:
        return None
    y = (col[valid] == mod_code).astype(np.float64)
    w = weights[valid].astype(np.float64)
    return y, w


# ---------- HDF5 helpers ----------


def _read_mod_codes(h5: h5py.File) -> dict[str, int]:
    """Read modification codes from an open HDF5 file."""
    return {
        mod_str: int(code)
        for mod_str, code in h5["modification_codes"].attrs.items()
    }


def _validate_mod_codes(
    mod_maps: list[dict[str, int]],
    filenames: list[str],
) -> dict[str, int]:
    """Verify all HDF5 files have identical modification codes."""
    reference = mod_maps[0]
    for i, code_map in enumerate(mod_maps[1:], start=1):
        if code_map != reference:
            ref_str = "; ".join(
                f"{k}={v}" for k, v in sorted(reference.items())
            )
            file_str = "; ".join(
                f"{k}={v}" for k, v in sorted(code_map.items())
            )
            raise ValueError(
                f"Modification codes in {filenames[i]} do not match "
                f"{filenames[0]}.\n"
                f"  {filenames[0]}: {ref_str}\n"
                f"  {filenames[i]}: {file_str}"
            )
    return reference


def _load_transcript_data(
    h5: h5py.File,
    tx_name: str,
    min_asp: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load matrix and weights for one transcript from one HDF5 file."""
    if tx_name not in h5["transcripts"]:
        return None
    grp = h5[f"transcripts/{tx_name}"]
    matrix = grp["matrix"][:]
    weights = grp["read_weights"][:]
    if min_asp > 0.0:
        mask = weights >= min_asp
        if mask.sum() == 0:
            return None
        matrix = matrix[mask]
        weights = weights[mask]
    return matrix, weights


def _validate_tx_lengths(
    tx_name: str,
    lengths: list[int | None],
    filenames: list[str],
) -> int:
    """Validate a transcript has consistent length across files."""
    ref_length = next(ln for ln in lengths if ln is not None)
    for i, (length, fname) in enumerate(zip(lengths, filenames)):
        if length is not None and length != ref_length:
            raise ValueError(
                f"Transcript '{tx_name}' has inconsistent lengths across "
                f"input files: {ref_length} in first file vs {length} in "
                f"{fname}. All files must use the same transcriptome "
                f"reference."
            )
    return ref_length


def _load_all_transcripts(
    h5_files: list[h5py.File],
    tx_names: set[str],
    min_asp: float,
    verbose: bool = False,
) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    """Pre-load pooled transcript data for all needed transcripts.

    Loads and pools matrices and weights across multiple HDF5 files for
    every transcript in *tx_names*.

    Returns
    -------
    dict
        ``{tx_name: (matrix, weights, tx_length)}``.
        Transcripts with zero reads after filtering are excluded.
    """
    h5_data: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    loaded = 0

    for tx_name in sorted(tx_names):
        matrices: list[np.ndarray] = []
        wlists: list[np.ndarray] = []
        tx_lengths_found: list[int | None] = []

        for h5 in h5_files:
            result = _load_transcript_data(h5, tx_name, min_asp)
            if result is not None:
                m, w = result
                matrices.append(m)
                wlists.append(w)
                tx_lengths_found.append(m.shape[1])
            else:
                tx_lengths_found.append(None)

        if not matrices:
            continue

        _validate_tx_lengths(
            tx_name, tx_lengths_found,
            [f.filename for f in h5_files],
        )

        if len(matrices) == 1:
            h5_data[tx_name] = (matrices[0], wlists[0], matrices[0].shape[1])
        else:
            h5_data[tx_name] = (
                np.vstack(matrices),
                np.concatenate(wlists),
                matrices[0].shape[1],
            )

        loaded += 1
        if verbose and loaded % 1000 == 0:
            print(
                f"[mod_dmt] Pre-loaded {loaded} transcripts...",
                file=sys.stderr,
            )

    if verbose:
        print(
            f"[mod_dmt] Pre-loaded {len(h5_data)} transcripts",
            file=sys.stderr,
        )
    return h5_data


# ---------- per-locus processing ----------


def process_locus_group(
    group_key: _LocusKey,
    tx_site_list: list[dict[str, Any]],
    h5_data: dict[str, tuple[np.ndarray, np.ndarray, int]],
    mod_code_map: dict[str, int],
) -> list[dict[str, Any]]:
    """Process one genomic locus group for DMT.

    Enumerates all isoform pairs and fits a weighted logistic regression
    per pair.

    Parameters
    ----------
    group_key : tuple
        ``(gene_id, chrom, gpos, strand, mod_type)``.
    tx_site_list : list of dict
        Site-summary rows for transcripts at this locus.
    h5_data : dict
        Pre-loaded ``{tx_name: (matrix, weights, tx_length)}`` mapping.
    mod_code_map : dict
        ``{mod_type_str: integer_code}``.

    Returns
    -------
    list of dict
        One dict per isoform pair with all ``_OUTPUT_COLS`` fields.
    """
    gene_id, chrom, gpos, strand, mod_type_str = group_key
    mod_code = mod_code_map.get(mod_type_str)
    if mod_code is None:
        return []

    # Filter to transcripts present in HDF5
    available = [
        s for s in tx_site_list
        if s["transcript_id"] in h5_data
    ]
    if len(available) < 2:
        return []

    rows: list[dict[str, Any]] = []

    for site_a, site_b in itertools.combinations(available, 2):
        tx_a = site_a["transcript_id"]
        tx_b = site_b["transcript_id"]
        pos_a = site_a["position"]
        pos_b = site_b["position"]

        matrix_a, weights_a, _len_a = h5_data[tx_a]
        matrix_b, weights_b, _len_b = h5_data[tx_b]

        reads_a = _extract_site_reads(matrix_a, weights_a, pos_a, mod_code)
        reads_b = _extract_site_reads(matrix_b, weights_b, pos_b, mod_code)

        if reads_a is None or reads_b is None:
            continue

        y_a, w_a = reads_a
        y_b, w_b = reads_b

        if len(y_a) == 0 or len(y_b) == 0:
            continue

        # Build design: transcript A = 0, transcript B = 1
        y = np.concatenate([y_a, y_b])
        x = np.concatenate([
            np.zeros(len(y_a), dtype=np.float64),
            np.ones(len(y_b), dtype=np.float64),
        ])
        w = np.concatenate([w_a, w_b])

        result = weighted_logistic_test(y, x, w)

        ml_a = site_a.get("mod_level")
        ml_b = site_b.get("mod_level")
        wml_a = site_a.get("wt_mod_level")
        wml_b = site_b.get("wt_mod_level")

        rows.append({
            "gene_id": gene_id,
            "chrom": chrom,
            "gpos": gpos,
            "strand": strand,
            "mod_type": mod_type_str,
            "transcript_id_1": tx_a,
            "transcript_id_2": tx_b,
            "position_1": pos_a,
            "position_2": pos_b,
            "mod_level_1": ml_a,
            "mod_level_2": ml_b,
            "wt_mod_level_1": wml_a,
            "wt_mod_level_2": wml_b,
            "delta_mod_level": (
                round(ml_b - ml_a, 6)
                if ml_a is not None and ml_b is not None
                else None
            ),
            "delta_wt_mod_level": (
                round(wml_b - wml_a, 6)
                if wml_a is not None and wml_b is not None
                else None
            ),
            "n_modified_1": site_a.get("n_modified"),
            "n_unmodified_1": site_a.get("n_unmodified"),
            "n_modified_2": site_b.get("n_modified"),
            "n_unmodified_2": site_b.get("n_unmodified"),
            "wt_modified_1": site_a.get("wt_modified"),
            "wt_unmodified_1": site_a.get("wt_unmodified"),
            "wt_modified_2": site_b.get("wt_modified"),
            "wt_unmodified_2": site_b.get("wt_unmodified"),
            "log2_or": result["log2_or"],
            "p_value": result["p_value"],
            "q_value": 0.0,  # filled after global BH correction
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
            ("gpos", pa.int32()),
            ("strand", pa.string()),
            ("mod_type", pa.string()),
            ("transcript_id_1", pa.string()),
            ("transcript_id_2", pa.string()),
            ("position_1", pa.int32()),
            ("position_2", pa.int32()),
            ("mod_level_1", pa.float64()),
            ("mod_level_2", pa.float64()),
            ("wt_mod_level_1", pa.float64()),
            ("wt_mod_level_2", pa.float64()),
            ("delta_mod_level", pa.float64()),
            ("delta_wt_mod_level", pa.float64()),
            ("n_modified_1", pa.int32()),
            ("n_unmodified_1", pa.int32()),
            ("n_modified_2", pa.int32()),
            ("n_unmodified_2", pa.int32()),
            ("wt_modified_1", pa.float64()),
            ("wt_unmodified_1", pa.float64()),
            ("wt_modified_2", pa.float64()),
            ("wt_unmodified_2", pa.float64()),
            ("log2_or", pa.float64()),
            ("p_value", pa.float64()),
            ("q_value", pa.float64()),
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
        if col in (
            "gene_id", "chrom", "strand", "mod_type",
            "transcript_id_1", "transcript_id_2",
        ):
            arrays[col] = pa.array(values)
        elif col in ("gpos", "position_1", "position_2"):
            arrays[col] = pa.array(values, type=pa.int32())
        elif col.startswith("n_"):
            arrays[col] = pa.array(values, type=pa.int32())
        else:
            arrays[col] = pa.array(values, type=pa.float64())
    pq.write_table(pa.table(arrays), path)


# ---------- main ----------


def main(args: argparse.Namespace | None = None) -> None:
    """Differential modification testing between transcript isoforms.

    Reads a pooled set of HDF5 files and a site summary (with genomic
    coordinates), groups sites by genomic locus, tests all isoform pairs
    within each group via weighted logistic regression, and writes
    results with global BH FDR correction.
    """
    if args is None:
        args = parse_args()

    # ---- 1. Read and group site summary ----
    if args.verbose:
        print("[mod_dmt] Reading site summary...", file=sys.stderr)
    locus_groups = read_sites_grouped_by_locus(args.sites)
    validate_input(locus_groups)

    if args.verbose:
        n_sites = sum(len(v) for v in locus_groups.values())
        print(
            f"[mod_dmt] {len(locus_groups)} locus groups, "
            f"{n_sites} sites",
            file=sys.stderr,
        )

    # ---- 2. Collect transcript names from locus groups ----
    site_tx_names: set[str] = set()
    for group_rows in locus_groups.values():
        for r in group_rows:
            site_tx_names.add(r["transcript_id"])

    # ---- 3. Open HDF5 files ----
    with ExitStack() as stack:
        h5_files = [
            stack.enter_context(h5py.File(f, "r")) for f in args.h5
        ]

        if args.verbose:
            print(
                f"[mod_dmt] Opened {len(h5_files)} HDF5 file(s)",
                file=sys.stderr,
            )

        # ---- 4. Validate modification codes ----
        all_mod_maps = [_read_mod_codes(h5) for h5 in h5_files]
        try:
            mod_code_map = _validate_mod_codes(
                all_mod_maps, list(args.h5)
            )
        except ValueError as exc:
            print(f"[mod_dmt] Error: {exc}", file=sys.stderr)
            sys.exit(1)

        if args.verbose:
            print(
                f"[mod_dmt] {len(mod_code_map)} modification types: "
                f"{sorted(mod_code_map.keys())}",
                file=sys.stderr,
            )

        # ---- 5. Determine transcript set ----
        h5_tx_union = set.union(
            *[set(h["transcripts"].keys()) for h in h5_files]
        )
        tx_to_load = h5_tx_union & site_tx_names

        if args.transcripts is not None:
            requested = set(args.transcripts)
            tx_to_load &= requested

        if args.verbose:
            print(
                f"[mod_dmt] {len(tx_to_load)} transcripts to load",
                file=sys.stderr,
            )

        # ---- 6. Pre-load transcript data ----
        h5_data = _load_all_transcripts(
            h5_files, tx_to_load, args.min_asp, args.verbose
        )

        # ---- 7. Process each locus group ----
        all_rows: list[dict[str, Any]] = []
        groups_processed = 0

        for group_key, group_rows in locus_groups.items():
            pair_rows = process_locus_group(
                group_key, group_rows, h5_data, mod_code_map
            )
            all_rows.extend(pair_rows)
            groups_processed += 1

            if args.verbose and groups_processed % 1000 == 0:
                print(
                    f"[mod_dmt] Processed {groups_processed}/"
                    f"{len(locus_groups)} locus groups...",
                    file=sys.stderr,
                )

    # ---- 8. Global BH FDR correction ----
    if all_rows:
        p_values = [r["p_value"] for r in all_rows]
        q_values = bh_fdr(p_values)
        for r, qv in zip(all_rows, q_values):
            r["q_value"] = round(qv, 6)

    if args.verbose:
        print(
            f"[mod_dmt] Total pairs tested: {len(all_rows)}",
            file=sys.stderr,
        )

    # ---- 9. Write output ----
    if args.format == "tsv":
        _write_tsv(all_rows, args.output, args.gzip)
    else:
        _write_parquet(all_rows, args.output)

    if args.verbose:
        print(
            f"[mod_dmt] Done. Output written to {args.output}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
