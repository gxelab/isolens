#!/usr/bin/env python3
"""mod_gene: Gene-level aggregation of modification site summaries.

Part of the isolens toolkit.  Reads a site summary file produced by
``mod_sites.py`` (which must have been run with ``--gtf`` to include
genomic coordinates) and aggregates per-transcript, per-position rows
into per-gene, per-genomic-position rows.

Aggregation logic
-----------------
Rows are grouped by ``(gene_id, chrom, strand, gpos, mod_type)``.
Within each group:

* All count columns (``n_modified``, ``n_unmodified``, …) are **summed**.
* All weighted-count columns (``wt_modified``, ``wt_unmodified``, …)
  are **summed**.
* ``mod_level`` is recomputed as
  ``n_modified / (n_modified + n_unmodified)`` (0 when the denominator
  is 0).
* ``wt_mod_level`` is recomputed analogously from the weighted columns.

Rows where ``gene_id`` or ``gpos`` is ``None``/``NA`` are silently
dropped (they cannot be assigned to a genomic position).

Input validation
----------------
Before processing, the module checks that the ``gene_id`` and ``chrom``
columns are present and contain at least one non-null value.  If not,
it terminates with a message instructing the user to re-run
``mod_sites`` with the ``-g/--gtf`` flag.
"""

import argparse
import sys
from collections import defaultdict
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

try:
    from isolens._io import write_parquet, write_tsv
except ImportError:
    from _io import write_parquet, write_tsv  # type: ignore[no-redef]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for mod_gene."""
    parser = argparse.ArgumentParser(
        description="mod_gene: Gene-level aggregation of modification site summaries"
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input site summary from mod_sites (Parquet or TSV/TSV.GZ). "
        "Must have been produced with --gtf to include gene_id, chrom, "
        "strand, and gpos columns.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output file path",
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


# ---------- input reading ----------

# Columns expected in the input site summary (mod_sites output).
_INPUT_COLS = [
    "transcript_id",
    "position",
    "mod_type",
    "n_modified",
    "wt_modified",
    "n_unmodified",
    "wt_unmodified",
    "n_canonical",
    "wt_canonical",
    "n_othermod",
    "wt_othermod",
    "n_mismatch",
    "wt_mismatch",
    "n_deletion",
    "wt_deletion",
    "n_failed",
    "wt_failed",
    "mod_level",
    "wt_mod_level",
    "gene_id",
    "chrom",
    "strand",
    "gpos",
]

# Columns in the gene-level output.
_OUTPUT_COLS = [
    "gene_id",
    "chrom",
    "strand",
    "gpos",
    "mod_type",
    "n_modified",
    "wt_modified",
    "n_unmodified",
    "wt_unmodified",
    "n_canonical",
    "wt_canonical",
    "n_othermod",
    "wt_othermod",
    "n_mismatch",
    "wt_mismatch",
    "n_deletion",
    "wt_deletion",
    "n_failed",
    "wt_failed",
    "mod_level",
    "wt_mod_level",
]

_OUTPUT_HEADER = "\t".join(_OUTPUT_COLS)

# Count columns that are summed (int).
_COUNT_COLS = [
    "n_modified",
    "n_unmodified",
    "n_canonical",
    "n_othermod",
    "n_mismatch",
    "n_deletion",
    "n_failed",
]

# Weighted count columns that are summed (float).
_WT_COLS = [
    "wt_modified",
    "wt_unmodified",
    "wt_canonical",
    "wt_othermod",
    "wt_mismatch",
    "wt_deletion",
    "wt_failed",
]


def read_input(path: str) -> list[dict[str, Any]]:
    """Read a site summary file (Parquet or TSV/TSV.GZ).

    Args:
        path: Path to a Parquet or TSV/TSV.GZ file from ``mod_sites``.

    Returns:
        List of row dicts with keys from ``_INPUT_COLS``.
    """
    if path.endswith(".parquet"):
        return _read_parquet(path)
    return _read_tsv(path)


def _read_parquet(path: str) -> list[dict[str, Any]]:
    """Read a Parquet site summary file."""
    table = pq.read_table(path)
    rows: list[dict[str, Any]] = []
    for i in range(len(table)):
        row: dict[str, Any] = {}
        for col in table.column_names:
            val = table.column(col)[i].as_py()
            row[col] = val
        rows.append(row)
    return rows


def _read_tsv(path: str) -> list[dict[str, Any]]:
    """Read a TSV (optionally gzipped) site summary file."""
    import gzip

    open_func = gzip.open if path.endswith(".gz") else open
    mode = "rt" if path.endswith(".gz") else "r"

    with open_func(path, mode, encoding="utf-8") as f:
        header = f.readline().strip().split("\t")

        # Map column name → index
        col_idx: dict[str, int] = {}
        for col in _INPUT_COLS:
            try:
                col_idx[col] = header.index(col)
            except ValueError:
                col_idx[col] = -1

        rows: list[dict[str, Any]] = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            row: dict[str, Any] = {}
            for col in _INPUT_COLS:
                idx = col_idx[col]
                if idx < 0 or idx >= len(parts):
                    row[col] = None
                    continue
                raw = parts[idx]
                if col == "transcript_id":
                    row[col] = raw
                elif col == "mod_type":
                    row[col] = raw
                elif col in ("gene_id", "chrom", "strand"):
                    row[col] = None if raw == "NA" else raw
                elif col == "position":
                    row[col] = int(raw) if raw != "NA" else None
                elif col == "gpos":
                    row[col] = int(raw) if raw not in ("NA", "") else None
                elif col.startswith("n_"):
                    row[col] = int(raw)
                else:
                    row[col] = float(raw)
            rows.append(row)

    return rows


# ---------- validation ----------


def validate_input(rows: list[dict[str, Any]]) -> None:
    """Validate that the input has gene-level metadata.

    Checks that ``gene_id`` and ``chrom`` columns are present and contain
    at least one non-null value.  Exits with an error message if not.

    Args:
        rows: List of row dicts from :func:`read_input`.
    """
    if not rows:
        _fail_metadata()

    # Check that both columns exist and have at least one non-null value.
    has_gene_id = any(r.get("gene_id") is not None for r in rows)
    has_chrom = any(r.get("chrom") is not None for r in rows)

    if not has_gene_id or not has_chrom:
        _fail_metadata()


def _fail_metadata() -> None:
    """Print metadata error and exit."""
    print(
        "Error: Input file lacks necessary gene mapping. "
        "Ensure the site file was produced using the -g/--gtf flag.",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------- aggregation ----------


def aggregate_to_gene(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate transcript-level rows to gene level.

    Groups rows by ``(gene_id, chrom, strand, gpos, mod_type)``, sums
    count and weighted-count columns, and recomputes modification levels.

    Rows where ``gene_id`` or ``gpos`` is ``None`` are dropped (they
    cannot be assigned to a genomic position).

    Args:
        rows: List of row dicts from :func:`read_input`.

    Returns:
        List of gene-level row dicts with keys from ``_OUTPUT_COLS``,
        sorted by ``(gene_id, gpos, mod_type)``.
    """
    # Group key → accumulated counts
    groups: dict[tuple[str, str, str, int, str], dict[str, float]] = defaultdict(
        lambda: {c: 0 for c in _COUNT_COLS} | {c: 0.0 for c in _WT_COLS}
    )

    for row in rows:
        gene_id = row.get("gene_id")
        gpos = row.get("gpos")
        if gene_id is None or gpos is None:
            continue

        chrom = row.get("chrom", "")
        strand = row.get("strand", "")
        mod_type = row.get("mod_type", "")

        key = (gene_id, chrom or "", strand or "", gpos, mod_type)
        acc = groups[key]
        for c in _COUNT_COLS:
            acc[c] += row.get(c, 0) or 0
        for c in _WT_COLS:
            acc[c] += row.get(c, 0.0) or 0.0

    # Build output rows
    result: list[dict[str, Any]] = []
    for (gene_id, chrom, strand, gpos, mod_type), acc in sorted(groups.items()):
        n_mod = acc["n_modified"]
        n_unmod = acc["n_unmodified"]
        wt_mod = acc["wt_modified"]
        wt_unmod = acc["wt_unmodified"]

        denom = n_mod + n_unmod
        w_denom = wt_mod + wt_unmod

        mod_level = round(n_mod / denom, 6) if denom > 0 else 0.0
        wt_mod_level = round(wt_mod / w_denom, 6) if w_denom > 0 else 0.0

        result.append(
            {
                "gene_id": gene_id,
                "chrom": chrom,
                "strand": strand,
                "gpos": gpos,
                "mod_type": mod_type,
                "n_modified": int(n_mod),
                "wt_modified": round(wt_mod, 4),
                "n_unmodified": int(n_unmod),
                "wt_unmodified": round(wt_unmod, 4),
                "n_canonical": int(acc["n_canonical"]),
                "wt_canonical": round(acc["wt_canonical"], 4),
                "n_othermod": int(acc["n_othermod"]),
                "wt_othermod": round(acc["wt_othermod"], 4),
                "n_mismatch": int(acc["n_mismatch"]),
                "wt_mismatch": round(acc["wt_mismatch"], 4),
                "n_deletion": int(acc["n_deletion"]),
                "wt_deletion": round(acc["wt_deletion"], 4),
                "n_failed": int(acc["n_failed"]),
                "wt_failed": round(acc["wt_failed"], 4),
                "mod_level": mod_level,
                "wt_mod_level": wt_mod_level,
            }
        )

    return result


_GENE_SCHEMA = pa.schema(
    [
        ("gene_id", pa.string()),
        ("chrom", pa.string()),
        ("strand", pa.string()),
        ("gpos", pa.int32()),
        ("mod_type", pa.string()),
        ("n_modified", pa.int32()),
        ("wt_modified", pa.float64()),
        ("n_unmodified", pa.int32()),
        ("wt_unmodified", pa.float64()),
        ("n_canonical", pa.int32()),
        ("wt_canonical", pa.float64()),
        ("n_othermod", pa.int32()),
        ("wt_othermod", pa.float64()),
        ("n_mismatch", pa.int32()),
        ("wt_mismatch", pa.float64()),
        ("n_deletion", pa.int32()),
        ("wt_deletion", pa.float64()),
        ("n_failed", pa.int32()),
        ("wt_failed", pa.float64()),
        ("mod_level", pa.float64()),
        ("wt_mod_level", pa.float64()),
    ]
)


# ---------- main ----------


def main(args: argparse.Namespace | None = None) -> None:
    """Aggregate transcript-level modification site summaries to gene level.

    Reads a site summary file produced by ``mod_sites`` (which must have
    been run with ``--gtf``) and writes a gene-level aggregation with
    columns ``gene_id, chrom, strand, gpos, mod_type`` plus the usual
    count and modification-level columns.
    """
    if args is None:
        args = parse_args()

    # ---- 1. Read input ----

    if args.verbose:
        print("[mod_gene] Reading input...", file=sys.stderr)
    rows = read_input(args.input)

    if args.verbose:
        print(f"[mod_gene] Read {len(rows)} rows", file=sys.stderr)

    # ---- 2. Validate metadata ----

    validate_input(rows)

    # ---- 3. Aggregate ----

    if args.verbose:
        print("[mod_gene] Aggregating to gene level...", file=sys.stderr)
    gene_rows = aggregate_to_gene(rows)

    if args.verbose:
        n_genes = len({r["gene_id"] for r in gene_rows})
        print(
            f"[mod_gene] {len(gene_rows)} rows across {n_genes} genes",
            file=sys.stderr,
        )

    # ---- 4. Write output ----

    if not gene_rows:
        print(
            "[mod_gene] No gene-level sites found — writing empty file.",
            file=sys.stderr,
        )

    if args.format == "tsv":
        write_tsv(gene_rows, args.output, _OUTPUT_HEADER, _OUTPUT_COLS, args.gzip)
    else:
        write_parquet(gene_rows, args.output, _GENE_SCHEMA, _OUTPUT_COLS)

    if args.verbose:
        print(
            f"[mod_gene] Done. Output written to {args.output}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
