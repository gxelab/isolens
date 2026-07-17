#!/usr/bin/env python3
"""Merge two poly(A) estimation files and recalculate weighted lengths."""

import argparse
import sys
from typing import Any

import pyarrow as pa

try:
    from isolens._io import ensure_gz_suffix, write_parquet, write_tsv
    from isolens._parsing import calc_weighted_pa_len, parse_polyA_file
except ImportError:
    from _io import ensure_gz_suffix, write_parquet, write_tsv  # type: ignore[no-redef]

    from _parsing import (  # type: ignore[no-redef]
        calc_weighted_pa_len,
        parse_polyA_file,
    )


_OUTPUT_COLS = [
    "transcript_id",
    "n_reads",
    "total_wt",
    "wmlen",
    "weights",
    "lengths",
]
_TSV_HEADER = "\t".join(_OUTPUT_COLS)

_MERGE_SCHEMA = pa.schema(
    [
        ("transcript_id", pa.string()),
        ("n_reads", pa.int32()),
        ("total_wt", pa.float32()),
        ("wmlen", pa.float32()),
        ("weights", pa.list_(pa.float32())),
        ("lengths", pa.list_(pa.int32())),
    ]
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for polya_merge."""
    parser = argparse.ArgumentParser(
        description="Merge two poly(A) estimation files together "
        "and recalculate weighted lengths."
    )
    parser.add_argument(
        "-i1",
        "--input1",
        required=True,
        help="First input file (TSV/TSV.GZ or Parquet)",
    )
    parser.add_argument(
        "-i2",
        "--input2",
        required=True,
        help="Second input file (TSV/TSV.GZ or Parquet)",
    )
    parser.add_argument("-o", "--output", required=True, help="Output file path")
    parser.add_argument(
        "-f",
        "--format",
        choices=["parquet", "tsv"],
        default="tsv",
        help="Output format: tsv (default) or parquet",
    )
    parser.add_argument(
        "-z",
        "--gzip",
        action="store_true",
        help="Compress the output TSV file using gzip",
    )
    parser.add_argument(
        "-l",
        "--log",
        action="store_true",
        default=False,
        help="Apply log-transform (log(L+1)) to poly(A) tail lengths before "
        "computing weighted averages, then back-transform results. "
        "This computes a weighted geometric mean.",
    )
    return parser.parse_args()


def _read_polya_to_dict(filename: str) -> dict[str, dict[str, Any]]:
    """Read a poly(A) file (TSV/TSV.GZ or Parquet) into a per-transcript dict.

    Args:
        filename: Path to a poly(A) file with columns
            ``transcript_id, n_reads, total_wt, wmlen, weights, lengths``
            and optionally ``gene_id``.

    Returns:
        ``dict[str, dict]`` mapping ``transcript_id`` to
        ``{'weights': list[float], 'lengths': list[int]}``
        and optionally ``gene_id`` when present in the input.
    """
    _id_col, raw = parse_polyA_file(filename)
    data_dict: dict[str, dict[str, Any]] = {}
    for tx_name, info in raw.items():
        entry: dict[str, Any] = {
            "weights": info["weights"].tolist(),
            "lengths": info["lengths"].tolist(),
        }
        if "gene_id" in info:
            entry["gene_id"] = info["gene_id"]
        data_dict[tx_name] = entry
    return data_dict


def main(args: argparse.Namespace | None = None) -> None:
    """Merge two poly(A) TSV files and recompute weighted averages.

    Reads two poly(A) output files (from ``polya_calc``), pools reads
    per transcript across both files, and writes a merged TSV with
    recalculated per-transcript weighted average poly(A) lengths.
    """
    if args is None:
        args = parse_args()

    # Load data from both files (auto-detecting gzip)
    file1_data = _read_polya_to_dict(args.input1)
    file2_data = _read_polya_to_dict(args.input2)

    all_tx_names = sorted(set(file1_data.keys()) | set(file2_data.keys()))
    print(
        f"Merging information across {len(all_tx_names)} distinct transcripts...",
        file=sys.stderr,
    )

    # Detect whether gene_id is present in either input
    has_gene_id = any(
        "gene_id" in d for data in (file1_data, file2_data) for d in data.values()
    )

    # Build output structures (conditional gene_id)
    _output_cols = list(_OUTPUT_COLS)
    _schema_fields: list[tuple[str, pa.DataType]] = [
        ("transcript_id", pa.string()),
        ("n_reads", pa.int32()),
        ("total_wt", pa.float32()),
        ("wmlen", pa.float32()),
        ("weights", pa.list_(pa.float32())),
        ("lengths", pa.list_(pa.int32())),
    ]
    if has_gene_id:
        _output_cols.insert(0, "gene_id")
        _schema_fields.insert(0, ("gene_id", pa.string()))
    _tsv_header = "\t".join(_output_cols)
    _merge_schema = pa.schema(_schema_fields)

    # Accumulate rows
    all_rows: list[dict] = []

    for tx_name in all_tx_names:
        merged_weights: list[float] = []
        merged_lengths: list[int] = []
        gene_id: str | None = None

        for data in (file1_data, file2_data):
            if tx_name in data:
                d = data[tx_name]
                merged_weights.extend(d["weights"])
                merged_lengths.extend(d["lengths"])
                if has_gene_id and gene_id is None:
                    gene_id = d.get("gene_id")

        n_reads = len(merged_weights)
        total_wt = sum(merged_weights)
        wmlen = calc_weighted_pa_len(
            merged_weights, merged_lengths, use_log=getattr(args, "log", False)
        )

        row: dict[str, Any] = {}
        if has_gene_id:
            row["gene_id"] = gene_id or "NA"
        row["transcript_id"] = tx_name
        row["n_reads"] = n_reads
        row["total_wt"] = total_wt
        row["wmlen"] = wmlen
        row["weights"] = merged_weights
        row["lengths"] = merged_lengths
        all_rows.append(row)

    # Write output
    if args.format == "tsv":
        out_path = ensure_gz_suffix(args.output, args.gzip)
        write_tsv(all_rows, out_path, _tsv_header, _output_cols, args.gzip)
    else:
        write_parquet(all_rows, args.output, _merge_schema, _output_cols)

    print("Done merging seamlessly!", file=sys.stderr)


if __name__ == "__main__":
    main()
