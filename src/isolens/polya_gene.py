#!/usr/bin/env python3
"""Aggregate transcript-level poly(A) length estimates to the gene level."""

import argparse
import sys
from collections import defaultdict

import pyarrow as pa

try:
    from isolens._gtf import build_tx_to_gene
    from isolens._io import ensure_gz_suffix, write_parquet, write_tsv
    from isolens._parsing import calc_weighted_pa_len, parse_polyA_file
except ImportError:
    from _io import ensure_gz_suffix, write_parquet, write_tsv  # type: ignore[no-redef]

    from _gtf import build_tx_to_gene  # type: ignore[no-redef]
    from _parsing import (  # type: ignore[no-redef]
        calc_weighted_pa_len,
        parse_polyA_file,
    )


_OUTPUT_COLS = [
    "gene_id",
    "n_reads",
    "total_wt",
    "wmlen",
    "weights",
    "lengths",
]
_TSV_HEADER = "\t".join(_OUTPUT_COLS)

_GENE_SCHEMA = pa.schema(
    [
        ("gene_id", pa.string()),
        ("n_reads", pa.int32()),
        ("total_wt", pa.float32()),
        ("wmlen", pa.float32()),
        ("weights", pa.list_(pa.float32())),
        ("lengths", pa.list_(pa.int32())),
    ]
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for polya_gene."""
    parser = argparse.ArgumentParser(
        description="Aggregate transcript-level poly(A) length estimates "
        "to the gene level."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input transcript poly(A) file (TSV/TSV.GZ or Parquet)",
    )
    parser.add_argument(
        "-g",
        "--gtf",
        default=None,
        help="GTF annotation file for transcript-to-gene mapping "
        "(gzipped or raw). Required if the input file does not "
        "already contain a gene_id column.",
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Output gene-level file path"
    )
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


def main(args: argparse.Namespace | None = None) -> None:
    """Aggregate per-transcript poly(A) data to gene level.

    Reads a transcript-level poly(A) file (from ``polya_calc``).  If the
    input already contains a ``gene_id`` column (e.g. from ``polya_calc
    -g``), that column is used directly.  Otherwise a GTF annotation file
    must be provided via ``--gtf`` to map transcript names to gene IDs.
    """
    if args is None:
        args = parse_args()

    # Read transcript-level data (supports TSV, TSV.GZ, and Parquet)
    id_col_name, tx_data = parse_polyA_file(args.input)

    if not tx_data:
        print("No transcripts found in input file.", file=sys.stderr)
        sys.exit(0)

    # Determine gene-mapping strategy
    has_gene_id_in_data = any("gene_id" in v for v in tx_data.values())
    tx_to_gene: dict[str, str] = {}

    if has_gene_id_in_data:
        print("Using gene_id column from input file.", file=sys.stderr)
    elif args.gtf is not None:
        print(
            "No gene_id column in input — using GTF annotation.",
            file=sys.stderr,
        )
        tx_to_gene = build_tx_to_gene(args.gtf)
    else:
        print(
            "Error: Input file does not contain a 'gene_id' column "
            "and no --gtf annotation was provided.  "
            "Either run polya_calc with --gtf, or provide --gtf "
            "to polya_gene.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Pool reads by gene
    gene_pools: dict[str, dict[str, list]] = defaultdict(
        lambda: {"weights": [], "lengths": []}
    )
    unmapped_transcripts: set[str] = set()

    for tx_name, tx_info in tx_data.items():
        weights = tx_info["weights"].tolist()
        lengths = tx_info["lengths"].tolist()

        # Resolve gene_id
        if has_gene_id_in_data:
            gene_id = tx_info.get("gene_id")
            if gene_id is None:
                unmapped_transcripts.add(tx_name)
                continue
        elif tx_name in tx_to_gene:
            gene_id = tx_to_gene[tx_name]
        else:
            unmapped_transcripts.add(tx_name)
            continue

        gene_pools[gene_id]["weights"].extend(weights)
        gene_pools[gene_id]["lengths"].extend(lengths)

    if unmapped_transcripts:
        print(
            f"Warning: Ignored {len(unmapped_transcripts)} transcripts "
            "that could not be mapped to a gene.",
            file=sys.stderr,
        )

    # Compute gene-level statistics and accumulate rows
    all_rows: list[dict] = []

    for gene_id in sorted(gene_pools.keys()):
        weights = gene_pools[gene_id]["weights"]
        lengths = gene_pools[gene_id]["lengths"]

        n_reads = len(weights)
        total_wt = sum(weights)
        wmlen = calc_weighted_pa_len(
            weights, lengths, use_log=getattr(args, "log", False)
        )

        all_rows.append(
            {
                "gene_id": gene_id,
                "n_reads": n_reads,
                "total_wt": total_wt,
                "wmlen": wmlen,
                "weights": weights,
                "lengths": lengths,
            }
        )

    # Write output
    if args.format == "tsv":
        out_path = ensure_gz_suffix(args.output, args.gzip)
        write_tsv(all_rows, out_path, _TSV_HEADER, _OUTPUT_COLS, args.gzip)
    else:
        write_parquet(all_rows, args.output, _GENE_SCHEMA, _OUTPUT_COLS)

    print("Gene-level aggregation completed successfully!", file=sys.stderr)


if __name__ == "__main__":
    main()
