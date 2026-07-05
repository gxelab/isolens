#!/usr/bin/env python3
"""Aggregate transcript-level poly(A) length estimates to the gene level."""

import argparse
import sys
from collections import defaultdict

try:
    from isolens._gtf import build_tx_to_gene
    from isolens._io import ensure_gz_suffix
    from isolens._parsing import calc_weighted_pa_len, open_by_suffix
except ImportError:
    from _io import ensure_gz_suffix  # type: ignore[no-redef]

    from _gtf import build_tx_to_gene  # type: ignore[no-redef]
    from _parsing import (  # type: ignore[no-redef]
        calc_weighted_pa_len,
        open_by_suffix,
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
        help="Input transcript poly(A) TSV file (gzipped or raw)",
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
        "-o", "--output", required=True, help="Output gene-level TSV file path"
    )
    parser.add_argument(
        "-z",
        "--gzip",
        action="store_true",
        help="Compress the output TSV file using gzip",
    )
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    """Aggregate per-transcript poly(A) data to gene level.

    Reads a transcript-level poly(A) TSV (from ``polya_calc``).  If the
    input already contains a ``gene_id`` column (e.g. from ``polya_calc
    -g``), that column is used directly.  Otherwise a GTF annotation file
    must be provided via ``--gtf`` to map transcript names to gene IDs.
    """
    if args is None:
        args = parse_args()

    # Read header and detect columns
    print(
        f"Processing transcript poly(A) lengths from {args.input}...", file=sys.stderr
    )
    read_mode = "rt" if args.input.endswith(".gz") else "r"
    with open_by_suffix(args.input, read_mode) as f:
        header = f.readline().strip().split("\t")
        if (
            "transcript_id" not in header
            or "weights" not in header
            or "lengths" not in header
        ):
            print(
                "Error: Input file must contain 'transcript_id', 'weights', "
                "and 'lengths' headers.",
                file=sys.stderr,
            )
            sys.exit(1)

        tx_col = header.index("transcript_id")
        weights_col = header.index("weights")
        lengths_col = header.index("lengths")

        has_gene_id_col = "gene_id" in header
        gene_id_col = header.index("gene_id") if has_gene_id_col else -1

        # Determine mapping source
        if has_gene_id_col:
            # Use gene_id directly from input rows
            print("Using gene_id column from input file.", file=sys.stderr)
        elif args.gtf is not None:
            print(
                "No gene_id column in input — using GTF annotation.",
                file=sys.stderr,
            )
        else:
            print(
                "Error: Input file does not contain a 'gene_id' column "
                "and no --gtf annotation was provided.  "
                "Either run polya_calc with --gtf, or provide --gtf "
                "to polya_gene.",
                file=sys.stderr,
            )
            sys.exit(1)

        # Load GTF mapping if needed
        tx_to_gene: dict[str, str] = {}
        if not has_gene_id_col:
            tx_to_gene = build_tx_to_gene(args.gtf)

        # Read transcript data and pool by gene
        gene_pools: dict[str, dict[str, list]] = defaultdict(
            lambda: {"weights": [], "lengths": []}
        )
        unmapped_transcripts: set[str] = set()

        for line in f:
            parts = line.strip().split("\t")
            if len(parts) <= max(weights_col, lengths_col):
                continue

            if has_gene_id_col and len(parts) <= gene_id_col:
                continue

            tx_name = parts[tx_col]

            # Resolve gene_id
            if has_gene_id_col:
                gene_id = parts[gene_id_col]
                if gene_id in ("", "NA", "."):
                    unmapped_transcripts.add(tx_name)
                    continue
            elif tx_name in tx_to_gene:
                gene_id = tx_to_gene[tx_name]
            else:
                unmapped_transcripts.add(tx_name)
                continue

            weights = [float(p) for p in parts[weights_col].split(",")]
            lengths = [int(pa_len) for pa_len in parts[lengths_col].split(",")]

            gene_pools[gene_id]["weights"].extend(weights)
            gene_pools[gene_id]["lengths"].extend(lengths)

    if unmapped_transcripts:
        print(
            f"Warning: Ignored {len(unmapped_transcripts)} transcripts "
            "that could not be mapped to a gene.",
            file=sys.stderr,
        )

    # Compute gene-level statistics and write output
    output_filename = ensure_gz_suffix(args.output, args.gzip)

    print(f"Writing gene-level metrics to {output_filename}...", file=sys.stderr)

    write_mode = "wt" if output_filename.endswith(".gz") else "w"
    with open_by_suffix(output_filename, write_mode) as out_f:
        out_f.write("gene_id\tn_reads\ttotal_wt\twmlen\tweights\tlengths\n")

        for gene_id in sorted(gene_pools.keys()):
            weights = gene_pools[gene_id]["weights"]
            lengths = gene_pools[gene_id]["lengths"]

            n_reads = len(weights)
            total_wt = sum(weights)
            wmlen = calc_weighted_pa_len(weights, lengths)

            weights_str = ",".join(f"{w:.5g}" for w in weights)
            lengths_str = ",".join(str(pa_len) for pa_len in lengths)

            out_f.write(
                f"{gene_id}\t{n_reads}\t{total_wt:.2f}\t{wmlen:.2f}\t"
                f"{weights_str}\t{lengths_str}\n"
            )

    print("Gene-level aggregation completed successfully!", file=sys.stderr)


if __name__ == "__main__":
    main()
