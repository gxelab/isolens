#!/usr/bin/env python3
"""Aggregate transcript-level poly(A) length estimates to the gene level."""

import argparse
import sys
from collections import defaultdict

try:
    from isolens._gtf import build_tx_to_gene
    from isolens._parsing import calc_weighted_pa_len, open_by_suffix
except ImportError:
    from _gtf import build_tx_to_gene  # type: ignore[no-redef]
    from _parsing import (  # type: ignore[no-redef]
        calc_weighted_pa_len,
        open_by_suffix,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for polya_t2g."""
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
            or "probs" not in header
            or "pa_lens" not in header
        ):
            print(
                "Error: Input file must contain 'transcript_id', 'probs', "
                "and 'pa_lens' headers.",
                file=sys.stderr,
            )
            sys.exit(1)

        tx_col = header.index("transcript_id")
        probs_col = header.index("probs")
        lens_col = header.index("pa_lens")

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
                "to polya_t2g.",
                file=sys.stderr,
            )
            sys.exit(1)

        # Load GTF mapping if needed
        tx_to_gene: dict[str, str] = {}
        if not has_gene_id_col:
            tx_to_gene = build_tx_to_gene(args.gtf)

        # Read transcript data and pool by gene
        gene_pools: dict[str, dict[str, list]] = defaultdict(
            lambda: {"probs": [], "pa_lens": []}
        )
        unmapped_transcripts: set[str] = set()

        for line in f:
            parts = line.strip().split("\t")
            if len(parts) <= max(probs_col, lens_col):
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

            probs = [float(p) for p in parts[probs_col].split(",")]
            pa_lens = [int(pa_len) for pa_len in parts[lens_col].split(",")]

            gene_pools[gene_id]["probs"].extend(probs)
            gene_pools[gene_id]["pa_lens"].extend(pa_lens)

    if unmapped_transcripts:
        print(
            f"Warning: Ignored {len(unmapped_transcripts)} transcripts "
            "that could not be mapped to a gene.",
            file=sys.stderr,
        )

    # Compute gene-level statistics and write output
    output_filename = args.output
    if args.gzip:
        if not output_filename.endswith(".gz"):
            output_filename += ".gz"

    print(f"Writing gene-level metrics to {output_filename}...", file=sys.stderr)

    write_mode = "wt" if output_filename.endswith(".gz") else "w"
    with open_by_suffix(output_filename, write_mode) as out_f:
        out_f.write("gene_id\tn_reads\tpa_wlen\tprobs\tpa_lens\n")

        for gene_id in sorted(gene_pools.keys()):
            probs = gene_pools[gene_id]["probs"]
            pa_lens = gene_pools[gene_id]["pa_lens"]

            n_reads = len(probs)
            pa_wlen = calc_weighted_pa_len(probs, pa_lens)

            probs_str = ",".join(f"{p:.5g}" for p in probs)
            pa_lens_str = ",".join(str(pa_len) for pa_len in pa_lens)

            out_f.write(
                f"{gene_id}\t{n_reads}\t{pa_wlen:.2f}\t{probs_str}\t{pa_lens_str}\n"
            )

    print("Gene-level aggregation completed successfully!", file=sys.stderr)


if __name__ == "__main__":
    main()
