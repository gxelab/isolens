#!/usr/bin/env python3
"""Aggregate transcript-level poly(A) length estimates to the gene level."""

import argparse
import sys
from collections import defaultdict

try:
    from isolens._parsing import calc_weighted_pa_len, open_by_suffix
except ImportError:
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
        "-m",
        "--map",
        required=True,
        help="Mapping file containing 'tx_name' and 'gene_id' columns "
        "(gzipped or raw TSV)",
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


def load_gene_mapping(map_file: str) -> dict[str, str]:
    """Parse a TSV mapping file and return ``{tx_name: gene_id}``.

    Args:
        map_file: Path to a TSV or TSV.GZ file with ``tx_name`` and
            ``gene_id`` columns.

    Returns:
        ``dict[str, str]`` mapping each transcript name to its gene ID.
    """
    print(f"Reading mapping file from {map_file}...", file=sys.stderr)
    tx_to_gene: dict[str, str] = {}

    read_mode = "rt" if map_file.endswith(".gz") else "r"
    with open_by_suffix(map_file, read_mode) as f:
        header = f.readline().strip().split("\t")

        if "tx_name" not in header or "gene_id" not in header:
            print(
                "Error: Mapping file must contain both 'tx_name' and "
                "'gene_id' column headers.",
                file=sys.stderr,
            )
            sys.exit(1)

        tx_col = header.index("tx_name")
        gene_col = header.index("gene_id")

        for line in f:
            parts = line.strip().split("\t")
            if len(parts) <= max(tx_col, gene_col):
                continue

            tx_to_gene[parts[tx_col]] = parts[gene_col]

    print(f"Loaded mappings for {len(tx_to_gene)} unique transcripts.", file=sys.stderr)
    return tx_to_gene


def main() -> None:
    """Aggregate per-transcript poly(A) data to gene level.

    Reads a transcript-level poly(A) TSV (from ``polya_calc``) and a
    ``tx_name → gene_id`` mapping file, pools reads by gene, and writes
    a gene-level poly(A) TSV with recalculated weighted average lengths.
    """
    args = parse_args()

    # Load transcript-to-gene relationships
    tx_to_gene = load_gene_mapping(args.map)

    # Read transcript data and pool by gene
    gene_pools: dict[str, dict[str, list]] = defaultdict(
        lambda: {"probs": [], "pa_lens": []}
    )
    unmapped_transcripts: set[str] = set()

    print(
        f"Processing transcript poly(A) lengths from {args.input}...", file=sys.stderr
    )
    read_mode = "rt" if args.input.endswith(".gz") else "r"
    with open_by_suffix(args.input, read_mode) as f:
        header = f.readline().strip().split("\t")
        if "tx_name" not in header or "probs" not in header or "pa_lens" not in header:
            print(
                "Error: Input file must contain 'tx_name', 'probs', "
                "and 'pa_lens' headers.",
                file=sys.stderr,
            )
            sys.exit(1)

        tx_col = header.index("tx_name")
        probs_col = header.index("probs")
        lens_col = header.index("pa_lens")

        for line in f:
            parts = line.strip().split("\t")
            if len(parts) <= max(probs_col, lens_col):
                continue

            tx_name = parts[tx_col]

            if tx_name in tx_to_gene:
                gene_id = tx_to_gene[tx_name]

                probs = [float(p) for p in parts[probs_col].split(",")]
                pa_lens = [int(pa_len) for pa_len in parts[lens_col].split(",")]

                gene_pools[gene_id]["probs"].extend(probs)
                gene_pools[gene_id]["pa_lens"].extend(pa_lens)
            else:
                unmapped_transcripts.add(tx_name)

    if unmapped_transcripts:
        print(
            f"Warning: Ignored {len(unmapped_transcripts)} transcripts "
            "that were missing from the mapping file.",
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
