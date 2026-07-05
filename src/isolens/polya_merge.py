#!/usr/bin/env python3
"""Merge two poly(A) estimation TSV files and recalculate weighted lengths."""

import argparse
import sys
from typing import Any

try:
    from isolens._io import ensure_gz_suffix
    from isolens._parsing import calc_weighted_pa_len, open_by_suffix
except ImportError:
    from _io import ensure_gz_suffix  # type: ignore[no-redef]

    from _parsing import (  # type: ignore[no-redef]
        calc_weighted_pa_len,
        open_by_suffix,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for polya_merge."""
    parser = argparse.ArgumentParser(
        description="Merge two poly(A) estimation TSV files together "
        "and recalculate weighted lengths."
    )
    parser.add_argument(
        "-i1", "--input1", required=True, help="First input TSV file (gzipped or raw)"
    )
    parser.add_argument(
        "-i2", "--input2", required=True, help="Second input TSV file (gzipped or raw)"
    )
    parser.add_argument("-o", "--output", required=True, help="Output file path")
    parser.add_argument(
        "-z",
        "--gzip",
        action="store_true",
        help="Compress the output TSV file using gzip",
    )
    return parser.parse_args()


def read_tsv_to_dict(filename: str) -> dict[str, dict[str, Any]]:
    """Read a poly(A) TSV file, auto-detecting gzip by suffix.

    Args:
        filename: Path to a TSV (or TSV.GZ) file with columns
            ``transcript_id, n_reads, total_wt, wmlen, weights, lengths``.

    Returns:
        ``dict[str, dict]`` mapping ``transcript_id`` to
        ``{'weights': list[float], 'lengths': list[int]}``.
    """
    data_dict: dict[str, dict[str, Any]] = {}
    print(f"Reading {filename}...", file=sys.stderr)

    read_mode = "rt" if filename.endswith(".gz") else "r"
    with open_by_suffix(filename, read_mode) as f:
        header = f.readline().strip().split("\t")
        if "transcript_id" not in header:
            print(
                f"Error: {filename} is missing 'transcript_id' column.",
                file=sys.stderr,
            )
            sys.exit(1)

        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 6:
                continue

            tx_name = parts[0]

            weights = [float(p) for p in parts[4].split(",")]
            lengths = [int(pa_len) for pa_len in parts[5].split(",")]

            data_dict[tx_name] = {
                "weights": weights,
                "lengths": lengths,
            }
    return data_dict


def main() -> None:
    """Merge two poly(A) TSV files and recompute weighted averages.

    Reads two poly(A) output files (from ``polya_calc``), pools reads
    per transcript across both files, and writes a merged TSV with
    recalculated per-transcript weighted average poly(A) lengths.
    """
    args = parse_args()

    # Load data from both files (auto-detecting gzip)
    file1_data = read_tsv_to_dict(args.input1)
    file2_data = read_tsv_to_dict(args.input2)

    all_tx_names = sorted(set(file1_data.keys()) | set(file2_data.keys()))
    print(
        f"Merging information across {len(all_tx_names)} distinct transcripts...",
        file=sys.stderr,
    )

    output_filename = ensure_gz_suffix(args.output, args.gzip)

    print(f"Writing re-estimated results to {output_filename}...", file=sys.stderr)

    write_mode = "wt" if output_filename.endswith(".gz") else "w"
    with open_by_suffix(output_filename, write_mode) as out_f:
        out_f.write(
            "transcript_id\tn_reads\ttotal_wt\twmlen\tweights\tlengths\n"
        )

        for tx_name in all_tx_names:
            merged_weights: list[float] = []
            merged_lengths: list[int] = []

            if tx_name in file1_data:
                merged_weights.extend(file1_data[tx_name]["weights"])
                merged_lengths.extend(file1_data[tx_name]["lengths"])

            if tx_name in file2_data:
                merged_weights.extend(file2_data[tx_name]["weights"])
                merged_lengths.extend(file2_data[tx_name]["lengths"])

            n_reads = len(merged_weights)
            total_wt = sum(merged_weights)
            wmlen = calc_weighted_pa_len(merged_weights, merged_lengths)

            weights_str = ",".join(f"{w:.5g}" for w in merged_weights)
            lengths_str = ",".join(str(pa_len) for pa_len in merged_lengths)

            out_f.write(
                f"{tx_name}\t{n_reads}\t{total_wt:.2f}\t{wmlen:.2f}\t"
                f"{weights_str}\t{lengths_str}\n"
            )

    print("Done merging seamlessly!", file=sys.stderr)


if __name__ == "__main__":
    main()
