#!/usr/bin/env python3
"""Merge two poly(A) estimation TSV files and recalculate weighted lengths."""

import argparse
import gzip
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge two poly(A) estimation TSV files together "
                    "and recalculate weighted lengths."
    )
    parser.add_argument(
        "-i1", "--input1", required=True,
        help="First input TSV file (gzipped or raw)")
    parser.add_argument(
        "-i2", "--input2", required=True,
        help="Second input TSV file (gzipped or raw)")
    parser.add_argument(
        "-o", "--output", required=True,
        help="Output file path")
    parser.add_argument(
        "-z", "--gzip", action="store_true",
        help="Compress the output TSV file using gzip")
    return parser.parse_args()


def read_tsv_to_dict(filename):
    """Read a TSV file, auto-detecting gzip by suffix.

    Returns:
        dict: ``tx_idx -> {'tx_name': name, 'probs': [float, ...],
        'pa_lens': [int, ...]}``
    """
    data_dict = {}
    print(f"Reading {filename}...", file=sys.stderr)

    if filename.endswith(".gz"):
        def open_func(f):
            return gzip.open(f, "rt", encoding="utf-8")
    else:
        def open_func(f):
            return open(f, encoding="utf-8")

    with open_func(filename) as f:
        header = f.readline().strip().split("\t")
        if len(header) < 6 or header[1] != "tx_idx":
            print(f"Error: {filename} header layout is unexpected or "
                  "malformed.", file=sys.stderr)
            sys.exit(1)

        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 6:
                continue

            tx_name = parts[0]
            tx_idx = int(parts[1])

            probs = [float(p) for p in parts[4].split(",")]
            pa_lens = [int(pa_len) for pa_len in parts[5].split(",")]

            data_dict[tx_idx] = {
                "tx_name": tx_name,
                "probs": probs,
                "pa_lens": pa_lens,
            }
    return data_dict


def main():
    args = parse_args()

    # Load data from both files (auto-detecting gzip)
    file1_data = read_tsv_to_dict(args.input1)
    file2_data = read_tsv_to_dict(args.input2)

    all_tx_indices = sorted(set(file1_data.keys()) | set(file2_data.keys()))
    print(f"Merging information across {len(all_tx_indices)} distinct "
          "transcripts...", file=sys.stderr)

    output_filename = args.output
    if args.gzip:
        if not output_filename.endswith(".gz"):
            output_filename += ".gz"
        def open_output_func(f):
            return gzip.open(f, "wt", encoding="utf-8")
    else:
        def open_output_func(f):
            return open(f, "w", encoding="utf-8")

    print(f"Writing re-estimated results to {output_filename}...",
          file=sys.stderr)

    with open_output_func(output_filename) as out_f:
        out_f.write("tx_name\ttx_idx\tn_reads\tpa_wlen\tprobs\tpa_lens\n")

        for tx_idx in all_tx_indices:
            tx_name = None
            merged_probs = []
            merged_lens = []

            if tx_idx in file1_data:
                tx_name = file1_data[tx_idx]["tx_name"]
                merged_probs.extend(file1_data[tx_idx]["probs"])
                merged_lens.extend(file1_data[tx_idx]["pa_lens"])

            if tx_idx in file2_data:
                if tx_name is None:
                    tx_name = file2_data[tx_idx]["tx_name"]
                merged_probs.extend(file2_data[tx_idx]["probs"])
                merged_lens.extend(file2_data[tx_idx]["pa_lens"])

            n_reads = len(merged_probs)

            sum_prob = sum(merged_probs)
            if sum_prob > 0:
                pa_wlen = sum(p * pa_len for p, pa_len in zip(
                    merged_probs, merged_lens)) / sum_prob
            else:
                pa_wlen = 0.0

            probs_str = ",".join(f"{p:.5g}" for p in merged_probs)
            pa_lens_str = ",".join(str(pa_len) for pa_len in merged_lens)

            out_f.write(
                f"{tx_name}\t{tx_idx}\t{n_reads}\t{pa_wlen:.2f}\t"
                f"{probs_str}\t{pa_lens_str}\n")

    print("Done merging seamlessly!", file=sys.stderr)


if __name__ == "__main__":
    main()
