#!/usr/bin/env python3
"""Estimate transcript isoform-specific poly(A) tail lengths from Oarfish
assignments and a Dorado BAM file with ``pt:i`` tags.
"""

import argparse
import gzip
import sys

import pysam

try:
    from isolens._parsing import parse_oarfish, read_id_to_int
except ImportError:
    from _parsing import parse_oarfish, read_id_to_int


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate transcript isoform-specific poly(A) length "
                    "using Oarfish assignments and Dorado BAM."
    )
    parser.add_argument(
        "-a", "--oarfish", required=True,
        help="Oarfish read assignment probability file (.lz4)")
    parser.add_argument(
        "-b", "--bam", required=True,
        help="Raw reads BAM file containing pt:i tags")
    parser.add_argument(
        "-o", "--output", required=True,
        help="Output TSV file")
    parser.add_argument(
        "-z", "--gzip", action="store_true",
        help="Compress the output TSV file using gzip")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Parsing Oarfish assignments from {args.oarfish}...", file=sys.stderr)
    tx_names, prob_map, name_to_id = parse_oarfish(args.oarfish)
    tx_idx_to_name = dict(enumerate(tx_names))

    n_assignments = len(prob_map)
    print(f"Loaded {len(tx_names)} transcripts and "
          f"{n_assignments} reads with assignments.", file=sys.stderr)

    if not prob_map:
        print("0 reads with assignments found. Exiting early without "
              "parsing the BAM file.", file=sys.stderr)
        sys.exit(0)

    # Initialize a dict to store poly(A) information mapped to transcripts
    tx_data = {tx_idx: [] for tx_idx in tx_idx_to_name}

    print(f"Processing BAM file {args.bam}...", file=sys.stderr)
    processed_reads = set()
    reads_scanned = 0

    # Read BAM file and extract pt:i tags
    with pysam.AlignmentFile(args.bam, "rb", check_sq=False) as bam:
        for read in bam.fetch(until_eof=True):
            reads_scanned += 1

            if reads_scanned % 200000 == 0:
                print(f"  ...scanned {reads_scanned} reads from BAM so far...",
                      file=sys.stderr)

            read_id_int = read_id_to_int(read.query_name)

            if read_id_int in processed_reads:
                continue

            if read_id_int in prob_map:
                if read.has_tag("pt"):
                    pt_val = read.get_tag("pt")

                    if pt_val > 0:
                        processed_reads.add(read_id_int)

                        for assignment in prob_map[read_id_int]:
                            tx_data[assignment.tx_id].append(
                                (assignment.prob, pt_val))

    print(f"Finished BAM parsing. Scanned {reads_scanned} total reads.",
          file=sys.stderr)
    print(f"Successfully extracted poly(A) lengths for "
          f"{len(processed_reads)} mapped reads.", file=sys.stderr)

    # Compute metrics and generate output TSV
    output_filename = args.output
    if args.gzip:
        if not output_filename.endswith(".gz"):
            output_filename += ".gz"
        def open_func(f):
            return gzip.open(f, "wt", encoding="utf-8")
    else:
        def open_func(f):
            return open(f, "w", encoding="utf-8")

    print(f"Writing results to {output_filename}...", file=sys.stderr)
    with open_func(output_filename) as out_f:
        out_f.write("tx_name\ttx_idx\tn_reads\tpa_wlen\tprobs\tpa_lens\n")

        for tx_idx, tx_name in tx_idx_to_name.items():
            data = tx_data.get(tx_idx, [])

            if not data:
                continue

            probs = [item[0] for item in data]
            pa_lens = [item[1] for item in data]

            n_reads = len(data)

            sum_prob = sum(probs)
            if sum_prob > 0:
                pa_wlen = sum(p * pa_len for p, pa_len in data) / sum_prob
            else:
                pa_wlen = 0.0

            probs_str = ",".join(f"{p:.5g}" for p in probs)
            pa_lens_str = ",".join(str(pa_len) for pa_len in pa_lens)

            out_f.write(
                f"{tx_name}\t{tx_idx}\t{n_reads}\t{pa_wlen:.3f}\t"
                f"{probs_str}\t{pa_lens_str}\n")

    print("Done!", file=sys.stderr)


if __name__ == "__main__":
    main()
