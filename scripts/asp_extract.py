#!/usr/bin/env python3
"""Extract a subset of an Oarfish assignment probability file.

Filters reads to only those assigned to one or more of the specified
transcripts, preserving the original file format: the header line is updated
with the new read count, all transcript name lines are kept, and only matching
read lines are written.

Usage:
    python asp_extract.py -i input.lz4 -t FBtr0073078,FBtr0073079 -o subset.lz4 --compress
"""

import argparse
import sys

import lz4.frame


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract Oarfish assignments for specific transcripts"
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input Oarfish assignment probability file (.lz4)",
    )
    parser.add_argument(
        "-t", "--transcripts",
        required=True,
        help="Comma-separated list of transcript names to retain",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output assignment probability file path",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="LZ4-compress the output (default: plain text)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Parse requested transcript names
    requested = set(t.strip() for t in args.transcripts.split(",") if t.strip())
    if not requested:
        print("Error: no valid transcript names provided.", file=sys.stderr)
        sys.exit(1)

    # ---- Parse input ----

    with lz4.frame.open(args.input, "rb") as f:
        # Header line: <m> <n>
        header_line = f.readline().decode("utf-8").strip()
        if not header_line:
            print("Error: empty input file.", file=sys.stderr)
            sys.exit(1)

        parts = header_line.split()
        num_transcripts = int(parts[0])

        # Transcript name lines
        tx_names = []
        name_to_idx = {}
        for i in range(num_transcripts):
            tx_name = f.readline().decode("utf-8").strip()
            tx_names.append(tx_name)
            name_to_idx[tx_name] = i

        # Identify target transcript indices
        target_indices = set()
        for tx_name in requested:
            if tx_name in name_to_idx:
                target_indices.add(name_to_idx[tx_name])
            else:
                print(f"Warning: transcript '{tx_name}' not found in input file.",
                      file=sys.stderr)

        if not target_indices:
            print("Error: none of the requested transcripts exist in the input.",
                  file=sys.stderr)
            sys.exit(1)

        # Read assignments
        kept_lines = []
        total_reads = 0

        for line in f:
            decoded = line.decode("utf-8").strip()
            if not decoded:
                continue

            tokens = decoded.split()
            read_name = tokens[0]
            num_targets = int(tokens[1])

            target_ids = tokens[2 : 2 + num_targets]
            probs = tokens[2 + num_targets : 2 + (2 * num_targets)]

            # Check if this read is assigned to any of the target transcripts
            for tid_str, prob_str in zip(target_ids, probs):
                if int(tid_str) in target_indices:
                    kept_lines.append(decoded)
                    total_reads += 1
                    break  # keep the line once, even if multiple targets match

        # ---- Write output ----

        if args.compress:
            out_fh = lz4.frame.open(args.output, mode="wt", encoding="utf-8")
        else:
            out_fh = open(args.output, "w", encoding="utf-8")

        with out_fh as out:
            # Header line with updated read count
            out.write(f"{num_transcripts}\t{total_reads}\n")

            # All transcript name lines preserved
            for tx_name in tx_names:
                out.write(f"{tx_name}\n")

            # Filtered read lines
            for kept in kept_lines:
                out.write(f"{kept}\n")

    print(f"Done. Kept {total_reads} reads (filtered from {len(tx_names)} "
          f"transcripts → {len(target_indices)} targets).", file=sys.stderr)


if __name__ == "__main__":
    main()
