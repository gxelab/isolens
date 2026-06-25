#!/usr/bin/env python3
"""Extract a subset of an Oarfish assignment probability file.

Filters reads to only those assigned to one or more of the specified
transcripts, preserving the original file format. The output header is slimmed
to only the requested transcripts, and target indices in read assignment lines
are remapped to the new, compact index space.

Supports minimum assignment probability cutoff (-p), maximum reads limit (-n),
and optional LZ4 compression (--compress).

Usage:
    python asp_extract.py -i input.lz4 -t FBtr0073078,FBtr0073079 -o subset.lz4 --compress
    python asp_extract.py -i input.lz4 -t FBtr0073078 -p 0.95 -n 1000 -o subset.lz4 --compress
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
        "-p", "--min-prob",
        type=float,
        default=0.0,
        help="Minimum assignment probability for a read to a target "
             "transcript (0.0–1.0, default: 0.0 = no cutoff)",
    )
    parser.add_argument(
        "-n", "--max-reads",
        type=int,
        default=0,
        help="Maximum number of reads to export (0 = unlimited). "
             "Reads are sorted by their best assignment probability "
             "to any target transcript before truncation.",
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

        # Build old→new index mapping for requested transcripts.
        # Preserve the input-file ordering.
        target_old_to_new = {}   # old_idx → new_idx
        target_new_names = []    # [name_at_new_idx_0, name_at_new_idx_1, …]
        for tx_name in requested:
            if tx_name in name_to_idx:
                target_old_to_new[name_to_idx[tx_name]] = len(target_new_names)
                target_new_names.append(tx_name)
            else:
                print(f"Warning: transcript '{tx_name}' not found in input file.",
                      file=sys.stderr)

        if not target_old_to_new:
            print("Error: none of the requested transcripts exist in the input.",
                  file=sys.stderr)
            sys.exit(1)

        target_old_indices = set(target_old_to_new.keys())
        num_out_transcripts = len(target_new_names)

        # Read assignments — collect (best_target_prob, remapped_line)
        kept_entries = []  # list of (best_prob, remapped_line)

        for line in f:
            decoded = line.decode("utf-8").strip()
            if not decoded:
                continue

            tokens = decoded.split()
            read_name = tokens[0]
            num_targets = int(tokens[1])

            target_ids = tokens[2 : 2 + num_targets]
            probs = tokens[2 + num_targets : 2 + (2 * num_targets)]

            # Filter to target transcripts, remap indices, and record
            # the best probability among them.
            remapped = []        # list of (new_idx, prob_str)
            best_prob = 0.0
            for tid_str, prob_str in zip(target_ids, probs):
                old_idx = int(tid_str)
                if old_idx in target_old_indices:
                    new_idx = target_old_to_new[old_idx]
                    remapped.append((new_idx, prob_str))
                    prob = float(prob_str)
                    if prob > best_prob:
                        best_prob = prob

            # Apply minimum probability cutoff.
            # Only keep reads that have at least one matching target
            # AND whose best probability meets the threshold.
            if remapped and best_prob >= args.min_prob:
                # Reconstruct the line with remapped indices
                new_num_targets = len(remapped)
                new_tids = " ".join(str(idx) for idx, _ in remapped)
                new_probs = " ".join(prob_str for _, prob_str in remapped)
                new_line = f"{read_name}\t{new_num_targets}\t{new_tids}\t{new_probs}"
                kept_entries.append((best_prob, new_line))

        # Apply maximum reads cutoff: sort by best probability descending,
        # then keep at most N reads.
        total_matched = len(kept_entries)
        if args.max_reads > 0 and total_matched > args.max_reads:
            kept_entries.sort(key=lambda x: x[0], reverse=True)
            kept_entries = kept_entries[: args.max_reads]

        total_reads = len(kept_entries)

        # ---- Write output ----

        if args.compress:
            out_fh = lz4.frame.open(args.output, mode="wt", encoding="utf-8")
        else:
            out_fh = open(args.output, "w", encoding="utf-8")

        with out_fh as out:
            # Header line with slimmed transcript count
            out.write(f"{num_out_transcripts}\t{total_reads}\n")

            # Only the requested transcript names (in input-file order)
            for tx_name in target_new_names:
                out.write(f"{tx_name}\n")

            # Filtered and index-remapped read lines
            # (sorted by best prob descending when max-reads was applied,
            #  otherwise in input-file order)
            for _, kept in kept_entries:
                out.write(f"{kept}\n")

    print(f"Done. Kept {total_reads} reads ({num_out_transcripts} "
          f"transcripts out of {len(tx_names)} total).", file=sys.stderr)
    if total_matched > total_reads:
        print(f"  Truncated from {total_matched} matching reads "
              f"(max-reads={args.max_reads}).", file=sys.stderr)


if __name__ == "__main__":
    main()
