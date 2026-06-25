#!/usr/bin/env python3
"""Filter a BAM file to only reads present in Oarfish assignment probability files.

Extracts read UUIDs and transcript names from one or more assignment (LZ4)
files, then writes a new BAM containing only alignments whose query name matches
a UUID in the set AND whose reference sequence name matches a transcript in the
assignment header. Reads present in the assignment files but missing from the
BAM are reported as warnings on stderr.

Usage:
    python bam_filter.py -i input.bam -a subset.lz4 -o filtered.bam
    python bam_filter.py -i input.bam -a rep1.lz4 -a rep2.lz4 -o filtered.bam
"""

import argparse
import sys
import uuid

import lz4.frame
import pysam


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter a BAM to reads present in Oarfish assignment files"
    )
    parser.add_argument(
        "-i", "--input-bam",
        required=True,
        help="Input BAM file (transcriptome-aligned)",
    )
    parser.add_argument(
        "-a", "--assignments",
        required=True,
        action="append",
        help="Oarfish assignment probability file (.lz4). "
             "May be specified multiple times to merge read sets.",
    )
    parser.add_argument(
        "-o", "--output-bam",
        required=True,
        help="Output filtered BAM file",
    )
    return parser.parse_args()


def parse_assignment(path):
    """Extract read UUIDs and transcript names from an Oarfish assignment file.

    Returns
    -------
    tuple[set[str], set[str]]
        (read_uuids, transcript_names) found in the assignment file.
    """
    uuids = set()
    tx_names = set()
    with lz4.frame.open(path, "rb") as fh:
        # Header line: <num_transcripts> <num_reads>
        header = fh.readline().decode("utf-8").strip()
        if not header:
            print(f"Warning: empty assignment file '{path}'.", file=sys.stderr)
            return uuids, tx_names

        parts = header.split()
        num_transcripts = int(parts[0])

        # Transcript name lines
        for _ in range(num_transcripts):
            tx_name = fh.readline().decode("utf-8").strip()
            tx_names.add(tx_name)

        # Collect read UUIDs (first token of each read line)
        for line in fh:
            decoded = line.decode("utf-8").strip()
            if not decoded:
                continue
            read_name = decoded.split(maxsplit=1)[0]
            uuids.add(read_name)

    return uuids, tx_names


def main():
    args = parse_args()

    # ---- Collect read UUIDs and transcript names from all assignment files ----

    all_uuids = set()        # set of int UUIDs for BAM lookup
    uuid_to_str = {}         # int -> original string (for warning messages)
    all_tx_names = set()     # union of transcript names across all assignment files

    for path in args.assignments:
        uuids, tx_names = parse_assignment(path)
        for u in uuids:
            try:
                uid_int = uuid.UUID(u).int
            except ValueError:
                print(f"Warning: skipping non-UUID read name '{u}' "
                      f"in '{path}'.", file=sys.stderr)
                continue
            all_uuids.add(uid_int)
            uuid_to_str[uid_int] = u
        all_tx_names.update(tx_names)
        print(f"  Loaded {len(uuids)} read UUIDs and {len(tx_names)} "
              f"transcripts from '{path}'.", file=sys.stderr)

    if not all_uuids:
        print("Error: no valid read UUIDs found in assignment files.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Total: {len(all_uuids)} unique read UUIDs, "
          f"{len(all_tx_names)} unique transcripts "
          f"across all assignment files.", file=sys.stderr)

    # ---- Filter BAM ----

    found_uuids = set()   # int UUIDs found in the BAM
    total_records = 0
    written_records = 0

    with (
        pysam.AlignmentFile(args.input_bam, "rb", check_sq=False) as in_bam,
        pysam.AlignmentFile(args.output_bam, "wb", template=in_bam) as out_bam,
    ):
        for record in in_bam:
            total_records += 1

            if total_records % 500000 == 0:
                print(f"  ...scanned {total_records} BAM records...",
                      file=sys.stderr)

            try:
                uid_int = uuid.UUID(record.query_name).int
            except ValueError:
                # Non-UUID query name — skip unless it's a string match
                # (unlikely, but handle gracefully)
                continue

            if uid_int not in all_uuids:
                continue

            # Check that the reference sequence matches a transcript
            # in the assignment file(s).
            if record.reference_id < 0:
                continue  # unmapped — no reference to match
            ref_name = in_bam.get_reference_name(record.reference_id)
            if ref_name not in all_tx_names:
                continue

            out_bam.write(record)
            written_records += 1
            found_uuids.add(uid_int)

    # ---- Report ----

    missing = all_uuids - found_uuids
    for uid_int in sorted(missing):
        original = uuid_to_str.get(uid_int, str(uid_int))
        print(f"Warning: read '{original}' found in assignment "
              f"but not in BAM.", file=sys.stderr)

    print(f"Done. Wrote {written_records} alignments from "
          f"{len(found_uuids)} unique reads "
          f"(scanned {total_records} BAM records).", file=sys.stderr)
    if missing:
        print(f"  {len(missing)} read(s) present in assignment files "
              f"were not found in the BAM.", file=sys.stderr)


if __name__ == "__main__":
    main()
