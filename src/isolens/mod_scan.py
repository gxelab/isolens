#!/usr/bin/env python3
"""mod_scan: Generate HDF5 transcript-specific read x position modification matrices.

Part of the isolens toolkit. See notebooks/01_mod.md for the full specification.
"""

import argparse
import sys
import uuid
from collections import defaultdict

import h5py
import numpy as np
import pysam

try:
    from isolens._parsing import parse_oarfish
except ImportError:
    from _parsing import parse_oarfish  # running as standalone script

# ---------- matrix encoding constants ----------

CODE_UNCOVERED = 0
CODE_CANONICAL = 1
CODE_MISMATCH = 2
CODE_DELETION = 3
# modification types start at 4

# ---------- CIGAR operator constants (pysam cigartuples) ----------

_BAM_CMATCH = 0  # M
_BAM_CINS = 1  # I
_BAM_CDEL = 2  # D
_BAM_CREF_SKIP = 3  # N
_BAM_CSOFT_CLIP = 4  # S
_BAM_CHARD_CLIP = 5  # H
_BAM_CPAD = 6  # P
_BAM_CEQUAL = 7  # =
_BAM_CDIFF = 8  # X


# ---------- CIGAR parsing ----------


def parse_cigar_for_row(record, tx_length):
    """Build a uint8 matrix row and read-to-transcript position map from CIGAR.

    Args:
        record: ``pysam.AlignedSegment``.
        tx_length: int — length of the target transcript in bases.

    Returns:
        (row, read_to_tx_map) where:
        - *row* is a ``numpy.ndarray`` of shape ``(tx_length,)``, dtype uint8,
          filled with states (0=uncovered, 1=match, 2=mismatch, 3=deletion).
        - *read_to_tx_map* is a ``list[int | None]`` of the same length as
          ``record.query_alignment_sequence``.  Each entry is the 1-based
          transcript position or ``None`` (for insertions / soft-clipped bases).
    """
    row = np.zeros(tx_length, dtype=np.uint8)
    read_to_tx_map = []

    ref_pos = record.reference_start  # 0-based
    read_pos = 0

    # Validate reference_start
    if ref_pos is None:
        return row, read_to_tx_map

    for op, length in record.cigartuples or []:
        if op == _BAM_CEQUAL:  # =
            for _ in range(length):
                if 0 <= ref_pos < tx_length:
                    row[ref_pos] = CODE_CANONICAL
                    read_to_tx_map.append(ref_pos + 1)  # 1-based
                else:
                    read_to_tx_map.append(None)
                ref_pos += 1
                read_pos += 1

        elif op == _BAM_CDIFF:  # X
            for _ in range(length):
                if 0 <= ref_pos < tx_length:
                    row[ref_pos] = CODE_MISMATCH
                    read_to_tx_map.append(ref_pos + 1)
                else:
                    read_to_tx_map.append(None)
                ref_pos += 1
                read_pos += 1

        elif op == _BAM_CMATCH:  # M (legacy — no =/X distinction)
            for _ in range(length):
                if 0 <= ref_pos < tx_length:
                    row[ref_pos] = CODE_CANONICAL  # best-effort
                    read_to_tx_map.append(ref_pos + 1)
                else:
                    read_to_tx_map.append(None)
                ref_pos += 1
                read_pos += 1

        elif op == _BAM_CDEL:  # D
            for _ in range(length):
                if 0 <= ref_pos < tx_length:
                    row[ref_pos] = CODE_DELETION
                ref_pos += 1
            # No entries in read_to_tx_map — deletions have no read base

        elif op == _BAM_CINS:  # I
            for _ in range(length):
                read_to_tx_map.append(None)
                read_pos += 1

        elif op == _BAM_CSOFT_CLIP:  # S
            read_pos += length
            for _ in range(length):
                read_to_tx_map.append(None)

        elif op == _BAM_CREF_SKIP:  # N (intron / splice junction)
            ref_pos += length

        elif op in (_BAM_CHARD_CLIP, _BAM_CPAD):  # H, P
            pass  # consume nothing

    return row, read_to_tx_map


# ---------- modification parsing ----------


def parse_modifications(record, row, read_to_tx_map, mod_cutoff_u8,
                        mod_code_map, seen_mod_types):
    """Parse MM/ML tags and override *row* in-place with modification codes.

    Args:
        record: ``pysam.AlignedSegment``.
        row: ``numpy.ndarray`` of shape ``(tx_length,)``, dtype uint8.
            Modified in-place — positions that pass the probability threshold
            are overwritten with the corresponding modification code (≥4).
        read_to_tx_map: ``list[int | None]`` — 1-based transcript positions
            indexed by read position (same length as ``query_alignment_sequence``).
        mod_cutoff_u8: int — raw ML threshold in 0-255 space
            (e.g. ``round(0.95 * 255) = 242``).
        mod_code_map: ``dict[str, int]`` — mutated in-place.
            Maps modification type string → integer code (starting at 4).
        seen_mod_types: ``set[str]`` — mutated in-place.
            Set of all modification type strings encountered.
    """
    # Read MM tag (prefer uppercase, fall back to lowercase)
    mm_str = None
    if record.has_tag("MM"):
        mm_str = record.get_tag("MM")
    elif record.has_tag("mm"):
        mm_str = record.get_tag("mm")

    if not mm_str:
        return

    # Read ML tag (prefer uppercase, fall back to lowercase)
    ml_bytes = None
    if record.has_tag("ML"):
        ml_bytes = record.get_tag("ML")
    elif record.has_tag("ml"):
        ml_bytes = record.get_tag("ml")

    # query_alignment_sequence recovers the sequence block even on
    # secondary / supplementary entries
    seq = record.query_alignment_sequence
    if seq is None:
        return

    total_mod_instance_idx = 0

    for mod_group in mm_str.split(";"):
        if not mod_group:
            continue
        parts = mod_group.split(",")
        if not parts:
            continue

        meta = parts[0]
        if len(meta) < 3:
            continue
        target_base = meta[0]
        mod_type = meta[2:].rstrip(".")
        seen_mod_types.add(mod_type)

        # Assign a stable integer code for this modification type
        if mod_type not in mod_code_map:
            mod_code_map[mod_type] = len(mod_code_map) + 4  # 4, 5, 6, ...

        try:
            skips = [int(s) for s in parts[1:]]
        except ValueError:
            continue

        skip_idx = 0
        current_skip = skips[skip_idx] if skip_idx < len(skips) else None
        occurrences_found = 0

        for read_pos_0 in range(len(seq)):
            if seq[read_pos_0] == target_base:
                if current_skip is not None and occurrences_found == current_skip:
                    passes_cutoff = True

                    if ml_bytes is not None and total_mod_instance_idx < len(ml_bytes):
                        raw_prob = ml_bytes[total_mod_instance_idx]
                        if raw_prob < mod_cutoff_u8:
                            passes_cutoff = False

                    if passes_cutoff and read_pos_0 < len(read_to_tx_map):
                        tx_pos_1 = read_to_tx_map[read_pos_0]
                        if tx_pos_1 is not None:
                            tx_pos_0 = tx_pos_1 - 1  # convert to 0-based
                            if 0 <= tx_pos_0 < len(row):
                                row[tx_pos_0] = mod_code_map[mod_type]

                    total_mod_instance_idx += 1
                    skip_idx += 1
                    current_skip = (
                        skips[skip_idx] if skip_idx < len(skips) else None
                    )
                    occurrences_found = 0
                else:
                    occurrences_found += 1


# ---------- HDF5 output ----------


def write_transcript_group(h5, tx_name, rows, read_ids, weights):
    """Write one transcript's matrix and metadata into an open HDF5 file.

    Args:
        h5: ``h5py.File`` open for writing.
        tx_name: str — transcript name (used as group name).
        rows: ``list[numpy.ndarray]`` — each of shape ``(tx_length,)``, dtype uint8.
        read_ids: ``list[str]`` — original read UUID strings.
        weights: ``list[float]`` — assignment probabilities.

    If *rows* is empty no group is created.
    """
    if not rows:
        return

    n_reads = len(rows)
    tx_length = rows[0].shape[0]

    # Verify all rows have the same length
    for r in rows:
        if r.shape[0] != tx_length:
            raise ValueError(
                f"Row length mismatch for transcript '{tx_name}': "
                f"expected {tx_length}, got {r.shape[0]}"
            )

    grp = h5.create_group(f"transcripts/{tx_name}")

    # Stack rows into a contiguous 2D matrix
    matrix = np.stack(rows, axis=0)  # shape (n_reads, tx_length), dtype uint8

    # Chunk rows based on transcript length
    if tx_length > 10000:
        chunk_rows = min(512, max(1, n_reads))
    elif tx_length > 1000:
        chunk_rows = min(1024, max(1, n_reads))
    else:
        chunk_rows = min(4096, max(1, n_reads))

    grp.create_dataset(
        "matrix",
        data=matrix,
        dtype=np.uint8,
        compression="gzip",
        shuffle=True,
        chunks=(chunk_rows, tx_length),
    )

    # Variable-length UTF-8 strings
    grp.create_dataset(
        "read_ids",
        data=np.array(read_ids, dtype=h5py.string_dtype()),
    )

    grp.create_dataset(
        "read_weights",
        data=np.array(weights, dtype=np.float32),
        compression="gzip",
    )


# ---------- CLI ----------


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="mod_scan: Generate HDF5 read x position modification matrices"
    )
    parser.add_argument(
        "-b", "--bam",
        required=True,
        help="Path to transcriptome BAM alignment file",
    )
    parser.add_argument(
        "-p", "--oarfish",
        required=True,
        help="Path to Oarfish isoform assignment probability file (.lz4)",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output HDF5 file path",
    )
    parser.add_argument(
        "-c", "--mod-cutoff",
        type=float,
        default=0.95,
        help="Modification probability cutoff [default: 0.95]",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print progress to stderr",
    )
    return parser.parse_args()


# ---------- main pipeline ----------


def main():
    args = parse_args()
    mod_cutoff_u8 = round(args.mod_cutoff * 255.0)

    # ---- 1. Load Oarfish assignments ----

    if args.verbose:
        print("[mod_scan] Loading Oarfish assignments into memory...", file=sys.stderr)

    tx_names, prob_map, name_to_id = parse_oarfish(args.oarfish)

    if args.verbose:
        print(f"[mod_scan] Loaded {len(tx_names)} transcripts, "
              f"{len(prob_map)} reads with assignments", file=sys.stderr)

    # ---- 2. Open BAM and map references ----

    bam = pysam.AlignmentFile(args.bam, "rb")

    # Build lookups from BAM reference index to Oarfish transcript ID and length
    bam_ref_to_tx_id = [name_to_id.get(ref, None) for ref in bam.references]
    # Also store transcript length per Oarfish tx_id for HDF5 writing
    tx_lengths = {}
    for i, ref in enumerate(bam.references):
        tx_id = name_to_id.get(ref)
        if tx_id is not None:
            tx_lengths[tx_id] = bam.lengths[i]

    # ---- 3. Per-transcript accumulation ----

    # mod_code_map: {mod_type_string: integer_code_starting_at_4}
    mod_code_map = {}
    seen_mod_types = set()

    # reads_by_tx[tx_id] = [(read_id_str, row_uint8, weight_float), ...]
    reads_by_tx = defaultdict(list)

    total_records = 0
    matched_reads = 0

    for record in bam:
        total_records += 1
        if args.verbose and total_records % 500_000 == 0:
            print(f"[mod_scan] Scanned {total_records} alignments...", file=sys.stderr)

        if record.is_unmapped:
            continue

        # Look up read in Oarfish by UUID
        try:
            read_id_int = uuid.UUID(record.query_name).int
        except ValueError:
            continue

        assignments = prob_map.get(read_id_int)
        if not assignments:
            continue

        tx_index = record.reference_id
        if tx_index is None or tx_index < 0:
            continue

        tx_id = bam_ref_to_tx_id[tx_index]
        if tx_id is None:
            continue

        # Find the exact assignment for this transcript
        assignment = next((a for a in assignments if a.tx_id == tx_id), None)
        if not assignment:
            continue

        matched_reads += 1

        # ---- Build matrix row ----
        tx_length = bam.lengths[tx_index]
        row, read_to_tx_map = parse_cigar_for_row(record, tx_length)
        parse_modifications(
            record, row, read_to_tx_map,
            mod_cutoff_u8, mod_code_map, seen_mod_types,
        )

        reads_by_tx[tx_id].append(
            (record.query_name, row, assignment.prob)
        )

    bam.close()

    if args.verbose:
        print(f"[mod_scan] Total alignments scanned: {total_records}", file=sys.stderr)
        print(f"[mod_scan] Reads matched to Oarfish assignments: {matched_reads}",
              file=sys.stderr)
        print(f"[mod_scan] Transcripts with reads: {len(reads_by_tx)}", file=sys.stderr)
        print(f"[mod_scan] Modification types found: {sorted(seen_mod_types)}",
              file=sys.stderr)
        print(f"[mod_scan] Modification code map: {mod_code_map}", file=sys.stderr)

    # ---- 4. Write HDF5 ----

    if args.verbose:
        print("[mod_scan] Writing HDF5 output...", file=sys.stderr)

    with h5py.File(args.output, "w") as h5:
        # Global /modification_codes — stored as attributes on a group
        codes_grp = h5.create_group("modification_codes")
        for mod_type, code in sorted(mod_code_map.items(), key=lambda x: x[1]):
            codes_grp.attrs[mod_type] = code

        # Global /metadata
        meta = h5.create_group("metadata")
        meta.attrs["mod_cutoff"] = args.mod_cutoff
        meta.attrs["pipeline_version"] = "0.1.0"
        meta.attrs["n_transcripts"] = len(reads_by_tx)
        meta.attrs["n_assignments"] = sum(len(v) for v in reads_by_tx.values())
        meta.attrs["modification_codes"] = str(
            dict(sorted(mod_code_map.items(), key=lambda x: x[1]))
        )

        # Per-transcript groups (write one at a time)
        written = 0
        n_total = len(reads_by_tx)

        for tx_id, read_data in reads_by_tx.items():
            tx_name = tx_names[tx_id]
            read_id_strs = [d[0] for d in read_data]
            rows = [d[1] for d in read_data]
            weights = [d[2] for d in read_data]

            write_transcript_group(h5, tx_name, rows, read_id_strs, weights)

            written += 1
            if args.verbose and written % 1000 == 0:
                print(f"[mod_scan] Wrote {written}/{n_total} transcript groups...",
                      file=sys.stderr)

    if args.verbose:
        print(f"[mod_scan] Done. Output written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
