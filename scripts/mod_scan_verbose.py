#!/usr/bin/env python3
"""Verbose per-position debug output for mod_scan.

For each mapped read with an Oarfish assignment, prints one TSV row per
read position that maps to a transcript position:

    tx_name  tx_pos  read_name  asp  read_pos  base  is_modified  mod_type  mod_prob

tx_pos and read_pos are 1-based.

Standalone — no isolens import required.
"""

import argparse
import sys
import uuid

import lz4.frame
import pysam


# ---------- Oarfish parsing (inlined from isolens._parsing) ----------


class TargetAssignment:
    """Lightweight struct for a single (transcript_id, probability) pair."""

    __slots__ = ["tx_id", "prob"]

    def __init__(self, tx_id: int, prob: float):
        self.tx_id = tx_id
        self.prob = prob


def parse_oarfish(path):
    """Parse an LZ4-compressed Oarfish assignment probability file.

    Returns:
        tx_names: list[str] — transcript names, index = Oarfish internal tx_id.
        prob_map: dict[int, list[TargetAssignment]] —
            keyed by ``uuid.UUID(read_name).int``.
        name_to_id: dict[str, int] — transcript name → Oarfish tx_id.
    """
    tx_names = []
    name_to_id = {}
    prob_map = {}

    with lz4.frame.open(path, "rb") as f:
        header_line = f.readline().decode("utf-8").strip()
        if not header_line:
            raise ValueError("Empty Oarfish allocation file.")

        num_transcripts = int(header_line.split()[0])

        for i in range(num_transcripts):
            tx_name = f.readline().decode("utf-8").strip()
            name_to_id[tx_name] = i
            tx_names.append(tx_name)

        for line in f:
            tokens = line.decode("utf-8").strip().split()
            if not tokens:
                continue

            read_id_int = uuid.UUID(tokens[0]).int
            num_targets = int(tokens[1])

            target_ids = tokens[2 : 2 + num_targets]
            probs = tokens[2 + num_targets : 2 + (2 * num_targets)]

            assignments = []
            for t_id, p_val in zip(target_ids, probs):
                assignments.append(TargetAssignment(int(t_id), float(p_val)))

            prob_map[read_id_int] = assignments

    return tx_names, prob_map, name_to_id


# ---------- CIGAR operator constants ----------

_BAM_CMATCH = 0      # M
_BAM_CINS = 1        # I
_BAM_CDEL = 2        # D
_BAM_CREF_SKIP = 3   # N
_BAM_CSOFT_CLIP = 4  # S
_BAM_CHARD_CLIP = 5  # H
_BAM_CPAD = 6        # P
_BAM_CEQUAL = 7      # =
_BAM_CDIFF = 8       # X


# ---------- CIGAR parsing (adapted from isolens.mod_scan) ----------


def parse_cigar_for_row(record, tx_length):
    """Build read-to-transcript position map from CIGAR.

    Returns:
        read_to_tx_map: list[int | None] — same length as
            ``record.query_sequence``.  Each entry is the 1-based
            transcript position or ``None`` (insertion / soft-clipped base).
    """
    read_to_tx_map = []

    ref_pos = record.reference_start  # 0-based
    if ref_pos is None:
        return read_to_tx_map

    for op, length in record.cigartuples or []:
        if op in (_BAM_CEQUAL, _BAM_CDIFF, _BAM_CMATCH):
            n_before = max(0, -min(ref_pos, 0))
            valid_start = max(ref_pos, 0)
            valid_end = min(ref_pos + length, tx_length)
            n_valid = max(0, valid_end - valid_start)
            n_after = length - n_before - n_valid

            read_to_tx_map.extend([None] * n_before)
            if n_valid > 0:
                read_to_tx_map.extend(range(valid_start + 1, valid_end + 1))
            read_to_tx_map.extend([None] * n_after)

            ref_pos += length

        elif op == _BAM_CDEL:
            ref_pos += length
            # No entries — deletions consume reference but not read

        elif op == _BAM_CINS:
            read_to_tx_map.extend([None] * length)

        elif op == _BAM_CSOFT_CLIP:
            read_to_tx_map.extend([None] * length)

        elif op == _BAM_CREF_SKIP:
            ref_pos += length

        elif op in (_BAM_CHARD_CLIP, _BAM_CPAD):
            pass

    return read_to_tx_map


# ---------- modification parsing (verbose variant) ----------


def parse_modifications_verbose(record, read_to_tx_map, mod_cutoff):
    """Parse MM/ML tags and return per-position modification details.

    Args:
        record: ``pysam.AlignedSegment``.
        read_to_tx_map: ``list[int | None]`` — from ``parse_cigar_for_row``.
        mod_cutoff: float — probability threshold (0.0–1.0).

    Returns:
        dict[int, tuple[str, float]] —
            keyed by 0-based read position,
            value = (mod_type_string, probability_float).
    """
    mod_dict = {}

    # Read MM tag
    mm_str = None
    if record.has_tag("MM"):
        mm_str = record.get_tag("MM")
    elif record.has_tag("mm"):
        mm_str = record.get_tag("mm")

    if not mm_str:
        return mod_dict

    # Read ML tag
    ml_bytes = None
    if record.has_tag("ML"):
        ml_bytes = record.get_tag("ML")
    elif record.has_tag("ml"):
        ml_bytes = record.get_tag("ml")

    seq = record.query_sequence
    if seq is None:
        return mod_dict

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

        try:
            skips = [int(s) for s in parts[1:]]
        except ValueError:
            continue

        # Find all positions of target_base in the read sequence
        positions = []
        pos = seq.find(target_base)
        while pos != -1:
            positions.append(pos)
            pos = seq.find(target_base, pos + 1)

        # Apply skip pattern
        occ_idx = 0
        for _skip_val in skips:
            occ_idx += _skip_val
            if occ_idx >= len(positions):
                break

            read_pos_0 = positions[occ_idx]

            # Check ML probability threshold
            prob = 1.0  # default if no ML tag
            if ml_bytes is not None and total_mod_instance_idx < len(ml_bytes):
                raw = ml_bytes[total_mod_instance_idx]
                prob = raw / 255.0

            if prob >= mod_cutoff and read_pos_0 < len(read_to_tx_map):
                tx_pos_1 = read_to_tx_map[read_pos_0]
                if tx_pos_1 is not None:
                    mod_dict[read_pos_0] = (mod_type, prob)

            total_mod_instance_idx += 1
            occ_idx += 1

    return mod_dict


# ---------- main ----------


def parse_args():
    parser = argparse.ArgumentParser(
        description="mod_scan_verbose: per-position debug output for mod_scan"
    )
    parser.add_argument(
        "-b", "--bam", required=True,
        help="Path to transcriptome BAM alignment file",
    )
    parser.add_argument(
        "-a", "--oarfish", required=True,
        help="Path to Oarfish isoform assignment probability file (.lz4)",
    )
    parser.add_argument(
        "-c", "--mod-cutoff", type=float, default=0.95,
        help="Modification probability cutoff [default: 0.95]",
    )
    parser.add_argument(
        "-m", "--min-asp", type=float, default=0.0,
        help="Minimum Oarfish assignment probability [default: 0.0 (no filter)]",
    )
    parser.add_argument(
        "--header", action="store_true",
        help="Print a header line as the first row of output",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ---- 1. Load Oarfish assignments ----
    tx_names, prob_map, name_to_id = parse_oarfish(args.oarfish)

    # ---- 2. Open BAM and build reference lookup ----
    bam = pysam.AlignmentFile(args.bam, "rb")
    bam_ref_to_tx_id = [name_to_id.get(ref, None) for ref in bam.references]

    # ---- 3. Print header if requested ----
    if args.header:
        print(
            "tx_name", "tx_pos", "read_name", "asp",
            "read_pos", "base", "is_modified", "mod_type", "mod_prob",
            sep="\t",
        )

    # ---- 4. Stream through BAM ----
    for record in bam:
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
        assign_dict = {a.tx_id: a for a in assignments}
        assignment = assign_dict.get(tx_id)
        if not assignment:
            continue

        # Assignment probability filter
        if args.min_asp > 0.0 and assignment.prob < args.min_asp:
            continue

        tx_name = tx_names[tx_id]
        tx_length = bam.lengths[tx_index]
        seq = record.query_sequence
        if seq is None:
            continue

        # Parse CIGAR → read-to-transcript position map
        read_to_tx_map = parse_cigar_for_row(record, tx_length)

        # Parse modifications → read_pos → (mod_type, prob)
        mod_dict = parse_modifications_verbose(record, read_to_tx_map, args.mod_cutoff)

        # Output per-position rows
        for read_pos_0, tx_pos_1 in enumerate(read_to_tx_map):
            if tx_pos_1 is None:
                continue

            base = seq[read_pos_0]
            mod_info = mod_dict.get(read_pos_0)
            if mod_info is not None:
                mod_type, mod_prob = mod_info
                print(
                    tx_name, tx_pos_1, record.query_name,
                    f"{assignment.prob:.4f}",
                    read_pos_0 + 1,
                    base,
                    "True",
                    mod_type,
                    f"{mod_prob:.4f}",
                    sep="\t",
                )
            else:
                print(
                    tx_name, tx_pos_1, record.query_name,
                    f"{assignment.prob:.4f}",
                    read_pos_0 + 1,
                    base,
                    "False",
                    "",
                    "",
                    sep="\t",
                )

    bam.close()


if __name__ == "__main__":
    main()
