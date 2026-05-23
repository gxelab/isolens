#!/usr/bin/env python3
import sys
import argparse
import gzip
import uuid
from collections import defaultdict
import lz4.frame
import pysam

# --- Custom Internal Structures ---
class TargetAssignment:
    __slots__ = ['tx_id', 'prob']
    def __init__(self, tx_id: int, prob: float):
        self.tx_id = tx_id
        self.prob = prob

class PositionStats:
    __slots__ = ['n_read', 'sum_probs', 'n_nomod', 'wt_nomod', 'mods']
    def __init__(self):
        self.n_read = 0
        self.sum_probs = 0.0
        self.n_nomod = 0
        self.wt_nomod = 0.0
        # Maps modification type string -> [count, weighted_count]
        self.mods = {}

class PolyAStats:
    __slots__ = ['n_reads', 'sum_weights', 'pa_reads', 'pa_weights', 'sum_weighted_pa_len', 'probs', 'pa_lens']
    def __init__(self):
        self.n_reads = 0
        self.sum_weights = 0.0
        self.pa_reads = 0
        self.pa_weights = 0.0
        self.sum_weighted_pa_len = 0.0
        self.probs = []
        self.pa_lens = []

def parse_oarfish(path):
    """
    Parses the LZ4 compressed Oarfish file.
    Converts UUID read names to 128-bit integers to minimize RAM footprint.
    """
    tx_names = []
    name_to_id = {}
    prob_map = {}

    with lz4.frame.open(path, 'rb') as f:
        header_line = f.readline().decode('utf-8').strip()
        if not header_line:
            raise ValueError("Empty Oarfish allocation file.")

        num_transcripts = int(header_line.split()[0])

        for i in range(num_transcripts):
            tx_name = f.readline().decode('utf-8').strip()
            name_to_id[tx_name] = i
            tx_names.append(tx_name)

        for line in f:
            tokens = line.decode('utf-8').strip().split()
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

def map_read_to_tx(aligned_pairs, tx_start_0):
    """
    Maps read sequence alignment pairings.
    Returns a list mapping read positions (0-indexed) to transcript positions (1-indexed).
    """
    read_positions = []
    for r_pos, ref_pos in aligned_pairs:
        if r_pos is not None:
            tx_pos_1 = (ref_pos + 1) if ref_pos is not None else None
            read_positions.append(tx_pos_1)
    return read_positions

def create_writer(path, use_gzip):
    if use_gzip:
        return gzip.open(path, 'wt', encoding='utf-8')
    return open(path, 'w', encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description="isolens: High-performance modification pipeline (Python Edition)")
    parser.add_argument("-b", "--bam", required=True, help="Path to input BAM alignment file")
    parser.add_argument("-p", "--prob", required=True, help="Path to oarfish assignment probability map (.lz4)")
    parser.add_argument("-m", "--mod", required=True, help="Output TSV path for positional modification summary")
    parser.add_argument("-a", "--polya", required=True, help="Output TSV path for poly(A) statistics summary")
    parser.add_argument("--out-per-base", help="Optional path to write raw per-read modification lines")
    parser.add_argument("-t", "--threshold", type=float, default=0.95, help="Modification probability threshold")
    parser.add_argument("-z", "--gzip", action="store_true", help="Compress outputs using gzip")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show progression metrics")
    args = parser.parse_args()

    ml_threshold_u8 = round(args.threshold * 255.0)

    if args.verbose:
        print("[isolens] Loading Oarfish allocations into memory...", file=sys.stderr)
    tx_names, probabilities, name_to_id = parse_oarfish(args.prob)

    if args.verbose:
        print("[isolens] Processing BAM alignments...", file=sys.stderr)

    bam_file = pysam.AlignmentFile(args.bam, "rb")
    bam_ref_to_tx_id = [name_to_id.get(ref, None) for ref in bam_file.references]

    modification_summary = defaultdict(PositionStats)
    poly_a_summary = defaultdict(PolyAStats)
    seen_mod_types = set()

    per_base_writer = None
    if args.out_per_base:
        per_base_writer = create_writer(args.out_per_base, args.gzip)
        per_base_writer.write("read_id\tpos_read\tmod_type\tx_name\tprob\tpos_tx\tmod_prob\n")

    total_records = 0

    for record in bam_file:
        total_records += 1
        if args.verbose and total_records % 1000000 == 0:
            print(f"[isolens] Audited {total_records} alignments...", file=sys.stderr)

        if record.is_unmapped:
            continue

        try:
            read_id_int = uuid.UUID(record.query_name).int
        except ValueError:
            continue

        assignments = probabilities.get(read_id_int)
        if not assignments:
            continue

        tx_index = record.reference_id
        if tx_index < 0:
            continue

        tx_id = bam_ref_to_tx_id[tx_index]
        if tx_id is None:
            continue

        assignment = next((a for a in assignments if a.tx_id == tx_id), None)
        if not assignment:
            continue

        # --- Core Process 1: Track Poly(A) Tail Features ---
        pa_length = -1
        if record.has_tag("pt"):
            pa_length = int(record.get_tag("pt"))

        pa_entry = poly_a_summary[tx_id]
        pa_entry.n_reads += 1
        pa_entry.sum_weights += assignment.prob
        pa_entry.probs.append(assignment.prob)
        pa_entry.pa_lens.append(pa_length)
        if pa_length > 0:
            pa_entry.pa_reads += 1
            pa_entry.pa_weights += assignment.prob
            pa_entry.sum_weighted_pa_len += float(pa_length) * assignment.prob

        # --- Core Process 2: Parse Alignments and Modification Spaces ---
        mm_str = None
        if record.has_tag("MM"):
            mm_str = record.get_tag("MM")
        elif record.has_tag("mm"):
            mm_str = record.get_tag("mm")

        ml_bytes = None
        if record.has_tag("ML"):
            ml_bytes = record.get_tag("ML")
        elif record.has_tag("ml"):
            ml_bytes = record.get_tag("ml")

        # Use matches_only=True to ensure alignment indexing lines up with query_alignment_sequence
        aligned_pairs = record.get_aligned_pairs(matches_only=True)
        read_to_tx_map = map_read_to_tx(aligned_pairs, record.reference_start)

        for tx_pos_1 in read_to_tx_map:
            if tx_pos_1 is not None:
                coord_stats = modification_summary[(tx_id, tx_pos_1)]
                coord_stats.n_read += 1
                coord_stats.sum_probs += assignment.prob

        if mm_str:
            # query_alignment_sequence safely recovers the sequence block on secondary/supplementary entries
            seq = record.query_alignment_sequence
            if seq is None:
                continue

            total_mod_instance_idx = 0
            modified_positions_in_read = {}

            for mod_group in mm_str.split(';'):
                if not mod_group:
                    continue
                parts = mod_group.split(',')
                if not parts:
                    continue

                meta = parts[0]
                if len(meta) < 3:
                    continue
                target_base = meta[0]
                mod_type = meta[2:].rstrip('.')
                seen_mod_types.add(mod_type)

                try:
                    skips = [int(s) for s in parts[1:]]
                except ValueError:
                    continue

                skip_idx = 0
                current_skip = skips[skip_idx] if skip_idx < len(skips) else None
                occurrences_found = 0

                for read_pos_0 in range(len(seq)):
                    if seq[read_pos_0] == target_base:
                        if current_skip is not None:
                            if occurrences_found == current_skip:
                                passes_cutoff = True
                                base_prob = 1.0

                                if ml_bytes is not None:
                                    if total_mod_instance_idx < len(ml_bytes):
                                        raw_prob = ml_bytes[total_mod_instance_idx]
                                        if raw_prob < ml_threshold_u8:
                                            passes_cutoff = False
                                        base_prob = raw_prob / 255.0

                                if read_pos_0 < len(read_to_tx_map):
                                    tx_pos_1 = read_to_tx_map[read_pos_0]
                                    if tx_pos_1 is not None:
                                        if passes_cutoff:
                                            modified_positions_in_read[(tx_pos_1, mod_type)] = assignment.prob

                                            # Writing raw lines specifically when passing cutoff criteria
                                            if per_base_writer:
                                                per_base_writer.write(
                                                    f"{record.query_name}\t{read_pos_0 + 1}\t{mod_type}\t"
                                                    f"{tx_names[tx_id]}\t{assignment.prob:.4f}\t{tx_pos_1}\t{base_prob:.4f}\n"
                                                )

                                total_mod_instance_idx += 1
                                skip_idx += 1
                                current_skip = skips[skip_idx] if skip_idx < len(skips) else None
                                occurrences_found = 0
                            else:
                                occurrences_found += 1

            for tx_pos_1 in read_to_tx_map:
                if tx_pos_1 is not None:
                    coord_stats = modification_summary[(tx_id, tx_pos_1)]
                    matched_any_mod = False
                    for m_type in seen_mod_types:
                        read_weight = modified_positions_in_read.get((tx_pos_1, m_type))
                        if read_weight is not None:
                            if m_type not in coord_stats.mods:
                                coord_stats.mods[m_type] = [0, 0.0]
                            coord_stats.mods[m_type][0] += 1
                            coord_stats.mods[m_type][1] += read_weight
                            matched_any_mod = True

                    if not matched_any_mod:
                        coord_stats.n_nomod += 1
                        coord_stats.wt_nomod += assignment.prob
        else:
            for tx_pos_1 in read_to_tx_map:
                if tx_pos_1 is not None:
                    coord_stats = modification_summary[(tx_id, tx_pos_1)]
                    coord_stats.n_nomod += 1
                    coord_stats.wt_nomod += assignment.prob

    bam_file.close()
    if per_base_writer:
        per_base_writer.close()

    # --- Output Generation 1: Positional Summaries ---
    if args.verbose:
        print("[isolens] Writing Output 1 (Positional Summaries)...", file=sys.stderr)

    sorted_mod_types = sorted(list(seen_mod_types))
    out1_header = "tx_name\ttx_pos\tn_read\tsum_probs\tn_nomod\twt_nomod"
    for m_type in sorted_mod_types:
        out1_header += f"\tn_{m_type.lower()}\twt_{m_type.lower()}"

    with create_writer(args.mod, args.gzip) as out1:
        out1.write(out1_header + "\n")
        for (tx_id, pos), stats in modification_summary.items():
            tx_name = tx_names[tx_id]
            line = f"{tx_name}\t{pos}\t{stats.n_read}\t{stats.sum_probs:.4f}\t{stats.n_nomod}\t{stats.wt_nomod:.4f}"
            for m_type in sorted_mod_types:
                mod_vals = stats.mods.get(m_type, [0, 0.0])
                line += f"\t{mod_vals[0]}\t{mod_vals[1]:.4f}"
            out1.write(line + "\n")

    # --- Output Generation 2: Poly(A) Tail Distribution Summaries ---
    if args.verbose:
        print("[isolens] Writing Output 2 (PolyA Profiles)...", file=sys.stderr)

    with create_writer(args.polya, args.gzip) as out2:
        out2.write("tx_name\tn_reads\tsum_weights\tpa_reads\tpa_weights\tpa_wlen\tprobs\tpa_lens\n")
        for tx_id, stats in poly_a_summary.items():
            tx_name = tx_names[tx_id]
            weighted_avg_len = (stats.sum_weighted_pa_len / stats.pa_weights) if stats.pa_weights > 0.0 else 0.0

            probs_str = ",".join(f"{p:.4f}" for p in stats.probs)
            lens_str = ",".join(str(l) for l in stats.pa_lens)

            out2.write(
                f"{tx_name}\t{stats.n_reads}\t{stats.sum_weights:.4f}\t{stats.pa_reads}\t"
                f"{stats.pa_weights:.4f}\t{weighted_avg_len:.2f}\t{probs_str}\t{lens_str}\n"
            )

if __name__ == "__main__":
    main()
