"""Shared parsing utilities for the isolens pipeline.

Used by mod_scan.py, polya_calc.py, and downstream analysis modules.
"""

import hashlib
import uuid

import lz4.frame


class TargetAssignment:
    """Lightweight struct for a single (transcript_id, probability) pair."""

    __slots__ = ["tx_id", "prob"]

    def __init__(self, tx_id: int, prob: float):
        self.tx_id = tx_id
        self.prob = prob


def read_id_to_int(read_id_str: str) -> int:
    """Convert a read name to a 128-bit integer for memory-efficient lookups.

    First tries UUID parsing. If *read_id_str* is not a valid UUID, falls
    back to MD5 hashing.
    """
    try:
        return uuid.UUID(read_id_str).int
    except ValueError:
        return int(hashlib.md5(read_id_str.encode("utf-8")).hexdigest(), 16)


def parse_oarfish(path):
    """Parse an LZ4-compressed Oarfish assignment probability file.

    Returns:
        tx_names: list[str] — transcript names, index = Oarfish internal tx_id.
        prob_map: dict[int, list[TargetAssignment]] —
            keyed by ``read_id_to_int(read_name)``.
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
