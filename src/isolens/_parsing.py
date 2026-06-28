"""Shared parsing utilities for the isolens pipeline.

Used by mod_scan.py, polya_calc.py, and downstream analysis modules.
"""

import gzip
import hashlib
import sys
import typing
import uuid
from typing import Any

import lz4.frame
import numpy as np


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


def parse_oarfish(
    path: str,
) -> tuple[list[str], dict[int, list[TargetAssignment]], dict[str, int]]:
    """Parse an LZ4-compressed Oarfish assignment probability file.

    Args:
        path: Path to the LZ4-compressed Oarfish ``.lz4`` file.

    Returns:
        A 3-tuple ``(tx_names, prob_map, name_to_id)`` where:

        * *tx_names*: ``list[str]`` — transcript names, index = Oarfish
          internal tx_id.
        * *prob_map*: ``dict[int, list[TargetAssignment]]`` — keyed by
          ``read_id_to_int(read_name)``.
        * *name_to_id*: ``dict[str, int]`` — transcript name → Oarfish tx_id.

    Raises:
        ValueError: If the file is empty.
    """
    tx_names: list[str] = []
    name_to_id: dict[str, int] = {}
    prob_map: dict[int, list[TargetAssignment]] = {}

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


# ---------- shared I/O utilities ----------


def open_by_suffix(path: str, mode: str = "r") -> typing.IO:
    """Open a file for reading or writing, auto-detecting gzip by suffix.

    Args:
        path: File path. If it ends with ``.gz``, ``gzip.open`` is used
            with the appropriate text/binary mode.
        mode: I/O mode (e.g. ``"r"``, ``"rt"``, ``"w"``, ``"wt"``).
            Default ``"r"``.

    Returns:
        A file-like object (``gzip.GzipFile`` for ``.gz`` paths,
        regular file handle otherwise).
    """
    if path.endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def calc_weighted_pa_len(probs: list[float], pa_lens: list[int]) -> float:
    """Compute the assignment-probability-weighted poly(A) tail length.

    Args:
        probs: Oarfish assignment probabilities (one per read).
        pa_lens: Raw poly(A) tail lengths (one per read, same order).

    Returns:
        Weighted average poly(A) length, or 0.0 if the sum of
        probabilities is zero.
    """
    if not probs:
        return 0.0
    sum_prob = sum(probs)
    if sum_prob <= 0:
        return 0.0
    return sum(p * pl for p, pl in zip(probs, pa_lens)) / sum_prob


def parse_polyA_file(filename: str) -> tuple[str, dict[str, dict[str, Any]]]:
    """Parse a poly(A) TSV file and return ``(id_column_name, data_dict)``.

    Handles both transcript-level (``transcript_id`` column) and gene-level
    (``gene_id`` column) input formats.  Auto-detects gzip by ``.gz``
    suffix.

    Args:
        filename: Path to the input TSV or TSV.GZ file.

    Returns:
        ``(id_col_name, data_dict)`` where *data_dict* maps feature IDs
        to dicts with keys ``n_reads``, ``pa_wlen``, ``probs``, ``pa_lens``.
    """
    print(f"Loading data from {filename}...", file=sys.stderr)
    data_dict: dict[str, dict[str, Any]] = {}

    with open_by_suffix(filename, "rt" if filename.endswith(".gz") else "r") as f:
        header = f.readline().strip().split("\t")

        # Detect whether transcript-level or gene-level output
        id_col_name = "transcript_id" if "transcript_id" in header else "gene_id"
        id_col = header.index(id_col_name)
        probs_col = header.index("probs")
        lens_col = header.index("pa_lens")

        for line in f:
            parts = line.strip().split("\t")
            if len(parts) <= max(probs_col, lens_col):
                continue

            feature_id = parts[id_col]
            probs = np.array([float(p) for p in parts[probs_col].split(",")])
            pa_lens = np.array(
                [int(pa_len) for pa_len in parts[lens_col].split(",")]
            )

            n_reads = len(probs)
            sum_prob = float(np.sum(probs))
            pa_wlen = (
                float(np.sum(probs * pa_lens) / sum_prob)
                if sum_prob > 0
                else 0.0
            )

            data_dict[feature_id] = {
                "n_reads": n_reads,
                "pa_wlen": pa_wlen,
                "probs": probs,
                "pa_lens": pa_lens,
            }

    return id_col_name, data_dict
