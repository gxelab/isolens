"""Shared parsing utilities for the isolens pipeline.

Used by mod_scan.py, polya_calc.py, and downstream analysis modules.
"""

import gzip
import hashlib
import io
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
    """Parse an Oarfish assignment probability file (LZ4-compressed or
    plain text).

    Args:
        path: Path to the Oarfish file.  If the path ends with ``.lz4``
            it is treated as LZ4-compressed; otherwise as plain text.

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
    # Detect format by suffix and read content
    if path.endswith(".lz4"):
        with lz4.frame.open(path, "rb") as raw_f:
            content = raw_f.read().decode("utf-8")
    else:
        with open(path, encoding="utf-8") as raw_f:
            content = raw_f.read()

    tx_names: list[str] = []
    name_to_id: dict[str, int] = {}
    prob_map: dict[int, list[TargetAssignment]] = {}

    with io.StringIO(content) as f:
        header_line = f.readline().strip()
        if not header_line:
            raise ValueError("Empty Oarfish allocation file.")

        num_transcripts = int(header_line.split()[0])

        for i in range(num_transcripts):
            tx_name = f.readline().strip()
            name_to_id[tx_name] = i
            tx_names.append(tx_name)

        for line in f:
            tokens = line.strip().split()
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


def calc_weighted_pa_len(weights: list[float], lengths: list[int]) -> float:
    """Compute the assignment-probability-weighted poly(A) tail length.

    Args:
        weights: Oarfish assignment probabilities (one per read).
        lengths: Raw poly(A) tail lengths (one per read, same order).

    Returns:
        Weighted average poly(A) length, or 0.0 if the sum of
        weights is zero.
    """
    if not weights:
        return 0.0
    sum_wt = sum(weights)
    if sum_wt <= 0:
        return 0.0
    return sum(w * pl for w, pl in zip(weights, lengths)) / sum_wt


def parse_polyA_file(filename: str) -> tuple[str, dict[str, dict[str, Any]]]:
    """Parse a poly(A) TSV file and return ``(id_column_name, data_dict)``.

    Handles both transcript-level (``transcript_id`` column) and gene-level
    (``gene_id`` column) input formats.  Auto-detects gzip by ``.gz``
    suffix.

    Args:
        filename: Path to the input TSV or TSV.GZ file.

    Returns:
        ``(id_col_name, data_dict)`` where *data_dict* maps feature IDs
        to dicts with keys ``n_reads``, ``total_wt``, ``wmlen``,
        ``weights``, ``lengths``.
    """
    print(f"Loading data from {filename}...", file=sys.stderr)
    data_dict: dict[str, dict[str, Any]] = {}

    with open_by_suffix(filename, "rt" if filename.endswith(".gz") else "r") as f:
        header = f.readline().strip().split("\t")

        # Detect whether transcript-level or gene-level output
        id_col_name = "transcript_id" if "transcript_id" in header else "gene_id"
        id_col = header.index(id_col_name)
        weights_col = header.index("weights")
        lengths_col = header.index("lengths")

        for line in f:
            parts = line.strip().split("\t")
            if len(parts) <= max(weights_col, lengths_col):
                continue

            feature_id = parts[id_col]
            weights = np.array([float(p) for p in parts[weights_col].split(",")])
            lengths = np.array(
                [int(pa_len) for pa_len in parts[lengths_col].split(",")]
            )

            n_reads = len(weights)
            total_wt = float(np.sum(weights))
            wmlen = float(np.sum(weights * lengths) / total_wt) if total_wt > 0 else 0.0

            data_dict[feature_id] = {
                "n_reads": n_reads,
                "total_wt": total_wt,
                "wmlen": wmlen,
                "weights": weights,
                "lengths": lengths,
            }

    return id_col_name, data_dict
