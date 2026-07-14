"""Shared parsing utilities for the isolens pipeline.

Used by mod_scan.py, polya_calc.py, and downstream analysis modules.
"""

import gzip
import hashlib
import io
import math
import sys
import typing
import uuid
from typing import Any

import lz4.frame
import numpy as np
import pyarrow.parquet as pq


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


def calc_weighted_pa_len(
    weights: list[float], lengths: list[int], use_log: bool = False
) -> float:
    """Compute the assignment-probability-weighted poly(A) tail length.

    Args:
        weights: Oarfish assignment probabilities (one per read).
        lengths: Raw poly(A) tail lengths (one per read, same order).
        use_log: If True, compute the weighted geometric mean via
            log-transform (log(L+1)) and back-transform (exp(result)-1).

    Returns:
        Weighted average poly(A) length, or 0.0 if the sum of
        weights is zero.
    """
    if not weights:
        return 0.0
    sum_wt = sum(weights)
    if sum_wt <= 0:
        return 0.0
    if use_log:
        log_lengths = [math.log(pl + 1.0) for pl in lengths]
        log_wm = sum(w * ll for w, ll in zip(weights, log_lengths)) / sum_wt
        return math.exp(log_wm) - 1.0
    return sum(w * pl for w, pl in zip(weights, lengths)) / sum_wt


def parse_polyA_file(filename: str) -> tuple[str, dict[str, dict[str, Any]]]:
    """Parse a poly(A) file and return ``(id_column_name, data_dict)``.

    Handles both transcript-level (``transcript_id`` column) and gene-level
    (``gene_id`` column) input formats.  Supports TSV (optionally gzipped,
    auto-detected by ``.gz`` suffix) and Parquet (auto-detected by
    ``.parquet`` or ``.pq`` suffix).

    Args:
        filename: Path to the input file (TSV, TSV.GZ, or Parquet).

    Returns:
        ``(id_col_name, data_dict)`` where *data_dict* maps feature IDs
        to dicts with keys ``n_reads``, ``total_wt``, ``wmlen``,
        ``weights``, ``lengths``, and optionally ``gene_id`` (when
        that column is present in the input).
    """
    print(f"Loading data from {filename}...", file=sys.stderr)

    if filename.endswith((".parquet", ".pq")):
        return _parse_polyA_parquet(filename)
    return _parse_polyA_tsv(filename)


def _parse_polyA_parquet(filename: str) -> tuple[str, dict[str, dict[str, Any]]]:
    """Read a poly(A) Parquet file."""
    table = pq.read_table(filename)
    col_names = table.schema.names
    has_gene_id = "gene_id" in col_names

    id_col_name = "transcript_id" if "transcript_id" in col_names else "gene_id"

    data_dict: dict[str, dict[str, Any]] = {}
    for row in table.to_pylist():
        feature_id = row[id_col_name]
        weights = np.array(row["weights"], dtype=np.float64)
        lengths = np.array(row["lengths"], dtype=np.int64)

        n_reads = len(weights)
        total_wt = float(np.sum(weights))
        wmlen = float(np.sum(weights * lengths) / total_wt) if total_wt > 0 else 0.0

        entry: dict[str, Any] = {
            "n_reads": n_reads,
            "total_wt": total_wt,
            "wmlen": wmlen,
            "weights": weights,
            "lengths": lengths,
        }
        if has_gene_id:
            entry["gene_id"] = row["gene_id"]

        data_dict[feature_id] = entry

    return id_col_name, data_dict


def _parse_polyA_tsv(filename: str) -> tuple[str, dict[str, dict[str, Any]]]:
    """Read a poly(A) TSV (or TSV.GZ) file."""
    data_dict: dict[str, dict[str, Any]] = {}

    with open_by_suffix(filename, "rt" if filename.endswith(".gz") else "r") as f:
        header = f.readline().strip().split("\t")

        # Detect whether transcript-level or gene-level output
        if "transcript_id" in header:
            id_col_name = "transcript_id"
        elif "gene_id" in header:
            id_col_name = "gene_id"
        else:
            print(
                "Error: Input file must contain 'transcript_id' or 'gene_id' column.",
                file=sys.stderr,
            )
            sys.exit(1)

        if "weights" not in header or "lengths" not in header:
            print(
                "Error: Input file must contain 'weights' and 'lengths' columns.",
                file=sys.stderr,
            )
            sys.exit(1)

        id_col = header.index(id_col_name)
        weights_col = header.index("weights")
        lengths_col = header.index("lengths")

        has_gene_id = "gene_id" in header
        gene_id_col = header.index("gene_id") if has_gene_id else -1

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

            entry: dict[str, Any] = {
                "n_reads": n_reads,
                "total_wt": total_wt,
                "wmlen": wmlen,
                "weights": weights,
                "lengths": lengths,
            }
            if has_gene_id and len(parts) > gene_id_col:
                gene_id = parts[gene_id_col]
                if gene_id not in ("", "NA", "."):
                    entry["gene_id"] = gene_id

            data_dict[feature_id] = entry

    return id_col_name, data_dict
