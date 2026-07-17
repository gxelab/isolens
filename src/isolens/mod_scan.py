#!/usr/bin/env python3
"""mod_scan: Generate HDF5 transcript-specific read x position modification matrices.

Part of the isolens toolkit. See notebooks/01_mod.md for the full specification.
"""

import argparse
import concurrent.futures
import functools
import os
import sys
import uuid
from typing import Any

import h5py
import numpy as np
import pysam

try:
    from isolens import __version__
    from isolens._parsing import parse_oarfish
except ImportError:
    from _parsing import parse_oarfish  # running as standalone script

    __version__ = "0.0.0"

# ---------- matrix encoding constants ----------
# Each position in the read × transcript matrix is encoded as a uint8:
#
#   CODE_UNCOVERED  = 0   no read coverage at this position
#   CODE_CANONICAL  = 1   canonical (unmodified) base, high confidence
#   CODE_MISMATCH   = 2   reference mismatch (CIGAR X operator)
#   CODE_DELETION   = 3   deletion relative to reference (CIGAR D)
#   4, 5, 6, ...          tracked modification types (assigned at runtime)
#   CODE_OTHERMOD   = 254 untracked modification won above threshold
#   CODE_FAIL       = 255 all states below probability threshold

CODE_UNCOVERED = 0
CODE_CANONICAL = 1
CODE_MISMATCH = 2
CODE_DELETION = 3
CODE_OTHERMOD = 254  # untracked modification won above threshold
CODE_FAIL = 255
# tracked modification types start at 4

# ---------- CIGAR operator constants (pysam cigartuples) ----------


@functools.cache
def _uuid_to_int(name: str) -> int:
    """Convert a read name to a 128-bit integer with LRU caching.

    Cached wrapper around ``uuid.UUID(name).int`` to avoid redundant
    parsing when the same read name appears multiple times in the BAM
    (e.g. primary + supplementary alignments).
    """
    return uuid.UUID(name).int


_BAM_CMATCH = 0  # M
_BAM_CINS = 1  # I
_BAM_CDEL = 2  # D
_BAM_CREF_SKIP = 3  # N
_BAM_CSOFT_CLIP = 4  # S
_BAM_CHARD_CLIP = 5  # H
_BAM_CPAD = 6  # P
_BAM_CEQUAL = 7  # =
_BAM_CDIFF = 8  # X


class _ReadRecord:
    """Lightweight, pickleable substitute for ``pysam.AlignedSegment``.

    Carries the subset of fields needed by ``parse_cigar_for_row`` and
    ``parse_modifications`` so that CPU-heavy parsing can be offloaded
    to worker threads without passing live pysam objects.
    """

    __slots__ = (
        "reference_start",
        "cigartuples",
        "query_sequence",
        "_mm_tag",
        "_ml_bytes",
    )

    def __init__(
        self,
        *,
        reference_start: int | None = None,
        cigartuples: list[tuple[int, int]] | None = None,
        query_sequence: str | None = None,
        mm_tag: str | None = None,
        ml_bytes: bytes | None = None,
    ):
        self.reference_start = reference_start
        self.cigartuples = cigartuples
        self.query_sequence = query_sequence
        self._mm_tag = mm_tag
        self._ml_bytes = ml_bytes

    def has_tag(self, tag: str) -> bool:
        if tag in ("MM", "mm"):
            return self._mm_tag is not None
        if tag in ("ML", "ml"):
            return self._ml_bytes is not None
        return False

    def get_tag(self, tag: str) -> str | bytes:
        if tag in ("MM", "mm"):
            return self._mm_tag
        if tag in ("ML", "ml"):
            return self._ml_bytes
        raise KeyError(tag)


def _extract_record(record: pysam.AlignedSegment) -> _ReadRecord:
    """Extract fields from a live pysam record into a ``_ReadRecord``.

    Must be called from the main thread (or any thread that owns the
    pysam iterator).  The returned ``_ReadRecord`` is safe to pass to
    worker threads.
    """
    mm_tag = None
    if record.has_tag("MM"):
        mm_tag = record.get_tag("MM")
    elif record.has_tag("mm"):
        mm_tag = record.get_tag("mm")

    ml_bytes = None
    if record.has_tag("ML"):
        ml_bytes = record.get_tag("ML")
    elif record.has_tag("ml"):
        ml_bytes = record.get_tag("ml")

    return _ReadRecord(
        reference_start=record.reference_start,
        cigartuples=record.cigartuples,
        query_sequence=record.query_sequence,
        mm_tag=mm_tag,
        ml_bytes=ml_bytes,
    )


# ---------- CIGAR parsing ----------


def parse_cigar_for_row(
    record: pysam.AlignedSegment | _ReadRecord,
    tx_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a uint8 matrix row and read-to-transcript position map from CIGAR.

    Uses NumPy slice assignments instead of per-position Python loops,
    giving ~10-50× speedup on long alignments.

    Args:
        record: ``pysam.AlignedSegment`` or ``_ReadRecord``.
        tx_length: int — length of the target transcript in bases.

    Returns:
        ``(row, read_to_tx_map)`` where:

        - *row* is a ``numpy.ndarray`` of shape ``(tx_length,)``, dtype
          uint8, filled with states (0=uncovered, 1=match, 2=mismatch,
          3=deletion).
        - *read_to_tx_map* is a ``numpy.ndarray`` of shape
          ``(len(query_sequence),)``, dtype int32.  Each entry is the
          1-based transcript position or -1 (sentinel for insertions /
          soft-clipped bases).
    """
    row = np.zeros(tx_length, dtype=np.uint8)

    seq = record.query_sequence
    ref_pos = record.reference_start  # 0-based

    # If we can't build a read-to-transcript map, just build the row from
    # CIGAR (if possible) and return an empty map.  Some reads have
    # query_sequence = None but still have valid CIGAR data.
    if seq is None or ref_pos is None:
        if ref_pos is not None:
            for op, length in record.cigartuples or []:
                if op in (_BAM_CEQUAL, _BAM_CDIFF, _BAM_CMATCH):
                    code = CODE_CANONICAL if op != _BAM_CDIFF else CODE_MISMATCH
                    valid_start = max(ref_pos, 0)
                    valid_end = min(ref_pos + length, tx_length)
                    if valid_start < valid_end:
                        row[valid_start:valid_end] = code
                    ref_pos += length
                elif op == _BAM_CDEL:
                    valid_start = max(ref_pos, 0)
                    valid_end = min(ref_pos + length, tx_length)
                    if valid_start < valid_end:
                        row[valid_start:valid_end] = CODE_DELETION
                    ref_pos += length
                elif op == _BAM_CREF_SKIP:
                    ref_pos += length
        return row, np.empty(0, dtype=np.int32)

    read_to_tx_map = np.full(len(seq), -1, dtype=np.int32)

    read_idx = 0  # current position in the read (0-based)

    for op, length in record.cigartuples or []:
        if op in (_BAM_CEQUAL, _BAM_CDIFF, _BAM_CMATCH):  # =, X, M
            code = CODE_CANONICAL if op != _BAM_CDIFF else CODE_MISMATCH

            # Positions before transcript start (ref_pos < 0)
            n_before = max(0, -min(ref_pos, 0))
            # Valid positions within [0, tx_length)
            valid_start = max(ref_pos, 0)
            valid_end = min(ref_pos + length, tx_length)
            n_valid = max(0, valid_end - valid_start)
            n_after = length - n_before - n_valid

            # n_before: positions already initialized to -1
            read_idx += n_before
            if n_valid > 0:
                row[valid_start:valid_end] = code
                read_to_tx_map[read_idx : read_idx + n_valid] = np.arange(
                    valid_start + 1, valid_end + 1, dtype=np.int32
                )
                read_idx += n_valid
            # n_after: positions already initialized to -1
            read_idx += n_after

            ref_pos += length

        elif op == _BAM_CDEL:  # D
            valid_start = max(ref_pos, 0)
            valid_end = min(ref_pos + length, tx_length)
            if valid_start < valid_end:
                row[valid_start:valid_end] = CODE_DELETION
            ref_pos += length
            # No entries in read_to_tx_map — deletions have no read base

        elif op == _BAM_CINS:  # I
            read_idx += length  # positions already -1

        elif op == _BAM_CSOFT_CLIP:  # S
            read_idx += length  # positions already -1

        elif op == _BAM_CREF_SKIP:  # N (intron / splice junction)
            ref_pos += length

        elif op in (_BAM_CHARD_CLIP, _BAM_CPAD):  # H, P
            pass  # consume nothing

    return row, read_to_tx_map


# ---------- modification parsing ----------


def _ensure_mod_code(mod_type: str, mod_code_map: dict[str, int]) -> int:
    """Assign a stable integer code for a modification type.

    Returns the integer code for *mod_type* (≥4).
    """
    if mod_type not in mod_code_map:
        mod_code_map[mod_type] = len(mod_code_map) + 4
    return mod_code_map[mod_type]


def parse_modifications(
    record: pysam.AlignedSegment | _ReadRecord,
    row: np.ndarray,
    read_to_tx_map: np.ndarray,
    filter_threshold: float,
    tracked_mod_types: frozenset,
    mod_code_map: dict[str, int],
    seen_mod_types: set,
) -> None:
    """Parse MM/ML tags and override *row* in-place with modification codes.

    Follows the modkit state classification pipeline:

    1. **CIGAR diff** — ``parse_cigar_for_row`` sets ``CODE_MISMATCH`` for
       reference mismatches (``X``) and ``CODE_DELETION`` for deletions
       (``D``).  These are preserved — ``parse_modifications`` never
       overwrites them.
    2. **Decode ML** — convert 8-bit integers to probabilities via
       ``P = (q + 0.5) / 256.0``.
    3. **Canonical remainder** — ``P_canonical = max(0, 1 - sum(P_mods))``.
    4. **Winning state** — pick the state (ALL modifications present at
       this position + canonical) with the highest probability.
    5. **Filter threshold** — if ``max_prob < filter_threshold``, mark as
       ``CODE_FAIL``.
    6. **Categorize** — canonical → ``CODE_CANONICAL``; tracked mod →
       its assigned integer code (≥4); untracked mod → ``CODE_OTHERMOD``.

    Uses flat parallel lists instead of per-position dicts to eliminate
    Python object allocation overhead in the hot path.

    Args:
        record: ``pysam.AlignedSegment`` or ``_ReadRecord``.
        row: ``numpy.ndarray`` of shape ``(tx_length,)``, dtype uint8.
            Modified in-place.
        read_to_tx_map: ``numpy.ndarray`` of shape ``(len(query_sequence),)``,
            dtype int32.  1-based transcript positions indexed by read
            position, with -1 sentinel for insertions / soft-clipped bases.
        filter_threshold: float — probability cutoff (e.g. ``0.95``).
        tracked_mod_types: ``frozenset[str]`` — modification types that
            participate in winner selection (from ``--mod-type``).
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

    # query_sequence recovers the sequence block even on
    # secondary / supplementary entries
    seq = record.query_sequence
    if seq is None:
        return

    # ---- Pass 1: aggregate modification instances in-place per position ----
    # Each entry is [sum_raw, max_prob, winning_code, winning_tracked].
    # Aggregating during collection avoids storing per-instance tuples and
    # eliminates the inner aggregation loop in Pass 2.

    per_position: dict[int, list] = {}
    total_mod_instance_idx = 0

    base_positions_cache: dict[str, list[int]] = {}

    for group_idx, mod_group in enumerate(mm_str.split(";")):
        if not mod_group:
            continue
        parts = mod_group.split(",")
        if not parts:
            continue

        meta = parts[0]
        if len(meta) < 3:
            continue
        target_base = meta[0]
        mod_type = meta[2:].split(".")[0]
        seen_mod_types.add(mod_type)

        try:
            skips = [int(s) for s in parts[1:]]
        except ValueError:
            continue

        # ---- Pre-compute target-base positions via C-level str.find ----
        # Cache per base character — multiple MM groups may target the
        # same base (e.g. A+a and A+17596 both target 'A').
        if target_base not in base_positions_cache:
            positions = []
            pos = seq.find(target_base)
            while pos != -1:
                positions.append(pos)
                pos = seq.find(target_base, pos + 1)
            base_positions_cache[target_base] = positions
        positions = base_positions_cache[target_base]

        # ---- Apply skip pattern over target-base positions ----
        occ_idx = 0  # index into *positions*
        for _skip_val in skips:
            occ_idx += _skip_val
            if occ_idx >= len(positions):
                break

            read_pos_0 = positions[occ_idx]

            # Step 2: Decode ML tag — convert 8-bit integer to probability
            #         P = (q + 0.5) / 256.0
            prob = 0.0
            raw_byte = 0
            if ml_bytes is not None and total_mod_instance_idx < len(ml_bytes):
                raw_byte = ml_bytes[total_mod_instance_idx]
                prob = (raw_byte + 0.5) / 256.0

            # Assign integer code and record whether this type is tracked.
            mod_type_code = _ensure_mod_code(mod_type, mod_code_map)
            is_tracked = mod_type in tracked_mod_types

            # Aggregate in-place: first instance at a position creates the
            # entry; subsequent instances update max_prob and sum_raw.
            entry = per_position.get(read_pos_0)
            if entry is None:
                per_position[read_pos_0] = [raw_byte, prob, mod_type_code, is_tracked]
            else:
                entry[0] += raw_byte
                if prob > entry[1]:
                    entry[1] = prob
                    entry[2] = mod_type_code
                    entry[3] = is_tracked

            total_mod_instance_idx += 1
            occ_idx += 1  # move past the marked occurrence

    if not per_position:
        return

    # ---- Pass 2: iterate position groups (no sort needed) ----
    # Pre-compute integer threshold for modkit-compatible comparison.
    # modkit converts probabilities to raw-byte space (floor(P * 256))
    # before comparing against the filter threshold.
    threshold_raw = int(filter_threshold * 256)

    row_len = len(row)
    tx_map_len = len(read_to_tx_map)

    for read_pos_0, entry in per_position.items():
        # Map read position to transcript position.
        if read_pos_0 >= tx_map_len:
            continue
        tx_pos_1 = read_to_tx_map[read_pos_0]
        if tx_pos_1 == -1:
            # -1 is the sentinel for insertions / soft-clipped bases
            continue
        tx_pos_0 = int(tx_pos_1) - 1  # convert to 0-based int
        if not (0 <= tx_pos_0 < row_len):
            continue

        # Preserve CIGAR reference mismatches (modkit's 'diff').
        if row[tx_pos_0] == CODE_MISMATCH:
            continue

        # ---- Pre-aggregated values from Pass 1 ----
        sum_raw = entry[0]
        max_prob = entry[1]
        winning_code = entry[2]  # -1 sentinel if we override to canonical
        winning_tracked = entry[3]

        # Step 3: Calculate canonical remainder (raw-byte space,
        #         matching modkit's canonical_qual = 255 - Σq_i).
        canonical_qual = max(0, 255 - sum_raw)
        p_canonical = (canonical_qual + 0.5) / 256.0

        # Step 4: canonical vs best mod
        if p_canonical > max_prob:
            max_prob = p_canonical
            winning_code = -1  # canonical
            winning_tracked = False

        # Step 5: Apply filter threshold (integer comparison matching modkit).
        if int(max_prob * 256) < threshold_raw:
            row[tx_pos_0] = CODE_FAIL
        elif winning_code == -1:
            # Step 6: Categorize — canonical.
            row[tx_pos_0] = CODE_CANONICAL
        elif winning_tracked:
            # Step 6: Categorize — modified (tracked mod won).
            row[tx_pos_0] = winning_code
        else:
            # Step 6: Categorize — othermod (untracked mod won
            #         above threshold, but isn't one we're tabulating).
            row[tx_pos_0] = CODE_OTHERMOD


# ---------- HDF5 output ----------


def write_transcript_group(
    h5: h5py.File,
    tx_name: str,
    rows: list[np.ndarray],
    read_ids: list[str],
    weights: list[float],
) -> None:
    """Write one transcript's matrix and metadata into an open HDF5 file.

    Args:
        h5: ``h5py.File`` open for writing.
        tx_name: str — transcript name (used as group name).
        rows: ``list[numpy.ndarray]`` — each of shape ``(tx_length,)``,
            dtype uint8.
        read_ids: ``list[str]`` — original read UUID strings.
        weights: ``list[float]`` — assignment probabilities.

    If *rows* is empty no group is created.
    """
    if not rows:
        return

    n_reads = len(rows)
    tx_length = rows[0].shape[0]

    grp = h5.create_group(f"transcripts/{tx_name}")

    # Stack rows into a contiguous 2D matrix
    # (np.stack raises ValueError if row shapes mismatch)
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


# ---------- parallel processing helpers ----------


def _process_transcript(
    tx_length: int,
    read_records: list[_ReadRecord],
    filter_threshold: float,
    tracked_mod_types: frozenset,
    mod_code_map: dict[str, int],
) -> tuple[list[np.ndarray], set]:
    """Process all reads for one transcript (worker-process entry point).

    Args:
        tx_length: int — length of the target transcript in bases.
        read_records: ``list[_ReadRecord]`` — one per read assigned to
            this transcript.
        filter_threshold: float — probability cutoff (e.g. ``0.95``).
        tracked_mod_types: ``frozenset[str]`` — modification types that
            participate in winner selection (from ``--mod-type``).
        mod_code_map: ``dict[str, int]`` — mapping from modification-type
            string to integer code (≥4).  All tracked types are pre-filled;
            ``_ensure_mod_code`` may add newly-encountered types at runtime.

    Returns:
        ``(rows, local_seen)`` where *rows* is a ``list[numpy.ndarray]``
        of uint8 matrix rows and *local_seen* is a ``set[str]`` of
        modification types observed by this worker.
    """
    rows = []
    local_seen: set[str] = set()
    for record in read_records:
        row, read_to_tx_map = parse_cigar_for_row(record, tx_length)
        parse_modifications(
            record,
            row,
            read_to_tx_map,
            filter_threshold,
            tracked_mod_types,
            mod_code_map,
            local_seen,
        )
        rows.append(row)
    return rows, local_seen


def _submit_batch(
    executor: concurrent.futures.ProcessPoolExecutor,
    pending: dict,
    records: list[_ReadRecord],
    read_ids: list[str],
    weights: list[float],
    tx_length: int,
    tx_name: str,
    filter_threshold: float,
    tracked_mod_types: frozenset,
    mod_code_map: dict[str, int],
) -> None:
    """Submit one transcript batch to the process pool.

    Stores the resulting ``Future`` in *pending* keyed to
    ``(tx_name, read_ids, weights)`` so the drain helpers can write
    the HDF5 group once processing completes.

    Args:
        executor: ``ProcessPoolExecutor``.
        pending: ``dict[Future, tuple]`` — mutated in-place.
        records: ``list[_ReadRecord]`` — read records for this batch.
        read_ids: ``list[str]`` — read UUID strings.
        weights: ``list[float]`` — assignment probabilities.
        tx_length: int — transcript length.
        tx_name: str — transcript name.
        filter_threshold: float — probability cutoff.
        tracked_mod_types: ``frozenset[str]`` — mod types to track.
        mod_code_map: ``dict[str, int]`` — mod type → code mapping.
    """
    future = executor.submit(
        _process_transcript,
        tx_length,
        records,
        filter_threshold,
        tracked_mod_types,
        mod_code_map,
    )
    pending[future] = (tx_name, read_ids, weights)


def _drain_one(
    pending: dict,
    h5: h5py.File,
    global_seen: set,
    verbose: bool,
) -> tuple[int, int]:
    """Wait for at least one pending future and write its result to HDF5.

    Returns ``(n_tx_flushed, n_assign_flushed)``.
    """
    done, _ = concurrent.futures.wait(
        pending,
        return_when=concurrent.futures.FIRST_COMPLETED,
    )
    n_tx = 0
    n_assign = 0
    for future in done:
        tx_name, read_ids, weights = pending.pop(future)
        rows, local_seen = future.result()
        global_seen |= local_seen
        if rows:
            write_transcript_group(h5, tx_name, rows, read_ids, weights)
            n_tx += 1
            n_assign += len(rows)
    return n_tx, n_assign


def _drain_all(
    pending: dict,
    h5: h5py.File,
    global_seen: set,
    verbose: bool,
) -> tuple[int, int]:
    """Wait for all pending futures and write their results to HDF5.

    Returns ``(n_tx_flushed, n_assign_flushed)``.
    """
    n_tx = 0
    n_assign = 0
    while pending:
        dt, da = _drain_one(pending, h5, global_seen, verbose)
        n_tx += dt
        n_assign += da
    return n_tx, n_assign


# ---------- CLI ----------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for mod_scan."""
    parser = argparse.ArgumentParser(
        description="mod_scan: Generate HDF5 read x position modification matrices"
    )
    parser.add_argument(
        "-b",
        "--bam",
        required=True,
        help="Path to transcriptome BAM alignment file",
    )
    parser.add_argument(
        "-a",
        "--oarfish",
        required=True,
        help="Path to Oarfish isoform assignment probability file (.lz4 or plain text)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output HDF5 file path",
    )
    parser.add_argument(
        "-c",
        "--mod-cutoff",
        type=float,
        default=0.95,
        help="Modification probability cutoff [default: 0.95]",
    )
    parser.add_argument(
        "-p",
        "--min-asp",
        type=float,
        default=0.0,
        help="Minimum Oarfish assignment probability for a read to be "
        "included [default: 0.0 (no filter)]",
    )
    parser.add_argument(
        "-d",
        "--max-depth",
        type=int,
        default=5000,
        help="Maximum number of reads per transcript. When the number of "
        "reads mapped to a transcript exceeds this limit, only the "
        "first N reads are retained [default: 5000]",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=min(2, os.cpu_count() or 1),
        help="Number of worker processes for parallel transcript processing "
        "[default: 2]",
    )
    parser.add_argument(
        "-m",
        "--mod-type",
        nargs="*",
        default=["a", "m", "17596", "17802", "19228", "69426", "19229", "19227"],
        help="Modification types to scan for (SAM code suffixes). "
        "Defaults to the standard RNA modification table: "
        "m6A (a), m5C (m), inosine (17596), pseU (17802), "
        "2OmeC (19228), 2OmeA (69426), 2OmeG (19229), "
        "2OmeU (19227)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print progress to stderr",
    )
    return parser.parse_args()


# ---------- shared helpers ----------


def _write_global_metadata(
    h5: h5py.File,
    mod_cutoff: float,
    min_asp: float,
    mod_code_map: dict[str, int],
    n_transcripts: int,
    n_assignments: int,
) -> None:
    """Write the ``/modification_codes`` and ``/metadata`` groups to HDF5.

    Args:
        h5: ``h5py.File`` open for writing.
        mod_cutoff: float — modification probability cutoff.
        min_asp: float — minimum assignment probability filter.
        mod_code_map: ``dict[str, int]`` — modification type → code.
        n_transcripts: int — number of transcripts written.
        n_assignments: int — number of read assignments written.
    """
    # Global /modification_codes
    codes_grp = h5.create_group("modification_codes")
    for mod_type, code in sorted(mod_code_map.items(), key=lambda x: x[1]):
        codes_grp.attrs[mod_type] = code

    # Global /metadata
    meta = h5.create_group("metadata")
    meta.attrs["mod_cutoff"] = mod_cutoff
    meta.attrs["min_asp"] = min_asp
    meta.attrs["pipeline_version"] = __version__
    meta.attrs["n_transcripts"] = n_transcripts
    meta.attrs["n_assignments"] = n_assignments
    meta.attrs["modification_codes"] = str(
        dict(sorted(mod_code_map.items(), key=lambda x: x[1]))
    )


def _resolve_read_assignment(
    record: pysam.AlignedSegment,
    prob_map: dict[int, list],
    bam_ref_to_tx_id: list[int | None],
) -> tuple[int, Any] | None:
    """Resolve a BAM record to its Oarfish transcript assignment.

    Looks up the read by UUID in the Oarfish probability map, then
    matches the BAM reference index to a transcript ID and finds the
    exact assignment for that transcript.

    Args:
        record: ``pysam.AlignedSegment`` from the BAM iterator.
        prob_map: Oarfish ``read_id_int → list[TargetAssignment]`` map.
        bam_ref_to_tx_id: ``list[int|None]`` — BAM reference index →
            Oarfish transcript ID.

    Returns:
        ``(tx_id, assignment)`` if a matching assignment is found, or
        ``None`` if any lookup fails.
    """
    # Look up read in Oarfish by UUID
    try:
        read_id_int = _uuid_to_int(record.query_name)
    except ValueError:
        return None

    assignments = prob_map.get(read_id_int)
    if not assignments:
        return None

    tx_index = record.reference_id
    if tx_index is None or tx_index < 0:
        return None

    tx_id = bam_ref_to_tx_id[tx_index]
    if tx_id is None:
        return None

    # Find the exact assignment for this transcript
    # (linear scan — assignment lists are small, 1–10 elements)
    assignment = next((a for a in assignments if a.tx_id == tx_id), None)
    if not assignment:
        return None

    return tx_id, assignment


# ---------- main pipeline ----------


# ---------- sequential path ----------


def _run_sequential(
    bam: pysam.AlignmentFile,
    args: argparse.Namespace,
    h5: h5py.File,
    tx_names: list[str],
    prob_map: dict[int, list],
    bam_ref_to_tx_id: list[int | None],
    filter_threshold: float,
    mod_code_map: dict[str, int],
) -> tuple[int, int, int, int, dict[str, int], set]:
    """Stream through the BAM transcript-by-transcript (single-threaded).

    *mod_code_map* is pre-filled with user-requested modification types
    and is mutated in-place as new types are discovered.

    Returns ``(n_tx, n_assign, total_records, matched_reads,
    mod_code_map, seen_mod_types)``.
    """
    seen_mod_types: set[str] = set()
    # Snapshot tracked modification types before any runtime mutations.
    # Only these types participate in winner selection (modkit Step 4).
    tracked_mod_types = frozenset(mod_code_map.keys())

    current_tx_id: int | None = None
    current_read_ids: list[str] = []
    current_rows: list[np.ndarray] = []
    current_weights: list[float] = []
    n_transcripts_written = 0
    n_assignments_written = 0

    total_records = 0
    matched_reads = 0

    for record in bam:
        total_records += 1
        if args.verbose and total_records % 500_000 == 0:
            print(
                f"[mod_scan] Scanned {total_records} alignments...",
                file=sys.stderr,
            )

        if record.is_unmapped:
            continue

        resolved = _resolve_read_assignment(record, prob_map, bam_ref_to_tx_id)
        if resolved is None:
            continue
        tx_id, assignment = resolved

        # ---- Transcript change: flush previous transcript ----
        if tx_id != current_tx_id:
            if current_tx_id is not None and current_rows:
                write_transcript_group(
                    h5,
                    tx_names[current_tx_id],
                    current_rows,
                    current_read_ids,
                    current_weights,
                )
                n_transcripts_written += 1
                n_assignments_written += len(current_rows)
                if args.verbose and n_transcripts_written % 1000 == 0:
                    print(
                        f"[mod_scan] Wrote {n_transcripts_written} "
                        "transcript groups...",
                        file=sys.stderr,
                    )
                current_read_ids = []
                current_rows = []
                current_weights = []
            current_tx_id = tx_id

        # Depth limit: skip if this transcript already has enough reads
        if args.max_depth is not None and len(current_rows) >= args.max_depth:
            continue

        # Assignment probability filter
        if args.min_asp > 0.0 and assignment.prob < args.min_asp:
            continue

        matched_reads += 1

        # ---- Build matrix row ----
        tx_length = bam.lengths[record.reference_id]
        # Extract to _ReadRecord to avoid redundant pysam C calls in
        # parse_modifications (which otherwise re-extracts MM/ML tags).
        extracted = _extract_record(record)
        row, read_to_tx_map = parse_cigar_for_row(extracted, tx_length)
        parse_modifications(
            extracted,
            row,
            read_to_tx_map,
            filter_threshold,
            tracked_mod_types,
            mod_code_map,
            seen_mod_types,
        )

        current_read_ids.append(record.query_name)
        current_rows.append(row)
        current_weights.append(assignment.prob)

    bam.close()

    # ---- Flush the last transcript ----
    if current_tx_id is not None and current_rows:
        write_transcript_group(
            h5,
            tx_names[current_tx_id],
            current_rows,
            current_read_ids,
            current_weights,
        )
        n_transcripts_written += 1
        n_assignments_written += len(current_rows)

    # ---- Global metadata ----
    _write_global_metadata(
        h5,
        args.mod_cutoff,
        args.min_asp,
        mod_code_map,
        n_transcripts_written,
        n_assignments_written,
    )

    return (
        n_transcripts_written,
        n_assignments_written,
        total_records,
        matched_reads,
        mod_code_map,
        seen_mod_types,
    )


# ---------- parallel path ----------


def _run_parallel(
    bam: pysam.AlignmentFile,
    args: argparse.Namespace,
    h5: h5py.File,
    tx_names: list[str],
    prob_map: dict[int, list],
    bam_ref_to_tx_id: list[int | None],
    tx_lengths: dict[int, int],
    filter_threshold: float,
    mod_code_map: dict[str, int],
) -> tuple[int, int, int, int, dict[str, int], set]:
    """Stream through the BAM and process transcripts in parallel.

    The main thread scans the BAM and submits transcript batches to a
    ``ProcessPoolExecutor``.  Workers do CIGAR + modification parsing
    with a **read-only** *mod_code_map* (pre-filled from the CLI
    ``--mod-type`` argument).  The main thread writes completed batches
    to HDF5.

    Returns ``(n_tx, n_assign, total_records, matched_reads,
    mod_code_map, seen_mod_types)``.
    """
    seen_mod_types: set[str] = set()
    # Snapshot tracked modification types before any runtime mutations.
    # Only these types participate in winner selection (modkit Step 4).
    tracked_mod_types = frozenset(mod_code_map.keys())

    max_pending = max(2, args.threads * 2)

    current_tx_id: int | None = None
    current_tx_length = 0
    current_records: list[_ReadRecord] = []
    current_read_ids: list[str] = []
    current_weights: list[float] = []

    n_transcripts_written = 0
    n_assignments_written = 0
    total_records = 0
    matched_reads = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.threads) as executor:
        pending: dict[concurrent.futures.Future, tuple] = {}

        for record in bam:
            total_records += 1
            if args.verbose and total_records % 500_000 == 0:
                print(
                    f"[mod_scan] Scanned {total_records} alignments...",
                    file=sys.stderr,
                )

            if record.is_unmapped:
                continue

            resolved = _resolve_read_assignment(record, prob_map, bam_ref_to_tx_id)
            if resolved is None:
                continue
            tx_id, assignment = resolved

            # ---- Transcript change ----
            if tx_id != current_tx_id:
                if current_tx_id is not None and current_records:
                    _submit_batch(
                        executor,
                        pending,
                        current_records,
                        current_read_ids,
                        current_weights,
                        current_tx_length,
                        tx_names[current_tx_id],
                        filter_threshold,
                        tracked_mod_types,
                        mod_code_map,
                    )
                    # Back-pressure: drain one if too many in-flight
                    if len(pending) >= max_pending:
                        dt, da = _drain_one(
                            pending,
                            h5,
                            seen_mod_types,
                            args.verbose,
                        )
                        n_transcripts_written += dt
                        n_assignments_written += da
                    current_records = []
                    current_read_ids = []
                    current_weights = []
                current_tx_id = tx_id
                current_tx_length = tx_lengths.get(
                    tx_id, bam.lengths[record.reference_id]
                )

            # Depth limit
            if args.max_depth is not None and len(current_records) >= args.max_depth:
                continue

            # Assignment probability filter
            if args.min_asp > 0.0 and assignment.prob < args.min_asp:
                continue

            matched_reads += 1

            # Extract read data for worker process
            read_record = _extract_record(record)
            current_records.append(read_record)
            current_read_ids.append(record.query_name)
            current_weights.append(assignment.prob)

        bam.close()

        # ---- Submit last transcript ----
        if current_tx_id is not None and current_records:
            _submit_batch(
                executor,
                pending,
                current_records,
                current_read_ids,
                current_weights,
                current_tx_length,
                tx_names[current_tx_id],
                filter_threshold,
                tracked_mod_types,
                mod_code_map,
            )

        # ---- Drain remaining ----
        dt, da = _drain_all(pending, h5, seen_mod_types, args.verbose)
        n_transcripts_written += dt
        n_assignments_written += da

    # ---- Global metadata ----
    _write_global_metadata(
        h5,
        args.mod_cutoff,
        args.min_asp,
        mod_code_map,
        n_transcripts_written,
        n_assignments_written,
    )

    return (
        n_transcripts_written,
        n_assignments_written,
        total_records,
        matched_reads,
        mod_code_map,
        seen_mod_types,
    )


# ---------- main dispatcher ----------


def main(args: argparse.Namespace | None = None) -> None:
    """Scan a transcriptome BAM for base modifications and write HDF5 output.

    Steps:
    1. Load Oarfish read-to-transcript assignment probabilities.
    2. Open the coordinate-sorted BAM and map reference indices.
    3. Build the modification code map from ``--mod-type`` arguments.
    4. Stream through the BAM (sequential or parallel), building per-
       transcript read × position matrices.
    5. Write global metadata (modification codes, pipeline version, counts).
    """
    if args is None:
        args = parse_args()
    filter_threshold = args.mod_cutoff

    # ---- 1. Load Oarfish assignments ----

    if args.verbose:
        print("[mod_scan] Loading Oarfish assignments into memory...", file=sys.stderr)

    tx_names, prob_map, name_to_id = parse_oarfish(args.oarfish)

    if args.verbose:
        print(
            f"[mod_scan] Loaded {len(tx_names)} transcripts, "
            f"{len(prob_map)} reads with assignments",
            file=sys.stderr,
        )

    # ---- 2. Open BAM and map references ----

    bam = pysam.AlignmentFile(args.bam, "rb")

    # Check sort order — streaming relies on coordinate-sorted BAM
    sort_order = bam.header.get("HD", {}).get("SO")
    if sort_order != "coordinate" and args.verbose:
        print(
            "[mod_scan] Warning: BAM is not coordinate-sorted; "
            "transcript-level streaming may produce incorrect results",
            file=sys.stderr,
        )

    # Build lookups from BAM reference index to Oarfish transcript ID
    bam_ref_to_tx_id: list[int | None] = [name_to_id.get(ref) for ref in bam.references]
    # Also build a tx_id → length lookup for the parallel path
    tx_lengths: dict[int, int] = {}
    for i, ref in enumerate(bam.references):
        tx_id = name_to_id.get(ref)
        if tx_id is not None:
            tx_lengths[tx_id] = bam.lengths[i]

    # ---- 3. Build initial modification code map ----

    # Pre-fill from user-provided (or default) modification types so that
    # codes are deterministic.  ``parse_modifications`` will add any
    # additional types it encounters at runtime.
    mod_code_map: dict[str, int] = {}
    for i, mod_type in enumerate(sorted(args.mod_type)):
        mod_code_map[mod_type] = i + 4  # 4, 5, 6, ...

    if args.verbose:
        print(
            f"[mod_scan] Modification types to scan: {sorted(mod_code_map.keys())}",
            file=sys.stderr,
        )

    # ---- 4. Process ----

    if args.verbose:
        print("[mod_scan] Writing HDF5 output...", file=sys.stderr)

    with h5py.File(args.output, "w") as h5:
        if args.threads <= 1:
            (n_tx, n_assign, total, matched, mod_code_map, seen_mod_types) = (
                _run_sequential(
                    bam,
                    args,
                    h5,
                    tx_names,
                    prob_map,
                    bam_ref_to_tx_id,
                    filter_threshold,
                    mod_code_map,
                )
            )
        else:
            (n_tx, n_assign, total, matched, mod_code_map, seen_mod_types) = (
                _run_parallel(
                    bam,
                    args,
                    h5,
                    tx_names,
                    prob_map,
                    bam_ref_to_tx_id,
                    tx_lengths,
                    filter_threshold,
                    mod_code_map,
                )
            )

    # ---- 5. Summary ----

    if args.verbose:
        print(
            f"[mod_scan] Total alignments scanned: {total}",
            file=sys.stderr,
        )
        print(
            f"[mod_scan] Reads matched to Oarfish assignments: {matched}",
            file=sys.stderr,
        )
        print(
            f"[mod_scan] Transcripts written: {n_tx}",
            file=sys.stderr,
        )
        print(
            f"[mod_scan] Modification types found: {sorted(seen_mod_types)}",
            file=sys.stderr,
        )
        print(
            f"[mod_scan] Modification code map: {mod_code_map}",
            file=sys.stderr,
        )
        print(
            f"[mod_scan] Done. Output written to {args.output}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
