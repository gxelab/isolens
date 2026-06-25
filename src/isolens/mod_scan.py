#!/usr/bin/env python3
"""mod_scan: Generate HDF5 transcript-specific read x position modification matrices.

Part of the isolens toolkit. See notebooks/01_mod.md for the full specification.
"""

import argparse
import concurrent.futures
import sys
import uuid

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
CODE_OTHERMOD = 254  # untracked modification won above threshold
CODE_FAIL = 255
# tracked modification types start at 4

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
        reference_start,
        cigartuples,
        query_sequence,
        mm_tag=None,
        ml_bytes=None,
    ):
        self.reference_start = reference_start
        self.cigartuples = cigartuples
        self.query_sequence = query_sequence
        self._mm_tag = mm_tag
        self._ml_bytes = ml_bytes

    def has_tag(self, tag):
        if tag in ("MM", "mm"):
            return self._mm_tag is not None
        if tag in ("ML", "ml"):
            return self._ml_bytes is not None
        return False

    def get_tag(self, tag):
        if tag in ("MM", "mm"):
            return self._mm_tag
        if tag in ("ML", "ml"):
            return self._ml_bytes
        raise KeyError(tag)


def _extract_record(record):
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


def parse_cigar_for_row(record, tx_length):
    """Build a uint8 matrix row and read-to-transcript position map from CIGAR.

    Uses NumPy slice assignments and ``list.extend`` instead of per-position
    Python loops, giving ~10-50× speedup on long alignments.

    Args:
        record: ``pysam.AlignedSegment`` or ``_ReadRecord``.
        tx_length: int — length of the target transcript in bases.

    Returns:
        (row, read_to_tx_map) where:
        - *row* is a ``numpy.ndarray`` of shape ``(tx_length,)``, dtype uint8,
          filled with states (0=uncovered, 1=match, 2=mismatch, 3=deletion).
        - *read_to_tx_map* is a ``list[int | None]`` of the same length as
          ``record.query_sequence``.  Each entry is the 1-based
          transcript position or ``None`` (for insertions / soft-clipped bases).
    """
    row = np.zeros(tx_length, dtype=np.uint8)
    read_to_tx_map = []

    ref_pos = record.reference_start  # 0-based

    # Validate reference_start
    if ref_pos is None:
        return row, read_to_tx_map

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

            read_to_tx_map.extend([None] * n_before)
            if n_valid > 0:
                row[valid_start:valid_end] = code
                read_to_tx_map.extend(range(valid_start + 1, valid_end + 1))
            read_to_tx_map.extend([None] * n_after)

            ref_pos += length

        elif op == _BAM_CDEL:  # D
            valid_start = max(ref_pos, 0)
            valid_end = min(ref_pos + length, tx_length)
            if valid_start < valid_end:
                row[valid_start:valid_end] = CODE_DELETION
            ref_pos += length
            # No entries in read_to_tx_map — deletions have no read base

        elif op == _BAM_CINS:  # I
            read_to_tx_map.extend([None] * length)

        elif op == _BAM_CSOFT_CLIP:  # S
            read_to_tx_map.extend([None] * length)

        elif op == _BAM_CREF_SKIP:  # N (intron / splice junction)
            ref_pos += length

        elif op in (_BAM_CHARD_CLIP, _BAM_CPAD):  # H, P
            pass  # consume nothing

    return row, read_to_tx_map


# ---------- modification parsing ----------


def _ensure_mod_code(mod_type, mod_code_map):
    """Assign a stable integer code for a modification type.

    Returns the integer code for *mod_type* (≥4).
    """
    if mod_type not in mod_code_map:
        mod_code_map[mod_type] = len(mod_code_map) + 4
    return mod_code_map[mod_type]


def parse_modifications(
    record,
    row,
    read_to_tx_map,
    filter_threshold,
    tracked_mod_types,
    mod_code_map,
    seen_mod_types,
):
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

    Uses a two-pass implementation: Pass 1 collects per-position data from
    the MM/ML tags (Step 2), Pass 2 applies Steps 3–6.

    Args:
        record: ``pysam.AlignedSegment`` or ``_ReadRecord``.
        row: ``numpy.ndarray`` of shape ``(tx_length,)``, dtype uint8.
            Modified in-place.
        read_to_tx_map: ``list[int | None]`` — 1-based transcript positions
            indexed by read position (same length as ``query_sequence``).
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

    # ---- Pass 1: collect per-position modification probabilities ----
    # per_position: read_pos_0 → {mod_type: (prob_float, raw_ml_byte)}
    per_position: dict[int, dict[str, tuple[float, int]]] = {}
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

            per_position.setdefault(read_pos_0, {})[mod_type] = (
                prob,
                raw_byte,
            )

            total_mod_instance_idx += 1
            occ_idx += 1  # move past the marked occurrence

    # Pre-compute integer threshold for modkit-compatible comparison.
    # modkit converts probabilities to raw-byte space (floor(P * 256))
    # before comparing against the filter threshold.  This is equivalent
    # to a float comparison for individual modifications (due to the
    # +0.5 alignment) but is slightly less strict for P_canonical at
    # multi-mod positions, matching modkit's behaviour exactly.
    threshold_raw = int(filter_threshold * 256)

    # ---- Pass 2: 6-step modkit classification pipeline ----
    for read_pos_0, mod_data in per_position.items():
        # Map read position to transcript position (used by every step
        # that assigns a code).  Skip positions that don't map.
        if read_pos_0 >= len(read_to_tx_map):
            continue
        tx_pos_1 = read_to_tx_map[read_pos_0]
        if tx_pos_1 is None:
            continue
        tx_pos_0 = tx_pos_1 - 1  # convert to 0-based
        if not (0 <= tx_pos_0 < len(row)):
            continue

        # Preserve CIGAR reference mismatches (modkit's 'diff').
        # If the read base differs from the reference at this position,
        # ``parse_cigar_for_row`` already flagged it as CODE_MISMATCH.
        # modkit locks these positions and refuses to tabulate them as
        # valid canonical, failed, or modified — do not overwrite.
        if row[tx_pos_0] == CODE_MISMATCH:
            continue

        # Step 3: Calculate canonical remainder (raw-byte space,
        #         matching modkit's canonical_qual = 255 - Σq_i).
        #         canonical_qual = max(0, 255 - sum of all raw ML bytes)
        #         P_canonical = (canonical_qual + 0.5) / 256.0
        sum_raw = sum(raw_byte for _, raw_byte in mod_data.values())
        canonical_qual = max(0, 255 - sum_raw)
        p_canonical = (canonical_qual + 0.5) / 256.0

        # Step 4: Find the highest-confidence state among ALL
        #         modifications at this position + canonical.
        #         (modkit considers every mod in the MM tag,
        #         regardless of --mod-type.)
        max_prob = p_canonical
        winning_state = "canonical"
        for mod_type, (prob, _raw_byte) in mod_data.items():
            if prob > max_prob:
                max_prob = prob
                winning_state = mod_type

        # Step 5: Apply filter threshold (integer comparison matching modkit).
        #         Convert max_prob to raw-byte space: floor(P * 256).
        #         If max_prob_raw < threshold_raw → FAIL.
        if int(max_prob * 256) < threshold_raw:
            row[tx_pos_0] = CODE_FAIL
        elif winning_state == "canonical":
            # Step 6: Categorize — canonical.
            row[tx_pos_0] = CODE_CANONICAL
        elif winning_state in tracked_mod_types:
            # Step 6: Categorize — modified (tracked mod won).
            row[tx_pos_0] = _ensure_mod_code(winning_state, mod_code_map)
        else:
            # Step 6: Categorize — othermod (untracked mod won
            #         above threshold, but isn't one we're tabulating).
            row[tx_pos_0] = CODE_OTHERMOD


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


def flush_transcript(h5, tx_name, read_data):
    """Unpack accumulated read data and write one transcript group to HDF5.

    Args:
        h5: ``h5py.File`` open for writing.
        tx_name: str — transcript name (used as group name).
        read_data: ``list[tuple[str, numpy.ndarray, float]]`` —
            each tuple is ``(read_id_str, row_uint8, weight_float)``.
    """
    read_id_strs = [d[0] for d in read_data]
    rows = [d[1] for d in read_data]
    weights = [d[2] for d in read_data]
    write_transcript_group(h5, tx_name, rows, read_id_strs, weights)


# ---------- parallel processing helpers ----------


def _process_transcript(
    tx_length, read_records, filter_threshold, tracked_mod_types, mod_code_map
):
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
    local_seen = set()
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
    executor,
    pending,
    batch,
    tx_length,
    tx_name,
    filter_threshold,
    tracked_mod_types,
    mod_code_map,
):
    """Submit one transcript batch to the process pool.

    Stores the resulting ``Future`` in *pending* keyed to
    ``(tx_name, read_ids, weights)`` so the drain helpers can write
    the HDF5 group once processing completes.
    """
    records = [item[0] for item in batch]
    read_ids = [item[1] for item in batch]
    weights = [item[2] for item in batch]
    future = executor.submit(
        _process_transcript,
        tx_length,
        records,
        filter_threshold,
        tracked_mod_types,
        mod_code_map,
    )
    pending[future] = (tx_name, read_ids, weights)


def _drain_one(pending, h5, global_seen, verbose):
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


def _drain_all(pending, h5, global_seen, verbose):
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


def parse_args():
    """Parse command-line arguments."""
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
        help="Path to Oarfish isoform assignment probability file (.lz4)",
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
        default=1,
        help="Number of worker threads for parallel transcript processing "
        "[default: 1 (sequential)]",
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


# ---------- main pipeline ----------


# ---------- sequential path ----------


def _run_sequential(
    bam, args, h5, tx_names, prob_map, bam_ref_to_tx_id, filter_threshold, mod_code_map
):
    """Stream through the BAM transcript-by-transcript (single-threaded).

    *mod_code_map* is pre-filled with user-requested modification types
    and is mutated in-place as new types are discovered.

    Returns ``(n_tx, n_assign, total_records, matched_reads,
    mod_code_map, seen_mod_types)``.
    """
    seen_mod_types = set()
    # Snapshot tracked modification types before any runtime mutations.
    # Only these types participate in winner selection (modkit Step 4).
    tracked_mod_types = frozenset(mod_code_map.keys())

    current_tx_id = None
    current_reads = []  # [(read_id_str, row_uint8, weight_float), ...]
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

        # ---- Transcript change: flush previous transcript ----
        if tx_id != current_tx_id:
            if current_tx_id is not None and current_reads:
                flush_transcript(
                    h5,
                    tx_names[current_tx_id],
                    current_reads,
                )
                n_transcripts_written += 1
                n_assignments_written += len(current_reads)
                if args.verbose and n_transcripts_written % 1000 == 0:
                    print(
                        f"[mod_scan] Wrote {n_transcripts_written} "
                        "transcript groups...",
                        file=sys.stderr,
                    )
                current_reads = []
            current_tx_id = tx_id

        # Find the exact assignment for this transcript
        # (linear scan — assignment lists are small, 1–10 elements)
        assignment = next((a for a in assignments if a.tx_id == tx_id), None)
        if not assignment:
            continue

        # Depth limit: skip if this transcript already has enough reads
        if args.max_depth is not None and len(current_reads) >= args.max_depth:
            continue

        # Assignment probability filter
        if args.min_asp > 0.0 and assignment.prob < args.min_asp:
            continue

        matched_reads += 1

        # ---- Build matrix row ----
        tx_length = bam.lengths[tx_index]
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

        current_reads.append((record.query_name, row, assignment.prob))

    bam.close()

    # ---- Flush the last transcript ----
    if current_tx_id is not None and current_reads:
        flush_transcript(h5, tx_names[current_tx_id], current_reads)
        n_transcripts_written += 1
        n_assignments_written += len(current_reads)

    # ---- Global /modification_codes ----
    codes_grp = h5.create_group("modification_codes")
    for mod_type, code in sorted(mod_code_map.items(), key=lambda x: x[1]):
        codes_grp.attrs[mod_type] = code

    # ---- Global /metadata ----
    meta = h5.create_group("metadata")
    meta.attrs["mod_cutoff"] = args.mod_cutoff
    meta.attrs["min_asp"] = args.min_asp
    meta.attrs["pipeline_version"] = "0.1.0"
    meta.attrs["n_transcripts"] = n_transcripts_written
    meta.attrs["n_assignments"] = n_assignments_written
    meta.attrs["modification_codes"] = str(
        dict(sorted(mod_code_map.items(), key=lambda x: x[1]))
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
    bam,
    args,
    h5,
    tx_names,
    prob_map,
    bam_ref_to_tx_id,
    tx_lengths,
    filter_threshold,
    mod_code_map,
):
    """Stream through the BAM and process transcripts in parallel.

    The main thread scans the BAM and submits transcript batches to a
    ``ProcessPoolExecutor``.  Workers do CIGAR + modification parsing
    with a **read-only** *mod_code_map* (pre-filled from the CLI
    ``--mod-type`` argument).  The main thread writes completed batches
    to HDF5.

    Returns ``(n_tx, n_assign, total_records, matched_reads,
    mod_code_map, seen_mod_types)``.
    """
    seen_mod_types = set()
    # Snapshot tracked modification types before any runtime mutations.
    # Only these types participate in winner selection (modkit Step 4).
    tracked_mod_types = frozenset(mod_code_map.keys())

    max_pending = max(2, args.threads * 2)

    current_tx_id = None
    current_tx_length = 0
    current_batch = []  # [(_ReadRecord, read_name_str, prob_float), ...]

    n_transcripts_written = 0
    n_assignments_written = 0
    total_records = 0
    matched_reads = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.threads) as executor:
        pending = {}  # Future → (tx_name, read_ids, weights)

        for record in bam:
            total_records += 1
            if args.verbose and total_records % 500_000 == 0:
                print(
                    f"[mod_scan] Scanned {total_records} alignments...",
                    file=sys.stderr,
                )

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

            # ---- Transcript change ----
            if tx_id != current_tx_id:
                if current_tx_id is not None and current_batch:
                    _submit_batch(
                        executor,
                        pending,
                        current_batch,
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
                    current_batch = []
                current_tx_id = tx_id
                current_tx_length = tx_lengths.get(tx_id, bam.lengths[tx_index])

            # Find the exact assignment for this transcript
            # (linear scan — assignment lists are small, 1–10 elements)
            assignment = next((a for a in assignments if a.tx_id == tx_id), None)
            if not assignment:
                continue

            # Depth limit
            if args.max_depth is not None and len(current_batch) >= args.max_depth:
                continue

            # Assignment probability filter
            if args.min_asp > 0.0 and assignment.prob < args.min_asp:
                continue

            matched_reads += 1

            # Extract read data for worker process
            read_record = _extract_record(record)
            current_batch.append((read_record, record.query_name, assignment.prob))

        bam.close()

        # ---- Submit last transcript ----
        if current_tx_id is not None and current_batch:
            _submit_batch(
                executor,
                pending,
                current_batch,
                current_tx_length,
                tx_names[current_tx_id],
                mod_cutoff_u8,
                mod_code_map,
            )

        # ---- Drain remaining ----
        dt, da = _drain_all(pending, h5, seen_mod_types, args.verbose)
        n_transcripts_written += dt
        n_assignments_written += da

    # ---- Global /modification_codes ----
    codes_grp = h5.create_group("modification_codes")
    for mod_type, code in sorted(mod_code_map.items(), key=lambda x: x[1]):
        codes_grp.attrs[mod_type] = code

    # ---- Global /metadata ----
    meta = h5.create_group("metadata")
    meta.attrs["mod_cutoff"] = args.mod_cutoff
    meta.attrs["min_asp"] = args.min_asp
    meta.attrs["pipeline_version"] = "0.1.0"
    meta.attrs["n_transcripts"] = n_transcripts_written
    meta.attrs["n_assignments"] = n_assignments_written
    meta.attrs["modification_codes"] = str(
        dict(sorted(mod_code_map.items(), key=lambda x: x[1]))
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


def main():
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
    bam_ref_to_tx_id = [name_to_id.get(ref, None) for ref in bam.references]
    # Also build a tx_id → length lookup for the parallel path
    tx_lengths = {}
    for i, ref in enumerate(bam.references):
        tx_id = name_to_id.get(ref)
        if tx_id is not None:
            tx_lengths[tx_id] = bam.lengths[i]

    # ---- 3. Build initial modification code map ----

    # Pre-fill from user-provided (or default) modification types so that
    # codes are deterministic.  ``parse_modifications`` will add any
    # additional types it encounters at runtime.
    mod_code_map = {}
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
