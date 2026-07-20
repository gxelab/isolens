#!/usr/bin/env python3
"""mod_sites: Per-position modification summaries from a mod_scan HDF5 file.

Part of the isolens toolkit.  Reads the HDF5 produced by ``mod_scan.py`` and
writes a Parquet file with one row per (transcript, position, modification_type).

For each position the output includes the following read-count categories
(see the :func:`compute_transcript_stats` docstring for definitions):

* **modified** — this modification type won and passed the threshold.
* **canonical** — the unmodified base won (all mod probs below canonical).
* **othermod** — another modification type won above threshold.
* **mismatch** — reference mismatch (CIGAR ``X``), preserved from CIGAR.
* **deletion** — deletion (CIGAR ``D``), preserved from CIGAR.
* **failed** — all states below the probability threshold.

See notebooks/01_mod.md for the full specification.
"""

import argparse
import concurrent.futures
import os
import sys
from contextlib import ExitStack

import h5py
import numpy as np
import pyarrow as pa

try:
    from isolens._gtf import load_gtf
    from isolens._hdf5_helpers import (
        pool_transcript_data,
        read_mod_codes,
        validate_mod_codes,
    )
    from isolens._io import write_parquet, write_tsv
    from isolens.mod_scan import (
        CODE_CANONICAL,
        CODE_DELETION,
        CODE_FAIL,
        CODE_MISMATCH,
    )
except ImportError:
    from _io import write_parquet, write_tsv  # type: ignore[no-redef]

    from _gtf import load_gtf  # type: ignore[no-redef]
    from _hdf5_helpers import (  # type: ignore[no-redef]
        pool_transcript_data,
        read_mod_codes,
        validate_mod_codes,
    )
    from mod_scan import (  # type: ignore[no-redef]
        CODE_CANONICAL,
        CODE_DELETION,
        CODE_FAIL,
        CODE_MISMATCH,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for mod_sites."""
    parser = argparse.ArgumentParser(
        description="mod_sites: Per-position modification summaries from HDF5"
    )
    parser.add_argument(
        "-i",
        "--h5",
        required=True,
        nargs="+",
        metavar="H5",
        help="Input HDF5 file(s) from mod_scan. When multiple files "
        "are provided, reads for the same transcript are pooled "
        "across all files before computing per-position statistics.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output file path",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=["parquet", "tsv"],
        default="parquet",
        help="Output format: parquet (default) or tsv",
    )
    parser.add_argument(
        "-z",
        "--gzip",
        action="store_true",
        help="Gzip-compress TSV output (ignored for parquet)",
    )
    parser.add_argument(
        "-s",
        "--sites",
        default=None,
        help="Predefined modification sites TSV (headerless; columns: "
        "transcript_id, position [1-based], mod_type [optional]). "
        "When mod_type is omitted, all modification types are emitted "
        "for that position.  Every input site appears in the output, "
        "even with zero reads.",
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
        "-x",
        "--transcripts",
        nargs="+",
        default=None,
        metavar="TX",
        help="Only process the specified transcript ID(s). "
        "[default: all transcripts in the HDF5]",
    )
    parser.add_argument(
        "-g",
        "--gtf",
        default=None,
        help="GTF annotation file for mapping transcript coordinates "
        "to genomic coordinates. When provided, three additional "
        "columns (chrom, strand, gpos) are appended to the output.",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Number of worker threads for parallel transcript "
        "processing [default: min(4, cpu_count)]. Set to 1 for "
        "serial (deterministic transcript ordering).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print progress to stderr",
    )
    return parser.parse_args()


def read_predefined_sites(path: str) -> dict[str, dict[int, set[str] | None]]:
    """Read a predefined modification sites TSV file.

    The file is headerless with 2+ tab-separated columns:
    ``transcript_id``, ``position`` (1-based), ``mod_type`` (optional).
    Extra columns are ignored.

    When *mod_type* is omitted or empty, every known modification type
    is emitted for that position.  When *mod_type* is provided, only
    that specific type is emitted.  If the same position appears both
    with and without *mod_type*, "all types" wins.

    Args:
        path: Path to the headerless TSV file.

    Returns:
        ``dict[str, dict[int, set[str] | None]]`` —
        ``{tx_name: {pos: mod_types_or_None}}`` where ``None`` means
        "all modification types" and a ``set[str]`` means only those
        types.
    """
    sites: dict[str, dict[int, set[str] | None]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            tx = parts[0].strip()
            try:
                pos = int(parts[1].strip())
            except ValueError:
                continue

            # Determine mod_type: None → all types, str → specific type
            if len(parts) < 3 or not parts[2].strip():
                mod_type = None
            else:
                mod_type = parts[2].strip()

            if tx not in sites:
                sites[tx] = {}

            pos_dict = sites[tx]
            if mod_type is None:
                # "all types" overrides any previous specific entries
                pos_dict[pos] = None
            elif pos not in pos_dict:
                pos_dict[pos] = {mod_type}
            elif pos_dict[pos] is not None:
                pos_dict[pos].add(mod_type)
    return sites


def compute_transcript_stats(
    matrix: np.ndarray,
    weights: np.ndarray,
    mod_codes: list[tuple[str, int]],
    predefined_mods: dict[int, set[str] | None] | None = None,
) -> dict[str, np.ndarray] | None:
    """Compute per-position statistics for a single transcript.

    For each (position, modification_type) pair the following columns
    are computed:

    * **n_modified** — number of reads where this mod type won (including
      weighted variant ``wt_modified``).
    * **n_canonical** — number of reads where the canonical (unmodified)
      base won (including ``wt_canonical``).
    * **n_othermod** — number of reads where a different, non-focal
      modification type won (including ``wt_othermod``).
    * **n_unmodified** = n_canonical + n_othermod (and ``wt_unmodified``).
    * **n_mismatch** — number of reads with a CIGAR reference mismatch
      at this position (and ``wt_mismatch``).
    * **n_deletion** — number of reads with a CIGAR deletion at this
      position (and ``wt_deletion``).
    * **n_failed** — number of reads where no state exceeded the
      probability threshold (and ``wt_failed``).
    * **mod_level** = n_modified / (n_modified + n_unmodified).
    * **wt_mod_level** — weighted variant using assignment probabilities.

    Args:
        matrix: ``numpy.ndarray`` of shape ``(n_reads, tx_length)``,
            dtype uint8.  Encoded per ``mod_scan.CODE_*`` constants.
        weights: ``numpy.ndarray`` of shape ``(n_reads,)``, dtype float32.
            Oarfish assignment probabilities for each read.
        mod_codes: ``list[(mod_type_str, code)]`` — modification codes
            (code ≥ 4) read from the HDF5 ``/modification_codes`` group.
        predefined_mods: Optional ``dict[int, set[str] | None]`` mapping
            1-based positions to the set of modification types to emit,
            or ``None`` to emit all known modification types at that
            position.  When ``None`` (the default), positions with
            ``n_modified > 0`` for the focal modification type are emitted.
            See :func:`read_predefined_sites` for the dict format.

    Returns:
        ``dict[str, np.ndarray]`` mapping column names to 1-D arrays,
        or ``None`` if no positions meet the emission criteria.
    """
    n_reads, tx_length = matrix.shape
    w64 = weights.astype(np.float64)  # (n_reads,) float64

    # ---- base stats (same for all modification types at each position) ----

    mismatch_mask = matrix == CODE_MISMATCH  # bool (n_reads, tx_length)
    n_mismatch = np.sum(mismatch_mask, axis=0, dtype=np.int32)
    w_mismatch = w64 @ mismatch_mask  # (tx_length,) float64

    deletion_mask = matrix == CODE_DELETION
    n_del = np.sum(deletion_mask, axis=0, dtype=np.int32)
    w_del = w64 @ deletion_mask

    failed_mask = matrix == CODE_FAIL
    n_failed = np.sum(failed_mask, axis=0, dtype=np.int32)
    w_failed = w64 @ failed_mask

    canonical_mask = matrix == CODE_CANONICAL  # bool (n_reads, tx_length)
    n_canonical = np.sum(canonical_mask, axis=0, dtype=np.int32)
    w_canonical = w64 @ canonical_mask

    # ---- pre-compute mask shared across all mod types ----
    any_mod_mask = (matrix >= 4) & (matrix != CODE_FAIL)
    n_any_mod = np.sum(any_mod_mask, axis=0, dtype=np.int32)
    w_any_mod = w64 @ any_mod_mask

    # ---- per-modification-type stats ----

    # Accumulate column arrays across mod types
    col_positions: list[np.ndarray] = []
    col_mod_type: list[np.ndarray] = []
    col_n_mod: list[np.ndarray] = []
    col_w_mod: list[np.ndarray] = []
    col_n_unmod: list[np.ndarray] = []
    col_w_unmod: list[np.ndarray] = []
    col_n_canon: list[np.ndarray] = []
    col_w_canon: list[np.ndarray] = []
    col_n_other: list[np.ndarray] = []
    col_w_other: list[np.ndarray] = []
    col_n_mm: list[np.ndarray] = []
    col_w_mm: list[np.ndarray] = []
    col_n_del: list[np.ndarray] = []
    col_w_del: list[np.ndarray] = []
    col_n_fail: list[np.ndarray] = []
    col_w_fail: list[np.ndarray] = []
    col_ml: list[np.ndarray] = []
    col_wml: list[np.ndarray] = []

    if predefined_mods is not None:
        # ---- Sites-driven path: emit exactly the requested (pos, mod_type) pairs ----
        code_map = dict(mod_codes)

        # Group requested positions by modification type for vectorized ops
        mod_to_positions: dict[str, list[int]] = {}
        for pos_1b, mod_types in predefined_mods.items():
            if pos_1b < 1 or pos_1b > tx_length:
                continue
            pos_0b = pos_1b - 1
            if mod_types is None:
                for mod_str, _code in mod_codes:
                    mod_to_positions.setdefault(mod_str, []).append(pos_0b)
            else:
                for mod_str in mod_types:
                    if mod_str in code_map:
                        mod_to_positions.setdefault(mod_str, []).append(pos_0b)

        for mod_str, positions_0b_list in mod_to_positions.items():
            code = code_map[mod_str]
            positions = np.array(sorted(positions_0b_list), dtype=np.intp)
            n_pos = len(positions)

            # Modification-specific stats at requested positions
            mod_mask = matrix[:, positions] == code  # (n_reads, n_pos)
            n_mod = np.sum(mod_mask, axis=0, dtype=np.int32)
            w_mod = w64 @ mod_mask

            n_othermod = n_any_mod[positions] - n_mod
            w_othermod = w_any_mod[positions] - w_mod

            n_unmod = n_canonical[positions] + n_othermod
            w_unmod = w_canonical[positions] + w_othermod

            denom = n_mod + n_unmod
            w_denom = w_mod + w_unmod

            ml = np.divide(
                n_mod.astype(np.float64),
                denom.astype(np.float64),
                where=denom > 0,
                out=np.zeros(n_pos, dtype=np.float64),
            )
            w_ml = np.divide(
                w_mod,
                w_denom,
                where=w_denom > 0,
                out=np.zeros(n_pos, dtype=np.float64),
            )

            col_positions.append(positions + 1)  # 1-based
            col_mod_type.append(np.full(n_pos, mod_str, dtype=object))
            col_n_mod.append(n_mod)
            col_w_mod.append(w_mod)
            col_n_unmod.append(n_unmod)
            col_w_unmod.append(w_unmod)
            col_n_canon.append(n_canonical[positions])
            col_w_canon.append(w_canonical[positions])
            col_n_other.append(n_othermod)
            col_w_other.append(w_othermod)
            col_n_mm.append(n_mismatch[positions])
            col_w_mm.append(w_mismatch[positions])
            col_n_del.append(n_del[positions])
            col_w_del.append(w_del[positions])
            col_n_fail.append(n_failed[positions])
            col_w_fail.append(w_failed[positions])
            col_ml.append(ml)
            col_wml.append(w_ml)
    else:
        # ---- Original path: emit positions with n_modified > 0 per mod type ----
        for mod_str, code in mod_codes:
            mod_mask = matrix == code  # bool (n_reads, tx_length)
            n_mod = np.sum(mod_mask, axis=0, dtype=np.int32)
            w_mod = w64 @ mod_mask

            # Other modifications: any mod except the focal type (derived by
            # subtraction — n_othermod = n_any_mod - n_mod is exact integer
            # arithmetic; w_othermod may differ by ≤ 1.1e-12 from the true
            # sum due to float64 cancellation, but this is erased by the
            # np.round(..., 4) applied before output.)
            n_othermod = n_any_mod - n_mod
            w_othermod = w_any_mod - w_mod

            # Unmodified = canonical + othermod
            n_unmod = n_canonical + n_othermod
            w_unmod = w_canonical + w_othermod

            positions = np.flatnonzero(n_mod > 0)
            if len(positions) == 0:
                continue

            n_pos = len(positions)

            # Modification level denominator
            denom = n_mod[positions] + n_unmod[positions]
            w_denom = w_mod[positions] + w_unmod[positions]

            ml = np.divide(
                n_mod[positions].astype(np.float64),
                denom.astype(np.float64),
                where=denom > 0,
                out=np.zeros(n_pos, dtype=np.float64),
            )
            w_ml = np.divide(
                w_mod[positions],
                w_denom,
                where=w_denom > 0,
                out=np.zeros(n_pos, dtype=np.float64),
            )

            col_positions.append(positions + 1)  # 1-based
            col_mod_type.append(np.full(n_pos, mod_str, dtype=object))
            col_n_mod.append(n_mod[positions])
            col_w_mod.append(w_mod[positions])
            col_n_unmod.append(n_unmod[positions])
            col_w_unmod.append(w_unmod[positions])
            col_n_canon.append(n_canonical[positions])
            col_w_canon.append(w_canonical[positions])
            col_n_other.append(n_othermod[positions])
            col_w_other.append(w_othermod[positions])
            col_n_mm.append(n_mismatch[positions])
            col_w_mm.append(w_mismatch[positions])
            col_n_del.append(n_del[positions])
            col_w_del.append(w_del[positions])
            col_n_fail.append(n_failed[positions])
            col_w_fail.append(w_failed[positions])
            col_ml.append(ml)
            col_wml.append(w_ml)

    if not col_positions:
        return None

    return {
        "position": np.concatenate(col_positions),
        "mod_type": np.concatenate(col_mod_type),
        "n_modified": np.concatenate(col_n_mod),
        "wt_modified": np.round(np.concatenate(col_w_mod), 4),
        "n_unmodified": np.concatenate(col_n_unmod),
        "wt_unmodified": np.round(np.concatenate(col_w_unmod), 4),
        "n_canonical": np.concatenate(col_n_canon),
        "wt_canonical": np.round(np.concatenate(col_w_canon), 4),
        "n_othermod": np.concatenate(col_n_other),
        "wt_othermod": np.round(np.concatenate(col_w_other), 4),
        "n_mismatch": np.concatenate(col_n_mm),
        "wt_mismatch": np.round(np.concatenate(col_w_mm), 4),
        "n_deletion": np.concatenate(col_n_del),
        "wt_deletion": np.round(np.concatenate(col_w_del), 4),
        "n_failed": np.concatenate(col_n_fail),
        "wt_failed": np.round(np.concatenate(col_w_fail), 4),
        "mod_level": np.round(np.concatenate(col_ml), 6),
        "wt_mod_level": np.round(np.concatenate(col_wml), 6),
    }


def _compute_transcript(
    tx_name: str,
    matrix: np.ndarray,
    weights: np.ndarray,
    mod_codes: list[tuple[str, int]],
    predefined_mods: dict[int, set[str] | None] | None,
) -> dict[str, np.ndarray] | None:
    """Compute stats for one transcript with transcript_id prepopulated.

    Thin wrapper around :func:`compute_transcript_stats` that also
    adds the ``transcript_id`` column.  This function only does pure
    NumPy work (no HDF5 I/O, no GTF lookups, no PyArrow) and is safe
    to call from worker threads.

    Args:
        tx_name: Transcript identifier.
        matrix: ``(n_reads, tx_length)`` uint8 array.
        weights: ``(n_reads,)`` float32 assignment probabilities.
        mod_codes: ``list[(mod_str, code)]`` sorted by code.
        predefined_mods: Optional dict of 1-based positions →
            set of mod types or None (see :func:`read_predefined_sites`).

    Returns:
        Dict of column→array, or ``None`` if no positions emitted.
    """
    col_arrays = compute_transcript_stats(
        matrix, weights, mod_codes, predefined_mods=predefined_mods
    )
    if col_arrays is None:
        return None
    col_arrays["transcript_id"] = np.full(
        len(col_arrays["position"]), tx_name, dtype=object
    )
    return col_arrays


def _make_zero_rows(
    tx_name: str,
    predefined_mods: dict[int, set[str] | None],
    mod_codes: list[tuple[str, int]],
) -> dict[str, np.ndarray] | None:
    """Generate all-zero stats rows for a transcript absent from all HDF5 files.

    When a transcript appears in the predefined sites file but has no reads
    in any input HDF5 file (or all its reads are filtered by *min_asp*),
    this function produces rows with every count set to zero so the
    completeness guarantee is upheld.

    Args:
        tx_name: Transcript identifier.
        predefined_mods: ``{pos_1b: set_of_mod_types | None}`` for this
            transcript (see :func:`read_predefined_sites`).
        mod_codes: ``list[(mod_str, code)]`` sorted by code.

    Returns:
        Dict of column→array, or ``None`` if no valid targets.
    """
    code_map = dict(mod_codes)

    targets: list[tuple[int, str]] = []  # (pos_1b, mod_str)
    for pos_1b, mod_types in predefined_mods.items():
        if mod_types is None:
            for mod_str, _code in mod_codes:
                targets.append((pos_1b, mod_str))
        else:
            for mod_str in mod_types:
                if mod_str in code_map:
                    targets.append((pos_1b, mod_str))

    if not targets:
        return None

    n = len(targets)
    positions_arr = np.array([p for p, _ in targets], dtype=np.int32)
    mod_types_arr = np.array([m for _, m in targets], dtype=object)

    return {
        "transcript_id": np.full(n, tx_name, dtype=object),
        "position": positions_arr,
        "mod_type": mod_types_arr,
        "n_modified": np.zeros(n, dtype=np.int32),
        "wt_modified": np.zeros(n, dtype=np.float64),
        "n_unmodified": np.zeros(n, dtype=np.int32),
        "wt_unmodified": np.zeros(n, dtype=np.float64),
        "n_canonical": np.zeros(n, dtype=np.int32),
        "wt_canonical": np.zeros(n, dtype=np.float64),
        "n_othermod": np.zeros(n, dtype=np.int32),
        "wt_othermod": np.zeros(n, dtype=np.float64),
        "n_mismatch": np.zeros(n, dtype=np.int32),
        "wt_mismatch": np.zeros(n, dtype=np.float64),
        "n_deletion": np.zeros(n, dtype=np.int32),
        "wt_deletion": np.zeros(n, dtype=np.float64),
        "n_failed": np.zeros(n, dtype=np.int32),
        "wt_failed": np.zeros(n, dtype=np.float64),
        "mod_level": np.zeros(n, dtype=np.float64),
        "wt_mod_level": np.zeros(n, dtype=np.float64),
    }


# ---------- main ----------


def main(args: argparse.Namespace | None = None) -> None:
    """Compute per-position modification summaries from mod_scan HDF5 files.

    Reads one or more HDF5 files produced by ``mod_scan`` and writes a
    Parquet (or TSV) file with one row per (transcript, position,
    modification_type) containing read counts and weighted modification
    levels.  When multiple HDF5 files are provided, reads for the same
    transcript are pooled across all files.
    """
    if args is None:
        args = parse_args()

    # ---- 0. Read predefined sites (optional) ----

    predefined_sites: dict[str, dict[int, set[str] | None]] | None = None
    if args.sites is not None:
        predefined_sites = read_predefined_sites(args.sites)
        if args.verbose:
            n_tx = len(predefined_sites)
            n_pos = sum(len(v) for v in predefined_sites.values())
            n_rows = sum(
                len(mod_types) if mod_types is not None else 0
                for pos_dict in predefined_sites.values()
                for mod_types in pos_dict.values()
            )
            print(
                f"[mod_sites] Predefined sites: {n_pos} positions across "
                f"{n_tx} transcripts ({n_rows} mod_type entries)",
                file=sys.stderr,
            )

    # ---- 0b. Parse GTF annotation (optional) ----

    gtf: dict | None = None
    if args.gtf is not None:
        gtf = load_gtf(args.gtf)
        if args.verbose:
            print(
                f"[mod_sites] Loaded {len(gtf)} transcripts from GTF",
                file=sys.stderr,
            )

    # ---- 1. Open all HDF5 files and validate modification codes ----

    with ExitStack() as stack:
        h5_files = [stack.enter_context(h5py.File(f, "r")) for f in args.h5]

        if args.verbose:
            print(
                f"[mod_sites] Opened {len(h5_files)} HDF5 file(s)",
                file=sys.stderr,
            )

        # Read and validate modification codes
        all_mod_maps = [read_mod_codes(h5) for h5 in h5_files]
        try:
            mod_code_map = validate_mod_codes(all_mod_maps, list(args.h5))
        except ValueError as exc:
            print(f"[mod_sites] Error: {exc}", file=sys.stderr)
            sys.exit(1)
        mod_codes = sorted(mod_code_map.items(), key=lambda x: x[1])

        if args.verbose:
            print(
                f"[mod_sites] {len(mod_codes)} modification types: "
                f"{[m for m, _c in mod_codes]}",
                file=sys.stderr,
            )

        # ---- 2. Build union of transcript names across all files ----

        all_tx_sets = [set(h5["transcripts"].keys()) for h5 in h5_files]
        tx_names = sorted(set.union(*all_tx_sets))

        if args.transcripts is not None:
            requested = set(args.transcripts)
            tx_names = sorted(tx for tx in tx_names if tx in requested)
            if args.verbose:
                print(
                    f"[mod_sites] Filtered to {len(tx_names)}/"
                    f"{len(requested)} requested transcripts",
                    file=sys.stderr,
                )
        n_transcripts = len(tx_names)

        if args.verbose:
            file_counts = ", ".join(
                f"{f}: {len(s)} tx" for f, s in zip(args.h5, all_tx_sets)
            )
            print(
                f"[mod_sites] {n_transcripts} unique transcripts across "
                f"{len(h5_files)} files ({file_counts})",
                file=sys.stderr,
            )

        # ---- 3. Process each transcript (pooling across files) ----

        all_results: list[dict[str, np.ndarray]] = []
        processed = 0

        # Helper to enrich a result dict with GTF / sentinel columns.
        def _add_gtf_columns(col_arrays: dict[str, np.ndarray], tx_name: str) -> None:
            n_rows = len(col_arrays["position"])
            if gtf is not None:
                tx_gtf = gtf.get(tx_name)
                if tx_gtf is None:
                    if args.verbose:
                        print(
                            f"[mod_sites] Warning: {tx_name} not found in GTF",
                            file=sys.stderr,
                        )
                    col_arrays["gene_id"] = np.full(n_rows, None, dtype=object)
                    col_arrays["chrom"] = np.full(n_rows, None, dtype=object)
                    col_arrays["strand"] = np.full(n_rows, None, dtype=object)
                    col_arrays["gpos"] = np.full(n_rows, -1, dtype=np.int32)
                else:
                    col_arrays["gene_id"] = np.full(
                        n_rows, tx_gtf.gene.gene_id, dtype=object
                    )
                    col_arrays["chrom"] = np.full(
                        n_rows, tx_gtf.gene.chrom, dtype=object
                    )
                    col_arrays["strand"] = np.full(
                        n_rows, tx_gtf.gene.strand, dtype=object
                    )
                    gpos_vals = np.array(
                        [tx_gtf.tpos_to_gpos(int(p)) for p in col_arrays["position"]],
                        dtype=np.int32,
                    )
                    gpos_vals[gpos_vals <= 0] = -1
                    col_arrays["gpos"] = gpos_vals
            else:
                col_arrays["gene_id"] = np.full(n_rows, None, dtype=object)
                col_arrays["chrom"] = np.full(n_rows, None, dtype=object)
                col_arrays["strand"] = np.full(n_rows, None, dtype=object)
                col_arrays["gpos"] = np.full(n_rows, -1, dtype=np.int32)

        if predefined_sites is not None:
            # ---- Sites-driven path: iterate over sites file transcripts ----
            sites_tx_names = sorted(predefined_sites.keys())
            if args.transcripts is not None:
                requested = set(args.transcripts)
                sites_tx_names = [t for t in sites_tx_names if t in requested]
                if args.verbose:
                    print(
                        f"[mod_sites] Filtered to {len(sites_tx_names)}/"
                        f"{len(requested)} requested transcripts",
                        file=sys.stderr,
                    )
            n_transcripts = len(sites_tx_names)

            for tx_name in sites_tx_names:
                mods_for_tx = predefined_sites[tx_name]
                pooled = pool_transcript_data(h5_files, tx_name, args.min_asp)

                if pooled is not None:
                    matrix, weights, _tx_len = pooled
                    col_arrays = _compute_transcript(
                        tx_name,
                        matrix,
                        weights,
                        mod_codes,
                        predefined_mods=mods_for_tx,
                    )
                else:
                    col_arrays = _make_zero_rows(
                        tx_name, mods_for_tx, mod_codes
                    )

                if col_arrays is None:
                    processed += 1
                    continue

                _add_gtf_columns(col_arrays, tx_name)
                all_results.append(col_arrays)

                processed += 1
                if args.verbose and processed % 1000 == 0:
                    print(
                        f"[mod_sites] Processed {processed}/{n_transcripts} "
                        f"transcripts...",
                        file=sys.stderr,
                    )
        elif getattr(args, "threads", 1) <= 1:
            # ---- Serial path (--threads 1, deterministic) ----
            for tx_name in tx_names:
                pooled = pool_transcript_data(h5_files, tx_name, args.min_asp)
                if pooled is None:
                    processed += 1
                    continue

                matrix, weights, _tx_len = pooled

                col_arrays = _compute_transcript(
                    tx_name,
                    matrix,
                    weights,
                    mod_codes,
                    predefined_mods=None,
                )

                if col_arrays is None:
                    processed += 1
                    continue

                _add_gtf_columns(col_arrays, tx_name)
                all_results.append(col_arrays)

                processed += 1
                if args.verbose and processed % 1000 == 0:
                    print(
                        f"[mod_sites] Processed {processed}/{n_transcripts} "
                        f"transcripts...",
                        file=sys.stderr,
                    )
        else:
            # ---- Parallel path (ThreadPoolExecutor) ----
            max_workers = getattr(args, "threads", 1)
            max_pending = max_workers * 2

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                future_to_tx: dict[concurrent.futures.Future, str] = {}
                tx_iter = iter(tx_names)
                tx_iter_exhausted = False

                while future_to_tx or not tx_iter_exhausted:
                    # Submit new work while below limit
                    while len(future_to_tx) < max_pending and not tx_iter_exhausted:
                        try:
                            tx_name = next(tx_iter)
                        except StopIteration:
                            tx_iter_exhausted = True
                            break
                        pooled = pool_transcript_data(h5_files, tx_name, args.min_asp)
                        if pooled is None:
                            processed += 1
                            continue
                        matrix, weights, _tx_len = pooled
                        future = executor.submit(
                            _compute_transcript,
                            tx_name,
                            matrix,
                            weights,
                            mod_codes,
                            None,
                        )
                        future_to_tx[future] = tx_name

                    if not future_to_tx:
                        break

                    # Wait for at least one completion
                    done, _ = concurrent.futures.wait(
                        future_to_tx,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in done:
                        tx_name = future_to_tx.pop(future)
                        try:
                            col_arrays = future.result()
                        except Exception:
                            for f in future_to_tx:
                                f.cancel()
                            raise
                        processed += 1
                        if col_arrays is None:
                            continue

                        # Enrich with GTF / sentinel columns (main thread)
                        _add_gtf_columns(col_arrays, tx_name)
                        all_results.append(col_arrays)

                        if args.verbose and processed % 1000 == 0:
                            print(
                                f"[mod_sites] Processed {processed}/"
                                f"{n_transcripts} transcripts...",
                                file=sys.stderr,
                            )

    # ---- 4. Build output and write ----

    if not all_results:
        # No modification sites found — write empty output
        empty_table = _SITES_SCHEMA.empty_table()
        print(
            "[mod_sites] No modification sites found — writing empty file.",
            file=sys.stderr,
        )
        if args.format == "tsv":
            write_tsv(empty_table, args.output, _TSV_HEADER, _TSV_COLS, args.gzip)
        else:
            write_parquet(empty_table, args.output, _SITES_SCHEMA, _TSV_COLS)
        if args.verbose:
            print(
                f"[mod_sites] Done. Output written to {args.output}",
                file=sys.stderr,
            )
        return

    # Concatenate numpy arrays across all transcripts (one pass per column)
    combined_arrays: dict[str, np.ndarray] = {}
    for col_name in _TSV_COLS:
        combined_arrays[col_name] = np.concatenate([r[col_name] for r in all_results])

    # Build a single pa.Table from the combined arrays
    pa_arrays: dict[str, pa.Array] = {}
    for col_name in _TSV_COLS:
        arr = combined_arrays[col_name]
        pa_type = _SITES_SCHEMA.field(col_name).type
        if col_name == "gpos":
            # Use pyarrow mask for vectorized null handling
            null_mask = arr == -1
            pa_arrays[col_name] = pa.array(arr, type=pa.int32(), mask=null_mask)
        elif pa_type == pa.string():
            # pa.array handles None → null in object arrays natively
            pa_arrays[col_name] = pa.array(arr, type=pa.string())
        elif pa_type == pa.int32():
            pa_arrays[col_name] = pa.array(arr, type=pa.int32())
        elif pa_type == pa.float64():
            pa_arrays[col_name] = pa.array(arr, type=pa.float64())
        else:
            pa_arrays[col_name] = pa.array(arr, type=pa_type)

    combined = pa.table(pa_arrays)

    if args.verbose:
        print(
            f"[mod_sites] Total rows to write: {len(combined)}",
            file=sys.stderr,
        )

    if args.format == "tsv":
        write_tsv(combined, args.output, _TSV_HEADER, _TSV_COLS, args.gzip)
    else:
        write_parquet(combined, args.output, _SITES_SCHEMA, _TSV_COLS)

    if args.verbose:
        print(f"[mod_sites] Done. Output written to {args.output}", file=sys.stderr)


# ---------- output writers ----------

_TSV_HEADER = (
    "transcript_id\tposition\tmod_type"
    "\tgene_id\tchrom\tstrand\tgpos"
    "\tn_modified\twt_modified"
    "\tn_unmodified\twt_unmodified\tn_canonical\twt_canonical"
    "\tn_othermod\twt_othermod\tn_mismatch\twt_mismatch"
    "\tn_deletion\twt_deletion\tn_failed\twt_failed"
    "\tmod_level\twt_mod_level"
)

_TSV_COLS = [
    "transcript_id",
    "position",
    "mod_type",
    "gene_id",
    "chrom",
    "strand",
    "gpos",
    "n_modified",
    "wt_modified",
    "n_unmodified",
    "wt_unmodified",
    "n_canonical",
    "wt_canonical",
    "n_othermod",
    "wt_othermod",
    "n_mismatch",
    "wt_mismatch",
    "n_deletion",
    "wt_deletion",
    "n_failed",
    "wt_failed",
    "mod_level",
    "wt_mod_level",
]


_SITES_SCHEMA = pa.schema(
    [
        ("transcript_id", pa.string()),
        ("position", pa.int32()),
        ("mod_type", pa.string()),
        ("gene_id", pa.string()),
        ("chrom", pa.string()),
        ("strand", pa.string()),
        ("gpos", pa.int32()),
        ("n_modified", pa.int32()),
        ("wt_modified", pa.float64()),
        ("n_unmodified", pa.int32()),
        ("wt_unmodified", pa.float64()),
        ("n_canonical", pa.int32()),
        ("wt_canonical", pa.float64()),
        ("n_othermod", pa.int32()),
        ("wt_othermod", pa.float64()),
        ("n_mismatch", pa.int32()),
        ("wt_mismatch", pa.float64()),
        ("n_deletion", pa.int32()),
        ("wt_deletion", pa.float64()),
        ("n_failed", pa.int32()),
        ("wt_failed", pa.float64()),
        ("mod_level", pa.float64()),
        ("wt_mod_level", pa.float64()),
    ]
)


if __name__ == "__main__":
    main()
