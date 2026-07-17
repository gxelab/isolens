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
        help="Predefined modification sites TSV (columns: tx_name, posn). "
        "When provided, only these positions are emitted, for all "
        "modification types, even if n_modified == 0.",
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
        "-v",
        "--verbose",
        action="store_true",
        help="Print progress to stderr",
    )
    return parser.parse_args()


def read_predefined_sites(path: str) -> dict[str, set[int]]:
    """Read a predefined modification sites TSV file.

    The file must have a header row with columns ``tx_name`` (transcript
    name) and ``posn`` (1-based position).  Additional columns are ignored.

    Args:
        path: Path to the TSV file.

    Returns:
        ``dict[str, set[int]]`` — ``{tx_name: {pos1, pos2, ...}}`` with
        1-based positions.
    """
    sites: dict[str, set[int]] = {}
    with open(path, encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        try:
            tx_col = header.index("tx_name")
            pos_col = header.index("posn")
        except ValueError as exc:
            raise ValueError(
                f"Sites file must have 'tx_name' and 'posn' columns; found: {header}"
            ) from exc
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) <= max(tx_col, pos_col):
                continue
            tx = parts[tx_col]
            try:
                pos = int(parts[pos_col])
            except ValueError:
                continue
            sites.setdefault(tx, set()).add(pos)
    return sites


def compute_transcript_stats(
    matrix: np.ndarray,
    weights: np.ndarray,
    mod_codes: list[tuple[str, int]],
    predefined_positions: set[int] | None = None,
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
        predefined_positions: Optional ``set[int]`` of 1-based positions to
            restrict output to.  When provided, only these positions are
            emitted for every modification type, even if ``n_modified == 0``.
            When ``None`` (the default), positions with ``n_modified > 0``
            for the focal modification type are emitted.

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

    for mod_str, code in mod_codes:
        mod_mask = matrix == code  # bool (n_reads, tx_length)
        n_mod = np.sum(mod_mask, axis=0, dtype=np.int32)
        w_mod = w64 @ mod_mask

        # Other modifications: any mod except the focal type
        othermod_mask = any_mod_mask & (matrix != code)
        n_othermod = np.sum(othermod_mask, axis=0, dtype=np.int32)
        w_othermod = w64 @ othermod_mask

        # Unmodified = canonical + othermod
        n_unmod = n_canonical + n_othermod
        w_unmod = w_canonical + w_othermod

        # Determine which positions to emit
        if predefined_positions is not None:
            positions_0b = sorted(
                p - 1 for p in predefined_positions if 1 <= p <= tx_length
            )
            if not positions_0b:
                continue
            positions = np.array(positions_0b, dtype=np.intp)
        else:
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

    predefined_sites: dict[str, set[int]] | None = None
    if args.sites is not None:
        predefined_sites = read_predefined_sites(args.sites)
        if args.verbose:
            n_tx = len(predefined_sites)
            n_pos = sum(len(v) for v in predefined_sites.values())
            print(
                f"[mod_sites] Predefined sites: {n_pos} positions across "
                f"{n_tx} transcripts",
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

        all_tables: list[pa.Table] = []
        processed = 0

        for tx_name in tx_names:
            pooled = pool_transcript_data(h5_files, tx_name, args.min_asp)
            if pooled is None:
                processed += 1
                continue

            matrix, weights, _tx_len = pooled

            if args.verbose and len(h5_files) > 1:
                print(
                    f"[mod_sites] {tx_name}: {matrix.shape[0]} reads "
                    f"pooled from {len(h5_files)} file(s)",
                    file=sys.stderr,
                )

            col_arrays = compute_transcript_stats(
                matrix,
                weights,
                mod_codes,
                predefined_positions=(
                    predefined_sites.get(tx_name)
                    if predefined_sites is not None
                    else None
                ),
            )

            if col_arrays is None:
                processed += 1
                continue

            n_rows = len(col_arrays["position"])

            # Add transcript_id column
            col_arrays["transcript_id"] = np.full(n_rows, tx_name, dtype=object)

            # ---- Enrich with genomic coordinates (if GTF provided) ----
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
                    gene_id = tx_gtf.gene.gene_id
                    chrom = tx_gtf.gene.chrom
                    strand = tx_gtf.gene.strand
                    gpos_vals = np.array(
                        [tx_gtf.tpos_to_gpos(int(p)) for p in col_arrays["position"]],
                        dtype=np.int32,
                    )
                    gpos_vals[gpos_vals <= 0] = -1
                    col_arrays["gene_id"] = np.full(n_rows, gene_id, dtype=object)
                    col_arrays["chrom"] = np.full(n_rows, chrom, dtype=object)
                    col_arrays["strand"] = np.full(n_rows, strand, dtype=object)
                    col_arrays["gpos"] = gpos_vals
            else:
                col_arrays["gene_id"] = np.full(n_rows, None, dtype=object)
                col_arrays["chrom"] = np.full(n_rows, None, dtype=object)
                col_arrays["strand"] = np.full(n_rows, None, dtype=object)
                col_arrays["gpos"] = np.full(n_rows, -1, dtype=np.int32)

            # Build pa.Table from column arrays
            pa_arrays = {}
            for col_name in _TSV_COLS:
                arr = col_arrays[col_name]
                pa_type = _SITES_SCHEMA.field(col_name).type
                if col_name == "gpos":
                    # Convert -1 sentinel back to None/null
                    mask = arr == -1
                    pa_arrays[col_name] = pa.array(
                        [None if m else int(v) for m, v in zip(mask, arr)],
                        type=pa.int32(),
                    )
                elif pa_type == pa.string():
                    pa_arrays[col_name] = pa.array(
                        [None if v is None else str(v) for v in arr],
                        type=pa.string(),
                    )
                elif pa_type == pa.int32():
                    pa_arrays[col_name] = pa.array(arr, type=pa.int32())
                elif pa_type == pa.float64():
                    pa_arrays[col_name] = pa.array(arr, type=pa.float64())
                else:
                    pa_arrays[col_name] = pa.array(arr, type=pa_type)

            all_tables.append(pa.table(pa_arrays))

            processed += 1
            if args.verbose and processed % 1000 == 0:
                print(
                    f"[mod_sites] Processed {processed}/{n_transcripts} transcripts...",
                    file=sys.stderr,
                )

    # ---- 4. Combine and write output ----
    if all_tables:
        combined = pa.concat_tables(all_tables)
        # Reorder columns to match schema
        combined = combined.select(_TSV_COLS)
    else:
        combined = _SITES_SCHEMA.empty_table()

    if args.verbose:
        print(
            f"[mod_sites] Total rows to write: {len(combined)}",
            file=sys.stderr,
        )

    if len(combined) == 0:
        print(
            "[mod_sites] No modification sites found — writing empty file.",
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
