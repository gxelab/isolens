#!/usr/bin/env python3
"""mod_sites: Per-position modification summaries from a mod_scan HDF5 file.

Part of the isolens toolkit.  Reads the HDF5 produced by ``mod_scan.py`` and
writes a Parquet file with one row per (transcript, position, modification_type).

See notebooks/01_mod.md for the full specification.
"""

import argparse
import gzip
import sys

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from isolens.mod_scan import CODE_CANONICAL, CODE_MISMATCH, CODE_DELETION
except ImportError:
    from mod_scan import CODE_CANONICAL, CODE_MISMATCH, CODE_DELETION


def parse_args():
    parser = argparse.ArgumentParser(
        description="mod_sites: Per-position modification summaries from HDF5"
    )
    parser.add_argument(
        "-i", "--h5",
        required=True,
        help="Input HDF5 file from mod_scan",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output file path",
    )
    parser.add_argument(
        "-f", "--format",
        choices=["parquet", "tsv"],
        default="parquet",
        help="Output format: parquet (default) or tsv",
    )
    parser.add_argument(
        "-z", "--gzip",
        action="store_true",
        help="Gzip-compress TSV output (ignored for parquet)",
    )
    parser.add_argument(
        "-p", "--min-asp", type=float, default=0.0,
        help="Minimum Oarfish assignment probability for a read to be "
             "included [default: 0.0 (no filter)]")
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print progress to stderr",
    )
    return parser.parse_args()


def compute_transcript_stats(matrix, weights, mod_codes):
    """Compute per-position statistics for a single transcript.

    Args:
        matrix: ``numpy.ndarray`` of shape ``(n_reads, tx_length)``, dtype uint8.
        weights: ``numpy.ndarray`` of shape ``(n_reads,)``, dtype float32.
        mod_codes: ``list[(mod_type_str, code)]`` — modification codes (code ≥ 4).

    Returns:
        ``list[dict]`` — one dict per (position, modification_type) with columns:
        position (1-based), modification_type, n_modified, weighted_modified,
        n_unmodified, weighted_unmodified, n_mismatch, weighted_mismatch,
        n_deletion, weighted_deletion, modification_level,
        weighted_modification_level.
    """
    n_reads, tx_length = matrix.shape
    weights_2d = weights[:, np.newaxis].astype(np.float64)  # (n_reads, 1)

    # ---- base stats (same for all modification types at each position) ----

    unmod_mask = (matrix == CODE_CANONICAL)       # bool (n_reads, tx_length)
    n_unmod = unmod_mask.sum(axis=0).astype(np.int32)
    w_unmod = (unmod_mask * weights_2d).sum(axis=0).astype(np.float64)

    mismatch_mask = (matrix == CODE_MISMATCH)
    n_mismatch = mismatch_mask.sum(axis=0).astype(np.int32)
    w_mismatch = (mismatch_mask * weights_2d).sum(axis=0).astype(np.float64)

    deletion_mask = (matrix == CODE_DELETION)
    n_del = deletion_mask.sum(axis=0).astype(np.int32)
    w_del = (deletion_mask * weights_2d).sum(axis=0).astype(np.float64)

    # ---- per-modification-type stats ----

    rows = []

    for mod_str, code in mod_codes:
        mod_mask = (matrix == code)               # bool (n_reads, tx_length)
        n_mod = mod_mask.sum(axis=0).astype(np.int32)
        w_mod = (mod_mask * weights_2d).sum(axis=0).astype(np.float64)

        # Only emit positions with at least one modification call
        positions = np.flatnonzero(n_mod > 0)
        if len(positions) == 0:
            continue

        n_mod_pos = n_mod[positions]
        w_mod_pos = w_mod[positions]

        n_unmod_pos = n_unmod[positions]
        w_unmod_pos = w_unmod[positions]

        # Modification level denominator: modified + unmodified only
        denom = n_mod_pos + n_unmod_pos
        w_denom = w_mod_pos + w_unmod_pos

        ml = np.divide(n_mod_pos, denom, where=denom > 0,
                       out=np.zeros_like(n_mod_pos, dtype=np.float64))
        w_ml = np.divide(w_mod_pos, w_denom, where=w_denom > 0,
                         out=np.zeros_like(w_mod_pos, dtype=np.float64))

        for i in range(len(positions)):
            rows.append({
                "transcript_id": "",         # filled by caller
                "position": int(positions[i]) + 1,  # 1-based
                "modification_type": mod_str,
                "n_modified": int(n_mod_pos[i]),
                "weighted_modified": float(round(w_mod_pos[i], 4)),
                "n_unmodified": int(n_unmod_pos[i]),
                "weighted_unmodified": float(round(w_unmod_pos[i], 4)),
                "n_mismatch": int(n_mismatch[positions[i]]),
                "weighted_mismatch": float(round(w_mismatch[positions[i]], 4)),
                "n_deletion": int(n_del[positions[i]]),
                "weighted_deletion": float(round(w_del[positions[i]], 4)),
                "modification_level": float(round(ml[i], 6)),
                "weighted_modification_level": float(round(w_ml[i], 6)),
            })

    return rows


def main():
    args = parse_args()

    # ---- 1. Read modification codes from HDF5 ----

    with h5py.File(args.h5, "r") as h5:
        # Parse modification codes
        mod_codes = []
        codes_grp = h5["modification_codes"]
        for mod_str, code in sorted(codes_grp.attrs.items(), key=lambda x: x[1]):
            mod_codes.append((mod_str, int(code)))

        tx_names = sorted(h5["transcripts"].keys())
        n_transcripts = len(tx_names)

        if args.verbose:
            print(f"[mod_sites] {n_transcripts} transcripts, "
                  f"{len(mod_codes)} modification types", file=sys.stderr)

        # ---- 2. Process each transcript ----

        all_rows = []
        processed = 0

        for tx_name in tx_names:
            grp = h5[f"transcripts/{tx_name}"]
            matrix = grp["matrix"][:]           # (n_reads, tx_length) uint8
            weights = grp["read_weights"][:]    # (n_reads,) float32

            if args.min_asp > 0.0:
                read_mask = weights >= args.min_asp
                if read_mask.sum() == 0:
                    processed += 1
                    continue
                matrix = matrix[read_mask]
                weights = weights[read_mask]

            tx_rows = compute_transcript_stats(matrix, weights, mod_codes)
            for row in tx_rows:
                row["transcript_id"] = tx_name
            all_rows.extend(tx_rows)

            processed += 1
            if args.verbose and processed % 1000 == 0:
                print(f"[mod_sites] Processed {processed}/{n_transcripts} "
                      f"transcripts...", file=sys.stderr)

    if args.verbose:
        print(f"[mod_sites] Total rows to write: {len(all_rows)}", file=sys.stderr)

    # ---- 3. Write output ----

    if args.format == "tsv":
        _write_tsv(all_rows, args.output, args.gzip)
    else:
        _write_parquet(all_rows, args.output)

    if args.verbose:
        print(f"[mod_sites] Done. Output written to {args.output}", file=sys.stderr)


# ---------- output writers ----------

_TSV_HEADER = (
    "transcript_id\tposition\tmodification_type\tn_modified\tweighted_modified"
    "\tn_unmodified\tweighted_unmodified\tn_mismatch\tweighted_mismatch"
    "\tn_deletion\tweighted_deletion\tmodification_level"
    "\tweighted_modification_level"
)

_TSV_COLS = [
    "transcript_id", "position", "modification_type",
    "n_modified", "weighted_modified", "n_unmodified", "weighted_unmodified",
    "n_mismatch", "weighted_mismatch", "n_deletion", "weighted_deletion",
    "modification_level", "weighted_modification_level",
]


def _write_tsv(all_rows, path, use_gzip):
    """Write rows as tab-separated values, optionally gzip-compressed."""
    open_func = gzip.open if use_gzip else open
    mode = "wt" if use_gzip else "w"

    with open_func(path, mode, encoding="utf-8") as f:
        f.write(_TSV_HEADER + "\n")
        for row in all_rows:
            f.write("\t".join(str(row[c]) for c in _TSV_COLS) + "\n")


def _write_parquet(all_rows, path):
    """Write rows as a Parquet file via pyarrow."""
    if not all_rows:
        print("[mod_sites] No modification sites found — writing empty file.",
              file=sys.stderr)
        schema = pa.schema([
            ("transcript_id", pa.string()),
            ("position", pa.int32()),
            ("modification_type", pa.string()),
            ("n_modified", pa.int32()),
            ("weighted_modified", pa.float64()),
            ("n_unmodified", pa.int32()),
            ("weighted_unmodified", pa.float64()),
            ("n_mismatch", pa.int32()),
            ("weighted_mismatch", pa.float64()),
            ("n_deletion", pa.int32()),
            ("weighted_deletion", pa.float64()),
            ("modification_level", pa.float64()),
            ("weighted_modification_level", pa.float64()),
        ])
        with pq.ParquetWriter(path, schema) as writer:
            writer.write_table(pa.table({k: pa.array([], type=schema.field(k).type)
                                         for k in schema.names}))
        return

    columns = {}
    for col in _TSV_COLS:
        values = [r[col] for r in all_rows]
        if col == "transcript_id" or col == "modification_type":
            columns[col] = pa.array(values)
        elif col == "position":
            columns[col] = pa.array(values, type=pa.int32())
        elif "n_" in col:
            columns[col] = pa.array(values, type=pa.int32())
        else:
            columns[col] = pa.array(values, type=pa.float64())

    pq.write_table(pa.table(columns), path)


if __name__ == "__main__":
    main()
