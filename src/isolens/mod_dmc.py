#!/usr/bin/env python3
"""mod_dmc: Differential modification calling between two conditions.

Compares modification levels between two experimental conditions at
each (transcript, position, modification-type) site using read-level
weighted logistic regression.  Reads from multiple HDF5 files are
pooled within each condition before testing.

Site key: ``(transcript_id, position, mod_type)``.
"""

import argparse
import sys
from contextlib import ExitStack
from typing import Any

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from isolens._stats import bh_fdr, weighted_logistic_test
    from isolens.mod_scan import (
        CODE_DELETION,
        CODE_FAIL,
        CODE_MISMATCH,
        CODE_UNCOVERED,
    )
except ImportError:
    from _stats import bh_fdr, weighted_logistic_test  # type: ignore[no-redef]
    from mod_scan import (  # type: ignore[no-redef]
        CODE_DELETION,
        CODE_FAIL,
        CODE_MISMATCH,
        CODE_UNCOVERED,
    )

# ---------- constants ----------

_OUTPUT_COLS = [
    "transcript_id",
    "position",
    "mod_type",
    "gene_id",
    "chrom",
    "strand",
    "gpos",
    "n_modified_1",
    "n_unmodified_1",
    "n_modified_2",
    "n_unmodified_2",
    "wt_modified_1",
    "wt_unmodified_1",
    "wt_modified_2",
    "wt_unmodified_2",
    "mod_level_1",
    "mod_level_2",
    "wt_mod_level_1",
    "wt_mod_level_2",
    "delta_mod_level",
    "delta_wt_mod_level",
    "log2_or",
    "p_value",
    "q_value",
]

_TSV_HEADER = "\t".join(_OUTPUT_COLS)

# Columns we read from the site summary Parquet / TSV
_SITE_COLS = [
    "transcript_id",
    "position",
    "mod_type",
    "n_modified",
    "wt_modified",
    "n_unmodified",
    "wt_unmodified",
    "mod_level",
    "wt_mod_level",
    "gene_id",
    "chrom",
    "strand",
    "gpos",
]


# ---------- CLI ----------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for mod_dmc."""
    parser = argparse.ArgumentParser(
        description="mod_dmc: Differential modification calling between "
        "two conditions"
    )
    parser.add_argument(
        "--h5-1",
        required=True,
        nargs="+",
        metavar="H5",
        help="Input HDF5 file(s) for condition 1.  When multiple files "
        "are provided, reads for the same transcript are pooled.",
    )
    parser.add_argument(
        "--h5-2",
        required=True,
        nargs="+",
        metavar="H5",
        help="Input HDF5 file(s) for condition 2.  When multiple files "
        "are provided, reads for the same transcript are pooled.",
    )
    parser.add_argument(
        "--sites-1",
        required=True,
        metavar="FILE",
        help="Pooled site summary for condition 1 (Parquet or TSV/TSV.GZ "
        "from mod_sites)",
    )
    parser.add_argument(
        "--sites-2",
        required=True,
        metavar="FILE",
        help="Pooled site summary for condition 2 (Parquet or TSV/TSV.GZ "
        "from mod_sites)",
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Output file path"
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
        help="Only process the specified transcript ID(s).  "
        "[default: all transcripts in the HDF5]",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print progress to stderr",
    )
    return parser.parse_args()


# ---------- site-summary readers ----------


def _read_sites_parquet(path: str) -> dict[tuple, dict[str, Any]]:
    """Read a Parquet site summary into a flat dict keyed by
    ``(transcript_id, position, mod_type)``."""
    table = pq.read_table(path, columns=_SITE_COLS)
    sites: dict[tuple, dict[str, Any]] = {}
    col_data = {c: table.column(c) for c in _SITE_COLS}
    for i in range(len(table)):
        tx = col_data["transcript_id"][i].as_py()
        pos = col_data["position"][i].as_py()
        mod = col_data["mod_type"][i].as_py()
        key = (tx, pos, mod)
        row = {}
        for c in _SITE_COLS:
            val = col_data[c][i].as_py()
            row[c] = val
        sites[key] = row
    return sites


def _read_sites_tsv(path: str) -> dict[tuple, dict[str, Any]]:
    """Read a TSV/TSV.GZ site summary into a flat dict keyed by
    ``(transcript_id, position, mod_type)``."""
    import gzip

    open_func = gzip.open if path.endswith(".gz") else open
    mode = "rt" if path.endswith(".gz") else "r"
    sites: dict[tuple, dict[str, Any]] = {}
    with open_func(path, mode, encoding="utf-8") as fh:
        header = fh.readline().strip().split("\t")
        col_idx = {c: header.index(c) for c in _SITE_COLS if c in header}
        for line in fh:
            parts = line.strip().split("\t")
            if not parts:
                continue
            tx = parts[col_idx["transcript_id"]]
            pos = int(parts[col_idx["position"]])
            mod = parts[col_idx["mod_type"]]
            key = (tx, pos, mod)
            row: dict[str, Any] = {}
            for c in _SITE_COLS:
                if c not in col_idx:
                    row[c] = None
                    continue
                val = parts[col_idx[c]]
                if val == "NA":
                    row[c] = None
                elif c in ("n_modified", "n_unmodified", "position"):
                    row[c] = int(val)
                elif c in (
                    "wt_modified", "wt_unmodified",
                    "mod_level", "wt_mod_level",
                ):
                    row[c] = float(val)
                elif c == "gpos":
                    row[c] = int(val) if val != "NA" else None
                else:
                    row[c] = val
            sites[key] = row
    return sites


def read_site_summary_full(path: str) -> dict[tuple, dict[str, Any]]:
    """Read a modification site summary file (Parquet or TSV/TSV.GZ).

    Returns a dict keyed by ``(transcript_id, position, mod_type)``,
    where each value is a dict with all site-summary columns.
    """
    if path.endswith(".parquet"):
        return _read_sites_parquet(path)
    return _read_sites_tsv(path)


# ---------- per-site read extraction ----------


def _extract_site_reads(
    matrix: np.ndarray,
    weights: np.ndarray,
    position_1b: int,
    mod_code: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract binary modified/unmodified vector and weights for a site.

    Filters out uncovered, mismatch, deletion, failed, and other-mod
    reads so that only *modified* (this mod type) and *unmodified*
    (canonical) reads remain.

    Parameters
    ----------
    matrix : ndarray of shape (n_reads, tx_length), dtype uint8
    weights : ndarray of shape (n_reads,), dtype float32 or float64
    position_1b : int
        1-based position in the transcript.
    mod_code : int
        Integer code for the focal modification type (≥ 4).

    Returns
    -------
    (y, w) or None
        *y* is a float64 array of 0.0 / 1.0 for valid reads only.
        *w* is the corresponding float64 weight vector.
        Returns ``None`` when no valid reads remain.
    """
    col = matrix[:, position_1b - 1]
    valid = (
        (col != CODE_UNCOVERED)
        & (col != CODE_MISMATCH)
        & (col != CODE_DELETION)
        & (col != CODE_FAIL)
    )
    # Exclude reads where a *different* tracked modification won
    other_mod = (col >= 4) & (col != mod_code) & (col != CODE_FAIL)
    valid = valid & (~other_mod)
    if valid.sum() == 0:
        return None
    y = (col[valid] == mod_code).astype(np.float64)
    w = weights[valid].astype(np.float64)
    return y, w


# ---------- HDF5 helpers ----------


def _read_mod_codes(h5: h5py.File) -> dict[str, int]:
    """Read modification codes from an open HDF5 file."""
    return {
        mod_str: int(code)
        for mod_str, code in h5["modification_codes"].attrs.items()
    }


def _validate_mod_codes(
    mod_maps: list[dict[str, int]],
    filenames: list[str],
) -> dict[str, int]:
    """Verify all HDF5 files have identical modification codes."""
    reference = mod_maps[0]
    for i, code_map in enumerate(mod_maps[1:], start=1):
        if code_map != reference:
            ref_str = "; ".join(
                f"{k}={v}" for k, v in sorted(reference.items())
            )
            file_str = "; ".join(
                f"{k}={v}" for k, v in sorted(code_map.items())
            )
            raise ValueError(
                f"Modification codes in {filenames[i]} do not match "
                f"{filenames[0]}.\n"
                f"  {filenames[0]}: {ref_str}\n"
                f"  {filenames[i]}: {file_str}"
            )
    return reference


def _load_transcript_data(
    h5: h5py.File,
    tx_name: str,
    min_asp: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load matrix and weights for one transcript from one HDF5 file."""
    if tx_name not in h5["transcripts"]:
        return None
    grp = h5[f"transcripts/{tx_name}"]
    matrix = grp["matrix"][:]
    weights = grp["read_weights"][:]
    if min_asp > 0.0:
        mask = weights >= min_asp
        if mask.sum() == 0:
            return None
        matrix = matrix[mask]
        weights = weights[mask]
    return matrix, weights


def _validate_tx_lengths(
    tx_name: str,
    lengths: list[int | None],
    filenames: list[str],
) -> int:
    """Validate a transcript has consistent length across files."""
    ref_length = next(ln for ln in lengths if ln is not None)
    for i, (length, fname) in enumerate(zip(lengths, filenames)):
        if length is not None and length != ref_length:
            raise ValueError(
                f"Transcript '{tx_name}' has inconsistent lengths across "
                f"input files: {ref_length} in first file vs {length} in "
                f"{fname}. All files must use the same transcriptome "
                f"reference."
            )
    return ref_length


def _pool_transcript_data(
    h5_files: list[h5py.File],
    tx_name: str,
    min_asp: float,
    label: str = "",
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Load and pool matrix/weights for a transcript across HDF5 files.

    Returns ``(matrix, weights, tx_length)`` or ``None`` if the
    transcript has no reads in any file.
    """
    matrices: list[np.ndarray] = []
    weights_list: list[np.ndarray] = []
    tx_lengths_found: list[int | None] = []

    for h5 in h5_files:
        result = _load_transcript_data(h5, tx_name, min_asp)
        if result is not None:
            m, w = result
            matrices.append(m)
            weights_list.append(w)
            tx_lengths_found.append(m.shape[1])
        else:
            tx_lengths_found.append(None)

    if not matrices:
        return None

    _validate_tx_lengths(tx_name, tx_lengths_found,
                         [f.filename for f in h5_files])

    if len(matrices) == 1:
        return matrices[0], weights_list[0], matrices[0].shape[1]
    return (
        np.vstack(matrices),
        np.concatenate(weights_list),
        matrices[0].shape[1],
    )


# ---------- per-transcript processing ----------


def process_transcript(
    tx_name: str,
    matrix_1: np.ndarray,
    weights_1: np.ndarray,
    matrix_2: np.ndarray,
    weights_2: np.ndarray,
    sites_1_for_tx: dict[tuple, dict[str, Any]],
    sites_2_for_tx: dict[tuple, dict[str, Any]],
    mod_code_map: dict[str, int],
) -> list[dict[str, Any]]:
    """Process one transcript for DMC.

    Parameters
    ----------
    tx_name : str
        Transcript name.
    matrix_1, matrix_2 : ndarray
        Pooled HDF5 matrices for condition 1 and 2.
    weights_1, weights_2 : ndarray
        Pooled read weights for condition 1 and 2.
    sites_1_for_tx, sites_2_for_tx : dict
        Site summary entries for this transcript, keyed by
        ``(tx_name, position, mod_type)``.
    mod_code_map : dict
        ``{mod_type_str: integer_code}`` from the HDF5.

    Returns
    -------
    list of dict
        One dict per matched site with all ``_OUTPUT_COLS`` fields.
    """
    # Intersection of site keys for this transcript
    common_keys = sorted(set(sites_1_for_tx.keys())
                         & set(sites_2_for_tx.keys()))
    if not common_keys:
        return []

    rows: list[dict[str, Any]] = []
    for key in common_keys:
        _tx, pos, mod_str = key
        mod_code = mod_code_map.get(mod_str)
        if mod_code is None:
            continue

        # Extract per-read data from each condition
        reads_1 = _extract_site_reads(matrix_1, weights_1, pos, mod_code)
        reads_2 = _extract_site_reads(matrix_2, weights_2, pos, mod_code)

        if reads_1 is None or reads_2 is None:
            continue

        y_1, w_1 = reads_1
        y_2, w_2 = reads_2

        if len(y_1) == 0 or len(y_2) == 0:
            continue

        # Build design: condition 1 = 0, condition 2 = 1
        y = np.concatenate([y_1, y_2])
        x = np.concatenate([
            np.zeros(len(y_1), dtype=np.float64),
            np.ones(len(y_2), dtype=np.float64),
        ])
        w = np.concatenate([w_1, w_2])

        result = weighted_logistic_test(y, x, w)

        # Site summary rows for effect sizes
        s1 = sites_1_for_tx[key]
        s2 = sites_2_for_tx[key]

        ml1 = s1.get("mod_level")
        ml2 = s2.get("mod_level")
        wml1 = s1.get("wt_mod_level")
        wml2 = s2.get("wt_mod_level")

        rows.append({
            "transcript_id": tx_name,
            "position": pos,
            "mod_type": mod_str,
            "gene_id": s1.get("gene_id"),
            "chrom": s1.get("chrom"),
            "strand": s1.get("strand"),
            "gpos": s1.get("gpos"),
            "n_modified_1": s1.get("n_modified"),
            "n_unmodified_1": s1.get("n_unmodified"),
            "n_modified_2": s2.get("n_modified"),
            "n_unmodified_2": s2.get("n_unmodified"),
            "wt_modified_1": s1.get("wt_modified"),
            "wt_unmodified_1": s1.get("wt_unmodified"),
            "wt_modified_2": s2.get("wt_modified"),
            "wt_unmodified_2": s2.get("wt_unmodified"),
            "mod_level_1": ml1,
            "mod_level_2": ml2,
            "wt_mod_level_1": wml1,
            "wt_mod_level_2": wml2,
            "delta_mod_level": (
                round(ml2 - ml1, 6)
                if ml1 is not None and ml2 is not None
                else None
            ),
            "delta_wt_mod_level": (
                round(wml2 - wml1, 6)
                if wml1 is not None and wml2 is not None
                else None
            ),
            "log2_or": result["log2_or"],
            "p_value": result["p_value"],
            "q_value": 0.0,  # filled after global BH correction
        })

    return rows


# ---------- output writers ----------


def _write_tsv(
    all_rows: list[dict[str, Any]], path: str, use_gzip: bool
) -> None:
    """Write rows as tab-separated values."""
    import gzip

    open_func = gzip.open if use_gzip else open
    mode = "wt" if use_gzip else "w"
    with open_func(path, mode, encoding="utf-8") as fh:
        fh.write(_TSV_HEADER + "\n")
        for row in all_rows:
            fh.write(
                "\t".join(
                    "NA" if row[c] is None else str(row[c])
                    for c in _OUTPUT_COLS
                )
                + "\n"
            )


def _write_parquet(all_rows: list[dict[str, Any]], path: str) -> None:
    """Write rows as a Parquet file via pyarrow."""
    if not all_rows:
        schema = pa.schema([
            ("transcript_id", pa.string()),
            ("position", pa.int32()),
            ("mod_type", pa.string()),
            ("gene_id", pa.string()),
            ("chrom", pa.string()),
            ("strand", pa.string()),
            ("gpos", pa.int32()),
            ("n_modified_1", pa.int32()),
            ("n_unmodified_1", pa.int32()),
            ("n_modified_2", pa.int32()),
            ("n_unmodified_2", pa.int32()),
            ("wt_modified_1", pa.float64()),
            ("wt_unmodified_1", pa.float64()),
            ("wt_modified_2", pa.float64()),
            ("wt_unmodified_2", pa.float64()),
            ("mod_level_1", pa.float64()),
            ("mod_level_2", pa.float64()),
            ("wt_mod_level_1", pa.float64()),
            ("wt_mod_level_2", pa.float64()),
            ("delta_mod_level", pa.float64()),
            ("delta_wt_mod_level", pa.float64()),
            ("log2_or", pa.float64()),
            ("p_value", pa.float64()),
            ("q_value", pa.float64()),
        ])
        with pq.ParquetWriter(path, schema) as w:
            w.write_table(
                pa.table({
                    k: pa.array([], type=schema.field(k).type)
                    for k in schema.names
                })
            )
        return

    arrays: dict[str, pa.Array] = {}
    for col in _OUTPUT_COLS:
        values = [r[col] for r in all_rows]
        if col in ("transcript_id", "mod_type", "gene_id"):
            arrays[col] = pa.array(values)
        elif col in ("chrom", "strand"):
            arrays[col] = pa.array(values, type=pa.string())
        elif col == "position":
            arrays[col] = pa.array(values, type=pa.int32())
        elif col == "gpos":
            arrays[col] = pa.array(values, type=pa.int32())
        elif col.startswith("n_"):
            arrays[col] = pa.array(values, type=pa.int32())
        else:
            arrays[col] = pa.array(values, type=pa.float64())
    pq.write_table(pa.table(arrays), path)


# ---------- main ----------


def main(args: argparse.Namespace | None = None) -> None:
    """Differential modification calling between two conditions.

    Reads HDF5 files and site summaries for two conditions, matches
    sites by ``(transcript_id, position, mod_type)``, fits a weighted
    logistic regression per site, and writes results with global BH
    FDR correction.
    """
    if args is None:
        args = parse_args()

    # ---- 1. Read site summaries ----
    if args.verbose:
        print("[mod_dmc] Reading site summaries...", file=sys.stderr)
    sites_1 = read_site_summary_full(args.sites_1)
    sites_2 = read_site_summary_full(args.sites_2)

    if args.verbose:
        print(
            f"[mod_dmc] Condition 1: {len(sites_1)} sites",
            file=sys.stderr,
        )
        print(
            f"[mod_dmc] Condition 2: {len(sites_2)} sites",
            file=sys.stderr,
        )

    # ---- 2. Open all HDF5 files ----
    with ExitStack() as stack:
        h5_1 = [
            stack.enter_context(h5py.File(f, "r")) for f in args.h5_1
        ]
        h5_2 = [
            stack.enter_context(h5py.File(f, "r")) for f in args.h5_2
        ]

        if args.verbose:
            print(
                f"[mod_dmc] Opened {len(h5_1)}+{len(h5_2)} HDF5 files",
                file=sys.stderr,
            )

        # ---- 3. Validate modification codes ----
        all_h5 = h5_1 + h5_2
        all_paths = list(args.h5_1) + list(args.h5_2)
        all_mod_maps = [_read_mod_codes(h5) for h5 in all_h5]
        try:
            mod_code_map = _validate_mod_codes(all_mod_maps, all_paths)
        except ValueError as exc:
            print(f"[mod_dmc] Error: {exc}", file=sys.stderr)
            sys.exit(1)

        if args.verbose:
            print(
                f"[mod_dmc] {len(mod_code_map)} modification types: "
                f"{sorted(mod_code_map.keys())}",
                file=sys.stderr,
            )

        # ---- 4. Build transcript sets ----
        h5_1_tx = set.union(*[set(h["transcripts"].keys()) for h in h5_1])
        h5_2_tx = set.union(*[set(h["transcripts"].keys()) for h in h5_2])
        sites_1_tx = {k[0] for k in sites_1}
        sites_2_tx = {k[0] for k in sites_2}

        common_tx = sorted(
            h5_1_tx & h5_2_tx & sites_1_tx & sites_2_tx
        )

        if args.transcripts is not None:
            requested = set(args.transcripts)
            common_tx = sorted(tx for tx in common_tx if tx in requested)

        if args.verbose:
            print(
                f"[mod_dmc] {len(common_tx)} transcripts in common "
                f"across all inputs",
                file=sys.stderr,
            )

        # ---- 5. Process each transcript ----
        all_rows: list[dict[str, Any]] = []
        processed = 0

        for tx_name in common_tx:
            # Pool reads within each condition
            pooled_1 = _pool_transcript_data(
                h5_1, tx_name, args.min_asp, "cond1"
            )
            pooled_2 = _pool_transcript_data(
                h5_2, tx_name, args.min_asp, "cond2"
            )

            if pooled_1 is None or pooled_2 is None:
                processed += 1
                continue

            matrix_1, weights_1, _len1 = pooled_1
            matrix_2, weights_2, _len2 = pooled_2

            if args.verbose and (
                len(
                    [h for h in h5_1
                     if tx_name in h["transcripts"]]
                ) > 1
                or len(
                    [h for h in h5_2
                     if tx_name in h["transcripts"]]
                ) > 1
            ):
                print(
                    f"[mod_dmc] {tx_name}: cond1={matrix_1.shape[0]} "
                    f"reads, cond2={matrix_2.shape[0]} reads",
                    file=sys.stderr,
                )

            # Filter site keys to this transcript
            st1 = {
                k: v for k, v in sites_1.items() if k[0] == tx_name
            }
            st2 = {
                k: v for k, v in sites_2.items() if k[0] == tx_name
            }

            tx_rows = process_transcript(
                tx_name, matrix_1, weights_1,
                matrix_2, weights_2,
                st1, st2, mod_code_map,
            )
            all_rows.extend(tx_rows)
            processed += 1

            if args.verbose and processed % 1000 == 0:
                print(
                    f"[mod_dmc] Processed {processed}/{len(common_tx)} "
                    f"transcripts...",
                    file=sys.stderr,
                )

    # ---- 6. Global BH FDR correction ----
    if all_rows:
        p_values = [r["p_value"] for r in all_rows]
        q_values = bh_fdr(p_values)
        for r, qv in zip(all_rows, q_values):
            r["q_value"] = round(qv, 6)

    if args.verbose:
        print(
            f"[mod_dmc] Total tests: {len(all_rows)}", file=sys.stderr
        )

    # ---- 7. Write output ----
    if args.format == "tsv":
        _write_tsv(all_rows, args.output, args.gzip)
    else:
        _write_parquet(all_rows, args.output)

    if args.verbose:
        print(
            f"[mod_dmc] Done. Output written to {args.output}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
