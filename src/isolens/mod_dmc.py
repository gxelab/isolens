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
from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from isolens._hdf5_helpers import (
        extract_site_reads,
        load_transcript_data,
        nullable_float,
        nullable_str,
        read_mod_codes,
        validate_mod_codes,
        validate_tx_lengths,
    )
    from isolens._io import write_parquet, write_tsv
    from isolens._stats import bh_fdr, weighted_logistic_test
except ImportError:
    from _io import write_parquet, write_tsv  # type: ignore[no-redef]

    from _hdf5_helpers import (  # type: ignore[no-redef]
        extract_site_reads,
        load_transcript_data,
        nullable_float,
        nullable_str,
        read_mod_codes,
        validate_mod_codes,
        validate_tx_lengths,
    )
    from _stats import bh_fdr, weighted_logistic_test  # type: ignore[no-redef]

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


# ---------- compact site data ----------


@dataclass(slots=True)
class TxSiteData:
    """Memory-efficient site data for a single transcript.

    Stores per-site values as numpy arrays instead of Python dicts
    to avoid the ~608-byte-per-row dict overhead.  All arrays have
    the same length (``n_sites``).
    """

    positions: np.ndarray  # int32  (n_sites,)
    mod_types: np.ndarray  # object (n_sites,) — short shared strings
    n_modified: np.ndarray  # int32
    wt_modified: np.ndarray  # float64
    n_unmodified: np.ndarray  # int32
    wt_unmodified: np.ndarray  # float64
    mod_level: np.ndarray  # float64
    wt_mod_level: np.ndarray  # float64
    gene_id: np.ndarray  # object (n_sites,) — str or None
    chrom: np.ndarray  # object
    strand: np.ndarray  # object
    gpos: np.ndarray  # float64 (NaN for NA / missing)

    @property
    def n_sites(self) -> int:
        return len(self.positions)


def _build_site_index(data: TxSiteData) -> dict[tuple[int, str], int]:
    """Build ``(position, mod_type) → array_index`` for fast intersection."""
    return {
        (int(data.positions[i]), str(data.mod_types[i])): i for i in range(data.n_sites)
    }


# ---------- CLI ----------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for mod_dmc."""
    parser = argparse.ArgumentParser(
        description="mod_dmc: Differential modification calling between two conditions"
    )
    parser.add_argument(
        "-i1",
        "--h5-1",
        required=True,
        nargs="+",
        metavar="H5",
        help="Input HDF5 file(s) for condition 1.  When multiple files "
        "are provided, reads for the same transcript are pooled.",
    )
    parser.add_argument(
        "-i2",
        "--h5-2",
        required=True,
        nargs="+",
        metavar="H5",
        help="Input HDF5 file(s) for condition 2.  When multiple files "
        "are provided, reads for the same transcript are pooled.",
    )
    parser.add_argument(
        "-s1",
        "--sites-1",
        required=True,
        metavar="FILE",
        help="Pooled site summary for condition 1 (Parquet or TSV/TSV.GZ "
        "from mod_sites)",
    )
    parser.add_argument(
        "-s2",
        "--sites-2",
        required=True,
        metavar="FILE",
        help="Pooled site summary for condition 2 (Parquet or TSV/TSV.GZ "
        "from mod_sites)",
    )
    parser.add_argument("-o", "--output", required=True, help="Output file path")
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


def _read_sites_parquet_grouped(path: str) -> dict[str, TxSiteData]:
    """Read a Parquet site summary grouped by transcript.

    Builds a ``transcript → TxSiteData`` mapping without creating
    per-row Python dicts — data moves directly from PyArrow columns
    into numpy arrays.
    """
    table = pq.read_table(path, columns=_SITE_COLS)

    # --- pass 1: find transcript boundaries ---
    tx_col = table.column("transcript_id")
    tx_ranges: dict[str, tuple[int, int]] = {}
    n_rows = len(table)
    if n_rows == 0:
        return {}
    cur_tx = tx_col[0].as_py()
    start = 0
    for i in range(1, n_rows):
        tx = tx_col[i].as_py()
        if tx != cur_tx:
            tx_ranges[cur_tx] = (start, i)
            cur_tx = tx
            start = i
    tx_ranges[cur_tx] = (start, n_rows)

    # --- pass 2: extract numpy arrays per transcript ---
    result: dict[str, TxSiteData] = {}
    for tx, (lo, hi) in tx_ranges.items():
        result[tx] = TxSiteData(
            positions=table.column("position")[lo:hi].to_numpy().astype(np.int32),
            mod_types=table.column("mod_type")[lo:hi].to_numpy(),
            n_modified=table.column("n_modified")[lo:hi].to_numpy().astype(np.int32),
            wt_modified=table.column("wt_modified")[lo:hi]
            .to_numpy()
            .astype(np.float64),
            n_unmodified=table.column("n_unmodified")[lo:hi]
            .to_numpy()
            .astype(np.int32),
            wt_unmodified=table.column("wt_unmodified")[lo:hi]
            .to_numpy()
            .astype(np.float64),
            mod_level=table.column("mod_level")[lo:hi].to_numpy().astype(np.float64),
            wt_mod_level=table.column("wt_mod_level")[lo:hi]
            .to_numpy()
            .astype(np.float64),
            gene_id=table.column("gene_id")[lo:hi].to_numpy(),
            chrom=table.column("chrom")[lo:hi].to_numpy(),
            strand=table.column("strand")[lo:hi].to_numpy(),
            gpos=_col_to_float64_nullable(table.column("gpos")[lo:hi]),
        )
    return result


def _col_to_float64_nullable(col: pa.ChunkedArray) -> np.ndarray:
    """Convert a PyArrow int column to float64 with NaN for nulls."""
    arr = col.to_numpy()
    if hasattr(col, "null_count") and col.null_count > 0:
        out = np.full(len(arr), np.nan, dtype=np.float64)
        mask = arr != np.array(None)
        # PyArrow nullable int → numpy object array with None for nulls
        if arr.dtype == object:
            valid = np.array([v is not None for v in arr])
            out[valid] = arr[valid].astype(np.float64)
        else:
            out[mask] = arr[mask].astype(np.float64)
        return out
    if arr.dtype == object:
        return np.array(
            [np.nan if v is None else np.float64(v) for v in arr], dtype=np.float64
        )
    return arr.astype(np.float64)


def _read_sites_tsv_grouped(path: str) -> dict[str, TxSiteData]:
    """Read a TSV/TSV.GZ site summary grouped by transcript.

    Streams the file one line at a time, buffers rows for the current
    transcript in plain Python lists, then converts to numpy arrays when
    the transcript changes.  Peak memory is one transcript's worth of
    raw values plus the final ``TxSiteData``.
    """
    import gzip

    open_func = gzip.open if path.endswith(".gz") else open
    mode = "rt" if path.endswith(".gz") else "r"

    tx_sites: dict[str, TxSiteData] = {}

    with open_func(path, mode, encoding="utf-8") as fh:
        header = fh.readline().strip().split("\t")
        col_idx = {c: header.index(c) for c in _SITE_COLS if c in header}

        # --- column buffers for the current transcript ---
        cur_tx: str | None = None
        cur_positions: list[int] = []
        cur_mod_types: list[str] = []
        cur_n_modified: list[int] = []
        cur_wt_modified: list[float] = []
        cur_n_unmodified: list[int] = []
        cur_wt_unmodified: list[float] = []
        cur_mod_level: list[float] = []
        cur_wt_mod_level: list[float] = []
        cur_gene_id: list[str | None] = []
        cur_chrom: list[str | None] = []
        cur_strand: list[str | None] = []
        cur_gpos: list[float] = []  # NaN for NA

        def _flush() -> None:
            """Convert buffered lists to TxSiteData and store."""
            nonlocal cur_tx
            if cur_tx is not None and cur_positions:
                tx_sites[cur_tx] = TxSiteData(
                    positions=np.array(cur_positions, dtype=np.int32),
                    mod_types=np.array(cur_mod_types, dtype=object),
                    n_modified=np.array(cur_n_modified, dtype=np.int32),
                    wt_modified=np.array(cur_wt_modified, dtype=np.float64),
                    n_unmodified=np.array(cur_n_unmodified, dtype=np.int32),
                    wt_unmodified=np.array(cur_wt_unmodified, dtype=np.float64),
                    mod_level=np.array(cur_mod_level, dtype=np.float64),
                    wt_mod_level=np.array(cur_wt_mod_level, dtype=np.float64),
                    gene_id=np.array(cur_gene_id, dtype=object),
                    chrom=np.array(cur_chrom, dtype=object),
                    strand=np.array(cur_strand, dtype=object),
                    gpos=np.array(cur_gpos, dtype=np.float64),
                )
            cur_positions.clear()
            cur_mod_types.clear()
            cur_n_modified.clear()
            cur_wt_modified.clear()
            cur_n_unmodified.clear()
            cur_wt_unmodified.clear()
            cur_mod_level.clear()
            cur_wt_mod_level.clear()
            cur_gene_id.clear()
            cur_chrom.clear()
            cur_strand.clear()
            cur_gpos.clear()

        def _parse_str(col: str) -> str | None:
            i = col_idx.get(col)
            if i is None:
                return None
            val = parts[i]
            return None if val == "NA" else val

        def _parse_int(col: str) -> int:
            return int(parts[col_idx[col]])

        def _parse_float(col: str) -> float:
            return float(parts[col_idx[col]])

        def _parse_gpos(col: str) -> float:
            i = col_idx.get(col)
            if i is None:
                return np.nan
            val = parts[i]
            return np.float64(val) if val != "NA" else np.nan

        for line in fh:
            parts = line.strip().split("\t")
            if not parts:
                continue

            tx = parts[col_idx["transcript_id"]]

            if tx != cur_tx:
                _flush()
                cur_tx = tx

            cur_positions.append(_parse_int("position"))
            cur_mod_types.append(parts[col_idx["mod_type"]])
            cur_n_modified.append(_parse_int("n_modified"))
            cur_wt_modified.append(_parse_float("wt_modified"))
            cur_n_unmodified.append(_parse_int("n_unmodified"))
            cur_wt_unmodified.append(_parse_float("wt_unmodified"))
            cur_mod_level.append(_parse_float("mod_level"))
            cur_wt_mod_level.append(_parse_float("wt_mod_level"))
            cur_gene_id.append(_parse_str("gene_id"))
            cur_chrom.append(_parse_str("chrom"))
            cur_strand.append(_parse_str("strand"))
            cur_gpos.append(_parse_gpos("gpos"))

        _flush()  # last transcript

    return tx_sites


def read_site_summary_full(path: str) -> dict[str, TxSiteData]:
    """Read a modification site summary file (Parquet or TSV/TSV.GZ).

    Returns a dict mapping ``transcript_id`` to ``TxSiteData``,
    a compact numpy-backed container with all site-summary columns.
    """
    if path.endswith(".parquet"):
        return _read_sites_parquet_grouped(path)
    return _read_sites_tsv_grouped(path)


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
        result = load_transcript_data(h5, tx_name, min_asp)
        if result is not None:
            m, w = result
            matrices.append(m)
            weights_list.append(w)
            tx_lengths_found.append(m.shape[1])
        else:
            tx_lengths_found.append(None)

    if not matrices:
        return None

    validate_tx_lengths(tx_name, tx_lengths_found, [f.filename for f in h5_files])

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
    data_1: TxSiteData,
    data_2: TxSiteData,
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
    data_1, data_2 : TxSiteData
        Compact site data for this transcript in each condition.
    mod_code_map : dict
        ``{mod_type_str: integer_code}`` from the HDF5.

    Returns
    -------
    list of dict
        One dict per matched site with all ``_OUTPUT_COLS`` fields.
    """
    # Build temporary per-transcript lookup dicts for intersection
    idx1 = _build_site_index(data_1)
    idx2 = _build_site_index(data_2)
    common_keys = sorted(set(idx1.keys()) & set(idx2.keys()))
    if not common_keys:
        return []

    rows: list[dict[str, Any]] = []
    for pos, mod_str in common_keys:
        i1 = idx1[(pos, mod_str)]
        i2 = idx2[(pos, mod_str)]

        mod_code = mod_code_map.get(mod_str)
        if mod_code is None:
            continue

        # Extract per-read data from each condition
        reads_1 = extract_site_reads(matrix_1, weights_1, pos, mod_code)
        reads_2 = extract_site_reads(matrix_2, weights_2, pos, mod_code)

        if reads_1 is None or reads_2 is None:
            continue

        y_1, w_1 = reads_1
        y_2, w_2 = reads_2

        if len(y_1) == 0 or len(y_2) == 0:
            continue

        # Build design: condition 1 = 0, condition 2 = 1
        y = np.concatenate([y_1, y_2])
        x = np.concatenate(
            [
                np.zeros(len(y_1), dtype=np.float64),
                np.ones(len(y_2), dtype=np.float64),
            ]
        )
        w = np.concatenate([w_1, w_2])

        result = weighted_logistic_test(y, x, w)

        # Site summary values from numpy arrays
        ml1 = nullable_float(data_1.mod_level[i1])
        ml2 = nullable_float(data_2.mod_level[i2])
        wml1 = nullable_float(data_1.wt_mod_level[i1])
        wml2 = nullable_float(data_2.wt_mod_level[i2])
        gpos_val = data_1.gpos[i1]
        gene_id = nullable_str(data_1.gene_id[i1])
        chrom = nullable_str(data_1.chrom[i1])
        strand = nullable_str(data_1.strand[i1])

        rows.append(
            {
                "transcript_id": tx_name,
                "position": pos,
                "mod_type": mod_str,
                "gene_id": gene_id,
                "chrom": chrom,
                "strand": strand,
                "gpos": None if np.isnan(gpos_val) else int(gpos_val),
                "n_modified_1": int(data_1.n_modified[i1]),
                "n_unmodified_1": int(data_1.n_unmodified[i1]),
                "n_modified_2": int(data_2.n_modified[i2]),
                "n_unmodified_2": int(data_2.n_unmodified[i2]),
                "wt_modified_1": float(data_1.wt_modified[i1]),
                "wt_unmodified_1": float(data_1.wt_unmodified[i1]),
                "wt_modified_2": float(data_2.wt_modified[i2]),
                "wt_unmodified_2": float(data_2.wt_unmodified[i2]),
                "mod_level_1": ml1,
                "mod_level_2": ml2,
                "wt_mod_level_1": wml1,
                "wt_mod_level_2": wml2,
                "delta_mod_level": (
                    round(ml2 - ml1, 6) if ml1 is not None and ml2 is not None else None
                ),
                "delta_wt_mod_level": (
                    round(wml2 - wml1, 6)
                    if wml1 is not None and wml2 is not None
                    else None
                ),
                "log2_or": result["log2_or"],
                "p_value": result["p_value"],
                "q_value": 0.0,  # filled after global BH correction
            }
        )

    return rows


_DMC_SCHEMA = pa.schema(
    [
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
    ]
)


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
        n_sites_1 = sum(d.n_sites for d in sites_1.values())
        n_sites_2 = sum(d.n_sites for d in sites_2.values())
        print(
            f"[mod_dmc] Condition 1: {len(sites_1)} transcripts, {n_sites_1} sites",
            file=sys.stderr,
        )
        print(
            f"[mod_dmc] Condition 2: {len(sites_2)} transcripts, {n_sites_2} sites",
            file=sys.stderr,
        )

    # ---- 2. Open all HDF5 files ----
    with ExitStack() as stack:
        h5_1 = [stack.enter_context(h5py.File(f, "r")) for f in args.h5_1]
        h5_2 = [stack.enter_context(h5py.File(f, "r")) for f in args.h5_2]

        if args.verbose:
            print(
                f"[mod_dmc] Opened {len(h5_1)}+{len(h5_2)} HDF5 files",
                file=sys.stderr,
            )

        # ---- 3. Validate modification codes ----
        all_h5 = h5_1 + h5_2
        all_paths = list(args.h5_1) + list(args.h5_2)
        all_mod_maps = [read_mod_codes(h5) for h5 in all_h5]
        try:
            mod_code_map = validate_mod_codes(all_mod_maps, all_paths)
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
        sites_1_tx = set(sites_1.keys())
        sites_2_tx = set(sites_2.keys())

        common_tx = sorted(h5_1_tx & h5_2_tx & sites_1_tx & sites_2_tx)

        if args.transcripts is not None:
            requested = set(args.transcripts)
            common_tx = sorted(tx for tx in common_tx if tx in requested)

        if args.verbose:
            print(
                f"[mod_dmc] {len(common_tx)} transcripts in common across all inputs",
                file=sys.stderr,
            )

        # ---- 5. Process each transcript ----
        all_rows: list[dict[str, Any]] = []
        processed = 0

        for tx_name in common_tx:
            # Pool reads within each condition
            pooled_1 = _pool_transcript_data(h5_1, tx_name, args.min_asp, "cond1")
            pooled_2 = _pool_transcript_data(h5_2, tx_name, args.min_asp, "cond2")

            if pooled_1 is None or pooled_2 is None:
                processed += 1
                continue

            matrix_1, weights_1, _len1 = pooled_1
            matrix_2, weights_2, _len2 = pooled_2

            if args.verbose and (
                len([h for h in h5_1 if tx_name in h["transcripts"]]) > 1
                or len([h for h in h5_2 if tx_name in h["transcripts"]]) > 1
            ):
                print(
                    f"[mod_dmc] {tx_name}: cond1={matrix_1.shape[0]} "
                    f"reads, cond2={matrix_2.shape[0]} reads",
                    file=sys.stderr,
                )

            # Look up sites for this transcript (O(1) dict access)
            data_1 = sites_1.get(tx_name)
            data_2 = sites_2.get(tx_name)
            if data_1 is None or data_2 is None:
                processed += 1
                continue

            tx_rows = process_transcript(
                tx_name,
                matrix_1,
                weights_1,
                matrix_2,
                weights_2,
                data_1,
                data_2,
                mod_code_map,
            )
            all_rows.extend(tx_rows)
            processed += 1

            if args.verbose and processed % 1000 == 0:
                print(
                    f"[mod_dmc] Processed {processed}/{len(common_tx)} transcripts...",
                    file=sys.stderr,
                )

    # ---- 6. Global BH FDR correction ----
    if all_rows:
        p_values = [r["p_value"] for r in all_rows]
        q_values = bh_fdr(p_values)
        for r, qv in zip(all_rows, q_values):
            r["q_value"] = qv

    if args.verbose:
        print(f"[mod_dmc] Total tests: {len(all_rows)}", file=sys.stderr)

    # ---- 7. Write output ----
    if args.format == "tsv":
        write_tsv(all_rows, args.output, _TSV_HEADER, _OUTPUT_COLS, args.gzip)
    else:
        write_parquet(all_rows, args.output, _DMC_SCHEMA, _OUTPUT_COLS)

    if args.verbose:
        print(
            f"[mod_dmc] Done. Output written to {args.output}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
