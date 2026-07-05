#!/usr/bin/env python3
"""mod_corr: Pairwise modification site correlation analysis.

Identifies cooperative or antagonistic relationships between modification
sites within the same transcript — both within a single modification type
and across different modification types.  Reads the HDF5 from ``mod_scan.py``
and the site summary from ``mod_sites.py``.

Each pair of sites is summarised with a 2×2 contingency table counting
reads where each site is modified (1) or not (0):

    ===== ==== ====
    Count site1 site2
    ===== ==== ====
    n11   1     1
    n10   1     0
    n01   0     1
    n00   0     0
    ===== ==== ====

Both unweighted (raw counts) and weighted (sum of assignment probabilities)
variants are computed.  Association metrics include the Phi coefficient,
odds ratio (with Haldane-Anscombe correction for zero cells), p-value
(Fisher's exact test), BH FDR q-value, and mutual information.

See notebooks/01_mod.md for the full specification.
"""

import argparse
import os
import sys
from collections import defaultdict
from contextlib import ExitStack
from typing import Any

import h5py
import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

matplotlib.use("Agg")  # non-interactive backend for PDF output
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.stats import t as t_dist

try:
    from isolens._hdf5_helpers import (
        load_transcript_data,
        read_mod_codes,
        validate_mod_codes,
        validate_tx_lengths,
    )
    from isolens._io import write_parquet, write_tsv
    from isolens._stats import bh_fdr
    from isolens.mod_scan import (
        CODE_DELETION,
        CODE_FAIL,
        CODE_MISMATCH,
        CODE_UNCOVERED,
    )
except ImportError:
    from _io import write_parquet, write_tsv  # type: ignore[no-redef]

    from _hdf5_helpers import (  # type: ignore[no-redef]
        load_transcript_data,
        read_mod_codes,
        validate_mod_codes,
        validate_tx_lengths,
    )
    from _stats import bh_fdr  # type: ignore[no-redef]
    from mod_scan import (  # type: ignore[no-redef]
        CODE_DELETION,
        CODE_FAIL,
        CODE_MISMATCH,
        CODE_UNCOVERED,
    )

# ---------- constants ----------

_HALDANE_ANSCOMBE = 0.5  # Haldane-Anscombe correction for odds ratio zeros

_OUTPUT_COLS = [
    "transcript_id",
    "site1",
    "site2",
    "mod_type1",
    "mod_type2",
    "wt_mod_level1",
    "wt_mod_level2",
    "n11",
    "n10",
    "n01",
    "n00",
    "w11",
    "w10",
    "w01",
    "w00",
    "corr",
    "pvalue",
    "qvalue",
    "wcorr",
    "wpvalue",
    "wqvalue",
    "mi",
    "wmi",
    "or",
    "wor",
]

_TSV_HEADER = "\t".join(_OUTPUT_COLS)

_METRIC_LABELS: dict[str, str] = {
    "corr": "Pearson r",
    "wcorr": "Weighted Pearson r",
    "mi": "Mutual Information (bits)",
    "wmi": "Weighted Mutual Information (bits)",
    "or": "Log$_2$ Odds Ratio",
    "wor": "Weighted Log$_2$ Odds Ratio",
}

# SAM modification codes → human-readable names (from notebooks/01_mod.md).
# 2'-O-methyl variants on C/A/G/U are combined under a single "2Ome" label.
_SAM_TO_HUMAN = {
    "m": "m5C",
    "a": "m6A",
    "17596": "inosine",
    "17802": "pseU",
    "19227": "2Ome",
    "19228": "2Ome",
    "19229": "2Ome",
    "69426": "2Ome",
}

# Consistent, publication-friendly colours per modification type (ColorBrewer Set1).
_MOD_COLORS = {
    "m5C": "#377EB8",  # blue
    "m6A": "#E41A1C",  # red
    "inosine": "#4DAF4A",  # green
    "pseU": "#FF7F00",  # orange
    "2Ome": "#984EA3",  # purple
}


# ---------- helpers ----------


def _sam_to_human(sam_code: str) -> str:
    """Convert a SAM modification code to a human-readable name."""
    return _SAM_TO_HUMAN.get(sam_code, sam_code)


def _nice_tick_step(rna_length: int) -> int:
    """Choose a regular tick interval for the transcript axis.

    Returns a step size that yields ~3-7 ticks across the transcript.
    """
    for step in (100, 200, 500, 1000, 2000, 5000, 10000):
        if rna_length / step <= 6:
            return step
    return 10000


# ---------- CLI ----------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for mod_corr."""
    parser = argparse.ArgumentParser(
        description="mod_corr: Pairwise modification site correlation analysis"
    )
    parser.add_argument(
        "-i",
        "--h5",
        required=True,
        nargs="+",
        metavar="H5",
        help="Input HDF5 file(s) from mod_scan. When multiple files "
        "are provided, reads for the same transcript are pooled "
        "across all files before computing pairwise correlations.",
    )
    parser.add_argument(
        "-s",
        "--sites",
        required=True,
        help="Input modification sites from mod_sites (Parquet or TSV/TSV.GZ)",
    )
    parser.add_argument("-o", "--output", required=True, help="Output file path")
    parser.add_argument(
        "-m",
        "--min-mod-reads",
        type=int,
        default=2,
        help="Minimum number of modified reads for a site to be "
        "considered [default: 2]",
    )
    parser.add_argument(
        "-l",
        "--min-mod-level",
        type=float,
        default=0.05,
        help="Minimum modification level for a site to be considered [default: 0.05]",
    )
    parser.add_argument(
        "-c",
        "--min-coverage",
        type=int,
        default=10,
        help="Minimum total depth for a site to be considered [default: 10]",
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
        "-d",
        "--plot-dir",
        metavar="DIR",
        default=None,
        help="Generate rotated triangular heatmap PDFs per transcript "
        "in the given output directory",
    )
    parser.add_argument(
        "-t",
        "--metric",
        choices=["corr", "wcorr", "mi", "wmi", "or", "wor"],
        default="wcorr",
        help="Association statistic to visualize in heatmaps [default: wcorr]",
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
        "-v", "--verbose", action="store_true", help="Print progress to stderr"
    )
    return parser.parse_args()


# ---------- site-summary reader ----------


def read_site_summary(path: str) -> dict[str, dict[str, list[dict[str, int | float]]]]:
    """Read a modification site summary file (Parquet or TSV).

    Args:
        path: Path to a Parquet or TSV/TSV.GZ file from ``mod_sites``.

    Returns:
        Nested dict ``{tx_name: {mod_type: [{"pos": int, "n_mod": int,
        "mod_level": float, "depth": int}, ...]}}``.
    """
    if path.endswith(".parquet"):
        return _read_sites_parquet(path)
    else:
        return _read_sites_tsv(path)


def _read_sites_parquet(
    path: str,
) -> dict[str, dict[str, list[dict[str, int | float]]]]:
    table = pq.read_table(path)
    sites: dict[str, dict[str, list[dict[str, int | float]]]] = {}
    for i in range(len(table)):
        tx = table.column("transcript_id")[i].as_py()
        pos = table.column("position")[i].as_py()
        mod = table.column("mod_type")[i].as_py()
        n_mod = table.column("n_modified")[i].as_py()
        n_unmod = table.column("n_unmodified")[i].as_py()
        n_mismatch = table.column("n_mismatch")[i].as_py()
        n_del = table.column("n_deletion")[i].as_py()
        n_failed = table.column("n_failed")[i].as_py()
        mod_level = table.column("mod_level")[i].as_py()
        wt_mod_level = table.column("wt_mod_level")[i].as_py()
        depth = n_mod + n_unmod + n_mismatch + n_del + n_failed
        sites.setdefault(tx, {}).setdefault(mod, []).append(
            {
                "pos": pos,
                "n_mod": n_mod,
                "mod_level": mod_level,
                "wt_mod_level": wt_mod_level,
                "depth": depth,
            }
        )
    return sites


def _read_sites_tsv(
    path: str,
) -> dict[str, dict[str, list[dict[str, int | float]]]]:
    import gzip

    open_func = gzip.open if path.endswith(".gz") else open
    mode = "rt" if path.endswith(".gz") else "r"
    sites: dict[str, dict[str, list[dict[str, int | float]]]] = {}
    with open_func(path, mode, encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        tx_col = header.index("transcript_id")
        pos_col = header.index("position")
        mod_col = header.index("mod_type")
        nmod_col = header.index("n_modified")
        nunmod_col = header.index("n_unmodified")
        nmis_col = header.index("n_mismatch")
        ndel_col = header.index("n_deletion")
        nfail_col = header.index("n_failed")
        ml_col = header.index("mod_level")
        wt_ml_col = header.index("wt_mod_level")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) <= max(
                tx_col,
                pos_col,
                mod_col,
                nmod_col,
                nunmod_col,
                nmis_col,
                ndel_col,
                nfail_col,
                ml_col,
                wt_ml_col,
            ):
                continue
            tx = parts[tx_col]
            n_mod = int(parts[nmod_col])
            n_unmod = int(parts[nunmod_col])
            n_mismatch = int(parts[nmis_col])
            n_del = int(parts[ndel_col])
            n_failed = int(parts[nfail_col])
            mod_level = float(parts[ml_col])
            wt_mod_level = float(parts[wt_ml_col])
            depth = n_mod + n_unmod + n_mismatch + n_del + n_failed
            sites.setdefault(tx, {}).setdefault(parts[mod_col], []).append(
                {
                    "pos": int(parts[pos_col]),
                    "n_mod": n_mod,
                    "mod_level": mod_level,
                    "wt_mod_level": wt_mod_level,
                    "depth": depth,
                }
            )
    return sites


# ---------- statistics ----------


def _pearson_r_from_counts(n11: float, n10: float, n01: float, n00: float) -> float:
    """Compute Pearson correlation from a 2×2 contingency table.

    For binary (0/1) variables, Pearson's r is mathematically equivalent
    to the Phi coefficient:

        r = (n11*n00 - n10*n01) / sqrt(n1• * n0• * n•1 * n•0)

    Returns 0.0 if any marginal total is zero.
    """
    n1x, n0x = n11 + n10, n01 + n00
    nx1, nx0 = n11 + n01, n10 + n00
    denom = n1x * n0x * nx1 * nx0
    if denom <= 0:
        return 0.0
    return (n11 * n00 - n10 * n01) / np.sqrt(denom)


def _pearson_pvalue(r: float, n: float) -> float:
    """Compute two-sided p-value for Pearson's r from the t-distribution.

    ``t = r * sqrt((n-2) / (1 - r²))``, with *n* - 2 degrees of freedom.

    Returns 0.0 when |r| ≈ 1 and 1.0 when n ≤ 2.
    """
    if n <= 2:
        return 1.0
    r_abs = abs(r)
    if r_abs >= 1.0 - 1e-15:
        return 0.0
    denom = 1.0 - r * r
    if denom <= 0.0:
        return 0.0 if r_abs > 1e-15 else 1.0
    t_stat = r_abs * np.sqrt((n - 2.0) / denom)
    return float(2.0 * t_dist.sf(t_stat, n - 2))


def _effective_sample_size(weights: np.ndarray) -> float:
    """Compute Kish's effective sample size from a weight vector.

    ``n_eff = (Σ w)² / Σ(w²)``
    """
    w_sum = weights.sum()
    if w_sum <= 0:
        return 0.0
    return float(w_sum * w_sum / (weights * weights).sum())


def _weighted_pearson_r(
    x: np.ndarray, y: np.ndarray, w: np.ndarray
) -> tuple[float, float]:
    """Compute weighted Pearson correlation between two binary vectors.

    Uses weighted means, covariance, and variances.  Returns ``(r, n_eff)``
    where *n_eff* is the effective sample size for p-value computation.

    Returns ``(0.0, 0.0)`` when the total weight is zero or either weighted
    variance is zero.
    """
    w_sum = w.sum()
    if w_sum <= 0:
        return 0.0, 0.0
    x_f = x.astype(np.float64)
    y_f = y.astype(np.float64)
    mx = (w * x_f).sum() / w_sum
    my = (w * y_f).sum() / w_sum
    xc = x_f - mx
    yc = y_f - my
    cov_w = (w * xc * yc).sum() / w_sum
    var_x = (w * xc * xc).sum() / w_sum
    var_y = (w * yc * yc).sum() / w_sum
    denom_sqrt = np.sqrt(var_x * var_y)
    if denom_sqrt <= 0:
        return 0.0, _effective_sample_size(w)
    r = cov_w / denom_sqrt
    # Clamp to [-1, 1] to guard against floating-point drift
    r = max(-1.0, min(1.0, r))
    return r, _effective_sample_size(w)


def _odds_ratio(n11: float, n10: float, n01: float, n00: float) -> float:
    """Compute the odds ratio with Haldane-Anscombe correction.

    OR = ((n11 + 0.5)*(n00 + 0.5)) / ((n10 + 0.5)*(n01 + 0.5))

    The correction prevents division by zero and infinite estimates.
    """
    a, b = n11 + _HALDANE_ANSCOMBE, n10 + _HALDANE_ANSCOMBE
    c, d = n01 + _HALDANE_ANSCOMBE, n00 + _HALDANE_ANSCOMBE
    return (a * d) / (b * c)


def _mutual_information(n11: float, n10: float, n01: float, n00: float) -> float:
    """Compute mutual information from a 2×2 contingency table.

    MI = Σ p_ij * log2(p_ij / (p_i• * p_•j))
    """
    total = n11 + n10 + n01 + n00
    if total <= 0:
        return 0.0
    p11, p10, p01, p00 = n11 / total, n10 / total, n01 / total, n00 / total
    p1x, p0x = p11 + p10, p01 + p00
    px1, px0 = p11 + p01, p10 + p00
    mi = 0.0
    for pj, pr, pc in [
        (p11, p1x, px1),
        (p10, p1x, px0),
        (p01, p0x, px1),
        (p00, p0x, px0),
    ]:
        if pj > 0 and pr > 0 and pc > 0:
            mi += pj * np.log2(pj / (pr * pc))
    return mi


# ---------- per-transcript processing ----------


def process_transcript(
    tx_name: str,
    matrix: np.ndarray,
    weights: np.ndarray,
    sites_by_mod: dict[str, list[dict[str, int | float]]],
    mod_code_map: dict[str, int],
    min_mod_reads: int,
    min_asp: float = 0.0,
    min_mod_level: float = 0.0,
    min_depth: int = 0,
) -> list[dict[str, Any]]:
    """Compute pairwise correlation statistics for one transcript.

    Computes both same-type and cross-type pairs.

    Args:
        tx_name: Transcript name (used as output label).
        matrix: ``(n_reads, tx_length)`` uint8 array from HDF5.
        weights: ``(n_reads,)`` float32 Oarfish assignment probabilities.
        sites_by_mod: ``{mod_type: [{"pos": int, "n_mod": int,
            "mod_level": float, "depth": int}, ...]}`` from
            the site summary file.
        mod_code_map: ``{mod_type_string: integer_code}`` from the HDF5
            ``/modification_codes`` group.
        min_mod_reads: Minimum ``n_modified`` for a site to be included.
        min_asp: Minimum assignment probability for a read to be included.
        min_mod_level: Minimum modification level for a site to be included.
        min_depth: Minimum total depth for a site to be included.

    Returns:
        List of dicts, one per site pair, with columns from ``_OUTPUT_COLS``.
    """
    # ---- filter reads by minimum assignment probability ----
    if min_asp > 0.0:
        read_mask = weights >= min_asp
        if read_mask.sum() == 0:
            return []
        matrix = matrix[read_mask]
        weights = weights[read_mask]

    # ---- flatten all candidates across modification types ----
    candidates: list[tuple[int, str, int, float]] = []
    for mod_str, site_list in sites_by_mod.items():
        mod_code = mod_code_map.get(mod_str)
        if mod_code is None:
            continue
        candidates.extend(
            (site["pos"], mod_str, mod_code, site.get("wt_mod_level", 0.0))
            for site in site_list
            if site["n_mod"] >= min_mod_reads
            and site["mod_level"] >= min_mod_level
            and site["depth"] >= min_depth
        )
    if len(candidates) < 2:
        return []

    # Sort by position for deterministic output
    candidates.sort(key=lambda x: x[0])

    # ---- pre-compute per-candidate masks and binary arrays ----
    w = weights.astype(np.float64)
    pre: list[tuple[int, str, int, np.ndarray, np.ndarray, float]] = []
    for pos_1b, mod_str, mod_code, wt_mod_level in candidates:
        col = matrix[:, pos_1b - 1]
        valid = (
            (col != CODE_UNCOVERED)
            & (col != CODE_MISMATCH)
            & (col != CODE_DELETION)
            & (col != CODE_FAIL)
        )
        other_mod = (col >= 4) & (col != mod_code) & (col != CODE_FAIL)
        valid = valid & (~other_mod)
        binary = (col == mod_code).astype(np.int8)
        pre.append((pos_1b, mod_str, mod_code, valid, binary, wt_mod_level))
    k = len(candidates)

    pair_p_values: list[float] = []
    pair_wp_values: list[float] = []
    pair_rows: list[dict[str, Any]] = []

    for i in range(k):
        pos_i, mod_i, code_i, vi, bi, wt_ml_i = pre[i]
        for j in range(i + 1, k):
            pos_j, mod_j, code_j, vj, bj, wt_ml_j = pre[j]

            joint_valid = vi & vj
            if joint_valid.sum() < 2:
                continue

            bi_j = bi[joint_valid].astype(bool)
            bj_j = bj[joint_valid].astype(bool)
            w_joint = w[joint_valid]

            # Unweighted contingency table
            n11 = int((bi_j & bj_j).sum())
            n10 = int((bi_j & ~bj_j).sum())
            n01 = int((~bi_j & bj_j).sum())
            n00 = int((~bi_j & ~bj_j).sum())

            # Weighted contingency table
            bi_num = bi_j.astype(np.float64)
            bj_num = bj_j.astype(np.float64)
            w11 = (w_joint * bi_num * bj_num).sum()
            w10 = (w_joint * bi_num * (1.0 - bj_num)).sum()
            w01 = (w_joint * (1.0 - bi_num) * bj_num).sum()
            w00 = (w_joint * (1.0 - bi_num) * (1.0 - bj_num)).sum()

            # Unweighted statistics
            n_total = n11 + n10 + n01 + n00
            corr = _pearson_r_from_counts(n11, n10, n01, n00)
            p_val = _pearson_pvalue(corr, n_total)
            mi = _mutual_information(n11, n10, n01, n00)
            odds = _odds_ratio(n11, n10, n01, n00)

            # Weighted statistics
            wcorr, n_eff = _weighted_pearson_r(bi_j, bj_j, w_joint)
            w_p_val = _pearson_pvalue(wcorr, n_eff)
            w_mi = _mutual_information(w11, w10, w01, w00)
            w_odds = _odds_ratio(w11, w10, w01, w00)

            pair_p_values.append(p_val)
            pair_wp_values.append(w_p_val)
            pair_rows.append(
                {
                    "transcript_id": tx_name,
                    "site1": pos_i,
                    "site2": pos_j,
                    "mod_type1": mod_i,
                    "mod_type2": mod_j,
                    "wt_mod_level1": round(wt_ml_i, 6),
                    "wt_mod_level2": round(wt_ml_j, 6),
                    "n11": n11,
                    "n10": n10,
                    "n01": n01,
                    "n00": n00,
                    "w11": round(w11, 4),
                    "w10": round(w10, 4),
                    "w01": round(w01, 4),
                    "w00": round(w00, 4),
                    "corr": round(corr, 6),
                    "pvalue": p_val,
                    "qvalue": 0.0,
                    "wcorr": round(wcorr, 6),
                    "wpvalue": w_p_val,
                    "wqvalue": 0.0,
                    "mi": round(mi, 6),
                    "wmi": round(w_mi, 6),
                    "or": round(odds, 6),
                    "wor": round(w_odds, 6),
                }
            )

    # BH FDR correction — all pairs within this transcript
    if pair_p_values:
        q_values = bh_fdr(pair_p_values)
        for r, qv in zip(pair_rows, q_values):
            r["qvalue"] = round(qv, 6)
    if pair_wp_values:
        wq_values = bh_fdr(pair_wp_values)
        for r, qv in zip(pair_rows, wq_values):
            r["wqvalue"] = round(qv, 6)

    return pair_rows


_CORR_SCHEMA = pa.schema(
    [
        ("transcript_id", pa.string()),
        ("site1", pa.int32()),
        ("site2", pa.int32()),
        ("mod_type1", pa.string()),
        ("mod_type2", pa.string()),
        ("wt_mod_level1", pa.float64()),
        ("wt_mod_level2", pa.float64()),
        ("n11", pa.int32()),
        ("n10", pa.int32()),
        ("n01", pa.int32()),
        ("n00", pa.int32()),
        ("w11", pa.float64()),
        ("w10", pa.float64()),
        ("w01", pa.float64()),
        ("w00", pa.float64()),
        ("corr", pa.float64()),
        ("pvalue", pa.float64()),
        ("qvalue", pa.float64()),
        ("wcorr", pa.float64()),
        ("wpvalue", pa.float64()),
        ("wqvalue", pa.float64()),
        ("mi", pa.float64()),
        ("wmi", pa.float64()),
        ("or", pa.float64()),
        ("wor", pa.float64()),
    ]
)


# ---------- visualization ----------


def _plot_transcript_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    positions: np.ndarray,
    mod_types_str: list[str],
    type_colors: dict[str, str],
    rna_length: int,
    title: str,
    metric_label: str = "Pearson r",
) -> None:
    """Plot an upward-pointing pyramid heatmap for RNA modification associations.

    Follows the visualisation scheme from ``scripts/mod_plot.py``:
    matrix-index-based diamond grid (pyramid) above a horizontal transcript
    axis (5'→3') with downward-pointing coordinate ticks and physical
    nucleotide positions mapped proportionally.  Modification sites are
    marked as coloured dots on the axis with connecting lines up to the
    pyramid base.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    matrix : (N, N) np.ndarray
        Dense symmetric matrix of association scores.
    positions : (N,) array-like
        Nucleotide positions of the modification sites (sorted, 1-based).
    mod_types_str : (N,) list of str
        Modification type label for each site.
    type_colors : dict[str, str]
        Colour mapping per modification type.
    rna_length : int
        Total transcript length for proportional axis mapping.
    title : str
        Figure title.
    metric_label : str
        Label for the colour bar (e.g. "Weighted Pearson r").
    """
    N = len(positions)
    ax.set_aspect("equal")

    cmap = plt.get_cmap("bwr")
    vmin, vmax = -1.0, 1.0

    # ---- vertical offset from transcript axis to pyramid base ----
    offset = max(1.0, N * 0.15)

    # ---- diamond cells (pyramid) ----
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    for i in range(N):
        for j in range(i, N):
            val = matrix[i, j]
            if np.isnan(val):
                continue

            x_c = (i + j) / 2.0
            y_c = (j - i) / 2.0 + offset

            vertices = [
                (x_c, y_c + 0.5),  # Top
                (x_c + 0.5, y_c),  # Right
                (x_c, y_c - 0.5),  # Bottom
                (x_c - 0.5, y_c),  # Left
            ]

            color = cmap(norm(val))

            poly = Polygon(vertices, facecolor=color, edgecolor="white", linewidth=1)
            ax.add_patch(poly)

    # ---- RNA transcript axis (matrix-index space) ----
    axis_start_x = -0.5
    axis_end_x = N - 0.5
    axis_width = axis_end_x - axis_start_x

    ax.plot([axis_start_x, axis_end_x], [0, 0], color="black", lw=3, zorder=3)
    ax.text(axis_start_x - 0.2, 0, "5'", va="center", ha="right", fontsize=12)
    ax.text(axis_end_x + 0.2, 0, "3'", va="center", ha="left", fontsize=12)

    # ---- physical coordinate ticks (downward-pointing) ----
    step = _nice_tick_step(rna_length)
    tick_positions = np.arange(0, rna_length + 1, step)
    for tick in tick_positions:
        tick_fraction = tick / rna_length
        tick_x = axis_start_x + (tick_fraction * axis_width)
        ax.plot([tick_x, tick_x], [0, -0.15], color="black", lw=1, zorder=3)
        ax.text(tick_x, -0.35, f"{tick}", va="top", ha="center", fontsize=10)

    # ---- site dots and connecting lines ----
    seen: set[str] = set()
    for i, (pos, mt) in enumerate(zip(positions, mod_types_str)):
        col = type_colors.get(mt, "black")

        # Proportional x coordinate on the matrix-index axis
        fraction = pos / rna_length
        x_axis = axis_start_x + (fraction * axis_width)

        # Dot on the transcript axis
        label = mt if mt not in seen else None
        if label is not None:
            seen.add(mt)
        ax.scatter(x_axis, 0, color=col, s=40, zorder=4, label=label)

        # Connecting line from physical position to pyramid column base
        x_matrix = i
        y_matrix = offset - 0.5
        ax.plot(
            [x_axis, x_matrix],
            [0, y_matrix],
            color=col,
            linestyle="-",
            lw=0.8,
            zorder=2,
        )

    # ---- colour bar ----
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = ax.figure.colorbar(
        sm, ax=ax, orientation="horizontal", pad=0.08, shrink=0.4, aspect=20
    )
    cbar.set_label(metric_label)

    # ---- formatting ----
    ax.set_xlim(axis_start_x - 1.5, axis_end_x + 1.5)
    ax.set_ylim(-1.2, ((N - 1) / 2.0) + offset + 1)
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)
    ax.legend(
        title="Modification Type",
        loc="upper left",
        bbox_to_anchor=(0.85, 0.95),
        fontsize=8,
        title_fontsize=9,
    )


def _generate_plots(
    all_rows: list[dict[str, Any]],
    h5_paths: list[str],
    out_dir: str,
    all_sites: dict[str, dict[str, list[dict[str, int | float]]]],
    metric: str = "wcorr",
) -> None:
    """Generate pyramid heatmap PDFs per transcript.

    Prepares per-transcript data (dense association matrix, site positions,
    modification types) and delegates drawing to
    :func:`_plot_transcript_heatmap`, which follows the visualisation scheme
    from ``scripts/mod_plot.py``.

    Parameters
    ----------
    all_rows : list[dict]
        Correlation rows (one per site pair).
    h5_paths : list[str]
        Paths to HDF5 files (the first file is used for transcript lengths).
    out_dir : str
        Output directory for PDF files.
    all_sites : dict
        Site summary data, used to determine the dominant modification type
        (by ``mod_level``) at each site for colouring.
    metric : str
        Column name of the association statistic to visualise in the heatmap.
    """
    os.makedirs(out_dir, exist_ok=True)

    metric_label = _METRIC_LABELS.get(metric, metric)

    # ---- collect single mod types (human-readable) and assign colours ----
    raw_mod_types: set[str] = set()
    for row in all_rows:
        raw_mod_types.add(_sam_to_human(row["mod_type1"]))
        raw_mod_types.add(_sam_to_human(row["mod_type2"]))
    # Use predefined colours for known types; grey fallback for unknowns
    mod_color = {mt: _MOD_COLORS.get(mt, "#999999") for mt in sorted(raw_mod_types)}

    # ---- build dominant mod type lookup from site summary (by mod_level) ----
    # site_summary_dominant: {tx_name: {pos: human_readable_mod_type}}
    site_summary_dominant: dict[str, dict[int, str]] = {}
    for tx_name, mod_groups in all_sites.items():
        pos_best: dict[int, tuple[str, float]] = {}
        for mod_type, sites in mod_groups.items():
            for site in sites:
                pos = site["pos"]
                ml = site["mod_level"]
                if pos not in pos_best or ml > pos_best[pos][1]:
                    pos_best[pos] = (_sam_to_human(mod_type), ml)
        site_summary_dominant[tx_name] = {
            pos: mt for pos, (mt, _ml) in pos_best.items()
        }

    # ---- group by transcript ----
    tx_groups: dict[str, list] = defaultdict(list)
    for row in all_rows:
        tx_groups[row["transcript_id"]].append(
            (
                row["site1"],
                row["site2"],
                row[metric],
                row["mod_type1"],
                row["mod_type2"],
            )
        )

    # ---- get transcript lengths from the HDF5 ----
    tx_lengths: dict[str, int] = {}
    with h5py.File(h5_paths[0], "r") as h5:
        for tx_name in h5["transcripts"]:
            tx_lengths[tx_name] = h5[f"transcripts/{tx_name}/matrix"].shape[1]

    for tx_name in sorted(tx_groups.keys()):
        pairs = tx_groups[tx_name]
        rna_length = tx_lengths.get(tx_name)
        if rna_length is None:
            rna_length = max(max(s1, s2) for s1, s2, _, _, _ in pairs) + 100

        # ---- collect unique sites and their dominant modification type ----
        sites = sorted(set(s for p in pairs for s in (p[0], p[1])))
        n_sites = len(sites)
        if n_sites < 2:
            continue

        dominant_lookup = site_summary_dominant.get(tx_name, {})
        site_mod: dict[int, str] = {}
        for pos in sites:
            site_mod[pos] = dominant_lookup.get(pos, "?")

        site_to_idx = {s: i for i, s in enumerate(sites)}
        mod_types_str = [site_mod[s] for s in sites]

        # ---- build dense association matrix ----
        assoc = np.full((n_sites, n_sites), np.nan)
        for s1, s2, val, _, _ in pairs:
            i, j = site_to_idx[s1], site_to_idx[s2]
            if i < j:
                assoc[i, j] = val
            else:
                assoc[j, i] = val
        # Self-correlation / self-value for pyramid base
        diag_val = 1.0 if metric in ("corr", "wcorr") else 0.0
        np.fill_diagonal(assoc, diag_val)

        # ---- plot ----
        fig, ax = plt.subplots(figsize=(12, 8))
        _plot_transcript_heatmap(
            ax,
            assoc,
            sites,
            mod_types_str,
            mod_color,
            rna_length,
            tx_name,
            metric_label=metric_label,
        )
        fig.tight_layout()
        pdf_path = os.path.join(out_dir, f"{tx_name}.pdf")
        fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


# ---------- main ----------


def main(args: argparse.Namespace | None = None) -> None:
    """Compute pairwise modification site correlations.

    Reads one or more HDF5 matrices from ``mod_scan`` and the site summary
    from ``mod_sites``, then computes pairwise association statistics
    (Pearson r, odds ratio, mutual information) for each transcript.
    When multiple HDF5 files are provided, reads for the same transcript
    are pooled across all files.  Results are written as Parquet or TSV.
    """
    if args is None:
        args = parse_args()

    if args.verbose:
        print("[mod_corr] Reading site summary...", file=sys.stderr)
    all_sites = read_site_summary(args.sites)

    if args.verbose:
        n_tx, n_mod = len(all_sites), sum(len(m) for m in all_sites.values())
        print(
            f"[mod_corr] {n_tx} transcripts, {n_mod} mod-type groups",
            file=sys.stderr,
        )

    # ---- Open all HDF5 files ----

    with ExitStack() as stack:
        h5_files = [stack.enter_context(h5py.File(f, "r")) for f in args.h5]

        if args.verbose:
            print(
                f"[mod_corr] Opened {len(h5_files)} HDF5 file(s)",
                file=sys.stderr,
            )

        # Read and validate modification codes
        all_mod_maps = [read_mod_codes(h5) for h5 in h5_files]
        try:
            mod_code_map = validate_mod_codes(all_mod_maps, list(args.h5))
        except ValueError as exc:
            print(f"[mod_corr] Error: {exc}", file=sys.stderr)
            sys.exit(1)

        # Build union of transcript names across all HDF5 files
        all_h5_tx_sets = [set(h5["transcripts"].keys()) for h5 in h5_files]
        h5_tx_union = set.union(*all_h5_tx_sets)

        if args.transcripts is not None:
            requested = set(args.transcripts)
            h5_tx_union &= requested
            if args.verbose:
                print(
                    f"[mod_corr] Filtered to {len(h5_tx_union)}/"
                    f"{len(requested)} requested transcripts in HDF5 files",
                    file=sys.stderr,
                )

        site_tx = set(all_sites.keys())
        common_tx = sorted(h5_tx_union & site_tx)

        if args.verbose:
            file_counts = ", ".join(
                f"{f}: {len(s)} tx" for f, s in zip(args.h5, all_h5_tx_sets)
            )
            print(
                f"[mod_corr] {len(common_tx)} transcripts in common "
                f"across {len(h5_files)} files ({file_counts})",
                file=sys.stderr,
            )

        # ---- Process each transcript (pooling across files) ----

        all_rows: list[dict[str, Any]] = []
        processed = 0

        for tx_name in common_tx:
            matrices: list[np.ndarray] = []
            weights_list: list[np.ndarray] = []
            tx_lengths_found: list[int | None] = []

            for h5 in h5_files:
                result = load_transcript_data(h5, tx_name, args.min_asp)
                if result is not None:
                    matrix_f, weights_f = result
                    matrices.append(matrix_f)
                    weights_list.append(weights_f)
                    tx_lengths_found.append(matrix_f.shape[1])
                else:
                    tx_lengths_found.append(None)

            if not matrices:
                processed += 1
                continue

            try:
                validate_tx_lengths(tx_name, tx_lengths_found, list(args.h5))
            except ValueError as exc:
                print(
                    f"[mod_corr] Warning: {exc} — skipping transcript",
                    file=sys.stderr,
                )
                processed += 1
                continue

            if len(matrices) == 1:
                matrix = matrices[0]
                weights = weights_list[0]
            else:
                matrix = np.vstack(matrices)
                weights = np.concatenate(weights_list)

            if args.verbose and len(matrices) > 1:
                n_from = ", ".join(f"{m.shape[0]}" for m in matrices)
                print(
                    f"[mod_corr] {tx_name}: {matrix.shape[0]} reads "
                    f"pooled from {len(matrices)} file(s) ({n_from})",
                    file=sys.stderr,
                )

            tx_results = process_transcript(
                tx_name,
                matrix,
                weights,
                all_sites.get(tx_name, {}),
                mod_code_map,
                args.min_mod_reads,
                args.min_asp,
                args.min_mod_level,
                args.min_coverage,
            )
            all_rows.extend(tx_results)
            processed += 1
            if args.verbose and processed % 1000 == 0:
                print(
                    f"[mod_corr] Processed {processed}/{len(common_tx)} transcripts...",
                    file=sys.stderr,
                )

    if args.verbose:
        print(f"[mod_corr] Total pairs: {len(all_rows)}", file=sys.stderr)

    if args.format == "tsv":
        write_tsv(all_rows, args.output, _TSV_HEADER, _OUTPUT_COLS, args.gzip)
    else:
        write_parquet(all_rows, args.output, _CORR_SCHEMA, _OUTPUT_COLS)

    if args.plot_dir:
        if args.verbose:
            print(
                "[mod_corr] Generating rotated triangular heatmap PDFs...",
                file=sys.stderr,
            )
        _generate_plots(all_rows, list(args.h5), args.plot_dir, all_sites, args.metric)
        if args.verbose:
            print(f"[mod_corr] Plots written to {args.plot_dir}/", file=sys.stderr)

    if args.verbose:
        print(f"[mod_corr] Done. Output written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
