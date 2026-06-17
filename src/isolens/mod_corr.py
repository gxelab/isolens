#!/usr/bin/env python3
"""mod_corr: Pairwise modification site correlation analysis.

Identifies cooperative or antagonistic relationships between modification
sites within the same transcript — both within a single modification type
and across different modification types.  Reads the HDF5 from ``mod_scan.py``
and the site summary from ``mod_sites.py``.

See notebooks/01_mod.md for the full specification.
"""

import argparse
import gzip
import os
import sys
from collections import defaultdict

import h5py
import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

matplotlib.use("Agg")  # non-interactive backend for PDF output
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.stats import fisher_exact

try:
    from isolens.mod_scan import (CODE_UNCOVERED, CODE_CANONICAL,
                                   CODE_MISMATCH, CODE_DELETION)
except ImportError:
    from mod_scan import (  # type: ignore[no-redef]
        CODE_UNCOVERED, CODE_CANONICAL, CODE_MISMATCH, CODE_DELETION,
    )

# ---------- constants ----------

_HAC = 0.5  # Haldane-Anscombe correction for odds ratio zeros

_OUTPUT_COLS = [
    "transcript_id", "site1", "site2", "modification_type",
    "n11", "n10", "n01", "n00",
    "weighted_n11", "weighted_n10", "weighted_n01", "weighted_n00",
    "phi", "weighted_phi", "odds_ratio",
    "p_value", "q_value",
    "mutual_information", "weighted_mutual_information",
]

_TSV_HEADER = "\t".join(_OUTPUT_COLS)

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
    "m5C": "#377EB8",      # blue
    "m6A": "#E41A1C",      # red
    "inosine": "#4DAF4A",  # green
    "pseU": "#FF7F00",     # orange
    "2Ome": "#984EA3",     # purple
}


# ---------- helpers ----------


def _sam_to_human(sam_code):
    """Convert a SAM modification code to a human-readable name."""
    return _SAM_TO_HUMAN.get(sam_code, sam_code)


def _nice_tick_step(rna_length):
    """Choose a regular tick interval for the transcript axis.

    Returns a step size that yields ~3-7 ticks across the transcript.
    """
    for step in (100, 200, 500, 1000, 2000, 5000, 10000):
        if rna_length / step <= 6:
            return step
    return 10000


# ---------- CLI ----------


def parse_args():
    parser = argparse.ArgumentParser(
        description="mod_corr: Pairwise modification site correlation analysis"
    )
    parser.add_argument(
        "-i", "--h5", required=True, help="Input HDF5 file from mod_scan")
    parser.add_argument(
        "-s", "--site-summary", required=True,
        help="Input site summary from mod_sites (Parquet or TSV/TSV.GZ)")
    parser.add_argument(
        "-o", "--output", required=True, help="Output file path")
    parser.add_argument(
        "-m", "--min-support", type=int, default=10,
        help="Minimum n_modified for a site to be considered [default: 10]")
    parser.add_argument(
        "-p", "--min-asp", type=float, default=0.0,
        help="Minimum Oarfish assignment probability for a read to be "
             "included [default: 0.0 (no filter)]")
    parser.add_argument(
        "-f", "--format", choices=["parquet", "tsv"], default="parquet",
        help="Output format: parquet (default) or tsv")
    parser.add_argument(
        "-z", "--gzip", action="store_true",
        help="Gzip-compress TSV output (ignored for parquet)")
    parser.add_argument(
        "-P", "--plot", metavar="DIR", default=None,
        help="Generate rotated triangular heatmap PDFs per transcript "
             "in the given output directory")
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print progress to stderr")
    return parser.parse_args()


# ---------- site-summary reader ----------


def read_site_summary(path):
    if path.endswith(".parquet"):
        return _read_sites_parquet(path)
    else:
        return _read_sites_tsv(path)


def _read_sites_parquet(path):
    table = pq.read_table(path, columns=[
        "transcript_id", "position", "modification_type", "n_modified"])
    sites = {}
    for i in range(len(table)):
        tx = table.column("transcript_id")[i].as_py()
        pos = table.column("position")[i].as_py()
        mod = table.column("modification_type")[i].as_py()
        n_mod = table.column("n_modified")[i].as_py()
        sites.setdefault(tx, {}).setdefault(mod, []).append((pos, n_mod))
    return sites


def _read_sites_tsv(path):
    open_func = gzip.open if path.endswith(".gz") else open
    mode = "rt" if path.endswith(".gz") else "r"
    sites = {}
    with open_func(path, mode, encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        tx_col = header.index("transcript_id")
        pos_col = header.index("position")
        mod_col = header.index("modification_type")
        nmod_col = header.index("n_modified")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) <= max(tx_col, pos_col, mod_col, nmod_col):
                continue
            tx = parts[tx_col]
            sites.setdefault(tx, {}).setdefault(
                parts[mod_col], []).append(
                (int(parts[pos_col]), int(parts[nmod_col])))
    return sites


# ---------- statistics ----------


def _phi_coefficient(n11, n10, n01, n00):
    n1x, n0x = n11 + n10, n01 + n00
    nx1, nx0 = n11 + n01, n10 + n00
    denom = n1x * n0x * nx1 * nx0
    if denom <= 0:
        return 0.0
    return (n11 * n00 - n10 * n01) / np.sqrt(denom)


def _odds_ratio(n11, n10, n01, n00):
    a, b = n11 + _HAC, n10 + _HAC
    c, d = n01 + _HAC, n00 + _HAC
    return (a * d) / (b * c)


def _mutual_information(n11, n10, n01, n00):
    total = n11 + n10 + n01 + n00
    if total <= 0:
        return 0.0
    p11, p10, p01, p00 = n11 / total, n10 / total, n01 / total, n00 / total
    p1x, p0x = p11 + p10, p01 + p00
    px1, px0 = p11 + p01, p10 + p00
    mi = 0.0
    for pj, pr, pc in [(p11, p1x, px1), (p10, p1x, px0),
                        (p01, p0x, px1), (p00, p0x, px0)]:
        if pj > 0 and pr > 0 and pc > 0:
            mi += pj * np.log2(pj / (pr * pc))
    return mi


def _bh_fdr(p_values):
    n = len(p_values)
    if n == 0:
        return []
    idx = np.argsort(p_values)
    q = np.zeros(n, dtype=np.float64)
    for rank, i in enumerate(idx):
        q[i] = min(1.0, p_values[i] * n / (rank + 1))
    for k in range(n - 2, -1, -1):
        q[idx[k]] = min(q[idx[k]], q[idx[k + 1]])
    return q.tolist()


# ---------- per-transcript processing ----------


def process_transcript(tx_name, matrix, weights, sites_by_mod,
                       mod_code_map, min_support, min_asp=0.0):
    """Compute pairwise correlation statistics for one transcript.

    Computes both same-type and cross-type pairs.
    """
    # ---- filter reads by minimum assignment probability ----
    if min_asp > 0.0:
        read_mask = weights >= min_asp
        if read_mask.sum() == 0:
            return []
        matrix = matrix[read_mask]
        weights = weights[read_mask]

    # ---- flatten all candidates across modification types ----
    candidates = []  # (pos_1based, mod_str, mod_code)
    for mod_str, site_list in sites_by_mod.items():
        mod_code = mod_code_map.get(mod_str)
        if mod_code is None:
            continue
        candidates.extend(
            (pos, mod_str, mod_code)
            for pos, n_mod in site_list if n_mod >= min_support
        )
    if len(candidates) < 2:
        return []

    # Sort by position for deterministic output
    candidates.sort(key=lambda x: x[0])

    # ---- pre-compute per-candidate masks and binary arrays ----
    pre = []  # (pos, mod_str, mod_code, valid_mask, binary_int8)
    for pos_1b, mod_str, mod_code in candidates:
        col = matrix[:, pos_1b - 1]
        valid = ((col != CODE_UNCOVERED) & (col != CODE_MISMATCH)
                 & (col != CODE_DELETION))
        other_mod = (col >= 4) & (col != mod_code)
        valid = valid & (~other_mod)
        binary = (col == mod_code).astype(np.int8)
        pre.append((pos_1b, mod_str, mod_code, valid, binary))

    w = weights.astype(np.float64)
    k = len(candidates)

    pair_p_values = []
    pair_rows = []

    for i in range(k):
        pos_i, mod_i, code_i, vi, bi = pre[i]
        for j in range(i + 1, k):
            pos_j, mod_j, code_j, vj, bj = pre[j]

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

            phi = _phi_coefficient(n11, n10, n01, n00)
            w_phi = _phi_coefficient(w11, w10, w01, w00)
            odds = _odds_ratio(n11, n10, n01, n00)
            try:
                _, p_val = fisher_exact([[n11, n10], [n01, n00]])
            except (ValueError, OverflowError):
                p_val = 1.0
            mi = _mutual_information(n11, n10, n01, n00)
            w_mi = _mutual_information(w11, w10, w01, w00)

            # Label: single mod type or "mod_a:mod_b" for cross-type
            if mod_i == mod_j:
                mod_label = mod_i
            else:
                mod_label = f"{mod_i}:{mod_j}"

            pair_p_values.append(p_val)
            pair_rows.append({
                "transcript_id": tx_name,
                "site1": pos_i, "site2": pos_j,
                "modification_type": mod_label,
                "n11": n11, "n10": n10, "n01": n01, "n00": n00,
                "weighted_n11": round(w11, 4),
                "weighted_n10": round(w10, 4),
                "weighted_n01": round(w01, 4),
                "weighted_n00": round(w00, 4),
                "phi": round(phi, 6),
                "weighted_phi": round(w_phi, 6),
                "odds_ratio": round(odds, 6),
                "p_value": p_val,
                "q_value": 0.0,
                "mutual_information": round(mi, 6),
                "weighted_mutual_information": round(w_mi, 6),
            })

    # BH FDR correction — all pairs within this transcript
    if pair_p_values:
        q_values = _bh_fdr(pair_p_values)
        for r, qv in zip(pair_rows, q_values):
            r["q_value"] = round(qv, 6)

    return pair_rows


# ---------- output writers ----------


def _write_tsv(all_rows, path, use_gzip):
    open_func = gzip.open if use_gzip else open
    mode = "wt" if use_gzip else "w"
    with open_func(path, mode, encoding="utf-8") as f:
        f.write(_TSV_HEADER + "\n")
        for row in all_rows:
            f.write("\t".join(str(row[c]) for c in _OUTPUT_COLS) + "\n")


def _write_parquet(all_rows, path):
    if not all_rows:
        schema = pa.schema([
            ("transcript_id", pa.string()),
            ("site1", pa.int32()), ("site2", pa.int32()),
            ("modification_type", pa.string()),
            ("n11", pa.int32()), ("n10", pa.int32()),
            ("n01", pa.int32()), ("n00", pa.int32()),
            ("weighted_n11", pa.float64()), ("weighted_n10", pa.float64()),
            ("weighted_n01", pa.float64()), ("weighted_n00", pa.float64()),
            ("phi", pa.float64()), ("weighted_phi", pa.float64()),
            ("odds_ratio", pa.float64()),
            ("p_value", pa.float64()), ("q_value", pa.float64()),
            ("mutual_information", pa.float64()),
            ("weighted_mutual_information", pa.float64()),
        ])
        with pq.ParquetWriter(path, schema) as w:
            w.write_table(pa.table(
                {k: pa.array([], type=schema.field(k).type)
                 for k in schema.names}))
        return
    arrays = {}
    for col in _OUTPUT_COLS:
        values = [r[col] for r in all_rows]
        if col in ("transcript_id", "modification_type"):
            arrays[col] = pa.array(values)
        elif col in ("site1", "site2", "n11", "n10", "n01", "n00"):
            arrays[col] = pa.array(values, type=pa.int32())
        else:
            arrays[col] = pa.array(values, type=pa.float64())
    pq.write_table(pa.table(arrays), path)


# ---------- visualization ----------


def _plot_transcript_heatmap(ax, matrix, positions, mod_types_str,
                              type_colors, rna_length, title):
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
        Dense symmetric matrix of association scores (values in [-1, 1]).
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
    """
    N = len(positions)
    ax.set_aspect("equal")

    cmap = plt.get_cmap("bwr")

    # ---- vertical offset from transcript axis to pyramid base ----
    offset = max(1.0, N * 0.15)

    # ---- diamond cells (pyramid) ----
    for i in range(N):
        for j in range(i, N):
            val = matrix[i, j]
            if np.isnan(val):
                continue

            x_c = (i + j) / 2.0
            y_c = (j - i) / 2.0 + offset

            vertices = [
                (x_c, y_c + 0.5),       # Top
                (x_c + 0.5, y_c),       # Right
                (x_c, y_c - 0.5),       # Bottom
                (x_c - 0.5, y_c),       # Left
            ]

            color_val = (val + 1) / 2.0   # [-1, 1] → [0, 1]
            color = cmap(color_val)

            poly = Polygon(vertices, facecolor=color,
                          edgecolor="white", linewidth=1)
            ax.add_patch(poly)

    # ---- RNA transcript axis (matrix-index space) ----
    axis_start_x = -0.5
    axis_end_x = N - 0.5
    axis_width = axis_end_x - axis_start_x

    ax.plot([axis_start_x, axis_end_x], [0, 0], color="black", lw=3, zorder=3)
    ax.text(axis_start_x - 0.2, 0, "5'",
            va="center", ha="right", fontsize=12)
    ax.text(axis_end_x + 0.2, 0, "3'",
            va="center", ha="left", fontsize=12)

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
        ax.plot([x_axis, x_matrix], [0, y_matrix], color=col,
                linestyle="-", lw=0.8, zorder=2)

    # ---- colour bar ----
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, ax=ax, orientation="horizontal",
                              pad=0.08, shrink=0.4, aspect=20)
    cbar.set_label("Phi coefficient")

    # ---- formatting ----
    ax.set_xlim(axis_start_x - 1.5, axis_end_x + 1.5)
    ax.set_ylim(-1.2, ((N - 1) / 2.0) + offset + 1)
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)
    ax.legend(title="Modification Type", loc="upper left",
              bbox_to_anchor=(0.85, 0.95), fontsize=8, title_fontsize=9)


def _generate_plots(all_rows, h5_path, out_dir):
    """Generate pyramid heatmap PDFs per transcript.

    Prepares per-transcript data (dense correlation matrix, site positions,
    modification types) and delegates drawing to
    :func:`_plot_transcript_heatmap`, which follows the visualisation scheme
    from ``scripts/mod_plot.py``.
    """
    os.makedirs(out_dir, exist_ok=True)

    # ---- collect single mod types (human-readable) and assign colours ----
    raw_mod_types: set[str] = set()
    for row in all_rows:
        for part in row["modification_type"].split(":"):
            raw_mod_types.add(_sam_to_human(part))
    # Use predefined colours for known types; grey fallback for unknowns
    mod_color = {mt: _MOD_COLORS.get(mt, "#999999")
                 for mt in sorted(raw_mod_types)}

    # ---- group by transcript ----
    tx_groups: dict[str, list] = defaultdict(list)
    for row in all_rows:
        tx_groups[row["transcript_id"]].append(
            (row["site1"], row["site2"], row["phi"], row["modification_type"]))

    # ---- get transcript lengths from the HDF5 ----
    tx_lengths: dict[str, int] = {}
    with h5py.File(h5_path, "r") as h5:
        for tx_name in h5["transcripts"]:
            tx_lengths[tx_name] = h5[f"transcripts/{tx_name}/matrix"].shape[1]

    for tx_name in sorted(tx_groups.keys()):
        pairs = tx_groups[tx_name]
        rna_length = tx_lengths.get(tx_name)
        if rna_length is None:
            rna_length = max(max(s1, s2) for s1, s2, _, _ in pairs) + 100

        # ---- collect unique sites and their dominant modification type ----
        sites = sorted(set(s for p in pairs for s in (p[0], p[1])))
        n_sites = len(sites)
        if n_sites < 2:
            continue

        site_mod: dict[int, str] = {}
        for pos in sites:
            mods: list[str] = []
            for s1, s2, _, mt in pairs:
                if s1 == pos or s2 == pos:
                    mods.extend(mt.split(":"))
            site_mod[pos] = _sam_to_human(
                max(set(mods), key=mods.count)) if mods else "?"

        site_to_idx = {s: i for i, s in enumerate(sites)}
        mod_types_str = [site_mod[s] for s in sites]

        # ---- build dense correlation matrix ----
        corr = np.full((n_sites, n_sites), np.nan)
        for s1, s2, phi_val, _ in pairs:
            i, j = site_to_idx[s1], site_to_idx[s2]
            if i < j:
                corr[i, j] = phi_val
            else:
                corr[j, i] = phi_val
        np.fill_diagonal(corr, 1.0)  # self-correlation for pyramid base

        # ---- plot ----
        fig, ax = plt.subplots(figsize=(12, 8))
        _plot_transcript_heatmap(
            ax, corr, sites, mod_types_str,
            mod_color, rna_length, tx_name,
        )
        fig.tight_layout()
        pdf_path = os.path.join(out_dir, f"{tx_name}.pdf")
        fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


# ---------- main ----------


def main():
    args = parse_args()

    if args.verbose:
        print("[mod_corr] Reading site summary...", file=sys.stderr)
    all_sites = read_site_summary(args.site_summary)

    if args.verbose:
        n_tx, n_mod = len(all_sites), sum(len(m) for m in all_sites.values())
        print(f"[mod_corr] {n_tx} transcripts, {n_mod} mod-type groups",
              file=sys.stderr)

    with h5py.File(args.h5, "r") as h5:
        mod_code_map = {}
        for mod_str, code in h5["modification_codes"].attrs.items():
            mod_code_map[mod_str] = int(code)

        h5_tx = set(h5["transcripts"].keys())
        site_tx = set(all_sites.keys())
        common_tx = sorted(h5_tx & site_tx)

        if args.verbose:
            print(f"[mod_corr] {len(common_tx)} transcripts in common",
                  file=sys.stderr)

        all_rows = []
        processed = 0
        for tx_name in common_tx:
            grp = h5[f"transcripts/{tx_name}"]
            matrix = grp["matrix"][:]
            weights = grp["read_weights"][:]
            tx_results = process_transcript(
                tx_name, matrix, weights,
                all_sites.get(tx_name, {}),
                mod_code_map, args.min_support, args.min_asp)
            all_rows.extend(tx_results)
            processed += 1
            if args.verbose and processed % 1000 == 0:
                print(f"[mod_corr] Processed {processed}/{len(common_tx)} "
                      f"transcripts...", file=sys.stderr)

    if args.verbose:
        print(f"[mod_corr] Total pairs: {len(all_rows)}", file=sys.stderr)

    if args.format == "tsv":
        _write_tsv(all_rows, args.output, args.gzip)
    else:
        _write_parquet(all_rows, args.output)

    if args.plot:
        if args.verbose:
            print("[mod_corr] Generating rotated triangular heatmap PDFs...",
                  file=sys.stderr)
        _generate_plots(all_rows, args.h5, args.plot)
        if args.verbose:
            print(f"[mod_corr] Plots written to {args.plot}/", file=sys.stderr)

    if args.verbose:
        print(f"[mod_corr] Done. Output written to {args.output}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
