#!/usr/bin/env python3
"""Plot weighted density of poly(A) tail length distribution for a selected
transcript or gene.

Reads a poly(A) file (TSV, TSV.GZ, or Parquet) produced by ``polya_calc``
or ``polya_gene``, extracts the per-read weights and lengths for the
requested feature, and generates a weighted-density plot.

Usage:
    python polya_wden_plot.py -i results.pa.tsv.gz -f FBtr0073078 -o plot.png
    python polya_wden_plot.py -i results.pa.parquet -f GeneX -o plot.pdf
"""

import argparse
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Input reading
# ---------------------------------------------------------------------------


def _read_parquet_polya(path: str) -> tuple[str, dict[str, dict]]:
    """Read a poly(A) Parquet file and return ``(id_col_name, data_dict)``.

    Returns a structure matching :func:`parse_polyA_file` so downstream
    code can treat TSV and Parquet uniformly.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    columns = table.column_names
    rows = table.to_pylist()

    # Detect ID column
    id_col_name = "transcript_id" if "transcript_id" in columns else "gene_id"

    data_dict: dict[str, dict] = {}
    for row in rows:
        feature_id = str(row[id_col_name])
        weights = np.asarray(row["weights"], dtype=np.float64)
        lengths = np.asarray(row["lengths"], dtype=np.int64)

        n_reads = len(weights)
        total_wt = float(np.sum(weights))
        wmlen = (
            float(np.sum(weights * lengths) / total_wt) if total_wt > 0 else 0.0
        )

        data_dict[feature_id] = {
            "n_reads": n_reads,
            "total_wt": total_wt,
            "wmlen": wmlen,
            "weights": weights,
            "lengths": lengths,
        }

    return id_col_name, data_dict


def read_polya_input(path: str) -> tuple[str, dict[str, dict]]:
    """Read a poly(A) file (TSV, TSV.GZ, or Parquet).

    Args:
        path: Input file path.  Format is detected by suffix:
            ``.parquet`` → Parquet, ``.tsv`` / ``.tsv.gz`` → TSV.

    Returns:
        ``(id_col_name, data_dict)`` — see :func:`parse_polyA_file`.
    """
    if path.endswith(".parquet"):
        print(f"Reading Parquet input from {path}...", file=sys.stderr)
        return _read_parquet_polya(path)

    # TSV path — use the shared parser from _parsing
    try:
        from isolens._parsing import parse_polyA_file
    except ImportError:
        from _parsing import parse_polyA_file  # type: ignore[no-redef]

    return parse_polyA_file(path)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _weighted_kde(
    weights: np.ndarray, lengths: np.ndarray, bw_method: str | float
) -> tuple:
    """Compute a weighted Gaussian KDE.

    Uses ``scipy.stats.gaussian_kde`` which supports per-sample weights.

    Args:
        weights: Per-read assignment probabilities.
        lengths: Per-read poly(A) tail lengths.
        bw_method: Bandwidth selection method for ``gaussian_kde``.

    Returns:
        ``(kde, x_grid, y_grid)`` where *kde* is the fitted
        ``gaussian_kde`` object, and *x_grid*/*y_grid* are the
        evaluation points and density values for plotting.
    """
    from scipy.stats import gaussian_kde

    # gaussian_kde expects 2-D data; weights sum is automatically
    # normalised internally (neff = sum(weights) is used as the
    # effective sample size for bandwidth scaling).
    kde = gaussian_kde(lengths.reshape(1, -1), weights=weights, bw_method=bw_method)

    x_min = max(0, lengths.min() - 5)
    x_max = lengths.max() + 5
    x_grid = np.linspace(x_min, x_max, 500)
    y_grid = kde.evaluate(x_grid.reshape(1, -1)).ravel()

    return kde, x_grid, y_grid


def plot_weighted_density(
    weights: np.ndarray,
    lengths: np.ndarray,
    title: str,
    bw_method: str | float,
    use_log: bool,
    output_path: str,
) -> None:
    """Generate and save a weighted poly(A) density plot.

    Produces a figure with:
    - Weighted histogram (semi-transparent bars)
    - Weighted KDE curve (solid line)
    - Vertical dashed line at the weighted mean

    Args:
        weights: Per-read assignment probabilities.
        lengths: Per-read poly(A) tail lengths (same length as *weights*).
        title: Plot title.
        bw_method: KDE bandwidth method (``"scott"``, ``"silverman"``, or a
            float).
        use_log: If ``True``, use log-scale on the x-axis (log(L+1) with
            log-scale ticks).
        output_path: File path for the saved figure (format inferred from
            suffix, e.g. ``.png``, ``.pdf``, ``.svg``).
    """
    import matplotlib.pyplot as plt

    # ---- data preparation ----
    mask = (weights > 0) & (lengths >= 0)
    weights = weights[mask]
    lengths = lengths[mask]

    if len(lengths) == 0:
        print("Error: no valid reads after filtering.", file=sys.stderr)
        sys.exit(1)

    # For log-scale visualisation, transform the data
    if use_log:
        plot_lengths = np.log(lengths.astype(np.float64) + 1.0)
        x_label = "log(poly(A) tail length + 1)"
    else:
        plot_lengths = lengths.astype(np.float64)
        x_label = "Poly(A) tail length (nt)"

    total_wt = float(np.sum(weights))
    wm = float(np.sum(weights * plot_lengths) / total_wt) if total_wt > 0 else 0.0

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(10, 6))

    # Weighted histogram
    bins = max(15, min(80, int(np.sqrt(len(lengths))) * 2))
    counts, bin_edges, patches = ax.hist(
        plot_lengths,
        bins=bins,
        weights=weights,
        alpha=0.4,
        color="#3a7eb0",
        edgecolor="white",
        linewidth=0.5,
        density=True,
        label="Weighted histogram",
    )

    # Weighted KDE
    try:
        kde, x_grid, y_grid = _weighted_kde(plot_lengths, weights, bw_method)
        ax.plot(
            x_grid,
            y_grid,
            color="#c44e52",
            linewidth=2,
            label=f"KDE (bw={bw_method})",
        )
    except Exception as exc:
        print(f"Warning: KDE failed ({exc}), showing histogram only.", file=sys.stderr)

    # Weighted mean line
    ax.axvline(
        wm,
        color="#4c4c4c",
        linestyle="--",
        linewidth=1.5,
        label=f"Weighted mean ({wm:.1f})",
    )

    # ---- labels & styling ----
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Weighted density", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)

    # Annotate with read count
    ax.text(
        0.98,
        0.95,
        f"n={len(lengths)} reads\nwt_sum={total_wt:.1f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {output_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot weighted poly(A) tail length density for a "
        "selected transcript or gene."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input poly(A) file (.tsv, .tsv.gz, or .parquet)",
    )
    parser.add_argument(
        "-f",
        "--feature",
        required=True,
        help="Transcript ID or gene ID to plot",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output plot file path (.png, .pdf, .svg, etc.)",
    )
    parser.add_argument(
        "-t",
        "--title",
        default=None,
        help="Custom plot title [default: feature ID]",
    )
    parser.add_argument(
        "--bw",
        default="silverman",
        help="KDE bandwidth method: 'scott', 'silverman', or a float "
        "[default: silverman]",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        default=False,
        help="Use log-scale x-axis (log(L+1)) for visualisation",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()

    # Parse bw_method — try as float first, then fall back to string
    try:
        bw_method: str | float = float(args.bw)
    except ValueError:
        bw_method = args.bw

    # Read input
    id_col_name, data_dict = read_polya_input(args.input)
    print(
        f"Loaded {len(data_dict)} features ({id_col_name}-level data).",
        file=sys.stderr,
    )

    # Look up the requested feature
    if args.feature not in data_dict:
        print(
            f"Error: feature '{args.feature}' not found in input file.",
            file=sys.stderr,
        )
        preview = list(data_dict.keys())[:20]
        suffix = "..." if len(data_dict) > 20 else ""
        print(
            f"Available features ({len(data_dict)} total): "
            f"{', '.join(preview)}{suffix}",
            file=sys.stderr,
        )
        sys.exit(1)

    d = data_dict[args.feature]
    title = args.title if args.title is not None else args.feature

    plot_weighted_density(
        weights=d["weights"],
        lengths=d["lengths"],
        title=title,
        bw_method=bw_method,
        use_log=args.log,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
