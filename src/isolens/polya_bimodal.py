#!/usr/bin/env python3
"""Detect bimodal poly(A) tail length distributions per transcript or gene.

Uses a weighted 1-D Gaussian Mixture Model (EM algorithm) and a weighted
KDE peak-detection approach.  A consensus bimodal call is reported when
both methods agree.
"""

import argparse
import sys

import numpy as np
from scipy.signal import find_peaks
from scipy.special import logsumexp
from scipy.stats import gaussian_kde

try:
    from isolens._io import ensure_gz_suffix
    from isolens._parsing import open_by_suffix, parse_polyA_file
except ImportError:
    from _io import ensure_gz_suffix  # type: ignore[no-redef]

    from _parsing import (  # type: ignore[no-redef]
        open_by_suffix,
        parse_polyA_file,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for polya_bimodal."""
    parser = argparse.ArgumentParser(
        description="Detect bimodal poly(A) tail length distributions "
        "per transcript or gene using weighted GMM and KDE."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input poly(A) TSV file (from polya_calc or polya_t2g, gzipped or raw)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output bimodality TSV results file",
    )
    parser.add_argument(
        "-z",
        "--gzip",
        action="store_true",
        help="Compress the output TSV file using gzip",
    )
    parser.add_argument(
        "-l",
        "--min-length",
        type=float,
        default=0.0,
        help="Drop reads with poly(A) length below this threshold (default: 0)",
    )
    parser.add_argument(
        "-p",
        "--min-asp",
        type=float,
        default=0.1,
        help="Drop reads with assignment probability below this "
        "threshold (default: 0.1)",
    )
    parser.add_argument(
        "-e",
        "--min-ess",
        type=float,
        default=30.0,
        help="Skip feature if effective sample size (sum of remaining "
        "weights) is below this threshold (default: 30)",
    )
    parser.add_argument(
        "-k",
        "--kde-prominence",
        type=float,
        default=0.05,
        help="Prominence threshold for KDE peak detection (default: 0.05)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Core algorithm helpers
# ---------------------------------------------------------------------------


def _log_gaussian_pdf(x: np.ndarray, mean: float, var: float) -> np.ndarray:
    """Logarithm of the univariate Gaussian PDF.

    Args:
        x: 1-D array of values.
        mean: Mean of the Gaussian.
        var: Variance of the Gaussian (clamped to ≥ 1e-12).

    Returns:
        1-D array of log-probability densities.
    """
    var = max(var, 1e-12)
    return -0.5 * np.log(2.0 * np.pi * var) - 0.5 * (x - mean) ** 2 / var


def _fit_weighted_gmm_1d(
    x: np.ndarray,
    weights: np.ndarray,
    n_components: int,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit a weighted 1-D Gaussian Mixture Model via EM.

    Args:
        x: 1-D data values, shape ``(n,)``.
        weights: Non-negative observation weights, shape ``(n,)``.
        n_components: Number of mixture components (1 or 2).
        max_iter: Maximum EM iterations (k=2 only).
        tol: Relative convergence tolerance on log-likelihood.

    Returns:
        ``(means, variances, mix_weights, weighted_log_likelihood)``.
        *means*, *variances*, *mix_weights* are 1-D arrays of length
        *n_components*.  The log-likelihood is ``nan`` when the fit
        is degenerate (e.g. a collapsed component).
    """
    ess = float(np.sum(weights))
    if ess <= 0:
        return (
            np.full(n_components, np.nan),
            np.full(n_components, np.nan),
            np.full(n_components, np.nan),
            float("nan"),
        )

    if n_components == 1:
        mean = float(np.average(x, weights=weights))
        var = float(np.average((x - mean) ** 2, weights=weights))
        var = max(var, 1e-6)
        ll = float(np.sum(weights * _log_gaussian_pdf(x, mean, var)))
        return np.array([mean]), np.array([var]), np.array([1.0]), ll

    # --- n_components == 2: EM algorithm ---

    # Initialization: split at weighted median
    sorter = np.argsort(x)
    x_s = x[sorter]
    w_s = weights[sorter]
    cum_w = np.cumsum(w_s)
    half = ess / 2.0
    split_idx = int(np.searchsorted(cum_w, half))
    split_idx = max(1, min(split_idx, len(x) - 1))

    x_left, w_left = x_s[:split_idx], w_s[:split_idx]
    x_right, w_right = x_s[split_idx:], w_s[split_idx:]

    mean1 = float(np.average(x_left, weights=w_left))
    mean2 = float(np.average(x_right, weights=w_right))
    if np.isnan(mean2):
        mean2 = mean1 + 1.0

    means = np.array([mean1, mean2])
    variances = np.array(
        [
            max(float(np.average((x_left - mean1) ** 2, weights=w_left)), 1e-6),
            max(float(np.average((x_right - mean2) ** 2, weights=w_right)), 1e-6),
        ]
    )
    sw_left = float(np.sum(w_left))
    mix_weights = np.array(
        [
            max(sw_left / ess, 0.05),
            max(1.0 - sw_left / ess, 0.05),
        ]
    )
    mix_weights /= mix_weights.sum()

    prev_ll = -np.inf

    for _iter in range(max_iter):
        # E-step (log-space)
        log_resp = np.empty((len(x), 2))
        for k in range(2):
            log_resp[:, k] = np.log(mix_weights[k]) + _log_gaussian_pdf(
                x, means[k], variances[k]
            )

        log_prob = logsumexp(log_resp, axis=1)
        weighted_ll = float(np.sum(weights * log_prob))

        # Convergence check
        if abs(weighted_ll - prev_ll) < tol * (abs(weighted_ll) + 1e-10):
            break
        prev_ll = weighted_ll

        # Responsibilities in linear space
        log_resp -= log_prob[:, np.newaxis]  # normalise
        resp = np.exp(log_resp)
        resp = np.clip(resp, 0.0, 1.0)

        # M-step (weighted)
        for k in range(2):
            Nk = float(np.sum(weights * resp[:, k]))
            if Nk > 1e-10:
                means[k] = float(np.sum(weights * resp[:, k] * x) / Nk)
                diff = x - means[k]
                variances[k] = float(np.sum(weights * resp[:, k] * diff**2) / Nk)
                variances[k] = max(variances[k], 1e-6)
            else:
                variances[k] = 1e-6
            mix_weights[k] = Nk / ess

        mix_weights /= mix_weights.sum()

    else:
        # Did not converge — log warning
        print(
            f"Warning: EM did not converge after {max_iter} iterations.",
            file=sys.stderr,
        )

    # Degeneracy check: at least 1 effective observation per component
    for k in range(2):
        if mix_weights[k] * ess < 1.0:
            return means, variances, mix_weights, float("nan")

    return means, variances, mix_weights, weighted_ll


def _compute_bic(weighted_ll: float, n_params: int, ess: float) -> float:
    """Compute BIC from weighted log-likelihood.

    Args:
        weighted_ll: Weighted log-likelihood.
        n_params: Number of free parameters.
        ess: Effective sample size (sum of weights).

    Returns:
        BIC value, or ``nan`` if *weighted_ll* is ``nan``.
    """
    if np.isnan(weighted_ll):
        return float("nan")
    return -2.0 * weighted_ll + n_params * np.log(ess)


def _find_peaks_kde(
    x: np.ndarray,
    weights: np.ndarray,
    prominence: float = 0.05,
    n_grid: int = 512,
) -> int:
    """Count density peaks via weighted KDE.

    Args:
        x: 1-D data values.
        weights: Observation weights.
        prominence: Minimum prominence for ``scipy.signal.find_peaks``.
        n_grid: Number of evaluation grid points.

    Returns:
        Number of peaks detected, or 0 when data is insufficient.
    """
    if len(x) < 2 or np.sum(weights) <= 0:
        return 0

    if np.std(x) < 1e-10:
        return 1  # all identical → single peak

    kde = gaussian_kde(x, weights=weights)

    lo, hi = float(np.min(x)), float(np.max(x))
    pad = 0.2 * (hi - lo) if hi > lo else 1.0
    grid = np.linspace(lo - pad, hi + pad, n_grid)
    density = kde.evaluate(grid)

    peaks, _props = find_peaks(density, prominence=prominence)
    return int(len(peaks))


# ---------------------------------------------------------------------------
# Per-feature processing
# ---------------------------------------------------------------------------


def _process_feature(
    feature_id: str,
    id_type: str,
    probs: np.ndarray,
    pa_lens: np.ndarray,
    min_length: float,
    min_asp: float,
    min_ess: float,
    kde_prominence: float,
) -> dict | None:
    """Run the bimodality pipeline for a single transcript or gene.

    Args:
        feature_id: Transcript or gene identifier.
        id_type: ``"transcript_id"`` or ``"gene_id"``.
        probs: Array of assignment probabilities.
        pa_lens: Array of poly(A) tail lengths (same length as *probs*).
        min_length: Minimum poly(A) length threshold.
        min_asp: Minimum assignment probability threshold.
        min_ess: Minimum effective sample size.
        kde_prominence: Prominence threshold for KDE peak detection.

    Returns:
        A dict with result columns, or ``None`` if the feature should
        be skipped (insufficient data after filtering).
    """
    n_reads_raw = len(pa_lens)

    # Filtering
    mask = (pa_lens >= min_length) & (probs >= min_asp)
    n_reads_filtered = int(np.sum(mask))
    if n_reads_filtered == 0:
        return None

    probs_f = probs[mask].astype(np.float64)
    pa_lens_f = pa_lens[mask].astype(np.float64)

    ess = float(np.sum(probs_f))
    if ess < min_ess:
        return None

    # Variance-stabilising log-transform
    x_log = np.log(pa_lens_f + 1.0)

    # GMM k=1 (always computable)
    _means1, _vars1, _mix1, ll_k1 = _fit_weighted_gmm_1d(x_log, probs_f, n_components=1)
    bic_k1 = _compute_bic(ll_k1, n_params=2, ess=ess)

    # GMM k=2
    _means2, _vars2, _mix2, ll_k2 = _fit_weighted_gmm_1d(x_log, probs_f, n_components=2)
    bic_k2 = _compute_bic(ll_k2, n_params=5, ess=ess)

    delta_bic = bic_k1 - bic_k2  # positive = evidence for k=2
    bimodal_gmm = bool((not np.isnan(delta_bic)) and delta_bic > 10.0)

    # KDE peak detection
    n_kde_peaks = _find_peaks_kde(x_log, probs_f, prominence=kde_prominence)
    bimodal_kde = n_kde_peaks >= 2

    return {
        "feature_id": feature_id,
        "id_type": id_type,
        "n_reads_raw": n_reads_raw,
        "n_reads_filtered": n_reads_filtered,
        "ess": ess,
        "delta_bic": delta_bic,
        "bic_k1": bic_k1,
        "bic_k2": bic_k2,
        "ll_k1": ll_k1,
        "ll_k2": ll_k2,
        "n_kde_peaks": n_kde_peaks,
        "bimodal_gmm": bimodal_gmm,
        "bimodal_kde": bimodal_kde,
        "bimodal_call": bimodal_gmm and bimodal_kde,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace | None = None) -> None:
    """Detect bimodal poly(A) tail length distributions.

    Reads a poly(A) TSV file (from ``polya_calc`` or ``polya_t2g``),
    applies weighted GMM and KDE bimodality tests to each feature, and
    writes a consensus bimodality call table.
    """
    if args is None:
        args = parse_args()

    id_col_name, data_dict = parse_polyA_file(args.input)

    if not data_dict:
        print("No features found in input file.", file=sys.stderr)
        sys.exit(0)

    print(
        f"Testing {len(data_dict)} features for bimodal poly(A) distributions...",
        file=sys.stderr,
    )

    results: list[dict] = []
    for feat_id in sorted(data_dict.keys()):
        d = data_dict[feat_id]
        row = _process_feature(
            feature_id=feat_id,
            id_type=id_col_name,
            probs=d["probs"],
            pa_lens=d["pa_lens"],
            min_length=args.min_length,
            min_asp=args.min_asp,
            min_ess=args.min_ess,
            kde_prominence=args.kde_prominence,
        )
        if row is not None:
            results.append(row)

    if not results:
        print(
            "No features passed the filtering thresholds.",
            file=sys.stderr,
        )
        sys.exit(0)

    # Write output
    output_filename = ensure_gz_suffix(args.output, args.gzip)

    print(
        f"Writing bimodality results to {output_filename}...",
        file=sys.stderr,
    )

    write_mode = "wt" if output_filename.endswith(".gz") else "w"
    with open_by_suffix(output_filename, write_mode) as out_f:
        out_f.write(
            "feature_id\tid_type\tn_reads_raw\tn_reads_filtered\t"
            "ess\tdelta_bic\tbic_k1\tbic_k2\tll_k1\tll_k2\t"
            "n_kde_peaks\tbimodal_gmm\tbimodal_kde\tbimodal_call\n"
        )

        def _fmt(v, fmt_str=".6f"):
            if isinstance(v, float) and np.isnan(v):
                return "NA"
            if isinstance(v, bool):
                return str(v)
            if isinstance(v, float):
                return f"{v:{fmt_str}}"
            return str(v)

        for row in results:
            out_f.write(
                f"{row['feature_id']}\t"
                f"{row['id_type']}\t"
                f"{row['n_reads_raw']}\t"
                f"{row['n_reads_filtered']}\t"
                f"{_fmt(row['ess'], '.2f')}\t"
                f"{_fmt(row['delta_bic'], '.4f')}\t"
                f"{_fmt(row['bic_k1'], '.2f')}\t"
                f"{_fmt(row['bic_k2'], '.2f')}\t"
                f"{_fmt(row['ll_k1'], '.4f')}\t"
                f"{_fmt(row['ll_k2'], '.4f')}\t"
                f"{row['n_kde_peaks']}\t"
                f"{row['bimodal_gmm']}\t"
                f"{row['bimodal_kde']}\t"
                f"{row['bimodal_call']}\n"
            )

    n_bimodal = sum(1 for r in results if r["bimodal_call"])
    print(
        f"Bimodality analysis complete.  "
        f"{n_bimodal} / {len(results)} features called bimodal.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
