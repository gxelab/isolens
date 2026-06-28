"""Shared statistical testing backend for poly(A) distribution comparison.

Provides weighted two-sample tests for comparing poly(A) length
distributions between conditions or isoforms:

- :func:`weighted_ks_test` — difference in **shape** (distribution)
- :func:`weighted_t_test` — difference in **mean** (location)
- :func:`weighted_rank_sum_test` — difference in **median** (location)

All functions accept 1-D *values* and *weights* arrays and return
``(statistic, p_value)``, returning ``(nan, nan)`` for degenerate
cases (zero total weight, insufficient effective sample size, etc.).
"""

import numpy as np
from scipy.stats import kstwobign, norm
from scipy.stats import t as t_dist


def weighted_ecdf(
    values: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the weighted Empirical Cumulative Distribution Function.

    Args:
        values: 1-D array of observed values.
        weights: 1-D array of weights (same length as *values*).

    Returns:
        ``(sorted_values, cdf)`` where *cdf* runs from 0 to 1.
    """
    if np.sum(weights) <= 0:
        return np.array([]), np.array([])

    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]

    cum_weights = np.cumsum(weights)
    cdf = cum_weights / cum_weights[-1]
    return values, cdf


def weighted_ks_test(
    v1: np.ndarray,
    w1: np.ndarray,
    v2: np.ndarray,
    w2: np.ndarray,
) -> tuple[float, float]:
    """Two-sample weighted KS test using Kish's effective sample sizes.

    Args:
        v1, v2: Observed values for samples 1 and 2.
        w1, w2: Weights for samples 1 and 2 (same lengths as *v1*, *v2*).

    Returns:
        ``(ks_statistic, p_value)`` where *ks_statistic* is the maximum
        absolute difference between the weighted ECDFs and *p_value* is
        computed via the Kolmogorov distribution with effective sample
        sizes.  Returns ``(nan, nan)`` when either group has zero total
        weight or zero effective sample size.
    """
    sw1 = np.sum(w1)
    sw2 = np.sum(w2)
    if sw1 <= 0 or sw2 <= 0:
        return float("nan"), float("nan")

    n1_eff = sw1**2 / np.sum(w1**2)
    n2_eff = sw2**2 / np.sum(w2**2)
    if n1_eff <= 0 or n2_eff <= 0:
        return float("nan"), float("nan")

    all_vals = np.unique(np.concatenate([v1, v2]))

    _, cdf1 = weighted_ecdf(v1, w1)
    _, cdf2 = weighted_ecdf(v2, w2)

    cdf1_interp = np.interp(all_vals, v1, cdf1, left=0, right=1)
    cdf2_interp = np.interp(all_vals, v2, cdf2, left=0, right=1)

    ks_stat = np.max(np.abs(cdf1_interp - cdf2_interp))

    en = np.sqrt((n1_eff * n2_eff) / (n1_eff + n2_eff))
    p_val = kstwobign.sf(ks_stat * (en + 0.12 + 0.11 / en))

    return float(ks_stat), float(min(1.0, max(0.0, p_val)))


def weighted_t_test(
    v1: np.ndarray,
    w1: np.ndarray,
    v2: np.ndarray,
    w2: np.ndarray,
) -> tuple[float, float]:
    """Weighted Welch's t-test for difference in means.

    Uses Kish's effective sample size and the Welch-Satterthwaite
    approximation for degrees of freedom.

    Args:
        v1, v2: Observed values for samples 1 and 2.
        w1, w2: Weights for samples 1 and 2 (same lengths as *v1*, *v2*).

    Returns:
        ``(t_statistic, p_value)``.  Returns ``(nan, nan)`` when either
        group has zero total weight, fewer than 2 effective observations,
        or zero variance.
    """
    sw1 = np.sum(w1)
    sw2 = np.sum(w2)
    if sw1 <= 0 or sw2 <= 0:
        return float("nan"), float("nan")

    n1_eff = sw1**2 / np.sum(w1**2)
    n2_eff = sw2**2 / np.sum(w2**2)
    if n1_eff < 2 or n2_eff < 2:
        return float("nan"), float("nan")

    # Weighted means
    mean1 = float(np.average(v1, weights=w1))
    mean2 = float(np.average(v2, weights=w2))

    # Unbiased weighted variance (reliability weights correction)
    var1 = np.average((v1 - mean1) ** 2, weights=w1) * n1_eff / (n1_eff - 1)
    var2 = np.average((v2 - mean2) ** 2, weights=w2) * n2_eff / (n2_eff - 1)

    if var1 <= 0 or var2 <= 0:
        return float("nan"), float("nan")

    # Standard error of the difference in means
    se = np.sqrt(var1 / n1_eff + var2 / n2_eff)
    if se <= 0:
        return float("nan"), float("nan")

    t_stat = (mean1 - mean2) / se

    # Welch-Satterthwaite degrees of freedom
    vn1 = var1 / n1_eff
    vn2 = var2 / n2_eff
    num = (vn1 + vn2) ** 2
    denom = vn1**2 / (n1_eff - 1) + vn2**2 / (n2_eff - 1)
    df = num / denom

    if df <= 0 or np.isnan(df):
        return float("nan"), float("nan")

    p_val = 2.0 * t_dist.sf(abs(t_stat), df)
    return float(t_stat), float(min(1.0, max(0.0, p_val)))


def weighted_rank_sum_test(
    v1: np.ndarray,
    w1: np.ndarray,
    v2: np.ndarray,
    w2: np.ndarray,
) -> tuple[float, float]:
    """Weighted Mann-Whitney U test (rank-sum) for difference in location.

    Uses fractional ranks with tie correction and a normal approximation
    with Kish effective sample sizes.

    Args:
        v1, v2: Observed values for samples 1 and 2.
        w1, w2: Weights for samples 1 and 2 (same lengths as *v1*, *v2*).

    Returns:
        ``(z_statistic, p_value)``.  Returns ``(nan, nan)`` when either
        group has zero total weight or zero effective sample size.
    """
    sw1 = np.sum(w1)
    sw2 = np.sum(w2)
    if sw1 <= 0 or sw2 <= 0:
        return float("nan"), float("nan")

    n1_eff = sw1**2 / np.sum(w1**2)
    n2_eff = sw2**2 / np.sum(w2**2)
    if n1_eff <= 0 or n2_eff <= 0:
        return float("nan"), float("nan")

    # Combine and sort
    vals = np.concatenate([v1, v2])
    weights = np.concatenate([w1, w2])
    group = np.concatenate([np.zeros(len(v1)), np.ones(len(v2))])

    sorter = np.argsort(vals)
    vals = vals[sorter]
    weights = weights[sorter]
    group = group[sorter]

    # Fractional ranks with tie correction (average ranks)
    n = len(vals)
    ranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j < n and vals[j] == vals[i]:
            j += 1
        # Average rank for tied group (1-based)
        mean_rank = (i + j - 1) / 2.0 + 1.0
        ranks[i:j] = mean_rank
        i = j

    # Weighted rank sum for group 1
    g1_mask = group == 0
    w_rank_sum = np.sum(weights[g1_mask] * ranks[g1_mask])

    # Expected rank sum under null
    n_eff = n1_eff + n2_eff
    expected = n1_eff * (n_eff + 1.0) / 2.0

    # Variance with tie correction based on nominal tie groups
    tie_sum = 0.0
    i = 0
    while i < n:
        j = i
        while j < n and vals[j] == vals[i]:
            j += 1
        t = j - i  # tie group size
        if t > 1:
            tie_sum += float(t**3 - t)
        i = j

    tie_correction = 1.0
    if n_eff > 1:
        tie_correction = 1.0 - tie_sum / (n_eff**3 - n_eff)

    var_wrs = n1_eff * n2_eff / 12.0 * (n_eff + 1.0) * tie_correction

    if var_wrs <= 0:
        return float("nan"), float("nan")

    z_stat = (w_rank_sum - expected) / np.sqrt(var_wrs)

    p_val = 2.0 * norm.sf(abs(z_stat))
    return float(z_stat), float(min(1.0, max(0.0, p_val)))
