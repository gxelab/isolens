"""Shared statistics backend for isolens.

Provides:

- Weighted two-sample tests for comparing poly(A) length distributions
  between conditions or isoforms (ECDF, KS, t-test, rank-sum).
- Weighted logistic regression for a single binary predictor.
- Benjamini-Hochberg FDR correction.

Used by ``mod_dmc``, ``mod_dmt``, ``mod_dmcg``, ``mod_corr``,
``polya_dpc``, and ``polya_dpt``.
"""

import numpy as np
from scipy.stats import kstwobign, norm
from scipy.stats import norm as _norm_dist
from scipy.stats import t as t_dist


def weighted_logistic_test(
    y: np.ndarray,
    x: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    """Weighted logistic regression MLE for a single binary predictor.

    Fits :math:`\\operatorname{logit}(P(y=1)) = \\beta_0 + \\beta_1 x`
    where *x* ∈ {0, 1} with observation weights *w*.  Because the
    predictor is binary the MLE has a closed form — no iterative
    optimisation is needed.

    A Haldane-Anscombe-style correction (add 0.5 to the weighted
    modified count and 1.0 to the total weighted depth per group)
    prevents infinite log-odds when all reads in a group share the
    same state.  The Wald test supplies the two-sided *p*-value.

    Parameters
    ----------
    y : ndarray of float64, shape (n_reads,)
        Binary response: 1.0 = modified, 0.0 = unmodified.
    x : ndarray of float64, shape (n_reads,)
        Binary predictor: 0.0 or 1.0 (e.g. condition, isoform).
    weights : ndarray of float64, shape (n_reads,)
        Non-negative observation weights (assignment probabilities).

    Returns
    -------
    dict
        ``log2_or`` (:class:`float`) — :math:`\\beta_1 / \\ln(2)`.
        ``p_value`` (:class:`float`) — two-sided Wald *p*-value.
        ``beta0`` (:class:`float`) — intercept (log-odds in group 0).
        ``beta1`` (:class:`float`) — coefficient for *x* (log odds ratio).
        ``se_beta1`` (:class:`float`) — standard error of ``beta1``.
        All values are ``nan`` when either group has zero total weight.

    Notes
    -----
    The closed-form solution is exact for a single binary predictor:

    .. math::

        p_k = \\frac{\\sum_{x_i=k} w_i y_i + 0.5}
                   {\\sum_{x_i=k} w_i + 1.0}, \\quad k \\in \\{0, 1\\}

        \\beta_0 = \\operatorname{logit}(p_0)

        \\beta_1 = \\operatorname{logit}(p_1) - \\operatorname{logit}(p_0)

        \\operatorname{Var}(\\beta_1) =
            \\frac{1}{p_0(1-p_0)W_0} + \\frac{1}{p_1(1-p_1)W_1}

    where :math:`W_k = \\sum_{x_i=k} w_i`.
    """
    mask0 = x == 0
    mask1 = x == 1

    w0_total = weights[mask0].sum()
    w1_total = weights[mask1].sum()

    if w0_total <= 0.0 or w1_total <= 0.0:
        return {
            "log2_or": float("nan"),
            "p_value": float("nan"),
            "beta0": float("nan"),
            "beta1": float("nan"),
            "se_beta1": float("nan"),
        }

    # Weighted modified counts with Haldane-Anscombe correction
    w_mod_0 = (weights[mask0] * y[mask0]).sum() + 0.5
    w_mod_1 = (weights[mask1] * y[mask1]).sum() + 0.5
    w_total_0_adj = w0_total + 1.0
    w_total_1_adj = w1_total + 1.0

    p0 = np.clip(w_mod_0 / w_total_0_adj, 1e-10, 1.0 - 1e-10)
    p1 = np.clip(w_mod_1 / w_total_1_adj, 1e-10, 1.0 - 1e-10)

    # Log-odds
    def _logit(p: float) -> float:
        return float(np.log(p / (1.0 - p)))

    beta0 = _logit(p0)
    beta1 = _logit(p1) - _logit(p0)

    # Wald variance
    var0 = 1.0 / (p0 * (1.0 - p0) * w0_total)
    var1 = 1.0 / (p1 * (1.0 - p1) * w1_total)
    var_beta1 = var0 + var1
    se_beta1 = float(np.sqrt(var_beta1))

    # Wald test
    if se_beta1 <= 0.0:
        p_value = 1.0
    else:
        z = abs(beta1) / se_beta1
        p_value = float(2.0 * _norm_dist.sf(z))

    log2_or = beta1 / float(np.log(2.0))

    return {
        "log2_or": log2_or,
        "p_value": p_value,
        "beta0": beta0,
        "beta1": beta1,
        "se_beta1": se_beta1,
    }


def bh_fdr(p_values: list[float]) -> list[float]:
    """Apply Benjamini-Hochberg FDR correction to a list of *p*-values.

    Parameters
    ----------
    p_values : list of float
        Raw *p*-values.

    Returns
    -------
    list of float
        *q*-values (FDR-adjusted) in the same order as *p_values*.
    """
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


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute the weighted median of *values* with observation *weights*.

    Sorts values, computes the cumulative sum of the corresponding weights,
    and returns the value where the cumulative weight first reaches or
    exceeds half the total weight.

    Args:
        values: 1-D array of observed values.
        weights: 1-D array of non-negative weights (same length as *values*).

    Returns:
        The weighted median, or ``nan`` if total weight ≤ 0 or no data.
    """
    if len(values) == 0:
        return float("nan")
    total_wt = np.sum(weights)
    if total_wt <= 0:
        return float("nan")

    sorter = np.argsort(values)
    cum_wt = np.cumsum(weights[sorter])
    half = total_wt / 2.0
    idx = int(np.searchsorted(cum_wt, half))
    return float(values[sorter][idx])


# ---------------------------------------------------------------------------
# Weighted two-sample tests (poly(A) distribution comparison)
# ---------------------------------------------------------------------------


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
