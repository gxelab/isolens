"""Shared statistics backend for isolens differential modification modules.

Provides weighted logistic regression for a single binary predictor
and Benjamini-Hochberg FDR correction.  Used by ``mod_dmc`` and ``mod_dmt``.
"""

import numpy as np
from scipy.stats import norm as _norm_dist


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
