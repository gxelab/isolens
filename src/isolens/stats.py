"""Re-export weighted two-sample tests from ``_stats`` for backward compatibility.

Prefer importing directly from ``isolens._stats`` in new code.
"""

from isolens._stats import (  # noqa: F401
    weighted_ecdf,
    weighted_ks_test,
    weighted_rank_sum_test,
    weighted_t_test,
)
