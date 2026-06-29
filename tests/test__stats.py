"""Tests for _stats — shared statistics backend."""

import os
import sys

import numpy as np
import pytest

try:
    from isolens._stats import bh_fdr, weighted_logistic_test
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from isolens._stats import (  # type: ignore[no-redef]
        bh_fdr,
        weighted_logistic_test,
    )


# ---------- weighted_logistic_test ----------


class TestWeightedLogisticTest:
    """Tests for weighted_logistic_test()."""

    def test_balanced_no_effect(self):
        """Equal weighted proportions → log2OR ≈ 0, p ≈ 1."""
        y = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64)
        x = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
        w = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)

        result = weighted_logistic_test(y, x, w)
        assert result["log2_or"] == pytest.approx(0.0, abs=1e-6)
        assert result["p_value"] == pytest.approx(1.0, abs=0.01)
        assert result["beta1"] == pytest.approx(0.0, abs=1e-6)

    def test_clear_effect(self):
        """Different weighted proportions → non-zero log2OR, small p."""
        # 20 reads per group for adequate power with Haldane correction
        n = 20
        y = np.array([0.0] * n + [1.0] * n, dtype=np.float64)
        x = np.array([0.0] * n + [1.0] * n, dtype=np.float64)
        w = np.ones(2 * n, dtype=np.float64)

        result = weighted_logistic_test(y, x, w)

        # With enough data, log2OR should be large and p-value small
        assert result["log2_or"] > 3.0
        assert result["p_value"] < 0.01

    def test_weighted_effect(self):
        """Non-uniform weights produce correct weighted estimates."""
        # Use enough reads per group so the weighted effect is significant
        n = 20
        y = np.array([0.0] * n + [1.0] * n, dtype=np.float64)
        x = np.array([0.0] * n + [1.0] * n, dtype=np.float64)
        # Group 0 has low weights, group 1 high weights
        w = np.array([0.1] * n + [2.0] * n, dtype=np.float64)

        result = weighted_logistic_test(y, x, w)

        # Group 1 has much higher weighted modification
        assert result["log2_or"] > 2.0
        assert result["p_value"] < 0.05

    def test_zero_total_weight_group0(self):
        """Zero total weight in group 0 → NaN results."""
        y = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
        x = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
        w = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)

        result = weighted_logistic_test(y, x, w)
        assert np.isnan(result["log2_or"])
        assert np.isnan(result["p_value"])
        assert np.isnan(result["beta0"])
        assert np.isnan(result["beta1"])

    def test_zero_total_weight_both_groups(self):
        """Both groups have zero total weight → NaN results."""
        y = np.array([1.0, 0.0], dtype=np.float64)
        x = np.array([0.0, 1.0], dtype=np.float64)
        w = np.array([0.0, 0.0], dtype=np.float64)

        result = weighted_logistic_test(y, x, w)
        assert np.isnan(result["log2_or"])
        assert np.isnan(result["p_value"])

    def test_all_modified_in_both_groups(self):
        """Haldane correction prevents infinite log-odds."""
        y = np.ones(10, dtype=np.float64)
        x = np.array([0.0] * 5 + [1.0] * 5, dtype=np.float64)
        w = np.ones(10, dtype=np.float64)

        result = weighted_logistic_test(y, x, w)
        # Both groups all-modified → log2OR should be near 0
        assert abs(result["log2_or"]) < 0.5
        # Not a significant difference
        assert result["p_value"] > 0.5

    def test_all_unmodified_in_both_groups(self):
        """Haldane correction when no modifications in either group."""
        y = np.zeros(10, dtype=np.float64)
        x = np.array([0.0] * 5 + [1.0] * 5, dtype=np.float64)
        w = np.ones(10, dtype=np.float64)

        result = weighted_logistic_test(y, x, w)
        assert abs(result["log2_or"]) < 0.5
        assert result["p_value"] > 0.5

    def test_extreme_effect_with_haldane(self):
        """All modified in one group, all unmodified in other → large log2OR."""
        y = np.array([0.0] * 5 + [1.0] * 5, dtype=np.float64)
        x = np.array([0.0] * 5 + [1.0] * 5, dtype=np.float64)
        w = np.ones(10, dtype=np.float64)

        result = weighted_logistic_test(y, x, w)
        # With Haldane: p0≈0.5/6≈0.083, p1≈5.5/6≈0.917
        # log2OR ≈ log2(0.917/0.083 * 0.917/0.083) ≈ log2(121) ≈ 6.9
        assert result["log2_or"] > 4.0
        assert result["p_value"] < 0.05

    def test_empty_input(self):
        """Empty input arrays — should handle gracefully."""
        y = np.array([], dtype=np.float64)
        x = np.array([], dtype=np.float64)
        w = np.array([], dtype=np.float64)

        result = weighted_logistic_test(y, x, w)
        # Both groups have zero total weight
        assert np.isnan(result["log2_or"])

    def test_single_read_per_group(self):
        """One read per group with clear difference."""
        y = np.array([0.0, 1.0], dtype=np.float64)
        x = np.array([0.0, 1.0], dtype=np.float64)
        w = np.array([1.0, 1.0], dtype=np.float64)

        result = weighted_logistic_test(y, x, w)
        # With Haldane: p0≈0.5/2=0.25, p1≈1.5/2=0.75, log2OR≈log2(9)≈3.17
        assert result["log2_or"] > 0.5
        # Small sample → large p-value
        assert result["p_value"] > 0.1

    def test_returns_all_keys(self):
        """Result dict contains all expected keys."""
        y = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64)
        x = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
        w = np.ones(4, dtype=np.float64)

        result = weighted_logistic_test(y, x, w)
        assert set(result.keys()) == {
            "log2_or",
            "p_value",
            "beta0",
            "beta1",
            "se_beta1",
        }

    def test_log2_or_sign_matches_direction(self):
        """log2_or > 0 when group 1 has higher weighted modification."""
        y = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
        x = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
        w = np.ones(4, dtype=np.float64)

        result = weighted_logistic_test(y, x, w)
        assert result["log2_or"] > 0.0

        # Reverse: group 0 now has higher modification → log2_or < 0
        x_rev = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float64)
        result_rev = weighted_logistic_test(y, x_rev, w)
        assert result_rev["log2_or"] < 0.0
        # Symmetric: log2_or_rev ≈ -log2_or
        assert result_rev["log2_or"] == pytest.approx(-result["log2_or"])


# ---------- bh_fdr ----------


class TestBhFdr:
    """Tests for Benjamini-Hochberg FDR correction."""

    def test_single_p_value(self):
        q = bh_fdr([0.01])
        assert q == pytest.approx([0.01])

    def test_multiple_sorted(self):
        p = [0.01, 0.02, 0.03]
        q = bh_fdr(p)
        assert q[0] == pytest.approx(0.03)
        assert q[1] == pytest.approx(0.03)
        assert q[2] == pytest.approx(0.03)

    def test_empty_list(self):
        assert bh_fdr([]) == []

    def test_high_p_values(self):
        """High p-values → q-values may still be <1 after FDR correction."""
        q = bh_fdr([0.5, 0.8, 0.9])
        for v in q:
            assert v == pytest.approx(0.9)

    def test_monotonic(self):
        """Q-values should be monotonic with respect to sorted p-values."""
        import random

        p = [random.uniform(0.001, 0.1) for _ in range(20)]
        q = bh_fdr(p)
        # After sorting by p-value, q-values should be non-decreasing
        sorted_idx = np.argsort(p)
        sorted_q = [q[i] for i in sorted_idx]
        for i in range(len(sorted_q) - 1):
            assert sorted_q[i] <= sorted_q[i + 1] + 1e-10

    def test_values_in_range(self):
        """All q-values should be in [0, 1]."""
        p = [0.001, 0.01, 0.05, 0.1, 0.5, 0.9]
        q = bh_fdr(p)
        for v in q:
            assert 0.0 <= v <= 1.0
