"""Tests for the shared _stats.py statistical testing backend."""

import os
import sys

import numpy as np
import pytest

try:
    from isolens._stats import (
        weighted_ecdf,
        weighted_ks_test,
        weighted_rank_sum_test,
        weighted_t_test,
    )
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from _stats import (  # type: ignore[no-redef]
        weighted_ecdf,
        weighted_ks_test,
        weighted_rank_sum_test,
        weighted_t_test,
    )


class TestWeightedEcdf:
    """Tests for weighted_ecdf()."""

    def test_uniform_weights(self):
        vals = np.array([1.0, 2.0, 3.0])
        wts = np.array([1.0, 1.0, 1.0])
        sv, cdf = weighted_ecdf(vals, wts)
        assert len(sv) == 3
        assert cdf[-1] == pytest.approx(1.0)
        assert cdf[0] == pytest.approx(1.0 / 3.0)

    def test_varying_weights(self):
        vals = np.array([1.0, 2.0])
        wts = np.array([0.9, 0.1])
        sv, cdf = weighted_ecdf(vals, wts)
        assert cdf[0] == pytest.approx(0.9)

    def test_single_element(self):
        vals = np.array([5.0])
        wts = np.array([3.0])
        sv, cdf = weighted_ecdf(vals, wts)
        assert len(sv) == 1
        assert cdf[-1] == pytest.approx(1.0)

    def test_zero_total_weights(self):
        vals = np.array([1.0, 2.0, 3.0])
        wts = np.array([0.0, 0.0, 0.0])
        sv, cdf = weighted_ecdf(vals, wts)
        assert len(sv) == 0
        assert len(cdf) == 0

    def test_empty_arrays(self):
        vals = np.array([])
        wts = np.array([])
        sv, cdf = weighted_ecdf(vals, wts)
        assert len(sv) == 0


class TestWeightedKsTest:
    """Tests for weighted_ks_test()."""

    def test_identical_distributions(self):
        v = np.array([1.0, 2.0, 3.0])
        w = np.array([1.0, 1.0, 1.0])
        stat, p = weighted_ks_test(v, w, v, w)
        assert stat == pytest.approx(0.0, abs=1e-9)
        assert p == pytest.approx(1.0, abs=0.1)

    def test_shifted_distributions(self):
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([4.0, 5.0, 6.0])
        w = np.array([1.0, 1.0, 1.0])
        stat, p = weighted_ks_test(v1, w, v2, w)
        assert stat > 0.5
        assert p < 0.5

    def test_single_element(self):
        stat, p = weighted_ks_test(
            np.array([1.0]),
            np.array([1.0]),
            np.array([5.0]),
            np.array([1.0]),
        )
        assert stat == pytest.approx(1.0)
        assert 0.0 <= p <= 1.0

    def test_overlapping_distributions(self):
        v1 = np.arange(1, 6, dtype=float)
        v2 = np.arange(2, 7, dtype=float)
        w = np.ones(5)
        stat, p = weighted_ks_test(v1, w, v2, w)
        assert 0.0 < stat < 1.0
        assert 0.0 <= p <= 1.0

    def test_zero_weights_in_one(self):
        v1 = np.array([1.0, 2.0])
        v2 = np.array([3.0, 4.0])
        stat, p = weighted_ks_test(v1, np.ones(2), v2, np.zeros(2))
        assert np.isnan(stat)
        assert np.isnan(p)

    def test_zero_weights_both(self):
        v = np.array([1.0, 2.0])
        w = np.zeros(2)
        stat, p = weighted_ks_test(v, w, v, w)
        assert np.isnan(stat)
        assert np.isnan(p)


class TestWeightedTTest:
    """Tests for weighted_t_test()."""

    def test_identical_means(self):
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        w = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        t_stat, p = weighted_t_test(v, w, v, w)
        assert t_stat == pytest.approx(0.0, abs=1e-9)
        assert p == pytest.approx(1.0, abs=0.1)

    def test_shifted_means(self):
        v1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        v2 = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        w = np.ones(5)
        t_stat, p = weighted_t_test(v1, w, v2, w)
        assert abs(t_stat) > 5.0
        assert p < 0.01

    def test_uniform_weights(self):
        v1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        v2 = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        w = np.ones(5)
        t_stat, p = weighted_t_test(v1, w, v2, w)
        assert abs(t_stat) > 0.0
        assert 0.0 <= p <= 1.0

    def test_weighted_effect(self):
        v1 = np.array([1.0, 100.0])
        v2 = np.array([50.0, 60.0])
        # Equal weighting: mean is pulled toward 100
        w_eq = np.ones(2)
        # Group 1: weight heavily on 1.0 → mean near 1
        w_skew1 = np.array([0.99, 0.01])
        t_eq, _ = weighted_t_test(v1, w_eq, v2, w_eq)
        t_skew, _ = weighted_t_test(v1, w_skew1, v2, w_eq)
        # The skewed weighting changes the test statistic
        assert t_eq != pytest.approx(t_skew)

    def test_zero_weights_in_one(self):
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([4.0, 5.0, 6.0])
        t_stat, p = weighted_t_test(v1, np.ones(3), v2, np.zeros(3))
        assert np.isnan(t_stat)
        assert np.isnan(p)

    def test_insufficient_effective_size(self):
        v = np.array([1.0])
        w = np.array([1.0])
        t_stat, p = weighted_t_test(v, w, v, w)
        assert np.isnan(t_stat)
        assert np.isnan(p)

    def test_all_same_values_one_group(self):
        v1 = np.array([1.0, 1.0, 1.0])
        v2 = np.array([1.0, 2.0, 3.0])
        w = np.ones(3)
        t_stat, p = weighted_t_test(v1, w, v2, w)
        assert np.isnan(t_stat)
        assert np.isnan(p)

    def test_direction_positive(self):
        v1 = np.array([5.0, 6.0, 7.0])
        v2 = np.array([1.0, 2.0, 3.0])
        w = np.ones(3)
        t_stat, _ = weighted_t_test(v1, w, v2, w)
        assert t_stat > 0.0

    def test_direction_negative(self):
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([5.0, 6.0, 7.0])
        w = np.ones(3)
        t_stat, _ = weighted_t_test(v1, w, v2, w)
        assert t_stat < 0.0

    def test_symmetry(self):
        v1 = np.array([1.0, 2.0, 3.0, 4.0])
        v2 = np.array([2.0, 3.0, 5.0, 6.0])
        w1 = np.array([0.5, 0.8, 1.0, 0.3])
        w2 = np.array([0.7, 0.9, 0.4, 1.0])
        t1, p1 = weighted_t_test(v1, w1, v2, w2)
        t2, p2 = weighted_t_test(v2, w2, v1, w1)
        assert t1 == pytest.approx(-t2)
        assert p1 == pytest.approx(p2)


class TestWeightedRankSumTest:
    """Tests for weighted_rank_sum_test()."""

    def test_identical_distributions(self):
        v = np.array([1.0, 2.0, 3.0])
        w = np.array([1.0, 1.0, 1.0])
        z_stat, p = weighted_rank_sum_test(v, w, v, w)
        assert z_stat == pytest.approx(0.0, abs=1e-9)
        assert p == pytest.approx(1.0, abs=0.1)

    def test_shifted_distributions(self):
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([4.0, 5.0, 6.0])
        w = np.ones(3)
        z_stat, p = weighted_rank_sum_test(v1, w, v2, w)
        assert abs(z_stat) > 1.0
        assert p < 0.5

    def test_uniform_weights(self):
        v1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        v2 = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        w = np.ones(5)
        z_stat, p = weighted_rank_sum_test(v1, w, v2, w)
        assert abs(z_stat) > 0.0
        assert 0.0 <= p <= 1.0

    def test_weighted_effect(self):
        v1 = np.array([1.0, 2.0, 100.0])
        v2 = np.array([3.0, 4.0, 5.0])
        w_eq = np.ones(3)
        w_skew = np.array([0.0, 0.0, 1.0])  # only the extreme value
        z_eq, _ = weighted_rank_sum_test(v1, w_eq, v2, w_eq)
        z_skew, _ = weighted_rank_sum_test(v1, w_skew, v2, w_eq)
        # Different weighting changes the statistic
        assert z_eq != pytest.approx(z_skew)

    def test_zero_weights_in_one(self):
        v1 = np.array([1.0, 2.0])
        v2 = np.array([3.0, 4.0])
        z_stat, p = weighted_rank_sum_test(
            v1,
            np.ones(2),
            v2,
            np.zeros(2),
        )
        assert np.isnan(z_stat)
        assert np.isnan(p)

    def test_all_same_values(self):
        v1 = np.array([5.0, 5.0, 5.0])
        v2 = np.array([5.0, 5.0, 5.0])
        w = np.ones(3)
        # All ties → zero variance after tie correction
        z_stat, p = weighted_rank_sum_test(v1, w, v2, w)
        # With all ties, variance collapses: return nan
        assert np.isnan(z_stat)
        assert np.isnan(p)

    def test_single_element_per_group(self):
        v1 = np.array([10.0])
        v2 = np.array([20.0])
        w = np.array([1.0])
        z_stat, p = weighted_rank_sum_test(v1, w, v2, w)
        assert 0.0 <= p <= 1.0

    def test_direction_positive(self):
        v1 = np.array([5.0, 6.0, 7.0])
        v2 = np.array([1.0, 2.0, 3.0])
        w = np.ones(3)
        z_stat, _ = weighted_rank_sum_test(v1, w, v2, w)
        assert z_stat > 0.0

    def test_direction_negative(self):
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([5.0, 6.0, 7.0])
        w = np.ones(3)
        z_stat, _ = weighted_rank_sum_test(v1, w, v2, w)
        assert z_stat < 0.0

    def test_ties(self):
        v1 = np.array([1.0, 1.0, 3.0, 5.0])
        v2 = np.array([1.0, 2.0, 4.0, 6.0])
        w = np.ones(4)
        z_stat, p = weighted_rank_sum_test(v1, w, v2, w)
        assert not np.isnan(z_stat)
        assert 0.0 <= p <= 1.0
