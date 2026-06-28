"""Tests for polya_diff — weighted KS test for poly(A) comparisons."""

import gzip
import os
import sys

import numpy as np
import pytest

try:
    from isolens.polya_diff import (
        parse_polyA_file,
        weighted_ecdf,
        weighted_ks_test,
    )
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from isolens.polya_diff import (  # type: ignore[no-redef]
        parse_polyA_file,
        weighted_ecdf,
        weighted_ks_test,
    )


# ---------- weighted_ecdf ----------


class TestWeightedEcdf:
    """Tests for weighted_ecdf()."""

    def test_uniform_weights(self):
        """Uniform weights → standard ECDF."""
        vals = np.array([1.0, 2.0, 3.0])
        w = np.array([1.0, 1.0, 1.0])
        sorted_vals, cdf = weighted_ecdf(vals, w)
        assert cdf[-1] == pytest.approx(1.0)
        assert cdf[0] == pytest.approx(1.0 / 3.0)

    def test_varying_weights(self):
        """Varying weights → weighted CDF."""
        vals = np.array([1.0, 2.0])
        w = np.array([0.9, 0.1])
        sorted_vals, cdf = weighted_ecdf(vals, w)
        assert sorted_vals[0] == 1.0
        assert cdf[0] == pytest.approx(0.9)

    def test_single_element(self):
        """Single element should work."""
        vals = np.array([5.0])
        w = np.array([3.0])
        sorted_vals, cdf = weighted_ecdf(vals, w)
        assert cdf[-1] == pytest.approx(1.0)
        assert len(cdf) == 1


# ---------- weighted_ks_test ----------


class TestWeightedKsTest:
    """Tests for weighted_ks_test()."""

    def test_identical_distributions(self):
        """Same values, same weights → KS ≈ 0."""
        v1 = np.array([1.0, 2.0, 3.0])
        w1 = np.array([1.0, 1.0, 1.0])
        stat, p_val = weighted_ks_test(v1, w1, v1, w1)
        assert stat == pytest.approx(0.0, abs=1e-10)
        assert p_val == pytest.approx(1.0, abs=0.1)

    def test_shifted_distributions(self):
        """Shifted values → KS > 0."""
        v1 = np.array([1.0, 2.0, 3.0])
        w1 = np.array([1.0, 1.0, 1.0])
        v2 = np.array([4.0, 5.0, 6.0])
        w2 = np.array([1.0, 1.0, 1.0])
        stat, p_val = weighted_ks_test(v1, w1, v2, w2)
        assert stat > 0.5
        assert p_val < 0.5

    def test_single_element(self):
        """Single-element inputs."""
        v1 = np.array([1.0])
        w1 = np.array([1.0])
        v2 = np.array([5.0])
        w2 = np.array([1.0])
        stat, p_val = weighted_ks_test(v1, w1, v2, w2)
        assert stat == pytest.approx(1.0)
        # p should be valid [0, 1]
        assert 0.0 <= p_val <= 1.0

    def test_overlapping_distributions(self):
        """Overlapping distributions with deterministic data → moderate KS."""
        v1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        v2 = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        w1 = np.ones(5)
        w2 = np.ones(5)
        stat, p_val = weighted_ks_test(v1, w1, v2, w2)
        # Overlapping: only the first and last elements differ
        assert 0.0 < stat < 1.0
        assert 0.0 <= p_val <= 1.0


# ---------- parse_polyA_file ----------


class TestParsePolyaFile:
    """Tests for parse_polyA_file()."""

    def test_valid_tsv(self, tmp_path):
        path = tmp_path / "test.tsv"
        path.write_text(
            "transcript_id\ttx_idx\tn_reads\tpa_wlen\tprobs\tpa_lens\n"
            "TX1\t0\t2\t100.5\t0.5,0.5\t100,101\n"
        )
        id_name, data = parse_polyA_file(str(path))
        assert id_name == "transcript_id"
        assert "TX1" in data
        assert data["TX1"]["n_reads"] == 2
        assert len(data["TX1"]["probs"]) == 2
        np.testing.assert_array_equal(data["TX1"]["pa_lens"], np.array([100, 101]))

    def test_gene_level_tsv(self, tmp_path):
        path = tmp_path / "gene.tsv"
        path.write_text(
            "gene_id\tn_reads\tpa_wlen\tprobs\tpa_lens\nGENE1\t1\t42.0\t1.0\t42\n"
        )
        id_name, data = parse_polyA_file(str(path))
        assert id_name == "gene_id"
        assert "GENE1" in data

    def test_gzipped_tsv(self, tmp_path):
        path = tmp_path / "test.tsv.gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(
                "transcript_id\ttx_idx\tn_reads\tpa_wlen\tprobs\tpa_lens\n"
                "TX1\t0\t1\t100.0\t1.0\t100\n"
            )
        id_name, data = parse_polyA_file(str(path))
        assert "TX1" in data

    def test_empty_file_after_header(self, tmp_path):
        path = tmp_path / "empty.tsv"
        path.write_text("transcript_id\ttx_idx\tn_reads\tpa_wlen\tprobs\tpa_lens\n")
        _, data = parse_polyA_file(str(path))
        assert data == {}
