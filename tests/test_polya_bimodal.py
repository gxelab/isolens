"""Tests for polya_bimodal — bimodal poly(A) distribution detection."""

import argparse
import gzip
import os
import sys

import numpy as np
import pytest
from scipy.stats import norm

try:
    from isolens.polya_bimodal import (
        _compute_bic,
        _find_peaks_kde,
        _fit_weighted_gmm_1d,
        _log_gaussian_pdf,
        _process_feature,
        main,
    )
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from polya_bimodal import (  # type: ignore[no-redef]
        _compute_bic,
        _find_peaks_kde,
        _fit_weighted_gmm_1d,
        _log_gaussian_pdf,
        _process_feature,
        main,
    )


def _make_polya_tsv(path, lines: list[str], gzip_output: bool = False):
    """Write polya TSV content to *path*, optionally gzip-compressed."""
    content = "".join(line + "\n" for line in lines)
    if gzip_output:
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(content)
    else:
        path.write_text(content)


# ---------------------------------------------------------------------------
# Unit tests for _log_gaussian_pdf
# ---------------------------------------------------------------------------


class TestLogGaussianPdf:
    """Tests for _log_gaussian_pdf()."""

    def test_against_scipy(self):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        got = _log_gaussian_pdf(x, mean=1.0, var=2.0)
        expected = norm.logpdf(x, loc=1.0, scale=np.sqrt(2.0))
        np.testing.assert_allclose(got, expected)

    def test_zero_variance_clamped(self):
        x = np.array([1.0, 2.0, 3.0])
        got = _log_gaussian_pdf(x, mean=2.0, var=0.0)
        assert not np.any(np.isnan(got))
        assert not np.any(np.isinf(got))


# ---------------------------------------------------------------------------
# Unit tests for _compute_bic
# ---------------------------------------------------------------------------


class TestComputeBic:
    """Tests for _compute_bic()."""

    def test_formula(self):
        bic = _compute_bic(weighted_ll=-50.0, n_params=2, ess=100.0)
        expected = -2.0 * (-50.0) + 2.0 * np.log(100.0)
        assert bic == pytest.approx(expected)

    def test_nan_ll(self):
        bic = _compute_bic(weighted_ll=float("nan"), n_params=5, ess=50.0)
        assert np.isnan(bic)


# ---------------------------------------------------------------------------
# Unit tests for _fit_weighted_gmm_1d
# ---------------------------------------------------------------------------


class TestFitWeightedGmm1d:
    """Tests for _fit_weighted_gmm_1d()."""

    def test_k1_identical_points(self):
        x = np.array([2.0, 2.0, 2.0, 2.0])
        w = np.ones(4)
        means, variances, mix, ll = _fit_weighted_gmm_1d(x, w, 1)
        assert means[0] == pytest.approx(2.0)
        assert mix[0] == pytest.approx(1.0)
        assert not np.isnan(ll)

    def test_k1_uniform_weights(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        w = np.ones(5)
        means, variances, mix, ll = _fit_weighted_gmm_1d(x, w, 1)
        assert means[0] == pytest.approx(3.0)
        assert not np.isnan(ll)

    def test_k1_weighted(self):
        x = np.array([1.0, 10.0])
        w = np.array([0.99, 0.01])
        means, _variances, _mix, ll = _fit_weighted_gmm_1d(x, w, 1)
        assert means[0] == pytest.approx(1.0, abs=0.2)
        assert not np.isnan(ll)

    def test_k1_zero_weights(self):
        x = np.array([1.0, 2.0])
        w = np.zeros(2)
        means, variances, mix, ll = _fit_weighted_gmm_1d(x, w, 1)
        assert np.all(np.isnan(means))

    def test_k2_well_separated_bimodal(self):
        np.random.seed(42)
        n = 200
        half = n // 2
        # Two well-separated Gaussians on log scale
        x = np.concatenate([
            np.random.normal(2.0, 0.2, half),   # short tail
            np.random.normal(5.0, 0.2, half),   # long tail
        ])
        w = np.ones(n)
        means, variances, mix, ll = _fit_weighted_gmm_1d(x, w, 2)
        # Should recover two components
        assert len(means) == 2
        assert not np.isnan(ll)
        # Means should be near 2 and 5
        sorted_means = np.sort(means)
        assert sorted_means[0] == pytest.approx(2.0, abs=0.5)
        assert sorted_means[1] == pytest.approx(5.0, abs=0.5)
        # Mixing weights should be roughly 0.5 each
        for mw in mix:
            assert mw == pytest.approx(0.5, abs=0.2)

    def test_k2_unimodal_data(self):
        np.random.seed(123)
        n = 100
        x = np.random.normal(3.0, 0.5, n)
        w = np.ones(n)
        means, variances, mix, ll = _fit_weighted_gmm_1d(x, w, 2)
        # Should not crash; may or may not be degenerate
        assert len(means) == 2
        if not np.isnan(ll):
            # Both components should be near 3
            assert np.all(np.abs(means - 3.0) < 2.0)

    def test_k2_small_sample(self):
        x = np.array([1.0, 2.0, 3.0])
        w = np.ones(3)
        means, variances, mix, ll = _fit_weighted_gmm_1d(x, w, 2)
        # May be degenerate with so few points — should not crash
        assert len(means) == 2

    def test_k2_weighted(self):
        np.random.seed(99)
        n = 300
        # 90% from one mode, 10% from another
        n1 = int(n * 0.9)
        n2 = n - n1
        x = np.concatenate([
            np.random.normal(3.0, 0.3, n1),
            np.random.normal(5.5, 0.2, n2),
        ])
        w = np.ones(n)
        means, variances, mix, ll = _fit_weighted_gmm_1d(x, w, 2)
        assert not np.isnan(ll)
        sorted_means = np.sort(means)
        assert sorted_means[0] == pytest.approx(3.0, abs=0.5)
        assert sorted_means[1] == pytest.approx(5.5, abs=0.5)
        # Minor component weight ~0.1
        assert np.min(mix) == pytest.approx(0.1, abs=0.1)


# ---------------------------------------------------------------------------
# Unit tests for _find_peaks_kde
# ---------------------------------------------------------------------------


class TestFindPeaksKde:
    """Tests for _find_peaks_kde()."""

    def test_unimodal(self):
        np.random.seed(7)
        x = np.random.normal(3.0, 0.5, 100)
        w = np.ones(100)
        n = _find_peaks_kde(x, w, prominence=0.05)
        assert n == 1

    def test_well_separated_bimodal(self):
        np.random.seed(8)
        x = np.concatenate([
            np.random.normal(2.0, 0.2, 100),
            np.random.normal(5.0, 0.2, 100),
        ])
        w = np.ones(200)
        n = _find_peaks_kde(x, w, prominence=0.05)
        assert n == 2

    def test_insufficient_data(self):
        n = _find_peaks_kde(np.array([1.0]), np.array([1.0]))
        assert n == 0

    def test_zero_weights(self):
        x = np.array([1.0, 2.0, 3.0])
        w = np.zeros(3)
        n = _find_peaks_kde(x, w)
        assert n == 0

    def test_identical_values(self):
        x = np.array([3.0, 3.0, 3.0, 3.0])
        w = np.ones(4)
        n = _find_peaks_kde(x, w)
        assert n == 1

    def test_weighted_kde(self):
        np.random.seed(11)
        x = np.concatenate([
            np.random.normal(2.0, 0.3, 100),
            np.random.normal(5.0, 0.3, 100),
        ])
        w = np.ones(200)
        n = _find_peaks_kde(x, w, prominence=0.05)
        assert n == 2


# ---------------------------------------------------------------------------
# Unit tests for _process_feature
# ---------------------------------------------------------------------------


class TestProcessFeature:
    """Tests for _process_feature()."""

    def test_below_min_ess(self):
        probs = np.array([0.5, 0.5])
        pa_lens = np.array([100, 200])
        result = _process_feature(
            "TX1", "transcript_id", probs, pa_lens,
            min_length=0.0, min_asp=0.0, min_ess=30.0, kde_prominence=0.05,
        )
        assert result is None

    def test_below_min_length(self):
        probs = np.ones(50)
        pa_lens = np.full(50, -1.0)
        result = _process_feature(
            "TX1", "transcript_id", probs, pa_lens,
            min_length=0.0, min_asp=0.0, min_ess=10.0, kde_prominence=0.05,
        )
        assert result is None

    def test_unimodal_feature(self):
        np.random.seed(13)
        n = 100
        pa_lens = np.random.normal(80, 10, n).astype(float)
        pa_lens = np.clip(pa_lens, 0, None)
        probs = np.ones(n)
        result = _process_feature(
            "TX1", "transcript_id", probs, pa_lens,
            min_length=0.0, min_asp=0.0, min_ess=30.0, kde_prominence=0.05,
        )
        assert result is not None
        assert result["feature_id"] == "TX1"
        assert result["n_reads_raw"] == n
        assert result["ess"] == pytest.approx(float(n))
        # Should not be called bimodal
        assert result["bimodal_call"] is False

    def test_bimodal_feature(self):
        np.random.seed(14)
        n = 200
        half = n // 2
        pa_lens = np.concatenate([
            np.random.normal(30, 5, half),    # short tail
            np.random.normal(150, 10, half),  # long tail
        ]).astype(float)
        pa_lens = np.clip(pa_lens, 0, None)
        probs = np.ones(n)
        result = _process_feature(
            "GENE1", "gene_id", probs, pa_lens,
            min_length=0.0, min_asp=0.0, min_ess=30.0, kde_prominence=0.03,
        )
        assert result is not None
        assert result["id_type"] == "gene_id"
        assert result["ess"] == pytest.approx(float(n))
        assert result["n_kde_peaks"] == 2
        assert result["bimodal_kde"] is True
        # Delta BIC should be strongly positive for such separated modes
        assert result["delta_bic"] > 10.0
        assert result["bimodal_gmm"] is True
        assert result["bimodal_call"] is True

    def test_all_keys_present(self):
        np.random.seed(15)
        pa_lens = np.random.normal(80, 10, 50).astype(float)
        pa_lens = np.clip(pa_lens, 0, None)
        probs = np.ones(50)
        result = _process_feature(
            "TX1", "transcript_id", probs, pa_lens,
            min_length=0.0, min_asp=0.0, min_ess=30.0, kde_prominence=0.05,
        )
        expected_keys = {
            "feature_id", "id_type", "n_reads_raw", "n_reads_filtered",
            "ess", "delta_bic", "bic_k1", "bic_k2",
            "ll_k1", "ll_k2", "n_kde_peaks",
            "bimodal_gmm", "bimodal_kde", "bimodal_call",
        }
        assert set(result.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Integration tests for main()
# ---------------------------------------------------------------------------


class TestMainIntegration:
    """Integration tests for the main() entry point."""

    def test_no_bimodal_genes(self, tmp_path):
        in_path = tmp_path / "in.tsv"
        out_path = tmp_path / "out.tsv"
        np.random.seed(16)
        polya_lines = [
            "transcript_id\ttx_idx\tn_reads\tpa_wlen\tprobs\tpa_lens",
        ]
        for i in range(3):
            n = 50
            pl = np.random.normal(80, 10, n).astype(float)
            pl = np.clip(pl, 0, None)
            polya_lines.append(
                f"TX{i}\t{i}\t{n}\t{pl.mean():.1f}\t"
                f"{','.join('1.0' for _ in range(n))}\t"
                f"{','.join(str(int(x)) for x in pl)}"
            )
        _make_polya_tsv(in_path, polya_lines)

        args = argparse.Namespace(
            input=str(in_path), output=str(out_path),
            gzip=False, min_length=0.0, min_asp=0.0,
            min_ess=30.0, kde_prominence=0.05,
        )
        main(args)

        lines = out_path.read_text().strip().split("\n")
        assert len(lines) == 4  # header + 3 features
        hdr = lines[0].split("\t")
        assert "feature_id" in hdr
        assert "bimodal_call" in hdr
        # None should be called bimodal (all unimodal)
        for line in lines[1:]:
            parts = line.split("\t")
            assert parts[-1] == "False"  # bimodal_call column

    def test_bimodal_feature_detected(self, tmp_path):
        in_path = tmp_path / "in.tsv"
        out_path = tmp_path / "out.tsv"
        np.random.seed(17)
        n = 200
        half = n // 2
        pa_lens = np.concatenate([
            np.random.normal(25, 4, half),
            np.random.normal(150, 10, half),
        ]).astype(float)
        pa_lens = np.clip(pa_lens, 0, None)
        polya_lines = [
            "gene_id\tn_reads\tpa_wlen\tprobs\tpa_lens",
            f"GENE_A\t{n}\t{pa_lens.mean():.1f}\t"
            f"{','.join('1.0' for _ in range(n))}\t"
            f"{','.join(str(int(x)) for x in pa_lens)}",
        ]
        _make_polya_tsv(in_path, polya_lines)

        args = argparse.Namespace(
            input=str(in_path), output=str(out_path),
            gzip=False, min_length=0.0, min_asp=0.0,
            min_ess=30.0, kde_prominence=0.03,
        )
        main(args)

        lines = out_path.read_text().strip().split("\n")
        assert len(lines) == 2  # header + 1 feature
        parts = lines[1].split("\t")
        hdr = lines[0].split("\t")
        bc_idx = hdr.index("bimodal_call")
        assert parts[bc_idx] == "True"

    def test_gzipped_output(self, tmp_path):
        in_path = tmp_path / "in.tsv"
        out_path = tmp_path / "out.tsv"
        np.random.seed(18)
        pa_lens = np.random.normal(80, 10, 50).astype(float)
        pa_lens = np.clip(pa_lens, 0, None)
        polya_lines = [
            "transcript_id\ttx_idx\tn_reads\tpa_wlen\tprobs\tpa_lens",
            f"TX1\t0\t50\t{pa_lens.mean():.1f}\t"
            f"{','.join('1.0' for _ in range(50))}\t"
            f"{','.join(str(int(x)) for x in pa_lens)}",
        ]
        _make_polya_tsv(in_path, polya_lines)

        args = argparse.Namespace(
            input=str(in_path), output=str(out_path),
            gzip=True, min_length=0.0, min_asp=0.0,
            min_ess=30.0, kde_prominence=0.05,
        )
        main(args)
        gz_path = tmp_path / "out.tsv.gz"
        assert gz_path.exists()

    def test_empty_input(self, tmp_path):
        in_path = tmp_path / "in.tsv"
        out_path = tmp_path / "out.tsv"
        _make_polya_tsv(in_path, [
            "transcript_id\ttx_idx\tn_reads\tpa_wlen\tprobs\tpa_lens",
        ])
        args = argparse.Namespace(
            input=str(in_path), output=str(out_path),
            gzip=False, min_length=0.0, min_asp=0.0,
            min_ess=30.0, kde_prominence=0.05,
        )
        with pytest.raises(SystemExit):
            main(args)

    def test_min_asp_filters_out_feature(self, tmp_path):
        in_path = tmp_path / "in.tsv"
        out_path = tmp_path / "out.tsv"
        pa_lens = np.random.normal(80, 10, 50).astype(float)
        pa_lens = np.clip(pa_lens, 0, None)
        polya_lines = [
            "transcript_id\ttx_idx\tn_reads\tpa_wlen\tprobs\tpa_lens",
            f"TX1\t0\t50\t{pa_lens.mean():.1f}\t"
            f"{','.join('0.05' for _ in range(50))}\t"
            f"{','.join(str(int(x)) for x in pa_lens)}",
        ]
        _make_polya_tsv(in_path, polya_lines)

        args = argparse.Namespace(
            input=str(in_path), output=str(out_path),
            gzip=False, min_length=0.0, min_asp=0.1,
            min_ess=30.0, kde_prominence=0.05,
        )
        with pytest.raises(SystemExit):  # no features pass
            main(args)

    def test_min_ess_threshold(self, tmp_path):
        in_path = tmp_path / "in.tsv"
        out_path = tmp_path / "out.tsv"
        pa_lens = np.array([50.0, 150.0, 80.0, 120.0])
        polya_lines = [
            "transcript_id\ttx_idx\tn_reads\tpa_wlen\tprobs\tpa_lens",
            f"TX1\t0\t4\t{pa_lens.mean():.1f}\t"
            f"{','.join('1.0' for _ in range(4))}\t"
            f"{','.join(str(int(x)) for x in pa_lens)}",
        ]
        _make_polya_tsv(in_path, polya_lines)

        args = argparse.Namespace(
            input=str(in_path), output=str(out_path),
            gzip=False, min_length=0.0, min_asp=0.0,
            min_ess=30.0, kde_prominence=0.05,
        )
        with pytest.raises(SystemExit):  # ESS=4 < 30
            main(args)

    def test_output_column_order(self, tmp_path):
        in_path = tmp_path / "in.tsv"
        out_path = tmp_path / "out.tsv"
        np.random.seed(19)
        pa_lens = np.random.normal(80, 10, 50).astype(float)
        pa_lens = np.clip(pa_lens, 0, None)
        polya_lines = [
            "transcript_id\ttx_idx\tn_reads\tpa_wlen\tprobs\tpa_lens",
            f"TX1\t0\t50\t{pa_lens.mean():.1f}\t"
            f"{','.join('1.0' for _ in range(50))}\t"
            f"{','.join(str(int(x)) for x in pa_lens)}",
        ]
        _make_polya_tsv(in_path, polya_lines)

        args = argparse.Namespace(
            input=str(in_path), output=str(out_path),
            gzip=False, min_length=0.0, min_asp=0.0,
            min_ess=30.0, kde_prominence=0.05,
        )
        main(args)

        lines = out_path.read_text().strip().split("\n")
        hdr = lines[0].split("\t")
        expected_header = [
            "feature_id", "id_type", "n_reads_raw", "n_reads_filtered",
            "ess", "delta_bic", "bic_k1", "bic_k2", "ll_k1", "ll_k2",
            "n_kde_peaks", "bimodal_gmm", "bimodal_kde", "bimodal_call",
        ]
        assert hdr == expected_header
