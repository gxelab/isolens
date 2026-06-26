"""Tests for mod_corr — pairwise modification site correlation analysis."""

import os
import sys
import tempfile

import numpy as np
import pyarrow.parquet as pq
import pytest

try:
    from isolens.mod_corr import (
        _bh_fdr,
        _mutual_information,
        _odds_ratio,
        _phi_coefficient,
        _write_parquet,
        _write_tsv,
        process_transcript,
        read_site_summary,
    )
    from isolens.mod_scan import (
        CODE_CANONICAL,
        CODE_FAIL,
        CODE_MISMATCH,
    )
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from isolens.mod_corr import (  # type: ignore[no-redef]
        _bh_fdr,
        _mutual_information,
        _odds_ratio,
        _phi_coefficient,
        _write_parquet,
        _write_tsv,
        process_transcript,
        read_site_summary,
    )
    from isolens.mod_scan import (  # type: ignore[no-redef]
        CODE_CANONICAL,
        CODE_FAIL,
        CODE_MISMATCH,
    )


# ---------- _phi_coefficient ----------


class TestPhiCoefficient:
    """Tests for _phi_coefficient()."""

    def test_perfect_positive(self):
        """When n10=n01=0, Phi=+1."""
        assert _phi_coefficient(10, 0, 0, 10) == pytest.approx(1.0)

    def test_perfect_negative(self):
        """When n11=n00=0, Phi=-1."""
        assert _phi_coefficient(0, 10, 10, 0) == pytest.approx(-1.0)

    def test_independence(self):
        """When rows/cols are independent, Phi≈0."""
        # n11 = 5, n10 = 5, n01 = 5, n00 = 5 → independent
        assert _phi_coefficient(5, 5, 5, 5) == pytest.approx(0.0)

    def test_zero_denominator(self):
        """When a marginal is zero, returns 0."""
        assert _phi_coefficient(0, 0, 0, 0) == 0.0

    def test_float_inputs(self):
        """Works with float (weighted) inputs."""
        result = _phi_coefficient(10.5, 2.0, 3.0, 8.5)
        expected = (10.5 * 8.5 - 2.0 * 3.0) / np.sqrt(12.5 * 11.5 * 13.5 * 10.5)
        assert result == pytest.approx(expected)


# ---------- _odds_ratio ----------


class TestOddsRatio:
    """Tests for _odds_ratio()."""

    def test_symmetric(self):
        """4 * 10 / (2 * 2) with HAC = 4.5 * 10.5 / (2.5 * 2.5)."""
        result = _odds_ratio(4, 2, 2, 10)
        expected = (4.5 * 10.5) / (2.5 * 2.5)
        assert result == pytest.approx(expected)

    def test_zero_cells(self):
        """HAC prevents division by zero / infinite OR."""
        result = _odds_ratio(10, 0, 0, 10)
        # (10.5 * 10.5) / (0.5 * 0.5) = 110.25 / 0.25 = 441
        assert result == pytest.approx(441.0)
        assert not np.isinf(result)


# ---------- _mutual_information ----------


class TestMutualInformation:
    """Tests for _mutual_information()."""

    def test_perfect_correlation(self):
        """MI for perfect correlation."""
        mi = _mutual_information(10, 0, 0, 10)
        assert mi > 0.9  # near 1 bit for high correlation

    def test_independence(self):
        """MI for independent variables ≈ 0."""
        mi = _mutual_information(5, 5, 5, 5)
        assert mi == pytest.approx(0.0, abs=1e-10)

    def test_zero_total(self):
        """Zero total observations → 0."""
        assert _mutual_information(0, 0, 0, 0) == 0.0


# ---------- _bh_fdr ----------


class TestBhFdr:
    """Tests for Benjamini-Hochberg FDR correction."""

    def test_single_p_value(self):
        q = _bh_fdr([0.01])
        assert q == pytest.approx([0.01])

    def test_multiple_sorted(self):
        p = [0.01, 0.02, 0.03]
        q = _bh_fdr(p)
        # q[0] = min(1, 0.01*3/1)=0.03, non-decreasing sweep
        assert q[0] == pytest.approx(0.03)
        assert q[1] == pytest.approx(0.03)
        assert q[2] == pytest.approx(0.03)

    def test_empty_list(self):
        assert _bh_fdr([]) == []

    def test_high_p_values(self):
        """High p-values → q-values may still be <1 after FDR correction."""
        q = _bh_fdr([0.5, 0.8, 0.9])
        # Third p-value: 0.9 * 3/3 = 0.9, backward sweep → all 0.9
        for v in q:
            assert v == pytest.approx(0.9)


# ---------- process_transcript ----------


class TestProcessTranscript:
    """Tests for process_transcript()."""

    def _site(self, pos, n_mod, mod_level=0.5, depth=100):
        """Helper to create a site dict for tests."""
        return {"pos": pos, "n_mod": n_mod, "mod_level": mod_level, "depth": depth}

    def test_two_sites_same_type(self):
        """Two sites of same mod type produce one pair."""
        matrix = np.array(
            [[4, 4, CODE_CANONICAL], [4, CODE_CANONICAL, 4], [CODE_CANONICAL, 4, 4]],
            dtype=np.uint8,
        )
        weights = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        sites_by_mod = {
            "a": [self._site(1, 3), self._site(2, 3), self._site(3, 2)]
        }
        mod_code_map = {"a": 4}

        rows = process_transcript(
            "TX1",
            matrix,
            weights,
            sites_by_mod,
            mod_code_map,
            min_mod_reads=2,
        )

        # All 3 sites qualify (n_mod >= 2) → 3 choose 2 = 3 pairs
        assert len(rows) == 3

    def test_cross_type_pair(self):
        """Sites from different mod types produce cross-type pairs."""
        matrix = np.array(
            [[4, 5, CODE_CANONICAL], [4, CODE_CANONICAL, 5]],
            dtype=np.uint8,
        )
        weights = np.array([1.0, 1.0], dtype=np.float32)
        sites_by_mod = {"a": [self._site(1, 2)], "m": [self._site(2, 2)]}
        mod_code_map = {"a": 4, "m": 5}

        rows = process_transcript(
            "TX1",
            matrix,
            weights,
            sites_by_mod,
            mod_code_map,
            min_mod_reads=1,
        )

        assert len(rows) == 1
        assert rows[0]["modification_type"] == "a:m"

    def test_below_min_mod_reads(self):
        """Sites with n_modified < min_mod_reads are excluded."""
        matrix = np.array([[4, 4]], dtype=np.uint8)
        weights = np.array([1.0], dtype=np.float32)
        sites_by_mod = {"a": [self._site(1, 1), self._site(2, 1)]}
        mod_code_map = {"a": 4}

        rows = process_transcript(
            "TX1",
            matrix,
            weights,
            sites_by_mod,
            mod_code_map,
            min_mod_reads=2,  # both sites have n_mod=1 < 2
        )
        assert rows == []

    def test_single_candidate(self):
        """Only one candidate site → no pairs possible."""
        matrix = np.array([[4, CODE_CANONICAL]], dtype=np.uint8)
        weights = np.array([1.0], dtype=np.float32)
        sites_by_mod = {"a": [self._site(1, 1)]}
        mod_code_map = {"a": 4}

        rows = process_transcript(
            "TX1",
            matrix,
            weights,
            sites_by_mod,
            mod_code_map,
            min_mod_reads=1,
        )
        assert rows == []

    def test_min_asp_filtering(self):
        """Reads below min_asp are excluded."""
        matrix = np.array(
            [[4, 4], [4, CODE_CANONICAL], [CODE_CANONICAL, 4]],
            dtype=np.uint8,
        )
        weights = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        sites_by_mod = {"a": [self._site(1, 3), self._site(2, 2)]}
        mod_code_map = {"a": 4}

        rows = process_transcript(
            "TX1",
            matrix,
            weights,
            sites_by_mod,
            mod_code_map,
            min_mod_reads=1,
            min_asp=0.9,
        )
        # All reads excluded → no rows
        assert rows == []

    def test_output_columns(self):
        """Verify all expected columns are present."""
        matrix = np.array(
            [[4, 4], [4, CODE_CANONICAL], [CODE_CANONICAL, 4]],
            dtype=np.uint8,
        )
        weights = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        sites_by_mod = {"a": [self._site(1, 3), self._site(2, 2)]}
        mod_code_map = {"a": 4}

        rows = process_transcript(
            "TX1",
            matrix,
            weights,
            sites_by_mod,
            mod_code_map,
            min_mod_reads=2,
        )

        assert len(rows) == 1
        r = rows[0]
        assert r["transcript_id"] == "TX1"
        assert r["site1"] == 1
        assert r["site2"] == 2
        assert "n11" in r
        assert "phi" in r
        assert "p_value" in r
        assert "q_value" in r

    def test_few_valid_reads(self):
        """Pair skipped when fewer than 2 joint-valid reads."""
        matrix = np.array(
            [[CODE_FAIL, CODE_FAIL], [CODE_MISMATCH, CODE_MISMATCH]],
            dtype=np.uint8,
        )
        weights = np.array([1.0, 1.0], dtype=np.float32)
        sites_by_mod = {"a": [self._site(1, 2), self._site(2, 2)]}
        mod_code_map = {"a": 4}

        rows = process_transcript(
            "TX1",
            matrix,
            weights,
            sites_by_mod,
            mod_code_map,
            min_mod_reads=1,
        )
        assert rows == []

    def test_min_mod_level_filter(self):
        """Sites with mod_level below threshold are excluded."""
        matrix = np.array(
            [[4, 4], [4, CODE_CANONICAL]],
            dtype=np.uint8,
        )
        weights = np.array([1.0, 1.0], dtype=np.float32)
        sites_by_mod = {
            "a": [
                self._site(1, 2, mod_level=0.9, depth=2),
                self._site(2, 1, mod_level=0.1, depth=2),
            ]
        }
        mod_code_map = {"a": 4}

        rows = process_transcript(
            "TX1",
            matrix,
            weights,
            sites_by_mod,
            mod_code_map,
            min_mod_reads=1,
            min_mod_level=0.5,
        )
        # Only site 1 passes mod_level filter → single candidate
        assert rows == []

    def test_depth_filter(self):
        """Sites with depth below threshold are excluded."""
        matrix = np.array(
            [[4, 4], [4, CODE_CANONICAL]],
            dtype=np.uint8,
        )
        weights = np.array([1.0, 1.0], dtype=np.float32)
        sites_by_mod = {
            "a": [
                self._site(1, 2, mod_level=0.5, depth=100),
                self._site(2, 1, mod_level=0.5, depth=5),
            ]
        }
        mod_code_map = {"a": 4}

        rows = process_transcript(
            "TX1",
            matrix,
            weights,
            sites_by_mod,
            mod_code_map,
            min_mod_reads=1,
            min_depth=10,
        )
        # Only site 1 passes depth filter → single candidate
        assert rows == []

    def test_combined_filters(self):
        """All three filters applied together."""
        matrix = np.array(
            [[4, 4, 4], [4, CODE_CANONICAL, 4]],
            dtype=np.uint8,
        )
        weights = np.array([1.0, 1.0], dtype=np.float32)
        sites_by_mod = {
            "a": [
                self._site(1, 2, mod_level=0.9, depth=100),  # passes all
                self._site(2, 2, mod_level=0.1, depth=100),  # fails mod_level
                self._site(3, 2, mod_level=0.9, depth=5),    # fails depth
            ]
        }
        mod_code_map = {"a": 4}

        rows = process_transcript(
            "TX1",
            matrix,
            weights,
            sites_by_mod,
            mod_code_map,
            min_mod_reads=2,
            min_mod_level=0.5,
            min_depth=10,
        )
        # Only site 1 passes all filters → single candidate, no pairs
        assert rows == []


# ---------- read_site_summary ----------


class TestReadSiteSummary:
    """Tests for read_site_summary()."""

    def test_parquet_input(self, tmp_path):
        import pyarrow as pa

        table = pa.table(
            {
                "transcript_id": ["TX1", "TX1"],
                "position": [42, 100],
                "mod_type": ["a", "a"],
                "n_modified": [10, 5],
                "n_unmodified": [90, 45],
                "n_mismatch": [0, 0],
                "n_deletion": [0, 0],
                "n_failed": [0, 0],
                "mod_level": [0.1, 0.1],
            }
        )
        path = tmp_path / "sites.parquet"
        pq.write_table(table, str(path))

        sites = read_site_summary(str(path))
        assert "TX1" in sites
        assert "a" in sites["TX1"]
        assert sites["TX1"]["a"] == [
            {"pos": 42, "n_mod": 10, "mod_level": 0.1, "depth": 100},
            {"pos": 100, "n_mod": 5, "mod_level": 0.1, "depth": 50},
        ]

    def test_tsv_input(self, tmp_path):
        path = tmp_path / "sites.tsv"
        path.write_text(
            "transcript_id\tposition\tmod_type\tn_modified\t"
            "n_unmodified\tn_mismatch\tn_deletion\tn_failed\tmod_level\n"
            "TX1\t42\ta\t10\t90\t0\t0\t0\t0.1\n"
            "TX1\t100\tm\t5\t40\t3\t2\t0\t0.1\n"
        )
        sites = read_site_summary(str(path))
        assert sites["TX1"]["a"] == [
            {"pos": 42, "n_mod": 10, "mod_level": 0.1, "depth": 100}
        ]
        assert sites["TX1"]["m"] == [
            {"pos": 100, "n_mod": 5, "mod_level": 0.1, "depth": 50}
        ]


# ---------- _write_parquet ----------


class TestWriteParquet:
    """Tests for _write_parquet()."""

    def test_empty_rows(self):
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tf:
            tmp_path = tf.name
        try:
            _write_parquet([], tmp_path)
            table = pq.read_table(tmp_path)
            assert len(table) == 0
        finally:
            os.unlink(tmp_path)

    def test_non_empty_rows(self):
        rows = [
            {
                "transcript_id": "TX1",
                "site1": 1,
                "site2": 2,
                "modification_type": "a",
                "n11": 5,
                "n10": 2,
                "n01": 3,
                "n00": 10,
                "weighted_n11": 4.5,
                "weighted_n10": 1.5,
                "weighted_n01": 2.5,
                "weighted_n00": 9.0,
                "phi": 0.5,
                "weighted_phi": 0.48,
                "odds_ratio": 2.5,
                "p_value": 0.01,
                "q_value": 0.05,
                "mutual_information": 0.3,
                "weighted_mutual_information": 0.28,
            }
        ]
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tf:
            tmp_path = tf.name
        try:
            _write_parquet(rows, tmp_path)
            table = pq.read_table(tmp_path)
            assert len(table) == 1
        finally:
            os.unlink(tmp_path)


# ---------- _write_tsv ----------


class TestWriteTsv:
    """Tests for _write_tsv()."""

    def test_empty_rows(self, tmp_path):
        path = tmp_path / "out.tsv"
        _write_tsv([], str(path), use_gzip=False)
        content = path.read_text()
        assert "transcript_id" in content

    def test_non_empty_rows(self, tmp_path):
        rows = [
            {
                "transcript_id": "TX1",
                "site1": 1,
                "site2": 2,
                "modification_type": "a",
                "n11": 5,
                "n10": 2,
                "n01": 3,
                "n00": 10,
                "weighted_n11": 4.5,
                "weighted_n10": 1.5,
                "weighted_n01": 2.5,
                "weighted_n00": 9.0,
                "phi": 0.5,
                "weighted_phi": 0.48,
                "odds_ratio": 2.5,
                "p_value": 0.01,
                "q_value": 0.05,
                "mutual_information": 0.3,
                "weighted_mutual_information": 0.28,
            }
        ]
        path = tmp_path / "out.tsv"
        _write_tsv(rows, str(path), use_gzip=False)
        content = path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 2  # header + data
