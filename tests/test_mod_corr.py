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
        _effective_sample_size,
        _mutual_information,
        _odds_ratio,
        _pearson_pvalue,
        _pearson_r_from_counts,
        _weighted_pearson_r,
        _write_parquet,
        _write_tsv,
        parse_args,
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
        _effective_sample_size,
        _mutual_information,
        _odds_ratio,
        _pearson_pvalue,
        _pearson_r_from_counts,
        _weighted_pearson_r,
        _write_parquet,
        _write_tsv,
        parse_args,
        process_transcript,
        read_site_summary,
    )
    from isolens.mod_scan import (  # type: ignore[no-redef]
        CODE_CANONICAL,
        CODE_FAIL,
        CODE_MISMATCH,
    )


# ---------- _pearson_r_from_counts ----------


class TestPearsonRFromCounts:
    """Tests for _pearson_r_from_counts()."""

    def test_perfect_positive(self):
        """When n10=n01=0, r=+1."""
        assert _pearson_r_from_counts(10, 0, 0, 10) == pytest.approx(1.0)

    def test_perfect_negative(self):
        """When n11=n00=0, r=-1."""
        assert _pearson_r_from_counts(0, 10, 10, 0) == pytest.approx(-1.0)

    def test_independence(self):
        """When rows/cols are independent, r≈0."""
        assert _pearson_r_from_counts(5, 5, 5, 5) == pytest.approx(0.0)

    def test_zero_denominator(self):
        """When a marginal is zero, returns 0."""
        assert _pearson_r_from_counts(0, 0, 0, 0) == 0.0

    def test_float_inputs(self):
        """Works with float (weighted) inputs."""
        result = _pearson_r_from_counts(10.5, 2.0, 3.0, 8.5)
        expected = (10.5 * 8.5 - 2.0 * 3.0) / np.sqrt(
            12.5 * 11.5 * 13.5 * 10.5
        )
        assert result == pytest.approx(expected)


# ---------- _pearson_pvalue ----------


class TestPearsonPvalue:
    """Tests for _pearson_pvalue()."""

    def test_perfect_correlation(self):
        """|r|=1 → p=0 even with handful of observations."""
        assert _pearson_pvalue(1.0, 10) == pytest.approx(0.0)
        assert _pearson_pvalue(-1.0, 10) == pytest.approx(0.0)

    def test_zero_correlation(self):
        """r=0 → large p-value."""
        p = _pearson_pvalue(0.0, 30)
        assert p == pytest.approx(1.0)

    def test_n_too_small(self):
        """n ≤ 2 → return 1.0 (insufficient df)."""
        assert _pearson_pvalue(0.9, 2) == pytest.approx(1.0)
        assert _pearson_pvalue(0.9, 1) == pytest.approx(1.0)

    def test_moderate_correlation(self):
        """Moderate r, moderate n → reasonable p-value."""
        p = _pearson_pvalue(0.5, 30)
        assert 0.001 < p < 0.05


# ---------- _effective_sample_size ----------


class TestEffectiveSampleSize:
    """Tests for _effective_sample_size()."""

    def test_uniform_weights(self):
        """Uniform weights → n_eff = len(weights)."""
        w = np.ones(10, dtype=np.float64)
        assert _effective_sample_size(w) == pytest.approx(10.0)

    def test_zero_total(self):
        """All zeros → n_eff = 0."""
        assert _effective_sample_size(np.zeros(5, dtype=np.float64)) == 0.0

    def test_non_uniform(self):
        """n_eff should be smaller than n for unequal weights."""
        w = np.array([1.0, 1.0, 0.1], dtype=np.float64)
        n_eff = _effective_sample_size(w)
        assert n_eff < 3.0
        assert n_eff > 0.0  # should be around 2.02


# ---------- _weighted_pearson_r ----------


class TestWeightedPearsonR:
    """Tests for _weighted_pearson_r()."""

    def test_perfect_correlation_uniform(self):
        """Perfectly correlated binary vectors with uniform weights."""
        x = np.array([True, True, False, False])
        y = np.array([True, True, False, False])
        w = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        r, n_eff = _weighted_pearson_r(x, y, w)
        assert r == pytest.approx(1.0)
        assert n_eff == pytest.approx(4.0)

    def test_perfect_anti_correlation(self):
        """Perfectly anti-correlated binary vectors."""
        x = np.array([True, True, False, False])
        y = np.array([False, False, True, True])
        w = np.ones(4, dtype=np.float64)
        r, n_eff = _weighted_pearson_r(x, y, w)
        assert r == pytest.approx(-1.0)
        assert n_eff == pytest.approx(4.0)

    def test_zero_weights(self):
        """All zero weights → return (0, 0)."""
        x = np.array([True, False])
        y = np.array([True, False])
        w = np.zeros(2, dtype=np.float64)
        r, n_eff = _weighted_pearson_r(x, y, w)
        assert r == 0.0
        assert n_eff == 0.0

    def test_non_uniform_weights(self):
        """Non-uniform weights give valid correlation in [-1, 1]."""
        x = np.array([True, False, True, False])
        y = np.array([True, False, False, True])
        w = np.array([2.0, 1.0, 0.5, 0.5], dtype=np.float64)
        r, n_eff = _weighted_pearson_r(x, y, w)
        assert -1.0 <= r <= 1.0
        assert n_eff > 0


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
        assert result == pytest.approx(441.0)
        assert not np.isinf(result)


# ---------- _mutual_information ----------


class TestMutualInformation:
    """Tests for _mutual_information()."""

    def test_perfect_correlation(self):
        """MI for perfect correlation."""
        mi = _mutual_information(10, 0, 0, 10)
        assert mi > 0.9

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
        assert q[0] == pytest.approx(0.03)
        assert q[1] == pytest.approx(0.03)
        assert q[2] == pytest.approx(0.03)

    def test_empty_list(self):
        assert _bh_fdr([]) == []

    def test_high_p_values(self):
        """High p-values → q-values may still be <1 after FDR correction."""
        q = _bh_fdr([0.5, 0.8, 0.9])
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
        assert rows[0]["mod_type1"] == "a"
        assert rows[0]["mod_type2"] == "m"

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
            min_mod_reads=2,
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
        assert r["mod_type1"] == "a"
        assert r["mod_type2"] == "a"
        assert "n11" in r
        assert "w11" in r
        assert "corr" in r
        assert "pvalue" in r
        assert "qvalue" in r
        assert "wcorr" in r
        assert "wpvalue" in r
        assert "wqvalue" in r
        assert "mi" in r
        assert "wmi" in r
        assert "or" in r
        assert "wor" in r

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
                self._site(1, 2, mod_level=0.9, depth=100),
                self._site(2, 2, mod_level=0.1, depth=100),
                self._site(3, 2, mod_level=0.9, depth=5),
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
        assert rows == []

    def test_weighted_statistics_present(self):
        """Weighted correlation and p-value are computed."""
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
        assert r["wcorr"] == pytest.approx(r["corr"])  # uniform weights
        assert r["wpvalue"] > 0  # p-value computed
        assert r["wqvalue"] >= 0  # q-value computed
        assert r["wor"] > 0  # weighted odds ratio

    def test_multiple_mod_types_same_position(self):
        """Each mod type at a site is reported independently.

        Position 1 has two mod types (a and m) observed on different reads.
        Position 2 has only mod type a.  Candidates (pos=1, a) with (pos=2, a)
        produces a same-type row; (pos=1, m) with (pos=2, a) produces a
        cross-type row.
        """
        # 4 reads: reads 0-1 have mod "a" at pos1, reads 2-3 have mod "m" at pos1
        #          all reads have mod "a" at pos2
        matrix = np.array(
            [[4, 4], [4, 4], [5, 4], [5, 4]],
            dtype=np.uint8,
        )
        weights = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        sites_by_mod = {
            "a": [self._site(1, 4), self._site(2, 4)],
            "m": [self._site(1, 4)],
        }
        mod_code_map = {"a": 4, "m": 5}

        rows = process_transcript(
            "TX1",
            matrix,
            weights,
            sites_by_mod,
            mod_code_map,
            min_mod_reads=1,
        )

        # 3 candidates: (1,a), (1,m), (2,a) → (1,a)-(1,m) skipped
        # (valid sets disjoint), (1,a)-(2,a) kept, (1,m)-(2,a) kept → 2 rows
        assert len(rows) == 2, f"Expected 2 rows, got {len(rows)}"
        types = {(r["mod_type1"], r["mod_type2"]) for r in rows}
        assert ("a", "a") in types  # same-type pair
        # Cross-type pair: (1,m)-(2,a) → mod_type1=m, mod_type2=a
        assert ("m", "a") in types or ("a", "m") in types


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
            # Verify schema has the new columns
            assert "mod_type1" in table.column_names
            assert "w11" in table.column_names
            assert "corr" in table.column_names
            assert "wor" in table.column_names
        finally:
            os.unlink(tmp_path)

    def test_non_empty_rows(self):
        rows = [
            {
                "transcript_id": "TX1",
                "site1": 1,
                "site2": 2,
                "mod_type1": "a",
                "mod_type2": "a",
                "n11": 5,
                "n10": 2,
                "n01": 3,
                "n00": 10,
                "w11": 4.5,
                "w10": 1.5,
                "w01": 2.5,
                "w00": 9.0,
                "corr": 0.5,
                "pvalue": 0.01,
                "qvalue": 0.05,
                "wcorr": 0.48,
                "wpvalue": 0.02,
                "wqvalue": 0.06,
                "mi": 0.3,
                "wmi": 0.28,
                "or": 2.5,
                "wor": 2.3,
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
        assert "mod_type1" in content
        assert "corr" in content

    def test_non_empty_rows(self, tmp_path):
        rows = [
            {
                "transcript_id": "TX1",
                "site1": 1,
                "site2": 2,
                "mod_type1": "a",
                "mod_type2": "a",
                "n11": 5,
                "n10": 2,
                "n01": 3,
                "n00": 10,
                "w11": 4.5,
                "w10": 1.5,
                "w01": 2.5,
                "w00": 9.0,
                "corr": 0.5,
                "pvalue": 0.01,
                "qvalue": 0.05,
                "wcorr": 0.48,
                "wpvalue": 0.02,
                "wqvalue": 0.06,
                "mi": 0.3,
                "wmi": 0.28,
                "or": 2.5,
                "wor": 2.3,
            }
        ]
        path = tmp_path / "out.tsv"
        _write_tsv(rows, str(path), use_gzip=False)
        content = path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 2  # header + data


# ---------- parse_args ----------


class TestParseArgs:
    """Tests for parse_args()."""

    def test_default_metric(self):
        """Default metric is wcorr."""
        import sys as _sys

        _sys.argv = [
            "mod_corr", "-i", "test.h5", "-s", "sites.parquet", "-o", "out.parquet"
        ]
        args = parse_args()
        assert args.metric == "wcorr"

    def test_custom_metric(self):
        """Custom metric is accepted."""
        import sys as _sys

        _sys.argv = [
            "mod_corr",
            "-i", "test.h5",
            "-s", "sites.parquet",
            "-o", "out.parquet",
            "-t", "mi",
        ]
        args = parse_args()
        assert args.metric == "mi"

    def test_plot_flag_renamed(self):
        """-d/--plot-dir is the flag name for plot output directory."""
        import sys as _sys

        _sys.argv = [
            "mod_corr",
            "-i", "test.h5",
            "-s", "sites.parquet",
            "-o", "out.parquet",
            "-d", "/tmp/plots",
            "-t", "corr",
        ]
        args = parse_args()
        assert args.plot_dir == "/tmp/plots"
        assert args.metric == "corr"

    def test_invalid_metric_rejected(self):
        """Invalid metric choice causes argparse error."""
        import sys as _sys

        _sys.argv = [
            "mod_corr",
            "-i", "test.h5",
            "-s", "sites.parquet",
            "-o", "out.parquet",
            "-t", "invalid",
        ]
        with pytest.raises(SystemExit):
            parse_args()
