"""Tests for mod_dmcg — gene-level differential modification calling."""

import argparse
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.stats import fisher_exact

try:
    from isolens.mod_dmcg import (
        _OUTPUT_COLS,
        _fisher_test,
        _write_parquet,
        _write_tsv,
        main,
        parse_args,
        process_matched_sites,
        read_gene_summary,
    )
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from isolens.mod_dmcg import (  # type: ignore[no-redef]
        _OUTPUT_COLS,
        _fisher_test,
        _write_parquet,
        _write_tsv,
        main,
        parse_args,
        process_matched_sites,
        read_gene_summary,
    )


# ---------- helpers ----------


def _make_site(
    gene_id="G1",
    chrom="2L",
    strand="+",
    gpos=100,
    mod_type="a",
    n_modified=10,
    n_unmodified=90,
    wt_modified=8.5,
    wt_unmodified=85.0,
    mod_level=0.1,
    wt_mod_level=0.091,
):
    """Create a gene-level site dict matching mod_gene output columns."""
    return {
        "gene_id": gene_id,
        "chrom": chrom,
        "strand": strand,
        "gpos": gpos,
        "mod_type": mod_type,
        "n_modified": n_modified,
        "n_unmodified": n_unmodified,
        "wt_modified": wt_modified,
        "wt_unmodified": wt_unmodified,
        "mod_level": mod_level,
        "wt_mod_level": wt_mod_level,
    }


def _write_gene_parquet(path, rows):
    """Write a list of gene-level site dicts as a Parquet file."""
    cols = [
        "gene_id",
        "chrom",
        "strand",
        "gpos",
        "mod_type",
        "n_modified",
        "wt_modified",
        "n_unmodified",
        "wt_unmodified",
        "mod_level",
        "wt_mod_level",
    ]
    arrays = {}
    for c in cols:
        values = [r.get(c) for r in rows]
        if c in ("gene_id", "chrom", "strand", "mod_type"):
            arrays[c] = pa.array(values)
        elif c == "gpos":
            arrays[c] = pa.array(values, type=pa.int32())
        elif c.startswith("n_"):
            arrays[c] = pa.array(values, type=pa.int32())
        else:
            arrays[c] = pa.array(values, type=pa.float64())
    pq.write_table(pa.table(arrays), path)


# ---------- _fisher_test ----------


class TestFisherTest:
    """Tests for _fisher_test()."""

    def test_clear_effect(self):
        """Strong difference → small p-value, large log2OR."""
        result = _fisher_test(0, 100, 100, 0)
        assert result["log2_or"] < -4.0  # odds ratio ≈ 0
        assert result["p_value"] < 0.001

    def test_no_difference(self):
        """Equal proportions → p ≈ 1, log2OR ≈ 0."""
        result = _fisher_test(50, 50, 50, 50)
        assert abs(result["log2_or"]) < 0.1
        assert result["p_value"] > 0.5

    def test_all_zeros(self):
        """Degenerate table → NaN results."""
        result = _fisher_test(0, 0, 0, 0)
        assert np.isnan(result["log2_or"])
        assert np.isnan(result["p_value"])

    def test_zero_row(self):
        """One row all zeros → NaN."""
        result = _fisher_test(0, 0, 10, 10)
        assert np.isnan(result["log2_or"])
        assert np.isnan(result["p_value"])

    def test_zero_column(self):
        """One column all zeros → NaN."""
        result = _fisher_test(10, 0, 10, 0)
        assert np.isnan(result["log2_or"])

    def test_matches_scipy(self):
        """Results match scipy.stats.fisher_exact directly."""
        n1_mod, n1_unmod = 5, 15
        n2_mod, n2_unmod = 15, 5
        result = _fisher_test(n1_mod, n1_unmod, n2_mod, n2_unmod)
        odds, p = fisher_exact([[n1_mod, n1_unmod], [n2_mod, n2_unmod]])
        assert result["p_value"] == pytest.approx(p)
        assert result["log2_or"] == pytest.approx(np.log2(odds))


# ---------- read_gene_summary ----------


class TestReadGeneSummary:
    """Tests for read_gene_summary()."""

    def test_parquet(self, tmp_path):
        path = str(tmp_path / "genes.parquet")
        _write_gene_parquet(
            path,
            [
                _make_site(gene_id="G1", gpos=100, mod_type="a"),
                _make_site(gene_id="G1", gpos=200, mod_type="m"),
            ],
        )
        sites = read_gene_summary(path)
        assert len(sites) == 2
        assert ("G1", "2L", "+", 100, "a") in sites
        assert ("G1", "2L", "+", 200, "m") in sites

    def test_tsv(self, tmp_path):
        path = str(tmp_path / "genes.tsv")
        with open(path, "w") as f:
            f.write(
                "gene_id\tchrom\tstrand\tgpos\tmod_type\t"
                "n_modified\twt_modified\tn_unmodified\twt_unmodified\t"
                "mod_level\twt_mod_level\n"
            )
            f.write("G1\t2L\t+\t100\ta\t10\t8.5\t90\t85.0\t0.1\t0.091\n")
        sites = read_gene_summary(path)
        assert len(sites) == 1
        s = sites[("G1", "2L", "+", 100, "a")]
        assert s["n_modified"] == 10
        assert s["wt_modified"] == pytest.approx(8.5)

    def test_tsv_na_values(self, tmp_path):
        """TSV with NA for null mod_level."""
        path = str(tmp_path / "genes.tsv")
        with open(path, "w") as f:
            f.write(
                "gene_id\tchrom\tstrand\tgpos\tmod_type\t"
                "n_modified\twt_modified\tn_unmodified\twt_unmodified\t"
                "mod_level\twt_mod_level\n"
            )
            f.write("G1\t2L\t+\t100\ta\t0\t0.0\t0\t0.0\tNA\tNA\n")
        sites = read_gene_summary(path)
        s = sites[("G1", "2L", "+", 100, "a")]
        assert s["mod_level"] is None
        assert s["wt_mod_level"] is None


# ---------- process_matched_sites ----------


class TestProcessMatchedSites:
    """Tests for process_matched_sites()."""

    def test_single_match_clear_effect(self):
        """One matched gene-site with strong difference."""
        sites_1 = {
            ("G1", "2L", "+", 100, "a"): _make_site(
                n_modified=0,
                n_unmodified=100,
                wt_modified=0.0,
                wt_unmodified=100.0,
                mod_level=0.0,
                wt_mod_level=0.0,
            ),
        }
        sites_2 = {
            ("G1", "2L", "+", 100, "a"): _make_site(
                n_modified=100,
                n_unmodified=0,
                wt_modified=100.0,
                wt_unmodified=0.0,
                mod_level=1.0,
                wt_mod_level=1.0,
            ),
        }
        rows = process_matched_sites(sites_1, sites_2)
        assert len(rows) == 1
        r = rows[0]
        assert r["gene_id"] == "G1"
        assert r["mod_type"] == "a"
        assert r["mod_level_1"] == 0.0
        assert r["mod_level_2"] == 1.0
        assert r["delta_mod_level"] == pytest.approx(1.0)
        assert r["log2_or"] < 0  # cond1 lower mod → negative log2OR
        assert r["p_value"] < 0.001

    def test_no_difference(self):
        """Equal modification levels → large p-value, log2OR ≈ 0."""
        site = _make_site(
            n_modified=50,
            n_unmodified=50,
            wt_modified=50.0,
            wt_unmodified=50.0,
            mod_level=0.5,
            wt_mod_level=0.5,
        )
        sites_1 = {("G1", "2L", "+", 100, "a"): site}
        sites_2 = {("G1", "2L", "+", 100, "a"): site}
        rows = process_matched_sites(sites_1, sites_2)
        assert len(rows) == 1
        r = rows[0]
        assert abs(r["log2_or"]) < 0.1
        assert r["p_value"] > 0.5

    def test_site_only_in_one_condition_skipped(self):
        """Site in cond1 but not cond2 → no output."""
        sites_1 = {
            ("G1", "2L", "+", 100, "a"): _make_site(),
        }
        sites_2 = {}
        rows = process_matched_sites(sites_1, sites_2)
        assert rows == []

    def test_zero_counts_skipped(self):
        """Zero total counts in one condition → skipped."""
        sites_1 = {
            ("G1", "2L", "+", 100, "a"): _make_site(
                n_modified=0,
                n_unmodified=0,
            ),
        }
        sites_2 = {
            ("G1", "2L", "+", 100, "a"): _make_site(
                n_modified=10,
                n_unmodified=10,
            ),
        }
        rows = process_matched_sites(sites_1, sites_2)
        assert rows == []

    def test_weighted_rounding(self):
        """Weighted counts are correctly rounded."""
        sites_1 = {
            ("G1", "2L", "+", 100, "a"): _make_site(
                n_modified=10,
                n_unmodified=90,
                wt_modified=10.4,
                wt_unmodified=89.6,
            ),
        }
        sites_2 = {
            ("G1", "2L", "+", 100, "a"): _make_site(
                n_modified=90,
                n_unmodified=10,
                wt_modified=89.6,
                wt_unmodified=10.4,
            ),
        }
        rows = process_matched_sites(sites_1, sites_2)
        assert len(rows) == 1
        r = rows[0]
        # Weighted test uses rounded counts: 10 vs 90 → should be significant
        assert r["w_p_value"] < 0.001
        # Unweighted test also significant
        assert r["p_value"] < 0.001

    def test_multiple_genes(self):
        """Multiple matched genes produce correct number of rows."""
        sites_1 = {
            ("G1", "2L", "+", 100, "a"): _make_site(),
            ("G1", "2L", "+", 200, "m"): _make_site(mod_type="m"),
            ("G2", "3R", "-", 500, "a"): _make_site(
                gene_id="G2",
                chrom="3R",
                strand="-",
                gpos=500,
            ),
        }
        sites_2 = {
            ("G1", "2L", "+", 100, "a"): _make_site(
                n_modified=20,
                n_unmodified=80,
                wt_modified=18.0,
                wt_unmodified=75.0,
            ),
            ("G1", "2L", "+", 200, "m"): _make_site(
                mod_type="m",
                n_modified=5,
                n_unmodified=95,
            ),
            # G2 only in cond1 → not matched
        }
        rows = process_matched_sites(sites_1, sites_2)
        assert len(rows) == 2

    def test_output_columns_complete(self):
        """All expected output columns are present."""
        sites_1 = {
            ("G1", "2L", "+", 100, "a"): _make_site(),
        }
        sites_2 = {
            ("G1", "2L", "+", 100, "a"): _make_site(
                n_modified=20,
                n_unmodified=80,
            ),
        }
        rows = process_matched_sites(sites_1, sites_2)
        assert len(rows) == 1
        expected = {
            "gene_id",
            "chrom",
            "strand",
            "gpos",
            "mod_type",
            "n_modified_1",
            "n_unmodified_1",
            "n_modified_2",
            "n_unmodified_2",
            "wt_modified_1",
            "wt_unmodified_1",
            "wt_modified_2",
            "wt_unmodified_2",
            "mod_level_1",
            "mod_level_2",
            "wt_mod_level_1",
            "wt_mod_level_2",
            "delta_mod_level",
            "delta_wt_mod_level",
            "log2_or",
            "p_value",
            "q_value",
            "w_log2_or",
            "w_p_value",
            "w_q_value",
        }
        assert set(rows[0].keys()) == expected

    def test_q_values_initialized_zero(self):
        """q_value and w_q_value are 0.0 before BH correction."""
        sites_1 = {
            ("G1", "2L", "+", 100, "a"): _make_site(),
        }
        sites_2 = {
            ("G1", "2L", "+", 100, "a"): _make_site(
                n_modified=20,
                n_unmodified=80,
            ),
        }
        rows = process_matched_sites(sites_1, sites_2)
        assert rows[0]["q_value"] == 0.0
        assert rows[0]["w_q_value"] == 0.0

    def test_delta_mod_level_sign(self):
        """delta_mod_level = mod_level_2 - mod_level_1."""
        sites_1 = {
            ("G1", "2L", "+", 100, "a"): _make_site(
                mod_level=0.2,
                wt_mod_level=0.19,
            ),
        }
        sites_2 = {
            ("G1", "2L", "+", 100, "a"): _make_site(
                mod_level=0.8,
                wt_mod_level=0.81,
            ),
        }
        rows = process_matched_sites(sites_1, sites_2)
        assert rows[0]["delta_mod_level"] == pytest.approx(0.6)
        assert rows[0]["delta_wt_mod_level"] == pytest.approx(0.62)

    def test_null_mod_levels(self):
        """NULL mod_levels produce NULL deltas."""
        sites_1 = {
            ("G1", "2L", "+", 100, "a"): _make_site(
                mod_level=None,
                wt_mod_level=None,
            ),
        }
        sites_2 = {
            ("G1", "2L", "+", 100, "a"): _make_site(
                mod_level=None,
                wt_mod_level=None,
            ),
        }
        rows = process_matched_sites(sites_1, sites_2)
        assert rows[0]["delta_mod_level"] is None
        assert rows[0]["delta_wt_mod_level"] is None


# ---------- output writers ----------


class TestWriteOutput:
    """Tests for _write_parquet and _write_tsv."""

    def _make_row(self, **overrides):
        defaults = {
            "gene_id": "G1",
            "chrom": "2L",
            "strand": "+",
            "gpos": 100,
            "mod_type": "a",
            "n_modified_1": 10,
            "n_unmodified_1": 90,
            "n_modified_2": 30,
            "n_unmodified_2": 70,
            "wt_modified_1": 8.5,
            "wt_unmodified_1": 85.0,
            "wt_modified_2": 28.0,
            "wt_unmodified_2": 68.0,
            "mod_level_1": 0.1,
            "mod_level_2": 0.3,
            "wt_mod_level_1": 0.091,
            "wt_mod_level_2": 0.292,
            "delta_mod_level": 0.2,
            "delta_wt_mod_level": 0.201,
            "log2_or": -2.5,
            "p_value": 0.001,
            "q_value": 0.01,
            "w_log2_or": -2.3,
            "w_p_value": 0.002,
            "w_q_value": 0.015,
        }
        defaults.update(overrides)
        return defaults

    def test_write_parquet(self, tmp_path):
        path = str(tmp_path / "out.parquet")
        rows = [self._make_row()]
        _write_parquet(rows, path)
        table = pq.read_table(path)
        assert len(table) == 1
        assert table.column("log2_or")[0].as_py() == pytest.approx(-2.5)
        assert table.column("w_log2_or")[0].as_py() == pytest.approx(-2.3)

    def test_write_empty_parquet(self, tmp_path):
        path = str(tmp_path / "out.parquet")
        _write_parquet([], path)
        table = pq.read_table(path)
        assert len(table) == 0
        # Check schema has all expected columns
        assert set(table.column_names) == set(_OUTPUT_COLS)

    def test_write_tsv(self, tmp_path):
        path = str(tmp_path / "out.tsv")
        rows = [self._make_row()]
        _write_tsv(rows, path, use_gzip=False)
        with open(path) as f:
            header = f.readline()
            data = f.readline()
        assert "gene_id" in header
        assert "G1" in data
        assert "w_log2_or" in header

    def test_write_tsv_null_values(self, tmp_path):
        path = str(tmp_path / "out.tsv")
        row = self._make_row(delta_mod_level=None, delta_wt_mod_level=None)
        _write_tsv([row], path, use_gzip=False)
        with open(path) as f:
            f.readline()  # header
            data = f.readline()
        assert "NA" in data


# ---------- CLI ----------


class TestCLI:
    """Tests for parse_args()."""

    def test_required_args(self):
        argv = [
            "--sites-1",
            "a.parquet",
            "--sites-2",
            "b.parquet",
            "-o",
            "out.parquet",
        ]
        import sys as _sys

        _sys.argv = ["mod_dmcg"] + argv
        args = parse_args()
        assert args.sites_1 == "a.parquet"
        assert args.sites_2 == "b.parquet"
        assert args.format == "parquet"

    def test_optional_args(self):
        argv = [
            "--sites-1",
            "a.parquet",
            "--sites-2",
            "b.parquet",
            "-o",
            "out.tsv",
            "-f",
            "tsv",
            "-z",
            "-v",
        ]
        import sys as _sys

        _sys.argv = ["mod_dmcg"] + argv
        args = parse_args()
        assert args.format == "tsv"
        assert args.gzip is True
        assert args.verbose is True


# ---------- integration tests ----------


class TestMainIntegration:
    """End-to-end integration tests for main()."""

    def test_end_to_end_parquet(self, tmp_path):
        """Two gene summaries → valid differential output."""
        sites_1_path = str(tmp_path / "genes1.parquet")
        sites_2_path = str(tmp_path / "genes2.parquet")
        out_path = str(tmp_path / "out.parquet")

        # Condition 1: low modification
        _write_gene_parquet(
            sites_1_path,
            [
                _make_site(
                    gene_id="G1",
                    gpos=100,
                    mod_type="a",
                    n_modified=5,
                    n_unmodified=95,
                    wt_modified=4.5,
                    wt_unmodified=90.0,
                    mod_level=0.05,
                    wt_mod_level=0.048,
                ),
            ],
        )
        # Condition 2: high modification
        _write_gene_parquet(
            sites_2_path,
            [
                _make_site(
                    gene_id="G1",
                    gpos=100,
                    mod_type="a",
                    n_modified=80,
                    n_unmodified=20,
                    wt_modified=75.0,
                    wt_unmodified=18.0,
                    mod_level=0.8,
                    wt_mod_level=0.806,
                ),
            ],
        )

        args = argparse.Namespace(
            sites_1=sites_1_path,
            sites_2=sites_2_path,
            output=out_path,
            format="parquet",
            gzip=False,
            verbose=False,
        )
        main(args)

        table = pq.read_table(out_path)
        assert len(table) == 1
        r = {c: table.column(c)[0].as_py() for c in table.column_names}
        assert r["gene_id"] == "G1"
        assert r["delta_mod_level"] == pytest.approx(0.75)
        assert r["log2_or"] < 0  # cond1 lower
        assert 0 <= r["q_value"] <= 1.0
        assert 0 <= r["w_q_value"] <= 1.0

    def test_end_to_end_empty_result(self, tmp_path):
        """No matched gene-sites → schema-only output."""
        sites_1_path = str(tmp_path / "g1.parquet")
        sites_2_path = str(tmp_path / "g2.parquet")
        out_path = str(tmp_path / "out.parquet")

        _write_gene_parquet(
            sites_1_path,
            [
                _make_site(gene_id="G1", gpos=100),
            ],
        )
        _write_gene_parquet(
            sites_2_path,
            [
                _make_site(gene_id="G2", gpos=200),  # different gene
            ],
        )

        args = argparse.Namespace(
            sites_1=sites_1_path,
            sites_2=sites_2_path,
            output=out_path,
            format="parquet",
            gzip=False,
            verbose=False,
        )
        main(args)

        table = pq.read_table(out_path)
        assert len(table) == 0
        assert "w_log2_or" in table.column_names

    def test_end_to_end_tsv(self, tmp_path):
        """TSV output format works."""
        sites_path = str(tmp_path / "g.parquet")
        out_path = str(tmp_path / "out.tsv")

        _write_gene_parquet(
            sites_path,
            [
                _make_site(n_modified=10, n_unmodified=90),
            ],
        )

        args = argparse.Namespace(
            sites_1=sites_path,
            sites_2=sites_path,
            output=out_path,
            format="tsv",
            gzip=False,
            verbose=False,
        )
        main(args)

        with open(out_path) as f:
            lines = f.readlines()
        assert len(lines) == 2
        assert "w_log2_or" in lines[0]

    def test_end_to_end_bh_correction(self, tmp_path):
        """Multiple tests get BH-corrected q-values."""
        sites_1_path = str(tmp_path / "g1.parquet")
        sites_2_path = str(tmp_path / "g2.parquet")
        out_path = str(tmp_path / "out.parquet")

        # Three gene-sites with varying effect sizes
        _write_gene_parquet(
            sites_1_path,
            [
                _make_site(
                    gene_id="G1", gpos=100, mod_type="a", n_modified=0, n_unmodified=100
                ),
                _make_site(
                    gene_id="G2", gpos=200, mod_type="a", n_modified=30, n_unmodified=70
                ),
                _make_site(
                    gene_id="G3", gpos=300, mod_type="a", n_modified=45, n_unmodified=55
                ),
            ],
        )
        _write_gene_parquet(
            sites_2_path,
            [
                _make_site(
                    gene_id="G1", gpos=100, mod_type="a", n_modified=100, n_unmodified=0
                ),
                _make_site(
                    gene_id="G2", gpos=200, mod_type="a", n_modified=70, n_unmodified=30
                ),
                _make_site(
                    gene_id="G3", gpos=300, mod_type="a", n_modified=55, n_unmodified=45
                ),
            ],
        )

        args = argparse.Namespace(
            sites_1=sites_1_path,
            sites_2=sites_2_path,
            output=out_path,
            format="parquet",
            gzip=False,
            verbose=False,
        )
        main(args)

        table = pq.read_table(out_path)
        assert len(table) == 3
        q_vals = table.column("q_value").to_pylist()
        w_q_vals = table.column("w_q_value").to_pylist()
        # All q-values should be in [0, 1]
        for q in q_vals:
            assert 0.0 <= q <= 1.0
        for q in w_q_vals:
            assert 0.0 <= q <= 1.0
        # At least one should be significant (G1: 0 vs 100)
        assert any(q < 0.05 for q in q_vals if not np.isnan(q))
