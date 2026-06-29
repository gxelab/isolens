"""Tests for mod_dmc — differential modification calling."""

import argparse
import os
import sys

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

try:
    from isolens.mod_dmc import (
        _extract_site_reads,
        _read_mod_codes,
        _validate_mod_codes,
        _validate_tx_lengths,
        _write_parquet,
        _write_tsv,
        main,
        parse_args,
        process_transcript,
        read_site_summary_full,
    )
    from isolens.mod_scan import (
        CODE_CANONICAL,
        CODE_DELETION,
        CODE_FAIL,
        CODE_MISMATCH,
        CODE_UNCOVERED,
    )
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from isolens.mod_dmc import (  # type: ignore[no-redef]
        _extract_site_reads,
        _read_mod_codes,
        _validate_mod_codes,
        _validate_tx_lengths,
        _write_parquet,
        _write_tsv,
        main,
        parse_args,
        process_transcript,
        read_site_summary_full,
    )
    from isolens.mod_scan import (  # type: ignore[no-redef]
        CODE_CANONICAL,
        CODE_DELETION,
        CODE_FAIL,
        CODE_MISMATCH,
        CODE_UNCOVERED,
    )


# ---------- helpers ----------


def _make_h5(path, tx_data, mod_codes):
    """Create a minimal HDF5 file for testing.

    tx_data: dict of {tx_name: (matrix, weights)}.
    mod_codes: dict of {mod_str: code}.
    """
    with h5py.File(path, "w") as h5:
        codes_grp = h5.create_group("modification_codes")
        for mod_str, code in mod_codes.items():
            codes_grp.attrs[mod_str] = code

        for tx_name, (matrix, weights) in tx_data.items():
            grp = h5.create_group(f"transcripts/{tx_name}")
            grp.create_dataset(
                "matrix", data=matrix, dtype=np.uint8, compression="gzip"
            )
            grp.create_dataset(
                "read_weights",
                data=weights,
                dtype=np.float32,
                compression="gzip",
            )
            grp.create_dataset(
                "read_ids",
                data=np.array(
                    [f"read_{i}" for i in range(len(weights))],
                    dtype=h5py.string_dtype(),
                ),
            )


def _make_sites_parquet(path, sites_data):
    """Create a minimal site-summary Parquet file.

    sites_data: list of dicts with keys matching mod_sites output columns.
    """
    cols = [
        "transcript_id",
        "position",
        "mod_type",
        "n_modified",
        "wt_modified",
        "n_unmodified",
        "wt_unmodified",
        "n_canonical",
        "wt_canonical",
        "n_othermod",
        "wt_othermod",
        "n_mismatch",
        "wt_mismatch",
        "n_deletion",
        "wt_deletion",
        "n_failed",
        "wt_failed",
        "mod_level",
        "wt_mod_level",
        "gene_id",
        "chrom",
        "strand",
        "gpos",
    ]
    arrays = {}
    for c in cols:
        values = [s.get(c, None) for s in sites_data]
        if c in ("transcript_id", "mod_type", "gene_id"):
            arrays[c] = pa.array(values)
        elif c in ("chrom", "strand"):
            arrays[c] = pa.array(values, type=pa.string())
        elif c == "position":
            arrays[c] = pa.array(values, type=pa.int32())
        elif c == "gpos":
            arrays[c] = pa.array(values, type=pa.int32())
        elif c.startswith("n_"):
            arrays[c] = pa.array(values, type=pa.int32())
        else:
            arrays[c] = pa.array(values, type=pa.float64())
    pq.write_table(pa.table(arrays), path)


# ---------- _extract_site_reads ----------


class TestExtractSiteReads:
    """Tests for _extract_site_reads()."""

    def test_simple_modified(self):
        """All reads have the modification at position 1."""
        matrix = np.array([[4, CODE_CANONICAL], [4, CODE_CANONICAL]], dtype=np.uint8)
        weights = np.array([1.0, 0.5], dtype=np.float32)
        y, w = _extract_site_reads(matrix, weights, 1, 4)
        assert len(y) == 2
        assert np.all(y == 1.0)
        assert w[0] == pytest.approx(1.0)
        assert w[1] == pytest.approx(0.5)

    def test_mixed_modified_unmodified(self):
        """Mix of modified and canonical reads."""
        matrix = np.array([[4], [CODE_CANONICAL], [4]], dtype=np.uint8)
        weights = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        y, w = _extract_site_reads(matrix, weights, 1, 4)
        assert len(y) == 3
        assert y.tolist() == [1.0, 0.0, 1.0]

    def test_filters_failed(self):
        """Failed reads are excluded."""
        matrix = np.array([[4], [CODE_FAIL], [CODE_CANONICAL]], dtype=np.uint8)
        weights = np.ones(3, dtype=np.float32)
        y, w = _extract_site_reads(matrix, weights, 1, 4)
        assert len(y) == 2
        assert y.tolist() == [1.0, 0.0]

    def test_filters_mismatch(self):
        """Mismatch reads are excluded."""
        matrix = np.array([[4], [CODE_MISMATCH]], dtype=np.uint8)
        weights = np.ones(2, dtype=np.float32)
        y, w = _extract_site_reads(matrix, weights, 1, 4)
        assert len(y) == 1
        assert y[0] == 1.0

    def test_filters_deletion(self):
        """Deletion reads are excluded."""
        matrix = np.array([[4], [CODE_DELETION]], dtype=np.uint8)
        weights = np.ones(2, dtype=np.float32)
        y, w = _extract_site_reads(matrix, weights, 1, 4)
        assert len(y) == 1
        assert y[0] == 1.0

    def test_filters_othermod(self):
        """Reads with a different modification are excluded."""
        matrix = np.array([[4], [5], [CODE_CANONICAL]], dtype=np.uint8)
        weights = np.ones(3, dtype=np.float32)
        y, w = _extract_site_reads(matrix, weights, 1, 4)
        # Read with mod_code=5 is excluded (othermod for code 4)
        assert len(y) == 2
        assert y.tolist() == [1.0, 0.0]

    def test_all_invalid(self):
        """All reads are invalid → None."""
        matrix = np.array([[CODE_FAIL], [CODE_FAIL]], dtype=np.uint8)
        weights = np.ones(2, dtype=np.float32)
        result = _extract_site_reads(matrix, weights, 1, 4)
        assert result is None

    def test_uncovered(self):
        """Uncovered positions are excluded."""
        matrix = np.array([[CODE_UNCOVERED], [4]], dtype=np.uint8)
        weights = np.ones(2, dtype=np.float32)
        y, w = _extract_site_reads(matrix, weights, 1, 4)
        assert len(y) == 1
        assert y[0] == 1.0


# ---------- read_site_summary_full ----------


class TestReadSiteSummaryFull:
    """Tests for read_site_summary_full()."""

    def test_parquet(self, tmp_path):
        path = str(tmp_path / "sites.parquet")
        _make_sites_parquet(
            path,
            [
                {
                    "transcript_id": "TX1",
                    "position": 10,
                    "mod_type": "a",
                    "n_modified": 5,
                    "wt_modified": 4.5,
                    "n_unmodified": 15,
                    "wt_unmodified": 14.0,
                    "n_canonical": 10,
                    "wt_canonical": 9.0,
                    "n_othermod": 5,
                    "wt_othermod": 5.0,
                    "n_mismatch": 2,
                    "wt_mismatch": 1.8,
                    "n_deletion": 1,
                    "wt_deletion": 0.9,
                    "n_failed": 0,
                    "wt_failed": 0.0,
                    "mod_level": 0.25,
                    "wt_mod_level": 0.2432,
                    "gene_id": "G1",
                    "chrom": "2L",
                    "strand": "+",
                    "gpos": 100,
                },
            ],
        )
        sites = read_site_summary_full(path)
        key = ("TX1", 10, "a")
        assert key in sites
        assert sites[key]["n_modified"] == 5
        assert sites[key]["gene_id"] == "G1"
        assert sites[key]["gpos"] == 100

    def test_tsv(self, tmp_path):
        path = str(tmp_path / "sites.tsv")
        with open(path, "w") as f:
            f.write(
                "transcript_id\tposition\tmod_type\t"
                "n_modified\twt_modified\tn_unmodified\twt_unmodified\t"
                "n_canonical\twt_canonical\tn_othermod\twt_othermod\t"
                "n_mismatch\twt_mismatch\tn_deletion\twt_deletion\t"
                "n_failed\twt_failed\tmod_level\twt_mod_level\t"
                "gene_id\tchrom\tstrand\tgpos\n"
            )
            f.write(
                "TX1\t10\ta\t5\t4.5\t15\t14.0\t"
                "10\t9.0\t5\t5.0\t2\t1.8\t1\t0.9\t"
                "0\t0.0\t0.25\t0.2432\t"
                "G1\t2L\t+\t100\n"
            )
        sites = read_site_summary_full(path)
        key = ("TX1", 10, "a")
        assert key in sites
        assert sites[key]["n_modified"] == 5
        assert sites[key]["gene_id"] == "G1"

    def test_tsv_na_values(self, tmp_path):
        """TSV with NA values for null genomic coordinates."""
        path = str(tmp_path / "sites.tsv")
        with open(path, "w") as f:
            f.write(
                "transcript_id\tposition\tmod_type\t"
                "n_modified\twt_modified\tn_unmodified\twt_unmodified\t"
                "mod_level\twt_mod_level\t"
                "gene_id\tchrom\tstrand\tgpos\n"
            )
            f.write("TX1\t5\tm\t3\t2.5\t7\t6.5\t0.3\t0.28\tNA\tNA\tNA\tNA\n")
        sites = read_site_summary_full(path)
        key = ("TX1", 5, "m")
        assert key in sites
        assert sites[key]["gene_id"] is None
        assert sites[key]["gpos"] is None

    def test_multiple_sites(self, tmp_path):
        path = str(tmp_path / "sites.parquet")
        _make_sites_parquet(
            path,
            [
                {
                    "transcript_id": "TX1",
                    "position": 10,
                    "mod_type": "a",
                    "n_modified": 5,
                    "wt_modified": 4.5,
                    "n_unmodified": 15,
                    "wt_unmodified": 14.0,
                    "n_canonical": 10,
                    "wt_canonical": 9.0,
                    "n_othermod": 5,
                    "wt_othermod": 5.0,
                    "n_mismatch": 0,
                    "wt_mismatch": 0.0,
                    "n_deletion": 0,
                    "wt_deletion": 0.0,
                    "n_failed": 0,
                    "wt_failed": 0.0,
                    "mod_level": 0.25,
                    "wt_mod_level": 0.2432,
                    "gene_id": "G1",
                    "chrom": "2L",
                    "strand": "+",
                    "gpos": 100,
                },
                {
                    "transcript_id": "TX1",
                    "position": 20,
                    "mod_type": "a",
                    "n_modified": 8,
                    "wt_modified": 7.0,
                    "n_unmodified": 12,
                    "wt_unmodified": 11.0,
                    "n_canonical": 8,
                    "wt_canonical": 7.5,
                    "n_othermod": 4,
                    "wt_othermod": 3.5,
                    "n_mismatch": 0,
                    "wt_mismatch": 0.0,
                    "n_deletion": 0,
                    "wt_deletion": 0.0,
                    "n_failed": 0,
                    "wt_failed": 0.0,
                    "mod_level": 0.4,
                    "wt_mod_level": 0.389,
                    "gene_id": "G1",
                    "chrom": "2L",
                    "strand": "+",
                    "gpos": 110,
                },
            ],
        )
        sites = read_site_summary_full(path)
        assert len(sites) == 2
        assert ("TX1", 10, "a") in sites
        assert ("TX1", 20, "a") in sites


# ---------- process_transcript ----------


class TestProcessTranscript:
    """Tests for process_transcript()."""

    def test_single_site_both_conditions(self):
        """One matched site with clear difference between conditions."""
        n = 10
        # Condition 1: all unmodified
        matrix_1 = np.array([[CODE_CANONICAL]] * n, dtype=np.uint8)
        weights_1 = np.ones(n, dtype=np.float32)

        # Condition 2: all modified (code 4)
        matrix_2 = np.array([[4]] * n, dtype=np.uint8)
        weights_2 = np.ones(n, dtype=np.float32)

        sites_1 = {
            ("TX1", 1, "a"): {
                "n_modified": 0,
                "wt_modified": 0.0,
                "n_unmodified": n,
                "wt_unmodified": float(n),
                "mod_level": 0.0,
                "wt_mod_level": 0.0,
                "gene_id": "G1",
                "chrom": "2L",
                "strand": "+",
                "gpos": 100,
            }
        }
        sites_2 = {
            ("TX1", 1, "a"): {
                "n_modified": n,
                "wt_modified": float(n),
                "n_unmodified": 0,
                "wt_unmodified": 0.0,
                "mod_level": 1.0,
                "wt_mod_level": 1.0,
                "gene_id": "G1",
                "chrom": "2L",
                "strand": "+",
                "gpos": 100,
            }
        }
        mod_code_map = {"a": 4}

        rows = process_transcript(
            "TX1",
            matrix_1,
            weights_1,
            matrix_2,
            weights_2,
            sites_1,
            sites_2,
            mod_code_map,
        )

        assert len(rows) == 1
        r = rows[0]
        assert r["transcript_id"] == "TX1"
        assert r["position"] == 1
        assert r["mod_type"] == "a"
        assert r["mod_level_1"] == 0.0
        assert r["mod_level_2"] == 1.0
        assert r["delta_mod_level"] == pytest.approx(1.0)
        # Condition 2 has higher modification → log2OR > 0
        assert r["log2_or"] > 0
        assert r["p_value"] < 0.05
        assert "q_value" in r

    def test_site_only_in_one_condition(self):
        """Site present in cond1 but not cond2 → no test."""
        matrix_1 = np.array([[4], [4]], dtype=np.uint8)
        weights_1 = np.ones(2, dtype=np.float32)
        matrix_2 = np.array([[CODE_CANONICAL]], dtype=np.uint8)
        weights_2 = np.ones(1, dtype=np.float32)

        sites_1 = {
            ("TX1", 1, "a"): {
                "n_modified": 2,
                "n_unmodified": 0,
                "mod_level": 1.0,
                "wt_mod_level": 1.0,
                "wt_modified": 2.0,
                "wt_unmodified": 0.0,
                "gene_id": None,
                "chrom": None,
                "strand": None,
                "gpos": None,
            }
        }
        sites_2 = {}  # No sites for this transcript in condition 2

        rows = process_transcript(
            "TX1",
            matrix_1,
            weights_1,
            matrix_2,
            weights_2,
            sites_1,
            sites_2,
            {"a": 4},
        )
        assert rows == []

    def test_no_valid_reads_in_one_condition(self):
        """All reads in condition 1 are failed → skip."""
        matrix_1 = np.array([[CODE_FAIL], [CODE_FAIL]], dtype=np.uint8)
        weights_1 = np.ones(2, dtype=np.float32)
        matrix_2 = np.array([[4], [4]], dtype=np.uint8)
        weights_2 = np.ones(2, dtype=np.float32)

        sites_1 = {
            ("TX1", 1, "a"): {
                "n_modified": 0,
                "n_unmodified": 0,
                "mod_level": 0.0,
                "wt_mod_level": 0.0,
                "wt_modified": 0.0,
                "wt_unmodified": 0.0,
                "gene_id": None,
                "chrom": None,
                "strand": None,
                "gpos": None,
            }
        }
        sites_2 = {
            ("TX1", 1, "a"): {
                "n_modified": 2,
                "n_unmodified": 0,
                "mod_level": 1.0,
                "wt_mod_level": 1.0,
                "wt_modified": 2.0,
                "wt_unmodified": 0.0,
                "gene_id": None,
                "chrom": None,
                "strand": None,
                "gpos": None,
            }
        }

        rows = process_transcript(
            "TX1",
            matrix_1,
            weights_1,
            matrix_2,
            weights_2,
            sites_1,
            sites_2,
            {"a": 4},
        )
        assert rows == []

    def test_no_sites_in_common(self):
        """Different site keys between conditions."""
        matrix_1 = np.array([[4]], dtype=np.uint8)
        weights_1 = np.ones(1, dtype=np.float32)
        matrix_2 = np.array([[4]], dtype=np.uint8)
        weights_2 = np.ones(1, dtype=np.float32)

        sites_1 = {
            ("TX1", 1, "a"): {
                "n_modified": 1,
                "n_unmodified": 0,
                "mod_level": 1.0,
                "wt_mod_level": 1.0,
                "wt_modified": 1.0,
                "wt_unmodified": 0.0,
                "gene_id": None,
                "chrom": None,
                "strand": None,
                "gpos": None,
            }
        }
        sites_2 = {
            ("TX1", 2, "a"): {  # Different position
                "n_modified": 1,
                "n_unmodified": 0,
                "mod_level": 1.0,
                "wt_mod_level": 1.0,
                "wt_modified": 1.0,
                "wt_unmodified": 0.0,
                "gene_id": None,
                "chrom": None,
                "strand": None,
                "gpos": None,
            }
        }

        rows = process_transcript(
            "TX1",
            matrix_1,
            weights_1,
            matrix_2,
            weights_2,
            sites_1,
            sites_2,
            {"a": 4},
        )
        assert rows == []

    def test_output_columns_complete(self):
        """Verify all expected output columns are present."""
        matrix_1 = np.array([[4], [CODE_CANONICAL]], dtype=np.uint8)
        weights_1 = np.ones(2, dtype=np.float32)
        matrix_2 = np.array([[CODE_CANONICAL], [CODE_CANONICAL]], dtype=np.uint8)
        weights_2 = np.ones(2, dtype=np.float32)

        site = {
            "n_modified": 1,
            "wt_modified": 0.8,
            "n_unmodified": 1,
            "wt_unmodified": 0.9,
            "mod_level": 0.5,
            "wt_mod_level": 0.471,
            "gene_id": "G1",
            "chrom": "2L",
            "strand": "+",
            "gpos": 100,
        }

        rows = process_transcript(
            "TX1",
            matrix_1,
            weights_1,
            matrix_2,
            weights_2,
            {("TX1", 1, "a"): site},
            {
                ("TX1", 1, "a"): {
                    "n_modified": 0,
                    "wt_modified": 0.0,
                    "n_unmodified": 2,
                    "wt_unmodified": 1.8,
                    "mod_level": 0.0,
                    "wt_mod_level": 0.0,
                    "gene_id": "G1",
                    "chrom": "2L",
                    "strand": "+",
                    "gpos": 100,
                }
            },
            {"a": 4},
        )

        assert len(rows) == 1
        r = rows[0]
        expected_cols = {
            "transcript_id",
            "position",
            "mod_type",
            "gene_id",
            "chrom",
            "strand",
            "gpos",
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
        }
        assert set(r.keys()) == expected_cols

    def test_unknown_mod_type_skipped(self):
        """Site with mod_type not in mod_code_map is skipped."""
        matrix_1 = np.array([[4]], dtype=np.uint8)
        weights_1 = np.ones(1, dtype=np.float32)
        matrix_2 = np.array([[4]], dtype=np.uint8)
        weights_2 = np.ones(1, dtype=np.float32)

        site = {
            "n_modified": 1,
            "n_unmodified": 0,
            "mod_level": 1.0,
            "wt_mod_level": 1.0,
            "wt_modified": 1.0,
            "wt_unmodified": 0.0,
            "gene_id": None,
            "chrom": None,
            "strand": None,
            "gpos": None,
        }

        rows = process_transcript(
            "TX1",
            matrix_1,
            weights_1,
            matrix_2,
            weights_2,
            {("TX1", 1, "unknown"): site},
            {("TX1", 1, "unknown"): site},
            {"a": 4},  # "unknown" not in map
        )
        assert rows == []


# ---------- output writers ----------


class TestWriteOutput:
    """Tests for _write_parquet and _write_tsv."""

    def _make_row(self, **overrides):
        defaults = {
            "transcript_id": "TX1",
            "position": 1,
            "mod_type": "a",
            "gene_id": "G1",
            "chrom": "2L",
            "strand": "+",
            "gpos": 100,
            "n_modified_1": 5,
            "n_unmodified_1": 15,
            "n_modified_2": 10,
            "n_unmodified_2": 10,
            "wt_modified_1": 4.5,
            "wt_unmodified_1": 14.0,
            "wt_modified_2": 9.0,
            "wt_unmodified_2": 9.0,
            "mod_level_1": 0.25,
            "mod_level_2": 0.5,
            "wt_mod_level_1": 0.243,
            "wt_mod_level_2": 0.5,
            "delta_mod_level": 0.25,
            "delta_wt_mod_level": 0.257,
            "log2_or": 1.5,
            "p_value": 0.01,
            "q_value": 0.05,
        }
        defaults.update(overrides)
        return defaults

    def test_write_parquet(self, tmp_path):
        path = str(tmp_path / "out.parquet")
        rows = [self._make_row()]
        _write_parquet(rows, path)
        table = pq.read_table(path)
        assert len(table) == 1
        assert table.column("log2_or")[0].as_py() == pytest.approx(1.5)

    def test_write_empty_parquet(self, tmp_path):
        path = str(tmp_path / "out.parquet")
        _write_parquet([], path)
        table = pq.read_table(path)
        assert len(table) == 0

    def test_write_tsv(self, tmp_path):
        path = str(tmp_path / "out.tsv")
        rows = [self._make_row()]
        _write_tsv(rows, path, use_gzip=False)
        with open(path) as f:
            header = f.readline()
            data = f.readline()
        assert "transcript_id" in header
        assert "TX1" in data

    def test_write_tsv_null_values(self, tmp_path):
        path = str(tmp_path / "out.tsv")
        row = self._make_row(gene_id=None, gpos=None)
        _write_tsv([row], path, use_gzip=False)
        with open(path) as f:
            f.readline()  # header
            data = f.readline()
        assert "NA" in data


# ---------- HDF5 helpers ----------


class TestHDF5Helpers:
    """Tests for HDF5 utility functions."""

    def test_read_mod_codes(self, tmp_path):
        path = str(tmp_path / "test.h5")
        _make_h5(
            path,
            {
                "TX1": (
                    np.array([[4]], dtype=np.uint8),
                    np.array([1.0], dtype=np.float32),
                )
            },
            {"a": 4, "m": 5},
        )
        with h5py.File(path, "r") as h5:
            codes = _read_mod_codes(h5)
        assert codes == {"a": 4, "m": 5}

    def test_validate_mod_codes_match(self):
        codes = _validate_mod_codes(
            [{"a": 4, "m": 5}, {"a": 4, "m": 5}],
            ["f1.h5", "f2.h5"],
        )
        assert codes == {"a": 4, "m": 5}

    def test_validate_mod_codes_mismatch(self):
        with pytest.raises(ValueError):
            _validate_mod_codes(
                [{"a": 4}, {"a": 5}],
                ["f1.h5", "f2.h5"],
            )

    def test_validate_tx_lengths_consistent(self):
        length = _validate_tx_lengths(
            "TX1", [100, 100, None], ["f1.h5", "f2.h5", "f3.h5"]
        )
        assert length == 100

    def test_validate_tx_lengths_inconsistent(self):
        with pytest.raises(ValueError):
            _validate_tx_lengths("TX1", [100, 200], ["f1.h5", "f2.h5"])


# ---------- CLI ----------


class TestCLI:
    """Tests for parse_args()."""

    def test_required_args(self):
        argv = [
            "--h5-1",
            "a.h5",
            "--h5-2",
            "b.h5",
            "--sites-1",
            "a.parquet",
            "--sites-2",
            "b.parquet",
            "-o",
            "out.parquet",
        ]
        import sys as _sys

        _sys.argv = ["mod_dmc"] + argv
        args = parse_args()
        assert args.h5_1 == ["a.h5"]
        assert args.h5_2 == ["b.h5"]
        assert args.sites_1 == "a.parquet"
        assert args.format == "parquet"

    def test_multi_h5(self):
        argv = [
            "--h5-1",
            "a1.h5",
            "a2.h5",
            "--h5-2",
            "b1.h5",
            "b2.h5",
            "--sites-1",
            "a.parquet",
            "--sites-2",
            "b.parquet",
            "-o",
            "out.parquet",
        ]
        import sys as _sys

        _sys.argv = ["mod_dmc"] + argv
        args = parse_args()
        assert args.h5_1 == ["a1.h5", "a2.h5"]
        assert args.h5_2 == ["b1.h5", "b2.h5"]

    def test_optional_args(self):
        argv = [
            "--h5-1",
            "a.h5",
            "--h5-2",
            "b.h5",
            "--sites-1",
            "a.parquet",
            "--sites-2",
            "b.parquet",
            "-o",
            "out.tsv",
            "-f",
            "tsv",
            "-z",
            "-p",
            "0.5",
            "-x",
            "TX1",
            "TX2",
            "-v",
        ]
        import sys as _sys

        _sys.argv = ["mod_dmc"] + argv
        args = parse_args()
        assert args.format == "tsv"
        assert args.gzip is True
        assert args.min_asp == 0.5
        assert args.transcripts == ["TX1", "TX2"]
        assert args.verbose is True


# ---------- integration tests ----------


class TestMainIntegration:
    """End-to-end integration tests for main()."""

    @staticmethod
    def _make_h5_file(path, tx_data, mod_codes):
        _make_h5(path, tx_data, mod_codes)

    @staticmethod
    def _make_sites_file(path, sites_data):
        _make_sites_parquet(path, sites_data)

    def _make_site_row(
        self, tx, pos, mod, n_mod, n_unmod, mod_level, gene_id=None, gpos=None
    ):
        return {
            "transcript_id": tx,
            "position": pos,
            "mod_type": mod,
            "n_modified": n_mod,
            "wt_modified": float(n_mod),
            "n_unmodified": n_unmod,
            "wt_unmodified": float(n_unmod),
            "n_canonical": n_unmod,
            "wt_canonical": float(n_unmod),
            "n_othermod": 0,
            "wt_othermod": 0.0,
            "n_mismatch": 0,
            "wt_mismatch": 0.0,
            "n_deletion": 0,
            "wt_deletion": 0.0,
            "n_failed": 0,
            "wt_failed": 0.0,
            "mod_level": mod_level,
            "wt_mod_level": mod_level,
            "gene_id": gene_id,
            "chrom": "2L" if gene_id else None,
            "strand": "+" if gene_id else None,
            "gpos": gpos,
        }

    def test_end_to_end_parquet(self, tmp_path):
        """Full pipeline with two conditions produces valid output."""
        # Condition 1: mostly unmodified (1 modified out of 5)
        matrix_1 = np.array(
            [
                [4],
                [CODE_CANONICAL],
                [CODE_CANONICAL],
                [CODE_CANONICAL],
                [CODE_CANONICAL],
            ],
            dtype=np.uint8,
        )
        weights_1 = np.ones(5, dtype=np.float32)

        # Condition 2: mostly modified (4 modified out of 5)
        matrix_2 = np.array(
            [[4], [4], [4], [4], [CODE_CANONICAL]],
            dtype=np.uint8,
        )
        weights_2 = np.ones(5, dtype=np.float32)

        h5_1_path = str(tmp_path / "cond1.h5")
        h5_2_path = str(tmp_path / "cond2.h5")
        sites_1_path = str(tmp_path / "sites1.parquet")
        sites_2_path = str(tmp_path / "sites2.parquet")
        out_path = str(tmp_path / "out.parquet")

        self._make_h5_file(h5_1_path, {"TX1": (matrix_1, weights_1)}, {"a": 4})
        self._make_h5_file(h5_2_path, {"TX1": (matrix_2, weights_2)}, {"a": 4})
        self._make_sites_file(
            sites_1_path, [self._make_site_row("TX1", 1, "a", 1, 4, 0.2, "G1", 100)]
        )
        self._make_sites_file(
            sites_2_path, [self._make_site_row("TX1", 1, "a", 4, 1, 0.8, "G1", 100)]
        )

        args = argparse.Namespace(
            h5_1=[h5_1_path],
            h5_2=[h5_2_path],
            sites_1=sites_1_path,
            sites_2=sites_2_path,
            output=out_path,
            format="parquet",
            gzip=False,
            min_asp=0.0,
            transcripts=None,
            verbose=False,
        )
        main(args)

        table = pq.read_table(out_path)
        assert len(table) == 1
        r = {c: table.column(c)[0].as_py() for c in table.column_names}
        assert r["transcript_id"] == "TX1"
        assert r["delta_mod_level"] == pytest.approx(0.6)
        # log2OR should be positive (cond2 has more modification)
        assert r["log2_or"] > 0
        assert 0 <= r["q_value"] <= 1.0

    def test_end_to_end_empty_result(self, tmp_path):
        """No matched sites → write empty schema-only file."""
        h5_1_path = str(tmp_path / "cond1.h5")
        h5_2_path = str(tmp_path / "cond2.h5")
        sites_1_path = str(tmp_path / "sites1.parquet")
        sites_2_path = str(tmp_path / "sites2.parquet")
        out_path = str(tmp_path / "out.parquet")

        self._make_h5_file(
            h5_1_path,
            {
                "TX1": (
                    np.array([[4]], dtype=np.uint8),
                    np.array([1.0], dtype=np.float32),
                )
            },
            {"a": 4},
        )
        self._make_h5_file(
            h5_2_path,
            {
                "TX1": (
                    np.array([[CODE_CANONICAL]], dtype=np.uint8),
                    np.array([1.0], dtype=np.float32),
                )
            },
            {"a": 4},
        )
        # Sites in different transcripts → no match
        self._make_sites_file(
            sites_1_path, [self._make_site_row("TX1", 1, "a", 1, 0, 1.0)]
        )
        self._make_sites_file(sites_2_path, [])  # No sites in cond2

        args = argparse.Namespace(
            h5_1=[h5_1_path],
            h5_2=[h5_2_path],
            sites_1=sites_1_path,
            sites_2=sites_2_path,
            output=out_path,
            format="parquet",
            gzip=False,
            min_asp=0.0,
            transcripts=None,
            verbose=False,
        )
        main(args)

        table = pq.read_table(out_path)
        assert len(table) == 0

    def test_end_to_end_tsv(self, tmp_path):
        """TSV output format works."""
        matrix = np.array([[4]], dtype=np.uint8)
        weights = np.ones(1, dtype=np.float32)

        h5_1_path = str(tmp_path / "c1.h5")
        h5_2_path = str(tmp_path / "c2.h5")
        sites_path = str(tmp_path / "s.parquet")
        out_path = str(tmp_path / "out.tsv")

        self._make_h5_file(h5_1_path, {"TX1": (matrix, weights)}, {"a": 4})
        self._make_h5_file(
            h5_2_path,
            {
                "TX1": (
                    np.array([[CODE_CANONICAL]], dtype=np.uint8),
                    np.ones(1, dtype=np.float32),
                )
            },
            {"a": 4},
        )
        self._make_sites_file(
            sites_path,
            [
                self._make_site_row("TX1", 1, "a", 1, 0, 1.0),
            ],
        )

        args = argparse.Namespace(
            h5_1=[h5_1_path],
            h5_2=[h5_2_path],
            sites_1=sites_path,
            sites_2=sites_path,
            output=out_path,
            format="tsv",
            gzip=False,
            min_asp=0.0,
            transcripts=None,
            verbose=False,
        )
        main(args)

        with open(out_path) as f:
            lines = f.readlines()
        assert len(lines) == 2  # header + 1 row
        assert "transcript_id" in lines[0]

    def test_transcripts_filter(self, tmp_path):
        """--transcripts filter limits processing."""
        matrix = np.array([[4]], dtype=np.uint8)
        weights = np.ones(1, dtype=np.float32)

        h5_1_path = str(tmp_path / "c1.h5")
        h5_2_path = str(tmp_path / "c2.h5")
        sites_path = str(tmp_path / "s.parquet")
        out_path = str(tmp_path / "out.parquet")

        self._make_h5_file(
            h5_1_path,
            {"TX1": (matrix, weights)},
            {"a": 4},
        )
        self._make_h5_file(
            h5_2_path,
            {"TX1": (matrix, weights)},
            {"a": 4},
        )
        self._make_sites_file(
            sites_path,
            [
                self._make_site_row("TX1", 1, "a", 1, 0, 1.0),
            ],
        )

        # Request a transcript not in the data
        args = argparse.Namespace(
            h5_1=[h5_1_path],
            h5_2=[h5_2_path],
            sites_1=sites_path,
            sites_2=sites_path,
            output=out_path,
            format="parquet",
            gzip=False,
            min_asp=0.0,
            transcripts=["TX2"],
            verbose=False,
        )
        main(args)

        table = pq.read_table(out_path)
        assert len(table) == 0
