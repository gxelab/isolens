"""Tests for mod_dmt — differential modification testing between isoforms."""

import argparse
import os
import sys

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

try:
    from isolens._hdf5_helpers import (
        extract_site_reads,
        read_mod_codes,
        validate_mod_codes,
    )
    from isolens._io import write_parquet, write_tsv
    from isolens.mod_dmt import (
        _DMT_SCHEMA,
        _OUTPUT_COLS,
        _TSV_HEADER,
        main,
        parse_args,
        process_locus_group,
        read_sites_grouped_by_locus,
        validate_input,
    )
    from isolens.mod_scan import (
        CODE_CANONICAL,
        CODE_FAIL,
    )
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from isolens._hdf5_helpers import (  # type: ignore[no-redef]
        extract_site_reads,
        read_mod_codes,
        validate_mod_codes,
    )
    from isolens._io import (  # type: ignore[no-redef]
        write_parquet,
        write_tsv,
    )
    from isolens.mod_dmt import (  # type: ignore[no-redef]
        _DMT_SCHEMA,
        _OUTPUT_COLS,
        _TSV_HEADER,
        main,
        parse_args,
        process_locus_group,
        read_sites_grouped_by_locus,
        validate_input,
    )
    from isolens.mod_scan import (  # type: ignore[no-redef]
        CODE_CANONICAL,
        CODE_FAIL,
    )


# ---------- helpers ----------


def _make_h5(path, tx_data, mod_codes):
    """Create a minimal HDF5 file for testing."""
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
    """Create a minimal site-summary Parquet file for DMT testing.

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


def _make_site(
    tx,
    pos,
    mod,
    n_mod,
    n_unmod,
    mod_level,
    gene_id="G1",
    chrom="2L",
    strand="+",
    gpos=100,
):
    """Create a site dict with all required fields."""
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
        "chrom": chrom,
        "strand": strand,
        "gpos": gpos,
    }


# ---------- read_sites_grouped_by_locus ----------


class TestReadSitesGroupedByLocus:
    """Tests for read_sites_grouped_by_locus()."""

    def test_groups_by_locus(self, tmp_path):
        """Sites at the same genomic locus are grouped together."""
        path = str(tmp_path / "sites.parquet")
        _make_sites_parquet(
            path,
            [
                _make_site("TX1", 10, "a", 5, 15, 0.25),
                _make_site("TX2", 20, "a", 8, 12, 0.4),
            ],
        )
        groups = read_sites_grouped_by_locus(path)
        assert len(groups) == 1
        key = ("G1", "2L", 100, "+", "a")
        assert key in groups
        assert len(groups[key]) == 2

    def test_different_genomic_positions_not_grouped(self, tmp_path):
        """Sites at different genomic positions are separate groups.

        Each genomic position needs ≥2 transcripts to form a group.
        """
        path = str(tmp_path / "sites.parquet")
        _make_sites_parquet(
            path,
            [
                _make_site("TX1", 1, "a", 5, 15, 0.25, gpos=100),
                _make_site("TX2", 1, "a", 8, 12, 0.4, gpos=100),
                _make_site("TX3", 1, "a", 3, 17, 0.15, gpos=200),
                _make_site("TX4", 1, "a", 6, 14, 0.3, gpos=200),
            ],
        )
        groups = read_sites_grouped_by_locus(path)
        assert len(groups) == 2
        keys = sorted(groups.keys(), key=lambda k: k[2])  # sort by gpos
        assert keys[0][2] == 100
        assert keys[1][2] == 200

    def test_different_mod_types_not_grouped(self, tmp_path):
        """Different modification types create separate groups.

        Each (genomic_pos, mod_type) needs ≥2 transcripts to form a group.
        """
        path = str(tmp_path / "sites.parquet")
        _make_sites_parquet(
            path,
            [
                _make_site("TX1", 1, "a", 5, 15, 0.25),
                _make_site("TX2", 1, "a", 8, 12, 0.4),
                _make_site("TX3", 1, "m", 3, 17, 0.15),
                _make_site("TX4", 1, "m", 6, 14, 0.3),
            ],
        )
        groups = read_sites_grouped_by_locus(path)
        assert len(groups) == 2
        mod_types = {k[4] for k in groups}
        assert mod_types == {"a", "m"}

    def test_single_transcript_dropped(self, tmp_path):
        """Groups with only one transcript are excluded."""
        path = str(tmp_path / "sites.parquet")
        _make_sites_parquet(
            path,
            [
                _make_site("TX1", 10, "a", 5, 15, 0.25),
            ],
        )
        groups = read_sites_grouped_by_locus(path)
        assert len(groups) == 0

    def test_null_genomic_coords_dropped(self, tmp_path):
        """Rows with null gene_id or gpos are dropped."""
        path = str(tmp_path / "sites.parquet")
        _make_sites_parquet(
            path,
            [
                _make_site("TX1", 10, "a", 5, 15, 0.25, gene_id=None, gpos=None),
                _make_site("TX2", 20, "a", 8, 12, 0.4, gene_id=None, gpos=None),
            ],
        )
        groups = read_sites_grouped_by_locus(path)
        assert len(groups) == 0

    def test_three_transcripts_at_locus(self, tmp_path):
        """Three transcripts at the same locus → 3 pairs."""
        path = str(tmp_path / "sites.parquet")
        _make_sites_parquet(
            path,
            [
                _make_site("TX1", 10, "a", 5, 15, 0.25),
                _make_site("TX2", 20, "a", 8, 12, 0.4),
                _make_site("TX3", 30, "a", 3, 17, 0.15),
            ],
        )
        groups = read_sites_grouped_by_locus(path)
        assert len(groups) == 1
        key = ("G1", "2L", 100, "+", "a")
        assert len(groups[key]) == 3

    def test_parquet_and_tsv(self, tmp_path):
        """Both Parquet and TSV input formats work."""
        # TSV
        tsv_path = str(tmp_path / "sites.tsv")
        with open(tsv_path, "w") as f:
            f.write(
                "transcript_id\tposition\tmod_type\t"
                "n_modified\twt_modified\tn_unmodified\twt_unmodified\t"
                "mod_level\twt_mod_level\t"
                "gene_id\tchrom\tstrand\tgpos\n"
            )
            f.write("TX1\t10\ta\t5\t5.0\t15\t15.0\t0.25\t0.25\tG1\t2L\t+\t100\n")
            f.write("TX2\t20\ta\t8\t8.0\t12\t12.0\t0.4\t0.4\tG1\t2L\t+\t100\n")
        groups = read_sites_grouped_by_locus(tsv_path)
        assert len(groups) == 1
        key = ("G1", "2L", 100, "+", "a")
        assert len(groups[key]) == 2


# ---------- validate_input ----------


class TestValidateInput:
    """Tests for validate_input()."""

    def test_non_empty_groups_passes(self):
        validate_input({("G1", "2L", 100, "+", "a"): [{"transcript_id": "TX1"}]})

    def test_empty_groups_exits(self):
        with pytest.raises(SystemExit):
            validate_input({})


# ---------- _extract_site_reads ----------


class TestExtractSiteReads:
    """Tests for _extract_site_reads (same logic as mod_dmc)."""

    def test_simple_modified(self):
        matrix = np.array([[4, CODE_CANONICAL], [4, CODE_CANONICAL]], dtype=np.uint8)
        weights = np.array([1.0, 0.5], dtype=np.float32)
        y, w = extract_site_reads(matrix, weights, 1, 4)
        assert len(y) == 2
        assert np.all(y == 1.0)

    def test_mixed(self):
        matrix = np.array([[4], [CODE_CANONICAL], [4]], dtype=np.uint8)
        weights = np.ones(3, dtype=np.float32)
        y, w = extract_site_reads(matrix, weights, 1, 4)
        assert y.tolist() == [1.0, 0.0, 1.0]

    def test_all_invalid(self):
        matrix = np.array([[CODE_FAIL], [CODE_FAIL]], dtype=np.uint8)
        weights = np.ones(2, dtype=np.float32)
        assert extract_site_reads(matrix, weights, 1, 4) is None


# ---------- process_locus_group ----------


class TestProcessLocusGroup:
    """Tests for process_locus_group()."""

    def test_two_transcripts_different_mod(self):
        """Two transcripts with clearly different modification patterns."""
        # TX1 at pos 1: all unmodified
        matrix_1 = np.array([[CODE_CANONICAL]] * 5, dtype=np.uint8)
        weights_1 = np.ones(5, dtype=np.float32)

        # TX2 at pos 1: all modified (code 4)
        matrix_2 = np.array([[4]] * 5, dtype=np.uint8)
        weights_2 = np.ones(5, dtype=np.float32)

        h5_data = {
            "TX1": (matrix_1, weights_1, 50),
            "TX2": (matrix_2, weights_2, 50),
        }

        tx_site_list = [
            _make_site("TX1", 1, "a", 0, 5, 0.0),
            _make_site("TX2", 1, "a", 5, 0, 1.0),
        ]

        rows = process_locus_group(
            ("G1", "2L", 100, "+", "a"),
            tx_site_list,
            h5_data,
            {"a": 4},
        )

        assert len(rows) == 1
        r = rows[0]
        assert r["gene_id"] == "G1"
        assert r["transcript_id_1"] == "TX1"
        assert r["transcript_id_2"] == "TX2"
        assert r["mod_level_1"] == 0.0
        assert r["mod_level_2"] == 1.0
        assert r["delta_mod_level"] == pytest.approx(1.0)
        # TX2 has higher modification → log2OR > 0
        assert r["log2_or"] > 0
        assert r["p_value"] < 0.05

    def test_three_transcripts_three_pairs(self):
        """Three transcripts → 3 choose 2 = 3 pairs."""
        matrix = np.array([[4]] * 3, dtype=np.uint8)
        weights = np.ones(3, dtype=np.float32)

        h5_data = {
            "TX1": (matrix, weights, 50),
            "TX2": (matrix, weights, 50),
            "TX3": (matrix, weights, 50),
        }

        tx_site_list = [
            _make_site("TX1", 1, "a", 3, 0, 1.0),
            _make_site("TX2", 1, "a", 3, 0, 1.0),
            _make_site("TX3", 1, "a", 3, 0, 1.0),
        ]

        rows = process_locus_group(
            ("G1", "2L", 100, "+", "a"),
            tx_site_list,
            h5_data,
            {"a": 4},
        )

        assert len(rows) == 3
        # All pairs should have both transcripts' info
        transcript_pairs = {(r["transcript_id_1"], r["transcript_id_2"]) for r in rows}
        assert ("TX1", "TX2") in transcript_pairs
        assert ("TX1", "TX3") in transcript_pairs
        assert ("TX2", "TX3") in transcript_pairs

    def test_transcript_not_in_h5_data(self):
        """Transcript missing from HDF5 → excluded from pairs."""
        matrix = np.array([[4]] * 3, dtype=np.uint8)
        weights = np.ones(3, dtype=np.float32)

        h5_data = {
            "TX1": (matrix, weights, 50),
            # TX2 not loaded
        }

        tx_site_list = [
            _make_site("TX1", 1, "a", 3, 0, 1.0),
            _make_site("TX2", 1, "a", 3, 0, 1.0),
        ]

        rows = process_locus_group(
            ("G1", "2L", 100, "+", "a"),
            tx_site_list,
            h5_data,
            {"a": 4},
        )

        # Only 1 transcript available → no pairs
        assert rows == []

    def test_zero_valid_reads_skips_pair(self):
        """Transcript with no valid reads at its position → pair skipped."""
        matrix_1 = np.array([[CODE_FAIL]] * 5, dtype=np.uint8)
        weights_1 = np.ones(5, dtype=np.float32)
        matrix_2 = np.array([[4]] * 5, dtype=np.uint8)
        weights_2 = np.ones(5, dtype=np.float32)

        h5_data = {
            "TX1": (matrix_1, weights_1, 50),
            "TX2": (matrix_2, weights_2, 50),
        }

        tx_site_list = [
            _make_site("TX1", 1, "a", 0, 0, 0.0),
            _make_site("TX2", 1, "a", 5, 0, 1.0),
        ]

        rows = process_locus_group(
            ("G1", "2L", 100, "+", "a"),
            tx_site_list,
            h5_data,
            {"a": 4},
        )

        assert rows == []

    def test_unknown_mod_type_skipped(self):
        """mod_type not in mod_code_map → no testing."""
        matrix = np.array([[4]] * 3, dtype=np.uint8)
        weights = np.ones(3, dtype=np.float32)

        h5_data = {
            "TX1": (matrix, weights, 50),
            "TX2": (matrix, weights, 50),
        }

        tx_site_list = [
            _make_site("TX1", 1, "unknown", 3, 0, 1.0),
            _make_site("TX2", 1, "unknown", 3, 0, 1.0),
        ]

        rows = process_locus_group(
            ("G1", "2L", 100, "+", "unknown"),
            tx_site_list,
            h5_data,
            {"a": 4},  # "unknown" not in map
        )

        assert rows == []

    def test_output_columns_complete(self):
        """All expected output columns are present."""
        matrix = np.array([[4]] * 3, dtype=np.uint8)
        weights = np.ones(3, dtype=np.float32)

        h5_data = {
            "TX1": (matrix, weights, 50),
            "TX2": (
                np.array([[CODE_CANONICAL]] * 3, dtype=np.uint8),
                np.ones(3, dtype=np.float32),
                50,
            ),
        }

        tx_site_list = [
            _make_site("TX1", 1, "a", 3, 0, 1.0),
            _make_site("TX2", 1, "a", 0, 3, 0.0),
        ]

        rows = process_locus_group(
            ("G1", "2L", 100, "+", "a"),
            tx_site_list,
            h5_data,
            {"a": 4},
        )

        assert len(rows) == 1
        expected_cols = {
            "gene_id",
            "chrom",
            "gpos",
            "strand",
            "mod_type",
            "transcript_id_1",
            "transcript_id_2",
            "position_1",
            "position_2",
            "mod_level_1",
            "mod_level_2",
            "wt_mod_level_1",
            "wt_mod_level_2",
            "delta_mod_level",
            "delta_wt_mod_level",
            "n_modified_1",
            "n_unmodified_1",
            "n_modified_2",
            "n_unmodified_2",
            "wt_modified_1",
            "wt_unmodified_1",
            "wt_modified_2",
            "wt_unmodified_2",
            "log2_or",
            "p_value",
            "q_value",
        }
        assert set(rows[0].keys()) == expected_cols


# ---------- output writers ----------


class TestWriteOutput:
    """Tests for _write_parquet and _write_tsv."""

    def _make_row(self, **overrides):
        defaults = {
            "gene_id": "G1",
            "chrom": "2L",
            "gpos": 100,
            "strand": "+",
            "mod_type": "a",
            "transcript_id_1": "TX1",
            "transcript_id_2": "TX2",
            "position_1": 10,
            "position_2": 20,
            "mod_level_1": 0.2,
            "mod_level_2": 0.8,
            "wt_mod_level_1": 0.19,
            "wt_mod_level_2": 0.81,
            "delta_mod_level": 0.6,
            "delta_wt_mod_level": 0.62,
            "n_modified_1": 2,
            "n_unmodified_1": 8,
            "n_modified_2": 8,
            "n_unmodified_2": 2,
            "wt_modified_1": 1.8,
            "wt_unmodified_1": 7.5,
            "wt_modified_2": 7.8,
            "wt_unmodified_2": 1.5,
            "log2_or": 2.5,
            "p_value": 0.001,
            "q_value": 0.01,
        }
        defaults.update(overrides)
        return defaults

    def testwrite_parquet(self, tmp_path):
        path = str(tmp_path / "out.parquet")
        rows = [self._make_row()]
        write_parquet(rows, path, _DMT_SCHEMA, _OUTPUT_COLS)
        table = pq.read_table(path)
        assert len(table) == 1
        assert table.column("log2_or")[0].as_py() == pytest.approx(2.5)

    def test_write_empty_parquet(self, tmp_path):
        path = str(tmp_path / "out.parquet")
        write_parquet([], path, _DMT_SCHEMA, _OUTPUT_COLS)
        table = pq.read_table(path)
        assert len(table) == 0

    def testwrite_tsv(self, tmp_path):
        path = str(tmp_path / "out.tsv")
        rows = [self._make_row()]
        write_tsv(rows, path, _TSV_HEADER, _OUTPUT_COLS, use_gzip=False)
        with open(path) as f:
            header = f.readline()
            data = f.readline()
        assert "gene_id" in header
        assert "G1" in data


# ---------- HDF5 helpers ----------


class TestHDF5Helpers:
    """Tests for HDF5 utility functions."""

    def testread_mod_codes(self, tmp_path):
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
            codes = read_mod_codes(h5)
        assert codes == {"a": 4, "m": 5}

    def test_validate_mod_codes_mismatch(self):
        with pytest.raises(ValueError):
            validate_mod_codes(
                [{"a": 4}, {"a": 5}],
                ["f1.h5", "f2.h5"],
            )


# ---------- CLI ----------


class TestCLI:
    """Tests for parse_args()."""

    def test_required_args(self):
        argv = [
            "-i",
            "a.h5",
            "-s",
            "sites.parquet",
            "-o",
            "out.parquet",
        ]
        import sys as _sys

        _sys.argv = ["mod_dmt"] + argv
        args = parse_args()
        assert args.h5 == ["a.h5"]
        assert args.sites == "sites.parquet"
        assert args.format == "parquet"

    def test_multi_h5(self):
        argv = [
            "-i",
            "a1.h5",
            "a2.h5",
            "-s",
            "sites.parquet",
            "-o",
            "out.parquet",
        ]
        import sys as _sys

        _sys.argv = ["mod_dmt"] + argv
        args = parse_args()
        assert args.h5 == ["a1.h5", "a2.h5"]

    def test_optional_args(self):
        argv = [
            "-i",
            "a.h5",
            "-s",
            "s.parquet",
            "-o",
            "out.tsv",
            "-f",
            "tsv",
            "-z",
            "-p",
            "0.5",
            "-x",
            "TX1",
            "-v",
        ]
        import sys as _sys

        _sys.argv = ["mod_dmt"] + argv
        args = parse_args()
        assert args.format == "tsv"
        assert args.gzip is True
        assert args.min_asp == 0.5
        assert args.transcripts == ["TX1"]
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

    def test_end_to_end_two_isoforms(self, tmp_path):
        """Two isoforms at the same locus with different modification."""
        # TX1: mostly unmodified
        matrix_1 = np.array([[CODE_CANONICAL]] * 5, dtype=np.uint8)
        weights_1 = np.ones(5, dtype=np.float32)

        # TX2: mostly modified
        matrix_2 = np.array([[4]] * 5, dtype=np.uint8)
        weights_2 = np.ones(5, dtype=np.float32)

        h5_path = str(tmp_path / "test.h5")
        sites_path = str(tmp_path / "sites.parquet")
        out_path = str(tmp_path / "out.parquet")

        self._make_h5_file(
            h5_path,
            {"TX1": (matrix_1, weights_1), "TX2": (matrix_2, weights_2)},
            {"a": 4},
        )
        self._make_sites_file(
            sites_path,
            [
                _make_site("TX1", 1, "a", 0, 5, 0.0),
                _make_site("TX2", 1, "a", 5, 0, 1.0),
            ],
        )

        args = argparse.Namespace(
            h5=[h5_path],
            sites=sites_path,
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
        assert r["gene_id"] == "G1"
        assert r["transcript_id_1"] == "TX1"
        assert r["transcript_id_2"] == "TX2"
        assert r["delta_mod_level"] == pytest.approx(1.0)
        assert r["log2_or"] > 0
        assert 0 <= r["q_value"] <= 1.0

    def test_end_to_end_no_genomic_coords_exits(self, tmp_path):
        """Site summary without genomic coordinates → exit with error."""
        h5_path = str(tmp_path / "test.h5")
        sites_path = str(tmp_path / "sites.parquet")
        out_path = str(tmp_path / "out.parquet")

        self._make_h5_file(
            h5_path,
            {
                "TX1": (
                    np.array([[4]], dtype=np.uint8),
                    np.array([1.0], dtype=np.float32),
                )
            },
            {"a": 4},
        )
        # Sites without gene_id/gpos
        table = pa.table(
            {
                "transcript_id": ["TX1"],
                "position": [10],
                "mod_type": ["a"],
                "n_modified": [5],
                "wt_modified": [5.0],
                "n_unmodified": [15],
                "wt_unmodified": [15.0],
                "mod_level": [0.25],
                "wt_mod_level": [0.25],
                "gene_id": [None],
                "chrom": [None],
                "strand": [None],
                "gpos": [None],
            }
        )
        pq.write_table(table, sites_path)

        args = argparse.Namespace(
            h5=[h5_path],
            sites=sites_path,
            output=out_path,
            format="parquet",
            gzip=False,
            min_asp=0.0,
            transcripts=None,
            verbose=False,
        )
        with pytest.raises(SystemExit):
            main(args)

    def test_end_to_end_multi_h5(self, tmp_path):
        """Multiple HDF5 files are pooled correctly."""
        # Two HDF5 files with reads for the same transcript
        matrix_a = np.array([[4]], dtype=np.uint8)
        weights_a = np.array([1.0], dtype=np.float32)
        matrix_b = np.array([[4], [CODE_CANONICAL]], dtype=np.uint8)
        weights_b = np.array([1.0, 1.0], dtype=np.float32)

        h5_a_path = str(tmp_path / "a.h5")
        h5_b_path = str(tmp_path / "b.h5")
        sites_path = str(tmp_path / "sites.parquet")
        out_path = str(tmp_path / "out.parquet")

        self._make_h5_file(
            h5_a_path,
            {
                "TX1": (matrix_a, weights_a),
                "TX2": (
                    np.array([[CODE_CANONICAL]] * 2, dtype=np.uint8),
                    np.array([1.0, 1.0], dtype=np.float32),
                ),
            },
            {"a": 4},
        )
        self._make_h5_file(
            h5_b_path,
            {"TX1": (matrix_b, weights_b)},
            {"a": 4},
        )
        self._make_sites_file(
            sites_path,
            [
                _make_site("TX1", 1, "a", 1, 0, 1.0),
                _make_site("TX2", 1, "a", 0, 2, 0.0),
            ],
        )

        args = argparse.Namespace(
            h5=[h5_a_path, h5_b_path],
            sites=sites_path,
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

    def test_end_to_end_empty_result(self, tmp_path):
        """No matching transcripts in HDF5 → output empty schema-only file."""
        h5_path = str(tmp_path / "test.h5")
        sites_path = str(tmp_path / "sites.parquet")
        out_path = str(tmp_path / "out.parquet")

        # HDF5 with transcripts that don't match the site summary
        self._make_h5_file(
            h5_path,
            {
                "TX_OTHER": (
                    np.array([[4]], dtype=np.uint8),
                    np.array([1.0], dtype=np.float32),
                )
            },
            {"a": 4},
        )
        self._make_sites_file(
            sites_path,
            [
                _make_site("TX1", 1, "a", 5, 15, 0.25),
                _make_site("TX2", 1, "a", 8, 12, 0.4),
            ],
        )

        args = argparse.Namespace(
            h5=[h5_path],
            sites=sites_path,
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
        matrix_1 = np.array([[4]] * 5, dtype=np.uint8)
        weights_1 = np.ones(5, dtype=np.float32)
        matrix_2 = np.array([[CODE_CANONICAL]] * 5, dtype=np.uint8)
        weights_2 = np.ones(5, dtype=np.float32)

        h5_path = str(tmp_path / "test.h5")
        sites_path = str(tmp_path / "sites.parquet")
        out_path = str(tmp_path / "out.tsv")

        self._make_h5_file(
            h5_path,
            {"TX1": (matrix_1, weights_1), "TX2": (matrix_2, weights_2)},
            {"a": 4},
        )
        self._make_sites_file(
            sites_path,
            [
                _make_site("TX1", 1, "a", 5, 0, 1.0),
                _make_site("TX2", 1, "a", 0, 5, 0.0),
            ],
        )

        args = argparse.Namespace(
            h5=[h5_path],
            sites=sites_path,
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
        assert "gene_id" in lines[0]

    def test_transcripts_filter(self, tmp_path):
        """--transcripts filter limits which transcripts are loaded."""
        matrix_1 = np.array([[4]] * 5, dtype=np.uint8)
        weights_1 = np.ones(5, dtype=np.float32)
        matrix_2 = np.array([[CODE_CANONICAL]] * 5, dtype=np.uint8)
        weights_2 = np.ones(5, dtype=np.float32)

        h5_path = str(tmp_path / "test.h5")
        sites_path = str(tmp_path / "sites.parquet")
        out_path = str(tmp_path / "out.parquet")

        self._make_h5_file(
            h5_path,
            {
                "TX1": (matrix_1, weights_1),
                "TX2": (matrix_2, weights_2),
                "TX_OTHER": (
                    np.array([[4]], dtype=np.uint8),
                    np.array([1.0], dtype=np.float32),
                ),
            },
            {"a": 4},
        )
        self._make_sites_file(
            sites_path,
            [
                _make_site("TX1", 1, "a", 5, 0, 1.0),
                _make_site("TX2", 1, "a", 0, 5, 0.0),
            ],
        )

        # Only allow TX1 and TX2
        args = argparse.Namespace(
            h5=[h5_path],
            sites=sites_path,
            output=out_path,
            format="parquet",
            gzip=False,
            min_asp=0.0,
            transcripts=["TX1", "TX2"],
            verbose=False,
        )
        main(args)

        table = pq.read_table(out_path)
        assert len(table) == 1
