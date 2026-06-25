"""Tests for mod_sites — per-position modification summaries."""

import os
import sys
import tempfile

import numpy as np
import pyarrow.parquet as pq
import pytest

try:
    from isolens.mod_scan import (
        CODE_CANONICAL,
        CODE_DELETION,
        CODE_FAIL,
        CODE_MISMATCH,
    )
    from isolens.mod_sites import (
        _write_parquet,
        _write_tsv,
        compute_transcript_stats,
        read_predefined_sites,
    )
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from isolens.mod_scan import (  # type: ignore[no-redef]
        CODE_CANONICAL,
        CODE_DELETION,
        CODE_FAIL,
        CODE_MISMATCH,
    )
    from isolens.mod_sites import (  # type: ignore[no-redef]
        _write_parquet,
        _write_tsv,
        compute_transcript_stats,
        read_predefined_sites,
    )


# ---------- compute_transcript_stats ----------


class TestComputeTranscriptStats:
    """Tests for compute_transcript_stats()."""

    def test_single_mod_single_read(self):
        """One read, one modification type at position 1."""
        matrix = np.array([[4, CODE_CANONICAL, CODE_CANONICAL]], dtype=np.uint8)
        weights = np.array([0.5], dtype=np.float32)
        mod_codes = [("a", 4)]

        rows = compute_transcript_stats(matrix, weights, mod_codes)

        assert len(rows) == 1
        r = rows[0]
        assert r["position"] == 1
        assert r["mod_type"] == "a"
        assert r["n_modified"] == 1
        assert r["wt_modified"] == pytest.approx(0.5)
        assert r["n_canonical"] == 0
        assert r["n_othermod"] == 0
        assert r["n_unmodified"] == 0
        assert r["n_mismatch"] == 0
        assert r["n_deletion"] == 0
        assert r["n_failed"] == 0
        assert r["mod_level"] == pytest.approx(1.0)
        assert r["wt_mod_level"] == pytest.approx(1.0)

    def test_canonical_position(self):
        """Position that is canonical in all reads."""
        matrix = np.array(
            [[CODE_CANONICAL, CODE_CANONICAL], [CODE_CANONICAL, 4]],
            dtype=np.uint8,
        )
        weights = np.array([1.0, 1.0], dtype=np.float32)
        mod_codes = [("a", 4)]

        rows = compute_transcript_stats(matrix, weights, mod_codes)

        # Only position 2 has n_modified > 0
        assert len(rows) == 1
        r = rows[0]
        assert r["position"] == 2
        assert r["n_modified"] == 1
        assert r["n_canonical"] == 1
        assert r["mod_level"] == pytest.approx(0.5)

    def test_mismatch_and_deletion_tracked(self):
        """Mismatch and deletion counts are tracked per position."""
        matrix = np.array(
            [
                [CODE_MISMATCH, CODE_DELETION, CODE_FAIL, 4],
                [CODE_CANONICAL, CODE_CANONICAL, 4, CODE_CANONICAL],
            ],
            dtype=np.uint8,
        )
        weights = np.array([1.0, 1.0], dtype=np.float32)
        mod_codes = [("a", 4)]

        rows = compute_transcript_stats(matrix, weights, mod_codes)

        # Position 3 (1-based): has FAIL in row 0, 4 (mod 'a') in row 1
        r = [r for r in rows if r["position"] == 3][0]
        assert r["n_mismatch"] == 0
        assert r["n_deletion"] == 0
        assert r["n_failed"] == 1
        assert r["n_modified"] == 1

    def test_multiple_mod_types(self):
        """Two modification types at different positions."""
        matrix = np.array([[4, 5, CODE_CANONICAL]], dtype=np.uint8)
        weights = np.array([1.0], dtype=np.float32)
        mod_codes = [("a", 4), ("m", 5)]

        rows = compute_transcript_stats(matrix, weights, mod_codes)

        # Position 1 → mod 'a' wins, Position 2 → mod 'm' wins
        assert len(rows) == 2
        positions = {r["position"] for r in rows}
        assert positions == {1, 2}

    def test_othermod_counted(self):
        """othermod = any mod code (≥4) that is not the focal type."""
        matrix = np.array([[4, 5, CODE_CANONICAL]], dtype=np.uint8)
        weights = np.array([1.0], dtype=np.float32)
        mod_codes = [("a", 4), ("m", 5)]

        # Use predefined positions to force emission for mod 'a' at pos 2
        rows = compute_transcript_stats(
            matrix,
            weights,
            mod_codes,
            predefined_positions={1, 2},
        )

        # For mod 'a' at position 2: the entry is 'm' (5) → othermod = 1
        r_a_pos2 = [r for r in rows if r["mod_type"] == "a" and r["position"] == 2][0]
        assert r_a_pos2["n_modified"] == 0
        assert r_a_pos2["n_othermod"] == 1

    def test_predefined_positions(self):
        """When predefined_positions is given, only those are emitted."""
        matrix = np.array([[4, 4, CODE_CANONICAL, 4]], dtype=np.uint8)
        weights = np.array([1.0], dtype=np.float32)
        mod_codes = [("a", 4)]

        rows = compute_transcript_stats(
            matrix,
            weights,
            mod_codes,
            predefined_positions={1, 3},  # only positions 1 and 3
        )

        positions = {r["position"] for r in rows}
        assert positions == {1, 3}

    def test_predefined_out_of_bounds(self):
        """Positions beyond transcript length are silently ignored."""
        matrix = np.array([[4, CODE_CANONICAL]], dtype=np.uint8)
        weights = np.array([1.0], dtype=np.float32)
        mod_codes = [("a", 4)]

        rows = compute_transcript_stats(
            matrix,
            weights,
            mod_codes,
            predefined_positions={1, 100},  # 100 > tx_length=2
        )

        positions = {r["position"] for r in rows}
        assert positions == {1}

    def test_empty_matrix(self):
        """Zero reads produces empty output."""
        matrix = np.empty((0, 10), dtype=np.uint8)
        weights = np.empty((0,), dtype=np.float32)
        mod_codes = [("a", 4)]

        rows = compute_transcript_stats(matrix, weights, mod_codes)
        assert rows == []

    def test_no_mods_found(self):
        """When no positions have any modification calls, output is empty."""
        matrix = np.array(
            [[CODE_CANONICAL, CODE_CANONICAL], [CODE_CANONICAL, CODE_CANONICAL]],
            dtype=np.uint8,
        )
        weights = np.array([1.0, 1.0], dtype=np.float32)
        mod_codes = [("a", 4)]

        rows = compute_transcript_stats(matrix, weights, mod_codes)
        assert rows == []

    def test_mod_level_calculation(self):
        """Modification level = n_modified / (n_modified + n_unmodified)."""
        matrix = np.array(
            [[4, 4, CODE_CANONICAL, CODE_CANONICAL]],
            dtype=np.uint8,
        )
        weights = np.array([1.0], dtype=np.float32)
        mod_codes = [("a", 4)]

        rows = compute_transcript_stats(matrix, weights, mod_codes)
        r1 = [r for r in rows if r["position"] == 1][0]
        r2 = [r for r in rows if r["position"] == 2][0]

        assert r1["mod_level"] == pytest.approx(1.0)  # 1/1
        assert r2["mod_level"] == pytest.approx(1.0)  # 1/1


# ---------- read_predefined_sites ----------


class TestReadPredefinedSites:
    """Tests for read_predefined_sites()."""

    def test_valid_tsv(self, tmp_path):
        path = tmp_path / "sites.tsv"
        path.write_text(
            "tx_name\tposn\textra\nTX1\t42\tignored\nTX1\t100\tignored\nTX2\t5\tignored\n"
        )

        sites = read_predefined_sites(str(path))
        assert sites == {"TX1": {42, 100}, "TX2": {5}}

    def test_missing_columns(self, tmp_path):
        path = tmp_path / "bad.tsv"
        path.write_text("tx_name\textra\nTX1\t42\n")

        with pytest.raises(ValueError, match="must have 'tx_name' and 'posn'"):
            read_predefined_sites(str(path))

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.tsv"
        path.write_text("tx_name\tposn\n")

        sites = read_predefined_sites(str(path))
        assert sites == {}

    def test_non_integer_posn(self, tmp_path):
        path = tmp_path / "bad_pos.tsv"
        path.write_text("tx_name\tposn\nTX1\tnot_a_number\n")

        sites = read_predefined_sites(str(path))
        assert sites == {}


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
            expected_cols = {
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
            }
            assert set(table.column_names) == expected_cols
        finally:
            os.unlink(tmp_path)

    def test_non_empty_rows(self):
        rows = [
            {
                "transcript_id": "TX1",
                "position": 42,
                "mod_type": "a",
                "n_modified": 10,
                "wt_modified": 8.5,
                "n_unmodified": 90,
                "wt_unmodified": 85.0,
                "n_canonical": 85,
                "wt_canonical": 80.0,
                "n_othermod": 5,
                "wt_othermod": 5.0,
                "n_mismatch": 2,
                "wt_mismatch": 1.5,
                "n_deletion": 1,
                "wt_deletion": 0.8,
                "n_failed": 3,
                "wt_failed": 2.0,
                "mod_level": 0.1,
                "wt_mod_level": 0.091,
            }
        ]
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tf:
            tmp_path = tf.name
        try:
            _write_parquet(rows, tmp_path)
            table = pq.read_table(tmp_path)
            assert len(table) == 1
            assert table.column("position")[0].as_py() == 42
        finally:
            os.unlink(tmp_path)


# ---------- _write_tsv ----------


class TestWriteTsv:
    """Tests for _write_tsv()."""

    def test_empty_rows(self, tmp_path):
        path = tmp_path / "out.tsv"
        _write_tsv([], str(path), use_gzip=False)
        content = path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 1  # header only
        assert "transcript_id" in lines[0]

    def test_non_empty_rows(self, tmp_path):
        rows = [
            {
                "transcript_id": "TX1",
                "position": 42,
                "mod_type": "a",
                "n_modified": 1,
                "wt_modified": 0.5,
                "n_unmodified": 9,
                "wt_unmodified": 4.5,
                "n_canonical": 8,
                "wt_canonical": 4.0,
                "n_othermod": 1,
                "wt_othermod": 0.5,
                "n_mismatch": 0,
                "wt_mismatch": 0.0,
                "n_deletion": 0,
                "wt_deletion": 0.0,
                "n_failed": 0,
                "wt_failed": 0.0,
                "mod_level": 0.1,
                "wt_mod_level": 0.1,
            }
        ]
        path = tmp_path / "out.tsv"
        _write_tsv(rows, str(path), use_gzip=False)
        content = path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 2  # header + 1 data row
        assert "TX1" in lines[1]

    def test_gzip_output(self, tmp_path):
        import gzip

        rows = [
            {
                "transcript_id": "TX1",
                "position": 1,
                "mod_type": "a",
                "n_modified": 0,
                "wt_modified": 0.0,
                "n_unmodified": 0,
                "wt_unmodified": 0.0,
                "n_canonical": 0,
                "wt_canonical": 0.0,
                "n_othermod": 0,
                "wt_othermod": 0.0,
                "n_mismatch": 0,
                "wt_mismatch": 0.0,
                "n_deletion": 0,
                "wt_deletion": 0.0,
                "n_failed": 0,
                "wt_failed": 0.0,
                "mod_level": 0.0,
                "wt_mod_level": 0.0,
            }
        ]
        path = tmp_path / "out.tsv.gz"
        _write_tsv(rows, str(path), use_gzip=True)

        with gzip.open(path, "rt", encoding="utf-8") as f:
            content = f.read()
        assert "transcript_id" in content
