"""Tests for mod_sites — per-position modification summaries."""

import argparse
import os
import sys
import tempfile

import h5py
import numpy as np
import pyarrow.parquet as pq
import pytest

try:
    from isolens._hdf5_helpers import validate_mod_codes, validate_tx_lengths
    from isolens._io import write_parquet, write_tsv
    from isolens.mod_scan import (
        CODE_CANONICAL,
        CODE_DELETION,
        CODE_FAIL,
        CODE_MISMATCH,
    )
    from isolens.mod_sites import (
        _SITES_SCHEMA,
        _TSV_COLS,
        _TSV_HEADER,
        _make_zero_rows,
        compute_transcript_stats,
        main,
        read_predefined_sites,
    )
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from isolens._hdf5_helpers import (  # type: ignore[no-redef]
        validate_mod_codes,
        validate_tx_lengths,
    )
    from isolens._io import (  # type: ignore[no-redef]
        write_parquet,
        write_tsv,
    )
    from isolens.mod_scan import (  # type: ignore[no-redef]
        CODE_CANONICAL,
        CODE_DELETION,
        CODE_FAIL,
        CODE_MISMATCH,
    )
    from isolens.mod_sites import (  # type: ignore[no-redef]
        _SITES_SCHEMA,
        _TSV_COLS,
        _TSV_HEADER,
        _make_zero_rows,
        compute_transcript_stats,
        main,
        read_predefined_sites,
    )


# ---------- compute_transcript_stats ----------


class TestComputeTranscriptStats:
    """Tests for compute_transcript_stats()."""

    @staticmethod
    def _to_rows(col_arrays):
        """Convert column arrays to list of dicts for readable assertions."""
        if col_arrays is None:
            return []
        n = len(col_arrays["position"])
        cols = list(col_arrays.keys())
        rows = []
        for i in range(n):
            row = {}
            for c in cols:
                v = col_arrays[c][i]
                if isinstance(v, (np.integer,)):
                    row[c] = int(v)
                elif isinstance(v, (np.floating,)):
                    row[c] = float(v)
                elif isinstance(v, np.bool_):
                    row[c] = bool(v)
                else:
                    row[c] = v
            rows.append(row)
        return rows

    def test_single_mod_single_read(self):
        """One read, one modification type at position 1."""
        matrix = np.array([[4, CODE_CANONICAL, CODE_CANONICAL]], dtype=np.uint8)
        weights = np.array([0.5], dtype=np.float32)
        mod_codes = [("a", 4)]

        result = compute_transcript_stats(matrix, weights, mod_codes)
        rows = self._to_rows(result)

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

        result = compute_transcript_stats(matrix, weights, mod_codes)
        rows = self._to_rows(result)

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

        result = compute_transcript_stats(matrix, weights, mod_codes)
        rows = self._to_rows(result)

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

        result = compute_transcript_stats(matrix, weights, mod_codes)
        rows = self._to_rows(result)

        # Position 1 → mod 'a' wins, Position 2 → mod 'm' wins
        assert len(rows) == 2
        positions = {r["position"] for r in rows}
        assert positions == {1, 2}

    def test_othermod_counted(self):
        """othermod = any mod code (≥4) that is not the focal type."""
        matrix = np.array([[4, 5, CODE_CANONICAL]], dtype=np.uint8)
        weights = np.array([1.0], dtype=np.float32)
        mod_codes = [("a", 4), ("m", 5)]

        # Use predefined mods to force emission for mod 'a' at pos 2
        result = compute_transcript_stats(
            matrix,
            weights,
            mod_codes,
            predefined_mods={1: None, 2: None},
        )
        rows = self._to_rows(result)

        # For mod 'a' at position 2: the entry is 'm' (5) → othermod = 1
        r_a_pos2 = [r for r in rows if r["mod_type"] == "a" and r["position"] == 2][0]
        assert r_a_pos2["n_modified"] == 0
        assert r_a_pos2["n_othermod"] == 1

    def test_predefined_mods(self):
        """When predefined_mods is given, only those are emitted."""
        matrix = np.array([[4, 4, CODE_CANONICAL, 4]], dtype=np.uint8)
        weights = np.array([1.0], dtype=np.float32)
        mod_codes = [("a", 4)]

        result = compute_transcript_stats(
            matrix,
            weights,
            mod_codes,
            predefined_mods={1: None, 3: None},  # only positions 1 and 3
        )
        rows = self._to_rows(result)

        positions = {r["position"] for r in rows}
        assert positions == {1, 3}

    def test_predefined_out_of_bounds(self):
        """Positions beyond transcript length are silently ignored."""
        matrix = np.array([[4, CODE_CANONICAL]], dtype=np.uint8)
        weights = np.array([1.0], dtype=np.float32)
        mod_codes = [("a", 4)]

        result = compute_transcript_stats(
            matrix,
            weights,
            mod_codes,
            predefined_mods={1: None, 100: None},  # 100 > tx_length=2
        )
        rows = self._to_rows(result)

        positions = {r["position"] for r in rows}
        assert positions == {1}

    def test_predefined_mods_all_types(self):
        """None value emits all modification types at that position."""
        matrix = np.array([[4, 5, CODE_CANONICAL]], dtype=np.uint8)
        weights = np.array([1.0], dtype=np.float32)
        mod_codes = [("a", 4), ("m", 5)]

        result = compute_transcript_stats(
            matrix,
            weights,
            mod_codes,
            predefined_mods={1: None},
        )
        rows = self._to_rows(result)
        # Both "a" and "m" at position 1
        types_at_pos1 = {r["mod_type"] for r in rows if r["position"] == 1}
        assert types_at_pos1 == {"a", "m"}
        assert len(rows) == 2

    def test_predefined_mods_single_type(self):
        """Specific mod_type set emits only that type."""
        matrix = np.array([[4, 5, CODE_CANONICAL]], dtype=np.uint8)
        weights = np.array([1.0], dtype=np.float32)
        mod_codes = [("a", 4), ("m", 5)]

        result = compute_transcript_stats(
            matrix,
            weights,
            mod_codes,
            predefined_mods={2: {"m"}},
        )
        rows = self._to_rows(result)
        assert len(rows) == 1
        assert rows[0]["position"] == 2
        assert rows[0]["mod_type"] == "m"

    def test_predefined_mods_combined(self):
        """Mix of all-types and specific-type positions."""
        matrix = np.array([[4, 5, CODE_CANONICAL]], dtype=np.uint8)
        weights = np.array([1.0], dtype=np.float32)
        mod_codes = [("a", 4), ("m", 5)]

        result = compute_transcript_stats(
            matrix,
            weights,
            mod_codes,
            predefined_mods={1: None, 2: {"a"}},
        )
        rows = self._to_rows(result)
        # Position 1: both types; Position 2: only "a"
        assert len(rows) == 3

    def test_predefined_mods_unknown_type(self):
        """Mod type not in H5 mod_codes is silently skipped."""
        matrix = np.array([[4, CODE_CANONICAL]], dtype=np.uint8)
        weights = np.array([1.0], dtype=np.float32)
        mod_codes = [("a", 4)]

        result = compute_transcript_stats(
            matrix,
            weights,
            mod_codes,
            predefined_mods={1: {"unknown"}},
        )
        # Unknown mod_type not in code_map → no rows emitted
        assert result is None

    def test_empty_matrix(self):
        """Zero reads produces empty output."""
        matrix = np.empty((0, 10), dtype=np.uint8)
        weights = np.empty((0,), dtype=np.float32)
        mod_codes = [("a", 4)]

        result = compute_transcript_stats(matrix, weights, mod_codes)
        assert result is None

    def test_no_mods_found(self):
        """When no positions have any modification calls, output is empty."""
        matrix = np.array(
            [[CODE_CANONICAL, CODE_CANONICAL], [CODE_CANONICAL, CODE_CANONICAL]],
            dtype=np.uint8,
        )
        weights = np.array([1.0, 1.0], dtype=np.float32)
        mod_codes = [("a", 4)]

        result = compute_transcript_stats(matrix, weights, mod_codes)
        assert result is None

    def test_mod_level_calculation(self):
        """Modification level = n_modified / (n_modified + n_unmodified)."""
        matrix = np.array(
            [[4, 4, CODE_CANONICAL, CODE_CANONICAL]],
            dtype=np.uint8,
        )
        weights = np.array([1.0], dtype=np.float32)
        mod_codes = [("a", 4)]

        result = compute_transcript_stats(matrix, weights, mod_codes)
        rows = self._to_rows(result)
        r1 = [r for r in rows if r["position"] == 1][0]
        r2 = [r for r in rows if r["position"] == 2][0]

        assert r1["mod_level"] == pytest.approx(1.0)  # 1/1
        assert r2["mod_level"] == pytest.approx(1.0)  # 1/1


# ---------- read_predefined_sites ----------


class TestReadPredefinedSites:
    """Tests for read_predefined_sites() — headerless TSV format."""

    def test_two_col_no_mod_type(self, tmp_path):
        """Two columns, no mod_type → None (all types)."""
        path = tmp_path / "sites.tsv"
        path.write_text("TX1\t42\nTX2\t5\n")
        sites = read_predefined_sites(str(path))
        assert sites == {"TX1": {42: None}, "TX2": {5: None}}

    def test_three_col_with_mod_type(self, tmp_path):
        """Three columns with mod_type → specific set."""
        path = tmp_path / "sites.tsv"
        path.write_text("TX1\t42\tm\nTX1\t100\ta\n")
        sites = read_predefined_sites(str(path))
        assert sites == {"TX1": {42: {"m"}, 100: {"a"}}}

    def test_empty_mod_type_col(self, tmp_path):
        """Empty third column → None (all types)."""
        path = tmp_path / "sites.tsv"
        path.write_text("TX1\t42\t\nTX1\t43\tm\n")
        sites = read_predefined_sites(str(path))
        assert sites == {"TX1": {42: None, 43: {"m"}}}

    def test_extra_columns_ignored(self, tmp_path):
        """Columns beyond the 3rd are ignored."""
        path = tmp_path / "sites.tsv"
        path.write_text("TX1\t42\tm\textra1\textra2\n")
        sites = read_predefined_sites(str(path))
        assert sites == {"TX1": {42: {"m"}}}

    def test_same_pos_specific_then_all(self, tmp_path):
        """Specific entry + all-types entry at same pos → all wins."""
        path = tmp_path / "sites.tsv"
        path.write_text("TX1\t42\tm\nTX1\t42\n")
        sites = read_predefined_sites(str(path))
        assert sites == {"TX1": {42: None}}

    def test_multiple_specific_same_pos(self, tmp_path):
        """Multiple specific mod_types at same position accumulate."""
        path = tmp_path / "sites.tsv"
        path.write_text("TX1\t42\ta\nTX1\t42\tm\n")
        sites = read_predefined_sites(str(path))
        assert sites == {"TX1": {42: {"a", "m"}}}

    def test_all_then_specific_same_pos(self, tmp_path):
        """All-types entry first then specific → all still wins."""
        path = tmp_path / "sites.tsv"
        path.write_text("TX1\t42\nTX1\t42\tm\n")
        sites = read_predefined_sites(str(path))
        assert sites == {"TX1": {42: None}}

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.tsv"
        path.write_text("")
        sites = read_predefined_sites(str(path))
        assert sites == {}

    def test_non_int_position_skipped(self, tmp_path):
        path = tmp_path / "bad.tsv"
        path.write_text("TX1\tnot_a_number\tm\nTX1\t42\ta\n")
        sites = read_predefined_sites(str(path))
        assert sites == {"TX1": {42: {"a"}}}

    def test_too_few_columns_skipped(self, tmp_path):
        path = tmp_path / "few.tsv"
        path.write_text("TX1\t42\ta\nonly_one_col\nTX2\t5\n")
        sites = read_predefined_sites(str(path))
        assert sites == {"TX1": {42: {"a"}}, "TX2": {5: None}}


# ---------- _make_zero_rows ----------


class TestMakeZeroRows:
    """Tests for _make_zero_rows()."""

    @staticmethod
    def _to_rows(col_arrays):
        """Convert column arrays to list of dicts for readable assertions."""
        if col_arrays is None:
            return []
        n = len(col_arrays["position"])
        cols = list(col_arrays.keys())
        rows = []
        for i in range(n):
            row = {}
            for c in cols:
                v = col_arrays[c][i]
                if isinstance(v, (np.integer,)):
                    row[c] = int(v)
                elif isinstance(v, (np.floating,)):
                    row[c] = float(v)
                else:
                    row[c] = v
            rows.append(row)
        return rows

    def test_basic_single_type(self):
        """Single position with specific mod_type → one zero row."""
        result = _make_zero_rows(
            "TX1",
            predefined_mods={1: {"m"}},
            mod_codes=[("a", 4), ("m", 5)],
        )
        rows = self._to_rows(result)
        assert len(rows) == 1
        r = rows[0]
        assert r["transcript_id"] == "TX1"
        assert r["position"] == 1
        assert r["mod_type"] == "m"
        assert r["n_modified"] == 0
        assert r["wt_modified"] == 0.0
        assert r["n_unmodified"] == 0
        assert r["mod_level"] == 0.0
        assert r["wt_mod_level"] == 0.0

    def test_all_types(self):
        """None → all modification types get zero rows."""
        result = _make_zero_rows(
            "TX1",
            predefined_mods={1: None},
            mod_codes=[("a", 4), ("m", 5)],
        )
        rows = self._to_rows(result)
        assert len(rows) == 2
        mod_types = {r["mod_type"] for r in rows}
        assert mod_types == {"a", "m"}

    def test_multiple_positions(self):
        """Multiple positions with different mod_types."""
        result = _make_zero_rows(
            "TX2",
            predefined_mods={5: {"a"}, 10: {"m"}},
            mod_codes=[("a", 4), ("m", 5)],
        )
        rows = self._to_rows(result)
        assert len(rows) == 2
        positions = {r["position"] for r in rows}
        assert positions == {5, 10}

    def test_unknown_mod_type_skipped(self):
        """Mod type not in mod_codes is not emitted."""
        result = _make_zero_rows(
            "TX1",
            predefined_mods={1: {"unknown"}},
            mod_codes=[("a", 4)],
        )
        assert result is None

    def test_empty_mods(self):
        """Empty predefined_mods returns None."""
        result = _make_zero_rows(
            "TX1",
            predefined_mods={},
            mod_codes=[("a", 4)],
        )
        assert result is None


# ---------- write_parquet ----------


class TestWriteParquet:
    """Tests for write_parquet()."""

    def test_empty_rows(self):
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tf:
            tmp_path = tf.name
        try:
            write_parquet([], tmp_path, _SITES_SCHEMA, _TSV_COLS)
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
                "gene_id",
                "chrom",
                "strand",
                "gpos",
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
                "gene_id": None,
                "chrom": None,
                "strand": None,
                "gpos": None,
            }
        ]
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tf:
            tmp_path = tf.name
        try:
            write_parquet(rows, tmp_path, _SITES_SCHEMA, _TSV_COLS)
            table = pq.read_table(tmp_path)
            assert len(table) == 1
            assert table.column("position")[0].as_py() == 42
        finally:
            os.unlink(tmp_path)


# ---------- write_tsv ----------


class TestWriteTsv:
    """Tests for write_tsv()."""

    def test_empty_rows(self, tmp_path):
        path = tmp_path / "out.tsv"
        write_tsv([], str(path), _TSV_HEADER, _TSV_COLS, use_gzip=False)
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
                "gene_id": None,
                "chrom": None,
                "strand": None,
                "gpos": None,
            }
        ]
        path = tmp_path / "out.tsv"
        write_tsv(rows, str(path), _TSV_HEADER, _TSV_COLS, use_gzip=False)
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
                "gene_id": None,
                "chrom": None,
                "strand": None,
                "gpos": None,
            }
        ]
        path = tmp_path / "out.tsv.gz"
        write_tsv(rows, str(path), _TSV_HEADER, _TSV_COLS, use_gzip=True)

        with gzip.open(path, "rt", encoding="utf-8") as f:
            content = f.read()
        assert "transcript_id" in content


# ---------- GTF mapping ----------


def _make_gtf(tmp_path, lines: list[str]) -> str:
    """Write a minimal GTF file and return its path."""
    path = tmp_path / "test.gtf"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


class TestGtfMapping:
    """Tests for transcript-to-genomic coordinate mapping via gppy."""

    def test_single_exon_plus_strand(self, tmp_path):
        """Single-exon transcript on the + strand."""
        from gppy.gtf import parse_gtf

        gtf_path = _make_gtf(
            tmp_path,
            [
                "chr1\tgtf\texon\t101\t200\t.\t+\t.\t"
                'gene_id "G1"; transcript_id "TX1";',
            ],
        )
        gtf = parse_gtf(gtf_path)
        tx = gtf["TX1"]

        assert tx.gene.chrom == "chr1"
        assert tx.gene.strand == "+"
        # tpos 1 → gpos 101 (first base of exon)
        assert tx.tpos_to_gpos(1) == 101
        # tpos 100 → gpos 200 (last base of exon)
        assert tx.tpos_to_gpos(100) == 200
        # tpos 50 → gpos 150 (midpoint)
        assert tx.tpos_to_gpos(50) == 150

    def test_single_exon_minus_strand(self, tmp_path):
        """Single-exon transcript on the - strand — coordinates are reversed."""
        from gppy.gtf import parse_gtf

        gtf_path = _make_gtf(
            tmp_path,
            [
                "chr1\tgtf\texon\t101\t200\t.\t-\t.\t"
                'gene_id "G1"; transcript_id "TX1";',
            ],
        )
        gtf = parse_gtf(gtf_path)
        tx = gtf["TX1"]

        assert tx.gene.strand == "-"
        assert len(tx) == 100
        # tpos 1 (5' end) → last base of the exon on genome (gpos 200)
        assert tx.tpos_to_gpos(1) == 200
        # tpos 100 (3' end) → first base of the exon on genome (gpos 101)
        assert tx.tpos_to_gpos(100) == 101

    def test_multi_exon_plus_strand(self, tmp_path):
        """Two-exon transcript on the + strand."""
        from gppy.gtf import parse_gtf

        gtf_path = _make_gtf(
            tmp_path,
            [
                # exon 1: 101-150 (len 50), exon 2: 201-250 (len 50)
                "chr1\tgtf\texon\t101\t150\t.\t+\t.\t"
                'gene_id "G1"; transcript_id "TX1";',
                "chr1\tgtf\texon\t201\t250\t.\t+\t.\t"
                'gene_id "G1"; transcript_id "TX1";',
            ],
        )
        gtf = parse_gtf(gtf_path)
        tx = gtf["TX1"]

        assert len(tx) == 100
        # tpos 1 → first base of exon 1
        assert tx.tpos_to_gpos(1) == 101
        # tpos 50 → last base of exon 1
        assert tx.tpos_to_gpos(50) == 150
        # tpos 51 → first base of exon 2
        assert tx.tpos_to_gpos(51) == 201
        # tpos 100 → last base of exon 2
        assert tx.tpos_to_gpos(100) == 250

    def test_multi_exon_minus_strand(self, tmp_path):
        """Two-exon transcript on the - strand."""
        from gppy.gtf import parse_gtf

        gtf_path = _make_gtf(
            tmp_path,
            [
                # exon 1 (genomic): 101-150 (len 50), exon 2 (genomic): 201-250 (len 50)
                # On minus strand: tpos 1 = last base of exon 2 = 250
                #                  tpos 100 = first base of exon 1 = 101
                "chr1\tgtf\texon\t101\t150\t.\t-\t.\t"
                'gene_id "G1"; transcript_id "TX1";',
                "chr1\tgtf\texon\t201\t250\t.\t-\t.\t"
                'gene_id "G1"; transcript_id "TX1";',
            ],
        )
        gtf = parse_gtf(gtf_path)
        tx = gtf["TX1"]

        assert len(tx) == 100
        # tpos 1 (5') → last base of exon 2 (genomic 250)
        assert tx.tpos_to_gpos(1) == 250
        # tpos 50 → first base of exon 2 (genomic 201)
        assert tx.tpos_to_gpos(50) == 201
        # tpos 51 → last base of exon 1 (genomic 150)
        assert tx.tpos_to_gpos(51) == 150
        # tpos 100 (3') → first base of exon 1 (genomic 101)
        assert tx.tpos_to_gpos(100) == 101

    def test_missing_transcript(self, tmp_path):
        """Transcript not in GTF returns no match."""
        from gppy.gtf import parse_gtf

        gtf_path = _make_gtf(
            tmp_path,
            [
                "chr1\tgtf\texon\t101\t200\t.\t+\t.\t"
                'gene_id "G1"; transcript_id "TX1";',
            ],
        )
        gtf = parse_gtf(gtf_path)
        assert gtf.get("TX2") is None


class TestWriteWithGtfColumns:
    """Tests for Parquet/TSV output including chrom, strand, gpos columns."""

    def _make_row(self, **overrides):
        """Create a minimal row dict with all required columns."""
        row = {
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
            "gene_id": "G1",
            "chrom": "chr1",
            "strand": "+",
            "gpos": 142,
        }
        row.update(overrides)
        return row

    def test_parquet_with_gtf_columns(self):
        """Parquet output includes gene_id, chrom, strand, gpos."""
        rows = [self._make_row()]
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tf:
            tmp_path = tf.name
        try:
            write_parquet(rows, tmp_path, _SITES_SCHEMA, _TSV_COLS)
            table = pq.read_table(tmp_path)
            assert "gene_id" in table.column_names
            assert "chrom" in table.column_names
            assert "strand" in table.column_names
            assert "gpos" in table.column_names
            assert table.column("gene_id")[0].as_py() == "G1"
            assert table.column("chrom")[0].as_py() == "chr1"
            assert table.column("strand")[0].as_py() == "+"
            assert table.column("gpos")[0].as_py() == 142
        finally:
            os.unlink(tmp_path)

    def test_parquet_with_null_gtf_columns(self):
        """Parquet handles None values in GTF columns."""
        rows = [self._make_row(gene_id=None, chrom=None, strand=None, gpos=None)]
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tf:
            tmp_path = tf.name
        try:
            write_parquet(rows, tmp_path, _SITES_SCHEMA, _TSV_COLS)
            table = pq.read_table(tmp_path)
            assert table.column("gene_id")[0].as_py() is None
            assert table.column("chrom")[0].as_py() is None
            assert table.column("strand")[0].as_py() is None
            assert table.column("gpos")[0].as_py() is None
        finally:
            os.unlink(tmp_path)

    def test_tsv_with_gtf_columns(self, tmp_path):
        """TSV output includes gene_id, chrom, strand, gpos."""
        rows = [self._make_row()]
        path = tmp_path / "out.tsv"
        write_tsv(rows, str(path), _TSV_HEADER, _TSV_COLS, use_gzip=False)
        content = path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 2  # header + 1 data
        header = lines[0]
        assert "gene_id" in header
        assert "chrom" in header
        assert "strand" in header
        assert "gpos" in header
        # Verify the GTF columns are at the end
        cols = header.split("\t")
        assert cols[3:7] == ["gene_id", "chrom", "strand", "gpos"]
        data = lines[1].split("\t")
        assert data[3:7] == ["G1", "chr1", "+", "142"]

    def test_tsv_with_null_gtf_columns(self, tmp_path):
        """TSV writes 'NA' for None GTF values."""
        rows = [self._make_row(gene_id=None, chrom=None, strand=None, gpos=None)]
        path = tmp_path / "out.tsv"
        write_tsv(rows, str(path), _TSV_HEADER, _TSV_COLS, use_gzip=False)
        content = path.read_text()
        data = content.strip().split("\n")[1].split("\t")
        assert data[3:7] == ["NA", "NA", "NA", "NA"]

    def test_empty_parquet_has_gtf_columns(self):
        """Empty Parquet schema includes gene_id, chrom, strand, gpos."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tf:
            tmp_path = tf.name
        try:
            write_parquet([], tmp_path, _SITES_SCHEMA, _TSV_COLS)
            table = pq.read_table(tmp_path)
            assert "gene_id" in table.column_names
            assert "chrom" in table.column_names
            assert "strand" in table.column_names
            assert "gpos" in table.column_names
        finally:
            os.unlink(tmp_path)

    def test_tsv_cols_includes_gtf(self):
        """_TSV_COLS has gene_id, chrom, strand, gpos right after core ID columns."""
        assert _TSV_COLS[3:7] == ["gene_id", "chrom", "strand", "gpos"]


# ---------- validate_mod_codes ----------


class TestValidateModCodes:
    """Tests for validate_mod_codes()."""

    def test_identical_codes(self):
        codes = {"a": 4, "m": 5}
        result = validate_mod_codes([codes, codes], ["f1.h5", "f2.h5"])
        assert result == codes

    def test_mismatched_codes_raises(self):
        codes1 = {"a": 4}
        codes2 = {"a": 5}
        with pytest.raises(ValueError, match="do not match"):
            validate_mod_codes([codes1, codes2], ["f1.h5", "f2.h5"])

    def test_extra_code_raises(self):
        codes1 = {"a": 4}
        codes2 = {"a": 4, "m": 5}
        with pytest.raises(ValueError, match="do not match"):
            validate_mod_codes([codes1, codes2], ["f1.h5", "f2.h5"])

    def test_single_file_no_validation(self):
        codes = {"a": 4}
        result = validate_mod_codes([codes], ["f1.h5"])
        assert result == codes


# ---------- validate_tx_lengths ----------


class TestValidateTxLengths:
    """Tests for validate_tx_lengths()."""

    def test_identical_lengths(self):
        result = validate_tx_lengths(
            "TX1", [100, 100, 100], ["f1.h5", "f2.h5", "f3.h5"]
        )
        assert result == 100

    def test_with_none_absent(self):
        result = validate_tx_lengths(
            "TX1", [100, None, 100], ["f1.h5", "f2.h5", "f3.h5"]
        )
        assert result == 100

    def test_mismatch_raises(self):
        with pytest.raises(ValueError, match="inconsistent lengths"):
            validate_tx_lengths("TX1", [100, 200], ["f1.h5", "f2.h5"])


# ---------- multi-file integration ----------


class TestMainMultiFile:
    """Integration tests for main() with multiple HDF5 files."""

    @staticmethod
    def _make_h5(path, tx_data, mod_codes):
        """Create a minimal HDF5 file for testing.

        Args:
            path: Output file path.
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

    def test_single_file_unchanged(self, tmp_path):
        """Single HDF5 file produces expected output."""
        h5_path = str(tmp_path / "test.h5")
        out_path = str(tmp_path / "out.parquet")

        matrix = np.array([[4, 1], [4, 1]], dtype=np.uint8)
        weights = np.array([0.8, 0.9], dtype=np.float32)
        self._make_h5(h5_path, {"TX1": (matrix, weights)}, {"a": 4})

        args = argparse.Namespace(
            h5=[h5_path],
            output=out_path,
            format="parquet",
            gzip=False,
            sites=None,
            min_asp=0.0,
            transcripts=None,
            gtf=None,
            verbose=False,
        )
        main(args)

        table = pq.read_table(out_path)
        assert len(table) == 1  # position 1 has mod
        assert table.column("transcript_id")[0].as_py() == "TX1"

    def test_two_files_disjoint_transcripts(self, tmp_path):
        """Transcripts unique to each file both appear in output."""
        h5_a = str(tmp_path / "a.h5")
        h5_b = str(tmp_path / "b.h5")
        out_path = str(tmp_path / "out.parquet")

        self._make_h5(
            h5_a,
            {
                "TX1": (
                    np.array([[4, 1]], dtype=np.uint8),
                    np.array([0.8], dtype=np.float32),
                )
            },
            {"a": 4},
        )
        self._make_h5(
            h5_b,
            {
                "TX2": (
                    np.array([[1, 4]], dtype=np.uint8),
                    np.array([0.9], dtype=np.float32),
                )
            },
            {"a": 4},
        )

        args = argparse.Namespace(
            h5=[h5_a, h5_b],
            output=out_path,
            format="parquet",
            gzip=False,
            sites=None,
            min_asp=0.0,
            transcripts=None,
            gtf=None,
            verbose=False,
        )
        main(args)

        table = pq.read_table(out_path)
        txs = set(table.column("transcript_id").to_pylist())
        assert txs == {"TX1", "TX2"}

    def test_two_files_overlapping_transcript(self, tmp_path):
        """Reads for same transcript are pooled across files."""
        h5_a = str(tmp_path / "a.h5")
        h5_b = str(tmp_path / "b.h5")
        out_path = str(tmp_path / "out.parquet")

        self._make_h5(
            h5_a,
            {
                "TX1": (
                    np.array([[4, 1]], dtype=np.uint8),
                    np.array([0.8], dtype=np.float32),
                )
            },
            {"a": 4},
        )
        self._make_h5(
            h5_b,
            {
                "TX1": (
                    np.array([[4, 1]], dtype=np.uint8),
                    np.array([0.9], dtype=np.float32),
                )
            },
            {"a": 4},
        )

        args = argparse.Namespace(
            h5=[h5_a, h5_b],
            output=out_path,
            format="parquet",
            gzip=False,
            sites=None,
            min_asp=0.0,
            transcripts=None,
            gtf=None,
            verbose=False,
        )
        main(args)

        table = pq.read_table(out_path)
        r = table.to_pylist()[0]
        assert r["transcript_id"] == "TX1"
        assert r["position"] == 1
        assert r["n_modified"] == 2
        assert r["n_canonical"] == 0
        assert r["mod_level"] == pytest.approx(1.0)

    def test_filter_transcripts_union(self, tmp_path):
        """--transcripts filter works on the union across files."""
        h5_a = str(tmp_path / "a.h5")
        h5_b = str(tmp_path / "b.h5")
        out_path = str(tmp_path / "out.parquet")

        self._make_h5(
            h5_a,
            {
                "TX1": (
                    np.array([[4]], dtype=np.uint8),
                    np.array([0.8], dtype=np.float32),
                ),
                "TX2": (
                    np.array([[1]], dtype=np.uint8),
                    np.array([0.8], dtype=np.float32),
                ),
            },
            {"a": 4},
        )
        self._make_h5(
            h5_b,
            {
                "TX1": (
                    np.array([[4]], dtype=np.uint8),
                    np.array([0.9], dtype=np.float32),
                ),
                "TX3": (
                    np.array([[4]], dtype=np.uint8),
                    np.array([0.9], dtype=np.float32),
                ),
            },
            {"a": 4},
        )

        args = argparse.Namespace(
            h5=[h5_a, h5_b],
            output=out_path,
            format="parquet",
            gzip=False,
            sites=None,
            min_asp=0.0,
            transcripts=["TX1", "TX3"],
            gtf=None,
            verbose=False,
        )
        main(args)

        table = pq.read_table(out_path)
        txs = set(table.column("transcript_id").to_pylist())
        assert txs == {"TX1", "TX3"}

    def test_sites_two_col_no_mod_type(self, tmp_path):
        """Sites file (2 cols, no mod_type) emits all mod types."""
        h5_path = str(tmp_path / "test.h5")
        out_path = str(tmp_path / "out.parquet")
        sites_path = str(tmp_path / "sites.tsv")

        matrix = np.array([[4, 5, 1]], dtype=np.uint8)
        weights = np.array([0.8], dtype=np.float32)
        self._make_h5(h5_path, {"TX1": (matrix, weights)}, {"a": 4, "m": 5})
        with open(sites_path, "w") as f:
            f.write("TX1\t1\n")

        args = argparse.Namespace(
            h5=[h5_path], output=out_path, format="parquet",
            gzip=False, sites=sites_path, min_asp=0.0,
            transcripts=None, gtf=None, verbose=False,
        )
        main(args)

        table = pq.read_table(out_path)
        rows = table.to_pylist()
        mod_types = {r["mod_type"] for r in rows}
        assert mod_types == {"a", "m"}
        assert all(r["position"] == 1 for r in rows)

    def test_sites_with_mod_type_filter(self, tmp_path):
        """Sites file with mod_type emits only that type."""
        h5_path = str(tmp_path / "test.h5")
        out_path = str(tmp_path / "out.parquet")
        sites_path = str(tmp_path / "sites.tsv")

        matrix = np.array([[4, 5, 1]], dtype=np.uint8)
        weights = np.array([0.8], dtype=np.float32)
        self._make_h5(h5_path, {"TX1": (matrix, weights)}, {"a": 4, "m": 5})
        with open(sites_path, "w") as f:
            f.write("TX1\t2\tm\n")

        args = argparse.Namespace(
            h5=[h5_path], output=out_path, format="parquet",
            gzip=False, sites=sites_path, min_asp=0.0,
            transcripts=None, gtf=None, verbose=False,
        )
        main(args)

        table = pq.read_table(out_path)
        rows = table.to_pylist()
        assert len(rows) == 1
        assert rows[0]["position"] == 2
        assert rows[0]["mod_type"] == "m"
        assert rows[0]["n_modified"] == 1

    def test_sites_missing_transcript(self, tmp_path):
        """Transcript in sites file but absent from H5 → zero rows."""
        h5_path = str(tmp_path / "test.h5")
        out_path = str(tmp_path / "out.parquet")
        sites_path = str(tmp_path / "sites.tsv")

        self._make_h5(
            h5_path,
            {"TX1": (np.array([[4]], dtype=np.uint8),
                      np.array([0.8], dtype=np.float32))},
            {"a": 4},
        )
        with open(sites_path, "w") as f:
            f.write("MissingTX\t42\ta\n")

        args = argparse.Namespace(
            h5=[h5_path], output=out_path, format="parquet",
            gzip=False, sites=sites_path, min_asp=0.0,
            transcripts=None, gtf=None, verbose=False,
        )
        main(args)

        table = pq.read_table(out_path)
        rows = table.to_pylist()
        assert len(rows) == 1
        assert rows[0]["transcript_id"] == "MissingTX"
        assert rows[0]["position"] == 42
        assert rows[0]["mod_type"] == "a"
        assert rows[0]["n_modified"] == 0
        assert rows[0]["n_unmodified"] == 0

    def test_sites_mixed_present_and_absent(self, tmp_path):
        """Both present and absent transcripts appear in output."""
        h5_path = str(tmp_path / "test.h5")
        out_path = str(tmp_path / "out.parquet")
        sites_path = str(tmp_path / "sites.tsv")

        self._make_h5(
            h5_path,
            {"TX1": (np.array([[4]], dtype=np.uint8),
                      np.array([0.8], dtype=np.float32))},
            {"a": 4},
        )
        with open(sites_path, "w") as f:
            f.write("TX1\t1\ta\nMissingTX\t5\ta\n")

        args = argparse.Namespace(
            h5=[h5_path], output=out_path, format="parquet",
            gzip=False, sites=sites_path, min_asp=0.0,
            transcripts=None, gtf=None, verbose=False,
        )
        main(args)

        table = pq.read_table(out_path)
        rows = table.to_pylist()
        txs = {r["transcript_id"] for r in rows}
        assert txs == {"TX1", "MissingTX"}
        # TX1 has real data
        tx1_row = [r for r in rows if r["transcript_id"] == "TX1"][0]
        assert tx1_row["n_modified"] == 1
        # MissingTX has zero rows
        missing_row = [r for r in rows if r["transcript_id"] == "MissingTX"][0]
        assert missing_row["n_modified"] == 0

    def test_sites_with_transcripts_filter(self, tmp_path):
        """--transcripts filter applied to sites file transcripts."""
        h5_path = str(tmp_path / "test.h5")
        out_path = str(tmp_path / "out.parquet")
        sites_path = str(tmp_path / "sites.tsv")

        self._make_h5(
            h5_path,
            {"TX1": (np.array([[4]], dtype=np.uint8),
                      np.array([0.8], dtype=np.float32))},
            {"a": 4},
        )
        with open(sites_path, "w") as f:
            f.write("TX1\t1\ta\nTX2\t5\ta\n")

        args = argparse.Namespace(
            h5=[h5_path], output=out_path, format="parquet",
            gzip=False, sites=sites_path, min_asp=0.0,
            transcripts=["TX1"], gtf=None, verbose=False,
        )
        main(args)

        table = pq.read_table(out_path)
        txs = set(table.column("transcript_id").to_pylist())
        assert txs == {"TX1"}

    def test_sites_empty_file(self, tmp_path):
        """Empty sites file produces empty output."""
        h5_path = str(tmp_path / "test.h5")
        out_path = str(tmp_path / "out.parquet")
        sites_path = str(tmp_path / "sites.tsv")

        self._make_h5(
            h5_path,
            {"TX1": (np.array([[4]], dtype=np.uint8),
                      np.array([0.8], dtype=np.float32))},
            {"a": 4},
        )
        with open(sites_path, "w") as f:
            f.write("")

        args = argparse.Namespace(
            h5=[h5_path], output=out_path, format="parquet",
            gzip=False, sites=sites_path, min_asp=0.0,
            transcripts=None, gtf=None, verbose=False,
        )
        main(args)

        table = pq.read_table(out_path)
        assert len(table) == 0
