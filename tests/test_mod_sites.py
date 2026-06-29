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
    from isolens.mod_scan import (
        CODE_CANONICAL,
        CODE_DELETION,
        CODE_FAIL,
        CODE_MISMATCH,
    )
    from isolens.mod_sites import (
        _TSV_COLS,
        _validate_mod_codes,
        _validate_tx_lengths,
        _write_parquet,
        _write_tsv,
        compute_transcript_stats,
        main,
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
        _TSV_COLS,
        _validate_mod_codes,
        _validate_tx_lengths,
        _write_parquet,
        _write_tsv,
        compute_transcript_stats,
        main,
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
                "gene_id": None,
                "chrom": None,
                "strand": None,
                "gpos": None,
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
                "gene_id": None,
                "chrom": None,
                "strand": None,
                "gpos": None,
            }
        ]
        path = tmp_path / "out.tsv.gz"
        _write_tsv(rows, str(path), use_gzip=True)

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
            _write_parquet(rows, tmp_path)
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
            _write_parquet(rows, tmp_path)
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
        _write_tsv(rows, str(path), use_gzip=False)
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
        assert cols[-4:] == ["gene_id", "chrom", "strand", "gpos"]
        data = lines[1].split("\t")
        assert data[-4:] == ["G1", "chr1", "+", "142"]

    def test_tsv_with_null_gtf_columns(self, tmp_path):
        """TSV writes 'NA' for None GTF values."""
        rows = [self._make_row(gene_id=None, chrom=None, strand=None, gpos=None)]
        path = tmp_path / "out.tsv"
        _write_tsv(rows, str(path), use_gzip=False)
        content = path.read_text()
        data = content.strip().split("\n")[1].split("\t")
        assert data[-4:] == ["NA", "NA", "NA", "NA"]

    def test_empty_parquet_has_gtf_columns(self):
        """Empty Parquet schema includes gene_id, chrom, strand, gpos."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tf:
            tmp_path = tf.name
        try:
            _write_parquet([], tmp_path)
            table = pq.read_table(tmp_path)
            assert "gene_id" in table.column_names
            assert "chrom" in table.column_names
            assert "strand" in table.column_names
            assert "gpos" in table.column_names
        finally:
            os.unlink(tmp_path)

    def test_tsv_cols_includes_gtf(self):
        """_TSV_COLS has gene_id, chrom, strand, gpos as last four entries."""
        assert _TSV_COLS[-4:] == ["gene_id", "chrom", "strand", "gpos"]


# ---------- _validate_mod_codes ----------


class TestValidateModCodes:
    """Tests for _validate_mod_codes()."""

    def test_identical_codes(self):
        codes = [("a", 4), ("m", 5)]
        result = _validate_mod_codes([codes, codes], ["f1.h5", "f2.h5"])
        assert result == codes

    def test_mismatched_codes_raises(self):
        codes1 = [("a", 4)]
        codes2 = [("a", 5)]
        with pytest.raises(ValueError, match="do not match"):
            _validate_mod_codes([codes1, codes2], ["f1.h5", "f2.h5"])

    def test_extra_code_raises(self):
        codes1 = [("a", 4)]
        codes2 = [("a", 4), ("m", 5)]
        with pytest.raises(ValueError, match="do not match"):
            _validate_mod_codes([codes1, codes2], ["f1.h5", "f2.h5"])

    def test_single_file_no_validation(self):
        codes = [("a", 4)]
        result = _validate_mod_codes([codes], ["f1.h5"])
        assert result == codes


# ---------- _validate_tx_lengths ----------


class TestValidateTxLengths:
    """Tests for _validate_tx_lengths()."""

    def test_identical_lengths(self):
        result = _validate_tx_lengths(
            "TX1", [100, 100, 100], ["f1.h5", "f2.h5", "f3.h5"]
        )
        assert result == 100

    def test_with_none_absent(self):
        result = _validate_tx_lengths(
            "TX1", [100, None, 100], ["f1.h5", "f2.h5", "f3.h5"]
        )
        assert result == 100

    def test_mismatch_raises(self):
        with pytest.raises(ValueError, match="inconsistent lengths"):
            _validate_tx_lengths("TX1", [100, 200], ["f1.h5", "f2.h5"])


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
