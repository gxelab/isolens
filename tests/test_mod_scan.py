"""Tests for mod_scan module."""

import os
import sys
import tempfile

import h5py
import numpy as np
import pytest

# ---------- helpers to import mod_scan functions ----------

# We need to add the src directory so we can import isolens
# (this is already handled by the editable install, but being defensive)
try:
    from isolens.mod_scan import (
        _BAM_CDEL,
        _BAM_CDIFF,
        _BAM_CEQUAL,
        _BAM_CINS,
        _BAM_CMATCH,
        _BAM_CREF_SKIP,
        _BAM_CSOFT_CLIP,
        CODE_CANONICAL,
        CODE_DELETION,
        CODE_MISMATCH,
        CODE_UNCOVERED,
        parse_cigar_for_row,
        parse_modifications,
        write_transcript_group,
    )
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from isolens.mod_scan import (
        _BAM_CDEL,
        _BAM_CDIFF,
        _BAM_CEQUAL,
        _BAM_CINS,
        _BAM_CMATCH,
        _BAM_CREF_SKIP,
        _BAM_CSOFT_CLIP,
        CODE_CANONICAL,
        CODE_DELETION,
        CODE_MISMATCH,
        CODE_UNCOVERED,
        parse_cigar_for_row,
        parse_modifications,
        write_transcript_group,
    )


# ---------- mock AlignedSegment for unit tests ----------


class MockRecord:
    """Minimal mock of ``pysam.AlignedSegment`` for CIGAR / mod parsing tests."""

    def __init__(
        self,
        cigartuples,
        reference_start,
        query_sequence,
        mm_tag=None,
        ml_bytes=None,
        has_mm=True,
        has_ml=True,
    ):
        self.cigartuples = cigartuples
        self.reference_start = reference_start
        self.query_sequence = query_sequence
        self._mm_tag = mm_tag
        self._ml_bytes = ml_bytes
        self._has_mm = has_mm
        self._has_ml = has_ml

    def has_tag(self, tag):
        if tag in ("MM", "mm"):
            return self._has_mm and self._mm_tag is not None
        if tag in ("ML", "ml"):
            return self._has_ml and self._ml_bytes is not None
        return False

    def get_tag(self, tag):
        if tag in ("MM", "mm"):
            return self._mm_tag
        if tag in ("ML", "ml"):
            return self._ml_bytes
        raise KeyError(tag)


# ---------- CIGAR parsing tests ----------


class TestParseCigarForRow:
    """Unit tests for parse_cigar_for_row()."""

    def test_simple_match_all(self):
        """100= (exact match across 100 bases)."""
        record = MockRecord(
            cigartuples=[(_BAM_CEQUAL, 100)],
            reference_start=0,
            query_sequence="A" * 100,
        )
        row, r2t = parse_cigar_for_row(record, 100)

        assert row.shape == (100,)
        assert row.dtype == np.uint8
        assert np.all(row == CODE_CANONICAL)
        assert len(r2t) == 100
        assert r2t[0] == 1
        assert r2t[99] == 100

    def test_offset_start(self):
        """Alignment starts at reference position 50."""
        record = MockRecord(
            cigartuples=[(_BAM_CEQUAL, 10)],
            reference_start=50,
            query_sequence="A" * 10,
        )
        row, r2t = parse_cigar_for_row(record, 100)

        assert row.shape == (100,)
        assert np.all(row[:50] == CODE_UNCOVERED)
        assert np.all(row[50:60] == CODE_CANONICAL)
        assert np.all(row[60:] == CODE_UNCOVERED)
        assert r2t == [51, 52, 53, 54, 55, 56, 57, 58, 59, 60]

    def test_mismatch(self):
        """X operator produces CODE_MISMATCH."""
        record = MockRecord(
            cigartuples=[(_BAM_CEQUAL, 5), (_BAM_CDIFF, 3), (_BAM_CEQUAL, 5)],
            reference_start=0,
            query_sequence="A" * 13,
        )
        row, r2t = parse_cigar_for_row(record, 20)

        assert row[0] == CODE_CANONICAL
        assert row[4] == CODE_CANONICAL
        assert row[5] == CODE_MISMATCH
        assert row[7] == CODE_MISMATCH
        assert row[8] == CODE_CANONICAL

    def test_deletion(self):
        """D operator produces CODE_DELETION and no entries in read_to_tx_map."""
        record = MockRecord(
            cigartuples=[(_BAM_CEQUAL, 3), (_BAM_CDEL, 2), (_BAM_CEQUAL, 3)],
            reference_start=0,
            query_sequence="A" * 6,
        )
        row, r2t = parse_cigar_for_row(record, 10)

        assert row[0] == CODE_CANONICAL
        assert row[2] == CODE_CANONICAL
        assert row[3] == CODE_DELETION
        assert row[4] == CODE_DELETION
        assert row[5] == CODE_CANONICAL
        # Deletion positions consume no read bases
        assert len(r2t) == 6  # 3 before + 3 after deletion

    def test_insertion(self):
        """I operator produces None entries in read_to_tx_map."""
        record = MockRecord(
            cigartuples=[(_BAM_CEQUAL, 5), (_BAM_CINS, 2), (_BAM_CEQUAL, 5)],
            reference_start=0,
            query_sequence="A" * 12,
        )
        row, r2t = parse_cigar_for_row(record, 20)

        # Insertion does not consume reference positions
        assert row[5] == CODE_CANONICAL  # position right after first match block
        # read_to_tx_map has None for inserted bases
        assert len(r2t) == 12  # 5 match + 2 insert + 5 match
        assert r2t[4] == 5  # last match before insertion
        assert r2t[5] is None  # first inserted base
        assert r2t[6] is None  # second inserted base
        assert r2t[7] == 6  # first match after insertion

    def test_soft_clip(self):
        """Soft-clipped bases produce None in read_to_tx_map."""
        record = MockRecord(
            cigartuples=[(_BAM_CSOFT_CLIP, 4), (_BAM_CEQUAL, 10)],
            reference_start=0,
            query_sequence="A" * 14,
        )
        row, r2t = parse_cigar_for_row(record, 20)

        assert len(r2t) == 14  # 4 soft clip + 10 match
        assert r2t[0] is None  # soft-clipped
        assert r2t[3] is None  # soft-clipped
        assert r2t[4] == 1  # first aligned base

    def test_legacy_m_op(self):
        """M operator (no =/X) defaults to CODE_CANONICAL."""
        record = MockRecord(
            cigartuples=[(_BAM_CMATCH, 10)],
            reference_start=0,
            query_sequence="A" * 10,
        )
        row, r2t = parse_cigar_for_row(record, 10)
        assert np.all(row == CODE_CANONICAL)

    def test_ref_skip(self):
        """N operator (intron/skip) advances reference without read consumption."""
        record = MockRecord(
            cigartuples=[(_BAM_CEQUAL, 5), (_BAM_CREF_SKIP, 100), (_BAM_CEQUAL, 5)],
            reference_start=0,
            query_sequence="A" * 10,
        )
        row, r2t = parse_cigar_for_row(record, 200)

        assert np.all(row[:5] == CODE_CANONICAL)
        assert np.all(row[5:105] == CODE_UNCOVERED)
        assert np.all(row[105:110] == CODE_CANONICAL)
        assert len(r2t) == 10

    def test_out_of_bounds(self):
        """Positions beyond tx_length are silently ignored."""
        record = MockRecord(
            cigartuples=[(_BAM_CEQUAL, 10)],
            reference_start=95,
            query_sequence="A" * 10,
        )
        row, r2t = parse_cigar_for_row(record, 100)

        assert np.all(row[:95] == CODE_UNCOVERED)
        assert np.all(row[95:100] == CODE_CANONICAL)
        # Only 5 positions within bounds
        assert r2t[:5] == [96, 97, 98, 99, 100]
        assert r2t[5:] == [None, None, None, None, None]

    def test_none_reference_start(self):
        """Record with None reference_start returns empty."""
        record = MockRecord(
            cigartuples=[(_BAM_CEQUAL, 10)],
            reference_start=None,
            query_sequence="A" * 10,
        )
        row, r2t = parse_cigar_for_row(record, 100)
        assert np.all(row == CODE_UNCOVERED)
        assert r2t == []


# ---------- modification parsing tests ----------


class TestParseModifications:
    """Unit tests for parse_modifications()."""

    def test_no_mm_tag(self):
        """Record without MM tag leaves row unchanged."""
        record = MockRecord(
            cigartuples=[(_BAM_CEQUAL, 10)],
            reference_start=0,
            query_sequence="A" * 10,
            has_mm=False,
            has_ml=False,
        )
        row, r2t = parse_cigar_for_row(record, 10)
        original = row.copy()

        mod_code_map = {}
        seen = set()
        parse_modifications(record, row, r2t, 200, mod_code_map, seen)

        np.testing.assert_array_equal(row, original)
        assert mod_code_map == {}
        assert seen == set()

    def test_basic_mod_call(self):
        """Single modification type, single site passing threshold."""
        # MM tag: A+a,0 — modify first A base with type 'a'
        # ML tag: [255] — max probability, passes any threshold
        mm_tag = "A+a,0"
        ml_bytes = bytes([255])  # raw probability 255/255 = 1.0

        record = MockRecord(
            cigartuples=[(_BAM_CEQUAL, 3)],
            reference_start=0,
            query_sequence="AAA",
            mm_tag=mm_tag,
            ml_bytes=ml_bytes,
        )
        row, r2t = parse_cigar_for_row(record, 3)
        mod_code_map = {}
        seen = set()

        parse_modifications(record, row, r2t, 200, mod_code_map, seen)

        # First position should be overridden with mod code 4
        assert row[0] == 4
        # Other positions remain canonical match
        assert row[1] == CODE_CANONICAL
        assert row[2] == CODE_CANONICAL
        assert seen == {"a"}
        assert mod_code_map == {"a": 4}

    def test_mod_below_threshold(self):
        """Modification below ML threshold is not applied."""
        mm_tag = "A+a,0"
        ml_bytes = bytes([100])  # 100/255 ≈ 0.39, below most thresholds

        record = MockRecord(
            cigartuples=[(_BAM_CEQUAL, 3)],
            reference_start=0,
            query_sequence="AAA",
            mm_tag=mm_tag,
            ml_bytes=ml_bytes,
        )
        row, r2t = parse_cigar_for_row(record, 3)
        mod_code_map = {}
        seen = set()

        # threshold_u8 = 200 → raw >= 200 passes, 100 does not
        parse_modifications(record, row, r2t, 200, mod_code_map, seen)

        assert row[0] == CODE_CANONICAL  # unchanged
        assert seen == {"a"}

    def test_skip_multiple(self):
        """Skip list: modify the 3rd and 5th occurrences, skip others."""
        # A+a,2,1 — skip 2, then modify the 3rd A; then skip 1, then modify the 5th A
        mm_tag = "A+a,2,1"
        ml_bytes = bytes([255, 255])

        record = MockRecord(
            cigartuples=[(_BAM_CEQUAL, 10)],
            reference_start=0,
            query_sequence="AAAAAAAAAA",
            mm_tag=mm_tag,
            ml_bytes=ml_bytes,
        )
        row, r2t = parse_cigar_for_row(record, 10)
        mod_code_map = {}
        seen = set()

        parse_modifications(record, row, r2t, 100, mod_code_map, seen)

        # 3rd A (index 2, 0-based) should be modified
        assert row[2] == 4
        # 5th A (index 4, 0-based) should be modified
        assert row[4] == 4
        # Others remain canonical
        assert row[0] == CODE_CANONICAL
        assert row[1] == CODE_CANONICAL
        assert row[3] == CODE_CANONICAL

    def test_multiple_mod_types(self):
        """Two different modification types with separate codes."""
        mm_tag = "A+a,0;C+m,0"
        ml_bytes = bytes([255, 255])

        record = MockRecord(
            cigartuples=[(_BAM_CEQUAL, 3), (_BAM_CEQUAL, 3)],
            reference_start=0,
            query_sequence="AAACCC",
            mm_tag=mm_tag,
            ml_bytes=ml_bytes,
        )
        row, r2t = parse_cigar_for_row(record, 6)
        mod_code_map = {}
        seen = set()

        parse_modifications(record, row, r2t, 200, mod_code_map, seen)

        assert row[0] == 4  # 'a' mod on A
        assert row[3] == 5  # 'm' mod on C
        assert seen == {"a", "m"}
        assert mod_code_map == {"a": 4, "m": 5}

    def test_insertion_skip(self):
        """Modification skip-list correctly skips insertions in read_to_tx_map."""
        mm_tag = "A+a,0"
        ml_bytes = bytes([255])

        # CIGAR: 3 match, 2 insert, 1 match — 6 bases in query sequence
        record = MockRecord(
            cigartuples=[(_BAM_CEQUAL, 3), (_BAM_CINS, 2), (_BAM_CEQUAL, 1)],
            reference_start=0,
            query_sequence="GGGAAG",  # GGG = match, AA = inserted, G = match
            mm_tag=mm_tag,
            ml_bytes=ml_bytes,
        )
        row, r2t = parse_cigar_for_row(record, 10)
        mod_code_map = {}

        # seq has no 'A', so the mod skip "0" won't find the target base
        # This tests that read_to_tx_map length == len(query_sequence)
        assert len(r2t) == 6  # 3 match + 2 insert + 1 match

        parse_modifications(record, row, r2t, 200, mod_code_map, set())

        # No A in sequence → no modifications applied
        assert np.all(row[:4] == CODE_CANONICAL)


# ---------- HDF5 writing tests ----------


class TestWriteTranscriptGroup:
    """Tests for write_transcript_group()."""

    def test_basic_write(self):
        """Write a simple transcript group and verify structure."""
        rows = [
            np.array([0, 1, 1, 0, 2], dtype=np.uint8),
            np.array([1, 1, 0, 0, 1], dtype=np.uint8),
            np.array([0, 0, 1, 1, 3], dtype=np.uint8),
        ]
        read_ids = ["read-001", "read-002", "read-003"]
        weights = [0.5, 0.3, 0.2]

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tf:
            tmp_path = tf.name

        try:
            with h5py.File(tmp_path, "w") as h5:
                write_transcript_group(h5, "TEST_TX", rows, read_ids, weights)

            with h5py.File(tmp_path, "r") as h5:
                grp = h5["transcripts/TEST_TX"]
                matrix = grp["matrix"]
                assert matrix.shape == (3, 5)
                assert matrix.dtype == np.uint8
                assert matrix.compression == "gzip"
                assert matrix.shuffle is True

                # Verify matrix values
                m = matrix[:]
                assert m[0, 1] == 1
                assert m[0, 4] == 2
                assert m[2, 4] == 3

                # Verify read IDs (h5py returns bytes for variable-length strings)
                ids = grp["read_ids"]
                decoded_ids = [
                    x.decode() if isinstance(x, bytes) else x for x in ids[:]
                ]
                assert decoded_ids == read_ids

                # Verify weights
                w = grp["read_weights"]
                assert w.shape == (3,)
                assert w.dtype == np.float32
                expected = np.array(weights, dtype=np.float32)
                np.testing.assert_array_almost_equal(w[:], expected)
        finally:
            os.unlink(tmp_path)

    def test_empty_rows_skipped(self):
        """Empty row list produces no group."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tf:
            tmp_path = tf.name

        try:
            with h5py.File(tmp_path, "w") as h5:
                write_transcript_group(h5, "EMPTY_TX", [], [], [])

            with h5py.File(tmp_path, "r") as h5:
                assert "transcripts" not in h5
                assert "EMPTY_TX" not in h5
        finally:
            os.unlink(tmp_path)

    def test_row_length_mismatch_raises(self):
        """Rows of different lengths raise ValueError."""
        rows = [
            np.array([0, 1, 1], dtype=np.uint8),
            np.array([0, 1], dtype=np.uint8),  # shorter
        ]
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tf:
            tmp_path = tf.name

        try:
            with h5py.File(tmp_path, "w") as h5:
                with pytest.raises(ValueError, match="Row length mismatch"):
                    write_transcript_group(h5, "BAD_TX", rows, ["a", "b"], [0.5, 0.5])
        finally:
            os.unlink(tmp_path)


# ---------- integration test ----------


def test_integration_example_data():
    """Run mod_scan on example data and verify HDF5 output."""
    import subprocess

    example_dir = os.path.join(os.path.dirname(__file__), "..", "examples")
    bam_path = os.path.join(example_dir, "example.txmap.bam")
    lz4_path = os.path.join(example_dir, "example.lz4")

    if not os.path.isfile(bam_path) or not os.path.isfile(lz4_path):
        pytest.skip("Example data files not found")

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tf:
        out_path = tf.name

    try:
        # Run mod_scan as a subprocess
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "isolens.mod_scan",
                "-b",
                bam_path,
                "-a",
                lz4_path,
                "--output",
                out_path,
                "--mod-cutoff",
                "0.95",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert result.returncode == 0, f"mod_scan failed:\n{result.stderr}"

        # Verify HDF5 output
        with h5py.File(out_path, "r") as f:
            # Transcripts
            tx_list = list(f["transcripts"].keys())
            assert len(tx_list) == 2
            assert "FBtr0073078" in tx_list
            assert "FBtr0073079" in tx_list

            # Check FBtr0073078
            grp = f["transcripts/FBtr0073078"]
            matrix = grp["matrix"]
            assert matrix.ndim == 2
            assert matrix.dtype == np.uint8
            assert matrix.shape[0] >= 1  # at least one read
            assert matrix.shape[1] >= 2500  # transcript length
            assert matrix.compression == "gzip"
            assert matrix.shuffle

            # All values should be in valid range
            vals = matrix[:]
            assert vals.min() >= CODE_UNCOVERED
            assert vals.max() >= CODE_CANONICAL  # at minimum has canonical matches

            # read_ids exists and matches n_reads
            assert len(grp["read_ids"]) == matrix.shape[0]
            assert grp["read_weights"].shape[0] == matrix.shape[0]
            assert grp["read_weights"].dtype == np.float32

            # Check FBtr0073079
            grp2 = f["transcripts/FBtr0073079"]
            assert grp2["matrix"].shape[0] >= 1

            # Modification codes
            codes_grp = f["modification_codes"]
            assert len(codes_grp.attrs) >= 1

            # Metadata
            meta = f["metadata"]
            assert meta.attrs["mod_cutoff"] == 0.95
            assert "pipeline_version" in meta.attrs
            assert int(meta.attrs["n_transcripts"]) == 2
            assert int(meta.attrs["n_assignments"]) >= 1

    finally:
        os.unlink(out_path)
