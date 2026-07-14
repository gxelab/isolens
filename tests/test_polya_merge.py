"""Tests for polya_merge — merging two poly(A) TSV files."""

import gzip
import os
import sys

import pytest

try:
    from isolens.polya_merge import _read_polya_to_dict
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from isolens.polya_merge import (  # type: ignore[no-redef]
        _read_polya_to_dict,
    )


class TestReadPolyaToDict:
    """Tests for _read_polya_to_dict()."""

    def test_valid_tsv(self, tmp_path):
        path = tmp_path / "test.tsv"
        path.write_text(
            "transcript_id\tn_reads\ttotal_wt\twmlen\tweights\tlengths\n"
            "TX1\t2\t1.0\t100.5\t0.5,0.5\t100,101\n"
            "TX2\t1\t1.0\t200.0\t1.0\t200\n"
        )
        data = _read_polya_to_dict(str(path))
        assert len(data) == 2
        assert data["TX1"]["weights"] == [0.5, 0.5]
        assert data["TX1"]["lengths"] == [100, 101]
        assert "TX2" in data

    def test_gzipped_tsv(self, tmp_path):
        path = tmp_path / "test.tsv.gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(
                "transcript_id\tn_reads\ttotal_wt\twmlen\tweights\tlengths\n"
                "TX1\t1\t1.0\t42.0\t1.0\t42\n"
            )
        data = _read_polya_to_dict(str(path))
        assert data["TX1"]["lengths"] == [42]

    def test_missing_transcript_id(self, tmp_path):
        """Header without transcript_id should sys.exit(1)."""
        path = tmp_path / "bad.tsv"
        path.write_text("name\tcount\ttotal\tlen\tw\tl\nTX1\t1\t1.0\t1.0\t1.0\t1\n")
        with pytest.raises(SystemExit):
            _read_polya_to_dict(str(path))

    def test_empty_data(self, tmp_path):
        path = tmp_path / "empty.tsv"
        path.write_text("transcript_id\tn_reads\ttotal_wt\twmlen\tweights\tlengths\n")
        data = _read_polya_to_dict(str(path))
        assert data == {}

    def test_short_lines_skipped(self, tmp_path):
        path = tmp_path / "test.tsv"
        path.write_text(
            "transcript_id\tn_reads\ttotal_wt\twmlen\tweights\tlengths\n"
            "TX1\t2\t1.0\t100.5\t0.5,0.5\t100,101\n"
            "short\n"
            "TX2\t1\t1.0\t200.0\t1.0\t200\n"
        )
        data = _read_polya_to_dict(str(path))
        assert len(data) == 2  # short line skipped
