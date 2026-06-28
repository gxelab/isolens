"""Tests for polya_merge — merging two poly(A) TSV files."""

import gzip
import os
import sys

import pytest

try:
    from isolens.polya_merge import read_tsv_to_dict
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from isolens.polya_merge import (  # type: ignore[no-redef]
        read_tsv_to_dict,
    )


class TestReadTsvToDict:
    """Tests for read_tsv_to_dict()."""

    def test_valid_tsv(self, tmp_path):
        path = tmp_path / "test.tsv"
        path.write_text(
            "transcript_id\ttx_idx\tn_reads\tpa_wlen\tprobs\tpa_lens\n"
            "TX1\t0\t2\t100.5\t0.5,0.5\t100,101\n"
            "TX2\t1\t1\t200.0\t1.0\t200\n"
        )
        data = read_tsv_to_dict(str(path))
        assert len(data) == 2
        assert data[0]["tx_name"] == "TX1"
        assert data[0]["probs"] == [0.5, 0.5]
        assert data[0]["pa_lens"] == [100, 101]
        assert data[1]["tx_name"] == "TX2"

    def test_gzipped_tsv(self, tmp_path):
        path = tmp_path / "test.tsv.gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(
                "transcript_id\ttx_idx\tn_reads\tpa_wlen\tprobs\tpa_lens\n"
                "TX1\t0\t1\t42.0\t1.0\t42\n"
            )
        data = read_tsv_to_dict(str(path))
        assert data[0]["pa_lens"] == [42]

    def test_malformed_header(self, tmp_path):
        """Header without tx_idx in position 1 should sys.exit(1)."""
        path = tmp_path / "bad.tsv"
        path.write_text("name\tid\tcount\tlen\tp\tl\n1\t0\t1\t1.0\t1.0\t1\n")
        with pytest.raises(SystemExit):
            read_tsv_to_dict(str(path))

    def test_empty_data(self, tmp_path):
        path = tmp_path / "empty.tsv"
        path.write_text("transcript_id\ttx_idx\tn_reads\tpa_wlen\tprobs\tpa_lens\n")
        data = read_tsv_to_dict(str(path))
        assert data == {}

    def test_short_lines_skipped(self, tmp_path):
        path = tmp_path / "test.tsv"
        path.write_text(
            "transcript_id\ttx_idx\tn_reads\tpa_wlen\tprobs\tpa_lens\n"
            "TX1\t0\t2\t100.5\t0.5,0.5\t100,101\n"
            "short\n"
            "TX2\t1\t1\t200.0\t1.0\t200\n"
        )
        data = read_tsv_to_dict(str(path))
        assert len(data) == 2  # short line skipped
