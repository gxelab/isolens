"""Tests for polya_t2g — transcript-to-gene aggregation."""

import gzip
import os
import sys

import pytest

try:
    from isolens.polya_t2g import load_gene_mapping
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from isolens.polya_t2g import (  # type: ignore[no-redef]
        load_gene_mapping,
    )


class TestLoadGeneMapping:
    """Tests for load_gene_mapping()."""

    def test_valid_mapping(self, tmp_path):
        path = tmp_path / "map.tsv"
        path.write_text("tx_name\tgene_id\nTX1\tGENE_A\nTX2\tGENE_A\nTX3\tGENE_B\n")
        mapping = load_gene_mapping(str(path))
        assert mapping == {"TX1": "GENE_A", "TX2": "GENE_A", "TX3": "GENE_B"}

    def test_gzipped_mapping(self, tmp_path):
        path = tmp_path / "map.tsv.gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write("tx_name\tgene_id\nTX1\tGENE_A\n")
        mapping = load_gene_mapping(str(path))
        assert mapping == {"TX1": "GENE_A"}

    def test_missing_columns(self, tmp_path):
        path = tmp_path / "bad.tsv"
        path.write_text("tx_name\textra\nTX1\tval\n")
        with pytest.raises(SystemExit):
            load_gene_mapping(str(path))

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.tsv"
        path.write_text("tx_name\tgene_id\n")
        mapping = load_gene_mapping(str(path))
        assert mapping == {}

    def test_short_lines_skipped(self, tmp_path):
        path = tmp_path / "map.tsv"
        path.write_text("tx_name\tgene_id\nTX1\tGENE_A\nshort\nTX2\tGENE_B\n")
        mapping = load_gene_mapping(str(path))
        assert mapping == {"TX1": "GENE_A", "TX2": "GENE_B"}
