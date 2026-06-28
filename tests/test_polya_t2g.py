"""Tests for polya_t2g — transcript-to-gene aggregation via GTF."""

import argparse
import gzip
import os
import sys

import pytest

try:
    from isolens._gtf import build_tx_to_gene
    from isolens.polya_t2g import main
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from _gtf import build_tx_to_gene  # type: ignore[no-redef]

    from isolens.polya_t2g import (  # type: ignore[no-redef]
        main,
    )


# Minimal valid GTF lines — gppy requires gene_id and transcript_id
# in the attributes column (9th field).
_GTF_HEADER = "##gtf-version 2.2\n"


def _make_gtf_line(chrom, feature, start, end, strand, attrs):
    """Build a minimal GTF line with the given attributes string."""
    return f"{chrom}\t.\t{feature}\t{start}\t{end}\t.\t{strand}\t.\t{attrs}\n"


def _make_polya_tsv(path, lines: list[str], gzip_output: bool = False):
    """Write polya TSV content to *path*, optionally gzip-compressed."""
    content = "".join(line + "\n" for line in lines)
    if gzip_output:
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(content)
    else:
        path.write_text(content)


class TestBuildTxToGene:
    """Tests for build_tx_to_gene()."""

    def test_valid_gtf(self, tmp_path):
        """Three transcripts across two genes."""
        path = tmp_path / "test.gtf"
        lines = [
            _GTF_HEADER,
            _make_gtf_line("chr1", "gene", 1, 300, "+",
                           'gene_id "GENE_A";'),
            _make_gtf_line("chr1", "transcript", 1, 100, "+",
                           'gene_id "GENE_A"; transcript_id "TX1";'),
            _make_gtf_line("chr1", "exon", 1, 100, "+",
                           'gene_id "GENE_A"; transcript_id "TX1";'),
            _make_gtf_line("chr1", "transcript", 101, 200, "+",
                           'gene_id "GENE_A"; transcript_id "TX2";'),
            _make_gtf_line("chr1", "exon", 101, 200, "+",
                           'gene_id "GENE_A"; transcript_id "TX2";'),
            _make_gtf_line("chr1", "transcript", 201, 300, "+",
                           'gene_id "GENE_B"; transcript_id "TX3";'),
            _make_gtf_line("chr1", "exon", 201, 300, "+",
                           'gene_id "GENE_B"; transcript_id "TX3";'),
        ]
        path.write_text("".join(lines))
        mapping = build_tx_to_gene(str(path))
        assert mapping == {"TX1": "GENE_A", "TX2": "GENE_A", "TX3": "GENE_B"}

    def test_gzipped_gtf(self, tmp_path):
        """GTF can be gzip-compressed."""
        path = tmp_path / "test.gtf.gz"
        content = "".join([
            _GTF_HEADER,
            _make_gtf_line("chr1", "gene", 1, 200, "+",
                           'gene_id "GENE_X";'),
            _make_gtf_line("chr1", "transcript", 1, 100, "+",
                           'gene_id "GENE_X"; transcript_id "TX_A";'),
            _make_gtf_line("chr1", "exon", 1, 100, "+",
                           'gene_id "GENE_X"; transcript_id "TX_A";'),
            _make_gtf_line("chr1", "transcript", 101, 200, "+",
                           'gene_id "GENE_X"; transcript_id "TX_B";'),
            _make_gtf_line("chr1", "exon", 101, 200, "+",
                           'gene_id "GENE_X"; transcript_id "TX_B";'),
        ])
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(content)
        mapping = build_tx_to_gene(str(path))
        assert mapping == {"TX_A": "GENE_X", "TX_B": "GENE_X"}

    def test_empty_gtf(self, tmp_path):
        """GTF with no transcript entries returns empty dict."""
        path = tmp_path / "empty.gtf"
        path.write_text(_GTF_HEADER)
        mapping = build_tx_to_gene(str(path))
        assert mapping == {}

    def test_multiple_transcripts_one_gene(self, tmp_path):
        """All transcripts map to the same gene."""
        path = tmp_path / "test.gtf"
        lines = [_GTF_HEADER]
        for i, tx in enumerate(["TX_A", "TX_B", "TX_C"]):
            start = i * 100 + 1
            end = start + 99
            lines.append(
                _make_gtf_line("chr1", "transcript", start, end, "+",
                               f'gene_id "G1"; transcript_id "{tx}";')
            )
            lines.append(
                _make_gtf_line("chr1", "exon", start, end, "+",
                               f'gene_id "G1"; transcript_id "{tx}";')
            )
        path.write_text("".join(lines))
        mapping = build_tx_to_gene(str(path))
        assert mapping == {"TX_A": "G1", "TX_B": "G1", "TX_C": "G1"}


class TestMainIntegration:
    """Integration tests for main() with various input configurations."""

    def test_uses_gene_id_from_input(self, tmp_path):
        """Input has gene_id column — use it directly, no --gtf needed."""
        in_path = tmp_path / "in.tsv"
        out_path = tmp_path / "out.tsv"
        _make_polya_tsv(in_path, [
            "transcript_id\ttx_idx\tn_reads\tpa_wlen\tprobs\tpa_lens\tgene_id",
            "TX1\t0\t2\t150.0\t1.0,1.0\t100,200\tGENE_A",
            "TX2\t1\t1\t300.0\t1.0\t300\tGENE_A",
            "TX3\t2\t1\t50.0\t1.0\t50\tGENE_B",
        ])
        args = argparse.Namespace(
            input=str(in_path),
            output=str(out_path),
            gtf=None,
            gzip=False,
        )
        main(args)

        lines = out_path.read_text().strip().split("\n")
        assert len(lines) == 3  # header + 2 genes
        assert lines[0] == "gene_id\tn_reads\tpa_wlen\tprobs\tpa_lens"
        # Gene A pools TX1 (2 reads) + TX2 (1 read)
        assert lines[1].startswith("GENE_A\t3\t")
        # Gene B pools TX3 (1 read)
        assert lines[2].startswith("GENE_B\t1\t")

    def test_skips_na_gene_id(self, tmp_path):
        """gene_id values of '' / 'NA' / '.' are skipped with a warning."""
        in_path = tmp_path / "in.tsv"
        out_path = tmp_path / "out.tsv"
        _make_polya_tsv(in_path, [
            "transcript_id\ttx_idx\tn_reads\tpa_wlen\tprobs\tpa_lens\tgene_id",
            "TX1\t0\t2\t150.0\t1.0,1.0\t100,200\tGENE_A",
            "TX2\t1\t1\t300.0\t1.0\t300\tNA",
            "TX3\t2\t1\t50.0\t1.0\t50\t.",
            "TX4\t3\t1\t60.0\t1.0\t60\t",
        ])
        args = argparse.Namespace(
            input=str(in_path),
            output=str(out_path),
            gtf=None,
            gzip=False,
        )
        main(args)

        lines = out_path.read_text().strip().split("\n")
        assert len(lines) == 2  # header + 1 gene
        assert lines[1].startswith("GENE_A\t2\t")

    def test_missing_gene_id_and_gtf_exits(self, tmp_path):
        """No gene_id column and no --gtf → exits with error."""
        in_path = tmp_path / "in.tsv"
        out_path = tmp_path / "out.tsv"
        _make_polya_tsv(in_path, [
            "transcript_id\ttx_idx\tn_reads\tpa_wlen\tprobs\tpa_lens",
            "TX1\t0\t2\t150.0\t1.0,1.0\t100,200",
        ])
        args = argparse.Namespace(
            input=str(in_path),
            output=str(out_path),
            gtf=None,
            gzip=False,
        )
        with pytest.raises(SystemExit):
            main(args)

    def test_gtf_fallback(self, tmp_path):
        """No gene_id column but --gtf provided → uses GTF mapping."""
        in_path = tmp_path / "in.tsv"
        out_path = tmp_path / "out.tsv"
        gtf_path = tmp_path / "test.gtf"
        _make_polya_tsv(in_path, [
            "transcript_id\ttx_idx\tn_reads\tpa_wlen\tprobs\tpa_lens",
            "TX1\t0\t2\t150.0\t1.0,1.0\t100,200",
            "TX2\t1\t1\t300.0\t1.0\t300",
        ])
        lines = [
            _GTF_HEADER,
            _make_gtf_line("chr1", "gene", 1, 200, "+",
                           'gene_id "GENE_A";'),
            _make_gtf_line("chr1", "transcript", 1, 100, "+",
                           'gene_id "GENE_A"; transcript_id "TX1";'),
            _make_gtf_line("chr1", "exon", 1, 100, "+",
                           'gene_id "GENE_A"; transcript_id "TX1";'),
            _make_gtf_line("chr1", "transcript", 101, 200, "+",
                           'gene_id "GENE_A"; transcript_id "TX2";'),
            _make_gtf_line("chr1", "exon", 101, 200, "+",
                           'gene_id "GENE_A"; transcript_id "TX2";'),
        ]
        gtf_path.write_text("".join(lines))

        args = argparse.Namespace(
            input=str(in_path),
            output=str(out_path),
            gtf=str(gtf_path),
            gzip=False,
        )
        main(args)

        lines = out_path.read_text().strip().split("\n")
        assert len(lines) == 2  # header + 1 gene
        assert lines[1].startswith("GENE_A\t3\t")
