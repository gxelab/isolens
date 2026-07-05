"""Tests for polya_dpt — pairwise isoform poly(A) comparison."""

import argparse
import gzip
import os
import sys

import numpy as np
import pytest

try:
    from isolens.polya_dpt import main
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from isolens.polya_dpt import main  # type: ignore[no-redef]


def _make_polya_tsv(path, lines: list[str], gzip_output: bool = False):
    """Write polya TSV content to *path*, optionally gzip-compressed."""
    content = "".join(line + "\n" for line in lines)
    if gzip_output:
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(content)
    else:
        path.write_text(content)


_GTF_HEADER = "##gtf-version 2.2\n"


def _make_gtf_line(chrom, feature, start, end, strand, attrs):
    """Build a minimal GTF line with the given attributes string."""
    return f"{chrom}\t.\t{feature}\t{start}\t{end}\t.\t{strand}\t.\t{attrs}\n"


class TestMainIntegration:
    """Integration tests for polya_dpt main()."""

    def test_single_gene_two_transcripts(self, tmp_path):
        in_path = tmp_path / "in.tsv"
        out_path = tmp_path / "out.tsv"
        _make_polya_tsv(
            in_path,
            [
                "transcript_id\tn_reads\ttotal_wt\twmlen\tweights\tlengths\tgene_id",
                "TX1\t3\t3.0\t150.0\t1.0,1.0,1.0\t100,150,200\tGENE_A",
                "TX2\t3\t3.0\t120.0\t1.0,1.0,1.0\t80,120,160\tGENE_A",
            ],
        )
        args = argparse.Namespace(
            input=str(in_path),
            output=str(out_path),
            format="tsv",
            gtf=None,
            gzip=False,
            min_asp=0.0,
            min_pareads=1,
        )
        main(args)

        lines = out_path.read_text().strip().split("\n")
        assert len(lines) == 2  # header + 1 pair
        hdr = lines[0].split("\t")
        assert "gene_id" in hdr
        assert "transcript_1" in hdr
        assert "transcript_2" in hdr
        assert "ks_stat" in hdr
        assert "t_stat" in hdr
        assert "u_stat" in hdr

    def test_single_gene_three_transcripts(self, tmp_path):
        in_path = tmp_path / "in.tsv"
        out_path = tmp_path / "out.tsv"
        _make_polya_tsv(
            in_path,
            [
                "transcript_id\tn_reads\ttotal_wt\twmlen\tweights\tlengths\tgene_id",
                "TX1\t3\t3.0\t150.0\t1.0,1.0,1.0\t100,150,200\tGENE_A",
                "TX2\t3\t3.0\t140.0\t1.0,1.0,1.0\t90,140,190\tGENE_A",
                "TX3\t2\t3\t130.0\t1.0,1.0,1.0\t80,130,180\tGENE_A",
            ],
        )
        args = argparse.Namespace(
            input=str(in_path),
            output=str(out_path),
            format="tsv",
            gtf=None,
            gzip=False,
            min_asp=0.0,
            min_pareads=1,
        )
        main(args)

        lines = out_path.read_text().strip().split("\n")
        # 3 transcripts → 3 pairs
        assert len(lines) == 4  # header + 3 pairs

    def test_two_genes(self, tmp_path):
        in_path = tmp_path / "in.tsv"
        out_path = tmp_path / "out.tsv"
        _make_polya_tsv(
            in_path,
            [
                "transcript_id\tn_reads\ttotal_wt\twmlen\tweights\tlengths\tgene_id",
                "TX1\t2\t2.0\t150.0\t1.0,1.0\t100,200\tGENE_A",
                "TX2\t2\t2.0\t120.0\t1.0,1.0\t80,160\tGENE_A",
                "TX3\t2\t2\t300.0\t1.0,1.0\t250,350\tGENE_B",
                "TX4\t3\t2\t280.0\t1.0,1.0\t230,330\tGENE_B",
            ],
        )
        args = argparse.Namespace(
            input=str(in_path),
            output=str(out_path),
            format="tsv",
            gtf=None,
            gzip=False,
            min_asp=0.0,
            min_pareads=1,
        )
        main(args)

        lines = out_path.read_text().strip().split("\n")
        assert len(lines) == 3  # header + 2 pairs (1 per gene)

    def test_gtf_fallback(self, tmp_path):
        in_path = tmp_path / "in.tsv"
        out_path = tmp_path / "out.tsv"
        gtf_path = tmp_path / "test.gtf"
        _make_polya_tsv(
            in_path,
            [
                "transcript_id\tn_reads\ttotal_wt\twmlen\tweights\tlengths",
                "TX1\t3\t3.0\t150.0\t1.0,1.0,1.0\t100,150,200",
                "TX2\t3\t3.0\t120.0\t1.0,1.0,1.0\t80,120,160",
            ],
        )
        gtf_lines = [
            _GTF_HEADER,
            _make_gtf_line("chr1", "gene", 1, 200, "+", 'gene_id "GENE_A";'),
            _make_gtf_line(
                "chr1",
                "transcript",
                1,
                100,
                "+",
                'gene_id "GENE_A"; transcript_id "TX1";',
            ),
            _make_gtf_line(
                "chr1", "exon", 1, 100, "+", 'gene_id "GENE_A"; transcript_id "TX1";'
            ),
            _make_gtf_line(
                "chr1",
                "transcript",
                101,
                200,
                "+",
                'gene_id "GENE_A"; transcript_id "TX2";',
            ),
            _make_gtf_line(
                "chr1", "exon", 101, 200, "+", 'gene_id "GENE_A"; transcript_id "TX2";'
            ),
        ]
        gtf_path.write_text("".join(gtf_lines))

        args = argparse.Namespace(
            input=str(in_path),
            output=str(out_path),
            format="tsv",
            gtf=str(gtf_path),
            gzip=False,
            min_asp=0.0,
            min_pareads=1,
        )
        main(args)

        lines = out_path.read_text().strip().split("\n")
        assert len(lines) == 2  # header + 1 pair
        assert lines[1].startswith("GENE_A\t")

    def test_gene_id_column_present(self, tmp_path):
        in_path = tmp_path / "in.tsv"
        out_path = tmp_path / "out.tsv"
        _make_polya_tsv(
            in_path,
            [
                "transcript_id\tn_reads\ttotal_wt\twmlen\tweights\tlengths\tgene_id",
                "TX1\t3\t3.0\t150.0\t1.0,1.0,1.0\t100,150,200\tGENE_A",
                "TX2\t3\t3.0\t120.0\t1.0,1.0,1.0\t80,120,160\tGENE_A",
            ],
        )
        args = argparse.Namespace(
            input=str(in_path),
            output=str(out_path),
            format="tsv",
            gtf=None,
            gzip=False,
            min_asp=0.0,
            min_pareads=1,
        )
        main(args)

        lines = out_path.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_min_asp_filtering(self, tmp_path):
        in_path = tmp_path / "in.tsv"
        out_path = tmp_path / "out.tsv"
        # TX1: all weights < 0.5 → filtered out entirely
        _make_polya_tsv(
            in_path,
            [
                "transcript_id\tn_reads\ttotal_wt\twmlen\tweights\tlengths\tgene_id",
                "TX1\t3\t0.3\t150.0\t0.1,0.1,0.1\t100,150,200\tGENE_A",
                "TX2\t3\t3.0\t120.0\t1.0,1.0,1.0\t80,120,160\tGENE_A",
            ],
        )
        args = argparse.Namespace(
            input=str(in_path),
            output=str(out_path),
            format="tsv",
            gtf=None,
            gzip=False,
            min_asp=0.5,
            min_pareads=1,
        )
        with pytest.raises(SystemExit):
            main(args)

    def test_min_pareads_threshold(self, tmp_path):
        in_path = tmp_path / "in.tsv"
        out_path = tmp_path / "out.tsv"
        _make_polya_tsv(
            in_path,
            [
                "transcript_id\tn_reads\ttotal_wt\twmlen\tweights\tlengths\tgene_id",
                "TX1\t3\t3.0\t150.0\t1.0,1.0,1.0\t100,150,200\tGENE_A",
                "TX2\t2\t2.0\t120.0\t1.0,1.0\t80,160\tGENE_A",
            ],
        )
        args = argparse.Namespace(
            input=str(in_path),
            output=str(out_path),
            format="tsv",
            gtf=None,
            gzip=False,
            min_asp=0.0,
            min_pareads=5,
        )
        main(args)

        lines = out_path.read_text().strip().split("\n")
        assert len(lines) == 2  # header + 1 row, but row has NA stats
        parts = lines[1].split("\t")
        # ks_stat at index 11
        assert parts[11] == "NA"

    def test_single_transcript_gene_skipped(self, tmp_path):
        in_path = tmp_path / "in.tsv"
        out_path = tmp_path / "out.tsv"
        _make_polya_tsv(
            in_path,
            [
                "transcript_id\tn_reads\ttotal_wt\twmlen\tweights\tlengths\tgene_id",
                "TX1\t3\t3.0\t150.0\t1.0,1.0,1.0\t100,150,200\tGENE_A",
                "TX2\t3\t3.0\t120.0\t1.0,1.0,1.0\t80,120,160\tGENE_B",
            ],
        )
        args = argparse.Namespace(
            input=str(in_path),
            output=str(out_path),
            format="tsv",
            gtf=None,
            gzip=False,
            min_asp=0.0,
            min_pareads=1,
        )
        with pytest.raises(SystemExit):
            main(args)

    def test_missing_gene_id_no_gtf(self, tmp_path):
        in_path = tmp_path / "in.tsv"
        out_path = tmp_path / "out.tsv"
        _make_polya_tsv(
            in_path,
            [
                "transcript_id\tn_reads\ttotal_wt\twmlen\tweights\tlengths",
                "TX1\t3\t3.0\t150.0\t1.0,1.0,1.0\t100,150,200",
                "TX2\t3\t3.0\t120.0\t1.0,1.0,1.0\t80,120,160",
            ],
        )
        args = argparse.Namespace(
            input=str(in_path),
            output=str(out_path),
            format="tsv",
            gtf=None,
            gzip=False,
            min_asp=0.0,
            min_pareads=1,
        )
        with pytest.raises(SystemExit):
            main(args)

    def test_gzipped_output(self, tmp_path):
        in_path = tmp_path / "in.tsv"
        out_path = tmp_path / "out.tsv"
        _make_polya_tsv(
            in_path,
            [
                "transcript_id\tn_reads\ttotal_wt\twmlen\tweights\tlengths\tgene_id",
                "TX1\t3\t3.0\t150.0\t1.0,1.0,1.0\t100,150,200\tGENE_A",
                "TX2\t3\t3.0\t120.0\t1.0,1.0,1.0\t80,120,160\tGENE_A",
            ],
        )
        args = argparse.Namespace(
            input=str(in_path),
            output=str(out_path),
            format="tsv",
            gtf=None,
            gzip=True,
            min_asp=0.0,
            min_pareads=1,
        )
        main(args)
        gz_path = tmp_path / "out.tsv.gz"
        assert gz_path.exists()

    def test_bh_fdr_correction(self, tmp_path):
        in_path = tmp_path / "in.tsv"
        out_path = tmp_path / "out.tsv"
        np.random.seed(42)
        polya_lines = [
            "transcript_id\tn_reads\ttotal_wt\twmlen\tweights\tlengths\tgene_id",
        ]
        # Two genes, each with 3 transcripts → 3 + 3 = 6 pairs
        for gi, gene in enumerate(["GENE_A", "GENE_B"]):
            for ti in range(3):
                tx = f"TX{gi}{ti}"
                n = 10
                lens = np.random.normal(100 + 5 * ti, 20, n).astype(float)
                polya_lines.append(
                    f"{tx}\t{ti}\t{n}\t{lens.mean():.1f}\t"
                    f"{','.join('1.0' for _ in range(n))}\t"
                    f"{','.join(str(int(x)) for x in lens)}\t"
                    f"{gene}"
                )
        _make_polya_tsv(in_path, polya_lines)
        args = argparse.Namespace(
            input=str(in_path),
            output=str(out_path),
            format="tsv",
            gtf=None,
            gzip=False,
            min_asp=0.0,
            min_pareads=1,
        )
        main(args)

        lines = out_path.read_text().strip().split("\n")
        assert len(lines) > 1  # at least one pair
        hdr = lines[0].split("\t")
        ks_q_idx = hdr.index("ks_q_value")
        t_q_idx = hdr.index("t_q_value")
        u_q_idx = hdr.index("u_q_value")
        for line in lines[1:]:
            parts = line.split("\t")
            for qi in (ks_q_idx, t_q_idx, u_q_idx):
                val = parts[qi]
                if val != "NA":
                    assert 0.0 <= float(val) <= 1.0

    def test_na_gene_id_skipped(self, tmp_path):
        """Transcripts with gene_id='NA' are skipped."""
        in_path = tmp_path / "in.tsv"
        out_path = tmp_path / "out.tsv"
        _make_polya_tsv(
            in_path,
            [
                "transcript_id\tn_reads\ttotal_wt\twmlen\tweights\tlengths\tgene_id",
                "TX1\t3\t3.0\t150.0\t1.0,1.0,1.0\t100,150,200\tGENE_A",
                "TX2\t3\t3.0\t120.0\t1.0,1.0,1.0\t80,120,160\tGENE_A",
                "TX3\t2\t3\t100.0\t1.0,1.0,1.0\t60,100,140\tNA",
            ],
        )
        args = argparse.Namespace(
            input=str(in_path),
            output=str(out_path),
            format="tsv",
            gtf=None,
            gzip=False,
            min_asp=0.0,
            min_pareads=1,
        )
        main(args)
        lines = out_path.read_text().strip().split("\n")
        assert len(lines) == 2  # header + 1 pair (TX3 excluded)

    def test_negative_lengths_filtered(self, tmp_path):
        """Reads with negative lengths (-1) are excluded from effective count."""
        in_path = tmp_path / "in.tsv"
        out_path = tmp_path / "out.tsv"
        _make_polya_tsv(
            in_path,
            [
                "transcript_id\tn_reads\ttotal_wt\twmlen\tweights\tlengths\tgene_id",
                # TX1: 3 reads, only 1 effective (non-negative length)
                "TX1\t3\t3.0\t150.0\t1.0,1.0,1.0\t100,-1,-1\tGENE_A",
                "TX2\t3\t3.0\t120.0\t1.0,1.0,1.0\t80,120,160\tGENE_A",
            ],
        )
        args = argparse.Namespace(
            input=str(in_path),
            output=str(out_path),
            format="tsv",
            gtf=None,
            gzip=False,
            min_asp=0.0,
            min_pareads=1,
        )
        main(args)
        lines = out_path.read_text().strip().split("\n")
        assert len(lines) == 2
        parts = lines[1].split("\t")
        # n_reads_1 should be 1 (only the non-negative read)
        assert int(parts[3]) == 1
