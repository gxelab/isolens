"""Tests for polya_dpc — condition comparison poly(A) analysis."""

import argparse
import gzip
import os
import sys

import numpy as np

try:
    from isolens._parsing import parse_polyA_file
    from isolens.polya_dpc import main
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from _parsing import parse_polyA_file  # type: ignore[no-redef]

    from isolens.polya_dpc import main  # type: ignore[no-redef]


def _make_polya_tsv(path, lines: list[str], gzip_output: bool = False):
    """Write polya TSV content to *path*, optionally gzip-compressed."""
    content = "".join(line + "\n" for line in lines)
    if gzip_output:
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(content)
    else:
        path.write_text(content)


class TestParsePolyaFile:
    """Tests for parse_polyA_file()."""

    def test_valid_tsv(self, tmp_path):
        path = tmp_path / "test.tsv"
        path.write_text(
            "transcript_id\tn_reads\ttotal_wt\twmlen\tweights\tlengths\n"
            "TX1\t2\t1.0\t100.5\t0.5,0.5\t100,101\n"
        )
        id_name, data = parse_polyA_file(str(path))
        assert id_name == "transcript_id"
        assert "TX1" in data
        assert data["TX1"]["n_reads"] == 2
        assert len(data["TX1"]["weights"]) == 2
        np.testing.assert_array_equal(data["TX1"]["lengths"], np.array([100, 101]))

    def test_gene_level_tsv(self, tmp_path):
        path = tmp_path / "gene.tsv"
        path.write_text(
            "gene_id\tn_reads\ttotal_wt\twmlen\tweights\tlengths\nGENE1\t1\t1.0\t42.0\t1.0\t42\n"
        )
        id_name, data = parse_polyA_file(str(path))
        assert id_name == "gene_id"
        assert "GENE1" in data

    def test_gzipped_tsv(self, tmp_path):
        path = tmp_path / "test.tsv.gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(
                "transcript_id\tn_reads\ttotal_wt\twmlen\tweights\tlengths\n"
                "TX1\t1\t1.0\t100.0\t1.0\t100\n"
            )
        id_name, data = parse_polyA_file(str(path))
        assert "TX1" in data

    def test_empty_file_after_header(self, tmp_path):
        path = tmp_path / "empty.tsv"
        path.write_text("transcript_id\tn_reads\ttotal_wt\twmlen\tweights\tlengths\n")
        _, data = parse_polyA_file(str(path))
        assert data == {}


class TestMainIntegration:
    """Integration tests for polya_dpc main()."""

    def test_basic_comparison(self, tmp_path):
        c1 = tmp_path / "c1.tsv"
        c2 = tmp_path / "c2.tsv"
        out = tmp_path / "out.tsv"
        _make_polya_tsv(
            c1,
            [
                "transcript_id\tn_reads\ttotal_wt\twmlen\tweights\tlengths\tgene_id",
                "TX1\t3\t3.0\t150.0\t1.0,1.0,1.0\t100,150,200\tGENE_A",
                "TX2\t2\t1.5\t300.0\t1.0,0.5\t250,350\tGENE_B",
            ],
        )
        _make_polya_tsv(
            c2,
            [
                "transcript_id\tn_reads\ttotal_wt\twmlen\tweights\tlengths\tgene_id",
                "TX1\t3\t3.0\t120.0\t1.0,1.0,1.0\t80,120,160\tGENE_A",
                "TX2\t2\t2.0\t250.0\t1.0,1.0\t200,300\tGENE_B",
            ],
        )
        args = argparse.Namespace(
            condition1=str(c1),
            condition2=str(c2),
            output=str(out),
            format="tsv",
            gzip=False,
            min_asp=0.0,
            min_pareads=1,
        )
        main(args)

        lines = out.read_text().strip().split("\n")
        assert len(lines) == 3  # header + 2 features
        hdr = lines[0].split("\t")
        assert "transcript_id" in hdr
        assert "ks_stat" in hdr
        assert "ks_p_value" in hdr
        assert "ks_q_value" in hdr
        assert "t_stat" in hdr
        assert "t_p_value" in hdr
        assert "t_q_value" in hdr
        assert "u_stat" in hdr
        assert "u_p_value" in hdr
        assert "u_q_value" in hdr

    def test_min_asp_filtering(self, tmp_path):
        c1 = tmp_path / "c1.tsv"
        c2 = tmp_path / "c2.tsv"
        out = tmp_path / "out.tsv"
        # All reads below threshold after filtering
        _make_polya_tsv(
            c1,
            [
                "transcript_id\tn_reads\ttotal_wt\twmlen\tweights\tlengths",
                "TX1\t3\t0.3\t150.0\t0.1,0.1,0.1\t100,150,200",
            ],
        )
        _make_polya_tsv(
            c2,
            [
                "transcript_id\tn_reads\ttotal_wt\twmlen\tweights\tlengths",
                "TX1\t3\t0.6\t120.0\t0.2,0.2,0.2\t80,120,160",
            ],
        )
        args = argparse.Namespace(
            condition1=str(c1),
            condition2=str(c2),
            output=str(out),
            format="tsv",
            gzip=False,
            min_asp=0.5,
            min_pareads=1,
        )
        main(args)
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 2  # header + 1 feature with NA
        parts = lines[1].split("\t")
        # n_reads should reflect filtered counts
        assert int(parts[1]) == 0 or parts[9] == "NA"  # ks_stat is NA

    def test_min_pareads_threshold(self, tmp_path):
        c1 = tmp_path / "c1.tsv"
        c2 = tmp_path / "c2.tsv"
        out = tmp_path / "out.tsv"
        _make_polya_tsv(
            c1,
            [
                "transcript_id\tn_reads\ttotal_wt\twmlen\tweights\tlengths",
                "TX1\t2\t2.0\t150.0\t1.0,1.0\t100,200",
            ],
        )
        _make_polya_tsv(
            c2,
            [
                "transcript_id\tn_reads\ttotal_wt\twmlen\tweights\tlengths",
                "TX1\t2\t2.0\t120.0\t1.0,1.0\t80,160",
            ],
        )
        args = argparse.Namespace(
            condition1=str(c1),
            condition2=str(c2),
            output=str(out),
            format="tsv",
            gzip=False,
            min_asp=0.0,
            min_pareads=10,  # threshold higher than actual
        )
        main(args)
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 2  # header + 1 NA row
        parts = lines[1].split("\t")
        assert parts[9] == "NA"  # ks_stat

    def test_no_shared_features(self, tmp_path):
        c1 = tmp_path / "c1.tsv"
        c2 = tmp_path / "c2.tsv"
        out = tmp_path / "out.tsv"
        _make_polya_tsv(
            c1,
            [
                "transcript_id\tn_reads\ttotal_wt\twmlen\tweights\tlengths",
                "TX1\t2\t2.0\t150.0\t1.0,1.0\t100,200",
            ],
        )
        _make_polya_tsv(
            c2,
            [
                "transcript_id\tn_reads\ttotal_wt\twmlen\tweights\tlengths",
                "TX2\t1\t1.0\t300.0\t1.0\t300",
            ],
        )
        args = argparse.Namespace(
            condition1=str(c1),
            condition2=str(c2),
            output=str(out),
            format="tsv",
            gzip=False,
            min_asp=0.0,
            min_pareads=1,
        )
        main(args)
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 1  # header only

    def test_gzipped_output(self, tmp_path):
        c1 = tmp_path / "c1.tsv"
        c2 = tmp_path / "c2.tsv"
        out = tmp_path / "out.tsv"
        _make_polya_tsv(
            c1,
            [
                "transcript_id\tn_reads\ttotal_wt\twmlen\tweights\tlengths",
                "TX1\t3\t3.0\t150.0\t1.0,1.0,1.0\t100,150,200",
            ],
        )
        _make_polya_tsv(
            c2,
            [
                "transcript_id\tn_reads\ttotal_wt\twmlen\tweights\tlengths",
                "TX1\t3\t3.0\t120.0\t1.0,1.0,1.0\t80,120,160",
            ],
        )
        args = argparse.Namespace(
            condition1=str(c1),
            condition2=str(c2),
            output=str(out),
            format="tsv",
            gzip=True,
            min_asp=0.0,
            min_pareads=1,
        )
        main(args)
        # gzip appends .gz if not present
        gz_path = tmp_path / "out.tsv.gz"
        assert gz_path.exists()

    def test_bh_fdr_correction(self, tmp_path):
        c1 = tmp_path / "c1.tsv"
        c2 = tmp_path / "c2.tsv"
        out = tmp_path / "out.tsv"
        # Multiple features so BH FDR is non-trivial
        lines_c1 = [
            "transcript_id\tn_reads\ttotal_wt\twmlen\tweights\tlengths",
        ]
        lines_c2 = [
            "transcript_id\tn_reads\ttotal_wt\twmlen\tweights\tlengths",
        ]
        np.random.seed(42)
        for i in range(5):
            n = 10
            l1 = np.random.normal(100, 20, n).astype(float)
            l2 = np.random.normal(105, 20, n).astype(float)
            lines_c1.append(
                f"TX{i}\t{n}\t{float(n):.1f}\t{l1.mean():.1f}\t"
                f"{','.join('1.0' for _ in range(n))}\t"
                f"{','.join(str(int(x)) for x in l1)}"
            )
            lines_c2.append(
                f"TX{i}\t{i}\t{n}\t{l2.mean():.1f}\t"
                f"{','.join('1.0' for _ in range(n))}\t"
                f"{','.join(str(int(x)) for x in l2)}"
            )
        _make_polya_tsv(c1, lines_c1)
        _make_polya_tsv(c2, lines_c2)
        args = argparse.Namespace(
            condition1=str(c1),
            condition2=str(c2),
            output=str(out),
            format="tsv",
            gzip=False,
            min_asp=0.0,
            min_pareads=1,
        )
        main(args)

        lines = out.read_text().strip().split("\n")
        hdr = lines[0].split("\t")
        # Find q-value columns
        ks_q_idx = hdr.index("ks_q_value")
        t_q_idx = hdr.index("t_q_value")
        u_q_idx = hdr.index("u_q_value")
        for line in lines[1:]:
            parts = line.split("\t")
            # Check that q-values are in [0, 1] or "NA"
            for qi in (ks_q_idx, t_q_idx, u_q_idx):
                val = parts[qi]
                if val != "NA":
                    assert 0.0 <= float(val) <= 1.0

    def test_negative_lengths_filtered(self, tmp_path):
        """Reads with negative lengths are filtered out."""
        c1 = tmp_path / "c1.tsv"
        c2 = tmp_path / "c2.tsv"
        out = tmp_path / "out.tsv"
        # TX1: 3 reads, 1 effective (others negative)
        _make_polya_tsv(
            c1,
            [
                "transcript_id\tn_reads\ttotal_wt\twmlen\tweights\tlengths",
                "TX1\t3\t3.0\t150.0\t1.0,1.0,1.0\t100,-1,-1",
            ],
        )
        _make_polya_tsv(
            c2,
            [
                "transcript_id\tn_reads\ttotal_wt\twmlen\tweights\tlengths",
                "TX1\t3\t3.0\t120.0\t1.0,1.0,1.0\t80,-1,-1",
            ],
        )
        args = argparse.Namespace(
            condition1=str(c1),
            condition2=str(c2),
            output=str(out),
            format="tsv",
            gzip=False,
            min_asp=0.0,
            min_pareads=1,
        )
        main(args)
        lines = out.read_text().strip().split("\n")
        parts = lines[1].split("\t")
        # Each side should have 1 effective read (non-negative)
        assert int(parts[1]) == 1  # n_reads_1
        assert int(parts[5]) == 1  # n_reads_2
