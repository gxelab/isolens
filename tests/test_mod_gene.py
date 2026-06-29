"""Tests for mod_gene — gene-level aggregation of modification sites."""

import argparse
import os
import sys
import tempfile

import pyarrow.parquet as pq
import pytest

try:
    from isolens.mod_gene import (
        _OUTPUT_COLS,
        _write_parquet,
        _write_tsv,
        aggregate_to_gene,
        main,
        read_input,
        validate_input,
    )
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from isolens.mod_gene import (  # type: ignore[no-redef]
        _OUTPUT_COLS,
        _write_parquet,
        _write_tsv,
        aggregate_to_gene,
        main,
        read_input,
        validate_input,
    )


# ---------- helpers ----------


def _make_row(**overrides):
    """Create a minimal transcript-level row with all expected columns."""
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


def _write_sites_parquet(path, rows):
    """Write rows as a Parquet site summary file (mod_sites format)."""
    import pyarrow as pa

    columns = {}
    for col in [
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
    ]:
        values = [r.get(col) for r in rows]
        if col in ("transcript_id", "mod_type", "gene_id", "chrom", "strand"):
            columns[col] = pa.array(values)
        elif col in ("position", "gpos"):
            columns[col] = pa.array(values, type=pa.int32())
        elif col.startswith("n_"):
            columns[col] = pa.array(values, type=pa.int32())
        else:
            columns[col] = pa.array(values, type=pa.float64())
    pq.write_table(pa.table(columns), path)


def _write_sites_tsv(path, rows):
    """Write rows as a TSV site summary file (mod_sites format)."""
    header = (
        "transcript_id\tposition\tmod_type\tn_modified\twt_modified"
        "\tn_unmodified\twt_unmodified\tn_canonical\twt_canonical"
        "\tn_othermod\twt_othermod\tn_mismatch\twt_mismatch"
        "\tn_deletion\twt_deletion\tn_failed\twt_failed"
        "\tmod_level\twt_mod_level"
        "\tgene_id\tchrom\tstrand\tgpos"
    )
    cols = [
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
    ]
    with open(path, "w") as f:
        f.write(header + "\n")
        for row in rows:
            f.write(
                "\t".join("NA" if row.get(c) is None else str(row[c]) for c in cols)
                + "\n"
            )


# ---------- validate_input ----------


class TestValidateInput:
    """Tests for validate_input()."""

    def test_valid_rows_pass(self):
        """Rows with gene_id and chrom present pass validation."""
        rows = [_make_row()]
        validate_input(rows)  # should not raise

    def test_empty_rows_exit(self, capsys):
        """Empty input triggers exit."""
        with pytest.raises(SystemExit) as exc:
            validate_input([])
        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert "Input file lacks necessary gene mapping" in captured.err

    def test_all_null_gene_id_exits(self, capsys):
        """All-NA gene_id triggers exit."""
        rows = [_make_row(gene_id=None)]
        with pytest.raises(SystemExit) as exc:
            validate_input(rows)
        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert "Input file lacks necessary gene mapping" in captured.err

    def test_all_null_chrom_exits(self, capsys):
        """All-NA chrom triggers exit."""
        rows = [_make_row(chrom=None)]
        with pytest.raises(SystemExit) as exc:
            validate_input(rows)
        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert "Input file lacks necessary gene mapping" in captured.err

    def test_mixed_null_and_valid_passes(self):
        """At least one row with valid gene_id and chrom passes."""
        rows = [
            _make_row(gene_id=None, chrom=None),
            _make_row(gene_id="G1", chrom="chr1"),
        ]
        validate_input(rows)  # should not raise


# ---------- aggregate_to_gene ----------


class TestAggregateToGene:
    """Tests for aggregate_to_gene()."""

    def test_single_row_preserved(self):
        """A single row produces one gene-level row."""
        rows = [_make_row()]
        result = aggregate_to_gene(rows)
        assert len(result) == 1
        r = result[0]
        assert r["gene_id"] == "G1"
        assert r["chrom"] == "chr1"
        assert r["strand"] == "+"
        assert r["gpos"] == 142
        assert r["mod_type"] == "a"
        assert r["n_modified"] == 10

    def test_two_rows_same_gene_position_summed(self):
        """Two rows with same (gene, gpos, mod_type) are summed."""
        rows = [
            _make_row(transcript_id="TX1", position=42, n_modified=5, n_unmodified=45),
            _make_row(transcript_id="TX2", position=100, n_modified=5, n_unmodified=45),
        ]
        result = aggregate_to_gene(rows)
        assert len(result) == 1
        r = result[0]
        assert r["n_modified"] == 10
        assert r["n_unmodified"] == 90

    def test_different_genes_separate(self):
        """Rows from different genes produce separate output rows."""
        rows = [
            _make_row(gene_id="G1", gpos=100),
            _make_row(gene_id="G2", gpos=100),
        ]
        result = aggregate_to_gene(rows)
        assert len(result) == 2
        gene_ids = {r["gene_id"] for r in result}
        assert gene_ids == {"G1", "G2"}

    def test_different_mod_types_separate(self):
        """Same position, different mod types → separate rows."""
        rows = [
            _make_row(mod_type="a"),
            _make_row(mod_type="m"),
        ]
        result = aggregate_to_gene(rows)
        assert len(result) == 2
        mod_types = {r["mod_type"] for r in result}
        assert mod_types == {"a", "m"}

    def test_nulls_dropped(self):
        """Rows with None gene_id or gpos are dropped."""
        rows = [
            _make_row(gene_id="G1", gpos=100),
            _make_row(gene_id=None, gpos=200),
            _make_row(gene_id="G2", gpos=None),
        ]
        result = aggregate_to_gene(rows)
        assert len(result) == 1
        assert result[0]["gene_id"] == "G1"

    def test_mod_level_recomputed(self):
        """mod_level is recomputed from summed counts, not averaged."""
        # Row 1: 1 mod, 1 unmod → mod_level = 0.5
        # Row 2: 3 mod, 1 unmod → mod_level = 0.75
        # Pooled: 4 mod, 2 unmod → mod_level = 4/6 ≈ 0.666667
        rows = [
            _make_row(n_modified=1, n_unmodified=1),
            _make_row(n_modified=3, n_unmodified=1),
        ]
        result = aggregate_to_gene(rows)
        assert len(result) == 1
        assert result[0]["n_modified"] == 4
        assert result[0]["n_unmodified"] == 2
        assert result[0]["mod_level"] == pytest.approx(4.0 / 6.0)

    def test_wt_mod_level_recomputed(self):
        """wt_mod_level is recomputed from summed weighted counts."""
        rows = [
            _make_row(wt_modified=1.0, wt_unmodified=1.0),
            _make_row(wt_modified=3.0, wt_unmodified=1.0),
        ]
        result = aggregate_to_gene(rows)
        assert result[0]["wt_modified"] == pytest.approx(4.0)
        assert result[0]["wt_unmodified"] == pytest.approx(2.0)
        assert result[0]["wt_mod_level"] == pytest.approx(4.0 / 6.0)

    def test_zero_denominator(self):
        """When n_modified + n_unmodified = 0, mod_level is 0."""
        rows = [
            _make_row(n_modified=0, n_unmodified=0, wt_modified=0.0, wt_unmodified=0.0),
        ]
        result = aggregate_to_gene(rows)
        assert result[0]["mod_level"] == 0.0
        assert result[0]["wt_mod_level"] == 0.0

    def test_all_count_columns_summed(self):
        """All count and weighted-count columns are summed."""
        rows = [
            _make_row(
                n_modified=1,
                n_unmodified=2,
                n_canonical=3,
                n_othermod=4,
                n_mismatch=5,
                n_deletion=6,
                n_failed=7,
                wt_modified=0.1,
                wt_unmodified=0.2,
                wt_canonical=0.3,
                wt_othermod=0.4,
                wt_mismatch=0.5,
                wt_deletion=0.6,
                wt_failed=0.7,
            ),
            _make_row(
                n_modified=1,
                n_unmodified=2,
                n_canonical=3,
                n_othermod=4,
                n_mismatch=5,
                n_deletion=6,
                n_failed=7,
                wt_modified=0.1,
                wt_unmodified=0.2,
                wt_canonical=0.3,
                wt_othermod=0.4,
                wt_mismatch=0.5,
                wt_deletion=0.6,
                wt_failed=0.7,
            ),
        ]
        result = aggregate_to_gene(rows)
        r = result[0]
        assert r["n_modified"] == 2
        assert r["n_unmodified"] == 4
        assert r["n_canonical"] == 6
        assert r["n_othermod"] == 8
        assert r["n_mismatch"] == 10
        assert r["n_deletion"] == 12
        assert r["n_failed"] == 14
        assert r["wt_modified"] == pytest.approx(0.2)
        assert r["wt_unmodified"] == pytest.approx(0.4)

    def test_empty_input(self):
        """Empty input produces empty output."""
        assert aggregate_to_gene([]) == []

    def test_output_sorted(self):
        """Output is sorted by (gene_id, gpos, mod_type)."""
        rows = [
            _make_row(gene_id="B", gpos=200, mod_type="m"),
            _make_row(gene_id="A", gpos=100, mod_type="a"),
            _make_row(gene_id="B", gpos=100, mod_type="a"),
        ]
        result = aggregate_to_gene(rows)
        assert len(result) == 3
        assert (result[0]["gene_id"], result[0]["gpos"]) == ("A", 100)
        assert (result[1]["gene_id"], result[1]["gpos"]) == ("B", 100)
        assert (result[2]["gene_id"], result[2]["gpos"]) == ("B", 200)


# ---------- I/O roundtrip ----------


class TestReadInput:
    """Tests for read_input() with Parquet and TSV."""

    def test_parquet_roundtrip(self, tmp_path):
        """Read Parquet → write Parquet → verify."""
        rows = [
            _make_row(),
            _make_row(
                transcript_id="TX2",
                n_modified=5,
                n_unmodified=45,
                mod_level=0.1,
            ),
        ]
        in_path = str(tmp_path / "sites.parquet")
        _write_sites_parquet(in_path, rows)

        read = read_input(in_path)
        assert len(read) == 2
        assert read[0]["gene_id"] == "G1"
        assert read[1]["transcript_id"] == "TX2"

    def test_tsv_roundtrip(self, tmp_path):
        """Read TSV → verify."""
        rows = [_make_row()]
        in_path = str(tmp_path / "sites.tsv")
        _write_sites_tsv(in_path, rows)

        read = read_input(in_path)
        assert len(read) == 1
        assert read[0]["gene_id"] == "G1"
        assert read[0]["position"] == 42
        assert read[0]["n_modified"] == 10

    def test_tsv_na_values(self, tmp_path):
        """TSV 'NA' values become None for gene_id/chrom/strand/gpos."""
        rows = [
            _make_row(
                gene_id=None,
                chrom=None,
                strand=None,
                gpos=None,
                n_modified=0,
                wt_modified=0.0,
                n_unmodified=0,
                wt_unmodified=0.0,
            )
        ]
        in_path = str(tmp_path / "sites.tsv")
        _write_sites_tsv(in_path, rows)

        read = read_input(in_path)
        assert read[0]["gene_id"] is None
        assert read[0]["chrom"] is None
        assert read[0]["strand"] is None
        assert read[0]["gpos"] is None


# ---------- output writers ----------


class TestWriteParquet:
    """Tests for _write_parquet()."""

    def test_empty_rows(self):
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tf:
            tmp_path = tf.name
        try:
            _write_parquet([], tmp_path)
            table = pq.read_table(tmp_path)
            assert len(table) == 0
            for col in _OUTPUT_COLS:
                assert col in table.column_names
        finally:
            os.unlink(tmp_path)

    def test_non_empty_rows(self):
        gene_rows = aggregate_to_gene([_make_row()])
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tf:
            tmp_path = tf.name
        try:
            _write_parquet(gene_rows, tmp_path)
            table = pq.read_table(tmp_path)
            assert len(table) == 1
            assert table.column("gene_id")[0].as_py() == "G1"
            assert table.column("gpos")[0].as_py() == 142
            assert table.column("mod_level")[0].as_py() == pytest.approx(0.1)
        finally:
            os.unlink(tmp_path)


class TestWriteTsv:
    """Tests for _write_tsv()."""

    def test_empty_rows(self, tmp_path):
        path = tmp_path / "out.tsv"
        _write_tsv([], str(path), use_gzip=False)
        content = path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 1  # header only
        assert "gene_id" in lines[0]

    def test_non_empty_rows(self, tmp_path):
        gene_rows = aggregate_to_gene([_make_row()])
        path = tmp_path / "out.tsv"
        _write_tsv(gene_rows, str(path), use_gzip=False)
        content = path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 2  # header + 1 data
        assert "G1" in lines[1]

    def test_gzip_output(self, tmp_path):
        import gzip

        gene_rows = aggregate_to_gene([_make_row()])
        path = tmp_path / "out.tsv.gz"
        _write_tsv(gene_rows, str(path), use_gzip=True)
        with gzip.open(path, "rt", encoding="utf-8") as f:
            content = f.read()
        assert "gene_id" in content


# ---------- main integration ----------


class TestMain:
    """Integration tests for main()."""

    def test_parquet_to_parquet(self, tmp_path):
        """Full pipeline: Parquet input → Parquet output."""
        in_path = str(tmp_path / "sites.parquet")
        out_path = str(tmp_path / "gene.parquet")
        _write_sites_parquet(in_path, [_make_row()])

        args = argparse.Namespace(
            input=in_path,
            output=out_path,
            format="parquet",
            gzip=False,
            verbose=False,
        )
        main(args)

        table = pq.read_table(out_path)
        assert len(table) == 1
        assert table.column("gene_id")[0].as_py() == "G1"
        assert "transcript_id" not in table.column_names
        assert "position" not in table.column_names

    def test_tsv_to_tsv(self, tmp_path):
        """Full pipeline: TSV input → TSV output."""
        in_path = str(tmp_path / "sites.tsv")
        out_path = str(tmp_path / "gene.tsv")
        _write_sites_tsv(in_path, [_make_row()])

        args = argparse.Namespace(
            input=in_path,
            output=out_path,
            format="tsv",
            gzip=False,
            verbose=False,
        )
        main(args)

        content = open(out_path).read()
        lines = content.strip().split("\n")
        assert len(lines) == 2  # header + data

    def test_missing_metadata_exits(self, tmp_path, capsys):
        """Input without gene_id triggers error exit."""
        in_path = str(tmp_path / "no_gtf.parquet")
        out_path = str(tmp_path / "gene.parquet")
        _write_sites_parquet(in_path, [_make_row(gene_id=None, chrom=None)])

        args = argparse.Namespace(
            input=in_path,
            output=out_path,
            format="parquet",
            gzip=False,
            verbose=False,
        )
        with pytest.raises(SystemExit) as exc:
            main(args)
        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert "Input file lacks necessary gene mapping" in captured.err

    def test_null_gpos_dropped(self, tmp_path):
        """Rows with gpos=None are excluded from output."""
        in_path = str(tmp_path / "sites.parquet")
        out_path = str(tmp_path / "gene.parquet")
        _write_sites_parquet(
            in_path,
            [
                _make_row(gene_id="G1", gpos=100),
                _make_row(gene_id="G1", gpos=None, position=99, n_modified=99),
            ],
        )

        args = argparse.Namespace(
            input=in_path,
            output=out_path,
            format="parquet",
            gzip=False,
            verbose=False,
        )
        main(args)

        table = pq.read_table(out_path)
        assert len(table) == 1
        assert table.column("gpos")[0].as_py() == 100


# ---------- output columns constant ----------


class TestOutputCols:
    """Verify _OUTPUT_COLS structure."""

    def test_first_five_are_key_columns(self):
        assert _OUTPUT_COLS[:5] == ["gene_id", "chrom", "strand", "gpos", "mod_type"]

    def test_last_two_are_mod_levels(self):
        assert _OUTPUT_COLS[-2:] == ["mod_level", "wt_mod_level"]

    def test_no_transcript_id_or_position(self):
        assert "transcript_id" not in _OUTPUT_COLS
        assert "position" not in _OUTPUT_COLS

    def test_total_columns(self):
        assert len(_OUTPUT_COLS) == 21
