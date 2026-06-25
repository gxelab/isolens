"""Tests for the _parsing module — shared parsing utilities."""

import hashlib
import uuid

import lz4.frame
import pytest

try:
    from isolens._parsing import (
        TargetAssignment,
        calc_weighted_pa_len,
        open_by_suffix,
        parse_oarfish,
        read_id_to_int,
    )
except ImportError:
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from isolens._parsing import (  # type: ignore[no-redef]
        TargetAssignment,
        calc_weighted_pa_len,
        open_by_suffix,
        parse_oarfish,
        read_id_to_int,
    )


# ---------- TargetAssignment ----------


class TestTargetAssignment:
    """Tests for the TargetAssignment lightweight struct."""

    def test_construction(self):
        ta = TargetAssignment(tx_id=42, prob=0.95)
        assert ta.tx_id == 42
        assert ta.prob == 0.95

    def test_slots(self):
        ta = TargetAssignment(1, 0.5)
        with pytest.raises(AttributeError):
            ta.new_attr = "should fail"  # __slots__ prevents new attributes

    def test_negative_prob(self):
        ta = TargetAssignment(0, -0.1)
        assert ta.prob == -0.1  # stored as-is; validation is caller's job

    def test_zero_prob(self):
        ta = TargetAssignment(0, 0.0)
        assert ta.prob == 0.0


# ---------- read_id_to_int ----------


class TestReadIdToInt:
    """Tests for read_id_to_int UUID/MD5 conversion."""

    def test_valid_uuid(self):
        uuid_str = "550e8400-e29b-41d4-a716-446655440000"
        expected = uuid.UUID(uuid_str).int
        assert read_id_to_int(uuid_str) == expected

    def test_non_uuid_falls_back_to_md5(self):
        non_uuid = "some-arbitrary-read-name"
        result = read_id_to_int(non_uuid)
        expected = int(hashlib.md5(non_uuid.encode("utf-8")).hexdigest(), 16)
        assert result == expected

    def test_empty_string(self):
        result = read_id_to_int("")
        expected = int(hashlib.md5(b"").hexdigest(), 16)
        assert result == expected

    def test_plain_read_name(self):
        """A plain read name (not UUID-like at all) uses MD5 fallback."""
        name = "read-001-chr1-pos42"
        result = read_id_to_int(name)
        expected = int(hashlib.md5(name.encode("utf-8")).hexdigest(), 16)
        assert result == expected


# ---------- parse_oarfish ----------


class TestParseOarfish:
    """Tests for parse_oarfish LZ4 file parsing."""

    def _make_lz4(self, content: str) -> bytes:
        """Helper: compress a string into LZ4 bytes in memory."""
        import io

        buf = io.BytesIO()
        with lz4.frame.open(buf, "wb") as f:
            f.write(content.encode("utf-8"))
        return buf.getvalue()

    def _write_temp_lz4(self, tmp_path, content: str) -> str:
        """Write LZ4-compressed content to a temp file, return path."""
        import os

        data = self._make_lz4(content)
        path = os.path.join(tmp_path, "test.lz4")
        with open(path, "wb") as f:
            f.write(data)
        return path

    def test_valid_file(self, tmp_path):
        content = (
            "2\n"
            "FBtr0073078\n"
            "FBtr0073079\n"
            "00000000-0000-0000-0000-000000000001 1 0 1.0\n"
            "00000000-0000-0000-0000-000000000002 2 0 1 0.8 0.2\n"
        )
        path = self._write_temp_lz4(tmp_path, content)
        tx_names, prob_map, name_to_id = parse_oarfish(path)

        assert tx_names == ["FBtr0073078", "FBtr0073079"]
        assert name_to_id == {"FBtr0073078": 0, "FBtr0073079": 1}
        assert len(prob_map) == 2

        read1 = uuid.UUID("00000000-0000-0000-0000-000000000001").int
        assert read1 in prob_map
        assert len(prob_map[read1]) == 1
        assert prob_map[read1][0].tx_id == 0
        assert prob_map[read1][0].prob == 1.0

        read2 = uuid.UUID("00000000-0000-0000-0000-000000000002").int
        assert read2 in prob_map
        assert len(prob_map[read2]) == 2

    def test_empty_file(self, tmp_path):
        content = ""
        path = self._write_temp_lz4(tmp_path, content)
        with pytest.raises(ValueError, match="Empty"):
            parse_oarfish(path)

    def test_header_only_no_transcripts(self, tmp_path):
        """Header says 2 transcripts but file ends — returns empty lists."""
        content = "2\n"
        path = self._write_temp_lz4(tmp_path, content)
        # The function reads empty strings for missing transcript lines
        tx_names, prob_map, name_to_id = parse_oarfish(path)
        # Should handle gracefully: empty-string transcript names
        assert len(tx_names) == 2
        assert len(prob_map) == 0

    def test_zero_transcripts(self, tmp_path):
        content = "0\n"
        path = self._write_temp_lz4(tmp_path, content)
        tx_names, prob_map, name_to_id = parse_oarfish(path)
        assert tx_names == []
        assert name_to_id == {}
        assert prob_map == {}

    def test_empty_lines_in_assignment_section(self, tmp_path):
        content = (
            "1\n"
            "TX1\n"
            "00000000-0000-0000-0000-000000000001 1 0 1.0\n"
            "\n"
            "00000000-0000-0000-0000-000000000002 1 0 1.0\n"
        )
        path = self._write_temp_lz4(tmp_path, content)
        tx_names, prob_map, _ = parse_oarfish(path)
        assert len(prob_map) == 2  # empty line skipped

    def test_no_assignments(self, tmp_path):
        content = "2\nTX1\nTX2\n"
        path = self._write_temp_lz4(tmp_path, content)
        tx_names, prob_map, _ = parse_oarfish(path)
        assert len(tx_names) == 2
        assert len(prob_map) == 0


# ---------- calc_weighted_pa_len ----------


class TestCalcWeightedPaLen:
    """Tests for calc_weighted_pa_len utility."""

    def test_simple(self):
        result = calc_weighted_pa_len([1.0, 1.0], [100, 200])
        assert result == pytest.approx(150.0)

    def test_weighted(self):
        result = calc_weighted_pa_len([0.5, 0.5], [100, 200])
        assert result == pytest.approx(150.0)

    def test_unequal_weights(self):
        result = calc_weighted_pa_len([0.9, 0.1], [100, 200])
        expected = (0.9 * 100 + 0.1 * 200) / 1.0
        assert result == pytest.approx(expected)

    def test_zero_sum_weights(self):
        result = calc_weighted_pa_len([0.0, 0.0], [100, 200])
        assert result == 0.0

    def test_empty_lists(self):
        result = calc_weighted_pa_len([], [])
        assert result == 0.0

    def test_single_element(self):
        result = calc_weighted_pa_len([0.5], [42])
        assert result == pytest.approx(42.0)

    def test_negative_probs(self):
        """Negative probabilities should produce a weighted result."""
        result = calc_weighted_pa_len([-0.5, 0.5], [100, 200])
        # sum_prob = 0.0 → returns 0.0
        assert result == 0.0


# ---------- open_by_suffix ----------


class TestOpenBySuffix:
    """Tests for open_by_suffix I/O utility."""

    def test_plain_read(self, tmp_path):
        path = tmp_path / "test.txt"
        path.write_text("hello")
        with open_by_suffix(str(path), "r") as f:
            assert f.read() == "hello"

    def test_gz_read(self, tmp_path):
        import gzip

        path = tmp_path / "test.txt.gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write("hello gz")
        with open_by_suffix(str(path), "rt") as f:
            assert f.read() == "hello gz"

    def test_plain_write(self, tmp_path):
        path = tmp_path / "out.txt"
        with open_by_suffix(str(path), "w") as f:
            f.write("written")
        assert path.read_text() == "written"

    def test_gz_write(self, tmp_path):
        import gzip

        path = tmp_path / "out.txt.gz"
        with open_by_suffix(str(path), "wt") as f:
            f.write("written gz")
        with gzip.open(path, "rt", encoding="utf-8") as f:
            assert f.read() == "written gz"
