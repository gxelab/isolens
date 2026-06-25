"""Tests for polya_calc — poly(A) length extraction from BAM + Oarfish."""

import os
import sys

import pytest

try:
    from isolens._parsing import calc_weighted_pa_len
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from isolens._parsing import (  # type: ignore[no-redef]
        calc_weighted_pa_len,
    )


class TestCalcWeightedPaLen:
    """Tests for calc_weighted_pa_len (re-imported for module coverage)."""

    def test_simple_average(self):
        result = calc_weighted_pa_len([1.0, 1.0], [100, 200])
        assert result == pytest.approx(150.0)

    def test_weighted_average(self):
        result = calc_weighted_pa_len([0.8, 0.2], [100, 200])
        expected = (0.8 * 100 + 0.2 * 200) / 1.0
        assert result == pytest.approx(expected)

    def test_zero_total_weight(self):
        result = calc_weighted_pa_len([0.0, 0.0], [100, 200])
        assert result == 0.0

    def test_empty_input(self):
        result = calc_weighted_pa_len([], [])
        assert result == 0.0


# Integration test for polya_calc would require real BAM + LZ4 input files.
# The module's main logic is covered indirectly via the example data
# integration test in test_mod_scan.py and via unit tests of the shared
# utilities (parse_oarfish, calc_weighted_pa_len, read_id_to_int).
