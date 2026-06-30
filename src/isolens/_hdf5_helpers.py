"""Shared HDF5 helpers for isolens modification analysis modules.

Used by ``mod_sites``, ``mod_corr``, ``mod_dmc``, and ``mod_dmt``.
"""

from __future__ import annotations

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Modification code I/O
# ---------------------------------------------------------------------------


def read_mod_codes(h5: h5py.File) -> dict[str, int]:
    """Read modification codes from an open HDF5 file.

    Returns ``{mod_type_str: code}`` dict.
    """
    return {
        mod_str: int(code) for mod_str, code in h5["modification_codes"].attrs.items()
    }


def validate_mod_codes(
    mod_maps: list[dict[str, int]],
    filenames: list[str],
) -> dict[str, int]:
    """Verify all HDF5 files have identical modification codes.

    Args:
        mod_maps: One ``{mod_str: code}`` dict per input file.
        filenames: Corresponding file paths (for error messages).

    Returns:
        Canonical modification code map from the first file.

    Raises:
        ValueError: If any file's codes differ from the first file.
    """
    reference = mod_maps[0]
    for i, code_map in enumerate(mod_maps[1:], start=1):
        if code_map != reference:
            ref_str = "; ".join(f"{k}={v}" for k, v in sorted(reference.items()))
            file_str = "; ".join(f"{k}={v}" for k, v in sorted(code_map.items()))
            raise ValueError(
                f"Modification codes in {filenames[i]} do not match "
                f"{filenames[0]}.\n"
                f"  {filenames[0]}: {ref_str}\n"
                f"  {filenames[i]}: {file_str}"
            )
    return reference


# ---------------------------------------------------------------------------
# Transcript matrix/weight loading
# ---------------------------------------------------------------------------


def load_transcript_data(
    h5: h5py.File,
    tx_name: str,
    min_asp: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load matrix and weights for one transcript from one HDF5 file.

    Args:
        h5: Open HDF5 file handle.
        tx_name: Transcript name.
        min_asp: Minimum assignment probability filter.

    Returns:
        ``(matrix, weights)`` tuple, or ``None`` if the transcript is
        absent from this file or has zero reads after filtering.
    """
    if tx_name not in h5["transcripts"]:
        return None

    grp = h5[f"transcripts/{tx_name}"]
    matrix = grp["matrix"][:]  # (n_reads, tx_length) uint8
    weights = grp["read_weights"][:]  # (n_reads,) float32

    if min_asp > 0.0:
        mask = weights >= min_asp
        if mask.sum() == 0:
            return None
        matrix = matrix[mask]
        weights = weights[mask]

    return matrix, weights


def validate_tx_lengths(
    tx_name: str,
    lengths: list[int | None],
    filenames: list[str],
) -> int:
    """Validate that a transcript has consistent length across files.

    Args:
        tx_name: Transcript name (for error messages).
        lengths: Length from each file (``None`` if absent).
        filenames: Corresponding file paths.

    Returns:
        Canonical length from the first file that contains the transcript.

    Raises:
        ValueError: If lengths differ across files.
    """
    ref_length = next(ln for ln in lengths if ln is not None)

    for i, (length, fname) in enumerate(zip(lengths, filenames)):
        if length is not None and length != ref_length:
            raise ValueError(
                f"Transcript '{tx_name}' has inconsistent lengths across "
                f"input files: {ref_length} in first file vs {length} in "
                f"{fname}. All files must use the same transcriptome "
                f"reference."
            )
    return ref_length


# ---------------------------------------------------------------------------
# Per-site read extraction (for mod_dmc and mod_dmt)
# ---------------------------------------------------------------------------


def extract_site_reads(
    matrix: np.ndarray,
    weights: np.ndarray,
    position_1b: int,
    mod_code: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract binary modified/unmodified vector and weights for a site.

    Filters out uncovered, mismatch, deletion, failed, and other-mod
    reads so that only *modified* (this mod type) and *unmodified*
    (canonical) reads remain.

    Parameters
    ----------
    matrix : ndarray of shape (n_reads, tx_length), dtype uint8
    weights : ndarray of shape (n_reads,), dtype float32 or float64
    position_1b : int
        1-based position in the transcript.
    mod_code : int
        Integer code for the focal modification type (>= 4).

    Returns
    -------
    (y, w) or None
        *y* is a float64 array of 0.0 / 1.0 for valid reads only.
        *w* is the corresponding float64 weight vector.
        Returns ``None`` when no valid reads remain.
    """
    try:
        from isolens.mod_scan import (
            CODE_DELETION,
            CODE_FAIL,
            CODE_MISMATCH,
            CODE_UNCOVERED,
        )
    except ImportError:
        from mod_scan import (  # type: ignore[no-redef]
            CODE_DELETION,
            CODE_FAIL,
            CODE_MISMATCH,
            CODE_UNCOVERED,
        )

    col = matrix[:, position_1b - 1]
    valid = (
        (col != CODE_UNCOVERED)
        & (col != CODE_MISMATCH)
        & (col != CODE_DELETION)
        & (col != CODE_FAIL)
    )
    other_mod = (col >= 4) & (col != mod_code) & (col != CODE_FAIL)
    valid = valid & (~other_mod)
    if valid.sum() == 0:
        return None
    y = (col[valid] == mod_code).astype(np.float64)
    w = weights[valid].astype(np.float64)
    return y, w


# ---------------------------------------------------------------------------
# Nullable helpers (for mod_dmc and mod_dmt)
# ---------------------------------------------------------------------------


def nullable_float(val: float) -> float | None:
    """Return None if *val* is NaN, otherwise the float value."""
    return None if np.isnan(val) else float(val)


def nullable_str(val: str | None) -> str | None:
    """Return None for a None or empty string value."""
    if val is None:
        return None
    return str(val) if val else None
