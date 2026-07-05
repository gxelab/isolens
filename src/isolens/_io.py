"""Shared I/O utilities for isolens modules.

Used by ``mod_sites``, ``mod_corr``, ``mod_dmc``, ``mod_dmcg``,
``mod_dmt``, ``mod_gene``, and the ``polya_*`` modules.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def ensure_gz_suffix(path: str, use_gzip: bool) -> str:
    """Append ``.gz`` suffix if *use_gzip* is true and not already present.

    Args:
        path: Original output file path.
        use_gzip: Whether gzip compression is requested.

    Returns:
        Path with ``.gz`` appended if needed.
    """
    if use_gzip and not path.endswith(".gz"):
        path += ".gz"
    return path


# ---------------------------------------------------------------------------
# TSV writer
# ---------------------------------------------------------------------------


def write_tsv(
    all_rows: list[dict[str, Any]],
    path: str,
    header: str,
    columns: list[str],
    use_gzip: bool = False,
) -> None:
    """Write rows as tab-separated values, optionally gzip-compressed.

    ``None`` values and ``NaN`` floats are written as ``"NA"``.  Non-zero
    float values below ``1e-6`` are written in scientific notation to
    preserve significant digits.  The *path* is used as-is (callers should
    apply :func:`ensure_gz_suffix` beforehand if needed).

    Args:
        all_rows: List of row dicts.
        path: Output file path.
        header: Tab-separated header line (e.g. ``"col1\\tcol2"``).
        columns: Ordered column names defining the output field order.
        use_gzip: If ``True``, compress with gzip.
    """
    import gzip

    import numpy as np

    def _fmt(v: Any) -> str:
        if v is None:
            return "NA"
        if isinstance(v, list):
            return ",".join(_fmt(x) for x in v)
        if isinstance(v, float):
            if np.isnan(v):
                return "NA"
            if 0.0 < abs(v) < 5e-7:
                return f"{v:.6e}"
        return str(v)

    open_func = gzip.open if use_gzip else open
    mode = "wt" if use_gzip else "w"

    with open_func(path, mode, encoding="utf-8") as f:
        f.write(header + "\n")
        for row in all_rows:
            f.write("\t".join(_fmt(row[c]) for c in columns) + "\n")


# ---------------------------------------------------------------------------
# Parquet writer (schema-based, avoids re-encoding types per column)
# ---------------------------------------------------------------------------


def write_parquet(
    all_rows: list[dict[str, Any]],
    path: str,
    schema: pa.Schema,
    columns: list[str],
) -> None:
    """Write rows as a Parquet file via pyarrow.

    When *all_rows* is empty, writes a schema-only file so downstream
    tools can still open it without errors.  The *schema* is the single
    source of truth for column types (both empty and non-empty paths).

    Args:
        all_rows: List of row dicts.
        path: Output file path.
        schema: PyArrow schema with field names matching dict keys.
        columns: Ordered column names defining the output field order
            (used for the non-empty path).
    """
    if not all_rows:
        with pq.ParquetWriter(path, schema) as writer:
            writer.write_table(
                pa.table(
                    {k: pa.array([], type=schema.field(k).type) for k in schema.names}
                )
            )
        return

    arrays: dict[str, pa.Array] = {}
    for col in columns:
        values = [r[col] for r in all_rows]
        pa_type = schema.field(col).type
        arrays[col] = pa.array(values, type=pa_type)
    pq.write_table(pa.table(arrays), path)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_float(v: float, fmt: str) -> str:
    """Format a float value, returning ``"NA"`` for NaN.

    When *fmt* is a fixed-point specifier (ends with ``"f"``) and the
    absolute value is non-zero but below ``1e-6``, the format
    automatically switches to scientific notation with the same
    precision width to avoid losing all significant digits.

    Args:
        v: Float value (may be NaN).
        fmt: Format specifier (e.g. ``".2f"``, ``".5e"``).

    Returns:
        Formatted string or ``"NA"``.
    """
    import numpy as np

    if np.isnan(v):
        return "NA"
    if fmt.endswith("f") and 0.0 < abs(v) < 1e-6:
        width = fmt[:-1]  # e.g. ".6" from ".6f"
        return f"{v:{width}e}"
    return f"{v:{fmt}}"
