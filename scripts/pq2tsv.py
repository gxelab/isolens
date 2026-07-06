#!/usr/bin/env python3
"""Convert a Parquet file to TSV format, optionally gzip-compressed.

Reads a Parquet file and writes a tab-separated file with column headers.
Null values and NaN floats are written as ``"NA"``, matching the
conventions used by isolens modules.

Usage:
    python pq2tsv.py -i input.parquet -o output.tsv
    python pq2tsv.py -i input.parquet -o output.tsv.gz -z
"""

import argparse
import gzip
import sys

import pyarrow.parquet as pq


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Parquet to TSV"
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input Parquet file path",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output TSV file path",
    )
    parser.add_argument(
        "-z", "--gzip",
        action="store_true",
        help="Gzip-compress the output",
    )
    return parser.parse_args()


def _fmt(v):
    """Format a single value for TSV output, matching ``_io.write_tsv``."""
    import numpy as np

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


def main():
    args = parse_args()

    table = pq.read_table(args.input)
    columns = table.column_names
    header = "\t".join(columns)
    rows = table.to_pylist()

    open_func = gzip.open if args.gzip else open
    mode = "wt" if args.gzip else "w"

    with open_func(args.output, mode, encoding="utf-8") as f:
        f.write(header + "\n")
        for row in rows:
            f.write("\t".join(_fmt(row[c]) for c in columns) + "\n")

    print(
        f"Converted {args.input} → {args.output} "
        f"({len(rows)} rows, {len(columns)} columns)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
