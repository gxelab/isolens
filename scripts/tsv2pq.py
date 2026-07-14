#!/usr/bin/env python3
"""Convert a TSV file (optionally gzip-compressed) to Parquet format.

Reads a tab-separated file and writes a Parquet file with auto-inferred
column types.  Gzip-compressed input (``.tsv.gz``) is detected and
decompressed automatically.  The converter is schema-agnostic: it works
with any well-formed TSV with a header row.

Usage:
    python tsv2pq.py -i input.tsv -o output.parquet
    python tsv2pq.py -i input.tsv.gz -o output.parquet
"""

import argparse
import gzip
import os
import sys

import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.parquet as pq


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert TSV (or TSV.GZ) to Parquet"
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input TSV file (.tsv or .tsv.gz)",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output Parquet file path",
    )
    return parser.parse_args()


def _max_line_length(path: str) -> int:
    """Return the length (in bytes) of the longest line in *path*.

    Handles gzip-compressed files (detected by ``.gz`` suffix).
    """
    open_fn = gzip.open if path.endswith(".gz") else open
    max_len = 0
    with open_fn(path, "rb") as fh:
        for line in fh:
            if len(line) > max_len:
                max_len = len(line)
    return max_len


def _read_csv_with_retry(path: str, parse_options: csv.ParseOptions) -> pa.Table:
    """Read *path* as CSV, retrying with a larger block size if a field
    straddles PyArrow's default block boundary.
    """
    try:
        return csv.read_csv(path, parse_options=parse_options)
    except pa.lib.ArrowInvalid as exc:
        if "straddling object" not in str(exc):
            raise
        # Find the widest row so we can size the block to fit it.
        max_line = _max_line_length(path)
        block_size = max(64 * 1024 * 1024, max_line * 2)
        print(
            f"Retrying with block_size={block_size // (1024 * 1024)} MiB "
            f"(max line length={max_line} bytes)",
            file=sys.stderr,
        )
        read_options = csv.ReadOptions(block_size=block_size)
        return csv.read_csv(path, parse_options=parse_options,
                            read_options=read_options)


def main():
    args = parse_args()

    # Reject empty files early with a clear message
    if os.path.getsize(args.input) == 0:
        print(f"Error: input file '{args.input}' is empty.", file=sys.stderr)
        sys.exit(1)

    parse_options = csv.ParseOptions(delimiter="\t")
    table = _read_csv_with_retry(args.input, parse_options)

    # Header-only TSV: pyarrow infers every column as null type.
    # Promote null columns to string so ParquetWriter doesn't choke.
    if table.num_rows == 0 and any(
        pa.types.is_null(f.type) for f in table.schema
    ):
        fields = [
            pa.field(f.name, pa.string()) if pa.types.is_null(f.type) else f
            for f in table.schema
        ]
        schema = pa.schema(fields)
        with pq.ParquetWriter(args.output, schema) as writer:
            writer.write_table(
                pa.table(
                    {f.name: pa.array([], type=f.type) for f in fields}
                )
            )
    else:
        pq.write_table(table, args.output)

    print(
        f"Converted {args.input} → {args.output} "
        f"({table.num_rows} rows, {table.num_columns} columns)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
