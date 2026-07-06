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


def main():
    args = parse_args()

    # Reject empty files early with a clear message
    if os.path.getsize(args.input) == 0:
        print(f"Error: input file '{args.input}' is empty.", file=sys.stderr)
        sys.exit(1)

    parse_options = csv.ParseOptions(delimiter="\t")
    table = csv.read_csv(args.input, parse_options=parse_options)

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
