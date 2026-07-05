#!/usr/bin/env python3
"""Estimate transcript isoform-specific poly(A) tail lengths from Oarfish
assignments and a Dorado BAM file with ``pt:i`` tags.
"""

import argparse
import sys

import pysam

try:
    from isolens._gtf import build_tx_to_gene
    from isolens._io import ensure_gz_suffix
    from isolens._parsing import (
        calc_weighted_pa_len,
        open_by_suffix,
        parse_oarfish,
        read_id_to_int,
    )
except ImportError:
    from _io import ensure_gz_suffix  # type: ignore[no-redef]

    from _gtf import build_tx_to_gene  # type: ignore[no-redef]
    from _parsing import (  # type: ignore[no-redef]
        calc_weighted_pa_len,
        open_by_suffix,
        parse_oarfish,
        read_id_to_int,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for polya_calc."""
    parser = argparse.ArgumentParser(
        description="Estimate transcript isoform-specific poly(A) length "
        "using Oarfish assignments and Dorado BAM."
    )
    parser.add_argument(
        "-a",
        "--oarfish",
        required=True,
        help="Oarfish read assignment probability file (.lz4)",
    )
    parser.add_argument(
        "-b", "--bam", required=True, help="Raw reads BAM file containing pt:i tags"
    )
    parser.add_argument("-o", "--output", required=True, help="Output TSV file")
    parser.add_argument(
        "-z",
        "--gzip",
        action="store_true",
        help="Compress the output TSV file using gzip",
    )
    parser.add_argument(
        "-g",
        "--gtf",
        default=None,
        help="GTF annotation file for adding gene_id to output",
    )
    return parser.parse_args()


def main() -> None:
    """Extract per-transcript poly(A) lengths from Dorado BAM + Oarfish.

    Reads the Oarfish assignment probability file to map reads to
    transcripts, then scans the BAM for ``pt:i`` tags (poly(A) tail
    length estimates emitted by Dorado).  Writes a TSV with per-transcript
    weighted-average poly(A) lengths.
    """
    args = parse_args()

    print(f"Parsing Oarfish assignments from {args.oarfish}...", file=sys.stderr)
    tx_names, prob_map, name_to_id = parse_oarfish(args.oarfish)
    tx_idx_to_name = dict(enumerate(tx_names))

    n_assignments = len(prob_map)
    print(
        f"Loaded {len(tx_names)} transcripts and "
        f"{n_assignments} reads with assignments.",
        file=sys.stderr,
    )

    # Optionally load transcript-to-gene mapping from GTF
    tx_to_gene: dict[str, str] | None = None
    if args.gtf is not None:
        tx_to_gene = build_tx_to_gene(args.gtf)

    if not prob_map:
        print(
            "0 reads with assignments found. Exiting early without "
            "parsing the BAM file.",
            file=sys.stderr,
        )
        sys.exit(0)

    # Initialize a dict to store poly(A) information mapped to transcripts
    tx_data: dict[int, list[tuple[float, int]]] = {
        tx_idx: [] for tx_idx in tx_idx_to_name
    }

    print(f"Processing BAM file {args.bam}...", file=sys.stderr)
    processed_reads: set[int] = set()
    reads_scanned = 0

    # Read BAM file and extract pt:i tags
    with pysam.AlignmentFile(args.bam, "rb", check_sq=False) as bam:
        for read in bam.fetch(until_eof=True):
            reads_scanned += 1

            if reads_scanned % 200000 == 0:
                print(
                    f"  ...scanned {reads_scanned} reads from BAM so far...",
                    file=sys.stderr,
                )

            read_id_int = read_id_to_int(read.query_name)

            if read_id_int in processed_reads:
                continue

            if read_id_int in prob_map:
                if read.has_tag("pt"):
                    pt_val = read.get_tag("pt")

                    if pt_val > 0:
                        processed_reads.add(read_id_int)

                        for assignment in prob_map[read_id_int]:
                            tx_data[assignment.tx_id].append((assignment.prob, pt_val))

    print(
        f"Finished BAM parsing. Scanned {reads_scanned} total reads.", file=sys.stderr
    )
    print(
        f"Successfully extracted poly(A) lengths for "
        f"{len(processed_reads)} mapped reads.",
        file=sys.stderr,
    )

    # Compute metrics and generate output TSV
    output_filename = ensure_gz_suffix(args.output, args.gzip)

    print(f"Writing results to {output_filename}...", file=sys.stderr)
    write_mode = "wt" if output_filename.endswith(".gz") else "w"
    with open_by_suffix(output_filename, write_mode) as out_f:
        if tx_to_gene is not None:
            out_f.write(
                "transcript_id\tn_reads\ttotal_wt\twmlen\tweights\tlengths\tgene_id\n"
            )
        else:
            out_f.write(
                "transcript_id\tn_reads\ttotal_wt\twmlen\tweights\tlengths\n"
            )

        for tx_idx, tx_name in tx_idx_to_name.items():
            data = tx_data.get(tx_idx, [])

            if not data:
                continue

            weights = [item[0] for item in data]
            lengths = [item[1] for item in data]

            n_reads = len(data)
            total_wt = sum(weights)
            wmlen = calc_weighted_pa_len(weights, lengths)

            weights_str = ",".join(f"{w:.5g}" for w in weights)
            lengths_str = ",".join(str(pl) for pl in lengths)

            line = (
                f"{tx_name}\t{n_reads}\t{total_wt:.3f}\t{wmlen:.3f}\t"
                f"{weights_str}\t{lengths_str}"
            )
            if tx_to_gene is not None:
                gene_id = tx_to_gene.get(tx_name, "NA")
                line += f"\t{gene_id}"
            out_f.write(line + "\n")

    print("Done!", file=sys.stderr)


if __name__ == "__main__":
    main()
