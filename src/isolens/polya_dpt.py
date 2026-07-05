#!/usr/bin/env python3
"""Pairwise differential poly(A) length analysis between transcript
isoforms of the same gene.
"""

import argparse
import itertools
import sys

import numpy as np

try:
    from isolens._gtf import build_tx_to_gene
    from isolens._io import ensure_gz_suffix, format_float
    from isolens._parsing import open_by_suffix
    from isolens._stats import (
        bh_fdr,
        weighted_ks_test,
        weighted_median,
        weighted_rank_sum_test,
        weighted_t_test,
    )
except ImportError:
    from _io import ensure_gz_suffix, format_float  # type: ignore[no-redef]

    from _gtf import build_tx_to_gene  # type: ignore[no-redef]
    from _parsing import open_by_suffix  # type: ignore[no-redef]
    from _stats import (  # type: ignore[no-redef]
        bh_fdr,
        weighted_ks_test,
        weighted_median,
        weighted_rank_sum_test,
        weighted_t_test,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for polya_dpt."""
    parser = argparse.ArgumentParser(
        description="Pairwise differential poly(A) length analysis "
        "between transcript isoforms of the same gene."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Transcript-level poly(A) TSV file (gzipped or raw)",
    )
    parser.add_argument(
        "-g",
        "--gtf",
        default=None,
        help="GTF annotation file for transcript-to-gene mapping "
        "(gzipped or raw). Required if the input file does not "
        "already contain a gene_id column.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output pairwise TSV results file",
    )
    parser.add_argument(
        "-z",
        "--gzip",
        action="store_true",
        help="Compress the output TSV file using gzip",
    )
    parser.add_argument(
        "-p",
        "--min-asp",
        type=float,
        default=0.0,
        help="Minimum assignment probability threshold "
        "(default: 0.0, i.e. no filtering)",
    )
    parser.add_argument(
        "-n",
        "--min-pareads",
        type=int,
        default=5,
        help="Minimum number of reads with effective (non-negative) "
        "poly(A) length estimation (default: 5)",
    )
    return parser.parse_args()


def _load_and_group(
    filename: str,
    gtf_path: str | None,
    min_asp: float,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, list[str]],
]:
    """Parse transcript-level poly(A) data and group transcripts by gene.

    Returns:
        ``(tx_data, gene_groups)`` where *tx_data* maps transcript IDs
        to ``{weights, lengths}`` and *gene_groups* maps gene IDs to
        lists of transcript IDs.  Only genes with ≥ 2 transcripts
        after filtering are included.
    """
    # Detect gene_id source
    tx_to_gene: dict[str, str] | None = None
    has_gene_id_col = False

    read_mode = "rt" if filename.endswith(".gz") else "r"
    print(f"Loading data from {filename}...", file=sys.stderr)

    with open_by_suffix(filename, read_mode) as f:
        header = f.readline().strip().split("\t")

        if "transcript_id" not in header:
            print(
                "Error: Input file must contain 'transcript_id' column.  "
                "polya_dpt requires transcript-level data.",
                file=sys.stderr,
            )
            sys.exit(1)

        if "weights" not in header or "lengths" not in header:
            print(
                "Error: Input file must contain 'weights' and 'lengths' columns.",
                file=sys.stderr,
            )
            sys.exit(1)

        tx_col = header.index("transcript_id")
        weights_col = header.index("weights")
        lengths_col = header.index("lengths")

        has_gene_id_col = "gene_id" in header
        gene_id_col = header.index("gene_id") if has_gene_id_col else -1

        if has_gene_id_col:
            print("Using gene_id column from input file.", file=sys.stderr)
        elif gtf_path is not None:
            print(
                "No gene_id column in input — using GTF annotation.",
                file=sys.stderr,
            )
            tx_to_gene = build_tx_to_gene(gtf_path)
        else:
            print(
                "Error: Input file does not contain a 'gene_id' column "
                "and no --gtf annotation was provided.  "
                "Either run polya_calc with --gtf, or provide --gtf "
                "to polya_dpt.",
                file=sys.stderr,
            )
            sys.exit(1)

        # Parse and group
        tx_data: dict[str, dict[str, np.ndarray]] = {}
        gene_groups: dict[str, list[str]] = {}

        unmapped: set[str] = set()

        for line in f:
            parts = line.strip().split("\t")
            if len(parts) <= max(weights_col, lengths_col):
                continue

            tx_id = parts[tx_col]

            # Resolve gene_id
            if has_gene_id_col:
                if len(parts) <= gene_id_col:
                    unmapped.add(tx_id)
                    continue
                gene_id = parts[gene_id_col]
                if gene_id in ("", "NA", "."):
                    unmapped.add(tx_id)
                    continue
            else:
                gene_id = tx_to_gene.get(tx_id, "")  # type: ignore[union-attr]
                if not gene_id:
                    unmapped.add(tx_id)
                    continue

            weights = np.array([float(p) for p in parts[weights_col].split(",")])
            lengths = np.array([int(pl) for pl in parts[lengths_col].split(",")])

            # Apply min_asp filter and non-negative length filter
            mask = (weights >= min_asp) & (lengths >= 0)
            if not np.any(mask):
                continue

            tx_data[tx_id] = {
                "weights": weights[mask],
                "lengths": lengths[mask],
            }
            gene_groups.setdefault(gene_id, []).append(tx_id)

    if unmapped:
        print(
            f"Warning: Ignored {len(unmapped)} transcripts "
            "that could not be mapped to a gene.",
            file=sys.stderr,
        )

    # Remove genes with fewer than 2 transcripts
    gene_groups = {g: txs for g, txs in gene_groups.items() if len(txs) >= 2}

    return tx_data, gene_groups


def main(args: argparse.Namespace | None = None) -> None:
    """Compare poly(A) length distributions between isoform pairs within genes.

    Reads a transcript-level poly(A) TSV file, groups transcripts by
    gene, performs all pairwise weighted KS, t-test, and rank-sum tests,
    applies global BH FDR correction per test type, and writes a
    comparison table.
    """
    if args is None:
        args = parse_args()

    tx_data, gene_groups = _load_and_group(args.input, args.gtf, args.min_asp)

    if not gene_groups:
        print(
            "No genes with ≥ 2 transcripts found after filtering.",
            file=sys.stderr,
        )
        sys.exit(0)

    total_pairs = sum(
        len(list(itertools.combinations(txs, 2))) for txs in gene_groups.values()
    )
    print(
        f"Processing {len(gene_groups)} genes ({total_pairs} transcript pairs)...",
        file=sys.stderr,
    )

    results: list[dict] = []

    for gene_id in sorted(gene_groups.keys()):
        transcripts = gene_groups[gene_id]

        for tx_a, tx_b in itertools.combinations(transcripts, 2):
            d_a = tx_data[tx_a]
            d_b = tx_data[tx_b]

            eff_a = len(d_a["weights"])
            eff_b = len(d_b["weights"])

            row: dict = {
                "gene_id": gene_id,
                "transcript_1": tx_a,
                "transcript_2": tx_b,
                "n_reads_1": eff_a,
                "total_wt_1": float("nan"),
                "wmlen_1": float("nan"),
                "wmedlen_1": float("nan"),
                "n_reads_2": eff_b,
                "total_wt_2": float("nan"),
                "wmlen_2": float("nan"),
                "wmedlen_2": float("nan"),
                "ks_stat": float("nan"),
                "ks_p_value": float("nan"),
                "ks_q_value": float("nan"),
                "wmlen_diff": float("nan"),
                "t_stat": float("nan"),
                "t_p_value": float("nan"),
                "t_q_value": float("nan"),
                "wmedlen_diff": float("nan"),
                "u_stat": float("nan"),
                "u_p_value": float("nan"),
                "u_q_value": float("nan"),
            }

            if eff_a >= args.min_pareads and eff_b >= args.min_pareads:
                p_a = d_a["weights"]
                l_a = d_a["lengths"]
                p_b = d_b["weights"]
                l_b = d_b["lengths"]

                row["total_wt_1"] = float(p_a.sum())
                row["total_wt_2"] = float(p_b.sum())

                row["wmlen_1"] = (
                    float(np.average(l_a, weights=p_a))
                    if p_a.sum() > 0
                    else float("nan")
                )
                row["wmlen_2"] = (
                    float(np.average(l_b, weights=p_b))
                    if p_b.sum() > 0
                    else float("nan")
                )

                row["wmedlen_1"] = weighted_median(l_a, p_a)
                row["wmedlen_2"] = weighted_median(l_b, p_b)

                if not np.isnan(row["wmlen_1"]) and not np.isnan(row["wmlen_2"]):
                    row["wmlen_diff"] = row["wmlen_1"] - row["wmlen_2"]
                if not np.isnan(row["wmedlen_1"]) and not np.isnan(
                    row["wmedlen_2"]
                ):
                    row["wmedlen_diff"] = row["wmedlen_1"] - row["wmedlen_2"]

                ks_stat, ks_p = weighted_ks_test(l_a, p_a, l_b, p_b)
                row["ks_stat"] = ks_stat
                row["ks_p_value"] = ks_p

                t_stat, t_p = weighted_t_test(l_a, p_a, l_b, p_b)
                row["t_stat"] = t_stat
                row["t_p_value"] = t_p

                u_stat, u_p = weighted_rank_sum_test(l_a, p_a, l_b, p_b)
                row["u_stat"] = u_stat
                row["u_p_value"] = u_p

            results.append(row)

    # Global BH FDR correction per test type
    for test_key in ("ks", "t", "u"):
        p_key = f"{test_key}_p_value"
        q_key = f"{test_key}_q_value"
        valid_indices = [
            i for i, r in enumerate(results) if not np.isnan(r.get(p_key, float("nan")))
        ]
        if valid_indices:
            p_vals = [results[i][p_key] for i in valid_indices]
            q_vals = bh_fdr(p_vals)
            for i, qv in zip(valid_indices, q_vals):
                results[i][q_key] = round(qv, 6)

    # Write output
    output_filename = ensure_gz_suffix(args.output, args.gzip)

    print(
        f"Writing pairwise comparison results to {output_filename}...",
        file=sys.stderr,
    )

    write_mode = "wt" if output_filename.endswith(".gz") else "w"
    with open_by_suffix(output_filename, write_mode) as out_f:
        out_f.write(
            "gene_id\ttranscript_1\ttranscript_2\t"
            "n_reads_1\ttotal_wt_1\twmlen_1\twmedlen_1\t"
            "n_reads_2\ttotal_wt_2\twmlen_2\twmedlen_2\t"
            "ks_stat\tks_p_value\tks_q_value\t"
            "wmlen_diff\tt_stat\tt_p_value\tt_q_value\t"
            "wmedlen_diff\tu_stat\tu_p_value\tu_q_value\n"
        )

        for row in results:
            gene = row["gene_id"]
            tx1 = row["transcript_1"]
            tx2 = row["transcript_2"]
            n1 = row["n_reads_1"]
            n2 = row["n_reads_2"]

            twt1 = format_float(row["total_wt_1"], ".2f")
            twt2 = format_float(row["total_wt_2"], ".2f")
            wlen1 = format_float(row["wmlen_1"], ".2f")
            wlen2 = format_float(row["wmlen_2"], ".2f")
            wmed1 = format_float(row["wmedlen_1"], ".2f")
            wmed2 = format_float(row["wmedlen_2"], ".2f")

            ks_s = format_float(row["ks_stat"], ".5f")
            ks_p = format_float(row["ks_p_value"], ".5e")
            ks_q = format_float(row["ks_q_value"], ".6f")

            wl_diff = format_float(row["wmlen_diff"], ".2f")
            t_s = format_float(row["t_stat"], ".5f")
            t_p = format_float(row["t_p_value"], ".5e")
            t_q = format_float(row["t_q_value"], ".6f")

            wm_diff = format_float(row["wmedlen_diff"], ".2f")
            u_s = format_float(row["u_stat"], ".5f")
            u_p = format_float(row["u_p_value"], ".5e")
            u_q = format_float(row["u_q_value"], ".6f")

            out_f.write(
                f"{gene}\t{tx1}\t{tx2}\t"
                f"{n1}\t{twt1}\t{wlen1}\t{wmed1}\t"
                f"{n2}\t{twt2}\t{wlen2}\t{wmed2}\t"
                f"{ks_s}\t{ks_p}\t{ks_q}\t"
                f"{wl_diff}\t{t_s}\t{t_p}\t{t_q}\t"
                f"{wm_diff}\t{u_s}\t{u_p}\t{u_q}\n"
            )

    print("Pairwise isoform comparison complete!", file=sys.stderr)


if __name__ == "__main__":
    main()
