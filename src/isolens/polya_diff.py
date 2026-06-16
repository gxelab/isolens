#!/usr/bin/env python3
"""Genome-wide comparison of poly(A) length distributions between two
conditions using a weighted two-sample Kolmogorov-Smirnov test.
"""

import argparse
import gzip
import sys

import numpy as np
from scipy.stats import kstwobign


def parse_args():
    parser = argparse.ArgumentParser(
        description="Genome-wide statistical comparison of poly(A) length "
                    "distributions using a weighted KS test."
    )
    parser.add_argument(
        "-c1", "--condition1", required=True,
        help="Condition 1 TSV/TSV.GZ file")
    parser.add_argument(
        "-c2", "--condition2", required=True,
        help="Condition 2 TSV/TSV.GZ file")
    parser.add_argument(
        "-o", "--output", required=True,
        help="Output TSV results file")
    parser.add_argument(
        "-z", "--gzip", action="store_true",
        help="Compress the output TSV file using gzip")
    return parser.parse_args()


def get_open_func(filename):
    if filename.endswith(".gz"):
        return lambda f: gzip.open(f, "rt", encoding="utf-8")
    return lambda f: open(f, encoding="utf-8")


def parse_polyA_file(filename):
    """Parse a poly(A) TSV file and return (id_column_name, data_dict)."""
    print(f"Loading data from {filename}...", file=sys.stderr)
    data_dict = {}
    open_func = get_open_func(filename)

    with open_func(filename) as f:
        header = f.readline().strip().split("\t")

        # Detect whether transcript-level or gene-level output
        id_col_name = "tx_name" if "tx_name" in header else "gene_id"
        id_col = header.index(id_col_name)
        probs_col = header.index("probs")
        lens_col = header.index("pa_lens")

        for line in f:
            parts = line.strip().split("\t")
            if len(parts) <= max(probs_col, lens_col):
                continue

            feature_id = parts[id_col]
            probs = np.array([float(p) for p in parts[probs_col].split(",")])
            pa_lens = np.array([int(pa_len) for pa_len in parts[lens_col].split(",")])

            n_reads = len(probs)
            sum_prob = np.sum(probs)
            pa_wlen = (np.sum(probs * pa_lens) / sum_prob
                       if sum_prob > 0 else 0.0)

            data_dict[feature_id] = {
                "n_reads": n_reads,
                "pa_wlen": pa_wlen,
                "probs": probs,
                "pa_lens": pa_lens,
            }

    return id_col_name, data_dict


def weighted_ecdf(values, weights):
    """Compute the weighted Empirical Cumulative Distribution Function."""
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]

    cum_weights = np.cumsum(weights)
    cdf = cum_weights / cum_weights[-1]
    return values, cdf


def weighted_ks_test(v1, w1, v2, w2):
    """Two-sample weighted KS test using Kish's effective sample sizes."""
    all_vals = np.unique(np.concatenate([v1, v2]))

    _, cdf1 = weighted_ecdf(v1, w1)
    _, cdf2 = weighted_ecdf(v2, w2)

    cdf1_interp = np.interp(all_vals, v1, cdf1, left=0, right=1)
    cdf2_interp = np.interp(all_vals, v2, cdf2, left=0, right=1)

    ks_stat = np.max(np.abs(cdf1_interp - cdf2_interp))

    n1_eff = (np.sum(w1) ** 2) / np.sum(w1 ** 2)
    n2_eff = (np.sum(w2) ** 2) / np.sum(w2 ** 2)

    en = np.sqrt((n1_eff * n2_eff) / (n1_eff + n2_eff))
    p_val = kstwobign.sf(ks_stat * (en + 0.12 + 0.11 / en))

    return ks_stat, min(1.0, max(0.0, p_val))


def main():
    args = parse_args()

    id_name_1, cond1_data = parse_polyA_file(args.condition1)
    id_name_2, cond2_data = parse_polyA_file(args.condition2)

    id_col_header = id_name_1 if id_name_1 == id_name_2 else "feature_id"

    all_features = sorted(set(cond1_data.keys()) | set(cond2_data.keys()))
    print(f"Comparing {len(all_features)} total features genome-wide...",
          file=sys.stderr)

    output_filename = args.output
    if args.gzip:
        if not output_filename.endswith(".gz"):
            output_filename += ".gz"
        def open_output(f):
            return gzip.open(f, "wt", encoding="utf-8")
    else:
        def open_output(f):
            return open(f, "w", encoding="utf-8")

    print(f"Writing statistical test matrix to {output_filename}...",
          file=sys.stderr)

    with open_output(output_filename) as out_f:
        out_f.write(
            f"{id_col_header}\tn_reads_1\tpa_wlen_1\tn_reads_2\t"
            f"pa_wlen_2\tstat\tp_value\n")

        for feat_id in all_features:
            in_c1 = feat_id in cond1_data
            in_c2 = feat_id in cond2_data

            n1 = cond1_data[feat_id]["n_reads"] if in_c1 else 0
            wlen1 = (f"{cond1_data[feat_id]['pa_wlen']:.2f}"
                     if in_c1 else "0.0")

            n2 = cond2_data[feat_id]["n_reads"] if in_c2 else 0
            wlen2 = (f"{cond2_data[feat_id]['pa_wlen']:.2f}"
                     if in_c2 else "0.0")

            if not in_c1 or not in_c2:
                stat_str = "NA"
                p_str = "NA"
            else:
                f1 = cond1_data[feat_id]
                f2 = cond2_data[feat_id]

                try:
                    stat, p_val = weighted_ks_test(
                        f1["pa_lens"], f1["probs"],
                        f2["pa_lens"], f2["probs"])
                    stat_str = f"{stat:.5f}"
                    p_str = f"{p_val:.5e}"
                except Exception:
                    stat_str = "NA"
                    p_str = "NA"

            out_f.write(
                f"{feat_id}\t{n1}\t{wlen1}\t{n2}\t{wlen2}\t"
                f"{stat_str}\t{p_str}\n")

    print("Genome-wide testing pipeline complete!", file=sys.stderr)


if __name__ == "__main__":
    main()
