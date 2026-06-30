#!/usr/bin/env python3
"""Genome-wide comparison of poly(A) length distributions between two
conditions using weighted two-sample tests (KS, t-test, rank-sum).
"""

import argparse
import sys

import numpy as np

try:
    from isolens._io import ensure_gz_suffix, format_float
    from isolens._parsing import open_by_suffix, parse_polyA_file
    from isolens._stats import (
        bh_fdr,
        weighted_ks_test,
        weighted_rank_sum_test,
        weighted_t_test,
    )
except ImportError:
    from _io import ensure_gz_suffix, format_float  # type: ignore[no-redef]

    from _parsing import (  # type: ignore[no-redef]
        open_by_suffix,
        parse_polyA_file,
    )
    from _stats import (  # type: ignore[no-redef]
        bh_fdr,
        weighted_ks_test,
        weighted_rank_sum_test,
        weighted_t_test,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for polya_dpc."""
    parser = argparse.ArgumentParser(
        description="Genome-wide comparison of poly(A) length distributions "
        "between two conditions using weighted two-sample tests."
    )
    parser.add_argument(
        "-c1",
        "--condition1",
        required=True,
        help="Condition 1 TSV/TSV.GZ file",
    )
    parser.add_argument(
        "-c2",
        "--condition2",
        required=True,
        help="Condition 2 TSV/TSV.GZ file",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output TSV results file",
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


def main(args: argparse.Namespace | None = None) -> None:
    """Compare poly(A) length distributions between two conditions.

    Reads two poly(A) TSV files (from ``polya_calc`` or ``polya_t2g``),
    performs weighted KS, t-test, and rank-sum tests on each shared
    feature, applies global BH FDR correction per test type, and writes
    a comparison table.
    """
    if args is None:
        args = parse_args()

    id_name_1, cond1_data = parse_polyA_file(args.condition1)
    id_name_2, cond2_data = parse_polyA_file(args.condition2)

    id_col_header = id_name_1 if id_name_1 == id_name_2 else "feature_id"

    # Only compare features present in both conditions
    shared_features = sorted(set(cond1_data.keys()) & set(cond2_data.keys()))
    print(
        f"Comparing {len(shared_features)} shared features genome-wide...",
        file=sys.stderr,
    )

    results: list[dict] = []

    for feat_id in shared_features:
        f1 = cond1_data[feat_id]
        f2 = cond2_data[feat_id]

        # Filter by min_asp and non-negative pa_lens
        mask1 = (f1["probs"] >= args.min_asp) & (f1["pa_lens"] >= 0)
        mask2 = (f2["probs"] >= args.min_asp) & (f2["pa_lens"] >= 0)

        eff1 = int(np.sum(mask1))
        eff2 = int(np.sum(mask2))

        row: dict = {
            "feat_id": feat_id,
            "n_reads_1": eff1,
            "pa_wlen_1": float("nan"),
            "n_reads_2": eff2,
            "pa_wlen_2": float("nan"),
            "ks_stat": float("nan"),
            "ks_p_value": float("nan"),
            "ks_q_value": float("nan"),
            "t_stat": float("nan"),
            "t_p_value": float("nan"),
            "t_q_value": float("nan"),
            "u_stat": float("nan"),
            "u_p_value": float("nan"),
            "u_q_value": float("nan"),
        }

        if eff1 >= args.min_pareads and eff2 >= args.min_pareads:
            p1 = f1["probs"][mask1]
            l1 = f1["pa_lens"][mask1]
            p2 = f2["probs"][mask2]
            l2 = f2["pa_lens"][mask2]

            # Weighted mean poly(A) lengths
            row["pa_wlen_1"] = (
                float(np.average(l1, weights=p1)) if p1.sum() > 0 else float("nan")
            )
            row["pa_wlen_2"] = (
                float(np.average(l2, weights=p2)) if p2.sum() > 0 else float("nan")
            )

            # Run three tests
            ks_stat, ks_p = weighted_ks_test(l1, p1, l2, p2)
            row["ks_stat"] = ks_stat
            row["ks_p_value"] = ks_p

            t_stat, t_p = weighted_t_test(l1, p1, l2, p2)
            row["t_stat"] = t_stat
            row["t_p_value"] = t_p

            u_stat, u_p = weighted_rank_sum_test(l1, p1, l2, p2)
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
        f"Writing statistical test matrix to {output_filename}...",
        file=sys.stderr,
    )

    write_mode = "wt" if output_filename.endswith(".gz") else "w"
    with open_by_suffix(output_filename, write_mode) as out_f:
        out_f.write(
            f"{id_col_header}\tn_reads_1\tpa_wlen_1\t"
            f"n_reads_2\tpa_wlen_2\t"
            f"ks_stat\tks_p_value\tks_q_value\t"
            f"t_stat\tt_p_value\tt_q_value\t"
            f"u_stat\tu_p_value\tu_q_value\n"
        )

        for row in results:
            feat = row["feat_id"]
            n1 = row["n_reads_1"]
            n2 = row["n_reads_2"]

            # Format numeric columns
            wlen1 = format_float(row["pa_wlen_1"], ".2f")
            wlen2 = format_float(row["pa_wlen_2"], ".2f")

            ks_s = format_float(row["ks_stat"], ".5f")
            ks_p = format_float(row["ks_p_value"], ".5e")
            ks_q = format_float(row["ks_q_value"], ".6f")

            t_s = format_float(row["t_stat"], ".5f")
            t_p = format_float(row["t_p_value"], ".5e")
            t_q = format_float(row["t_q_value"], ".6f")

            u_s = format_float(row["u_stat"], ".5f")
            u_p = format_float(row["u_p_value"], ".5e")
            u_q = format_float(row["u_q_value"], ".6f")

            out_f.write(
                f"{feat}\t{n1}\t{wlen1}\t{n2}\t{wlen2}\t"
                f"{ks_s}\t{ks_p}\t{ks_q}\t"
                f"{t_s}\t{t_p}\t{t_q}\t"
                f"{u_s}\t{u_p}\t{u_q}\n"
            )

    print("Genome-wide comparison complete!", file=sys.stderr)


if __name__ == "__main__":
    main()
