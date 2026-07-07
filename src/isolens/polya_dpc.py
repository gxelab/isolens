#!/usr/bin/env python3
"""Genome-wide comparison of poly(A) length distributions between two
conditions using weighted two-sample tests (KS, t-test, rank-sum).
"""

import argparse
import sys

import numpy as np
import pyarrow as pa

try:
    from isolens._io import ensure_gz_suffix, write_parquet, write_tsv
    from isolens._parsing import parse_polyA_file
    from isolens._stats import (
        bh_fdr,
        weighted_ks_test,
        weighted_median,
        weighted_rank_sum_test,
        weighted_t_test,
    )
except ImportError:
    from _io import ensure_gz_suffix, write_parquet, write_tsv  # type: ignore[no-redef]

    from _parsing import parse_polyA_file  # type: ignore[no-redef]
    from _stats import (  # type: ignore[no-redef]
        bh_fdr,
        weighted_ks_test,
        weighted_median,
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
        "-f",
        "--format",
        choices=["parquet", "tsv"],
        default="tsv",
        help="Output format: tsv (default) or parquet",
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
    parser.add_argument(
        "-l",
        "--log",
        action="store_true",
        default=False,
        help="Apply log-transform (log(L+1)) to poly(A) tail lengths before "
        "computing weighted means, medians, and hypothesis tests, then "
        "back-transform results for wmlen and wmedlen.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    """Compare poly(A) length distributions between two conditions.

    Reads two poly(A) TSV files (from ``polya_calc`` or ``polya_gene``),
    performs weighted KS, t-test, and rank-sum tests on each shared
    feature, applies global BH FDR correction per test type, and writes
    a comparison table.
    """
    if args is None:
        args = parse_args()

    id_name_1, cond1_data = parse_polyA_file(args.condition1)
    id_name_2, cond2_data = parse_polyA_file(args.condition2)

    id_col_header = id_name_1 if id_name_1 == id_name_2 else "feature_id"

    _OUTPUT_COLS = [
        id_col_header,
        "n_reads_1",
        "total_wt_1",
        "wmlen_1",
        "wmedlen_1",
        "n_reads_2",
        "total_wt_2",
        "wmlen_2",
        "wmedlen_2",
        "ks_stat",
        "ks_p_value",
        "ks_q_value",
        "wmlen_diff",
        "t_stat",
        "t_p_value",
        "t_q_value",
        "wmedlen_diff",
        "u_stat",
        "u_p_value",
        "u_q_value",
    ]
    _TSV_HEADER = "\t".join(_OUTPUT_COLS)

    _SCHEMA = pa.schema(
        [
            (id_col_header, pa.string()),
            ("n_reads_1", pa.int32()),
            ("total_wt_1", pa.float64()),
            ("wmlen_1", pa.float64()),
            ("wmedlen_1", pa.float64()),
            ("n_reads_2", pa.int32()),
            ("total_wt_2", pa.float64()),
            ("wmlen_2", pa.float64()),
            ("wmedlen_2", pa.float64()),
            ("ks_stat", pa.float64()),
            ("ks_p_value", pa.float64()),
            ("ks_q_value", pa.float64()),
            ("wmlen_diff", pa.float64()),
            ("t_stat", pa.float64()),
            ("t_p_value", pa.float64()),
            ("t_q_value", pa.float64()),
            ("wmedlen_diff", pa.float64()),
            ("u_stat", pa.float64()),
            ("u_p_value", pa.float64()),
            ("u_q_value", pa.float64()),
        ]
    )

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

        # Filter by min_asp and non-negative lengths
        mask1 = (f1["weights"] >= args.min_asp) & (f1["lengths"] >= 0)
        mask2 = (f2["weights"] >= args.min_asp) & (f2["lengths"] >= 0)

        eff1 = int(np.sum(mask1))
        eff2 = int(np.sum(mask2))

        row: dict = {
            id_col_header: feat_id,
            "n_reads_1": eff1,
            "total_wt_1": float("nan"),
            "wmlen_1": float("nan"),
            "wmedlen_1": float("nan"),
            "n_reads_2": eff2,
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

        if eff1 >= args.min_pareads and eff2 >= args.min_pareads:
            p1 = f1["weights"][mask1]
            l1 = f1["lengths"][mask1]
            p2 = f2["weights"][mask2]
            l2 = f2["lengths"][mask2]

            use_log = getattr(args, "log", False)
            if use_log:
                l1 = np.log(l1 + 1.0)
                l2 = np.log(l2 + 1.0)

            row["total_wt_1"] = float(p1.sum())
            row["total_wt_2"] = float(p2.sum())

            # Weighted mean poly(A) lengths
            if use_log:
                row["wmlen_1"] = (
                    float(np.exp(np.average(l1, weights=p1)) - 1.0)
                    if p1.sum() > 0
                    else float("nan")
                )
                row["wmlen_2"] = (
                    float(np.exp(np.average(l2, weights=p2)) - 1.0)
                    if p2.sum() > 0
                    else float("nan")
                )
            else:
                row["wmlen_1"] = (
                    float(np.average(l1, weights=p1)) if p1.sum() > 0 else float("nan")
                )
                row["wmlen_2"] = (
                    float(np.average(l2, weights=p2)) if p2.sum() > 0 else float("nan")
                )

            # Weighted median poly(A) lengths
            if use_log:
                wm1 = weighted_median(l1, p1)
                wm2 = weighted_median(l2, p2)
                row["wmedlen_1"] = (
                    float(np.exp(wm1) - 1.0) if not np.isnan(wm1) else float("nan")
                )
                row["wmedlen_2"] = (
                    float(np.exp(wm2) - 1.0) if not np.isnan(wm2) else float("nan")
                )
            else:
                row["wmedlen_1"] = weighted_median(l1, p1)
                row["wmedlen_2"] = weighted_median(l2, p2)

            # Differences
            if not np.isnan(row["wmlen_1"]) and not np.isnan(row["wmlen_2"]):
                row["wmlen_diff"] = row["wmlen_1"] - row["wmlen_2"]
            if not np.isnan(row["wmedlen_1"]) and not np.isnan(row["wmedlen_2"]):
                row["wmedlen_diff"] = row["wmedlen_1"] - row["wmedlen_2"]

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
    if args.format == "tsv":
        out_path = ensure_gz_suffix(args.output, args.gzip)
        write_tsv(results, out_path, _TSV_HEADER, _OUTPUT_COLS, args.gzip)
    else:
        write_parquet(results, args.output, _SCHEMA, _OUTPUT_COLS)

    print("Genome-wide comparison complete!", file=sys.stderr)


if __name__ == "__main__":
    main()
