#!/usr/bin/env python3
"""mod_sites: Per-position modification summaries from a mod_scan HDF5 file.

Part of the isolens toolkit.  Reads the HDF5 produced by ``mod_scan.py`` and
writes a Parquet file with one row per (transcript, position, modification_type).

For each position the output includes the following read-count categories
(see the :func:`compute_transcript_stats` docstring for definitions):

* **modified** — this modification type won and passed the threshold.
* **canonical** — the unmodified base won (all mod probs below canonical).
* **othermod** — another modification type won above threshold.
* **mismatch** — reference mismatch (CIGAR ``X``), preserved from CIGAR.
* **deletion** — deletion (CIGAR ``D``), preserved from CIGAR.
* **failed** — all states below the probability threshold.

See notebooks/01_mod.md for the full specification.
"""

import argparse
import sys

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from isolens.mod_scan import (
        CODE_CANONICAL,
        CODE_DELETION,
        CODE_FAIL,
        CODE_MISMATCH,
    )
except ImportError:
    from mod_scan import (  # type: ignore[no-redef]
        CODE_CANONICAL,
        CODE_DELETION,
        CODE_FAIL,
        CODE_MISMATCH,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for mod_sites."""
    parser = argparse.ArgumentParser(
        description="mod_sites: Per-position modification summaries from HDF5"
    )
    parser.add_argument(
        "-i",
        "--h5",
        required=True,
        help="Input HDF5 file from mod_scan",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output file path",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=["parquet", "tsv"],
        default="parquet",
        help="Output format: parquet (default) or tsv",
    )
    parser.add_argument(
        "-z",
        "--gzip",
        action="store_true",
        help="Gzip-compress TSV output (ignored for parquet)",
    )
    parser.add_argument(
        "-s",
        "--sites",
        default=None,
        help="Predefined modification sites TSV (columns: tx_name, posn). "
        "When provided, only these positions are emitted, for all "
        "modification types, even if n_modified == 0.",
    )
    parser.add_argument(
        "-p",
        "--min-asp",
        type=float,
        default=0.0,
        help="Minimum Oarfish assignment probability for a read to be "
        "included [default: 0.0 (no filter)]",
    )
    parser.add_argument(
        "-x",
        "--transcripts",
        nargs="+",
        default=None,
        metavar="TX",
        help="Only process the specified transcript ID(s). "
        "[default: all transcripts in the HDF5]",
    )
    parser.add_argument(
        "-g",
        "--gtf",
        default=None,
        help="GTF annotation file for mapping transcript coordinates "
        "to genomic coordinates. When provided, three additional "
        "columns (chrom, strand, gpos) are appended to the output.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print progress to stderr",
    )
    return parser.parse_args()


def read_predefined_sites(path: str) -> dict[str, set[int]]:
    """Read a predefined modification sites TSV file.

    The file must have a header row with columns ``tx_name`` (transcript
    name) and ``posn`` (1-based position).  Additional columns are ignored.

    Args:
        path: Path to the TSV file.

    Returns:
        ``dict[str, set[int]]`` — ``{tx_name: {pos1, pos2, ...}}`` with
        1-based positions.
    """
    sites: dict[str, set[int]] = {}
    with open(path, encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        try:
            tx_col = header.index("tx_name")
            pos_col = header.index("posn")
        except ValueError as exc:
            raise ValueError(
                f"Sites file must have 'tx_name' and 'posn' columns; found: {header}"
            ) from exc
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) <= max(tx_col, pos_col):
                continue
            tx = parts[tx_col]
            try:
                pos = int(parts[pos_col])
            except ValueError:
                continue
            sites.setdefault(tx, set()).add(pos)
    return sites


def compute_transcript_stats(
    matrix: np.ndarray,
    weights: np.ndarray,
    mod_codes: list[tuple[str, int]],
    predefined_positions: set[int] | None = None,
) -> list[dict]:
    """Compute per-position statistics for a single transcript.

    For each (position, modification_type) pair the following columns
    are computed:

    * **n_modified** — number of reads where this mod type won (including
      weighted variant ``wt_modified``).
    * **n_canonical** — number of reads where the canonical (unmodified)
      base won (including ``wt_canonical``).
    * **n_othermod** — number of reads where a different, non-focal
      modification type won (including ``wt_othermod``).
    * **n_unmodified** = n_canonical + n_othermod (and ``wt_unmodified``).
    * **n_mismatch** — number of reads with a CIGAR reference mismatch
      at this position (and ``wt_mismatch``).
    * **n_deletion** — number of reads with a CIGAR deletion at this
      position (and ``wt_deletion``).
    * **n_failed** — number of reads where no state exceeded the
      probability threshold (and ``wt_failed``).
    * **mod_level** = n_modified / (n_modified + n_unmodified).
    * **wt_mod_level** — weighted variant using assignment probabilities.

    Args:
        matrix: ``numpy.ndarray`` of shape ``(n_reads, tx_length)``,
            dtype uint8.  Encoded per ``mod_scan.CODE_*`` constants.
        weights: ``numpy.ndarray`` of shape ``(n_reads,)``, dtype float32.
            Oarfish assignment probabilities for each read.
        mod_codes: ``list[(mod_type_str, code)]`` — modification codes
            (code ≥ 4) read from the HDF5 ``/modification_codes`` group.
        predefined_positions: Optional ``set[int]`` of 1-based positions to
            restrict output to.  When provided, only these positions are
            emitted for every modification type, even if ``n_modified == 0``.
            When ``None`` (the default), positions with ``n_modified > 0``
            for the focal modification type are emitted.

    Returns:
        ``list[dict]`` — one dict per (position, modification_type) with
        all columns listed above.
    """
    n_reads, tx_length = matrix.shape
    w64 = weights.astype(np.float64)  # (n_reads,) float64

    # ---- base stats (same for all modification types at each position) ----

    mismatch_mask = matrix == CODE_MISMATCH  # bool (n_reads, tx_length)
    n_mismatch = np.sum(mismatch_mask, axis=0, dtype=np.int32)
    w_mismatch = w64 @ mismatch_mask  # (tx_length,) float64

    deletion_mask = matrix == CODE_DELETION
    n_del = np.sum(deletion_mask, axis=0, dtype=np.int32)
    w_del = w64 @ deletion_mask

    failed_mask = matrix == CODE_FAIL
    n_failed = np.sum(failed_mask, axis=0, dtype=np.int32)
    w_failed = w64 @ failed_mask

    canonical_mask = matrix == CODE_CANONICAL  # bool (n_reads, tx_length)
    n_canonical = np.sum(canonical_mask, axis=0, dtype=np.int32)
    w_canonical = w64 @ canonical_mask

    # ---- per-modification-type stats ----

    rows: list[dict] = []

    for mod_str, code in mod_codes:
        mod_mask = matrix == code  # bool (n_reads, tx_length)
        n_mod = np.sum(mod_mask, axis=0, dtype=np.int32)
        w_mod = w64 @ mod_mask

        # Other modifications: any mod code (≥4) that is not the focal
        # type and not CODE_FAIL
        othermod_mask = (matrix >= 4) & (matrix != code) & (matrix != CODE_FAIL)
        n_othermod = np.sum(othermod_mask, axis=0, dtype=np.int32)
        w_othermod = w64 @ othermod_mask

        # Unmodified = canonical + othermod
        n_unmod = n_canonical + n_othermod
        w_unmod = w_canonical + w_othermod

        # Determine which positions to emit
        if predefined_positions is not None:
            # Only emit predefined positions (within transcript bounds)
            positions_0b = sorted(
                p - 1 for p in predefined_positions if 1 <= p <= tx_length
            )
            if not positions_0b:
                continue
            positions = np.array(positions_0b, dtype=np.intp)
        else:
            # Emit all positions with at least one focal modification call
            positions = np.flatnonzero(n_mod > 0)
            if len(positions) == 0:
                continue

        n_mod_pos = n_mod[positions]
        w_mod_pos = w_mod[positions]

        n_unmod_pos = n_unmod[positions]
        w_unmod_pos = w_unmod[positions]

        n_canonical_pos = n_canonical[positions]
        w_canonical_pos = w_canonical[positions]

        n_othermod_pos = n_othermod[positions]
        w_othermod_pos = w_othermod[positions]

        n_mismatch_pos = n_mismatch[positions]
        w_mismatch_pos = w_mismatch[positions]

        n_del_pos = n_del[positions]
        w_del_pos = w_del[positions]

        n_failed_pos = n_failed[positions]
        w_failed_pos = w_failed[positions]

        # Modification level denominator: modified + unmodified only
        # (failed, mismatch, deletion are excluded)
        denom = n_mod_pos + n_unmod_pos
        w_denom = w_mod_pos + w_unmod_pos

        ml = np.divide(
            n_mod_pos,
            denom,
            where=denom > 0,
            out=np.zeros_like(n_mod_pos, dtype=np.float64),
        )
        w_ml = np.divide(
            w_mod_pos,
            w_denom,
            where=w_denom > 0,
            out=np.zeros_like(w_mod_pos, dtype=np.float64),
        )

        for i in range(len(positions)):
            rows.append(
                {
                    "transcript_id": "",  # filled by caller
                    "position": int(positions[i]) + 1,  # 1-based
                    "mod_type": mod_str,
                    "n_modified": int(n_mod_pos[i]),
                    "wt_modified": float(round(w_mod_pos[i], 4)),
                    "n_unmodified": int(n_unmod_pos[i]),
                    "wt_unmodified": float(round(w_unmod_pos[i], 4)),
                    "n_canonical": int(n_canonical_pos[i]),
                    "wt_canonical": float(round(w_canonical_pos[i], 4)),
                    "n_othermod": int(n_othermod_pos[i]),
                    "wt_othermod": float(round(w_othermod_pos[i], 4)),
                    "n_mismatch": int(n_mismatch_pos[i]),
                    "wt_mismatch": float(round(w_mismatch_pos[i], 4)),
                    "n_deletion": int(n_del_pos[i]),
                    "wt_deletion": float(round(w_del_pos[i], 4)),
                    "n_failed": int(n_failed_pos[i]),
                    "wt_failed": float(round(w_failed_pos[i], 4)),
                    "mod_level": float(round(ml[i], 6)),
                    "wt_mod_level": float(round(w_ml[i], 6)),
                }
            )

    return rows


def main() -> None:
    """Compute per-position modification summaries from a mod_scan HDF5.

    Reads the HDF5 file produced by ``mod_scan`` and writes a Parquet
    (or TSV) file with one row per (transcript, position, modification_type)
    containing read counts and weighted modification levels.
    """
    args = parse_args()

    # ---- 0. Read predefined sites (optional) ----

    predefined_sites: dict[str, set[int]] | None = None
    if args.sites is not None:
        predefined_sites = read_predefined_sites(args.sites)
        if args.verbose:
            n_tx = len(predefined_sites)
            n_pos = sum(len(v) for v in predefined_sites.values())
            print(
                f"[mod_sites] Predefined sites: {n_pos} positions across "
                f"{n_tx} transcripts",
                file=sys.stderr,
            )

    # ---- 0b. Parse GTF annotation (optional) ----

    gtf: dict | None = None
    if args.gtf is not None:
        try:
            from gppy.gtf import parse_gtf  # type: ignore[import-untyped]
        except ImportError:
            print(
                "[mod_sites] Error: --gtf requires the 'gppy' package. "
                "Install it with: pip install gppy",
                file=sys.stderr,
            )
            sys.exit(1)

        gtf = parse_gtf(args.gtf)
        if args.verbose:
            print(
                f"[mod_sites] Loaded {len(gtf)} transcripts from GTF",
                file=sys.stderr,
            )

    # ---- 1. Read modification codes from HDF5 ----

    with h5py.File(args.h5, "r") as h5:
        # Parse modification codes
        mod_codes: list[tuple[str, int]] = []
        codes_grp = h5["modification_codes"]
        for mod_str, code in sorted(codes_grp.attrs.items(), key=lambda x: x[1]):
            mod_codes.append((mod_str, int(code)))

        tx_names = sorted(h5["transcripts"].keys())
        if args.transcripts is not None:
            requested = set(args.transcripts)
            tx_names = sorted(tx for tx in tx_names if tx in requested)
            if args.verbose:
                print(
                    f"[mod_sites] Filtered to {len(tx_names)}/"
                    f"{len(requested)} requested transcripts",
                    file=sys.stderr,
                )
        n_transcripts = len(tx_names)

        if args.verbose:
            print(
                f"[mod_sites] {n_transcripts} transcripts, "
                f"{len(mod_codes)} modification types",
                file=sys.stderr,
            )

        # ---- 2. Process each transcript ----

        all_rows: list[dict] = []
        processed = 0

        for tx_name in tx_names:
            grp = h5[f"transcripts/{tx_name}"]
            matrix = grp["matrix"][:]  # (n_reads, tx_length) uint8
            weights = grp["read_weights"][:]  # (n_reads,) float32

            if args.min_asp > 0.0:
                read_mask = weights >= args.min_asp
                if read_mask.sum() == 0:
                    processed += 1
                    continue
                matrix = matrix[read_mask]
                weights = weights[read_mask]

            tx_rows = compute_transcript_stats(
                matrix,
                weights,
                mod_codes,
                predefined_positions=(
                    predefined_sites.get(tx_name)
                    if predefined_sites is not None
                    else None
                ),
            )
            for row in tx_rows:
                row["transcript_id"] = tx_name

            # ---- Enrich with genomic coordinates (if GTF provided) ----
            if gtf is not None:
                tx_gtf = gtf.get(tx_name)
                if tx_gtf is None:
                    if args.verbose:
                        print(
                            f"[mod_sites] Warning: {tx_name} not found in GTF",
                            file=sys.stderr,
                        )
                    for row in tx_rows:
                        row["gene_id"] = None
                        row["chrom"] = None
                        row["strand"] = None
                        row["gpos"] = None
                else:
                    gene_id = tx_gtf.gene.gene_id
                    chrom = tx_gtf.gene.chrom
                    strand = tx_gtf.gene.strand
                    for row in tx_rows:
                        gpos = tx_gtf.tpos_to_gpos(row["position"])
                        row["gene_id"] = gene_id
                        row["chrom"] = chrom
                        row["strand"] = strand
                        row["gpos"] = gpos if gpos > 0 else None
            else:
                for row in tx_rows:
                    row["gene_id"] = None
                    row["chrom"] = None
                    row["strand"] = None
                    row["gpos"] = None

            all_rows.extend(tx_rows)

            processed += 1
            if args.verbose and processed % 1000 == 0:
                print(
                    f"[mod_sites] Processed {processed}/{n_transcripts} transcripts...",
                    file=sys.stderr,
                )

    if args.verbose:
        print(f"[mod_sites] Total rows to write: {len(all_rows)}", file=sys.stderr)

    # ---- 3. Write output ----

    if args.format == "tsv":
        _write_tsv(all_rows, args.output, args.gzip)
    else:
        _write_parquet(all_rows, args.output)

    if args.verbose:
        print(f"[mod_sites] Done. Output written to {args.output}", file=sys.stderr)


# ---------- output writers ----------

_TSV_HEADER = (
    "transcript_id\tposition\tmod_type\tn_modified\twt_modified"
    "\tn_unmodified\twt_unmodified\tn_canonical\twt_canonical"
    "\tn_othermod\twt_othermod\tn_mismatch\twt_mismatch"
    "\tn_deletion\twt_deletion\tn_failed\twt_failed"
    "\tmod_level\twt_mod_level"
    "\tgene_id\tchrom\tstrand\tgpos"
)

_TSV_COLS = [
    "transcript_id",
    "position",
    "mod_type",
    "n_modified",
    "wt_modified",
    "n_unmodified",
    "wt_unmodified",
    "n_canonical",
    "wt_canonical",
    "n_othermod",
    "wt_othermod",
    "n_mismatch",
    "wt_mismatch",
    "n_deletion",
    "wt_deletion",
    "n_failed",
    "wt_failed",
    "mod_level",
    "wt_mod_level",
    "gene_id",
    "chrom",
    "strand",
    "gpos",
]


def _write_tsv(all_rows: list[dict], path: str, use_gzip: bool) -> None:
    """Write rows as tab-separated values, optionally gzip-compressed."""
    import gzip

    open_func = gzip.open if use_gzip else open
    mode = "wt" if use_gzip else "w"

    with open_func(path, mode, encoding="utf-8") as f:
        f.write(_TSV_HEADER + "\n")
        for row in all_rows:
            f.write(
                "\t".join(
                    "NA" if row[c] is None else str(row[c]) for c in _TSV_COLS
                )
                + "\n"
            )


def _write_parquet(all_rows: list[dict], path: str) -> None:
    """Write rows as a Parquet file via pyarrow.

    When *all_rows* is empty, writes a schema-only file so downstream
    tools can still open it without errors.
    """
    if not all_rows:
        print(
            "[mod_sites] No modification sites found — writing empty file.",
            file=sys.stderr,
        )
        schema = pa.schema(
            [
                ("transcript_id", pa.string()),
                ("position", pa.int32()),
                ("mod_type", pa.string()),
                ("n_modified", pa.int32()),
                ("wt_modified", pa.float64()),
                ("n_unmodified", pa.int32()),
                ("wt_unmodified", pa.float64()),
                ("n_canonical", pa.int32()),
                ("wt_canonical", pa.float64()),
                ("n_othermod", pa.int32()),
                ("wt_othermod", pa.float64()),
                ("n_mismatch", pa.int32()),
                ("wt_mismatch", pa.float64()),
                ("n_deletion", pa.int32()),
                ("wt_deletion", pa.float64()),
                ("n_failed", pa.int32()),
                ("wt_failed", pa.float64()),
                ("mod_level", pa.float64()),
                ("wt_mod_level", pa.float64()),
                ("gene_id", pa.string()),
                ("chrom", pa.string()),
                ("strand", pa.string()),
                ("gpos", pa.int32()),
            ]
        )
        with pq.ParquetWriter(path, schema) as writer:
            writer.write_table(
                pa.table(
                    {k: pa.array([], type=schema.field(k).type) for k in schema.names}
                )
            )
        return

    columns = {}
    for col in _TSV_COLS:
        values = [r[col] for r in all_rows]
        if col in ("transcript_id", "mod_type", "gene_id"):
            columns[col] = pa.array(values)
        elif col in ("chrom", "strand"):
            columns[col] = pa.array(values, type=pa.string())
        elif col == "position":
            columns[col] = pa.array(values, type=pa.int32())
        elif col == "gpos":
            columns[col] = pa.array(values, type=pa.int32())
        elif col.startswith("n_"):
            columns[col] = pa.array(values, type=pa.int32())
        else:
            columns[col] = pa.array(values, type=pa.float64())

    pq.write_table(pa.table(columns), path)


if __name__ == "__main__":
    main()
