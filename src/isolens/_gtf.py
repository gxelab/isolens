"""Shared GTF parsing utilities for the isolens pipeline.

Used by mod_sites.py, polya_calc.py, polya_t2g.py, and other modules
that need transcript-to-gene mapping from GTF annotations.
"""

import sys


def load_gtf(gtf_path: str) -> dict:
    """Parse a GTF annotation file and return ``{tx_name: Transcript}``.

    Wraps ``gppy.gtf.parse_gtf`` with import guarding and progress
    logging.  Returns the full transcript-keyed dictionary so callers
    can access gene metadata, genomic coordinates, and coordinate
    conversion methods (e.g. ``tpos_to_gpos``).

    Args:
        gtf_path: Path to a GTF file (plain or ``.gz``).

    Returns:
        ``dict[str, gppy.gtf.Transcript]`` keyed by transcript name.
    """
    try:
        from gppy.gtf import parse_gtf  # type: ignore[import-untyped]
    except ImportError:
        print(
            "Error: --gtf requires the 'gppy' package. "
            "Install it with: pip install gppy",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Reading GTF annotation from {gtf_path}...", file=sys.stderr)
    return parse_gtf(gtf_path)


def build_tx_to_gene(gtf_path: str) -> dict[str, str]:
    """Parse a GTF file and return ``{tx_name: gene_id}``.

    Convenience wrapper around :func:`load_gtf` that extracts only the
    transcript-name → gene-id mapping.

    Args:
        gtf_path: Path to a GTF file (plain or ``.gz``).

    Returns:
        ``dict[str, str]`` mapping each transcript name to its gene ID.
    """
    gtf = load_gtf(gtf_path)
    tx_to_gene: dict[str, str] = {}
    for tx_name, tx in gtf.items():
        tx_to_gene[tx_name] = tx.gene.gene_id
    print(
        f"Loaded gene mappings for {len(tx_to_gene)} transcripts.",
        file=sys.stderr,
    )
    return tx_to_gene
