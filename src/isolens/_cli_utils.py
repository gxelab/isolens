"""Shared CLI argument definitions and utilities for isolens subcommands.

Each ``opt_*`` function returns a ``click.option(...)`` decorator so
subcommands can reuse common argument definitions without duplication.
Usage::

    @click.command()
    @opt_output()
    @opt_verbose()
    def my_cmd(output, verbose):
        ...
"""

import argparse

import click

# ---------------------------------------------------------------------------
# Shared option factories
# ---------------------------------------------------------------------------


def opt_output(**kwargs: object) -> click.option:
    """``-o / --output`` — output file path (required)."""
    opts: dict[str, object] = {"required": True, "help": "Output file path"}
    opts.update(kwargs)
    return click.option("-o", "--output", **opts)  # type: ignore[arg-type]


def opt_format(default: str = "parquet") -> click.option:
    """``-f / --format`` — output format (parquet | tsv)."""
    return click.option(
        "-f",
        "--format",
        type=click.Choice(["parquet", "tsv"]),
        default=default,
        show_default=True,
        help="Output format",
    )


def opt_gzip() -> click.option:
    """``-z / --gzip`` — gzip-compress TSV output."""
    return click.option(
        "-z",
        "--gzip",
        is_flag=True,
        help="Gzip-compress TSV output (ignored for Parquet)",
    )


def opt_verbose() -> click.option:
    """``-v / --verbose`` — print progress to stderr."""
    return click.option(
        "-v", "--verbose", is_flag=True, help="Print progress to stderr"
    )


def opt_min_asp(**kwargs: object) -> click.option:
    """``-p / --min-asp`` — minimum Oarfish assignment probability."""
    opts: dict[str, object] = {
        "type": float,
        "default": 0.0,
        "show_default": True,
        "help": "Minimum Oarfish assignment probability filter",
    }
    opts.update(kwargs)
    return click.option("-p", "--min-asp", **opts)  # type: ignore[arg-type]


def opt_transcripts() -> click.option:
    """``-x / --transcripts`` — restrict to specific transcript IDs (repeatable)."""
    return click.option(
        "-x",
        "--transcripts",
        multiple=True,
        default=None,
        metavar="TX",
        help="Only process specified transcript ID(s). [default: all transcripts]",
    )


def opt_gtf(**kwargs: object) -> click.option:
    """``-g / --gtf`` — GTF annotation file."""
    opts: dict[str, object] = {"default": None, "help": "GTF annotation file"}
    opts.update(kwargs)
    return click.option("-g", "--gtf", **opts)  # type: ignore[arg-type]


def opt_log() -> click.option:
    """``-l / --log`` — log-transform poly(A) tail lengths."""
    return click.option(
        "-l",
        "--log",
        is_flag=True,
        default=False,
        help="Apply log-transform to poly(A) tail lengths",
    )


def opt_h5(**kwargs: object) -> click.option:
    """``-i / --h5`` — input HDF5 file(s) (repeatable)."""
    opts: dict[str, object] = {
        "required": True,
        "multiple": True,
        "help": "Input HDF5 file from mod_scan (repeat for multiple files)",
    }
    opts.update(kwargs)
    return click.option("-i", "--h5", **opts)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Namespace bridge
# ---------------------------------------------------------------------------


def ns(**kwargs: object) -> argparse.Namespace:
    """Build an ``argparse.Namespace`` from click keyword arguments.

    Click ``multiple=True`` options produce tuples, but the existing
    modules expect lists (from argparse ``nargs="+"`` / ``nargs="*"``).
    This helper converts all tuples to lists so ``main(args)`` receives
    attributes it can index and iterate.

    Empty tuples from click are converted to ``None`` because the modules
    use ``if args.transcripts is not None`` (etc.) as the "was this option
    provided?" check, and an empty list would incorrectly trigger filtering.
    """
    converted: dict[str, object] = {}
    for k, v in kwargs.items():
        if isinstance(v, tuple):
            converted[k] = list(v) if v else None
        else:
            converted[k] = v
    return argparse.Namespace(**converted)
