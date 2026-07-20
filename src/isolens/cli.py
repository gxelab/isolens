"""IsoLens CLI — isoform-resolution epitranscriptome analysis.

Usage::

    isolens <subcommand> [options]

Available subcommands:

    Modification pipeline:
        mod_scan      Generate HDF5 modification matrices
        mod_sites     Per-position modification summaries
        mod_corr      Pairwise modification site correlation
        mod_gene      Gene-level modification aggregation

    Differential modification:
        mod_dmc       Differential modification (2-condition)
        mod_dmt       Differential modification (transcript-level)
        mod_dmcg      Differential modification (gene-level)

    Poly(A) estimation:
        polya_calc    Poly(A) tail length estimation
        polya_merge   Merge two poly(A) estimates
        polya_gene    Gene-level poly(A) aggregation
        polya_bimodal Bimodal poly(A) detection

    Differential poly(A):
        polya_dpc     Differential poly(A) (2-condition)
        polya_dpt     Differential poly(A) (transcript-level)
"""

import os

import click

from isolens import __version__
from isolens._cli_utils import (
    ns,
    opt_format,
    opt_gtf,
    opt_gzip,
    opt_h5,
    opt_log,
    opt_min_asp,
    opt_output,
    opt_transcripts,
    opt_verbose,
)

# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, "-V", "--version", message="%(version)s")
def main() -> None:
    """IsoLens: isoform-resolution epitranscriptome analysis."""


# ===========================================================================
# Modification pipeline
# ===========================================================================


@main.command("mod_scan")
@click.option(
    "-b", "--bam", required=True, help="Path to transcriptome BAM alignment file"
)
@click.option(
    "-a",
    "--oarfish",
    required=True,
    help="Path to Oarfish isoform assignment file (.lz4 or plain text)",
)
@opt_output(help="Output HDF5 file path")
@click.option(
    "-c",
    "--mod-cutoff",
    type=float,
    default=0.95,
    show_default=True,
    help="Modification probability cutoff",
)
@opt_min_asp()
@click.option(
    "-d",
    "--max-depth",
    type=int,
    default=5000,
    show_default=True,
    help="Maximum number of reads per transcript",
)
@click.option(
    "-t",
    "--threads",
    type=int,
    default=min(2, os.cpu_count() or 1),
    show_default=True,
    help="Number of worker processes",
)
@click.option(
    "-m",
    "--mod-type",
    multiple=True,
    default=["a", "m", "17596", "17802", "19228", "69426", "19229", "19227"],
    help="Modification type codes (repeatable). "
    "Defaults to standard RNA modification table.",
)
@opt_verbose()
def mod_scan(
    bam: str,
    oarfish: str,
    output: str,
    mod_cutoff: float,
    min_asp: float,
    max_depth: int,
    threads: int,
    mod_type: tuple[str, ...],
    verbose: bool,
) -> None:
    """Generate HDF5 read x position modification matrices."""
    from isolens import mod_scan as _m

    _m.main(
        ns(
            bam=bam,
            oarfish=oarfish,
            output=output,
            mod_cutoff=mod_cutoff,
            min_asp=min_asp,
            max_depth=max_depth,
            threads=threads,
            mod_type=mod_type,
            verbose=verbose,
        )
    )


@main.command("mod_sites")
@opt_h5()
@opt_output()
@opt_format(default="parquet")
@opt_gzip()
@click.option(
    "-s",
    "--sites",
    default=None,
    help="Predefined modification sites TSV (headerless: transcript_id, posn, [mod_type])",
)
@opt_min_asp()
@opt_transcripts()
@opt_gtf()
@opt_verbose()
def mod_sites(
    h5: tuple[str, ...],
    output: str,
    format: str,
    gzip: bool,
    sites: str | None,
    min_asp: float,
    transcripts: tuple[str, ...] | None,
    gtf: str | None,
    verbose: bool,
) -> None:
    """Per-position modification summaries from HDF5 matrices."""
    from isolens import mod_sites as _m

    _m.main(
        ns(
            h5=h5,
            output=output,
            format=format,
            gzip=gzip,
            sites=sites,
            min_asp=min_asp,
            transcripts=transcripts,
            gtf=gtf,
            verbose=verbose,
        )
    )


@main.command("mod_corr")
@opt_h5()
@click.option(
    "-s",
    "--sites",
    required=True,
    help="Site summary file from mod_sites (Parquet or TSV/TSV.GZ)",
)
@opt_output()
@click.option(
    "-m",
    "--min-mod-reads",
    type=int,
    default=2,
    show_default=True,
    help="Minimum number of modified reads for a site",
)
@click.option(
    "-l",
    "--min-mod-level",
    type=float,
    default=0.05,
    show_default=True,
    help="Minimum modification level for a site",
)
@click.option(
    "-c",
    "--min-coverage",
    type=int,
    default=10,
    show_default=True,
    help="Minimum total depth for a site",
)
@opt_min_asp()
@opt_format()
@opt_gzip()
@click.option(
    "-d",
    "--plot-dir",
    metavar="DIR",
    default=None,
    help="Generate rotated triangular heatmap PDFs per transcript "
    "in the given output directory",
)
@click.option(
    "-t",
    "--metric",
    type=click.Choice(["corr", "wcorr", "mi", "wmi", "or", "wor"]),
    default="wcorr",
    show_default=True,
    help="Association statistic to visualize in heatmaps",
)
@opt_transcripts()
@opt_verbose()
def mod_corr(
    h5: tuple[str, ...],
    sites: str,
    output: str,
    min_mod_reads: int,
    min_mod_level: float,
    min_coverage: int,
    min_asp: float,
    format: str,
    gzip: bool,
    plot_dir: str | None,
    metric: str,
    transcripts: tuple[str, ...] | None,
    verbose: bool,
) -> None:
    """Pairwise modification site correlation analysis."""
    from isolens import mod_corr as _m

    _m.main(
        ns(
            h5=h5,
            sites=sites,
            output=output,
            min_mod_reads=min_mod_reads,
            min_mod_level=min_mod_level,
            min_coverage=min_coverage,
            min_asp=min_asp,
            format=format,
            gzip=gzip,
            plot_dir=plot_dir,
            metric=metric,
            transcripts=transcripts,
            verbose=verbose,
        )
    )


@main.command("mod_gene")
@click.option(
    "-i",
    "--input",
    required=True,
    help="Site summary from mod_sites (Parquet or TSV/TSV.GZ). "
    "Must include genomic coordinate columns (run mod_sites with --gtf).",
)
@opt_output()
@opt_format(default="parquet")
@opt_gzip()
@opt_verbose()
def mod_gene(
    input: str,
    output: str,
    format: str,
    gzip: bool,
    verbose: bool,
) -> None:
    """Gene-level aggregation of modification site summaries."""
    from isolens import mod_gene as _m

    _m.main(ns(input=input, output=output, format=format, gzip=gzip, verbose=verbose))


# ===========================================================================
# Differential modification
# ===========================================================================


@main.command("mod_dmc")
@click.option(
    "-i1",
    "--h5-1",
    required=True,
    multiple=True,
    help="Input HDF5 file(s) for condition 1 (repeat for multiple files)",
)
@click.option(
    "-i2",
    "--h5-2",
    required=True,
    multiple=True,
    help="Input HDF5 file(s) for condition 2 (repeat for multiple files)",
)
@click.option(
    "-s1",
    "--sites-1",
    required=True,
    metavar="FILE",
    help="Pooled site summary for condition 1 (Parquet or TSV/TSV.GZ)",
)
@click.option(
    "-s2",
    "--sites-2",
    required=True,
    metavar="FILE",
    help="Pooled site summary for condition 2 (Parquet or TSV/TSV.GZ)",
)
@opt_output()
@opt_format(default="parquet")
@opt_gzip()
@opt_min_asp()
@opt_transcripts()
@opt_verbose()
def mod_dmc(
    h5_1: tuple[str, ...],
    h5_2: tuple[str, ...],
    sites_1: str,
    sites_2: str,
    output: str,
    format: str,
    gzip: bool,
    min_asp: float,
    transcripts: tuple[str, ...] | None,
    verbose: bool,
) -> None:
    """Differential modification calling between two conditions.

    Reads HDF5 files and site summaries for two conditions, matches
    sites by (transcript_id, position, mod_type), fits a weighted
    logistic regression per site, and writes results with global BH
    FDR correction.
    """
    from isolens import mod_dmc as _m

    _m.main(
        ns(
            h5_1=h5_1,
            h5_2=h5_2,
            sites_1=sites_1,
            sites_2=sites_2,
            output=output,
            format=format,
            gzip=gzip,
            min_asp=min_asp,
            transcripts=transcripts,
            verbose=verbose,
        )
    )


@main.command("mod_dmt")
@opt_h5()
@click.option(
    "-s",
    "--sites",
    required=True,
    metavar="FILE",
    help="Pooled site summary from mod_sites (Parquet or TSV/TSV.GZ). "
    "Must include genomic coordinate columns (run mod_sites with --gtf).",
)
@opt_output()
@opt_format(default="parquet")
@opt_gzip()
@opt_min_asp()
@opt_transcripts()
@opt_verbose()
def mod_dmt(
    h5: tuple[str, ...],
    sites: str,
    output: str,
    format: str,
    gzip: bool,
    min_asp: float,
    transcripts: tuple[str, ...] | None,
    verbose: bool,
) -> None:
    """Differential modification testing between transcript isoforms."""
    from isolens import mod_dmt as _m

    _m.main(
        ns(
            h5=h5,
            sites=sites,
            output=output,
            format=format,
            gzip=gzip,
            min_asp=min_asp,
            transcripts=transcripts,
            verbose=verbose,
        )
    )


@main.command("mod_dmcg")
@click.option(
    "-s1",
    "--sites-1",
    required=True,
    metavar="FILE",
    help="Gene-level site summary for condition 1 "
    "(Parquet or TSV/TSV.GZ from mod_gene)",
)
@click.option(
    "-s2",
    "--sites-2",
    required=True,
    metavar="FILE",
    help="Gene-level site summary for condition 2 "
    "(Parquet or TSV/TSV.GZ from mod_gene)",
)
@opt_output()
@opt_format(default="parquet")
@opt_gzip()
@opt_verbose()
def mod_dmcg(
    sites_1: str,
    sites_2: str,
    output: str,
    format: str,
    gzip: bool,
    verbose: bool,
) -> None:
    """Gene-level differential modification between two conditions."""
    from isolens import mod_dmcg as _m

    _m.main(
        ns(
            sites_1=sites_1,
            sites_2=sites_2,
            output=output,
            format=format,
            gzip=gzip,
            verbose=verbose,
        )
    )


# ===========================================================================
# Poly(A) estimation
# ===========================================================================


@main.command("polya_calc")
@click.option(
    "-a",
    "--oarfish",
    required=True,
    help="Oarfish read assignment probability file (.lz4 or plain text)",
)
@click.option(
    "-b",
    "--bam",
    required=True,
    help="Raw reads BAM file containing pt:i tags",
)
@opt_output()
@opt_format(default="tsv")
@opt_gzip()
@opt_min_asp()
@opt_gtf()
@opt_log()
def polya_calc(
    oarfish: str,
    bam: str,
    output: str,
    format: str,
    gzip: bool,
    min_asp: float,
    gtf: str | None,
    log: bool,
) -> None:
    """Estimate transcript poly(A) tail lengths from BAM + Oarfish."""
    from isolens import polya_calc as _m

    _m.main(
        ns(
            oarfish=oarfish,
            bam=bam,
            output=output,
            format=format,
            gzip=gzip,
            min_asp=min_asp,
            gtf=gtf,
            log=log,
        )
    )


@main.command("polya_merge")
@click.option("-i1", "--input1", required=True, help="First poly(A) estimation file")
@click.option("-i2", "--input2", required=True, help="Second poly(A) estimation file")
@opt_output()
@opt_format(default="tsv")
@opt_gzip()
@opt_log()
def polya_merge(
    input1: str,
    input2: str,
    output: str,
    format: str,
    gzip: bool,
    log: bool,
) -> None:
    """Merge two poly(A) estimation files and recompute weighted averages."""
    from isolens import polya_merge as _m

    _m.main(
        ns(
            input1=input1,
            input2=input2,
            output=output,
            format=format,
            gzip=gzip,
            log=log,
        )
    )


@main.command("polya_gene")
@click.option(
    "-i",
    "--input",
    required=True,
    help="Transcript-level poly(A) file from polya_calc (TSV/TSV.GZ or Parquet)",
)
@opt_output()
@opt_format(default="tsv")
@opt_gzip()
@opt_gtf()
@opt_log()
def polya_gene(
    input: str,
    output: str,
    format: str,
    gzip: bool,
    gtf: str | None,
    log: bool,
) -> None:
    """Aggregate transcript-level poly(A) estimates to gene level."""
    from isolens import polya_gene as _m

    _m.main(ns(input=input, output=output, format=format, gzip=gzip, gtf=gtf, log=log))


@main.command("polya_bimodal")
@click.option(
    "-i",
    "--input",
    required=True,
    help="Poly(A) file from polya_calc or polya_gene (TSV/TSV.GZ or Parquet)",
)
@opt_output(help="Output bimodality results file")
@opt_format(default="tsv")
@opt_gzip()
@click.option(
    "-l",
    "--min-length",
    type=float,
    default=0.0,
    show_default=True,
    help="Drop reads with poly(A) length below this threshold",
)
@opt_min_asp(default=0.1)
@click.option(
    "-e",
    "--min-ess",
    type=float,
    default=30.0,
    show_default=True,
    help="Skip feature if effective sample size below this threshold",
)
@click.option(
    "-k",
    "--kde-prominence",
    type=float,
    default=0.05,
    show_default=True,
    help="Prominence threshold for KDE peak detection",
)
def polya_bimodal(
    input: str,
    output: str,
    format: str,
    gzip: bool,
    min_length: float,
    min_asp: float,
    min_ess: float,
    kde_prominence: float,
) -> None:
    """Detect bimodal poly(A) tail length distributions."""
    from isolens import polya_bimodal as _m

    _m.main(
        ns(
            input=input,
            output=output,
            format=format,
            gzip=gzip,
            min_length=min_length,
            min_asp=min_asp,
            min_ess=min_ess,
            kde_prominence=kde_prominence,
        )
    )


# ===========================================================================
# Differential poly(A)
# ===========================================================================


@main.command("polya_dpc")
@click.option("-c1", "--condition1", required=True, help="Poly(A) file for condition 1")
@click.option("-c2", "--condition2", required=True, help="Poly(A) file for condition 2")
@opt_output()
@opt_format(default="tsv")
@opt_gzip()
@opt_min_asp()
@click.option(
    "-n",
    "--min-pareads",
    type=int,
    default=5,
    show_default=True,
    help="Minimum number of reads with effective poly(A) length",
)
@opt_log()
def polya_dpc(
    condition1: str,
    condition2: str,
    output: str,
    format: str,
    gzip: bool,
    min_asp: float,
    min_pareads: int,
    log: bool,
) -> None:
    """Differential poly(A) between two conditions using weighted two-sample tests."""
    from isolens import polya_dpc as _m

    _m.main(
        ns(
            condition1=condition1,
            condition2=condition2,
            output=output,
            format=format,
            gzip=gzip,
            min_asp=min_asp,
            min_pareads=min_pareads,
            log=log,
        )
    )


@main.command("polya_dpt")
@click.option(
    "-i",
    "--input",
    required=True,
    help="Transcript-level poly(A) file (TSV/TSV.GZ or Parquet)",
)
@opt_output()
@opt_format(default="tsv")
@opt_gzip()
@opt_gtf()
@opt_min_asp()
@click.option(
    "-n",
    "--min-pareads",
    type=int,
    default=5,
    show_default=True,
    help="Minimum number of reads with effective poly(A) length",
)
@opt_log()
def polya_dpt(
    input: str,
    output: str,
    format: str,
    gzip: bool,
    gtf: str | None,
    min_asp: float,
    min_pareads: int,
    log: bool,
) -> None:
    """Pairwise differential poly(A) between transcript isoforms."""
    from isolens import polya_dpt as _m

    _m.main(
        ns(
            input=input,
            output=output,
            format=format,
            gzip=gzip,
            gtf=gtf,
            min_asp=min_asp,
            min_pareads=min_pareads,
            log=log,
        )
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
