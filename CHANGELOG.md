# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-06-21

### Added

- `mod_scan` module — generates HDF5 transcript-specific read × position modification
  matrices from a coordinate-sorted transcriptome BAM and Oarfish assignment
  probabilities. Encodes alignment states (uncovered, canonical, mismatch,
  deletion, modification) as uint8 values with gzip+shuffle compression.
- `mod_sites` module — per-position modification summaries from a `mod_scan` HDF5
  file. Computes modification levels (fraction of reads modified) and tracks
  mismatches and deletions separately. Outputs Parquet or TSV.
- `mod_corr` module — pairwise modification site correlation analysis within and
  across modification types. Computes Phi coefficient, odds ratio (with
  Haldane-Anscombe correction), Fisher's exact test p-value, Benjamini-Hochberg
  FDR q-value, and mutual information. Optional per-transcript pyramid heatmap
  PDF generation.
- `polya_calc` module — transcript-level poly(A) tail length estimation from
  Dorado BAM `pt:i` tags weighted by Oarfish assignment probabilities.
- `polya_merge` module — merge two poly(A) TSV files from replicate experiments,
  recomputing weighted average tail lengths from pooled per-read data.
- `polya_diff` module — differential poly(A) length comparison between two
  conditions using a weighted two-sample KS test with Kish's effective sample
  size correction.
- `polya_t2g` module — aggregate transcript-level poly(A) estimates to gene
  level via a user-provided `tx_name → gene_id` mapping file.
- `_parsing` module — shared Oarfish LZ4 assignment file parser and
  `TargetAssignment` data structure.
- `--mod-type` (`-m`) option to `mod_scan` for specifying which modification
  types to scan (defaults to the standard RNA modification table: m6A, m5C,
  inosine, pseU, 2'-O-methyl variants).
- `--max-depth` (`-d`) option to `mod_scan` to cap reads per transcript,
  preventing memory blow-up on highly expressed transcripts.
- `--min-asp` (`-p`) option to `mod_scan`, `mod_sites`, and `mod_corr` for
  Oarfish assignment probability filtering.
- `--threads` (`-t`) option to `mod_scan` for parallel transcript processing via
  `ProcessPoolExecutor` with stream-drain back-pressure.
- `--sites` (`-s`) option to `mod_sites` for restricting output to
  user-provided predefined modification positions.
- `--transcripts` (`-x`) option to `mod_sites` and `mod_corr` for filtering
  to a subset of transcript IDs.
- `--plot` (`-P`) option to `mod_corr` for generating per-transcript pyramid
  heatmap PDFs with ColorBrewer-styled modification type colors and
  proportional transcript axes.
- `--format` (`-f`) option to `mod_sites` and `mod_corr` for choosing between
  Parquet and TSV output.
- `--gzip` (`-z`) option to `mod_sites`, `mod_corr`, `polya_calc`,
  `polya_merge`, `polya_diff`, and `polya_t2g` for compressed TSV output.
- Python API: `parse_oarfish()` function for programmatic access to Oarfish
  read-to-transcript assignment probabilities.
- Example test dataset in `examples/` (subset of two _Drosophila_ transcripts).
- `scripts/asp_extract.py` — extract Oarfish assignment subsets for specific
  transcripts.
- `scripts/mod_plot.py` — visualization utilities for modification data.
- CIGAR-based read-to-transcript position mapping supporting all standard
  operators (=, X, M, D, I, S, N, H, P) with proper edge-case handling.
- Support for both uppercase (`MM`/`ML`) and lowercase (`mm`/`ml`)
  base modification tag variants.
- `_ReadRecord` extraction pattern for safe multi-threaded BAM processing
  without passing live pysam objects to worker processes/threads.
- `notebooks/` — file format specification and workflow documentation for the
  modification and poly(A) pipelines.
- `docs/` — SAM/BAM format specification reference (SAMv1.pdf, SAMtags.pdf).
- `CLAUDE.md` — project architecture overview, build commands, and development
  guide.

### Changed

- Package renamed from `lrkit` to `isolens`.
- Poly(A) analysis modules migrated from standalone `scripts/` to
  package-installable modules (`python -m isolens.polya_*`).
- Repository URLs updated from `mt1022/isolens` to `gxelab/isolens`.
- Copyright holder updated from Hong Zhang to GxE Lab.
- Development dependencies updated: `pytest>=9.1.0`, `pytest-cov>=7.1.0`,
  `ruff>=0.15.17`.
- ruff target-version bumped from `py310` to `py314`.

### Removed

- Legacy monolithic `main.py` pipeline, superseded by the modular `mod_scan`,
  `mod_sites`, and `mod_corr` modules.

### Fixed

- BAM coordinate-sort validation: `mod_scan` now warns when the input BAM
  is not coordinate-sorted, since streaming relies on sort order.
- Memory optimization: `_ReadRecord` extraction avoids redundant pysam C
  calls for MM/ML tag access during multi-threaded processing.
- Correct handling of alignment edge cases: positions before transcript start
  (negative reference position), CIGAR operators spanning transcript
  boundaries, and soft/hard-clipped read bases.
- Read UUID parsing: graceful fallback for reads with non-UUID query names
  instead of crashing.

## [0.1.0] - 2026-04-10

### Added

- Initial package skeleton following modern Python packaging standards (PEP 517/518/621).
- `src/` layout with `isolens` package.
- `py.typed` marker for PEP 561 compliance.
- `pyproject.toml` with `hatchling` build backend.
- CI workflow for testing and publishing to PyPI.

[Unreleased]: https://github.com/gxelab/isolens/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/gxelab/isolens/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/gxelab/isolens/releases/tag/v0.1.0
