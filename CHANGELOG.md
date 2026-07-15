# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.1] - to be released

### Added

## Changed

## Fixed

## [0.5.4] - 2026-07-14

### Added

- Parquet input support for all poly(A) modules (`polya_gene`, `polya_dpc`,
  `polya_dpt`, `polya_bimodal`, `polya_merge`). Input format is auto-detected
  by `.parquet` / `.pq` suffix via the shared `parse_polyA_file()` function
  in `_parsing.py`.
- `gene_id` extraction in `parse_polyA_file()` — the column is included in
  returned data dicts when present, enabling downstream modules to use
  gene mappings without re-parsing.
- `polya_merge` now preserves the `gene_id` column through the merge when
  present in input files.

### Changed

- `polya_gene`, `polya_dpt`, and `polya_merge` refactored to use the shared
  `parse_polyA_file()` reader instead of inline TSV parsing.
- `polya_calc`: `gene_id` is now the first output column (before
  `transcript_id`) when `--gtf` is provided.
- `polya_calc`: added `-f` / `--format` flag for Parquet output.

### Fixed

- `scripts/tsv2pq.py`: "straddling object" crash on TSV files with very wide
  rows (e.g. polya_calc output with hundreds of comma-separated weights).
  Now retries with a larger `block_size` when the error is detected.
- README column counts corrected for `mod_corr` (23→25), `mod_dmc` (25→24),
  `mod_dmt` (25→26), and `polya_dpt` (21→22).
- README: `polya_gene` example command fixed (`-m` → `-g`).
- README: missing `-f`/`--format` and `-g`/`--gtf` flags added to `polya_calc`
  and `polya_merge` option tables.
- README: input format descriptions updated to mention Parquet support across
  all poly(A) modules.

## [0.5.3] - 2026-07-07

### Added

- `-l` / `--log` flag to `polya_calc`, `polya_merge`, `polya_gene`, `polya_dpc`,
  and `polya_dpt`. When enabled, poly(A) tail lengths are log-transformed
  (`log(L+1)`) before computing weighted means and medians, then
  back-transformed (`exp(result)-1`) to yield weighted geometric means/medians.
  In `polya_dpc` and `polya_dpt`, hypothesis tests (weighted KS, t-test,
  rank-sum) are run on the log-transformed data.

## [0.5.2] - 2026-07-06

### Added

- `scripts/pq2tsv.py` — general-purpose Parquet-to-TSV converter with optional
  gzip compression. Formatting conventions match `_io.write_tsv` (NA for null/NaN,
  scientific notation for tiny floats, comma-joined lists).
- `scripts/tsv2pq.py` — general-purpose TSV-to-Parquet converter. Auto-detects
  gzip-compressed input (`.tsv.gz`), infers column types via pyarrow, and handles
  header-only empty files.

## [0.5.1] - 2026-07-05

### Fixed

- Missing CHANGELOG update for 0.5.0.

## [0.5.0] - 2026-07-05

### Added

- Parquet output format support across modules.
- `float32` precision for read weights.
- Oarfish plain-text (uncompressed) input file support.
- `_io.py` — shared I/O utilities for reading/writing Parquet and TSV.
- `_hdf5_helpers.py` — shared HDF5 helper utilities.
- `polya_gene` module (renamed from `polya_t2g`) for transcript-to-gene aggregation.
- Weighted statistics in poly(A) output columns.
- Precise q-value reporting (not capped at significant-digit rounding).

### Changed

- `polya_t2g` renamed to `polya_gene` with simplified CLI args.
- Major refactor of `mod_dmc`, `mod_dmt`, `mod_dmcg`, `mod_gene`, `polya_bimodal`,
  `polya_dpc`, `polya_dpt` for improved efficiency and reduced memory usage.
- `mod_sites`, `mod_corr` refactored for code clarity and reuse.
- Poly(A) output columns renamed for consistency.
- `stats.py` merged into `_stats.py` (deduplication).
- Updated README documentation.

### Removed

- `polya_t2g` module (replaced by `polya_gene`).

### Fixed

- CI and lint issues (UP015, UP035).

## [0.4.0] - 2026-06-29

### Added

- `mod_gene` module — gene-level aggregation of transcript-level modification site
  summaries. Groups sites by `(gene_id, chrom, strand, gpos, mod_type)`, sums all
  count and weighted-count columns, and recomputes modification levels.
- `mod_dmc` module — differential modification calling between two experimental
  conditions. Matches sites by `(transcript_id, position, mod_type)`, pools
  reads from multiple HDF5 files per condition, and fits read-level weighted
  logistic regression per site. Reports log2 odds ratio, Wald p-value, and
  Benjamini-Hochberg FDR q-value with per-condition effect sizes.
- `mod_dmt` module — differential modification testing between transcript
  isoforms. Groups sites by genomic locus `(gene_id, chrom, gpos, strand,
  mod_type)`, enumerates isoform pairs, and fits weighted logistic regression
  per pair. Requires site summaries generated with `--gtf` for genomic
  coordinate mapping.
- `mod_dmcg` module — gene-level differential modification calling between two
  conditions. Takes gene-level site summaries from `mod_gene` as input and
  applies Fisher's exact test (both unweighted on integer counts and weighted on
  rounded `wt_modified` / `wt_unmodified`). No HDF5 or read-level data required.
- `polya_bimodal` module — bimodal poly(A) tail length detection via Gaussian
  mixture modeling and KDE peak-finding. Identifies transcripts with two
  distinct poly(A) length populations and reports per-mode statistics.
- `polya_dpc` module — compare poly(A) length distributions between two
  conditions using weighted mean difference and weighted two-sample KS test
  with Kish's effective sample size correction.
- `polya_dpt` module — differential poly(A) tail length testing with weighted
  t-test, Mann-Whitney U test, and Kolmogorov-Smirnov test backed by shared
  `stats.py`. Reports test statistics, p-values, and BH FDR q-values.
- `stats.py` — shared statistics backend for poly(A) differential analysis
  providing `WeightedTTest`, `WeightedMannWhitney`, and `WeightedKS` functions.
- `_stats` module — shared statistics backend providing `weighted_logistic_test`
  (closed-form weighted MLE for logistic regression with a single binary
  predictor, using Haldane-Anscombe correction and Wald test) and `bh_fdr`
  (Benjamini-Hochberg FDR correction). Used by `mod_dmc`, `mod_dmt`, and
  `mod_dmcg`.
- `_gtf.py` — shared GTF parsing utilities (`load_gtf`, `build_tx_to_gene`)
  with import guarding and progress logging.
- `--gtf` (`-g`) option to `mod_sites` for mapping transcript coordinates to
  genomic coordinates, adding `gene_id`, `chrom`, `strand`, and `gpos` columns
  to the output.
- `--gtf` / `--tx2gene` options to `polya_calc` and `polya_gene` for direct
  GTF-to-gene mapping.
- Multi-file HDF5 pooling support in `mod_sites` and `mod_corr`: `--h5` (`-i`)
  now accepts multiple files and pools reads for the same transcript across all
  files before computing statistics. Validates consistent modification codes
  and transcript lengths across files.
- `-k` short flag for `--kde-prominence` in `polya_bimodal`.

### Changed

- Replaced `polya_diff` with `polya_dpc` + `polya_dpt` backed by shared
  `stats.py`, providing richer statistical tests and shared infrastructure.
- Renamed TSV column `tx_name` to `transcript_id` in all poly(A) modules
  for consistency with the modification modules.
- Refactored GTF parsing from `polya_gene`, `polya_calc`, and `mod_sites` into
  shared `_gtf.py` module.
- `mod_corr` plot flag renamed from `-P`/`--plot` to `-d`/`--plot-dir` with
  added `-t`/`--metric` option to select the association statistic visualized
  in heatmaps (default: `wcorr`).
- `mod_corr` now computes both unweighted and weighted variants of all
  association metrics (Pearson r, p-value, q-value, mutual information, odds
  ratio).
- Updated README with comprehensive module documentation.

### Removed

- `polya_diff` module (replaced by `polya_dpc` + `polya_dpt`).
- Rust implementation draft (`rust/` directory).

### Fixed

- CI and lint issues.

## [0.3.0] - 2026-06-25

### Added

- Flexible LZ4 probability file and BAM file extractors in `_parsing.py`.

### Changed

- Revised alignment state classification to align with `modkit` conventions
  (canonical match, mismatch, deletion, modification).
- Improved efficiency and memory use in `mod_scan` and `mod_sites`.
- Improved code quality, test coverage, and cleanup across all modules.

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
- `polya_gene` module — aggregate transcript-level poly(A) estimates to gene
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
  `polya_merge`, `polya_diff`, and `polya_gene` for compressed TSV output.
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

[Unreleased]: https://github.com/gxelab/isolens/compare/v0.5.4...HEAD
[0.5.4]: https://github.com/gxelab/isolens/compare/v0.5.3...v0.5.4
[0.5.3]: https://github.com/gxelab/isolens/compare/v0.5.2...v0.5.3
[0.5.2]: https://github.com/gxelab/isolens/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/gxelab/isolens/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/gxelab/isolens/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/gxelab/isolens/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/gxelab/isolens/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/gxelab/isolens/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/gxelab/isolens/releases/tag/v0.1.0
