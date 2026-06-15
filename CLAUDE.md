# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`isolens` (Long-Read Kit) is a bioinformatics toolkit for analyzing long-read RNA sequencing data — specifically transcript-level base modification detection and poly(A) tail length profiling. It joins Oxford Nanopore BAM alignments (with MM/ML modification tags and pt:i poly(A) tags from Dorado) against Oarfish read-to-transcript assignment probabilities (LZ4-compressed), producing per-position modification summaries and per-transcript poly(A) statistics.

The project has both a Python reference implementation and a high-performance Rust rewrite. The Rust build is the primary tool for production use; the Python version serves as a readable reference and prototyping surface.

## Build & development commands

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests (all)
pytest

# Run tests with coverage
pytest --cov=isolens --cov-report=term-missing

# Run a single test
pytest tests/test_isolens.py::test_version

# Lint
ruff check src tests

# Format
ruff format src tests

# Build distribution (sdist + wheel)
python -m build

# --- Rust ---
cd rust

cargo build --release   # production binary
cargo check             # fast compile-check (no binary)
cargo test              # run Rust tests
cargo fmt               # format Rust code
cargo clippy            # lint Rust code
```

## Running the tool

```bash
# Python version
python src/isolens/main.py -b examples/example.txmap.bam -p examples/example.lz4 \
  -m example.isolens.mod.tsv.gz -a example.isolens.pa.tsv.gz \
  --out-per-base example.isolens.modpb.tsv.gz -z -v

# Rust version
./rust/target/release/isolens -b examples/example.txmap.bam -p examples/example.lz4 \
  --out1 example.isolens.mod.tsv.gz --out2 example.isolens.pa.tsv.gz \
  --out-per-base example.isolens.modpb.tsv.gz -z -v
```

The `examples/` directory contains a small test dataset (subset of two _Drosophila_ transcripts) suitable for verifying changes.

## Architecture

### Dual implementation

Both implementations implement the same pipeline: parse Oarfish LZ4 → stream BAM → join reads to transcript assignments → produce two TSV outputs (positional modification summary + poly(A) tail statistics).

| | Python | Rust |
|---|---|---|
| Entry point | `src/isolens/main.py` | `rust/src/main.rs` |
| BAM parsing | `pysam` | `noodles-bam`/`noodles-sam`/`noodles-util` |
| LZ4 handling | `lz4.frame` | `lz4_flex` |
| Hashing | built-in `dict` | `ahash` |
| CLI | `argparse` | `lexopt` |
| CLI flags | `-b`, `-p`, `-m`, `-a`, `--out-per-base` | `-b`, `-p`, `--out1`, `--out2`, `--out-per-base` |

**Important**: The Python and Rust CLIs use different flag names for outputs (`-m`/`-a` vs `--out1`/`--out2`). The Rust version also has `-j/--threads` for parallel decompression.

**UUID handling**: The Python version converts read UUIDs to 128-bit integers (via `uuid.UUID(...).int`) to minimize RAM; the Rust version keeps them as strings. This is relevant when porting changes between implementations.

### Data structures (identical across both implementations)

- **`TargetAssignment`**: Maps a transcript ID (`tx_id`) to an assignment probability (`prob`).
- **`PositionStats`**: Per-transcript-position counters — `n_read`, `sum_probs`, `n_nomod`, `wt_nomod`, and a `mods` dict keyed by modification type string (e.g., "m", "h") with `[count, weighted_count]`.
- **`PolyAStats`**: Per-transcript poly(A) counters — `n_reads`, `sum_weights`, `pa_reads`, `pa_weights`, `sum_weighted_pa_len`, and raw `probs`/`pa_lens` vectors.

### Pipeline flow

1. **`parse_oarfish()`** — Reads the LZ4-compressed Oarfish file. First line = transcript count, next N lines = transcript names, remaining lines = per-read assignments (read UUID, target count, target indices, probabilities). See `notebooks/03_file_format.md` for the full format specification with examples.
2. **BAM iteration** — Streams through alignments. For each mapped read with a matching Oarfish assignment: (a) records poly(A) tail length from the `pt` tag; (b) builds a read-to-transcript position map via CIGAR parsing; (c) parses `MM`/`ML` tags to identify modified positions passing the probability threshold. The SAM/BAM format and tag specifications are in `docs/SAMv1.pdf` and `docs/SAMtags.pdf`.
3. **Output generation** — Writes two TSVs: (1) per-position modification counts/weights grouped by modification type; (2) per-transcript poly(A) statistics with raw probability and length vectors.

### Scripts (`scripts/`)

Standalone Python scripts for poly(A) analysis workflows, independent of the main pipeline:

- **`polya_calc.py`** — Extract per-transcript poly(A) lengths from a Dorado BAM + Oarfish assignments. Output: `tx_name, tx_idx, n_reads, pa_wlen, probs, pa_lens`.
- **`polya_merge.py`** — Merge poly(A) TSVs from two replicates, recomputing weighted average lengths.
- **`polya_diff.py`** — Compare poly(A) length distributions between two conditions using a weighted two-sample KS test (Kish's effective sample size). Requires `numpy` and `scipy` (not declared in project config).
- **`polya_t2g.py`** — Aggregate transcript-level poly(A) data to gene level using a `tx_name → gene_id` mapping file.

### Package metadata

- Build backend: `hatchling`, `src/` layout with `packages = ["src/isolens"]`
- Python ≥ 3.10, tested in CI on 3.12/3.13/3.14
- Target Python version for ruff: 3.14
- Publish triggered by `v*` tags via GitHub trusted publishing (OIDC) to both PyPI and TestPyPI

### Key external tools

- **Oarfish** — Transcript quantification tool that produces the `.prob` (or `.prob.lz4`) read assignment probability map. Format documented in `notebooks/03_file_format.md`.
- **Dorado** — Oxford Nanopore basecaller that emits BAM with `MM`/`ML` (modification) and `pt` (poly(A) tail length) tags.
- **minimap2** — Used to map reads to a transcriptome index before running isolens. Typical invocation: `minimap2 --eqx -N 100 -ax map-ont -y`.

### Reference documentation (`docs/`)

- **`docs/SAMv1.pdf`** — SAM/BAM format specification (CIGAR operators, flags, header structure, alignment fields).
- **`docs/SAMtags.pdf`** — Standard SAM tags reference, including `MM`/`ML` (base modification) and `pt` (poly(A) tail length) tags consumed by this pipeline.

## Roadmap: planned modules

`notebooks/01_mod.md` defines the specification for three future Python programs that will form the next phase of the toolkit. These operate on the same input data but produce richer outputs (HDF5 matrices, Parquet summaries, correlation statistics):

### `mod_scan.py`
Generate a single HDF5 file containing transcript-specific read × position matrices.
- **Inputs**: `--bam` (transcriptome BAM), `--oarfish` (LZ4 assignments), `--output` (HDF5), `--mod-cutoff` (default 0.95)
- **Output**: HDF5 with `/transcripts/<tx_id>/matrix` (uint8), `/read_ids`, `/read_weights` (float32), plus global `/modification_codes` and `/metadata`
- **Encoding**: 0=uncovered, 1=canonical match, 2=mismatch, 3=deletion, 4+=modification types
- **Storage**: gzip compression with shuffle, chunked rows (~512–4096 rows per chunk)

### `mod_sites.py`
Transcript-position level modification summaries from the HDF5 file.
- **Inputs**: `--h5 transcripts.h5`, `--output site_summary.parquet`
- **Per-position output columns**: `transcript_id, position, modification_type, n_modified, weighted_modified, n_unmodified, weighted_unmodified, n_mismatch, weighted_mismatch, n_deletion, weighted_deletion, modification_level, weighted_modification_level`
- Only canonical-match reads count as unmodified; mismatches and deletions are tracked separately

### `mod_corr.py`
Identify cooperative or antagonistic relationships between modification sites within the same transcript.
- **Inputs**: `--h5 transcripts.h5`, `--site-summary site_summary.parquet`, `--output correlations.parquet`
- **Methods**: weighted contingency tables, Phi coefficient, Fisher's exact test, odds ratio, mutual information
- **Multiple testing**: Benjamini-Hochberg FDR correction, per-transcript
- **Candidate filtering**: only sites with `n_modified ≥ minimum_support` (default 10)

### Performance constraints (applies to all three)

- Target scale: 100,000 transcripts, 10,000,000 reads
- Process one transcript at a time — never load all matrices simultaneously
- Use NumPy vectorization; avoid Python loops over matrix elements
- Store matrices as `uint8`, weights as `float32`
- Prefer Parquet for tabular outputs
- Design for future multi-process transcript-level parallelization

## Known issues

- `pyproject.toml` declares `dependencies = []`, but `src/isolens/main.py` requires `lz4` and `pysam` at runtime. These should be added.
- `scripts/polya_diff.py` requires `numpy` and `scipy`, which are not declared in any project dependency group.
- The CHANGELOG references `epitk` (a previous project name) rather than `isolens`.
- CLI flag names differ between the Python and Rust implementations.
- `.gitignore` lists `dist/` as ignored, but the `dist/` directory exists in the repo with built artifacts.
