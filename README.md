<p align="center">
  <img src="logo.png" alt="IsoLens logo" width="320">
</p>

<p align="center">
  <strong>Isoform-aware RNA modification and poly(A) tail analysis for Oxford Nanopore direct RNA sequencing.</strong>
</p>

# IsoLens

[![PyPI - Version](https://img.shields.io/pypi/v/isolens)](https://pypi.org/project/isolens/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/isolens)](https://pypi.org/project/isolens/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/gxelab/isolens/actions/workflows/ci.yml/badge.svg)](https://github.com/gxelab/isolens/actions/workflows/ci.yml)

**IsoLens** is a Python toolkit for isoform-aware analysis of RNA modifications and poly(A) tail lengths from Oxford Nanopore direct RNA sequencing data, explicitly accounting for transcript assignment uncertainty to enable accurate transcript-level profiling.

---

## Why IsoLens?

Most long-read RNA analysis tools either:

- analyze RNA modifications or poly(A) tails without isoform uncertainty
- assign reads to transcripts using hard labels

IsoLens propagates transcript assignment probabilities from [Oarfish](https://github.com/COMBINE-lab/oarfish) throughout both modification and poly(A) analyses, enabling more accurate transcript-level estimates for genes with complex isoform structure.

Key capabilities:

- Isoform-aware RNA modification profiling at single-nucleotide resolution
- Transcript-level poly(A) tail length estimation with uncertainty propagation
- Modification site co-occurrence and correlation analysis
- Differential poly(A) testing between conditions
- Efficient HDF5 and Parquet outputs for large-scale studies
- Direct integration with Dorado BAM tags and Oarfish assignments

---

## Quick Start

Install:

```bash
pip install isolens
```

Build transcript-level modification matrices:

```bash
python -m isolens.mod_scan -b alignments.bam -a oarfish.lz4 -o mod_scan.h5
```

Summarize modification sites:

```bash
python -m isolens.mod_sites -i mod_scan.h5 -o sites.parquet
```

Estimate transcript-level poly(A) lengths:

```bash
python -m isolens.polya_calc -a oarfish.lz4 -b reads.bam -o polya.tsv.gz -z
```

---

## Contents

- [Pipeline Overview](#pipeline-overview)
- [Installation](#installation)
- [Modules](#modules)
- [Python API](#python-api)
- [Input Data Requirements](#input-data-requirements)
- [Development](#development)
- [Example Data](#example-data)
- [License](#license)

---

## Pipeline Overview

```
Oarfish (.lz4) ─────┐
                     ├── mod_scan ──► HDF5 ──┬── mod_sites ──► site summary (.parquet/.tsv)
Dorado BAM ─────────┘                        │
                                             └── mod_corr ──► correlations (.parquet/.tsv)
                                                                  + PDF heatmaps (─P)

Oarfish (.lz4) ─────┐
                     ├── polya_calc ──► polya TSV ──┬── polya_merge ──► merged TSV
Minimap2 BAM ───────┘                               │
                                                    ├── polya_diff ──► diff TSV
                                                    └── polya_t2g ───► gene-level TSV
```

### Outputs

| Command | Output |
|----------|----------|
| `mod_scan` | HDF5 read × position modification matrices |
| `mod_sites` | Per-site modification summaries |
| `mod_corr` | Pairwise modification site correlations |
| `polya_calc` | Transcript-level poly(A) estimates |
| `polya_merge` | Merged replicate poly(A) estimates |
| `polya_diff` | Differential poly(A) comparison |
| `polya_t2g` | Gene-level poly(A) summaries |

---

## Installation

```bash
pip install isolens
```

---

## Modules

### `mod_scan` — HDF5 read × position matrices

Generates a single HDF5 file containing transcript-specific read × position modification matrices. For each transcript, it builds a `(n_reads × tx_length)` uint8 matrix encoding the alignment state at every position.

**Encoding:** 0 = uncovered, 1 = canonical match, 2 = mismatch, 3 = deletion, 4+ = modification types (configurable).

```bash
python -m isolens.mod_scan \
  -b alignments.bam \
  -a oarfish.lz4 \
  -o mod_scan.h5 \
  -c 0.95 \
  -v
```

Key options:

| Flag | Description | Default |
|------|-------------|---------|
| `-b, --bam` | Transcriptome BAM alignment | (required) |
| `-a, --oarfish` | Oarfish assignment probability file (`.lz4`) | (required) |
| `-o, --output` | Output HDF5 path | (required) |
| `-c, --mod-cutoff` | Modification probability threshold | 0.95 |
| `-m, --mod-type` | Modification types to scan for (SAM codes) | `a,m,17596,17802,19228,69426,19229,19227` |
| `-p, --min-asp` | Minimum assignment probability filter | 0.0 |
| `-d, --max-depth` | Max reads per transcript | 5000 |
| `-t, --threads` | Worker processes for parallel processing | 1 |
| `-v, --verbose` | Print progress to stderr | off |

---

### `mod_sites` — Per-position modification summaries

Reads the HDF5 from `mod_scan` and produces a Parquet or TSV file with one row per `(transcript, position, modification type)`. Computes modification levels and tracks mismatches and deletions separately.

```bash
python -m isolens.mod_sites \
  -i mod_scan.h5 \
  -o sites.parquet
```

Key options:

| Flag | Description | Default |
|------|-------------|---------|
| `-i, --h5` | Input HDF5 from `mod_scan` | (required) |
| `-o, --output` | Output file | (required) |
| `-f, --format` | Output format: `parquet` or `tsv` | `parquet` |
| `-z, --gzip` | Gzip-compress TSV output | off |
| `-s, --sites` | Predefined modification sites TSV (`tx_name`, `posn`) | all sites |
| `-p, --min-asp` | Minimum assignment probability filter | 0.0 |
| `-x, --transcripts` | Only process specified transcript IDs | all |
| `-v, --verbose` | Print progress | off |

**Output columns:** `transcript_id`, `position`, `modification_type`, `n_modified`, `weighted_modified`, `n_unmodified`, `weighted_unmodified`, `n_mismatch`, `weighted_mismatch`, `n_deletion`, `weighted_deletion`, `modification_level`, `weighted_modification_level`.

---

### `mod_corr` — Pairwise modification site correlation

Identifies cooperative or antagonistic relationships between modification sites within the same transcript. Computes both within-type and cross-type correlations using weighted contingency tables.

**Metrics:** Phi coefficient, odds ratio, Fisher's exact test p-value, Benjamini-Hochberg FDR q-value, mutual information.

```bash
python -m isolens.mod_corr \
  -i mod_scan.h5 \
  -s sites.parquet \
  -o correlations.parquet \
  -m 10
```

Key options:

| Flag | Description | Default |
|------|-------------|---------|
| `-i, --h5` | Input HDF5 from `mod_scan` | (required) |
| `-s, --sites` | Site summary from `mod_sites` | (required) |
| `-o, --output` | Output file | (required) |
| `-m, --min-support` | Minimum `n_modified` for a site to be considered | 10 |
| `-p, --min-asp` | Minimum assignment probability filter | 0.0 |
| `-f, --format` | Output format: `parquet` or `tsv` | `parquet` |
| `-P, --plot` | Generate PDF heatmap per transcript in this directory | off |
| `-x, --transcripts` | Only process specified transcript IDs | all |
| `-v, --verbose` | Print progress | off |

**Output columns:** `transcript_id`, `site1`, `site2`, `modification_type`, `n11`, `n10`, `n01`, `n00` (and weighted variants), `phi`, `weighted_phi`, `odds_ratio`, `p_value`, `q_value`, `mutual_information`, `weighted_mutual_information`.

When `-P` is used, generates rotated triangular heatmap PDFs per transcript showing the correlation matrix and site positions along the transcript body.

---

### `polya_calc` — Poly(A) tail length estimation

Extracts poly(A) tail lengths from Dorado's `pt:i` BAM tags, weighted by Oarfish assignment probabilities.

```bash
python -m isolens.polya_calc \
  -a oarfish.lz4 \
  -b reads.bam \
  -o polya.tsv.gz \
  -z
```

**Output columns:** `tx_name`, `tx_idx`, `n_reads`, `pa_wlen` (weighted mean), `probs` (comma-separated), `pa_lens` (comma-separated).

---

### `polya_merge` — Merge poly(A) replicates

Combines two poly(A) TSV files from separate replicates, recomputing weighted average tail lengths from the pooled per-read data.

```bash
python -m isolens.polya_merge \
  -i1 rep1.tsv.gz \
  -i2 rep2.tsv.gz \
  -o merged.tsv.gz
```

---

### `polya_diff` — Differential poly(A) comparison

Compares poly(A) length distributions between two conditions using a weighted two-sample Kolmogorov-Smirnov test with Kish's effective sample size correction.

```bash
python -m isolens.polya_diff \
  -c1 control.tsv.gz \
  -c2 treatment.tsv.gz \
  -o diff.tsv
```

**Output columns:** `tx_name` (or `gene_id`), `n_reads_1`, `pa_wlen_1`, `n_reads_2`, `pa_wlen_2`, `stat` (KS statistic), `p_value`.

---

### `polya_t2g` — Transcript-to-gene aggregation

Aggregates transcript-level poly(A) estimates to the gene level using a user-provided `tx_name → gene_id` mapping file.

```bash
python -m isolens.polya_t2g \
  -i polya.tsv.gz \
  -m tx2gene.tsv \
  -o gene_polya.tsv.gz
```

---

## Python API

The core data structures and parsing functions are available for programmatic use:

```python
from isolens._parsing import parse_oarfish

tx_names, prob_map, name_to_id = parse_oarfish("assignments.lz4")

# tx_names: list[str]
# prob_map: dict[int, list[TargetAssignment]]
# name_to_id: dict[str, int]
```

---

## Input Data Requirements

| File | Source | Required tags / format |
|--------|--------|----------------------|
| Transcriptome BAM | minimap2 + Dorado | `MM`/`ML` (base modifications), `pt:i` (poly(A) tail length) |
| Oarfish assignments | Oarfish | LZ4-compressed read-to-transcript probability map |

The BAM should be coordinate-sorted and aligned to a transcriptome reference.

Typical preprocessing:

```bash
minimap2 --eqx -N 100 -ax map-ont -y transcriptome.fa reads.fastq \
  | samtools sort -o alignments.bam

samtools index alignments.bam
```

---

## Development

```bash
git clone https://github.com/gxelab/isolens.git
cd isolens
pip install -e ".[dev]"
```

Run without installing:

```bash
uv run python -m isolens.mod_scan \
  -b ... \
  -a ... \
  -o ...
```

Run tests:

```bash
pytest
```

Lint and format:

```bash
ruff check src tests
ruff format src tests
```

---

## Example Data

The `examples/` directory contains a small test dataset (subset of two *Drosophila* transcripts) suitable for verifying changes.

```bash
python -m isolens.mod_scan \
  -b examples/example.txmap.bam \
  -a examples/example.lz4 \
  -o example.mod_scan.h5 \
  -c 0.95 -v

python -m isolens.mod_sites \
  -i example.mod_scan.h5 \
  -o example.sites.parquet

python -m isolens.polya_calc \
  -a examples/example.lz4 \
  -b examples/example.txmap.bam \
  -o example.polya.tsv.gz -z
```

---

## License

Distributed under the [MIT License](LICENSE).