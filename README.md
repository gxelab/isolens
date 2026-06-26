<p align="center">
  <img src="logo.png" alt="IsoLens logo" width="320">
</p>

<p align="center">
  <strong>Isoform-aware RNA modification and poly(A) tail analysis for direct RNA sequencing data.</strong>
</p>

# IsoLens

[![PyPI - Version](https://img.shields.io/pypi/v/isolens)](https://pypi.org/project/isolens/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/isolens)](https://pypi.org/project/isolens/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/gxelab/isolens/actions/workflows/ci.yml/badge.svg)](https://github.com/gxelab/isolens/actions/workflows/ci.yml)

**IsoLens** is a Python toolkit for isoform-aware analysis of RNA modifications and poly(A) tail lengths from Oxford Nanopore direct RNA sequencing data, explicitly accounting for transcript assignment uncertainty to enable accurate transcript-level profiling.

---

## Why IsoLens?

Most long-read RNA analysis tools either analyze RNA modifications or poly(A) tails without discrimination of transcript isoforms or assign reads to transcripts using hard labels. IsoLens propagates transcript assignment probabilities from [Oarfish](https://github.com/COMBINE-lab/oarfish) throughout both modification and poly(A) analyses, enabling more accurate transcript-level estimates for genes with complex isoform structure.

Key capabilities:

- Isoform-aware RNA modification profiling at single-nucleotide resolution
- Transcript-level poly(A) tail length estimation with uncertainty propagation
- Modification site co-occurrence and correlation analysis
- Differential modification testing between conditions, isoforms, and genes
- Gene-level aggregation of transcript-level modification and poly(A) data
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
Dorado BAM ─────────┘                        │                  │
                                             │                  ├── mod_gene ──► gene-level summary (.parquet/.tsv)
                                             │                  │
                                             │                  ├── mod_corr ──► correlations (.parquet/.tsv)
                                             │                  │                  + PDF heatmaps (-d)
                                             │                  │
                                             │                  ├── mod_dmc ───► differential modification
                                             │                  │                  (condition comparison)
                                             │                  │
                                             │                  ├── mod_dmt ───► differential modification
                                             │                  │                  (isoform comparison)
                                             │                  │
                                             │                  └── mod_dmcg ──► gene-level differential
                                             │                                     modification
                                             │
Oarfish (.lz4) ─────┐                              │
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
| `mod_gene` | Gene-level modification summaries |
| `mod_corr` | Pairwise modification site correlations |
| `mod_dmc` | Differential modification between conditions |
| `mod_dmt` | Differential modification between isoforms |
| `mod_dmcg` | Gene-level differential modification |
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

Generates a single HDF5 file containing transcript-specific read-by-position modification matrices. For each transcript, IsoLens constructs an `(n_reads × transcript_length)` `uint8` matrix that encodes the nucleotide state at every position for every aligned read. Nucleotide states are parsed using logic consistent with that implemented in [modkit](https://github.com/nanoporetech/modkit), ensuring compatibility with standard Oxford Nanopore modification annotations.

**Encoding:** 0 = uncovered, 1 = canonical match, 2 = mismatch, 3 = deletion, 4+ = tracked modification types, 254 = untracked modification, 255 = failed (all states below probability threshold).

```bash
python -m isolens.mod_scan \
  -b alignments.bam \
  -a oarfish.lz4 \
  -o mod_scan.h5 \
  -c 0.95 \
  -t 4 -v
```

Key options:

| Flag | Description | Default |
|------|-------------|---------|
| `-b, --bam` | Transcriptome BAM alignment | (required) |
| `-a, --oarfish` | Oarfish assignment probability file (`.lz4`) | (required) |
| `-o, --output` | Output HDF5 path | (required) |
| `-c, --mod-cutoff` | Modification probability threshold | 0.95 |
| `-m, --mod-type` | Modification types to scan for (SAM code suffixes) | `a,m,17596,17802,19228,69426,19229,19227` |
| `-p, --min-asp` | Minimum assignment probability filter | 0.0 |
| `-d, --max-depth` | Max reads per transcript | 5000 |
| `-t, --threads` | Worker threads for parallel processing | 1 |
| `-v, --verbose` | Print progress to stderr | off |

**HDF5 structure:** `/transcripts/<tx_name>/matrix` (uint8), `read_ids` (string), `read_weights` (float32); `/modification_codes` (attrs); `/metadata` (attrs).

---

### `mod_sites` — Per-position modification summaries

Reads the HDF5 output generated by `mod_scan` and summarizes modification information into a Parquet or TSV file, with one row per `(transcript, position, modification_type)`. For each site, IsoLens reports modification levels, read coverage, and modification counts, while tracking mismatches, deletions, other-modifications, and failed calls as separate categories.

When the same modification probability threshold is used (`--mod-cutoff` in `mod_scan` and `--filter-threshold` in modkit), the combination of `mod_scan` and `mod_sites` produces unweighted modification counts that match those generated by `modkit pileup`.

Multiple HDF5 files can be provided — reads for the same transcript are pooled across all files before computing statistics.

```bash
python -m isolens.mod_sites \
  -i mod_scan.h5 \
  -o sites.parquet

# With genomic coordinate mapping
python -m isolens.mod_sites \
  -i mod_scan.h5 \
  -o sites.parquet \
  -g annotations.gtf
```

Key options:

| Flag | Description | Default |
|------|-------------|---------|
| `-i, --h5` | Input HDF5 file(s) from `mod_scan` (accepts multiple) | (required) |
| `-o, --output` | Output file | (required) |
| `-f, --format` | Output format: `parquet` or `tsv` | `parquet` |
| `-z, --gzip` | Gzip-compress TSV output | off |
| `-s, --sites` | Predefined modification sites TSV (`tx_name`, `posn`) | all sites |
| `-p, --min-asp` | Minimum assignment probability filter | 0.0 |
| `-x, --transcripts` | Only process specified transcript IDs | all |
| `-g, --gtf` | GTF annotation for genomic coordinate mapping | off |
| `-v, --verbose` | Print progress | off |

**Output columns (23):** `transcript_id`, `position`, `mod_type`, `n_modified`, `wt_modified`, `n_unmodified`, `wt_unmodified`, `n_canonical`, `wt_canonical`, `n_othermod`, `wt_othermod`, `n_mismatch`, `wt_mismatch`, `n_deletion`, `wt_deletion`, `n_failed`, `wt_failed`, `mod_level`, `wt_mod_level`, `gene_id`\*, `chrom`\*, `strand`\*, `gpos`\* (\*requires `--gtf`).

---

### `mod_corr` — Pairwise modification site correlation

Identifies cooperative or antagonistic relationships between modification sites within the same transcript. Computes both within-type and cross-type correlations using weighted 2×2 contingency tables. Multiple HDF5 files can be provided — reads for the same transcript are pooled across all files.

**Metrics:** Phi coefficient (Pearson's r for binary variables), odds ratio with Haldane-Anscombe correction, p-value via t-distribution, Benjamini-Hochberg FDR q-value (per-transcript), and mutual information. Both unweighted and assignment-probability-weighted variants are computed for every metric.

```bash
python -m isolens.mod_corr \
  -i mod_scan.h5 \
  -s sites.parquet \
  -o correlations.parquet \
  -m 10 -l 0.05 -c 10
```

Key options:

| Flag | Description | Default |
|------|-------------|---------|
| `-i, --h5` | Input HDF5 file(s) from `mod_scan` (accepts multiple) | (required) |
| `-s, --sites` | Site summary from `mod_sites` (Parquet or TSV/TSV.GZ) | (required) |
| `-o, --output` | Output file | (required) |
| `-m, --min-mod-reads` | Minimum `n_modified` for a site to be considered | 2 |
| `-l, --min-mod-level` | Minimum `mod_level` for a site to be considered | 0.05 |
| `-c, --min-coverage` | Minimum total depth for a site to be considered | 10 |
| `-p, --min-asp` | Minimum assignment probability filter | 0.0 |
| `-f, --format` | Output format: `parquet` or `tsv` | `parquet` |
| `-z, --gzip` | Gzip-compress TSV output | off |
| `-d, --plot-dir` | Generate pyramid heatmap PDFs per transcript in this directory | off |
| `-t, --metric` | Statistic to visualize in heatmaps (`corr`, `wcorr`, `mi`, `wmi`, `or`, `wor`) | `wcorr` |
| `-x, --transcripts` | Only process specified transcript IDs | all |
| `-v, --verbose` | Print progress | off |

**Output columns (23):** `transcript_id`, `site1`, `site2`, `mod_type1`, `mod_type2`, `n11`, `n10`, `n01`, `n00` (2×2 contingency counts), `w11`, `w10`, `w01`, `w00` (weighted), `corr`, `pvalue`, `qvalue` (unweighted Pearson + BH FDR), `wcorr`, `wpvalue`, `wqvalue` (weighted Pearson + BH FDR), `mi`, `wmi` (mutual information), `or`, `wor` (log2 odds ratio).

When `-d` is used, generates rotated triangular heatmap PDFs per transcript showing the correlation matrix and site positions along the transcript body.

---

### `mod_gene` — Gene-level modification aggregation

Aggregates transcript-level modification site summaries to the gene level by summing per-position counts grouped by `(gene_id, chrom, strand, gpos, mod_type)`. Requires the site summary to have been generated with `--gtf` so that genomic coordinate columns are present.

```bash
python -m isolens.mod_gene \
  -i sites.parquet \
  -o gene_sites.parquet
```

Key options:

| Flag | Description | Default |
|------|-------------|---------|
| `-i, --input` | Site summary from `mod_sites` (must have GTF columns) | (required) |
| `-o, --output` | Output file | (required) |
| `-f, --format` | Output format: `parquet` or `tsv` | `parquet` |
| `-z, --gzip` | Gzip-compress TSV output | off |
| `-v, --verbose` | Print progress | off |

**Output columns:** `gene_id`, `chrom`, `strand`, `gpos`, `mod_type`, and all per-position count/weight columns summed across transcripts. `mod_level` and `wt_mod_level` are recomputed from the summed counts.

---

### `mod_dmc` — Differential modification between conditions

Compares modification levels between two experimental conditions at each `(transcript, position, mod_type)` site using read-level weighted logistic regression. Reads from multiple HDF5 files are pooled within each condition before testing.

```bash
python -m isolens.mod_dmc \
  --h5-1 cond1_rep1.h5 cond1_rep2.h5 \
  --h5-2 cond2_rep1.h5 cond2_rep2.h5 \
  --sites-1 cond1_sites.parquet \
  --sites-2 cond2_sites.parquet \
  -o dmc_results.parquet -v
```

Key options:

| Flag | Description | Default |
|------|-------------|---------|
| `--h5-1` | HDF5 file(s) for condition 1 | (required) |
| `--h5-2` | HDF5 file(s) for condition 2 | (required) |
| `--sites-1` | Site summary for condition 1 | (required) |
| `--sites-2` | Site summary for condition 2 | (required) |
| `-o, --output` | Output file | (required) |
| `-f, --format` | Output format: `parquet` or `tsv` | `parquet` |
| `-z, --gzip` | Gzip-compress TSV output | off |
| `-p, --min-asp` | Minimum assignment probability filter | 0.0 |
| `-x, --transcripts` | Only process specified transcript IDs | all |
| `-v, --verbose` | Print progress | off |

**Output columns (25):** `transcript_id`, `position`, `mod_type`, `gene_id`, `chrom`, `strand`, `gpos`, `n_modified_1`, `n_unmodified_1`, `n_modified_2`, `n_unmodified_2`, `wt_modified_1`, `wt_unmodified_1`, `wt_modified_2`, `wt_unmodified_2`, `mod_level_1`, `mod_level_2`, `wt_mod_level_1`, `wt_mod_level_2`, `delta_mod_level`, `delta_wt_mod_level`, `log2_or`, `p_value`, `q_value` (BH FDR).

**Method:** Weighted logistic regression with Haldane-Anscombe correction for zero counts. Wald test p-values with global Benjamini-Hochberg FDR correction.

---

### `mod_dmt` — Differential modification between isoforms

Compares modification levels between transcript isoforms that share a genomic locus, using read-level weighted logistic regression. Transcripts are grouped by `(gene_id, chrom, gpos, strand, mod_type)` and all isoform pairs within each group are tested.

```bash
python -m isolens.mod_dmt \
  -i pooled.h5 \
  -s sites_with_gtf.parquet \
  -o dmt_results.parquet -v
```

Key options:

| Flag | Description | Default |
|------|-------------|---------|
| `-i, --h5` | Input HDF5 file(s) from `mod_scan` | (required) |
| `-s, --sites` | Site summary from `mod_sites` (must have `--gtf` columns) | (required) |
| `-o, --output` | Output file | (required) |
| `-f, --format` | Output format: `parquet` or `tsv` | `parquet` |
| `-z, --gzip` | Gzip-compress TSV output | off |
| `-p, --min-asp` | Minimum assignment probability filter | 0.0 |
| `-x, --transcripts` | Only consider specified transcript IDs | all |
| `-v, --verbose` | Print progress | off |

**Output columns (25):** `gene_id`, `chrom`, `gpos`, `strand`, `mod_type`, `transcript_id_1`, `transcript_id_2`, `position_1`, `position_2`, `mod_level_1`, `mod_level_2`, `wt_mod_level_1`, `wt_mod_level_2`, `delta_mod_level`, `delta_wt_mod_level`, `n_modified_1`, `n_unmodified_1`, `n_modified_2`, `n_unmodified_2`, `wt_modified_1`, `wt_unmodified_1`, `wt_modified_2`, `wt_unmodified_2`, `log2_or`, `p_value`, `q_value` (BH FDR).

**Method:** Same weighted logistic regression backend as `mod_dmc`. Transcripts are pre-loaded from HDF5 for efficient paired testing. Global BH FDR correction.

---

### `mod_dmcg` — Gene-level differential modification

Compares modification levels between two conditions at the gene level using Fisher's exact test. Takes gene-level site summaries from `mod_gene` as input — no HDF5 or read-level data required.

```bash
python -m isolens.mod_dmcg \
  --sites-1 cond1_genes.parquet \
  --sites-2 cond2_genes.parquet \
  -o dmcg_results.parquet -v
```

Key options:

| Flag | Description | Default |
|------|-------------|---------|
| `--sites-1` | Gene-level summary for condition 1 (from `mod_gene`) | (required) |
| `--sites-2` | Gene-level summary for condition 2 (from `mod_gene`) | (required) |
| `-o, --output` | Output file | (required) |
| `-f, --format` | Output format: `parquet` or `tsv` | `parquet` |
| `-z, --gzip` | Gzip-compress TSV output | off |
| `-v, --verbose` | Print progress | off |

**Output columns (25):** `gene_id`, `chrom`, `strand`, `gpos`, `mod_type`, `n_modified_1`, `n_unmodified_1`, `n_modified_2`, `n_unmodified_2`, `wt_modified_1`, `wt_unmodified_1`, `wt_modified_2`, `wt_unmodified_2`, `mod_level_1`, `mod_level_2`, `wt_mod_level_1`, `wt_mod_level_2`, `delta_mod_level`, `delta_wt_mod_level`, `log2_or`, `p_value`, `q_value` (unweighted Fisher + BH FDR), `w_log2_or`, `w_p_value`, `w_q_value` (weighted Fisher with rounded counts + BH FDR).

**Method:** Two Fisher's exact tests per matched gene-position — one on raw integer counts, one on `wt_modified` / `wt_unmodified` rounded to the nearest integer. Global BH FDR correction applied separately to each set of p-values.

---

### `polya_calc` — Poly(A) tail length estimation

Extracts poly(A) tail lengths from Dorado's `pt:i` BAM tags, weighted by Oarfish assignment probabilities.

```bash
python -m isolens.polya_calc \
  -a oarfish.lz4 \
  -b reads.bam \
  -o polya.tsv.gz -z
```

Key options:

| Flag | Description | Default |
|------|-------------|---------|
| `-a, --oarfish` | Oarfish assignment probability file (`.lz4`) | (required) |
| `-b, --bam` | BAM file with `pt:i` poly(A) tags (from Dorado) | (required) |
| `-o, --output` | Output TSV file | (required) |
| `-z, --gzip` | Gzip-compress output | off |

**Output columns (6):** `tx_name`, `tx_idx`, `n_reads`, `pa_wlen` (assignment-probability-weighted mean poly(A) length), `probs` (comma-separated assignment probabilities), `pa_lens` (comma-separated raw poly(A) lengths).

---

### `polya_merge` — Merge poly(A) replicates

Combines two poly(A) TSV files from separate replicates, recomputing weighted average tail lengths from the pooled per-read data.

```bash
python -m isolens.polya_merge \
  -i1 rep1.tsv.gz \
  -i2 rep2.tsv.gz \
  -o merged.tsv.gz -z
```

Key options:

| Flag | Description | Default |
|------|-------------|---------|
| `-i1, --input1` | First input TSV file (gzipped or raw) | (required) |
| `-i2, --input2` | Second input TSV file (gzipped or raw) | (required) |
| `-o, --output` | Output TSV file | (required) |
| `-z, --gzip` | Gzip-compress output | off |

**Output columns (6):** Same schema as `polya_calc` — `tx_name`, `tx_idx`, `n_reads`, `pa_wlen`, `probs`, `pa_lens`. Per-transcript probability and length lists from both files are concatenated before recalculating `pa_wlen`.

---

### `polya_diff` — Differential poly(A) comparison

Compares poly(A) length distributions between two conditions using a weighted two-sample Kolmogorov-Smirnov test with Kish's effective sample size correction.

```bash
python -m isolens.polya_diff \
  -c1 control.tsv.gz \
  -c2 treatment.tsv.gz \
  -o diff.tsv.gz -z
```

Key options:

| Flag | Description | Default |
|------|-------------|---------|
| `-c1, --condition1` | Condition 1 TSV/TSV.GZ file | (required) |
| `-c2, --condition2` | Condition 2 TSV/TSV.GZ file | (required) |
| `-o, --output` | Output TSV file | (required) |
| `-z, --gzip` | Gzip-compress output | off |

**Output columns (7):** `<feature_id>`, `n_reads_1`, `pa_wlen_1`, `n_reads_2`, `pa_wlen_2`, `stat` (weighted KS statistic), `p_value`. The feature ID column header is `tx_name` for transcript-level input, `gene_id` for gene-level input, or `feature_id` if the two files disagree.

---

### `polya_t2g` — Transcript-to-gene aggregation

Aggregates transcript-level poly(A) estimates to the gene level using a user-provided `tx_name → gene_id` mapping file. Per-transcript probability and length lists are pooled before recalculating the weighted average.

```bash
python -m isolens.polya_t2g \
  -i polya.tsv.gz \
  -m tx2gene.tsv \
  -o gene_polya.tsv.gz -z
```

Key options:

| Flag | Description | Default |
|------|-------------|---------|
| `-i, --input` | Input transcript poly(A) TSV file (gzipped or raw) | (required) |
| `-m, --map` | Mapping file with `tx_name` and `gene_id` columns (gzipped or raw TSV) | (required) |
| `-o, --output` | Output gene-level TSV file | (required) |
| `-z, --gzip` | Gzip-compress output | off |

**Output columns (5):** `gene_id`, `n_reads`, `pa_wlen` (recalculated weighted mean), `probs` (comma-separated pooled probabilities), `pa_lens` (comma-separated pooled lengths).

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