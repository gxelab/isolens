# Project Specification: Transcript-Level Modification Matrix Construction and Analysis

## Overview

Develop three Python programs:

1. `mod_scan.py`
2. `mod_sites.py`
3. `mod_corr.py`

Input data consist of:

* Transcriptome alignments produced by minimap2 (BAM format).
* Isoform assignment probabilities produced by oarfish (LZ4-compressed format).
* Modification annotations stored in BAM MM/ML tags.
* Transcriptome reference annotation.

The pipeline operates at the transcript level. Reads may be assigned probabilistically to multiple transcript isoforms. These assignment probabilities must be propagated throughout all downstream analyses as weights.

The objective is to construct transcript-specific read × position matrices, summarize modification frequencies, and identify associations between modification sites.

---

# General Requirements

## Read Assignment Weights

A read may be assigned to multiple transcript isoforms.

For each read:

```text
Σ assignment_probability = 1
```

Assignment probabilities must be interpreted as weights.

All weighted statistics must use these assignment probabilities.

Example:

```text
read_001
  transcript_A : 0.7
  transcript_B : 0.3
```

When processing transcript_A:

```text
read weight = 0.7
```

When processing transcript_B:

```text
read weight = 0.3
```

---

## Modification Probability Threshold

Default:

```text
--mod-cutoff 0.95
```

A modification call is considered present only if:

```text
ML_probability >= cutoff
```

Otherwise it is treated as unmodified.

The cutoff must be configurable.

---

## Supported Modification Types

Assume fewer than 10 modification classes.

Internally assign integer codes:

```text
0 = uncovered/not aligned
1 = canonical match
2 = mismatch
3 = deletion

4+ = modification types
```

Example:

```text
4 = m6A
5 = m5C
6 = pseudouridine
7 = m1C
8 = 2'-O-methyl
...
```

A global modification code table must be stored in the HDF5 file.

Use:

```python
numpy.uint8
```

for matrix storage.

No larger integer type should be used unless necessary.

---

## Coordinate System

All analyses must use transcript coordinates.

CIGAR operations must be projected onto transcript positions.

Insertions do not consume transcript positions.

Deletions consume transcript positions and should be represented explicitly.

---

# 1. mod_scan.py

## Goal

Generate a single HDF5 file containing transcript-specific read × position matrices.

---

## Inputs

```text
--bam           minimap2 transcriptome BAM
--oarfish       isoform assignment probabilities (lz4)
--output        output HDF5
--mod-cutoff    modification probability threshold
```

---

## Output Structure

Single HDF5 file:

```text
transcripts.h5
```

Structure:

```text
/transcripts

    /ENST000001
        matrix
        read_ids
        read_weights

    /ENST000002
        matrix
        read_ids
        read_weights

/modification_codes
/metadata
```

---

## Matrix Definition

For transcript T:

```python
matrix.shape = (
    n_reads_assigned_to_T,
    transcript_length
)
```

Datatype:

```python
uint8
```

Encoding:

```text
0 = uncovered
1 = canonical match
2 = mismatch
3 = deletion
4+ = modification types
```

Each row corresponds to one read assignment.

Each column corresponds to one transcript position.

---

## Read Weights

Store:

```python
read_weights.shape = (n_reads,)
dtype=float32
```

Weight equals oarfish assignment probability for that transcript.

Example:

```text
read_001 -> transcript_A weight=0.7
read_001 -> transcript_B weight=0.3
```

Both rows should exist independently in the corresponding transcript datasets.

---

## Storage Optimizations

Use HDF5 compression:

```python
compression="gzip"
shuffle=True
```

Recommended chunking:

```python
(chunk_rows, transcript_length)
```

where chunk_rows is approximately:

```text
512–4096
```

depending on transcript length.

Avoid Python object arrays.

Use contiguous NumPy arrays before writing.

---

## Alignment Parsing

For each alignment:

1. Parse CIGAR.
2. Project read bases onto transcript coordinates.
3. Populate matrix states.
4. Parse MM/ML tags.
5. Apply modification cutoff.
6. Override canonical state with modification state when modification passes threshold.
7. Record mismatches separately from modifications.
8. Record deletions explicitly.

---

## Metadata

Store:

```text
transcript_length
n_reads
modification_code_map
mod_cutoff
pipeline_version
```

---

# 2. mod_sites.py

## Goal

Generate transcript-position level modification summaries.

---

## Inputs

```text
--h5 transcripts.h5
--output site_summary.parquet
```

---

## Analysis

For every transcript position containing at least one modification call.

Modification types must be analyzed separately.

Example:

```text
ENST1
position=245
modification=m6A
```

---

## Counts

Calculate:

### Modified Reads

Unweighted:

```text
n_modified
```

Weighted:

```text
weighted_modified
```

Sum of read assignment probabilities.

---

### Unmodified Reads

Unmodified means:

```text
state == canonical match
```

Only.

Do NOT include:

```text
mismatch
deletion
uncovered
other modification types
```

Calculate:

```text
n_unmodified
weighted_unmodified
```

---

### Mismatch Statistics

Calculate separately:

```text
n_mismatch
weighted_mismatch
```

Mismatch reads must never be merged into the unmodified category.

---

### Deletion Statistics

Calculate separately:

```text
n_deletion
weighted_deletion
```

---

## Modification Level

Unweighted:

```text
ML = n_modified /
     (n_modified + n_unmodified)
```

Weighted:

```text
weighted_ML =
weighted_modified /
(weighted_modified + weighted_unmodified)
```

Do not include mismatch, deletion, or uncovered states in the denominator.

---

## Output Columns

Minimum:

```text
transcript_id
position
modification_type

n_modified
weighted_modified

n_unmodified
weighted_unmodified

n_mismatch
weighted_mismatch

n_deletion
weighted_deletion

modification_level
weighted_modification_level
```

Store as:

```text
Parquet
```

---

# 3. mod_corr.py

## Goal

Identify cooperative or antagonistic relationships between modification sites within the same transcript.

---

## Inputs

```text
--h5 transcripts.h5
--site-summary site_summary.parquet
--output correlations.parquet
```

---

## Candidate Sites

Only evaluate sites satisfying:

```text
n_modified >= minimum_support
```

Default:

```text
minimum_support = 10
```

Configurable.

---

## Pair Definition

For transcript:

```text
site_i
site_j
```

consider only reads covering both positions.

Reads with:

```text
uncovered
```

at either position are excluded.

---

## Binary Representation

For a given modification type:

```text
1 = modified
0 = canonical
```

Exclude:

```text
mismatch
deletion
other modifications
```

unless explicitly requested.

---

## Statistics

Calculate:

### Weighted Contingency Table

Using read assignment probabilities.

```text
n11
n10
n01
n00
```

and weighted equivalents.

---

### Phi Coefficient

Primary association statistic.

Compute both:

```text
phi
weighted_phi
```

---

### Fisher Exact Test

Unweighted significance:

```text
p_value
```

---

### Odds Ratio

```text
odds_ratio
```

Interpretation:

```text
odds_ratio > 1
    cooperative

odds_ratio < 1
    antagonistic
```

---

### Mutual Information

Optional but recommended:

```text
mutual_information
weighted_mutual_information
```

Useful for non-linear dependencies.

---

## Multiple Testing

Perform transcript-level correction:

```text
Benjamini-Hochberg FDR
```

Store:

```text
q_value
```

---

## Output Columns

```text
transcript_id

site1
site2

modification_type

n11
n10
n01
n00

weighted_n11
weighted_n10
weighted_n01
weighted_n00

phi
weighted_phi

odds_ratio

p_value
q_value

mutual_information
weighted_mutual_information
```

Store as Parquet.

---

# Performance Requirements

Target scale:

```text
100,000 transcripts
10,000,000 reads
```

Requirements:

* Process one transcript at a time.
* Never load all transcript matrices simultaneously.
* Use HDF5 chunked access.
* Use NumPy vectorization whenever possible.
* Avoid Python loops over matrix elements.
* Store matrices as uint8.
* Store weights as float32.
* Prefer Parquet for tabular outputs.
* Design all modules to support multi-process transcript-level parallelization in the future.
