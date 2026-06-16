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

### code of modifications

from https://software-docs.nanoporetech.com/dorado/latest/basecaller/mods/:

| Mod    | Name                      | SAM Code   | CHEBI       |
|--------|---------------------------|------------|-------------|
| m5C    | 5-Methylcytosine          | C+m        | CHEBI:27551 |
| m6A    | N(6)-Methyladenosine      | A+a        | CHEBI:21891 |
| inosine| Inosine                   | A+17596    | CHEBI:17596 |
| pseU   | Pseudouridine             | T+17802    | CHEBI:17802 |
| 2OmeC  | 2'-O-methylcytidine       | C+19228    | CHEBI:19228 |
| 2OmeA  | 2'-O-methyladenosine      | A+69426    | CHEBI:69426 |
| 2OmeG  | 2'-O-methylguanosine      | G+19229    | CHEBI:19229 |
| 2OmeU  | 2'-O-methyluridine        | T+19227    | CHEBI:19227 |
