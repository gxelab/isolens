# isolens

**Long-Read Kit** — a Python toolkit for analysis of long-read sequencing data.

[![PyPI - Version](https://img.shields.io/pypi/v/isolens)](https://pypi.org/project/isolens/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/isolens)](https://pypi.org/project/isolens/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/gxelab/isolens/actions/workflows/ci.yml/badge.svg)](https://github.com/gxelab/isolens/actions/workflows/ci.yml)

## Installation

```bash
pip install isolens
```

## Usage

### CLI (recommended)

```bash
# Generate HDF5 read × position matrices
python -m isolens.mod_scan \
  -b alignments.bam -a oarfish.lz4 -o mod_scan.h5 -c 0.95 -v

# Per-position modification summaries
python -m isolens.mod_sites -i mod_scan.h5 -o sites.parquet

# Poly(A) tail length estimation
python -m isolens.polya_calc \
  -b alignments.bam -a oarfish.lz4 -o polya.tsv.gz -z

# Merge poly(A) replicates
python -m isolens.polya_merge -i1 rep1.tsv.gz -i2 rep2.tsv.gz -o merged.tsv.gz

# Compare poly(A) distributions between conditions
python -m isolens.polya_diff -c1 ctrl.tsv.gz -c2 treat.tsv.gz -o diff.tsv

# Aggregate transcript-level poly(A) to gene level
python -m isolens.polya_t2g -i polya.tsv.gz -m tx2gene.tsv -o gene.tsv.gz
```

### Python API

```python
from isolens._parsing import parse_oarfish

tx_names, prob_map, name_to_id = parse_oarfish("oarfish.lz4")
```

## Development

Clone the repository and install in editable mode with the development extras:

```bash
git clone https://github.com/gxelab/isolens.git
cd isolens
pip install -e ".[dev]"
```

### Run without installing

When developing, you can run the tools directly from the source tree
without any installation:

```bash
# Using uv (auto-handles the src/ layout)
uv run python -m isolens.mod_scan -b ... -a ... -o ...

# Or set PYTHONPATH manually
PYTHONPATH=src python -m isolens.mod_scan -b ... -a ... -o ...
```

Run the test suite:

```bash
pytest
```

## Contributing

Contributions are welcome! Please open an issue or pull request on
[GitHub](https://github.com/gxelab/isolens).

## License

Distributed under the [MIT License](LICENSE).

## TODO
- [ ] mod_scan memory usage