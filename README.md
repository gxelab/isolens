# lrkit

**Long-Read Kit** — a Python toolkit for analysis of long-read sequencing data.

[![PyPI - Version](https://img.shields.io/pypi/v/lrkit)](https://pypi.org/project/lrkit/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lrkit)](https://pypi.org/project/lrkit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/mt1022/lrkit/actions/workflows/ci.yml/badge.svg)](https://github.com/mt1022/lrkit/actions/workflows/ci.yml)

## Installation

```bash
pip install lrkit
```

## Usage

```python
import lrkit

print(lrkit.__version__)
```

## Development

Clone the repository and install in editable mode with the development extras:

```bash
git clone https://github.com/mt1022/lrkit.git
cd lrkit
pip install -e ".[dev]"
```

Run the test suite:

```bash
pytest
```

## Contributing

Contributions are welcome! Please open an issue or pull request on
[GitHub](https://github.com/mt1022/lrkit).

## License

Distributed under the [MIT License](LICENSE).