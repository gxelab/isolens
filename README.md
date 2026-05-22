# isolens

**Long-Read Kit** — a Python toolkit for analysis of long-read sequencing data.

[![PyPI - Version](https://img.shields.io/pypi/v/isolens)](https://pypi.org/project/isolens/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/isolens)](https://pypi.org/project/isolens/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/mt1022/isolens/actions/workflows/ci.yml/badge.svg)](https://github.com/mt1022/isolens/actions/workflows/ci.yml)

## Installation

```bash
pip install isolens
```

## Usage

```python
import isolens

print(isolens.__version__)
```

## Development

Clone the repository and install in editable mode with the development extras:

```bash
git clone https://github.com/mt1022/isolens.git
cd isolens
pip install -e ".[dev]"
```

Run the test suite:

```bash
pytest
```

## Contributing

Contributions are welcome! Please open an issue or pull request on
[GitHub](https://github.com/mt1022/isolens).

## License

Distributed under the [MIT License](LICENSE).