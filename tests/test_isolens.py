"""Tests for isolens package."""

import isolens


def test_version():
    """Package exposes a __version__ string."""
    assert isinstance(isolens.__version__, str)
    assert isolens.__version__ != ""


def test_version_format():
    """Version follows basic semver format (MAJOR.MINOR.PATCH)."""
    parts = isolens.__version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)
