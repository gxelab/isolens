"""Tests for lrkit package."""

import lrkit


def test_version():
    """Package exposes a __version__ string."""
    assert isinstance(lrkit.__version__, str)
    assert lrkit.__version__ != ""


def test_version_format():
    """Version follows basic semver format (MAJOR.MINOR.PATCH)."""
    parts = lrkit.__version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)