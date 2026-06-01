"""Smoke tests: prove the package imports and exposes a valid version.

No business logic — establishes a green baseline for TDD.
"""

import re

import micha7


def test_package_imports():
    """Top-level package imports without error."""
    assert micha7 is not None


def test_version_is_valid_semver():
    """__version__ exists and is MAJOR.MINOR.PATCH."""
    assert hasattr(micha7, "__version__")
    assert re.fullmatch(r"\d+\.\d+\.\d+", micha7.__version__)
