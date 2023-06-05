"""Integration tests.

Tests that the package reports reasonable values for certain interior-atmosphere systems. These are
quite rudimentary tests, but at least confirm that nothing fundamental is broken with the code.

"""

from atmodeller import __version__


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"
