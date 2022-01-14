"""
Unit and regression test for the snrv package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import snrv


def test_snrv_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "snrv" in sys.modules
