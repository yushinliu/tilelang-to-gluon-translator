"""Packaging smoke tests for the public wheel-facing API."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import tilelang_to_gluon_translator as public_pkg
from src import (
    GluonKernelCache,
    TileLangGluonWrapper,
    TileLangToGluonTranslator,
    to_gluon,
)
from src.translator import main as translator_main
from tilelang_to_gluon_translator.cli import main as public_main


def test_public_package_re_exports_core_api():
    """The wheel-facing package should expose the same public API as src."""
    assert public_pkg.TileLangToGluonTranslator is TileLangToGluonTranslator
    assert public_pkg.to_gluon is to_gluon
    assert public_pkg.TileLangGluonWrapper is TileLangGluonWrapper
    assert public_pkg.GluonKernelCache is GluonKernelCache


def test_public_cli_forwards_to_translator_main():
    """The installed console entry point should use the existing translator CLI."""
    assert public_main is translator_main
