"""Packaging smoke tests for the public wheel-facing API."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import tilelang_to_gluon_translator as public_pkg
from tilelang_to_gluon_translator.decorator import (
    GluonKernelCache,
    TileLangGluonWrapper,
    to_gluon,
)
from tilelang_to_gluon_translator.cli import main as public_main
from tilelang_to_gluon_translator.translator import (
    TileLangToGluonTranslator,
    main as translator_main,
)


def test_public_modules_do_not_leak_src_namespace():
    """Public objects should be defined under the public package namespace."""
    assert "src." not in TileLangToGluonTranslator.__module__
    assert "src." not in to_gluon.__module__


def test_public_package_re_exports_core_api():
    """The wheel-facing package should expose its own stable public API."""
    assert public_pkg.TileLangToGluonTranslator is TileLangToGluonTranslator
    assert public_pkg.to_gluon is to_gluon
    assert public_pkg.TileLangGluonWrapper is TileLangGluonWrapper
    assert public_pkg.GluonKernelCache is GluonKernelCache
    assert public_pkg.TileLangToGluonTranslator.__module__.startswith("tilelang_to_gluon_translator")
    assert public_pkg.to_gluon.__module__.startswith("tilelang_to_gluon_translator")


def test_public_cli_forwards_to_translator_main():
    """The installed console entry point should use the existing translator CLI."""
    assert public_main is translator_main
