"""
Unit tests for Gluon code generator.
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parser import TileLangParser
from src.transformer import TileLangToGluonTransformer
from src.codegen import GluonCodeGenerator


class TestGluonCodeGenerator:
    """Test cases for Gluon code generator."""

    def test_generate_imports(self):
        """Test import generation."""
        generator = GluonCodeGenerator()

        source = '''
@T.prim_func
def simple_kernel(A: T.Tensor((1024,), T.float32)):
    with T.Kernel(1, threads=128):
        pass
'''
        parser = TileLangParser()
        transformer = TileLangToGluonTransformer()

        tilelang_kernel = parser.parse(source)
        gluon_kernel = transformer.transform(tilelang_kernel)
        code = generator.generate(gluon_kernel)

        # Check imports
        assert "import torch" in code
        assert "import triton" in code
        assert "from triton.experimental import gluon" in code
        assert "@gluon.jit" in code

    def test_generate_kernel_signature(self):
        """Test kernel signature generation."""
        generator = GluonCodeGenerator()

        source = '''
@T.prim_func
def test_kernel(A: T.Tensor((1024,), T.float32)):
    with T.Kernel(1, threads=128):
        pass
'''
        parser = TileLangParser()
        transformer = TileLangToGluonTransformer()

        tilelang_kernel = parser.parse(source)
        gluon_kernel = transformer.transform(tilelang_kernel)
        code = generator.generate(gluon_kernel)

        # Check kernel function
        assert "def test_kernel_kernel(" in code
        assert "num_warps: gl.constexpr = 4" in code

    def test_generate_shared_allocation(self):
        """Test shared memory allocation generation."""
        generator = GluonCodeGenerator()

        source = '''
@T.prim_func
def test_alloc(A: T.Tensor((1024,), T.float32)):
    with T.Kernel(1, threads=128):
        shared = T.alloc_shared([128, 64], T.float16)
'''
        parser = TileLangParser()
        transformer = TileLangToGluonTransformer()

        tilelang_kernel = parser.parse(source)
        gluon_kernel = transformer.transform(tilelang_kernel)
        code = generator.generate(gluon_kernel)

        # Check shared memory allocation
        assert "gl.allocate_shared_memory" in code
        assert "gl.float16" in code

    def test_generate_launcher(self):
        """Test launcher function generation."""
        generator = GluonCodeGenerator()

        source = '''
@T.prim_func
def test_launcher(A: T.Tensor((1024,), T.float32)):
    with T.Kernel(1, threads=128):
        pass
'''
        parser = TileLangParser()
        transformer = TileLangToGluonTransformer()

        tilelang_kernel = parser.parse(source)
        gluon_kernel = transformer.transform(tilelang_kernel)
        code = generator.generate(gluon_kernel)

        # Check launcher function
        assert "def test_launcher(" in code
        assert "TensorDescriptor" in code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
