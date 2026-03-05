"""
Unit and integration tests for Gluon code generator.

Includes:
- Unit tests for code generation (string-based)
- Decorator-based code generation tests
- GPU execution tests for generated code
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parser import TileLangParser
from src.transformer import TileLangToGluonTransformer
from src.codegen import GluonCodeGenerator
from src.decorator import to_gluon


class TestGluonCodeGenerator:
    """Unit tests for Gluon code generator (string-based)."""

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


class TestCodegenDecoratorIntegration:
    """Integration tests for code generator with @to_gluon decorator."""

    @pytest.mark.skip(reason="Requires actual TileLang environment with T.prim_func syntax")
    def test_decorator_generates_code(self):
        """Test that decorator generates Gluon code.

        Note: This test requires actual TileLang kernel syntax, not regular Python.
        """
        pass

    @pytest.mark.skip(reason="Requires actual TileLang environment with T.prim_func syntax")
    def test_generated_code_has_required_imports(self):
        """Test that generated code includes required imports.

        Note: This test requires actual TileLang kernel syntax, not regular Python.
        """
        pass


class TestCodegenGPUExecution:
    """GPU execution tests for generated code."""

    @pytest.mark.gpu
    @pytest.mark.skip(reason="Requires actual TileLang environment with T.prim_func syntax")
    def test_generated_kernel_gpu_execution(self, device, tensor_factory, verify_tensors):
        """Test that generated kernel can execute on GPU.

        Note: This test requires actual TileLang kernel syntax, not regular Python.
        """
        pass

    @pytest.mark.gpu
    @pytest.mark.skip(reason="Requires actual TileLang environment with T.prim_func syntax")
    def test_generated_code_correctness(self, device, tensor_factory, verify_tensors):
        """Test numerical correctness of generated code.

        Note: This test requires actual TileLang kernel syntax, not regular Python.
        """
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
