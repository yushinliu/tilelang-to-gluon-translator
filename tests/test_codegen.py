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
from src.codegen_pointer import GluonPointerCodeGenerator
from src.decorator import to_gluon


class TestGluonCodeGenerator:
    """Unit tests for Gluon code generator (string-based)."""

    def test_default_generator_uses_descriptor_mode(self):
        """Default codegen should follow the non-pointer Gluon path."""
        generator = GluonCodeGenerator()
        assert generator.use_pointer_mode is False

    def test_translator_default_output_avoids_pointer_signature(self):
        """Default translation should not silently fall back to pointer mode."""
        generator = GluonCodeGenerator()

        source = '''
@T.prim_func
def test_default_mode(A: T.Tensor((1024,), T.float32)):
    with T.Kernel(1, threads=128):
        pass
'''
        parser = TileLangParser()
        transformer = TileLangToGluonTransformer()

        tilelang_kernel = parser.parse(source)
        gluon_kernel = transformer.transform(tilelang_kernel)
        code = generator.generate(gluon_kernel)

        assert "TensorDescriptor" in code
        assert "tl.pointer_type" not in code
        assert "_ptr:" not in code

    def test_pointer_mode_requires_explicit_opt_in(self):
        """Pointer mode should only be enabled when explicitly requested."""
        generator = GluonCodeGenerator(use_pointer_mode=True)

        source = '''
@T.prim_func
def test_pointer_mode(A: T.Tensor((1024,), T.float32)):
    with T.Kernel(1, threads=128):
        pass
'''
        parser = TileLangParser()
        transformer = TileLangToGluonTransformer()

        tilelang_kernel = parser.parse(source)
        gluon_kernel = transformer.transform(tilelang_kernel)
        code = generator.generate(gluon_kernel)

        assert "tl.pointer_type" in code
        assert "A_ptr:" in code

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

    def test_pointer_mode_lowers_parallel_atomic_add_to_tile_atomic(self):
        """Pointer mode should avoid scalar indexing into distributed tensors."""
        generator = GluonPointerCodeGenerator()

        source = '''
@T.prim_func
def test_atomic_tile(C: T.Tensor((128, 128), T.float32)):
    with T.Kernel(1, 1, threads=128) as (bx, by):
        tmp = T.alloc_fragment([16, 16], T.float32)
        for i, j in T.Parallel(16, 16):
            T.atomic_add(C[by * 16 + i, bx * 16 + j], tmp[i, j])
'''
        parser = TileLangParser()
        transformer = TileLangToGluonTransformer()

        tilelang_kernel = parser.parse(source)
        gluon_kernel = transformer.transform(tilelang_kernel)
        code = generator.generate(gluon_kernel)

        assert "tmp_atomic_tmp = gl.convert_layout(tmp, gl.BlockedLayout([4, 1], [1, 32], [4, 1], [1, 0]))" in code
        assert "gl.atomic_add(ptr_tmp, tmp_atomic_tmp, mask=mask_tmp)" in code
        assert "tmp[i, j]" not in code
        assert "i_idx_tmp[:, None]" in code
        assert "j_idx_tmp[None, :]" in code

    def test_pointer_mode_emits_constexpr_dot_k_width_literal(self):
        """Dot operand layouts should use integer literals, not runtime tensor dtype access."""
        generator = GluonPointerCodeGenerator()

        source = '''
@T.prim_func
def test_dot_literal(
    A: T.Tensor((128, 128), T.float16),
    B: T.Tensor((128, 128), T.float16),
    C: T.Tensor((128, 128), T.float32),
):
    with T.Kernel(1, 1, threads=128):
        A_shared = T.alloc_shared((64, 32), T.float16)
        B_shared = T.alloc_shared((32, 64), T.float16)
        C_local = T.alloc_fragment((64, 64), T.float32)
        T.gemm(A_shared, B_shared, C_local)
'''
        parser = TileLangParser()
        transformer = TileLangToGluonTransformer()

        tilelang_kernel = parser.parse(source)
        gluon_kernel = transformer.transform(tilelang_kernel)
        code = generator.generate(gluon_kernel)

        assert "k_width=2" in code
        assert ".dtype.primitive_bitwidth" not in code
        assert "gl.NVMMADistributedLayout" in code

    def test_pointer_mode_clear_uses_static_layout_for_fragments(self):
        """Clear should preserve fragment layout instead of re-deriving it from a runtime value."""
        generator = GluonPointerCodeGenerator()

        source = '''
@T.prim_func
def test_clear_layout(A: T.Tensor((128, 128), T.float16)):
    with T.Kernel(1, 1, threads=128):
        C_local = T.alloc_fragment((64, 64), T.float32)
        T.clear(C_local)
'''
        parser = TileLangParser()
        transformer = TileLangToGluonTransformer()

        tilelang_kernel = parser.parse(source)
        gluon_kernel = transformer.transform(tilelang_kernel)
        code = generator.generate(gluon_kernel)

        assert "layout=gl.NVMMADistributedLayout" in code
        assert "layout=C_local.type.layout" not in code

    def test_pointer_mode_uses_mma_v2_when_runtime_supports_it(self):
        """Pointer mode should prefer the official Gluon mma_v2 path when available."""
        generator = GluonPointerCodeGenerator()
        generator.has_runtime_dot_operand_layout = True
        generator.has_runtime_mma_v2 = True

        source = '''
@T.prim_func
def test_mma_v2(
    A: T.Tensor((128, 128), T.float16),
    B: T.Tensor((128, 128), T.float16),
    C: T.Tensor((128, 128), T.float32),
):
    with T.Kernel(1, 1, threads=128):
        A_shared = T.alloc_shared((64, 32), T.float16)
        B_shared = T.alloc_shared((32, 64), T.float16)
        C_local = T.alloc_fragment((64, 64), T.float32)
        T.gemm(A_shared, B_shared, C_local)
'''
        parser = TileLangParser()
        transformer = TileLangToGluonTransformer()

        tilelang_kernel = parser.parse(source)
        gluon_kernel = transformer.transform(tilelang_kernel)
        code = generator.generate(gluon_kernel)

        assert "from triton.experimental.gluon.language.nvidia.ampere import mma_v2" in code
        assert "C_local = mma_v2(A_shared_dot, B_shared_dot, C_local)" in code
        assert "C_local_dot = tl.dot" not in code

    def test_pointer_mode_local_copy_converts_to_destination_layout(self):
        """Local/shared copies should respect the destination layout instead of aliasing the source tensor."""
        generator = GluonPointerCodeGenerator()

        source = '''
@T.prim_func
def test_copy_layout(A: T.Tensor((128, 128), T.float16)):
    with T.Kernel(1, 1, threads=128):
        C_shared = T.alloc_shared((64, 64), T.float32)
        C_local = T.alloc_fragment((64, 64), T.float32)
        T.copy(C_local, C_shared)
'''
        parser = TileLangParser()
        transformer = TileLangToGluonTransformer()

        tilelang_kernel = parser.parse(source)
        gluon_kernel = transformer.transform(tilelang_kernel)
        code = generator.generate(gluon_kernel)

        assert "C_shared = gl.convert_layout(C_local, gl.BlockedLayout" in code
        assert "C_shared = C_local" not in code


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
