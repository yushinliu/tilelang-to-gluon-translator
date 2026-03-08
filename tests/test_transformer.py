"""
Unit and integration tests for TileLang to Gluon transformer.

Includes:
- Unit tests for IR transformation (string-based)
- Decorator-based transformation tests
- GPU execution tests for transformed kernels
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parser import TileLangParser
from src.transformer import TileLangToGluonTransformer
from src.transformer import (
    GluonKernel, GluonAllocShared, GluonRegisterTensor,
    GluonMma, GluonTmaLoad, GluonLoop, GluonLocalCopy, GluonAtomicAdd, GluonProgramId
)
from src.decorator import to_gluon


class TestTileLangToGluonTransformer:
    """Unit tests for TileLang to Gluon transformer (string-based)."""

    def test_transform_simple_kernel(self):
        """Test transforming a simple kernel."""
        source = '''
@T.prim_func
def simple_kernel(A: T.Tensor((1024,), T.float32)):
    with T.Kernel(1, threads=128) as (bx,):
        shared = T.alloc_shared([128], T.float32)
'''
        parser = TileLangParser()
        transformer = TileLangToGluonTransformer()

        tilelang_kernel = parser.parse(source)
        gluon_kernel = transformer.transform(tilelang_kernel)

        assert isinstance(gluon_kernel, GluonKernel)
        assert gluon_kernel.name == "simple_kernel"
        assert gluon_kernel.num_warps == 4  # 128 threads / 32

    def test_transform_alloc_shared(self):
        """Test transforming shared memory allocation."""
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

        # Check shared allocation
        assert len(gluon_kernel.shared_allocs) > 0
        alloc = gluon_kernel.shared_allocs[0]
        assert alloc.shape == [128, 64]
        assert alloc.dtype == "gl.float16"
        assert "NVMMASharedLayout" in alloc.layout

    def test_transform_alloc_fragment(self):
        """Test transforming fragment allocation."""
        source = '''
@T.prim_func
def test_alloc(A: T.Tensor((1024,), T.float32)):
    with T.Kernel(1, threads=128):
        fragment = T.alloc_fragment([128, 128], T.float32)
'''
        parser = TileLangParser()
        transformer = TileLangToGluonTransformer()

        tilelang_kernel = parser.parse(source)
        gluon_kernel = transformer.transform(tilelang_kernel)

        # Find register tensor allocation in body
        alloc = None
        for stmt in gluon_kernel.body:
            if isinstance(stmt, GluonRegisterTensor):
                alloc = stmt
                break

        assert alloc is not None
        assert alloc.shape == [128, 128]
        assert alloc.dtype == "gl.float32"
        assert "NVMMADistributedLayout" in alloc.layout

    def test_transform_gemm(self):
        """Test transforming GEMM operation."""
        source = '''
@T.prim_func
def test_gemm(A: T.Tensor((128, 64), T.float16), B: T.Tensor((64, 128), T.float16)):
    with T.Kernel(1, threads=128):
        C = T.alloc_fragment([128, 128], T.float32)
        T.gemm(A, B, C, trans_A=False, trans_B=True)
'''
        parser = TileLangParser()
        transformer = TileLangToGluonTransformer()

        tilelang_kernel = parser.parse(source)
        gluon_kernel = transformer.transform(tilelang_kernel)

        # Find MMA operation in body
        mma = None
        for stmt in gluon_kernel.body:
            if isinstance(stmt, GluonMma):
                mma = stmt
                break

        assert mma is not None
        assert mma.A_desc == "A"
        assert mma.B_desc == "B"
        assert mma.acc == "C"
        assert mma.is_async == True

    def test_transform_grid(self):
        """Test grid computation."""
        source = '''
@T.prim_func
def test_grid(A: T.Tensor((1024, 1024), T.float32)):
    with T.Kernel(T.ceildiv(1024, 128), T.ceildiv(1024, 128), threads=128):
        pass
'''
        parser = TileLangParser()
        transformer = TileLangToGluonTransformer()

        tilelang_kernel = parser.parse(source)
        gluon_kernel = transformer.transform(tilelang_kernel)

        # Grid should be computed from block dimensions
        assert len(gluon_kernel.grid) == 2

    def test_transform_local_copy_and_atomic_add(self):
        """Shared/local copy and atomic add should be preserved as Gluon ops."""
        source = '''
@T.prim_func
def test_copy_atomic(C: T.Tensor((128, 128), T.float32)):
    with T.Kernel(1, threads=128):
        tmp = T.alloc_shared([16, 16], T.float32)
        frag = T.alloc_fragment([16, 16], T.float32)
        T.copy(frag, tmp)
        for i, j in T.Parallel(16, 16):
            T.atomic_add(C[i, j], tmp[i, j])
'''
        parser = TileLangParser()
        transformer = TileLangToGluonTransformer()

        tilelang_kernel = parser.parse(source)
        gluon_kernel = transformer.transform(tilelang_kernel)

        local_copy = next(stmt for stmt in gluon_kernel.body if isinstance(stmt, GluonLocalCopy))
        assert local_copy.src == "frag"
        assert local_copy.dst == "tmp"

        parallel_loop = next(stmt for stmt in gluon_kernel.body if isinstance(stmt, GluonLoop))
        atomic_add = next(stmt for stmt in parallel_loop.body[0].body if isinstance(stmt, GluonAtomicAdd))
        assert atomic_add.target == "C"
        assert atomic_add.value == "tmp"

    def test_transform_emits_program_ids(self):
        """Kernel launch indices should become explicit Gluon program IDs."""
        source = '''
@T.prim_func
def test_program_ids(A: T.Tensor((1024, 1024), T.float32)):
    with T.Kernel(4, 5, 6, threads=128) as (bx, by, bz):
        pass
'''
        parser = TileLangParser()
        transformer = TileLangToGluonTransformer()

        tilelang_kernel = parser.parse(source)
        gluon_kernel = transformer.transform(tilelang_kernel)

        program_ids = [stmt for stmt in gluon_kernel.body if isinstance(stmt, GluonProgramId)]
        assert [stmt.var_name for stmt in program_ids] == ["bx", "by", "bz"]
        assert [stmt.axis for stmt in program_ids] == [0, 1, 2]


class TestTransformerDecoratorIntegration:
    """Integration tests for transformer with @to_gluon decorator."""

    def test_decorator_creates_transformer(self):
        """Test that decorator creates transformer correctly."""
        @to_gluon(max_jobs=4, verify=False)
        def test_kernel():
            """Test kernel."""
            pass

        # Verify the wrapper was created with transformer
        assert test_kernel.translator is not None
        assert test_kernel.translator.transformer is not None

    @pytest.mark.skip(reason="Requires actual TileLang environment with T.prim_func syntax")
    def test_decorator_source_to_gluon(self):
        """Test that decorator can extract and transform source.

        Note: This test requires actual TileLang kernel syntax, not regular Python.
        """
        pass


class TestTransformerGPUExecution:
    """GPU execution tests for transformed kernels."""

    @pytest.mark.gpu
    @pytest.mark.skip(reason="Requires actual TileLang environment with T.prim_func syntax")
    def test_transformed_kernel_gpu_execution(self, device, tensor_factory, verify_tensors):
        """Test that transformed kernel can execute on GPU.

        Note: This test requires actual TileLang kernel syntax, not regular Python.
        """
        pass

    @pytest.mark.gpu
    @pytest.mark.skip(reason="Requires actual TileLang environment with T.prim_func syntax")
    def test_transformer_preserves_semantics(self, device, tensor_factory, verify_tensors):
        """Test that transformation preserves kernel semantics.

        Note: This test requires actual TileLang kernel syntax, not regular Python.
        """
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
