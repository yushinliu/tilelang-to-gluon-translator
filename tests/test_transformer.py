"""
Unit tests for TileLang to Gluon transformer.
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parser import TileLangParser
from src.transformer import TileLangToGluonTransformer
from src.transformer import (
    GluonKernel, GluonAllocShared, GluonRegisterTensor,
    GluonMma, GluonTmaLoad, GluonLoop
)


class TestTileLangToGluonTransformer:
    """Test cases for TileLang to Gluon transformer."""

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
