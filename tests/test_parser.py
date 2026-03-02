"""
Unit tests for TileLang parser.
"""

import ast
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parser import (
    TileLangParser, TileLangKernel, AllocShared, AllocFragment,
    CopyOp, GemmOp, ClearOp, ParallelLoop, PipelinedLoop
)


class TestTileLangParser:
    """Test cases for TileLang parser."""

    def test_parse_simple_kernel(self):
        """Test parsing a simple kernel."""
        source = '''
@T.prim_func
def simple_kernel(A: T.Tensor((1024,), T.float32)):
    with T.Kernel(1, threads=128) as (bx,):
        pass
'''
        parser = TileLangParser()
        kernel = parser.parse(source)

        assert isinstance(kernel, TileLangKernel)
        assert kernel.name == "simple_kernel"
        assert len(kernel.params) == 1
        assert kernel.thread_count == 128

    def test_parse_alloc_shared(self):
        """Test parsing shared memory allocation."""
        source = '''
@T.prim_func
def test_alloc(A: T.Tensor((1024,), T.float32)):
    with T.Kernel(1, threads=128):
        shared = T.alloc_shared([128, 64], T.float16)
'''
        parser = TileLangParser()
        kernel = parser.parse(source)

        # Find AllocShared statement
        alloc = None
        for stmt in kernel.body:
            if isinstance(stmt, AllocShared):
                alloc = stmt
                break

        assert alloc is not None
        assert alloc.shape == [128, 64]
        assert alloc.dtype == "float16"

    def test_parse_alloc_fragment(self):
        """Test parsing fragment allocation."""
        source = '''
@T.prim_func
def test_alloc(A: T.Tensor((1024,), T.float32)):
    with T.Kernel(1, threads=128):
        fragment = T.alloc_fragment([128, 128], T.float32)
'''
        parser = TileLangParser()
        kernel = parser.parse(source)

        # Find AllocFragment statement
        alloc = None
        for stmt in kernel.body:
            if isinstance(stmt, AllocFragment):
                alloc = stmt
                break

        assert alloc is not None
        assert alloc.shape == [128, 128]
        assert alloc.dtype == "float32"

    def test_parse_copy(self):
        """Test parsing copy operation."""
        source = '''
@T.prim_func
def test_copy(A: T.Tensor((1024,), T.float32)):
    with T.Kernel(1, threads=128):
        shared = T.alloc_shared([128], T.float32)
        T.copy(A[0:128], shared)
'''
        parser = TileLangParser()
        kernel = parser.parse(source)

        # Find CopyOp statement
        copy = None
        for stmt in kernel.body:
            if isinstance(stmt, CopyOp):
                copy = stmt
                break

        assert copy is not None
        assert copy.src == "A"
        assert copy.dst == "shared"

    def test_parse_gemm(self):
        """Test parsing GEMM operation."""
        source = '''
@T.prim_func
def test_gemm(A: T.Tensor((128, 64), T.float16), B: T.Tensor((64, 128), T.float16)):
    with T.Kernel(1, threads=128):
        C = T.alloc_fragment([128, 128], T.float32)
        T.gemm(A, B, C, trans_A=False, trans_B=True)
'''
        parser = TileLangParser()
        kernel = parser.parse(source)

        # Find GemmOp statement
        gemm = None
        for stmt in kernel.body:
            if isinstance(stmt, GemmOp):
                gemm = stmt
                break

        assert gemm is not None
        assert gemm.A == "A"
        assert gemm.B == "B"
        assert gemm.C == "C"
        assert gemm.trans_A == False
        assert gemm.trans_B == True

    def test_parse_clear(self):
        """Test parsing clear operation."""
        source = '''
@T.prim_func
def test_clear(A: T.Tensor((1024,), T.float32)):
    with T.Kernel(1, threads=128):
        fragment = T.alloc_fragment([128, 128], T.float32)
        T.clear(fragment)
'''
        parser = TileLangParser()
        kernel = parser.parse(source)

        # Find ClearOp statement
        clear = None
        for stmt in kernel.body:
            if isinstance(stmt, ClearOp):
                clear = stmt
                break

        assert clear is not None
        assert clear.buffer == "fragment"

    def test_parse_parallel_loop(self):
        """Test parsing parallel loop."""
        source = '''
@T.prim_func
def test_loop(A: T.Tensor((1024,), T.float32)):
    with T.Kernel(1, threads=128):
        for i in T.Parallel(10):
            pass
'''
        parser = TileLangParser()
        kernel = parser.parse(source)

        # Find ParallelLoop statement
        loop = None
        for stmt in kernel.body:
            if isinstance(stmt, ParallelLoop):
                loop = stmt
                break

        assert loop is not None
        assert loop.var == "i"
        assert loop.extent == 10

    def test_parse_pipelined_loop(self):
        """Test parsing pipelined loop."""
        source = '''
@T.prim_func
def test_pipelined(A: T.Tensor((1024, 1024), T.float16)):
    with T.Kernel(1, threads=128):
        for k in T.Pipelined(32, num_stages=2):
            pass
'''
        parser = TileLangParser()
        kernel = parser.parse(source)

        # Find PipelinedLoop statement
        loop = None
        for stmt in kernel.body:
            if isinstance(stmt, PipelinedLoop):
                loop = stmt
                break

        assert loop is not None
        assert loop.var == "k"
        assert loop.extent == 32
        assert loop.num_stages == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
