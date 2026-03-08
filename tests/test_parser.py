"""
Unit and integration tests for TileLang parser.

Includes:
- Unit tests for parsing logic (string-based)
- Integration tests with @to_gluon decorator
- GPU verification tests for parsed kernels
"""

import ast
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from tilelang_to_gluon_translator.parser import (
    TileLangParser, TileLangKernel, AllocShared, AllocFragment,
    CopyOp, GemmOp, ClearOp, AtomicAddOp, ParallelLoop, PipelinedLoop, SerialLoop
)
from tilelang_to_gluon_translator import to_gluon


class TestTileLangParser:
    """Unit tests for TileLang parser (string-based)."""

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
        assert kernel.block_vars == ["bx"]

    def test_parse_kernel_block_vars(self):
        """Test parsing block program IDs from T.Kernel context."""
        source = '''
@T.prim_func
def kernel_3d(A: T.Tensor((1024,), T.float32)):
    with T.Kernel(4, 5, 6, threads=128) as (bx, by, bz):
        pass
'''
        parser = TileLangParser()
        kernel = parser.parse(source)

        assert kernel.block_dims == [4, 5, 6]
        assert kernel.block_vars == ["bx", "by", "bz"]

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

    def test_parse_single_arg_serial_loop_defaults_start_to_zero(self):
        """T.serial(x) should parse like range(0, x)."""
        source = '''
@T.prim_func
def test_serial(A: T.Tensor((1024,), T.float32)):
    with T.Kernel(1, threads=128):
        for i in T.serial(8):
            tmp = i
'''
        parser = TileLangParser()
        kernel = parser.parse(source)

        loop = next(stmt for stmt in kernel.body if isinstance(stmt, SerialLoop))
        assert loop.start == 0
        assert loop.end == 8
        assert any(isinstance(stmt, ast.Assign) for stmt in loop.body)

    def test_parse_atomic_add(self):
        """Test parsing atomic add operation."""
        source = '''
@T.prim_func
def test_atomic_add(C: T.Tensor((128, 128), T.float32)):
    with T.Kernel(1, threads=128):
        tmp = T.alloc_shared([16, 16], T.float32)
        for i, j in T.Parallel(16, 16):
            T.atomic_add(C[i, j], tmp[i, j])
'''
        parser = TileLangParser()
        kernel = parser.parse(source)

        loop = next(stmt for stmt in kernel.body if isinstance(stmt, ParallelLoop))
        atomic_add = next(stmt for stmt in loop.body if isinstance(stmt, AtomicAddOp))
        assert atomic_add.target == "C"
        assert atomic_add.value == "tmp"
        assert atomic_add.target_indices == ["i", "j"]
        assert atomic_add.value_indices == ["i", "j"]

    def test_parse_numeric_floor_div_expression(self):
        """Numeric floor division should be folded instead of emitted as invalid text."""
        source = '''
@T.prim_func
def test_floor_div(A: T.Tensor((1024, 1024), T.float16)):
    with T.Kernel(1, threads=128):
        for k in T.Pipelined(T.ceildiv(256 // 2, 32), num_stages=0):
            pass
'''
        parser = TileLangParser()
        kernel = parser.parse(source)

        loop = next(stmt for stmt in kernel.body if isinstance(stmt, PipelinedLoop))
        assert loop.extent == 4


class TestParserDecoratorIntegration:
    """Integration tests for parser with @to_gluon decorator."""

    def test_decorator_extracts_source(self):
        """Test that @to_gluon decorator extracts source correctly."""
        @to_gluon
        def sample_kernel():
            """Sample kernel for testing."""
            pass

        assert sample_kernel.source_code is not None
        assert "sample_kernel" in sample_kernel.source_code
        assert "Sample kernel for testing" in sample_kernel.source_code

    def test_decorator_with_parser_options(self):
        """Test decorator with various parser-compatible options."""
        @to_gluon(max_jobs=4, verify=False)
        def kernel_with_options():
            """Kernel with decorator options."""
            pass

        assert kernel_with_options.max_jobs == 4
        assert kernel_with_options.source_code is not None


class TestParserGPUVerification:
    """GPU verification tests for parsed kernels."""

    @pytest.mark.gpu
    @pytest.mark.skip(reason="Requires actual TileLang environment with T.prim_func syntax")
    def test_simple_kernel_gpu_execution(self, device, tensor_factory, verify_tensors):
        """Test that parsed simple kernel can run on GPU.

        Note: This test requires actual TileLang kernel syntax, not regular Python.
        The @to_gluon decorator expects TileLang source code like:

            @T.prim_func
            def kernel(A: T.Tensor(...), ...):
                with T.Kernel(...) as (bx, by):
                    ...
        """
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
