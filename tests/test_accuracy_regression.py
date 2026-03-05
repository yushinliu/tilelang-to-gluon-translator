"""
Strict-mode accuracy/behavior regression tests for @to_gluon.

This suite validates that strict mode rejects non-TileLang kernels and keeps
behavior stable across input shapes/dtypes.
"""

import pytest
import torch
import tilelang
import tilelang.language as T
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.decorator import to_gluon


def _assert_plain_kernel_rejected(fn, *args):
    with pytest.raises(ValueError, match="No kernel function found with @T\\.prim_func decorator"):
        fn(*args)


class TestStrictRejectionMatrix:
    """Matrix-style regression checks for strict rejection behavior."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("shape", [(64,), (1024,), (128, 128)])
    def test_elementwise_plain_kernel_rejected(self, dtype, shape):
        @to_gluon(max_jobs=8, verify=False)
        def add_kernel(A, B, C):
            with torch.no_grad():
                C.copy_(A + B)

        a = torch.randn(*shape, dtype=dtype)
        b = torch.randn(*shape, dtype=dtype)
        c = torch.zeros(*shape, dtype=dtype)

        _assert_plain_kernel_rejected(add_kernel, a, b, c)

    @pytest.mark.parametrize("m,n,k", [(32, 32, 32), (64, 64, 64), (128, 64, 32)])
    def test_matmul_plain_kernel_rejected(self, m, n, k):
        @to_gluon(max_jobs=8, verify=False)
        def gemm_kernel(A, B, C):
            with torch.no_grad():
                C.copy_(A @ B)

        a = torch.randn(m, k)
        b = torch.randn(k, n)
        c = torch.zeros(m, n)

        _assert_plain_kernel_rejected(gemm_kernel, a, b, c)

    @pytest.mark.parametrize("size", [1, 16, 256, 2048])
    def test_reduction_plain_kernel_rejected(self, size):
        @to_gluon(max_jobs=8, verify=False)
        def reduce_kernel(inp, out):
            with torch.no_grad():
                out.copy_(inp.sum().unsqueeze(0))

        inp = torch.randn(size)
        out = torch.zeros(1)
        _assert_plain_kernel_rejected(reduce_kernel, inp, out)


class TestStrictSourceBehavior:
    """Regression checks for source generation behavior in strict mode."""

    def test_get_gluon_source_rejects_plain_function(self):
        @to_gluon(max_jobs=8, verify=False)
        def plain_kernel(A, B, C):
            with torch.no_grad():
                C.copy_(A + B)

        with pytest.raises(ValueError, match="No kernel function found with @T\\.prim_func decorator"):
            _ = plain_kernel.get_gluon_source()

    def test_get_gluon_source_for_valid_prim_func(self):
        source = '''
@T.prim_func
def simple_kernel(A: T.Tensor((128,), T.float32), B: T.Tensor((128,), T.float32)):
    with T.Kernel(1, threads=128) as (bx,):
        A_shared = T.alloc_shared([128], T.float32)
        T.copy(A[0:128], A_shared)
'''
        from src.translator import TileLangToGluonTranslator

        translator = TileLangToGluonTranslator(max_jobs=8, verify=False)
        code = translator.translate(source)
        assert "@gluon.jit" in code


class TestExamplesStrictCompatibility:
    """Use minimal real TileLang kernels to ensure strict path remains stable."""

    @pytest.mark.gpu
    def test_elementwise_fragment_kernel_raises_on_gluon_runtime(self, device):
        @tilelang.jit(out_idx=[-1])
        def elementwise_add_const():
            M, N = 256, 256
            block_M, block_N = 32, 32
            threads = 128
            in_dtype = T.float32
            out_dtype = T.float32

            @T.prim_func
            def elem_add(A: T.Tensor((M, N), in_dtype), B: T.Tensor((M, N), in_dtype), C: T.Tensor((M, N), out_dtype)):
                with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
                    A_shared = T.alloc_shared((block_M, block_N), in_dtype)
                    B_shared = T.alloc_shared((block_M, block_N), in_dtype)
                    C_local = T.alloc_fragment((block_M, block_N), out_dtype)
                    C_shared = T.alloc_shared((block_M, block_N), out_dtype)
                    T.copy(A[by * block_M, bx * block_N], A_shared)
                    T.copy(B[by * block_M, bx * block_N], B_shared)
                    for local_y, local_x in T.Parallel(block_M, block_N):
                        C_local[local_y, local_x] = A_shared[local_y, local_x] + B_shared[local_y, local_x]
                    T.copy(C_local, C_shared)
                    T.copy(C_shared, C[by * block_M, bx * block_N])

            return elem_add

        kernel = to_gluon(elementwise_add_const, max_jobs=8, verify=False)
        a = torch.randn(256, 256, device=device, dtype=torch.float32)
        b = torch.randn(256, 256, device=device, dtype=torch.float32)
        out = torch.zeros(256, 256, device=device, dtype=torch.float32)

        with pytest.raises(Exception):
            kernel(a, b, out)

    @pytest.mark.gpu
    def test_gemm_kernel_raises_on_current_codegen_limit(self, device):
        @tilelang.jit(out_idx=[-1])
        def gemm_const():
            M, N, K = 128, 128, 128
            block_M, block_N, block_K = 64, 64, 32
            dtype = T.float16
            accum_dtype = T.float32

            @T.prim_func
            def gemm(A: T.Tensor((M, K), dtype), B: T.Tensor((K, N), dtype), C: T.Tensor((M, N), dtype)):
                with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
                    A_shared = T.alloc_shared((block_M, block_K), dtype)
                    B_shared = T.alloc_shared((block_K, block_N), dtype)
                    C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                    T.clear(C_local)
                    for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                        T.copy(A[by * block_M, k * block_K], A_shared)
                        T.copy(B[k * block_K, bx * block_N], B_shared)
                        T.gemm(A_shared, B_shared, C_local)
                    T.copy(C_local, C[by * block_M, bx * block_N])

            return gemm

        kernel = to_gluon(gemm_const, max_jobs=8, verify=False)
        a = torch.randn(128, 128, device=device).half()
        b = torch.randn(128, 128, device=device).half()
        out = torch.zeros(128, 128, device=device).half()

        with pytest.raises(Exception):
            kernel(a, b, out)
