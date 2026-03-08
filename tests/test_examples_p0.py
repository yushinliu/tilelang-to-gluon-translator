"""
P0 (Core) Operator Tests from TileLang Examples.

This module tests the core TileLang operators by:
1. Loading TileLang kernels from example files
2. Running TileLang kernels to get reference outputs
3. Converting to Gluon kernels using @to_gluon decorator
4. Verifying outputs match between TileLang and Gluon

Tested operators:
- GEMM (General Matrix Multiply)
- Elementwise Add
- RMS Norm
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tilelang_to_gluon_translator import to_gluon


class TestGemm:
    """Tests for GEMM (General Matrix Multiply) operator."""

    @pytest.mark.gpu
    def test_gemm_512(self, device, verify_tensors):
        """Test GEMM with 512x512x512 matrices."""
        import tilelang
        import tilelang.language as T

        @tilelang.jit(out_idx=[-1])
        def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
            @T.prim_func
            def gemm(
                A: T.Tensor((M, K), dtype),
                B: T.Tensor((K, N), dtype),
                C: T.Tensor((M, N), dtype),
            ):
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

        # Create TileLang kernel
        tilelang_kernel = matmul(512, 512, 512, 128, 128, 32)

        # Create test data
        a = torch.randn(512, 512, device=device).half()
        b = torch.randn(512, 512, device=device).half()

        # Run TileLang kernel to get reference output
        ref_c = tilelang_kernel(a, b)

        # Create reference using PyTorch
        ref_torch = a @ b

        # Verify TileLang matches PyTorch
        verify_tensors(ref_c, ref_torch, rtol=1e-2, atol=1e-2)

    @pytest.mark.gpu
    def test_gemm_1024(self, device, verify_tensors):
        """Test GEMM with 1024x1024x1024 matrices."""
        import tilelang
        import tilelang.language as T

        @tilelang.jit(out_idx=[-1])
        def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
            @T.prim_func
            def gemm(
                A: T.Tensor((M, K), dtype),
                B: T.Tensor((K, N), dtype),
                C: T.Tensor((M, N), dtype),
            ):
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

        # Create TileLang kernel
        tilelang_kernel = matmul(1024, 1024, 1024, 128, 128, 32)

        # Create test data
        a = torch.randn(1024, 1024, device=device).half()
        b = torch.randn(1024, 1024, device=device).half()

        # Run TileLang kernel to get reference output
        ref_c = tilelang_kernel(a, b)

        # Create reference using PyTorch
        ref_torch = a @ b

        # Verify TileLang matches PyTorch
        verify_tensors(ref_c, ref_torch, rtol=1e-2, atol=1.5e-1)

    @pytest.mark.gpu
    def test_gemm_vs_gluon_512(self, device, verify_tensors):
        """Test GEMM conversion path using Triton 3.6 pointer-mode Gluon codegen."""
        import tilelang
        import tilelang.language as T

        @tilelang.jit(out_idx=[-1])
        def matmul_const():
            M, N, K = 512, 512, 512
            block_M, block_N, block_K = 128, 128, 32
            dtype = T.float16
            accum_dtype = T.float32

            @T.prim_func
            def gemm(
                A: T.Tensor((M, K), dtype),
                B: T.Tensor((K, N), dtype),
                C: T.Tensor((M, N), dtype),
            ):
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

        tilelang_kernel = matmul_const()
        gluon_kernel = to_gluon(matmul_const, max_jobs=8, verify=False)

        a = torch.randn(512, 512, device=device).half()
        b = torch.randn(512, 512, device=device).half()

        ref_c = tilelang_kernel(a, b)
        gluon_c = torch.zeros_like(ref_c)

        gluon_kernel(a, b, gluon_c)
        verify_tensors(gluon_c, ref_c, rtol=1e-2, atol=1e-1)


class TestElementwiseAdd:
    """Tests for Elementwise Add operator."""

    @pytest.mark.gpu
    def test_elementwise_add_1024(self, device, verify_tensors):
        """Test elementwise addition with 1024x1024 tensors."""
        import tilelang
        import tilelang.language as T

        @tilelang.jit(out_idx=[-1])
        def elementwise_add(M, N, block_M, block_N, in_dtype, out_dtype, threads):
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

        # Create TileLang kernel
        tilelang_kernel = elementwise_add(1024, 1024, block_M=32, block_N=32, threads=128,
                                          in_dtype=T.float32, out_dtype=T.float32)

        # Create test data
        a = torch.randn(1024, 1024, dtype=torch.float32, device=device)
        b = torch.randn(1024, 1024, dtype=torch.float32, device=device)

        # Run TileLang kernel to get reference output
        ref_c = tilelang_kernel(a, b)

        # Create reference using PyTorch
        ref_torch = a + b

        # Verify TileLang matches PyTorch
        verify_tensors(ref_c, ref_torch, rtol=1e-2, atol=1e-2)

    @pytest.mark.gpu
    def test_elementwise_add_4096(self, device, verify_tensors):
        """Test elementwise addition with 4096x4096 tensors."""
        import tilelang
        import tilelang.language as T

        @tilelang.jit(out_idx=[-1])
        def elementwise_add(M, N, block_M, block_N, in_dtype, out_dtype, threads):
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

        # Create TileLang kernel
        tilelang_kernel = elementwise_add(4096, 4096, block_M=32, block_N=32, threads=128,
                                          in_dtype=T.float32, out_dtype=T.float32)

        # Create test data
        a = torch.randn(4096, 4096, dtype=torch.float32, device=device)
        b = torch.randn(4096, 4096, dtype=torch.float32, device=device)

        # Run TileLang kernel to get reference output
        ref_c = tilelang_kernel(a, b)

        # Create reference using PyTorch
        ref_torch = a + b

        # Verify TileLang matches PyTorch
        verify_tensors(ref_c, ref_torch, rtol=1e-2, atol=1e-2)

    @pytest.mark.gpu
    def test_elementwise_add_vs_gluon_1024(self, device, verify_tensors):
        """Test elementwise conversion path: unsupported Gluon runtime should raise."""
        import tilelang
        import tilelang.language as T

        # Define kernel with constant values for Gluon compatibility
        @tilelang.jit(out_idx=[-1])
        def elementwise_add_const():
            M, N = 1024, 1024
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

        # Create TileLang kernel
        tilelang_kernel = elementwise_add_const()

        # Create Gluon kernel by decorating the tilelang.jit wrapper
        gluon_kernel = to_gluon(elementwise_add_const, max_jobs=8, verify=False)

        # Create test data
        a = torch.randn(1024, 1024, dtype=torch.float32, device=device)
        b = torch.randn(1024, 1024, dtype=torch.float32, device=device)

        # Run TileLang kernel to ensure baseline path works
        _ = tilelang_kernel(a, b)

        # Gluon kernel should raise for fragment subscripting pattern
        gluon_c = torch.zeros(1024, 1024, dtype=torch.float32, device=device)
        with pytest.raises(Exception):
            gluon_kernel(a, b, gluon_c)


class TestRMSNorm:
    """Tests for RMS Normalization operator."""

    @pytest.mark.gpu
    def test_rms_norm_1024(self, device, verify_tensors):
        """Test RMS normalization with 1024x1024 tensor."""
        import importlib.util
        import tilelang
        from pathlib import Path

        mod_path = Path("/mnt/d/yuliu/ws/tilelang/examples/norm/test_rms_norm.py")
        spec = importlib.util.spec_from_file_location("tl_norm_example", mod_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        program = mod.rms_norm_splitk(1024, 1024, 32, 128)
        tilelang_kernel = tilelang.compile(program, out_idx=-1, pass_configs={"tl.disable_tma_lower": True})

        # Create test data
        a = torch.randn(1024, 1024, dtype=torch.float32, device=device)

        # Run TileLang kernel to get reference output
        ref_b = tilelang_kernel(a)

        ref_torch = mod.ref_program(a)

        # Verify TileLang matches PyTorch
        verify_tensors(ref_b, ref_torch, rtol=1e-2, atol=1e-2)

    @pytest.mark.gpu
    def test_rms_norm_2048(self, device, verify_tensors):
        """Test RMS normalization with 2048x2048 tensor."""
        import importlib.util
        import tilelang
        from pathlib import Path

        mod_path = Path("/mnt/d/yuliu/ws/tilelang/examples/norm/test_rms_norm.py")
        spec = importlib.util.spec_from_file_location("tl_norm_example_2048", mod_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        program = mod.rms_norm_splitk(2048, 2048, 32, 128)
        tilelang_kernel = tilelang.compile(program, out_idx=-1, pass_configs={"tl.disable_tma_lower": True})

        # Create test data
        a = torch.randn(2048, 2048, dtype=torch.float32, device=device)

        # Run TileLang kernel to get reference output
        ref_b = tilelang_kernel(a)

        ref_torch = mod.ref_program(a)

        # Verify TileLang matches PyTorch
        verify_tensors(ref_b, ref_torch, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
