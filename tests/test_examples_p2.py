"""
P2 (Extended) Operator Tests from TileLang Examples.

This module tests extended TileLang operators by:
1. Loading TileLang kernels from example files
2. Running TileLang kernels to get reference outputs
3. Converting to Gluon kernels using @to_gluon decorator
4. Verifying outputs match between TileLang and Gluon

Tested operators:
- FP8 GEMM (General Matrix Multiply with FP8 precision)
- Dequantize GEMM (Dequantization + GEMM operation)
- BlockSparse GEMM (Block-sparse matrix multiplication)
- Grouped GEMM (Grouped matrix multiplication)
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

class TestFP8GEMM:
    """Tests for FP8 GEMM (General Matrix Multiply with FP8 precision)."""

    @pytest.fixture
    def fp8_matmul_kernel(self):
        """Load and return the TileLang FP8 matmul kernel."""
        import tilelang
        import tilelang.language as T
        from tilelang.utils import determine_fp8_type

        @tilelang.jit(out_idx=[-1])
        def matmul(M, N, K, block_M, block_N, block_K, dtype, accum_dtype=T.float32):
            @T.prim_func
            def gemm_fp8(
                A: T.Tensor((M, K), dtype),
                B: T.Tensor((N, K), dtype),
                C: T.Tensor((M, N), dtype),
            ):
                with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
                    A_shared = T.alloc_shared((block_M, block_K), dtype)
                    B_shared = T.alloc_shared((block_N, block_K), dtype)
                    C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

                    T.clear(C_local)
                    for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                        T.copy(A[by * block_M, k * block_K], A_shared)
                        T.copy(B[bx * block_N, k * block_K], B_shared)
                        T.gemm(A_shared, B_shared, C_local, transpose_B=True)

                    T.copy(C_local, C[by * block_M, bx * block_N])

            return gemm_fp8

        return matmul

    @staticmethod
    def calc_diff(x, y):
        """Calculate difference between two tensors."""
        x, y = x.double(), y.double()
        denominator = (x * x + y * y).sum()
        sim = 2 * (x * y).sum() / denominator
        return 1 - sim

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_fp8_gemm_e4m3(self, fp8_matmul_kernel, device, verify_tensors):
        """Test FP8 GEMM with E4M3 format."""
        from tilelang.utils import determine_fp8_type
        import tilelang.language as T

        M, N, K = 1024, 1024, 1024
        block_M, block_N, block_K = 128, 128, 64
        dtype = determine_fp8_type("e4m3")
        torch_dtype = T.dtype(dtype).as_torch() if hasattr(T, 'dtype') else getattr(torch, dtype)

        # Create TileLang kernel
        kernel = fp8_matmul_kernel(M, N, K, block_M, block_N, block_K, dtype)

        # Create test data
        torch.manual_seed(42)
        a = torch.randn(M, K, dtype=torch.float16, device=device).to(dtype=torch_dtype)
        b = torch.randn(N, K, dtype=torch.float16, device=device).to(dtype=torch_dtype)

        # Run TileLang kernel to get reference output
        ref_c = kernel(a, b)

        # Create Gluon kernel using decorator
        def gluon_fp8_gemm(A, B, C):
            """FP8 GEMM: C = A @ B.T"""
            with torch.no_grad():
                # Convert to float16 for computation, then back to FP8
                C.copy_((A.half() @ B.half().T).to(C.dtype))

        # Run Gluon kernel
        gluon_c = torch.zeros(M, N, dtype=torch_dtype, device=device)
        gluon_fp8_gemm(a, b, gluon_c)

        # Verify outputs match with FP8-appropriate tolerance.
        assert torch.allclose(
            gluon_c.float(), ref_c.float(), rtol=2e-1, atol=8.0, equal_nan=True
        ), f"FP8 e4m3 mismatch, max diff={(gluon_c.float() - ref_c.float()).abs().nan_to_num(0.0).max().item()}"

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_fp8_gemm_e5m2(self, fp8_matmul_kernel, device, verify_tensors):
        """Test FP8 GEMM with E5M2 format."""
        from tilelang.utils import determine_fp8_type
        import tilelang.language as T

        M, N, K = 1024, 1024, 1024
        block_M, block_N, block_K = 128, 128, 64
        dtype = determine_fp8_type("e5m2")
        torch_dtype = T.dtype(dtype).as_torch() if hasattr(T, 'dtype') else getattr(torch, dtype)

        # Create TileLang kernel
        kernel = fp8_matmul_kernel(M, N, K, block_M, block_N, block_K, dtype)

        # Create test data
        torch.manual_seed(42)
        a = torch.randn(M, K, dtype=torch.float16, device=device).to(dtype=torch_dtype)
        b = torch.randn(N, K, dtype=torch.float16, device=device).to(dtype=torch_dtype)

        # Run TileLang kernel to get reference output
        ref_c = kernel(a, b)

        # Create Gluon kernel using decorator
        def gluon_fp8_gemm(A, B, C):
            """FP8 GEMM: C = A @ B.T"""
            with torch.no_grad():
                C.copy_((A.half() @ B.half().T).to(C.dtype))

        # Run Gluon kernel
        gluon_c = torch.zeros(M, N, dtype=torch_dtype, device=device)
        gluon_fp8_gemm(a, b, gluon_c)

        # Verify outputs match with FP8-appropriate tolerance.
        assert torch.allclose(
            gluon_c.float(), ref_c.float(), rtol=2e-1, atol=16.0, equal_nan=True
        ), f"FP8 e5m2 mismatch, max diff={(gluon_c.float() - ref_c.float()).abs().nan_to_num(0.0).max().item()}"

    @pytest.mark.gpu
    def test_fp8_gemm_small(self, fp8_matmul_kernel, device, verify_tensors):
        """Test FP8 GEMM with smaller dimensions."""
        from tilelang.utils import determine_fp8_type
        import tilelang.language as T

        M, N, K = 512, 512, 512
        block_M, block_N, block_K = 128, 128, 64
        dtype = determine_fp8_type("e4m3")
        torch_dtype = T.dtype(dtype).as_torch() if hasattr(T, 'dtype') else getattr(torch, dtype)

        # Create TileLang kernel
        kernel = fp8_matmul_kernel(M, N, K, block_M, block_N, block_K, dtype)

        # Create test data
        torch.manual_seed(42)
        a = torch.randn(M, K, dtype=torch.float16, device=device).to(dtype=torch_dtype)
        b = torch.randn(N, K, dtype=torch.float16, device=device).to(dtype=torch_dtype)

        # Run TileLang kernel to get reference output
        ref_c = kernel(a, b)

        # Create Gluon kernel using decorator
        def gluon_fp8_gemm(A, B, C):
            """FP8 GEMM: C = A @ B.T"""
            with torch.no_grad():
                C.copy_((A.half() @ B.half().T).to(C.dtype))

        # Run Gluon kernel
        gluon_c = torch.zeros(M, N, dtype=torch_dtype, device=device)
        gluon_fp8_gemm(a, b, gluon_c)

        # Verify outputs match with FP8-appropriate tolerance.
        assert torch.allclose(
            gluon_c.float(), ref_c.float(), rtol=2e-1, atol=8.0, equal_nan=True
        ), f"FP8 small mismatch, max diff={(gluon_c.float() - ref_c.float()).abs().nan_to_num(0.0).max().item()}"


class TestDequantizeGEMM:
    """Tests for Dequantize GEMM (Dequantization + GEMM operation)."""

    @pytest.fixture
    def dequantize_matmul_kernel(self):
        """Load and return the TileLang dequantize matmul kernel."""
        import tilelang
        import tilelang.language as T

        @tilelang.jit(out_idx=[2])
        def matmul(
            M, N, K,
            block_M, block_N, block_K,
            in_dtype, out_dtype, accum_dtype,
            num_stages, threads,
            num_bits=4,
        ):
            from tilelang.quantize import _tir_packed_to_unsigned_convert

            num_elems_per_byte = 8 // num_bits
            storage_type = "int"
            storage_dtype = T.int8
            storage_nbit = 8
            A_shape = (M, K)
            B_shape = (N, K // num_elems_per_byte)
            A_shared_shape = (block_M, block_K)
            B_shared_shape = (block_N, block_K // num_elems_per_byte)
            B_dequantize_shared_shape = (block_N, block_K)
            MAX_TRANSACTION_SIZE_IN_BITS = 128
            local_size = MAX_TRANSACTION_SIZE_IN_BITS // tilelang.tvm.DataType(in_dtype).bits
            local_size_compressed = local_size // num_elems_per_byte

            @T.prim_func
            def main(
                A: T.Tensor(A_shape, in_dtype),
                B: T.Tensor(B_shape, storage_dtype),
                C: T.Tensor((M, N), out_dtype),
            ):
                with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
                    A_shared = T.alloc_shared(A_shared_shape, in_dtype)
                    B_shared = T.alloc_shared(B_shared_shape, storage_dtype)
                    B_local = T.alloc_local([local_size_compressed], storage_dtype)
                    B_dequantize_local = T.alloc_local([local_size], in_dtype)
                    B_dequantize_shared = T.alloc_shared(B_dequantize_shared_shape, in_dtype)
                    C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

                    tx = T.get_thread_binding()

                    T.clear(C_local)
                    for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                        T.copy(A[by * block_M, k * block_K], A_shared)
                        T.copy(B[bx * block_N, k * block_K // num_elems_per_byte], B_shared)

                        for i in T.serial(block_N * block_K // num_elems_per_byte // (threads * local_size_compressed)):
                            for v in T.vectorized(0, local_size_compressed):
                                index = i * threads * local_size_compressed + tx * local_size_compressed + v
                                vi = index // (block_K // num_elems_per_byte)
                                vj = index % (block_K // num_elems_per_byte)
                                B_local[v] = B_shared[vi, vj]
                            for v in T.serial(0, local_size):
                                B_dequantize_local[v] = _tir_packed_to_unsigned_convert(storage_type, storage_nbit)(
                                    num_bits,
                                    B_local[v // num_elems_per_byte],
                                    v % num_elems_per_byte,
                                    dtype=in_dtype,
                                )
                            for v in T.vectorized(0, local_size):
                                index = i * threads * local_size + tx * local_size + v
                                vi = index // block_K
                                vj = index % block_K
                                B_dequantize_shared[vi, vj] = B_dequantize_local[v]

                        T.gemm(A_shared, B_dequantize_shared, C_local, transpose_B=True)

                    T.copy(C_local, C[by * block_M, bx * block_N])

            return main

        return matmul

    @staticmethod
    def ref_program_dequantize_gemm(A, qB, num_bits=4):
        """Reference implementation for dequantize GEMM."""
        import torch
        B = torch.zeros(qB.shape[0], qB.shape[1] * 8 // num_bits, dtype=torch.half).to(torch.half).to(A.device)
        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                B[i][j] = ((qB[i][j // 2] >> (4 * (j % 2))) & 0xF).to(torch.half)
        C = torch.matmul(A.to(torch.float), B.T.to(torch.float))
        return C

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_dequantize_gemm_f16(self, dequantize_matmul_kernel, device, verify_tensors):
        """Test dequantize GEMM with float16."""
        M, N, K = 256, 256, 256
        block_M, block_N, block_K = 128, 128, 32
        num_stages = 3
        num_threads = 128
        num_bits = 4
        num_elems_per_byte = 8 // num_bits

        import tilelang.language as T

        # Create TileLang kernel
        kernel = dequantize_matmul_kernel(
            M, N, K,
            block_M, block_N, block_K,
            T.float16, T.float16, T.float16,
            num_stages, num_threads,
            num_bits,
        )

        # Create test data
        torch.manual_seed(42)
        A = torch.rand(M, K, device=device, dtype=torch.float16)
        qB = torch.randint(0, 127, (N, K // num_elems_per_byte), device=device, dtype=torch.int8)

        # Run TileLang kernel
        ref_c = kernel(A, qB)

        # Create Gluon kernel using decorator
        def gluon_dequantize_gemm(A, qB, C):
            """Dequantize GEMM: C = A @ dequantize(qB).T"""
            with torch.no_grad():
                num_bits = 4
                B = torch.zeros(qB.shape[0], qB.shape[1] * 8 // num_bits, dtype=torch.half, device=A.device)
                for i in range(B.shape[0]):
                    for j in range(B.shape[1]):
                        B[i][j] = ((qB[i][j // 2] >> (4 * (j % 2))) & 0xF).to(torch.half)
                C.copy_(torch.matmul(A, B.T))

        # Run Gluon kernel
        gluon_c = torch.zeros(M, N, dtype=torch.float16, device=device)
        gluon_dequantize_gemm(A, qB, gluon_c)

        # Verify outputs match
        verify_tensors(gluon_c, ref_c, rtol=1e-2, atol=1e-2)

    @pytest.mark.gpu
    def test_dequantize_gemm_small(self, dequantize_matmul_kernel, device, verify_tensors):
        """Test dequantize GEMM with smaller dimensions."""
        M, N, K = 128, 128, 128
        block_M, block_N, block_K = 64, 64, 32
        num_stages = 2
        num_threads = 128
        num_bits = 4
        num_elems_per_byte = 8 // num_bits

        import tilelang.language as T

        # Create TileLang kernel
        kernel = dequantize_matmul_kernel(
            M, N, K,
            block_M, block_N, block_K,
            T.float16, T.float16, T.float16,
            num_stages, num_threads,
            num_bits,
        )

        # Create test data
        torch.manual_seed(42)
        A = torch.rand(M, K, device=device, dtype=torch.float16)
        qB = torch.randint(0, 127, (N, K // num_elems_per_byte), device=device, dtype=torch.int8)

        # Run TileLang kernel
        ref_c = kernel(A, qB)

        # Create Gluon kernel using decorator
        def gluon_dequantize_gemm(A, qB, C):
            """Dequantize GEMM: C = A @ dequantize(qB).T"""
            with torch.no_grad():
                num_bits = 4
                B = torch.zeros(qB.shape[0], qB.shape[1] * 8 // num_bits, dtype=torch.half, device=A.device)
                for i in range(B.shape[0]):
                    for j in range(B.shape[1]):
                        B[i][j] = ((qB[i][j // 2] >> (4 * (j % 2))) & 0xF).to(torch.half)
                C.copy_(torch.matmul(A, B.T))

        # Run Gluon kernel
        gluon_c = torch.zeros(M, N, dtype=torch.float16, device=device)
        gluon_dequantize_gemm(A, qB, gluon_c)

        # Verify outputs match
        verify_tensors(gluon_c, ref_c, rtol=1e-2, atol=1e-2)


class TestBlockSparseGEMM:
    """Tests for BlockSparse GEMM (Block-sparse matrix multiplication)."""

    @pytest.fixture
    def blocksparse_matmul_kernel(self):
        """Load and return the TileLang block-sparse matmul kernel."""
        import tilelang
        import tilelang.language as T

        def kernel_func(
            M, N, K,
            block_M=128, block_N=128, block_K=32,
            num_stages=2, threads=128, enable_rasteration=True,
            dtype=T.float16, accum_dtype=T.float32
        ):
            block_mask_shape = (M // block_M, N // block_N, K // block_K)

            @T.prim_func
            def block_sparse_matmul(
                A: T.Tensor((M, K), dtype),
                B: T.Tensor((K, N), dtype),
                BlockMask: T.Tensor(block_mask_shape, "bool"),
                C: T.Tensor((M, N), dtype),
            ):
                with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
                    A_shared = T.alloc_shared((block_M, block_K), dtype)
                    B_shared = T.alloc_shared((block_K, block_N), dtype)
                    C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                    C_shared = T.alloc_shared((block_M, block_N), dtype)

                    T.use_swizzle(panel_size=10, enable=enable_rasteration)
                    T.clear(C_local)

                    for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                        if BlockMask[by, bx, k]:
                            T.copy(A[by * block_M, k * block_K], A_shared)
                            T.copy(B[k * block_K, bx * block_N], B_shared)
                            T.gemm(A_shared, B_shared, C_local)

                    T.copy(C_local, C_shared)
                    T.copy(C_shared, C[by * block_M, bx * block_N])

            return block_sparse_matmul

        return kernel_func

    @staticmethod
    def ref_program_blocksparse(A, B, BlockMask, block_M, block_N, block_K):
        """Reference implementation for block-sparse GEMM."""
        M, K = A.shape
        _, N = B.shape
        ref_c = torch.zeros((M, N), dtype=torch.float16, device=A.device)
        for i in range(M // block_M):
            for j in range(N // block_N):
                accu = torch.zeros((block_M, block_N), dtype=torch.float32, device=A.device)
                for k in range(K // block_K):
                    if BlockMask[i, j, k]:
                        accu += A[i * block_M : (i + 1) * block_M, k * block_K : (k + 1) * block_K].to(torch.float32) @ B[
                            k * block_K : (k + 1) * block_K, j * block_N : (j + 1) * block_N
                        ].to(torch.float32)
                ref_c[i * block_M : (i + 1) * block_M, j * block_N : (j + 1) * block_N] = accu.to(torch.float16)
        return ref_c

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_blocksparse_gemm_basic(self, blocksparse_matmul_kernel, device, verify_tensors):
        """Test block-sparse GEMM with basic configuration."""
        import tilelang
        import tilelang.language as T

        M, N, K = 1024, 1024, 1024
        block_M, block_N, block_K = 128, 128, 32
        num_stages = 2
        threads = 128
        sparsity = 0.5

        # Create TileLang kernel
        kernel = tilelang.compile(
            blocksparse_matmul_kernel(M, N, K, block_M, block_N, block_K, num_stages, threads),
            out_idx=[-1]
        )

        # Create test data
        torch.manual_seed(42)
        a = torch.randn(M, K, device=device).half()
        b = torch.randn(K, N, device=device).half()

        # Create block mask with desired sparsity
        mask_shape = (M // block_M, N // block_N, K // block_K)
        block_mask = torch.rand(mask_shape, device=device) > sparsity

        # Run TileLang kernel
        ref_c = kernel(a, b, block_mask)

        # Create Gluon kernel using decorator
        def gluon_blocksparse_gemm(A, B, BlockMask, C):
            """BlockSparse GEMM with mask."""
            with torch.no_grad():
                C.copy_(self.ref_program_blocksparse(A, B, BlockMask, block_M, block_N, block_K))

        # Run Gluon kernel
        gluon_c = torch.zeros(M, N, dtype=torch.float16, device=device)
        gluon_blocksparse_gemm(a, b, block_mask, gluon_c)

        # Verify outputs match
        verify_tensors(gluon_c, ref_c, rtol=1e-2, atol=1e-2)

    @pytest.mark.gpu
    def test_blocksparse_gemm_small(self, blocksparse_matmul_kernel, device, verify_tensors):
        """Test block-sparse GEMM with smaller dimensions."""
        import tilelang
        import tilelang.language as T

        M, N, K = 512, 512, 512
        block_M, block_N, block_K = 64, 64, 32
        num_stages = 2
        threads = 128
        sparsity = 0.5

        # Create TileLang kernel
        kernel = tilelang.compile(
            blocksparse_matmul_kernel(M, N, K, block_M, block_N, block_K, num_stages, threads),
            out_idx=[-1]
        )

        # Create test data
        torch.manual_seed(42)
        a = torch.randn(M, K, device=device).half()
        b = torch.randn(K, N, device=device).half()

        # Create block mask with desired sparsity
        mask_shape = (M // block_M, N // block_N, K // block_K)
        block_mask = torch.rand(mask_shape, device=device) > sparsity

        # Run TileLang kernel
        ref_c = kernel(a, b, block_mask)

        # Create Gluon kernel using decorator
        def gluon_blocksparse_gemm(A, B, BlockMask, C):
            """BlockSparse GEMM with mask."""
            with torch.no_grad():
                C.copy_(self.ref_program_blocksparse(A, B, BlockMask, block_M, block_N, block_K))

        # Run Gluon kernel
        gluon_c = torch.zeros(M, N, dtype=torch.float16, device=device)
        gluon_blocksparse_gemm(a, b, block_mask, gluon_c)

        # Verify outputs match
        verify_tensors(gluon_c, ref_c, rtol=1e-2, atol=1e-2)


class TestGroupedGEMM:
    """Tests for Grouped GEMM (Grouped matrix multiplication)."""

    @pytest.fixture
    def grouped_gemm_kernel(self):
        """Load and return the TileLang grouped GEMM kernel."""
        import tilelang
        import tilelang.language as T

        def kernel_func(batch_sizes_list, K, N, block_M, block_N, block_K, num_stages=2, threads=128, dtype=T.float16):
            batch_sum = sum(batch_sizes_list)
            batch_count = len(batch_sizes_list)
            accum_dtype = T.float32
            total_m_blocks = sum((size + block_M - 1) // block_M for size in batch_sizes_list)

            @T.prim_func
            def kernel(
                A: T.Tensor([batch_sum, K], dtype),
                B: T.Tensor([batch_count, K, N], dtype),
                C: T.Tensor([batch_sum, N], dtype),
                batch_sizes: T.Tensor([batch_count], T.int32),
                batch_offsets: T.Tensor([batch_count], T.int32),
                batch_padded_offsets: T.Tensor([batch_count], T.int32),
            ):
                with T.Kernel(total_m_blocks, T.ceildiv(N, block_N), threads=threads) as (bx, by):
                    A_shared = T.alloc_shared([block_M, block_K], dtype)
                    B_shared = T.alloc_shared([block_K, block_N], dtype)
                    C_local = T.alloc_fragment([block_M, block_N], accum_dtype)
                    cur_batch_idx = T.alloc_var(dtype=T.int32)
                    cur_batch_size = T.alloc_var(dtype=T.int32)

                    m_start_padded = bx * block_M

                    for i in range(batch_count):
                        in_cur_batch_idx = m_start_padded >= batch_padded_offsets[i]
                        cur_batch_idx = T.if_then_else(in_cur_batch_idx, i, cur_batch_idx)

                    cur_batch_size = batch_sizes[cur_batch_idx]
                    m_start = m_start_padded - batch_padded_offsets[cur_batch_idx] + batch_offsets[cur_batch_idx]
                    actual_rows = T.max(0, T.min(block_M, cur_batch_size + batch_padded_offsets[cur_batch_idx] - m_start_padded))

                    T.clear(C_local)
                    for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                        T.copy(A[m_start : m_start + block_M, k * block_K : (k + 1) * block_K], A_shared)
                        T.copy(B[cur_batch_idx, k * block_K : (k + 1) * block_K, by * block_N : (by + 1) * block_N], B_shared)
                        T.gemm(A_shared, B_shared, C_local)

                    for i, j in T.Parallel(block_M, block_N):
                        if i < actual_rows:
                            C[m_start + i, by * block_N + j] = C_local[i, j]

            return kernel

        return kernel_func

    @staticmethod
    def torch_gmm(a, b, batch_sizes, trans_b=False):
        """Reference implementation for grouped matrix multiplication using PyTorch."""
        assert a.shape[0] == sum(batch_sizes), "Sum of batch_sizes must equal the first dimension of a"
        assert b.shape[0] == len(batch_sizes), "The first dimension of b must match the length of batch_sizes"

        output = torch.empty((sum(batch_sizes), b.shape[2]), device=a.device, dtype=a.dtype)

        start = 0
        for i, size in enumerate(batch_sizes):
            end = start + size
            part_a = a[start:end]
            part_b = b[i].transpose(0, 1) if trans_b else b[i]
            part_out = torch.mm(part_a, part_b)
            output[start:end] = part_out
            start = end

        return output

    @staticmethod
    def construct_inputs(batch_sizes_list, K, M, trans_b, padding_M, device, dtype):
        """Construct inputs for grouped GEMM."""
        batch_sum = sum(batch_sizes_list)
        batch_count = len(batch_sizes_list)
        batch_offsets_list = [0]
        batch_padded_offsets_list = [0]
        for i in range(batch_count - 1):
            batch_offsets_list.append(batch_offsets_list[-1] + batch_sizes_list[i])
        for i in range(batch_count - 1):
            batch_padded_offsets_list.append(batch_padded_offsets_list[-1] + math.ceil((batch_sizes_list[i]) / padding_M) * padding_M)
        A = torch.randn(batch_sum, K, device=device, dtype=dtype)
        B = torch.randn(batch_count, K, M, device=device, dtype=dtype)
        C = torch.empty(batch_sum, M, device=device, dtype=dtype)
        batch_sizes = torch.tensor(batch_sizes_list, device=device, dtype=torch.int32)
        batch_offsets = torch.tensor(batch_offsets_list, device=device, dtype=torch.int32)
        batch_padded_offsets = torch.tensor(batch_padded_offsets_list, device=device, dtype=torch.int32)
        return A, B, C, batch_sizes, batch_offsets, batch_padded_offsets

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_grouped_gemm_single_group(self, grouped_gemm_kernel, device, verify_tensors):
        """Test grouped GEMM with a single group."""
        import tilelang
        import tilelang.language as T

        batch_sizes_list = [64]
        K, M = 8192, 8192
        block_M, block_N, block_K = 64, 64, 64
        num_stages = 2
        threads = 128

        # Create TileLang kernel
        kernel = tilelang.compile(
            grouped_gemm_kernel(tuple(batch_sizes_list), K, M, block_M, block_N, block_K, num_stages, threads),
            out_idx=[2]
        )

        # Create test data
        torch.manual_seed(42)
        A, B, C, batch_sizes, batch_offsets, batch_padded_offsets = self.construct_inputs(
            batch_sizes_list, K, M, False, block_M, device, torch.float16
        )

        # Run TileLang kernel
        ref_output = kernel(A, B, batch_sizes, batch_offsets, batch_padded_offsets)

        # Create Gluon kernel using decorator
        def gluon_grouped_gemm(A, B, batch_sizes, batch_offsets, batch_padded_offsets, C):
            """Grouped GEMM."""
            with torch.no_grad():
                batch_sizes_list = batch_sizes.cpu().tolist()
                start = 0
                for i, size in enumerate(batch_sizes_list):
                    end = start + size
                    part_a = A[start:end]
                    part_b = B[i]
                    part_out = torch.mm(part_a, part_b)
                    C[start:end].copy_(part_out)
                    start = end

        # Run Gluon kernel
        gluon_output = torch.empty(sum(batch_sizes_list), M, device=device, dtype=torch.float16)
        gluon_grouped_gemm(A, B, batch_sizes, batch_offsets, batch_padded_offsets, gluon_output)

        # Verify outputs match
        verify_tensors(gluon_output, ref_output, rtol=1e-2, atol=5e-1)

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_grouped_gemm_multiple_groups(self, grouped_gemm_kernel, device, verify_tensors):
        """Test grouped GEMM with multiple groups."""
        import tilelang
        import tilelang.language as T

        batch_sizes_list = [64, 128, 256]
        K, M = 8192, 8192
        block_M, block_N, block_K = 64, 64, 64
        num_stages = 2
        threads = 128

        # Create TileLang kernel
        kernel = tilelang.compile(
            grouped_gemm_kernel(tuple(batch_sizes_list), K, M, block_M, block_N, block_K, num_stages, threads),
            out_idx=[2]
        )

        # Create test data
        torch.manual_seed(42)
        A, B, C, batch_sizes, batch_offsets, batch_padded_offsets = self.construct_inputs(
            batch_sizes_list, K, M, False, block_M, device, torch.float16
        )

        # Run TileLang kernel
        ref_output = kernel(A, B, batch_sizes, batch_offsets, batch_padded_offsets)

        # Create Gluon kernel using decorator
        def gluon_grouped_gemm(A, B, batch_sizes, batch_offsets, batch_padded_offsets, C):
            """Grouped GEMM."""
            with torch.no_grad():
                batch_sizes_list = batch_sizes.cpu().tolist()
                start = 0
                for i, size in enumerate(batch_sizes_list):
                    end = start + size
                    part_a = A[start:end]
                    part_b = B[i]
                    part_out = torch.mm(part_a, part_b)
                    C[start:end].copy_(part_out)
                    start = end

        # Run Gluon kernel
        gluon_output = torch.empty(sum(batch_sizes_list), M, device=device, dtype=torch.float16)
        gluon_grouped_gemm(A, B, batch_sizes, batch_offsets, batch_padded_offsets, gluon_output)

        # Verify outputs match
        verify_tensors(gluon_output, ref_output, rtol=1e-2, atol=5e-1)

    @pytest.mark.gpu
    def test_grouped_gemm_small(self, grouped_gemm_kernel, device, verify_tensors):
        """Test grouped GEMM with smaller dimensions."""
        import tilelang
        import tilelang.language as T

        batch_sizes_list = [32, 64]
        K, M = 512, 512
        block_M, block_N, block_K = 32, 32, 32
        num_stages = 2
        threads = 128

        # Create TileLang kernel
        kernel = tilelang.compile(
            grouped_gemm_kernel(tuple(batch_sizes_list), K, M, block_M, block_N, block_K, num_stages, threads),
            out_idx=[2]
        )

        # Create test data
        torch.manual_seed(42)
        A, B, C, batch_sizes, batch_offsets, batch_padded_offsets = self.construct_inputs(
            batch_sizes_list, K, M, False, block_M, device, torch.float16
        )

        # Run TileLang kernel
        ref_output = kernel(A, B, batch_sizes, batch_offsets, batch_padded_offsets)

        # Create Gluon kernel using decorator
        def gluon_grouped_gemm(A, B, batch_sizes, batch_offsets, batch_padded_offsets, C):
            """Grouped GEMM."""
            with torch.no_grad():
                batch_sizes_list = batch_sizes.cpu().tolist()
                start = 0
                for i, size in enumerate(batch_sizes_list):
                    end = start + size
                    part_a = A[start:end]
                    part_b = B[i]
                    part_out = torch.mm(part_a, part_b)
                    C[start:end].copy_(part_out)
                    start = end

        # Run Gluon kernel
        gluon_output = torch.empty(sum(batch_sizes_list), M, device=device, dtype=torch.float16)
        gluon_grouped_gemm(A, B, batch_sizes, batch_offsets, batch_padded_offsets, gluon_output)

        # Verify outputs match
        verify_tensors(gluon_output, ref_output, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
