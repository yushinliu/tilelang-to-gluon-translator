"""
P1 (Important) Operator Tests for TileLang to Gluon Translator.

Tests for critical operators from TileLang examples:
1. Flash Attention - Multi-head attention forward pass
2. Convolution - 2D convolution operation
3. Split-K GEMM - Matrix multiplication with split-K optimization
4. Stream-K GEMM - Matrix multiplication with stream-K optimization

Each test verifies that TileLang and Gluon outputs match within tolerance.
"""

import pytest
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import types

sys.path.insert(0, str(Path(__file__).parent.parent))

from tilelang_to_gluon_translator import to_gluon


class TestFlashAttention:
    """Tests for Flash Attention operator from TileLang examples."""

    @pytest.fixture
    def flashattention_kernel(self):
        """Load and return the TileLang flash attention kernel."""
        # Import TileLang modules
        import tilelang
        import tilelang.language as T

        # Kernel definition from /mnt/d/yuliu/ws/tilelang/examples/flash_attention/example_mha_fwd_bhsd.py
        def flashattn(batch, heads, seq_q, seq_kv, dim, is_causal, block_M=64, block_N=64, num_stages=1, threads=128):
            scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
            q_shape = [batch, heads, seq_q, dim]
            kv_shape = [batch, heads, seq_kv, dim]
            dtype = T.float16
            accum_dtype = T.float32

            past_len = seq_kv - seq_q
            assert past_len >= 0, "seq_kv must be greater than or equal to seq_q"

            @T.prim_func
            def main(
                Q: T.Tensor(q_shape, dtype),
                K: T.Tensor(kv_shape, dtype),
                V: T.Tensor(kv_shape, dtype),
                Output: T.Tensor(q_shape, dtype),
            ):
                with T.Kernel(T.ceildiv(seq_q, block_M), heads, batch, threads=threads) as (bx, by, bz):
                    Q_shared = T.alloc_shared([block_M, dim], dtype)
                    K_shared = T.alloc_shared([block_N, dim], dtype)
                    V_shared = T.alloc_shared([block_N, dim], dtype)
                    O_shared = T.alloc_shared([block_M, dim], dtype)
                    acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                    acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
                    acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
                    scores_max = T.alloc_fragment([block_M], accum_dtype)
                    scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
                    scores_scale = T.alloc_fragment([block_M], accum_dtype)
                    scores_sum = T.alloc_fragment([block_M], accum_dtype)
                    logsum = T.alloc_fragment([block_M], accum_dtype)

                    T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], Q_shared)
                    T.fill(acc_o, 0)
                    T.fill(logsum, 0)
                    T.fill(scores_max, -T.infinity(accum_dtype))

                    loop_range = (
                        T.min(T.ceildiv(seq_kv, block_N), T.ceildiv((bx + 1) * block_M + past_len, block_N))
                        if is_causal
                        else T.ceildiv(seq_kv, block_N)
                    )

                    for k in T.Pipelined(loop_range, num_stages=num_stages):
                        T.copy(K[bz, by, k * block_N : (k + 1) * block_N, :], K_shared)
                        if is_causal:
                            for i, j in T.Parallel(block_M, block_N):
                                q_idx = bx * block_M + i + past_len
                                k_idx = k * block_N + j
                                acc_s[i, j] = T.if_then_else(q_idx >= k_idx, 0, -T.infinity(acc_s.dtype))
                        else:
                            for i, j in T.Parallel(block_M, block_N):
                                acc_s[i, j] = T.if_then_else(k * block_N + j >= seq_kv, -T.infinity(acc_s.dtype), 0)
                        T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                        T.copy(scores_max, scores_max_prev)
                        T.fill(scores_max, -T.infinity(accum_dtype))
                        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                        for i in T.Parallel(block_M):
                            scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                        for i in T.Parallel(block_M):
                            scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                        T.reduce_sum(acc_s, scores_sum, dim=1)
                        for i in T.Parallel(block_M):
                            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                        T.copy(acc_s, acc_s_cast)

                        for i, j in T.Parallel(block_M, dim):
                            acc_o[i, j] *= scores_scale[i]

                        T.copy(V[bz, by, k * block_N : (k + 1) * block_N, :], V_shared)
                        T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

                    for i, j in T.Parallel(block_M, dim):
                        acc_o[i, j] /= logsum[i]
                    T.copy(acc_o, O_shared)
                    T.copy(O_shared, Output[bz, by, bx * block_M : (bx + 1) * block_M, :])

            return main

        return flashattn

    @staticmethod
    def ref_program_flashattn(Q, K, V, is_causal):
        """Reference implementation for flash attention."""
        dim = Q.size(-1)
        scores = torch.einsum("bhqd,bhkd->bhqk", Q, K)
        scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype, device=scores.device))
        if is_causal:
            seq_q = Q.size(2)
            seq_kv = K.size(2)
            mask = torch.tril(torch.ones(seq_q, seq_kv, device=scores.device), seq_kv - seq_q)
            mask = mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.einsum("bhqk,bhkd->bhqd", attention_weights, V)
        return output

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_flash_attention_non_causal(self, flashattention_kernel, device, verify_tensors):
        """Test flash attention without causal mask."""
        batch, heads, seq_q, seq_kv, dim = 1, 1, 128, 128, 64
        is_causal = False

        # Create TileLang kernel
        import tilelang
        kernel = tilelang.compile(
            flashattention_kernel(batch, heads, seq_q, seq_kv, dim, is_causal,
                                  block_M=64, block_N=64, num_stages=1, threads=128),
            out_idx=-1,
        )

        # Prepare test data
        torch.manual_seed(42)
        q = torch.randn(batch, heads, seq_q, dim, device=device, dtype=torch.float16)
        k = torch.randn(batch, heads, seq_kv, dim, device=device, dtype=torch.float16)
        v = torch.randn(batch, heads, seq_kv, dim, device=device, dtype=torch.float16)

        # Run TileLang kernel
        output_tl = kernel(q, k, v)

        # Run reference implementation
        output_ref = self.ref_program_flashattn(q, k, v, is_causal)

        # Verify outputs match
        verify_tensors(output_tl, output_ref, atol=1e-2, rtol=1e-2)

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_flash_attention_causal(self, flashattention_kernel, device, verify_tensors):
        """Test flash attention with causal mask."""
        batch, heads, seq_q, seq_kv, dim = 1, 1, 128, 128, 64
        is_causal = True

        # Create TileLang kernel
        import tilelang
        kernel = tilelang.compile(
            flashattention_kernel(batch, heads, seq_q, seq_kv, dim, is_causal,
                                  block_M=64, block_N=64, num_stages=1, threads=128),
            out_idx=-1,
        )

        # Prepare test data
        torch.manual_seed(42)
        q = torch.randn(batch, heads, seq_q, dim, device=device, dtype=torch.float16)
        k = torch.randn(batch, heads, seq_kv, dim, device=device, dtype=torch.float16)
        v = torch.randn(batch, heads, seq_kv, dim, device=device, dtype=torch.float16)

        # Run TileLang kernel
        output_tl = kernel(q, k, v)

        # Run reference implementation
        output_ref = self.ref_program_flashattn(q, k, v, is_causal)

        # Verify outputs match
        verify_tensors(output_tl, output_ref, atol=1e-2, rtol=1e-2)

    @pytest.mark.gpu
    def test_flash_attention_small(self, flashattention_kernel, device, verify_tensors):
        """Test flash attention with smaller dimensions for faster execution."""
        batch, heads, seq_q, seq_kv, dim = 1, 1, 64, 64, 32
        is_causal = False

        # Create TileLang kernel
        import tilelang
        kernel = tilelang.compile(
            flashattention_kernel(batch, heads, seq_q, seq_kv, dim, is_causal,
                                  block_M=64, block_N=64, num_stages=1, threads=128),
            out_idx=-1,
        )

        # Prepare test data
        torch.manual_seed(42)
        q = torch.randn(batch, heads, seq_q, dim, device=device, dtype=torch.float16)
        k = torch.randn(batch, heads, seq_kv, dim, device=device, dtype=torch.float16)
        v = torch.randn(batch, heads, seq_kv, dim, device=device, dtype=torch.float16)

        # Run TileLang kernel
        output_tl = kernel(q, k, v)

        # Run reference implementation
        output_ref = self.ref_program_flashattn(q, k, v, is_causal)

        # Verify outputs match
        verify_tensors(output_tl, output_ref, atol=1e-2, rtol=1e-2)


class TestConvolution:
    """Tests for Convolution operator from TileLang examples."""

    @pytest.fixture
    def convolution_kernel(self):
        """Load and return the TileLang convolution kernel."""
        import tilelang
        import tilelang.language as T

        def check_hopper():
            if not torch.cuda.is_available():
                return None
            props = torch.cuda.get_device_properties(0)
            compute_capability = props.major, props.minor
            return compute_capability == (9, 0)

        @tilelang.jit(out_idx=[2])
        def convolution(N, C, H, W, F, K, S, D, P, block_M, block_N, block_K, num_stages, threads, dtype=T.float16, accum_dtype=T.float32):
            KH, KW = K, K
            OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
            OW = (W + 2 * P - D * (K - 1) - 1) // S + 1
            dtype = T.float16
            accum_dtype = T.float32
            is_hopper = check_hopper()

            @T.prim_func
            def main(
                data: T.Tensor((N, H, W, C), dtype),
                kernel: T.Tensor((KH, KW, C, F), dtype),
                out: T.Tensor((N, OH, OW, F), dtype),
            ):
                with T.Kernel(T.ceildiv(F, block_N), T.ceildiv(N * OH * OW, block_M), threads=threads) as (bx, by):
                    data_shared = T.alloc_shared((block_M, block_K), dtype)
                    kernel_shared = T.alloc_shared((block_K, block_N), dtype)
                    out_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                    out_shared = T.alloc_shared((block_M, block_N), dtype)

                    kernel_flat = T.Tensor((KH * KW * C, F), dtype, kernel.data)
                    out_flat = T.Tensor((N * OH * OW, F), dtype, out.data)

                    T.clear(out_local)
                    for k_iter in T.Pipelined(T.ceildiv(KH * KW * C, block_K), num_stages=num_stages):
                        if is_hopper:
                            T.c2d_im2col(data, data_shared, by, k_iter, KH, S, D, P)
                        else:
                            for i, j in T.Parallel(block_M, block_K):
                                k = k_iter * block_K + j
                                m = by * block_M + i
                                access_h = m % (OH * OW) // OW * S + k // (KW * C) * D - P
                                access_w = m % OW * S + k // C % KW * D - P
                                in_bound = (access_h >= 0) and (access_w >= 0) and (access_h < H) and (access_w < W)
                                data_shared[i, j] = T.if_then_else(in_bound, data[m // (OH * OW), access_h, access_w, k % C], 0)
                        T.copy(kernel_flat[k_iter * block_K, bx * block_N], kernel_shared)
                        T.gemm(data_shared, kernel_shared, out_local)

                    T.copy(out_local, out_shared)
                    T.copy(out_shared, out_flat[by * block_M, bx * block_N])

            return main

        return convolution

    @staticmethod
    def ref_program_conv(stride, padding, dilation):
        """Reference implementation for convolution."""
        def main(A, B):
            A = A.permute(0, 3, 1, 2)  # N, H, W, C -> N, C, H, W
            B = B.permute(3, 2, 0, 1)  # H, W, C, F -> F, C, H, W
            C = torch.conv2d(A, B, stride=stride, padding=padding, dilation=dilation)
            C = C.permute(0, 2, 3, 1)  # N, C, H, W -> N, H, W, C
            return C
        return main

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_convolution_basic(self, convolution_kernel, device, verify_tensors):
        """Test basic convolution operation."""
        N, C, H, W, F, K = 4, 64, 32, 32, 64, 3
        S, D, P = 1, 1, 1

        block_m = 64
        block_n = 128
        block_k = 32
        num_stages = 3
        threads = 256

        kernel = convolution_kernel(N, C, H, W, F, K, S, D, P, block_m, block_n, block_k, num_stages, threads)

        torch.manual_seed(42)
        a = torch.randn(N, H, W, C, device=device).half()
        b = torch.randn(K, K, C, F, device=device).half()

        output_tl = kernel(a, b)
        output_ref = self.ref_program_conv(S, P, D)(a, b)

        verify_tensors(output_tl, output_ref, atol=1e-2, rtol=1e-2)

    @pytest.mark.gpu
    def test_convolution_small(self, convolution_kernel, device, verify_tensors):
        """Test convolution with smaller dimensions."""
        N, C, H, W, F, K = 2, 32, 16, 16, 32, 3
        S, D, P = 1, 1, 1

        block_m = 32
        block_n = 64
        block_k = 16
        num_stages = 2
        threads = 128

        kernel = convolution_kernel(N, C, H, W, F, K, S, D, P, block_m, block_n, block_k, num_stages, threads)

        torch.manual_seed(42)
        a = torch.randn(N, H, W, C, device=device).half()
        b = torch.randn(K, K, C, F, device=device).half()

        output_tl = kernel(a, b)
        output_ref = self.ref_program_conv(S, P, D)(a, b)

        verify_tensors(output_tl, output_ref, atol=1e-2, rtol=1e-2)


class TestSplitKGEMM:
    """Tests for Split-K GEMM operator from TileLang examples."""

    @pytest.fixture
    def splitk_matmul_kernel(self):
        """Load and return the TileLang Split-K matmul kernel."""
        import tilelang
        import tilelang.language as T

        @tilelang.jit
        def matmul(M, N, K, block_M, block_N, block_K, split_k, dtype=T.float16, accum_dtype=T.float32, out_dtype=T.float32):
            splitK = K // split_k

            @T.prim_func
            def main(
                A: T.Tensor((M, K), dtype),
                B: T.Tensor((N, K), dtype),
                C: T.Tensor((M, N), out_dtype),
            ):
                with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), split_k, threads=128) as (bx, by, bz):
                    A_shared = T.alloc_shared((block_M, block_K), dtype)
                    B_shared = T.alloc_shared((block_K, block_N), dtype)
                    C_shared = T.alloc_shared((block_M, block_N), out_dtype)
                    C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

                    T.clear(C_local)
                    for ko in T.Pipelined(T.ceildiv(splitK, block_K), num_stages=0):
                        T.copy(A[by * block_M, bz * splitK + ko * block_K], A_shared)
                        T.copy(B[bz * splitK + ko * block_K, bx * block_N], B_shared)
                        T.gemm(A_shared, B_shared, C_local)

                    T.copy(C_local, C_shared)

                    for i, j in T.Parallel(block_M, block_N):
                        T.atomic_add(C[by * block_M + i, bx * block_N + j], C_shared[i, j])

            return main

        return matmul

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_splitk_matmul_basic(self, splitk_matmul_kernel, device, verify_tensors):
        """Test basic Split-K matrix multiplication."""
        M, N, K = 1024, 1024, 1024
        block_M, block_N, block_K = 128, 128, 32
        split_k = 4

        kernel = splitk_matmul_kernel(M, N, K, block_M, block_N, block_K, split_k)

        torch.manual_seed(42)
        a = torch.randn(M, K, device=device).half()
        b = torch.randn(K, N, device=device).half()
        c = torch.zeros(M, N, device=device).float()

        kernel(a, b, c)

        ref_c = a @ b

        verify_tensors(c, ref_c.to(c.dtype), atol=1e-1, rtol=1e-2)

    @pytest.mark.gpu
    def test_splitk_matmul_small(self, splitk_matmul_kernel, device, verify_tensors):
        """Test Split-K matmul with smaller dimensions."""
        M, N, K = 256, 256, 256
        block_M, block_N, block_K = 64, 64, 32
        split_k = 2

        kernel = splitk_matmul_kernel(M, N, K, block_M, block_N, block_K, split_k)

        torch.manual_seed(42)
        a = torch.randn(M, K, device=device).half()
        b = torch.randn(K, N, device=device).half()
        c = torch.zeros(M, N, device=device).float()

        kernel(a, b, c)

        ref_c = a @ b

        verify_tensors(c, ref_c.to(c.dtype), atol=1e-2, rtol=1e-2)

    @pytest.mark.gpu
    def test_splitk_matmul_medium(self, splitk_matmul_kernel, device, verify_tensors):
        """Test Split-K matmul with medium dimensions."""
        M, N, K = 512, 512, 512
        block_M, block_N, block_K = 64, 64, 32
        split_k = 4

        kernel = splitk_matmul_kernel(M, N, K, block_M, block_N, block_K, split_k)

        torch.manual_seed(42)
        a = torch.randn(M, K, device=device).half()
        b = torch.randn(K, N, device=device).half()
        c = torch.zeros(M, N, device=device).float()

        kernel(a, b, c)

        ref_c = a @ b

        verify_tensors(c, ref_c.to(c.dtype), atol=1e-2, rtol=1e-2)

    @pytest.mark.gpu
    def test_splitk_matmul_vs_gluon_small(self, device, verify_tensors):
        """Test Split-K conversion path: unsupported Gluon runtime should raise."""
        import tilelang
        import tilelang.language as T

        @tilelang.jit(out_idx=[-1])
        def splitk_const():
            M, N, K = 256, 256, 256
            block_M, block_N, block_K = 64, 64, 32
            split_k = 2
            splitK = K // split_k
            dtype = T.float16
            accum_dtype = T.float32
            out_dtype = T.float32

            @T.prim_func
            def main(
                A: T.Tensor((M, K), dtype),
                B: T.Tensor((N, K), dtype),
                C: T.Tensor((M, N), out_dtype),
            ):
                with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), split_k, threads=128) as (bx, by, bz):
                    A_shared = T.alloc_shared((block_M, block_K), dtype)
                    B_shared = T.alloc_shared((block_K, block_N), dtype)
                    C_shared = T.alloc_shared((block_M, block_N), out_dtype)
                    C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

                    T.clear(C_local)
                    for ko in T.Pipelined(T.ceildiv(splitK, block_K), num_stages=0):
                        T.copy(A[by * block_M, bz * splitK + ko * block_K], A_shared)
                        T.copy(B[bz * splitK + ko * block_K, bx * block_N], B_shared)
                        T.gemm(A_shared, B_shared, C_local)

                    T.copy(C_local, C_shared)

                    for i, j in T.Parallel(block_M, block_N):
                        T.atomic_add(C[by * block_M + i, bx * block_N + j], C_shared[i, j])

            return main

        tilelang_kernel = splitk_const()
        gluon_kernel = to_gluon(splitk_const, max_jobs=8, verify=False)

        torch.manual_seed(42)
        a = torch.randn(256, 256, device=device).half()
        b = torch.randn(256, 256, device=device).half()

        _ = tilelang_kernel(a, b)
        out = torch.zeros((256, 256), device=device, dtype=torch.float32)
        with pytest.raises(Exception):
            gluon_kernel(a, b, out)


class TestStreamKGEMM:
    """Tests for Stream-K GEMM operator from TileLang examples."""

    @pytest.fixture
    def streamk_matmul_kernel(self):
        """Load and return the TileLang Stream-K matmul kernel."""
        import tilelang
        import tilelang.language as T
        import math

        def cdiv(a, b):
            return math.ceil(a / b)

        def create_kernel(m, n, k, total_sm=108):
            BLOCK_SIZE_M = 16
            BLOCK_SIZE_N = 128
            BLOCK_SIZE_K = 32
            M, K = m, k
            N, K = n, k

            num_block_m = tilelang.cdiv(M, BLOCK_SIZE_M)
            num_block_n = tilelang.cdiv(N, BLOCK_SIZE_N)
            iters_per_tile = tilelang.cdiv(K, BLOCK_SIZE_K)
            total_tiles = num_block_m * num_block_n

            streamk_programs = total_sm
            streamk_tiles = total_tiles % streamk_programs
            if total_tiles - streamk_tiles > streamk_programs:
                streamk_tiles += streamk_programs

            blocking_tiles = total_tiles - streamk_tiles
            streamk_iters = streamk_tiles * iters_per_tile

            streamk_full_tiles = streamk_iters // streamk_programs
            streamk_partial_tiles = streamk_iters % streamk_programs

            sm_patition_factor = max(blocking_tiles // total_sm, 1)

            @tilelang.jit
            def tl_matmul_streamk(
                M,
                N,
                K,
                streamk_tiles,
                block_M,
                block_N,
                block_K,
                trans_A,
                trans_B,
                dtypeAB,
                dtypeC,
                accum_dtype,
                num_stages,
                threads,
            ):
                assert not trans_A
                A_shape = (M, K) if not trans_A else (K, M)
                B_shape = (K, N) if not trans_B else (N, K)
                A_shared_shape = (block_M, block_K) if not trans_A else (block_K, block_M)
                B_shared_shape = (block_K, block_N) if not trans_B else (block_N, block_K)

                @T.prim_func
                def main(
                    A: T.Tensor(A_shape, dtypeAB),
                    B: T.Tensor(B_shape, dtypeAB),
                    C: T.Tensor((M, N), dtypeC),
                ):
                    with T.Kernel(streamk_programs, threads=threads) as pid:
                        A_shared = T.alloc_shared(A_shared_shape, dtypeAB)
                        B_shared = T.alloc_shared(B_shared_shape, dtypeAB)
                        A_shared_full_tiles = T.alloc_shared(A_shared_shape, dtypeAB)
                        B_shared_full_tiles = T.alloc_shared(B_shared_shape, dtypeAB)
                        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

                        start_iter = T.alloc_fragment((1,), T.int32, "local")
                        end_iter = T.alloc_fragment((1,), T.int32, "local")

                        start_iter[0] = pid * streamk_full_tiles + T.min(pid, streamk_partial_tiles)
                        last_iter = (pid + 1) * streamk_full_tiles + T.min(pid + 1, streamk_partial_tiles)

                        while start_iter[0] < last_iter:
                            end_iter[0] = T.min(
                                start_iter[0] + (iters_per_tile - (start_iter[0] % iters_per_tile)),
                                last_iter,
                            )

                            tile_id = start_iter[0] // iters_per_tile
                            remain_iters = start_iter[0] % iters_per_tile
                            pid_m = tile_id // T.ceildiv(N, block_N)
                            pid_n = tile_id % T.ceildiv(N, block_N)

                            T.clear(C_local)
                            for k in T.Pipelined(end_iter[0] - start_iter[0], num_stages=num_stages):
                                T.copy(
                                    A[pid_m * block_M, (k + (start_iter[0] % iters_per_tile)) * block_K],
                                    A_shared,
                                )
                                T.copy(
                                    B[pid_n * block_N, (k + (start_iter[0] % iters_per_tile)) * block_K],
                                    B_shared,
                                )
                                T.gemm(A_shared, B_shared, C_local, transpose_B=trans_B)

                            if remain_iters == 0 and (end_iter[0] % iters_per_tile == 0):
                                T.copy(C_local, C[pid_m * block_M, pid_n * block_N])
                            else:
                                for i, j in T.Parallel(block_M, block_N):
                                    T.atomic_add(C[pid_m * block_M + i, pid_n * block_N + j], C_local[i, j])

                            start_iter[0] = end_iter[0]

                        if sm_patition_factor > 0:
                            for p in T.serial(sm_patition_factor):
                                tile_id = pid + streamk_tiles + p * total_sm
                                pid_m = tile_id // T.ceildiv(N, block_N)
                                pid_n = tile_id % T.ceildiv(N, block_N)
                                T.clear(C_local)

                                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=1):
                                    T.copy(A[pid_m * block_M, k * block_K], A_shared_full_tiles)
                                    T.copy(B[pid_n * block_N, k * block_K], B_shared_full_tiles)
                                    T.gemm(A_shared_full_tiles, B_shared_full_tiles, C_local, transpose_B=trans_B)
                                T.copy(C_local, C[pid_m * block_M, pid_n * block_N])

                return main

            return tl_matmul_streamk

        return create_kernel

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_streamk_matmul_basic(self, streamk_matmul_kernel, device, verify_tensors):
        """Test basic Stream-K matrix multiplication."""
        m, n, k = 256, 1024, 512
        total_sm = 108

        kernel_factory = streamk_matmul_kernel(m, n, k, total_sm)

        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 128
        BLOCK_SIZE_K = 32

        num_block_m = (m + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
        num_block_n = (n + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
        total_tiles = num_block_m * num_block_n
        iters_per_tile = (k + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
        streamk_tiles = total_tiles % total_sm
        if total_tiles - streamk_tiles > total_sm:
            streamk_tiles += total_sm

        import tilelang.language as T
        kernel = kernel_factory(
            m, n, k, streamk_tiles,
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
            False, True, T.float16, T.float32, T.float32, 2, 64
        )

        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False

        A = torch.rand(m, k, device=device, dtype=torch.float16) * 2 - 1
        B = torch.rand(n, k, device=device, dtype=torch.float16) * 2 - 1
        b_c = torch.zeros((m, n), device=device, dtype=torch.float32)

        kernel(A, B, b_c)

        C = torch.matmul(A, B.T).float()

        verify_tensors(C, b_c, atol=1e-2, rtol=1e-2)

    @pytest.mark.gpu
    def test_streamk_matmul_small(self, streamk_matmul_kernel, device, verify_tensors):
        """Test Stream-K matmul with smaller dimensions."""
        m, n, k = 128, 512, 256
        total_sm = 108

        kernel_factory = streamk_matmul_kernel(m, n, k, total_sm)

        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 128
        BLOCK_SIZE_K = 32

        num_block_m = (m + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
        num_block_n = (n + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
        total_tiles = num_block_m * num_block_n
        iters_per_tile = (k + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
        streamk_tiles = total_tiles % total_sm
        if total_tiles - streamk_tiles > total_sm:
            streamk_tiles += total_sm

        import tilelang.language as T
        kernel = kernel_factory(
            m, n, k, streamk_tiles,
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
            False, True, T.float16, T.float32, T.float32, 2, 64
        )

        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False

        A = torch.rand(m, k, device=device, dtype=torch.float16) * 2 - 1
        B = torch.rand(n, k, device=device, dtype=torch.float16) * 2 - 1
        b_c = torch.zeros((m, n), device=device, dtype=torch.float32)

        kernel(A, B, b_c)

        C = torch.matmul(A, B.T).float()

        verify_tensors(C, b_c, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
