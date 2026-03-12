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
import importlib.util
import types

sys.path.insert(0, str(Path(__file__).parent.parent))

from tilelang_to_gluon_translator import to_gluon


EXAMPLES_ROOT = Path("/mnt/d/yuliu/ws/tilelang/examples")


def _load_example_module(name: str, rel_path: str):
    module_path = EXAMPLES_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_example_module_with_stubs(name: str, rel_path: str, stubs: dict[str, object]):
    saved = {}
    try:
        for mod_name, stub in stubs.items():
            saved[mod_name] = sys.modules.get(mod_name)
            sys.modules[mod_name] = stub
        return _load_example_module(name, rel_path)
    finally:
        for mod_name, prev in saved.items():
            if prev is None:
                sys.modules.pop(mod_name, None)
            else:
                sys.modules[mod_name] = prev


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
        from tilelang_to_gluon_translator import TileLangToGluonTranslator

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
    def test_gemm_kernel_auto_falls_back_to_pointer_mode(self, device):
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
        ref = gemm_const()(a, b)

        kernel(a, b, out)
        assert torch.allclose(out, ref, rtol=1e-2, atol=1e-1)
        assert kernel.translator.use_pointer_mode is True

    @pytest.mark.gpu
    def test_simt_gemv_thread_binding_lowers_in_pointer_mode(self, device):
        @tilelang.jit(out_idx=[-1])
        def simt_gemv_const():
            N, K = 32, 32
            dtype = T.float16
            accum_dtype = T.float32

            @T.prim_func
            def gemv(A: T.Tensor((K,), dtype), B: T.Tensor((N, K), dtype), C: T.Tensor((N,), dtype)):
                with T.Kernel(1, threads=N) as bx:
                    tx = T.get_thread_binding(0)
                    C_local = T.alloc_local((1,), accum_dtype)
                    T.clear(C_local)
                    for k in T.serial(K):
                        C_local[0] += A[k].astype(accum_dtype) * B[tx, k].astype(accum_dtype)
                    C[tx] = C_local[0]

            return gemv

        kernel = to_gluon(simt_gemv_const, max_jobs=8, verify=False)
        a = torch.randn(32, device=device, dtype=torch.float16)
        b = torch.randn(32, 32, device=device, dtype=torch.float16)
        out = torch.zeros(32, device=device, dtype=torch.float16)
        ref = torch.matmul(b.to(torch.float32), a.to(torch.float32)).to(torch.float16)

        kernel(a, b, out)
        assert torch.allclose(out, ref, rtol=1e-2, atol=1e-2)
        assert kernel.translator.use_pointer_mode is True

    @pytest.mark.gpu
    @pytest.mark.xfail(
        reason="Instantiated TIR cast still relies on scalar fragment indexing not yet vectorized in pointer-mode Gluon lowering."
    )
    def test_instantiated_cast_example_jitkernel_matches_tilelang(self, device):
        module = _load_example_module("tl_example_cast", "cast/example_per_token_cast_to_fp8.py")

        tilelang_kernel = module.per_token_cast_to_fp8(256, 256, 8)
        gluon_kernel = to_gluon(tilelang_kernel, max_jobs=8, verify=False)

        x = torch.randn(256, 256, device=device, dtype=torch.float32)
        tl_fp8, tl_amax = tilelang_kernel(x)
        gl_fp8, gl_amax = gluon_kernel(x)

        assert torch.allclose(tl_fp8.to(torch.float32), gl_fp8.to(torch.float32), rtol=1e-2, atol=1e-2)
        assert torch.allclose(tl_amax, gl_amax, rtol=1e-2, atol=1e-2)
        assert gluon_kernel.translator.use_pointer_mode is True

    @pytest.mark.gpu
    def test_instantiated_topk_example_jitkernel_matches_tilelang(self, device):
        module = _load_example_module("tl_example_topk", "topk/example_topk.py")

        tilelang_kernel = module.tl_topk(M=320, N=128, topk=6, blk_m=64)
        gluon_kernel = to_gluon(tilelang_kernel, max_jobs=8, verify=False)

        logits = torch.rand(320, 128, device=device, dtype=torch.float32)
        tl_gates, tl_indices = tilelang_kernel(logits)
        gl_gates, gl_indices = gluon_kernel(logits)

        assert torch.allclose(tl_gates, gl_gates, rtol=1e-2, atol=1e-2)
        assert torch.equal(tl_indices, gl_indices)
        assert gluon_kernel.translator.use_pointer_mode is True

    @pytest.mark.gpu
    def test_instantiated_dynamic_shape_matmul_matches_tilelang(self, device):
        module = _load_example_module("tl_example_dynamic", "dynamic_shape/example_dynamic.py")

        tilelang_kernel = module.matmul_dynamic_mnk(64, 64, 32, False, False, "float16", "float16", "float32", 3, 128)
        gluon_kernel = to_gluon(tilelang_kernel, max_jobs=8, verify=False)

        a = torch.randn(128, 96, device=device, dtype=torch.float16)
        b = torch.randn(96, 160, device=device, dtype=torch.float16)
        tl_out = torch.zeros(128, 160, device=device, dtype=torch.float16)
        gl_out = torch.zeros(128, 160, device=device, dtype=torch.float16)

        tilelang_kernel(a, b, tl_out)
        result = gluon_kernel(a, b, gl_out)
        if isinstance(result, torch.Tensor):
            gl_out = result

        assert torch.allclose(tl_out, gl_out, rtol=1e-2, atol=1e-2)
        assert gluon_kernel.translator.use_pointer_mode is True

    @pytest.mark.gpu
    def test_instantiated_hadamard_example_jitkernel_matches_tilelang(self, device):
        scipy_stub = types.ModuleType("scipy")
        scipy_stub.linalg = types.SimpleNamespace(hadamard=lambda *args, **kwargs: None)
        module = _load_example_module_with_stubs(
            "tl_example_hadamard",
            "hadamard_transform/example_hadamard.py",
            {"scipy": scipy_stub},
        )

        tilelang_kernel = module.hadamard(2, 256, module.T.float32)
        gluon_kernel = to_gluon(tilelang_kernel, max_jobs=8, verify=False)

        x = torch.randn(2, 256, device=device, dtype=torch.float32)
        tl_out = tilelang_kernel(x)
        gl_out = gluon_kernel(x)

        assert torch.allclose(tl_out, gl_out, rtol=1e-2, atol=1e-2)
        assert gluon_kernel.translator.use_pointer_mode is True

    @pytest.mark.gpu
    def test_instantiated_flash_attention_bhsd_example_jitkernel_matches_tilelang(self, device):
        module = _load_example_module("tl_example_flash_attention_bhsd", "flash_attention/example_mha_fwd_bhsd.py")

        tilelang_kernel = module.flashattn(1, 1, 64, 64, 32, False, 64, 64, 1, 128)
        gluon_kernel = to_gluon(tilelang_kernel, max_jobs=8, verify=False)

        q = torch.randn(1, 1, 64, 32, device=device, dtype=torch.float16)
        k = torch.randn(1, 1, 64, 32, device=device, dtype=torch.float16)
        v = torch.randn(1, 1, 64, 32, device=device, dtype=torch.float16)
        tl_out = tilelang_kernel(q, k, v)
        gl_out = gluon_kernel(q, k, v)

        assert torch.allclose(tl_out, gl_out, rtol=1e-2, atol=1e-2)
        assert gluon_kernel.translator.use_pointer_mode is True

    @pytest.mark.gpu
    def test_instantiated_flash_attention_bshd_example_jitkernel_matches_tilelang(self, device):
        module = _load_example_module("tl_example_flash_attention_bshd", "flash_attention/example_mha_fwd_bshd.py")

        tilelang_kernel = module.flashattn(1, 1, 64, 32, False, 64, 64, 1, 128)
        gluon_kernel = to_gluon(tilelang_kernel, max_jobs=8, verify=False)

        q = torch.randn(1, 64, 1, 32, device=device, dtype=torch.float16)
        k = torch.randn(1, 64, 1, 32, device=device, dtype=torch.float16)
        v = torch.randn(1, 64, 1, 32, device=device, dtype=torch.float16)
        tl_out = tilelang_kernel(q, k, v)
        gl_out = gluon_kernel(q, k, v)

        assert torch.allclose(tl_out, gl_out, rtol=1e-2, atol=1e-2)
        assert gluon_kernel.translator.use_pointer_mode is True

    @pytest.mark.gpu
    def test_instantiated_flash_attention_gqa_bshd_example_jitkernel_matches_tilelang(self, device):
        module = _load_example_module("tl_example_flash_attention_gqa_bshd", "flash_attention/example_gqa_fwd_bshd.py")

        tilelang_kernel = module.flashattn(1, 4, 64, 32, False, 2, 64, 64, 2, 128)
        gluon_kernel = to_gluon(tilelang_kernel, max_jobs=8, verify=False)

        q = torch.randn(1, 64, 4, 32, device=device, dtype=torch.float16)
        k = torch.randn(1, 64, 2, 32, device=device, dtype=torch.float16)
        v = torch.randn(1, 64, 2, 32, device=device, dtype=torch.float16)
        tl_out = tilelang_kernel(q, k, v)
        gl_out = gluon_kernel(q, k, v)

        assert torch.allclose(tl_out, gl_out, rtol=1e-2, atol=1e-2)
        assert gluon_kernel.translator.use_pointer_mode is True

    @pytest.mark.gpu
    def test_instantiated_naive_gemv_example_jitkernel_matches_tilelang(self, device):
        module = _load_example_module("tl_example_gemv", "gemv/example_gemv.py")

        tilelang_kernel = module.naive_gemv(128, 128, 128, 128)
        gluon_kernel = to_gluon(tilelang_kernel, max_jobs=8, verify=False)

        a = torch.randn(128, device=device, dtype=torch.float16)
        b = torch.randn(128, 128, device=device, dtype=torch.float16)
        tl_out = tilelang_kernel(a, b)
        gl_out = gluon_kernel(a, b)

        assert torch.allclose(tl_out, gl_out, rtol=1e-2, atol=1e-2)
        assert gluon_kernel.translator.use_pointer_mode is True
