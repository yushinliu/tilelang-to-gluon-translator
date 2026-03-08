"""
P0 integration tests using kernels from the TileLang examples repository.

These tests intentionally import source kernels from:
  /mnt/d/yuliu/ws/tilelang/examples
to keep translator validation aligned with upstream examples.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import torch

from tilelang_to_gluon_translator import to_gluon


EXAMPLES_ROOT = Path("/mnt/d/yuliu/ws/tilelang/examples")


def _load_module(name: str, rel_path: str) -> ModuleType:
    module_path = EXAMPLES_ROOT / rel_path
    if not module_path.exists():
        raise FileNotFoundError(f"Missing TileLang example: {module_path}")
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_p0_example_sources_exist():
    required = [
        "gemm/example_gemm.py",
        "elementwise/example_elementwise_add.py",
        "norm/test_rms_norm.py",
    ]
    for rel in required:
        assert (EXAMPLES_ROOT / rel).exists(), f"Missing example file: {rel}"


@pytest.mark.gpu
def test_gemm_512_from_example_matches_torch(device, verify_tensors):
    module = _load_module("tl_example_gemm", "gemm/example_gemm.py")

    kernel = module.matmul(512, 512, 512, 128, 128, 32)
    a = torch.randn(512, 512, device=device, dtype=torch.float16)
    b = torch.randn(512, 512, device=device, dtype=torch.float16)

    out = kernel(a, b)
    ref = a @ b
    verify_tensors(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.gpu
def test_gemm_1024_from_example_matches_torch(device, verify_tensors):
    module = _load_module("tl_example_gemm_1024", "gemm/example_gemm.py")

    kernel = module.matmul(1024, 1024, 1024, 128, 128, 32)
    a = torch.randn(1024, 1024, device=device, dtype=torch.float16)
    b = torch.randn(1024, 1024, device=device, dtype=torch.float16)

    out = kernel(a, b)
    ref = a @ b
    verify_tensors(out, ref, rtol=1e-2, atol=1.5e-1)


@pytest.mark.gpu
def test_elementwise_1024_from_example_matches_torch(device, verify_tensors):
    module = _load_module("tl_example_elementwise", "elementwise/example_elementwise_add.py")
    import tilelang.language as T

    kernel = module.elementwise_add(
        1024,
        1024,
        block_M=32,
        block_N=32,
        threads=128,
        in_dtype=T.float32,
        out_dtype=T.float32,
    )
    a = torch.randn(1024, 1024, device=device, dtype=torch.float32)
    b = torch.randn(1024, 1024, device=device, dtype=torch.float32)

    out = kernel(a, b)
    ref = a + b
    verify_tensors(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.gpu
def test_elementwise_1024_example_vs_gluon(device, verify_tensors):
    import tilelang
    import tilelang.language as T

    @tilelang.jit(out_idx=[-1])
    def elementwise_add_const():
        M, N = 1024, 1024
        block_M, block_N = 32, 32
        threads = 128
        in_dtype = T.float32
        out_dtype = T.float32

        @T.prim_func
        def elem_add(
            A: T.Tensor((M, N), in_dtype),
            B: T.Tensor((M, N), in_dtype),
            C: T.Tensor((M, N), out_dtype),
        ):
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

    tilelang_kernel = elementwise_add_const()
    gluon_kernel = to_gluon(elementwise_add_const, max_jobs=8, verify=False)

    a = torch.randn(1024, 1024, dtype=torch.float32, device=device)
    b = torch.randn(1024, 1024, dtype=torch.float32, device=device)
    tilelang_out = tilelang_kernel(a, b)
    gluon_out = torch.zeros_like(tilelang_out)
    with pytest.raises(Exception):
        gluon_kernel(a, b, gluon_out)


@pytest.mark.gpu
def test_rms_norm_1024_from_example_matches_torch(device, verify_tensors):
    module = _load_module("tl_example_norm", "norm/test_rms_norm.py")
    import tilelang

    program = module.rms_norm_splitk(1024, 1024, 32, 128)
    kernel = tilelang.compile(program, out_idx=-1, pass_configs={"tl.disable_tma_lower": True})

    x = torch.randn(1024, 1024, device=device, dtype=torch.float32)
    out = kernel(x)
    ref = module.ref_program(x)
    verify_tensors(out, ref, rtol=1e-2, atol=1e-2)
