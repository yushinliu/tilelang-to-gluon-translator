# TileLang to Gluon Primitive Mapping

## Core Primitives Mapping

| TileLang Primitive | Gluon Equivalent | Notes |
|-------------------|------------------|-------|
| `T.prim_func` | `@gluon.jit` | Kernel decorator |
| `T.Kernel(grid, threads=...)` | `gl.program_id(0/1/2)` | Grid launch, manual PID |
| `T.Tensor(shape, dtype)` | `TensorDescriptor` | TMA descriptor for global memory |
| `T.alloc_shared(shape, dtype)` | `gl.allocate_shared_memory(dtype, shape, layout)` | Requires explicit layout |
| `T.alloc_fragment(shape, dtype)` | Register tensor with `NVMMADistributedLayout` | Accumulator layout |
| `T.alloc_var(dtype)` | Scalar variable | Direct assignment |
| `T.copy(src, dst)` | `tma.async_copy_global_to_shared` / `smem.load()` / `smem.store()` | Depends on memory levels |
| `T.gemm(A, B, C, trans_A, trans_B)` | `warpgroup_mma(a, b, c, is_async=True)` | WGMMA on Hopper |
| `T.clear(buffer)` | `gl.zeros(shape, dtype, layout)` | Zero initialization |
| `T.Parallel(extents)` | `gl.arange(start, end, layout)` with `BlockedLayout` | Parallel loop |
| `T.Pipelined(extent, num_stages)` | Manual double-buffering + `mbarrier` | Software pipelining |
| `T.Serial(start, stop)` | Python `range(start, stop)` | Serial loop |
| `T.Unroll(start, stop)` | Python `range()` with pragma | Unrolled loop |
| `T.if_then_else(cond, a, b)` | `gl.where(cond, a, b)` | Select operation |
| `T.max(a, b)` | `gl.maximum(a, b)` | Element-wise max |
| `T.min(a, b)` | `gl.minimum(a, b)` | Element-wise min |
| `T.ceildiv(a, b)` | `triton.cdiv(a, b)` | Ceiling division |
| `T.float16`, `T.float32`, etc. | `gl.float16`, `gl.float32`, etc. | Data types |
| `T.reduce_max/min/sum` | `gl.reduce` with appropriate op | Reduction operations |

## Layout Mapping

TileLang uses implicit layouts based on buffer scope, while Gluon requires explicit layouts:

```python
# TileLang (implicit)
A_shared = T.alloc_shared([128, 64], T.float16)  # Shared memory
C_local = T.alloc_fragment([128, 128], T.float32)  # Fragment/accumulator

# Gluon (explicit)
# Shared memory with NVMMASharedLayout for TMA/WGMMA compatibility
a_layout = gl.NVMMASharedLayout.get_default_for([128, 64], gl.float16)
a_smem = gl.allocate_shared_memory(gl.float16, [128, 64], a_layout)

# Accumulator with NVMMADistributedLayout
mma_layout = gl.NVMMADistributedLayout(
    version=[3, 0],
    warps_per_cta=[4, 1],
    instr_shape=[16, 64, 16],  # m, n, k
)
acc = gl.zeros((128, 128), dtype=gl.float32, layout=mma_layout)
```

## Example Translation

### TileLang Input

```python
@T.prim_func
def matmul(
    A: T.Tensor((M, K), T.float16),
    B: T.Tensor((K, N), T.float16),
    C: T.Tensor((M, N), T.float16),
):
    with T.Kernel(T.ceildiv(N, 128), T.ceildiv(M, 128), threads=128) as (bx, by):
        A_shared = T.alloc_shared([128, 32], T.float16)
        B_shared = T.alloc_shared([128, 32], T.float16)
        C_local = T.alloc_fragment([128, 128], T.float32)
        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, 32), num_stages=2):
            T.copy(A[by * 128, k * 32], A_shared)
            T.copy(B[k * 32, bx * 128], B_shared)
            T.gemm(A_shared, B_shared, C_local, False, True)
        T.copy(C_local, C[by * 128, bx * 128])
```

### Gluon Output

```python
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import (
    tma, mbarrier, fence_async_shared,
    warpgroup_mma_init, warpgroup_mma, warpgroup_mma_wait,
)

@gluon.jit
def matmul_kernel(a_desc, b_desc, c_desc, num_warps: gl.constexpr):
    BLOCK_M: gl.constexpr = 128
    BLOCK_N: gl.constexpr = 128
    BLOCK_K: gl.constexpr = 32
    K = a_desc.shape[1]

    # Allocate shared memory
    a_smem = gl.allocate_shared_memory(gl.float16, [BLOCK_M, BLOCK_K], a_desc.layout)
    b_smem = gl.allocate_shared_memory(gl.float16, [BLOCK_K, BLOCK_N], b_desc.layout)

    # Setup MMA layout
    mma_layout = gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=[4, 1],
        instr_shape=[16, 64, 16],
    )
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mma_layout)

    # Get program IDs
    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    # Barrier for pipelining
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    phase = 0

    # Pipelined loop
    for k in range(0, K, BLOCK_K):
        mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(a_desc, [off_m, k], bar, a_smem)
        tma.async_copy_global_to_shared(b_desc, [k, off_n], bar, b_smem)
        mbarrier.wait(bar, phase=phase)
        phase ^= 1

        acc = warpgroup_mma(a_smem, b_smem, acc, is_async=True)
        acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))

    mbarrier.invalidate(bar)

    # Store result
    c_smem = gl.allocate_shared_memory(gl.float16, [BLOCK_M, BLOCK_N], c_desc.layout)
    c_smem.store(acc.to(gl.float16))
    fence_async_shared()
    tma.async_copy_shared_to_global(c_desc, [off_m, off_n], c_smem)
    tma.store_wait(pendings=0)
```
