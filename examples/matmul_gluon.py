import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import (
    tma,
    mbarrier,
    fence_async_shared,
    warpgroup_mma_init,
    warpgroup_mma,
    warpgroup_mma_wait,
)

@gluon.jit
def matmul_kernel(
num_warps: gl.constexpr = 4,
    A_shared_layout: gl.constexpr,
    B_shared_layout: gl.constexpr,
    pipeline_bar_0: gl.constexpr):
    A_shared = gl.allocate_shared_memory(gl.float16, [128, 32], A_shared_layout)
    B_shared = gl.allocate_shared_memory(gl.float16, [128, 32], B_shared_layout)
    C_local = gl.zeros([128, 128], dtype=gl.float32, layout=gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 64, 16]))
    C_local = 0
    pipeline_bar_0 = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(pipeline_bar_0, count=2)
    # Pipelined loop with 2 stages
    phase = 0
    for k in range(0, ceildiv(K, 32), 1):
        C_local = warpgroup_mma(A_shared, B_shared, C_local, is_async=True)
        C_local = warpgroup_mma_wait(num_outstanding=0, deps=(C_local,))
        mbarrier.wait(pipeline_bar_0, phase=phase)
        phase ^= 1  # Toggle phase

def matmul(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor):
    grid = (ceildiv(N, 128), ceildiv(M, 128),)
    matmul_kernel[grid](A_shared_layout=gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2), B_shared_layout=gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2), pipeline_bar_0)
