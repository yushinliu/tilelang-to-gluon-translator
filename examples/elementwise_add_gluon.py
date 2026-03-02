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
def elementwise_add_kernel(
num_warps: gl.constexpr = 4,
    A_shared_layout: gl.constexpr,
    B_shared_layout: gl.constexpr,
    C_shared_layout: gl.constexpr):
    A_shared = gl.allocate_shared_memory(gl.float32, [128], A_shared_layout)
    B_shared = gl.allocate_shared_memory(gl.float32, [128], B_shared_layout)
    C_shared = gl.allocate_shared_memory(gl.float32, [128], C_shared_layout)
    for i in range(0, 128, 1):

def elementwise_add(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor):
    grid = (8,)
    elementwise_add_kernel[grid](A_shared_layout=gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=32, rank=1), B_shared_layout=gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=32, rank=1), C_shared_layout=gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=32, rank=1))
