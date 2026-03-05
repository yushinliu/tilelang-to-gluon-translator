"""
Transform TileLang AST to Gluon AST.
"""

import ast
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union

from .parser import (
    TileLangKernel, AllocShared, AllocFragment, AllocLocal,
    CopyOp, GemmOp, ClearOp, ParallelLoop, PipelinedLoop, SerialLoop
)


@dataclass
class GluonAllocShared:
    """Gluon shared memory allocation."""
    name: str
    shape: List[int]
    dtype: str
    layout: str


@dataclass
class GluonRegisterTensor:
    """Gluon register tensor allocation."""
    name: str
    shape: List[int]
    dtype: str
    layout: str


@dataclass
class GluonTensorDescriptor:
    """Gluon TMA tensor descriptor."""
    name: str
    tensor_name: str
    shape: List[int]
    dtype: str
    layout: str = ""


@dataclass
class GluonMma:
    """Gluon MMA/WGMMA operation."""
    A_desc: str
    B_desc: str
    acc: str
    is_async: bool = True


@dataclass
class GluonTmaLoad:
    """Gluon TMA async load."""
    desc: str
    offsets: List[Any]
    barrier: str
    smem: str


@dataclass
class GluonTmaStore:
    """Gluon TMA async store."""
    desc: str
    offsets: List[Any]
    smem: str


@dataclass
class GluonBarrier:
    """Gluon memory barrier."""
    name: str
    count: int


@dataclass
class GluonBarrierInit:
    """Gluon barrier initialization."""
    barrier: str
    count: int


@dataclass
class GluonBarrierWait:
    """Gluon barrier wait."""
    barrier: str
    phase: Any


@dataclass
class GluonLoop:
    """Gluon loop construct."""
    var: str
    start: Any
    end: Any
    step: Any
    body: List[Any] = field(default_factory=list)
    is_pipelined: bool = False
    num_stages: int = 1


@dataclass
class GluonClear:
    """Gluon clear/zero operation."""
    buffer: str
    dtype: str


@dataclass
class GluonProgramId:
    """Gluon program ID access."""
    axis: int
    var_name: str


@dataclass
class GluonKernel:
    """Represents a transformed Gluon kernel."""
    name: str
    params: List[Dict[str, Any]]
    grid: List[Any]
    num_warps: int
    body: List[Any] = field(default_factory=list)
    tensor_descriptors: List[GluonTensorDescriptor] = field(default_factory=list)
    shared_allocs: List[GluonAllocShared] = field(default_factory=list)
    register_tensors: List[GluonRegisterTensor] = field(default_factory=list)
    barriers: List[GluonBarrier] = field(default_factory=list)


GluonStmt = Union[
    GluonAllocShared, GluonRegisterTensor, GluonTensorDescriptor,
    GluonMma, GluonTmaLoad, GluonTmaStore, GluonBarrier, GluonBarrierInit,
    GluonBarrierWait, GluonLoop, GluonClear, GluonProgramId
]


class TileLangToGluonTransformer:
    """Transforms TileLang kernel to Gluon kernel."""

    # Dtype mapping from TileLang to Gluon
    DTYPE_MAP = {
        "float16": "gl.float16",
        "float32": "gl.float32",
        "float64": "gl.float64",
        "int8": "gl.int8",
        "int16": "gl.int16",
        "int32": "gl.int32",
        "int64": "gl.int64",
        "uint8": "gl.uint8",
        "uint16": "gl.uint16",
        "uint32": "gl.uint32",
        "uint64": "gl.uint64",
        "bfloat16": "gl.bfloat16",
    }

    def __init__(self):
        self.layout_map = {}
        self.buffer_to_descriptor = {}
        self.current_kernel = None
        self.var_counter = 0

    def _fresh_name(self, prefix: str) -> str:
        """Generate a fresh variable name."""
        name = f"{prefix}_{self.var_counter}"
        self.var_counter += 1
        return name

    def transform(self, kernel: TileLangKernel) -> GluonKernel:
        """Transform TileLang kernel to Gluon kernel."""
        # Handle both integer and string thread counts
        thread_count = kernel.thread_count
        if isinstance(thread_count, str):
            # If thread_count is a variable name, use default (128)
            thread_count = 128
        num_warps = max(thread_count // 32, 1)

        # Extract block dimensions from the first allocations (common pattern)
        block_M = 32
        block_N = 32
        in_dtype = "gl.float32"
        out_dtype = "gl.float32"

        gluon_kernel = GluonKernel(
            name=kernel.name,
            params=self._transform_params(kernel.params),
            grid=self._compute_grid(kernel.block_dims),
            num_warps=num_warps,
            body=[],
            tensor_descriptors=[],
            shared_allocs=[],
            barriers=[]
        )
        # Store additional attributes for codegen
        gluon_kernel.block_M = block_M
        gluon_kernel.block_N = block_N
        gluon_kernel.in_dtype = in_dtype
        gluon_kernel.out_dtype = out_dtype
        self.current_kernel = gluon_kernel

        # Create tensor descriptors for input/output tensors
        for param in kernel.params:
            if param.get("annotation", {}).get("type") == "Tensor":
                desc = GluonTensorDescriptor(
                    name=f"{param['name']}_desc",
                    tensor_name=param["name"],
                    shape=param["annotation"].get("shape", []),
                    dtype=self._map_dtype(param["annotation"].get("dtype", "float32")),
                    layout=""
                )
                gluon_kernel.tensor_descriptors.append(desc)
                self.buffer_to_descriptor[param["name"]] = desc.name

        # Transform kernel body
        for stmt in kernel.body:
            gluon_stmts = self._transform_stmt(stmt)
            if gluon_stmts:
                if isinstance(gluon_stmts, list):
                    gluon_kernel.body.extend(gluon_stmts)
                else:
                    gluon_kernel.body.append(gluon_stmts)

        return gluon_kernel

    def _transform_params(self, params: List[Dict]) -> List[Dict]:
        """Transform kernel parameters."""
        result = []
        for param in params:
            transformed = {
                "name": param["name"],
                "type": "tensor_descriptor" if param.get("annotation", {}).get("type") == "Tensor" else "other"
            }
            if "annotation" in param:
                transformed["shape"] = param["annotation"].get("shape", [])
                transformed["dtype"] = self._map_dtype(param["annotation"].get("dtype", "float32"))
            result.append(transformed)
        return result

    def _compute_grid(self, block_dims: List[Any]) -> List[Any]:
        """Compute Gluon grid from TileLang block dimensions."""
        return block_dims if block_dims else [1]

    def _map_dtype(self, dtype: str) -> str:
        """Map TileLang dtype to Gluon dtype."""
        return self.DTYPE_MAP.get(dtype, f"gl.{dtype}")

    def _get_base_dtype(self, gluon_dtype: str) -> str:
        """Get base dtype from Gluon dtype string."""
        return gluon_dtype.replace("gl.", "")

    def _transform_stmt(self, stmt: Any) -> Optional[Union[GluonStmt, List[GluonStmt]]]:
        """Transform a TileLang statement to Gluon."""
        if isinstance(stmt, AllocShared):
            return self._transform_alloc_shared(stmt)
        elif isinstance(stmt, AllocFragment):
            return self._transform_alloc_fragment(stmt)
        elif isinstance(stmt, AllocLocal):
            return self._transform_alloc_local(stmt)
        elif isinstance(stmt, CopyOp):
            return self._transform_copy(stmt)
        elif isinstance(stmt, GemmOp):
            return self._transform_gemm(stmt)
        elif isinstance(stmt, ClearOp):
            return self._transform_clear(stmt)
        elif isinstance(stmt, ParallelLoop):
            return self._transform_parallel_loop(stmt)
        elif isinstance(stmt, PipelinedLoop):
            return self._transform_pipelined_loop(stmt)
        elif isinstance(stmt, SerialLoop):
            return self._transform_serial_loop(stmt)
        elif isinstance(stmt, ast.AST):
            # Pass through Python AST nodes
            return None
        return None

    def _transform_alloc_shared(self, stmt: AllocShared) -> GluonAllocShared:
        """Transform shared memory allocation."""
        # Handle shape properly - if it's a tuple like ('block_M', 'block_N'), keep it as-is
        if isinstance(stmt.shape, tuple):
            shape = list(stmt.shape)
        elif isinstance(stmt.shape, list):
            shape = stmt.shape
        else:
            shape = [stmt.shape]
        dtype = self._map_dtype(stmt.dtype)

        # Infer layout based on shape and dtype for MMA compatibility
        layout = self._infer_shared_layout(shape, dtype)

        alloc = GluonAllocShared(
            name=stmt.name,
            shape=shape,
            dtype=dtype,
            layout=layout
        )
        self.current_kernel.shared_allocs.append(alloc)
        return alloc

    def _transform_alloc_fragment(self, stmt: AllocFragment) -> GluonRegisterTensor:
        """Transform fragment allocation to register tensor with MMA layout."""
        # Handle shape properly - if it's a tuple like ('block_M', 'block_N'), keep it as-is
        if isinstance(stmt.shape, tuple):
            shape = list(stmt.shape)
        elif isinstance(stmt.shape, list):
            shape = stmt.shape
        else:
            shape = [stmt.shape]
        dtype = self._map_dtype(stmt.dtype)

        # Use NVMMADistributedLayout for accumulators
        layout = self._infer_mma_layout(shape, dtype)

        reg_tensor = GluonRegisterTensor(
            name=stmt.name,
            shape=shape,
            dtype=dtype,
            layout=layout
        )
        self.current_kernel.register_tensors.append(reg_tensor)
        return reg_tensor

    def _transform_alloc_local(self, stmt: AllocLocal) -> GluonRegisterTensor:
        """Transform local allocation to register tensor."""
        # Handle shape properly - if it's a tuple like ('block_M', 'block_N'), keep it as-is
        if isinstance(stmt.shape, tuple):
            shape = list(stmt.shape)
        elif isinstance(stmt.shape, list):
            shape = stmt.shape
        else:
            shape = [stmt.shape]
        dtype = self._map_dtype(stmt.dtype)

        # Use BlockedLayout for local/thread memory
        layout = self._infer_blocked_layout(shape, dtype)

        reg_tensor = GluonRegisterTensor(
            name=stmt.name,
            shape=shape,
            dtype=dtype,
            layout=layout
        )
        self.current_kernel.register_tensors.append(reg_tensor)
        return reg_tensor

    def _infer_shared_layout(self, shape: List[int], dtype: str) -> str:
        """Infer NVMMASharedLayout for shared memory."""
        base_dtype = self._get_base_dtype(dtype)
        element_bits = 16 if base_dtype in ["float16", "bfloat16"] else 32

        # Default swizzle pattern for TMA/WGMMA compatibility
        swizzle_width = 128  # bytes

        return (f"gl.NVMMASharedLayout("
                f"swizzle_byte_width={swizzle_width}, "
                f"element_bitwidth={element_bits}, "
                f"rank={len(shape)})")

    def _infer_mma_layout(self, shape: List[int], dtype: str) -> str:
        """Infer layout for MMA operations (accumulators) - use shared memory layout."""
        # Use NVMMASharedLayout which works with gl.allocate_shared_memory
        base_dtype = self._get_base_dtype(dtype)
        element_bits = 16 if base_dtype in ["float16", "bfloat16"] else 32
        rank = len(shape) if shape else 2
        return (f"gl.NVMMASharedLayout("
                f"swizzle_byte_width=128, "
                f"element_bitwidth={element_bits}, "
                f"rank={rank})")

    def _infer_blocked_layout(self, shape: List[int], dtype: str) -> str:
        """Infer layout for general register tensors - use shared memory layout."""
        # Use NVMMASharedLayout which works with gl.allocate_shared_memory
        base_dtype = self._get_base_dtype(dtype)
        element_bits = 16 if base_dtype in ["float16", "bfloat16"] else 32
        rank = len(shape) if shape else 2
        return (f"gl.NVMMASharedLayout("
                f"swizzle_byte_width=128, "
                f"element_bitwidth={element_bits}, "
                f"rank={rank})")

    def _transform_copy(self, stmt: CopyOp) -> Optional[Union[GluonStmt, List[GluonStmt]]]:
        """Transform copy operation to TMA operations."""
        stmts = []

        # Determine if this is global->shared, shared->fragment, or shared->global
        src_is_global = stmt.src in self.buffer_to_descriptor
        dst_is_global = stmt.dst in self.buffer_to_descriptor

        if src_is_global:
            # Global to shared: TMA load
            desc_name = self.buffer_to_descriptor[stmt.src]
            barrier_name = self._fresh_name("bar")

            # Create barrier
            barrier = GluonBarrier(name=barrier_name, count=1)
            stmts.append(barrier)
            self.current_kernel.barriers.append(barrier)

            # Calculate offsets
            offsets = stmt.dst_indices if stmt.dst_indices else [0, 0]

            # TMA load
            tma_load = GluonTmaLoad(
                desc=desc_name,
                offsets=offsets,
                barrier=barrier_name,
                smem=stmt.dst
            )
            stmts.append(tma_load)

        elif dst_is_global:
            # Shared to global: TMA store
            desc_name = self.buffer_to_descriptor[stmt.dst]
            offsets = stmt.src_indices if stmt.src_indices else [0, 0]

            tma_store = GluonTmaStore(
                desc=desc_name,
                offsets=offsets,
                smem=stmt.src
            )
            stmts.append(tma_store)

        # Shared to fragment is handled by MMA operation directly

        return stmts if len(stmts) > 1 else (stmts[0] if stmts else None)

    def _transform_gemm(self, stmt: GemmOp) -> GluonMma:
        """Transform GEMM operation to WGMMA."""
        return GluonMma(
            A_desc=stmt.A,
            B_desc=stmt.B,
            acc=stmt.C,
            is_async=True
        )

    def _transform_clear(self, stmt: ClearOp) -> GluonClear:
        """Transform clear operation."""
        return GluonClear(buffer=stmt.buffer, dtype="gl.float32")

    def _transform_parallel_loop(self, stmt: ParallelLoop) -> GluonLoop:
        """Transform parallel loop."""
        transformed_body = []
        for s in stmt.body:
            transformed = self._transform_stmt(s)
            if transformed:
                transformed_body.append(transformed)
            elif isinstance(s, ast.AST):
                # Pass through raw AST nodes
                transformed_body.append(s)

        # Handle multi-dimensional parallel loops
        if stmt.extra_dims:
            # Create nested loops for extra dimensions
            inner_loop = GluonLoop(
                var=stmt.all_vars[-1] if stmt.all_vars else "j",
                start=0,
                end=stmt.extra_dims[0],
                step=1,
                body=transformed_body,
                is_pipelined=False
            )
            return GluonLoop(
                var=stmt.var,
                start=0,
                end=stmt.extent,
                step=1,
                body=[inner_loop],
                is_pipelined=False
            )

        return GluonLoop(
            var=stmt.var,
            start=0,
            end=stmt.extent,
            step=1,
            body=transformed_body,
            is_pipelined=False
        )

    def _transform_pipelined_loop(self, stmt: PipelinedLoop) -> List[GluonStmt]:
        """Transform pipelined loop with manual barrier management."""
        stmts = []

        # Create barrier for pipelining
        barrier_name = self._fresh_name("pipeline_bar")
        barrier = GluonBarrier(name=barrier_name, count=1)
        stmts.append(barrier)
        self.current_kernel.barriers.append(barrier)

        # Initialize barrier
        stmts.append(GluonBarrierInit(barrier=barrier_name, count=stmt.num_stages))

        # Create loop with pipelining
        loop = GluonLoop(
            var=stmt.var,
            start=0,
            end=stmt.extent,
            step=1,
            body=[],
            is_pipelined=True,
            num_stages=stmt.num_stages
        )

        # Transform loop body
        for s in stmt.body:
            transformed = self._transform_stmt(s)
            if transformed:
                if isinstance(transformed, list):
                    loop.body.extend(transformed)
                else:
                    loop.body.append(transformed)

        # Add barrier wait at end of loop iteration
        loop.body.append(GluonBarrierWait(barrier=barrier_name, phase=None))

        stmts.append(loop)
        return stmts

    def _transform_serial_loop(self, stmt: SerialLoop) -> GluonLoop:
        """Transform serial loop."""
        return GluonLoop(
            var=stmt.var,
            start=stmt.start,
            end=stmt.end,
            step=1,
            body=[self._transform_stmt(s) for s in stmt.body if self._transform_stmt(s)],
            is_pipelined=False
        )
