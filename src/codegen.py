"""
Generate Gluon kernel source code from Gluon AST.
"""

from typing import List, Any
from .transformer import (
    GluonKernel, GluonAllocShared, GluonRegisterTensor, GluonTensorDescriptor,
    GluonMma, GluonTmaLoad, GluonTmaStore, GluonBarrier, GluonBarrierInit,
    GluonBarrierWait, GluonLoop, GluonClear, GluonProgramId
)


class GluonCodeGenerator:
    """Generates Gluon kernel source code from Gluon AST."""

    def __init__(self):
        self.indent_level = 0
        self.lines = []

    def generate(self, kernel: GluonKernel) -> str:
        """Generate Gluon kernel source code."""
        self.lines = []
        self._generate_imports()
        self._generate_kernel(kernel)
        self._generate_launcher(kernel)
        return "\n".join(self.lines)

    def _indent(self) -> str:
        """Get current indentation string."""
        return "    " * self.indent_level

    def _generate_imports(self):
        """Generate import statements."""
        self.lines.extend([
            "import torch",
            "import triton",
            "from triton.experimental import gluon",
            "from triton.experimental.gluon import language as gl",
            "from triton.experimental.gluon.nvidia.hopper import TensorDescriptor",
            "from triton.experimental.gluon.language.nvidia.hopper import (",
            "    tma,",
            "    mbarrier,",
            "    fence_async_shared,",
            "    warpgroup_mma_init,",
            "    warpgroup_mma,",
            "    warpgroup_mma_wait,",
            ")",
            ""
        ])

    def _generate_kernel(self, kernel: GluonKernel):
        """Generate kernel function."""
        self.lines.append("@gluon.jit")

        # Build parameter list
        params = [f"num_warps: gl.constexpr = {kernel.num_warps}"]
        for desc in kernel.tensor_descriptors:
            params.append(f"{desc.name}: TensorDescriptor")
        for alloc in kernel.shared_allocs:
            params.append(f"{alloc.name}_layout: gl.constexpr")
        for barrier in kernel.barriers:
            params.append(f"{barrier.name}: gl.constexpr")

        self.lines.append(f"def {kernel.name}_kernel(")
        self.lines.append(",\n    ".join(params) + "):")
        self.indent_level += 1

        # Generate kernel body
        for stmt in kernel.body:
            self._generate_stmt(stmt)

        self.indent_level -= 1
        self.lines.append("")

    def _generate_stmt(self, stmt: Any):
        """Generate a single statement."""
        if isinstance(stmt, GluonAllocShared):
            self._generate_alloc_shared(stmt)
        elif isinstance(stmt, GluonRegisterTensor):
            self._generate_register_tensor(stmt)
        elif isinstance(stmt, GluonTensorDescriptor):
            pass  # Descriptors are parameters
        elif isinstance(stmt, GluonMma):
            self._generate_mma(stmt)
        elif isinstance(stmt, GluonTmaLoad):
            self._generate_tma_load(stmt)
        elif isinstance(stmt, GluonTmaStore):
            self._generate_tma_store(stmt)
        elif isinstance(stmt, GluonBarrier):
            self._generate_barrier(stmt)
        elif isinstance(stmt, GluonBarrierInit):
            self._generate_barrier_init(stmt)
        elif isinstance(stmt, GluonBarrierWait):
            self._generate_barrier_wait(stmt)
        elif isinstance(stmt, GluonLoop):
            self._generate_loop(stmt)
        elif isinstance(stmt, GluonClear):
            self._generate_clear(stmt)
        elif isinstance(stmt, GluonProgramId):
            self._generate_program_id(stmt)

    def _generate_alloc_shared(self, stmt: GluonAllocShared):
        """Generate shared memory allocation."""
        shape_str = str(stmt.shape).replace("'", "")
        self.lines.append(
            f"{self._indent()}{stmt.name} = gl.allocate_shared_memory("
            f"{stmt.dtype}, {shape_str}, {stmt.name}_layout)"
        )

    def _generate_register_tensor(self, stmt: GluonRegisterTensor):
        """Generate register tensor allocation."""
        shape_str = str(stmt.shape).replace("'", "")
        self.lines.append(
            f"{self._indent()}{stmt.name} = gl.zeros({shape_str}, "
            f"dtype={stmt.dtype}, layout={stmt.layout})"
        )

    def _generate_mma(self, stmt: GluonMma):
        """Generate MMA/WGMMA operation."""
        self.lines.append(
            f"{self._indent()}{stmt.acc} = warpgroup_mma("
            f"{stmt.A_desc}, {stmt.B_desc}, {stmt.acc}, is_async={stmt.is_async})"
        )
        self.lines.append(
            f"{self._indent()}{stmt.acc} = warpgroup_mma_wait("
            f"num_outstanding=0, deps=({stmt.acc},))"
        )

    def _generate_tma_load(self, stmt: GluonTmaLoad):
        """Generate TMA async load."""
        offsets_str = str(stmt.offsets).replace("'", "") if stmt.offsets else "[]"
        self.lines.append(
            f"{self._indent()}tma.async_copy_global_to_shared("
            f"{stmt.desc}, {offsets_str}, {stmt.barrier}, {stmt.smem})"
        )

    def _generate_tma_store(self, stmt: GluonTmaStore):
        """Generate TMA async store."""
        offsets_str = str(stmt.offsets).replace("'", "") if stmt.offsets else "[]"
        self.lines.append(
            f"{self._indent()}tma.async_copy_shared_to_global("
            f"{stmt.desc}, {offsets_str}, {stmt.smem})"
        )
        self.lines.append(f"{self._indent()}tma.store_wait(pendings=0)")

    def _generate_barrier(self, stmt: GluonBarrier):
        """Generate barrier allocation."""
        self.lines.append(
            f"{self._indent()}{stmt.name} = gl.allocate_shared_memory("
            f"gl.int64, [1], mbarrier.MBarrierLayout())"
        )

    def _generate_barrier_init(self, stmt: GluonBarrierInit):
        """Generate barrier initialization."""
        self.lines.append(
            f"{self._indent()}mbarrier.init({stmt.barrier}, count={stmt.count})"
        )

    def _generate_barrier_wait(self, stmt: GluonBarrierWait):
        """Generate barrier wait."""
        phase_str = str(stmt.phase) if stmt.phase is not None else "phase"
        self.lines.append(
            f"{self._indent()}mbarrier.wait({stmt.barrier}, phase={phase_str})"
        )

    def _generate_loop(self, stmt: GluonLoop):
        """Generate loop construct."""
        start_str = str(stmt.start)
        end_str = str(stmt.end)
        step_str = str(stmt.step)

        if stmt.is_pipelined:
            self.lines.append(f"{self._indent()}# Pipelined loop with {stmt.num_stages} stages")
            self.lines.append(f"{self._indent()}phase = 0")

        self.lines.append(
            f"{self._indent()}for {stmt.var} in range({start_str}, {end_str}, {step_str}):"
        )
        self.indent_level += 1

        for body_stmt in stmt.body:
            if body_stmt:
                self._generate_stmt(body_stmt)

        if stmt.is_pipelined:
            self.lines.append(f"{self._indent()}phase ^= 1  # Toggle phase")

        self.indent_level -= 1

    def _generate_clear(self, stmt: GluonClear):
        """Generate clear operation."""
        self.lines.append(f"{self._indent()}{stmt.buffer} = 0")

    def _generate_program_id(self, stmt: GluonProgramId):
        """Generate program ID access."""
        self.lines.append(
            f"{self._indent()}{stmt.var_name} = gl.program_id(axis={stmt.axis})"
        )

    def _generate_launcher(self, kernel: GluonKernel):
        """Generate host-side launcher function."""
        self.lines.append(f"def {kernel.name}(" + ", ".join(
            [f"{p['name']}: torch.Tensor" for p in kernel.params]
        ) + "):")
        self.indent_level += 1

        # Generate tensor descriptor creation
        for desc in kernel.tensor_descriptors:
            self.lines.append(
                f"{self._indent()}{desc.name} = TensorDescriptor({desc.tensor_name}, "
                f"{desc.shape})"
            )

        # Generate grid calculation
        grid_str = ", ".join([str(g) for g in kernel.grid]) if kernel.grid else "1"
        self.lines.append(f"{self._indent()}grid = ({grid_str},)")

        # Generate kernel launch
        desc_args = ", ".join([d.name for d in kernel.tensor_descriptors])
        layout_args = ", ".join([f"{a.name}_layout={a.layout}" for a in kernel.shared_allocs])
        barrier_args = ", ".join([b.name for b in kernel.barriers])

        all_args = ", ".join(filter(None, [desc_args, layout_args, barrier_args]))

        self.lines.append(
            f"{self._indent()}{kernel.name}_kernel[grid]({all_args})"
        )

        # Return output tensors
        output_params = [p['name'] for p in kernel.params if p.get('type') == 'tensor_descriptor']
        if output_params:
            self.lines.append(f"{self._indent()}return " + ", ".join(output_params))

        self.indent_level -= 1
        self.lines.append("")
