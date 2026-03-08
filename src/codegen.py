"""
Generate Gluon kernel source code from Gluon AST.
"""

import ast
import re
from typing import List, Any
from .transformer import (
    GluonKernel, GluonAllocShared, GluonRegisterTensor, GluonTensorDescriptor,
    GluonMma, GluonTmaLoad, GluonTmaStore, GluonBarrier, GluonBarrierInit,
    GluonBarrierWait, GluonLoop, GluonClear, GluonLocalCopy, GluonAtomicAdd, GluonProgramId
)


class GluonCodeGenerator:
    """Generates Gluon kernel source code from Gluon AST."""

    def __init__(self, use_pointer_mode: bool = False):
        self.indent_level = 0
        self.lines = []
        # Use pointer mode (tl.load/tl.store) instead of TensorDescriptor mode
        # This must be opt-in because the default translator path is intended to
        # emit Gluon descriptor/TMA-style code rather than Triton pointer code.
        self.use_pointer_mode = use_pointer_mode
        # Track tensor parameters for pointer mode
        self.tensor_params = []

    def generate(self, kernel: GluonKernel) -> str:
        """Generate Gluon kernel source code."""
        self.lines = []
        self._generate_imports()
        self._generate_kernel(kernel)
        self._generate_launcher(kernel)
        # Add helper function at the end (prefixed with _ to not interfere with launcher detection)
        self.lines.append("")
        self.lines.append("def _ceildiv(a, b):")
        self.lines.append("    return (a + b - 1) // b")
        return "\n".join(self.lines)

    def _indent(self) -> str:
        """Get current indentation string."""
        return "    " * self.indent_level

    def _generate_imports(self):
        """Generate import statements for Gluon 3.4.0."""
        self.lines.extend([
            "import torch",
            "import triton",
            "import triton.language as tl",
            "from triton.experimental import gluon",
            "from triton.experimental.gluon import language as gl",
            "from triton.experimental.gluon.nvidia.hopper import TensorDescriptor",
            "from triton.experimental.gluon.language.nvidia.hopper import (",
            "    tma,",
            "    mbarrier,",
            "    fence_async_shared,",
            ")",
            ""
        ])

    def _generate_kernel(self, kernel: GluonKernel):
        """Generate kernel function."""
        self.lines.append("@gluon.jit")

        # Build parameter list - all constexpr parameters need defaults
        # Order: non-constexpr params first, then constexpr with defaults
        params = []
        constexpr_params = []

        if self.use_pointer_mode:
            # Pointer mode: use raw pointers and dimensions
            # Note: transformer converts 'annotation' to 'type', so check 'type' field
            self.tensor_params = [p for p in kernel.params if p.get('type') == 'tensor_descriptor']
            for param in self.tensor_params:
                params.append(f"{param['name']}_ptr: tl.pointer_type")
            # Add dimension parameters
            if self.tensor_params:
                params.append("M: tl.constexpr")
                params.append("N: tl.constexpr")
                params.append("K: tl.constexpr")
        else:
            # TensorDescriptor mode (for TMA loads/stores)
            for desc in kernel.tensor_descriptors:
                params.append(f"{desc.name}: 'TensorDescriptor'")

        # constexpr parameters with defaults
        constexpr_params.append(f"num_warps: gl.constexpr = {kernel.num_warps}")

        # Extract constants from kernel or use defaults
        block_M = kernel.block_M if hasattr(kernel, 'block_M') and kernel.block_M else 32
        block_N = kernel.block_N if hasattr(kernel, 'block_N') and kernel.block_N else 32
        block_K = kernel.block_K if hasattr(kernel, 'block_K') and kernel.block_K else 32
        in_dtype = kernel.in_dtype if hasattr(kernel, 'in_dtype') and kernel.in_dtype else "gl.float32"
        out_dtype = kernel.out_dtype if hasattr(kernel, 'out_dtype') and kernel.out_dtype else "gl.float32"

        constexpr_params.append(f"block_M: gl.constexpr = {block_M}")
        constexpr_params.append(f"block_N: gl.constexpr = {block_N}")
        # Add missing block_* symbols referenced in shapes (e.g., block_K).
        for sym in self._collect_block_symbols(kernel):
            if sym not in {"block_M", "block_N"}:
                constexpr_params.append(f"{sym}: gl.constexpr = 32")

        # Add shared memory layouts for both pointer and TensorDescriptor modes
        # (needed for allocate_shared_memory calls in kernel body)
        for alloc in kernel.shared_allocs:
            element_bits = 32
            if 'float16' in alloc.dtype or 'bfloat16' in alloc.dtype:
                element_bits = 16
            rank = len(alloc.shape) if alloc.shape else 2
            constexpr_params.append(f"{alloc.name}_layout: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth={element_bits}, rank={rank})")
        for reg in getattr(kernel, 'register_tensors', []):
            element_bits = 32
            if 'float16' in reg.dtype or 'bfloat16' in reg.dtype:
                element_bits = 16
            rank = len(reg.shape) if reg.shape else 2
            constexpr_params.append(f"{reg.name}_layout: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth={element_bits}, rank={rank})")
        for barrier in kernel.barriers:
            constexpr_params.append(f"{barrier.name}: gl.constexpr = 0")

        # Combine: non-constexpr first, then constexpr with defaults
        all_params = params + constexpr_params

        self.lines.append(f"def {kernel.name}_kernel(")
        if all_params:
            self.lines.append("    " + ",\n    ".join(all_params))
        self.lines.append("):")
        self.indent_level += 1

        # Generate kernel body
        for stmt in kernel.body:
            self._generate_stmt(stmt, in_dtype=in_dtype, out_dtype=out_dtype)

        self.indent_level -= 1
        self.lines.append("")

    def _generate_stmt(self, stmt: Any, in_dtype: str = "gl.float32", out_dtype: str = "gl.float32"):
        """Generate a single statement."""
        if isinstance(stmt, GluonAllocShared):
            self._generate_alloc_shared(stmt, in_dtype=in_dtype, out_dtype=out_dtype)
        elif isinstance(stmt, GluonRegisterTensor):
            self._generate_register_tensor(stmt, in_dtype=in_dtype, out_dtype=out_dtype)
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
            self._generate_loop(stmt, in_dtype=in_dtype, out_dtype=out_dtype)
        elif isinstance(stmt, GluonClear):
            self._generate_clear(stmt)
        elif isinstance(stmt, GluonLocalCopy):
            self._generate_local_copy(stmt)
        elif isinstance(stmt, GluonAtomicAdd):
            self._generate_atomic_add(stmt)
        elif isinstance(stmt, GluonProgramId):
            self._generate_program_id(stmt)

    def _generate_alloc_shared(self, stmt: GluonAllocShared, in_dtype: str = "gl.float32", out_dtype: str = "gl.float32"):
        """Generate shared memory allocation."""
        # Format shape as tuple of variable names, not strings
        if stmt.shape and len(stmt.shape) == 1 and isinstance(stmt.shape[0], tuple):
            # Shape is [(block_M, block_N)] - unwrap it and format as (block_M, block_N)
            shape_str = "(" + ", ".join(str(x) for x in stmt.shape[0]) + ")"
        elif stmt.shape:
            # Shape is a list/tuple - format without quotes
            shape_str = "(" + ", ".join(str(x) for x in stmt.shape) + ")"
        else:
            shape_str = "(1,)"
        # Use the actual dtype from the statement, not the parameter defaults
        actual_dtype = self._normalize_dtype(stmt.dtype if stmt.dtype else "gl.float32", in_dtype, out_dtype)
        self.lines.append(
            f"{self._indent()}{stmt.name} = gl.allocate_shared_memory("
            f"{actual_dtype}, {shape_str}, {stmt.name}_layout)"
        )

    def _generate_register_tensor(self, stmt: GluonRegisterTensor, in_dtype: str = "gl.float32", out_dtype: str = "gl.float32"):
        """Generate register tensor allocation."""
        # Format shape as tuple of variable names, not strings
        if stmt.shape and len(stmt.shape) == 1 and isinstance(stmt.shape[0], tuple):
            # Shape is [(block_M, block_N)] - unwrap it and format as (block_M, block_N)
            shape_str = "(" + ", ".join(str(x) for x in stmt.shape[0]) + ")"
        elif stmt.shape and len(stmt.shape) > 1:
            # Shape is a list like ['block_M', 'block_N'] - format as (block_M, block_N)
            shape_str = "(" + ", ".join(str(x) for x in stmt.shape) + ")"
        else:
            shape_str = str(stmt.shape).replace("'", "") if stmt.shape else "(1,)"
        actual_dtype = self._normalize_dtype(stmt.dtype if stmt.dtype else out_dtype, in_dtype, out_dtype)
        # For register tensors (fragments), use shared memory with NVMMASharedLayout
        # This matches the Gluon example pattern
        layout_name = f"{stmt.name}_layout"
        self.lines.append(
            f"{self._indent()}{stmt.name} = gl.allocate_shared_memory("
            f"{actual_dtype}, {shape_str}, {layout_name})"
        )

    def _generate_mma(self, stmt: GluonMma):
        """Generate MMA/WGMMA operation using tl.dot for Gluon 3.4.0."""
        self.lines.append(
            f"{self._indent()}# MMA operation: {stmt.acc} = {stmt.A_desc} @ {stmt.B_desc}"
        )
        # Gluon 3.4.0 doesn't have warpgroup_mma, use tl.dot instead
        # Note: tl.dot may have different performance characteristics
        self.lines.append(
            f"{self._indent()}# Using tl.dot (Gluon 3.4.0 compatible)"
        )
        self.lines.append(
            f"{self._indent()}{stmt.acc} = tl.dot({stmt.A_desc}, {stmt.B_desc}, {stmt.acc})"
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

    def _generate_loop(self, stmt: GluonLoop, in_dtype: str = "gl.float32", out_dtype: str = "gl.float32"):
        """Generate loop construct."""
        start_str = str(stmt.start)
        end_str = str(stmt.end)
        step_str = str(stmt.step)

        # Replace ceildiv(a, b) with inline computation (a + b - 1) // b
        # Note: cannot use helper functions inside @jit kernel
        import re
        def replace_ceildiv(expr):
            # Match ceildiv(arg1, arg2) and replace with (arg1 + arg2 - 1) // arg2
            pattern = r'\bceildiv\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)'
            def replacer(match):
                a, b = match.group(1).strip(), match.group(2).strip()
                return f'(({a} + {b} - 1) // {b})'
            return re.sub(pattern, replacer, expr)

        end_str = replace_ceildiv(end_str)
        start_str = replace_ceildiv(start_str)
        step_str = replace_ceildiv(step_str)

        if stmt.is_pipelined:
            self.lines.append(f"{self._indent()}# Pipelined loop with {stmt.num_stages} stages")
            self.lines.append(f"{self._indent()}phase = 0")

        self.lines.append(
            f"{self._indent()}for {stmt.var} in range({start_str}, {end_str}, {step_str}):"
        )
        self.indent_level += 1

        # Generate loop body statements
        generated_body = False
        for body_stmt in stmt.body:
            if body_stmt:
                if isinstance(body_stmt, ast.AST):
                    # Convert Python AST to source code
                    try:
                        # Use ast.unparse (Python 3.9+) or fall back to astor
                        if hasattr(ast, 'unparse'):
                            source = ast.unparse(body_stmt).strip()
                        else:
                            import astor
                            source = astor.to_source(body_stmt).strip()
                        self.lines.append(f"{self._indent()}{source}")
                        generated_body = True
                    except Exception:
                        pass
                else:
                    self._generate_stmt(body_stmt, in_dtype=in_dtype, out_dtype=out_dtype)
                    generated_body = True

        # If no body statements were generated, add a pass statement
        if not generated_body:
            self.lines.append(f"{self._indent()}pass  # Empty loop body")

        if stmt.is_pipelined:
            self.lines.append(f"{self._indent()}phase ^= 1  # Toggle phase")

        self.indent_level -= 1

    def _generate_clear(self, stmt: GluonClear):
        """Generate clear operation."""
        self.lines.append(f"{self._indent()}{stmt.buffer} = 0")

    def _generate_local_copy(self, stmt: GluonLocalCopy):
        """Generate in-kernel local/shared/register copy."""
        self.lines.append(f"{self._indent()}{stmt.dst} = {stmt.src}")

    def _generate_atomic_add(self, stmt: GluonAtomicAdd):
        """Generate atomic add into global memory."""
        if stmt.target_indices:
            target_indices = ", ".join(str(i) for i in stmt.target_indices)
            target_expr = f"{stmt.target}[{target_indices}]"
        else:
            target_expr = stmt.target
        if stmt.value_indices:
            value_indices = ", ".join(str(i) for i in stmt.value_indices)
            value_expr = f"{stmt.value}[{value_indices}]"
        else:
            value_expr = stmt.value
        self.lines.append(f"{self._indent()}tl.atomic_add({target_expr}, {value_expr})")

    def _generate_program_id(self, stmt: GluonProgramId):
        """Generate program ID access."""
        self.lines.append(
            f"{self._indent()}{stmt.var_name} = gl.program_id(axis={stmt.axis})"
        )

    def _generate_launcher(self, kernel: GluonKernel):
        """Generate host-side launcher function."""
        # Extract dimensions from kernel or use defaults
        block_M = kernel.block_M if hasattr(kernel, 'block_M') and kernel.block_M else 32
        block_N = kernel.block_N if hasattr(kernel, 'block_N') and kernel.block_N else 32
        block_K = kernel.block_K if hasattr(kernel, 'block_K') and kernel.block_K else 32

        # Define constant values extracted from kernel (before launcher)
        self.lines.append(f"# Kernel constants for {kernel.name}")
        self.lines.append(f"BLOCK_M = {block_M}")
        self.lines.append(f"BLOCK_N = {block_N}")
        self.lines.append(f"BLOCK_K = {block_K}")
        self.lines.append("")

        # Launcher function takes torch.Tensor arguments (generate this first so it's detected as the launcher)
        tensor_params = [p for p in kernel.params]
        if tensor_params:
            self.lines.append(f"def {kernel.name}(")
            self.lines.append("    " + ",\n    ".join([f"{p['name']}: torch.Tensor" for p in tensor_params]))
            self.lines.append("):")
        else:
            self.lines.append(f"def {kernel.name}():")
        self.indent_level += 1

        if tensor_params:
            first_tensor = tensor_params[0]['name']
            # Calculate grid based on tensor shape
            self.lines.append(f"{self._indent()}M = {first_tensor}.shape[0] if {first_tensor}.dim() > 0 else 1")
            self.lines.append(f"{self._indent()}N = {first_tensor}.shape[1] if {first_tensor}.dim() > 1 else 1")
            if len(tensor_params) >= 2:
                second_tensor = tensor_params[1]['name']
                self.lines.append(f"{self._indent()}K = {second_tensor}.shape[0] if {second_tensor}.dim() > 0 else 1")
            else:
                self.lines.append(f"{self._indent()}K = M")  # Assume square matrix
            if kernel.grid:
                self.lines.append(f"{self._indent()}grid = ({self._format_grid(kernel.grid)})")
            else:
                self.lines.append(f"{self._indent()}grid = (_ceildiv(N, BLOCK_N), _ceildiv(M, BLOCK_M))")

            if self.use_pointer_mode:
                # Pass pointers and dimensions to kernel
                # Note: kernel expects {name}_ptr parameters
                ptr_args = ", ".join([f"{p['name']}" for p in tensor_params])
                dim_args = "M=M, N=N, K=K"
                all_args = ", ".join([ptr_args, dim_args])
                self.lines.append(
                    f"{self._indent()}{kernel.name}_kernel[grid]({all_args})"
                )
            else:
                # TensorDescriptor mode
                for desc in kernel.tensor_descriptors:
                    tensor_name = desc.tensor_name
                    self.lines.append(f"{self._indent()}# Create tensor descriptor for {tensor_name}")
                    self.lines.append(
                        f"{self._indent()}{desc.name} = TensorDescriptor("
                        f"{tensor_name}, "
                        f"list({tensor_name}.shape), "
                        f"[s if i == 0 else 1 for i, s in enumerate({tensor_name}.shape)], "  # strides
                    )
                    self.lines.append(
                        f"{self._indent()}    [{block_M}, {block_N}], "  # block_shape
                    )
                    self.lines.append(
                        f"{self._indent()}    gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)"
                    )
                    self.lines.append(f"{self._indent()})")
                    self.lines.append("")

                # Generate kernel launch for TensorDescriptor mode
                desc_args = ", ".join([d.name for d in kernel.tensor_descriptors])
                shared_layout_args = ", ".join([f"{a.name}_layout={a.layout}" for a in kernel.shared_allocs])
                reg_layout_args = ", ".join([f"{r.name}_layout={r.layout}" for r in getattr(kernel, 'register_tensors', [])])
                all_args = ", ".join(filter(None, [desc_args, shared_layout_args, reg_layout_args]))

                self.lines.append(
                    f"{self._indent()}{kernel.name}_kernel[grid]({all_args})"
                )
        else:
            # No tensor params - use default grid
            grid_str = ", ".join([str(g) for g in kernel.grid]) if kernel.grid else "1"
            self.lines.append(f"{self._indent()}grid = ({grid_str},)")
            self.lines.append(
                f"{self._indent()}{kernel.name}_kernel[grid]()"
            )

        # Return output tensors (last param is usually the output)
        if tensor_params:
            self.lines.append(f"{self._indent()}return {tensor_params[-1]['name']}")

        self.indent_level -= 1
        self.lines.append("")

    def _normalize_dtype(self, dtype_str: str, in_dtype: str, out_dtype: str) -> str:
        """Normalize symbolic dtype placeholders to concrete Gluon dtypes."""
        token = str(dtype_str).strip()
        if token in {"gl.in_dtype", "in_dtype", "gl.dtype", "dtype"}:
            return in_dtype
        if token in {"gl.out_dtype", "out_dtype"}:
            return out_dtype
        if token in {"gl.accum_dtype", "accum_dtype"}:
            return "gl.float32"
        return token

    def _collect_block_symbols(self, kernel: GluonKernel) -> set:
        """Collect block_* symbols referenced by shapes in generated AST."""
        syms = set()
        candidates = list(kernel.shared_allocs) + list(getattr(kernel, "register_tensors", []))
        for node in candidates:
            shape = getattr(node, "shape", None)
            if shape is None:
                continue
            text = str(shape)
            for m in re.findall(r"\bblock_[A-Za-z0-9_]+\b", text):
                syms.add(m)
        return syms

    def _format_grid(self, grid: List[Any]) -> str:
        """Format grid expressions for the host launcher."""
        rendered = []
        for dim in grid:
            expr = str(dim)
            expr = expr.replace("block_M", "BLOCK_M").replace("block_N", "BLOCK_N").replace("block_K", "BLOCK_K")
            expr = re.sub(r"\bceildiv\s*\(", "_ceildiv(", expr)
            rendered.append(expr)
        return ", ".join(rendered)
