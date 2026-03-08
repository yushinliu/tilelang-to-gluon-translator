"""
Generate Gluon kernel source code using pointer mode (tl.load/tl.store).

This is an alternative code generator that uses tl.load/tl.store + tl.dot
instead of TensorDescriptor + TMA operations. This is required for Gluon 3.4.0
compatibility since shared_memory_descriptor is not supported by tl.dot.
"""

import ast
import re
from typing import List, Any, Optional
from .transformer import (
    GluonKernel, GluonAllocShared, GluonRegisterTensor, GluonTensorDescriptor,
    GluonMma, GluonTmaLoad, GluonTmaStore, GluonBarrier, GluonBarrierInit,
    GluonBarrierWait, GluonLoop, GluonClear, GluonLocalCopy, GluonAtomicAdd, GluonProgramId
)


class GluonPointerCodeGenerator:
    """
    Generates Gluon kernel source code using pointer mode.

    This generator uses tl.load/tl.store for memory operations and tl.dot for
    matrix multiplication, avoiding shared_memory_descriptor which is not
    compatible with tl.dot in Gluon 3.4.0.
    """

    def __init__(self):
        self.indent_level = 0
        self.lines = []
        self.kernel = None
        self.block_M = 32
        self.block_N = 32
        self.block_K = 32
        self.has_runtime_dot_operand_layout = self._detect_dot_operand_layout()
        self.has_runtime_mma_v2 = self._detect_mma_v2()

    def generate(self, kernel: GluonKernel) -> str:
        """Generate Gluon kernel source code using pointer mode."""
        self.lines = []
        self.kernel = kernel
        self.allocs = {
            alloc.name: alloc for alloc in list(kernel.shared_allocs) + list(getattr(kernel, "register_tensors", []))
        }
        self._generate_imports()
        self._generate_kernel(kernel)
        self._generate_launcher(kernel)
        # Add helper function at the end
        self.lines.append("")
        self.lines.append("def _ceildiv(a, b):")
        self.lines.append("    return (a + b - 1) // b")
        return "\n".join(self.lines)

    def _indent(self) -> str:
        """Get current indentation string."""
        return "    " * self.indent_level

    def _generate_imports(self):
        """Generate import statements."""
        self.lines.extend([
            "import torch",
            "import triton",
            "import triton.language as tl",
            "from triton.experimental import gluon",
            "from triton.experimental.gluon import language as gl",
            ""
        ])
        if self.has_runtime_mma_v2:
            self.lines.insert(-1, "from triton.experimental.gluon.language.nvidia.ampere import mma_v2")
        if not self.has_runtime_dot_operand_layout:
            self.lines.insert(-1, "from tilelang_to_gluon_translator.gluon_compat import DotOperandLayout as _DotOperandLayout")
            self.lines.insert(-1, "DotOperandLayout = tl.constexpr(_DotOperandLayout)")

    def _generate_kernel(self, kernel: GluonKernel):
        """Generate kernel function using pointer mode."""
        self.lines.append("@gluon.jit")

        # Extract constants
        self.block_M = kernel.block_M if hasattr(kernel, 'block_M') and kernel.block_M else 128
        self.block_N = kernel.block_N if hasattr(kernel, 'block_N') and kernel.block_N else 128
        self.block_K = 32  # Default K block size

        # Get tensor parameters (transformer sets 'type' field to 'tensor_descriptor')
        tensor_params = [p for p in kernel.params if p.get('type') == 'tensor_descriptor' or p.get('annotation', {}).get('type') == 'Tensor']

        # Build parameter list - pass dimensions as constexpr from host
        params = []
        for param in tensor_params:
            params.append(f"{param['name']}: gl.pointer_type")

        # Add dimension parameters as constexpr (passed from launcher)
        constexpr_params = [
            f"M: gl.constexpr",
            f"N: gl.constexpr",
            f"K: gl.constexpr",
            f"num_warps: gl.constexpr = {kernel.num_warps}",
            f"BLOCK_M: gl.constexpr = {self.block_M}",
            f"BLOCK_N: gl.constexpr = {self.block_N}",
            f"BLOCK_K: gl.constexpr = {self.block_K}",
        ]

        # Add additional block symbols
        for sym in self._collect_block_symbols(kernel):
            if sym not in {"block_M", "block_N", "block_K", "BLOCK_M", "BLOCK_N", "BLOCK_K"}:
                constexpr_params.append(f"{sym}: gl.constexpr = 32")

        all_params = params + constexpr_params

        self.lines.append(f"def {kernel.name}_kernel(")
        if all_params:
            self.lines.append("    " + ",\n    ".join(all_params))
        self.lines.append("):")
        self.indent_level += 1

        # Store tensor params for later use
        self.tensor_params = tensor_params

        # Get program IDs
        self.lines.append(f"{self._indent()}pid_m = gl.program_id(0)")
        self.lines.append(f"{self._indent()}pid_n = gl.program_id(1)")

        # Compute offsets - use simple integer arithmetic for pointer mode
        # Note: In pointer mode, we don't need tl.arange with layout
        self.lines.append(f"{self._indent()}base_m = pid_m * BLOCK_M")
        self.lines.append(f"{self._indent()}base_n = pid_n * BLOCK_N")

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
        elif isinstance(stmt, GluonMma):
            self._generate_mma(stmt)
        elif isinstance(stmt, GluonTmaLoad):
            self._generate_tma_load(stmt)
        elif isinstance(stmt, GluonTmaStore):
            self._generate_tma_store(stmt)
        elif isinstance(stmt, GluonLoop):
            self._generate_loop(stmt)
        elif isinstance(stmt, GluonClear):
            self._generate_clear(stmt)
        elif isinstance(stmt, GluonLocalCopy):
            self._generate_local_copy(stmt)
        elif isinstance(stmt, GluonAtomicAdd):
            self._generate_atomic_add(stmt)
        elif isinstance(stmt, GluonProgramId):
            self._generate_program_id(stmt)
        elif isinstance(stmt, ast.AST):
            self._generate_raw_ast(stmt)

    def _generate_alloc_shared(self, stmt: GluonAllocShared):
        """Generate shared memory allocation (using local accumulator in pointer mode)."""
        # In pointer mode, we use local accumulators instead of shared memory
        # The allocation is skipped since we directly load from global memory
        pass

    def _generate_register_tensor(self, stmt: GluonRegisterTensor):
        """Generate register tensor allocation (local accumulator)."""
        # Register fragments must preserve the transformed layout metadata.
        shape_str = str(stmt.shape).replace("'", "") if stmt.shape else "(1,)"
        layout = stmt.layout if getattr(stmt, "layout", None) else (
            self._blocked_layout_expr(*stmt.shape[:2]) if len(stmt.shape) >= 2 else "gl.BlockedLayout([1], [32], [4], [0])"
        )
        self.lines.append(f"{self._indent()}# Initialize register tensor")
        self.lines.append(f"{self._indent()}{stmt.name} = gl.full({shape_str}, 0, {stmt.dtype}, layout={layout})")

    def _generate_mma(self, stmt: GluonMma):
        """Generate MMA operation using tl.dot."""
        # In pointer mode, we expect data to be loaded by _generate_tma_load
        # Use the loaded shared memory variables (A_shared, B_shared)
        # If they don't exist, fall back to loading directly from global memory

        # Determine the source variables for dot operation
        A_var = stmt.A_desc if hasattr(stmt, 'A_desc') and stmt.A_desc else "a"
        B_var = stmt.B_desc if hasattr(stmt, 'B_desc') and stmt.B_desc else "b"

        # Check if we have shared memory variables loaded by _generate_tma_load
        # The transformer sets A_desc/B_desc to descriptor names like "A_desc"
        # In pointer mode, we want to use the actual loaded variables
        if "_desc" in A_var.lower():
            # Fallback: try to find the corresponding shared memory variable
            # This assumes T.copy loads to A_shared, B_shared
            A_var = "A_shared" if any("A_shared" in line for line in self.lines) else A_var
        if "_desc" in B_var.lower():
            B_var = "B_shared" if any("B_shared" in line for line in self.lines) else B_var

        # If no shared memory loading happened, load directly from global memory
        if "_desc" in A_var.lower() or A_var == "a":
            tensor_params = [p['name'] for p in self.kernel.params if p.get('annotation', {}).get('type') == 'Tensor']
            A_name = tensor_params[0] if len(tensor_params) > 0 else "A"
            B_name = tensor_params[1] if len(tensor_params) > 1 else "B"

            self.lines.append(f"{self._indent()}# Load A and B blocks for current k")
            self.lines.append(f"{self._indent()}offs_k = gl.arange(0, BLOCK_K)")
            self.lines.append(f"{self._indent()}a_ptrs = {A_name} + (offs_m[:, None] * K + (k + offs_k[None, :]))")
            self.lines.append(f"{self._indent()}a_mask = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)")
            self.lines.append(f"{self._indent()}a = gl.load(a_ptrs, mask=a_mask)")

            self.lines.append(f"{self._indent()}b_ptrs = {B_name} + ((k + offs_k[:, None]) * N + offs_n[None, :]))")
            self.lines.append(f"{self._indent()}b_mask = ((k + offs_k[:, None]) < K) & (offs_n[None, :] < N)")
            self.lines.append(f"{self._indent()}b = gl.load(b_ptrs, mask=b_mask)")

            A_var = "a"
            B_var = "b"

        self.lines.append(f"{self._indent()}# MMA: {stmt.acc} = dot({A_var}, {B_var}, acc={stmt.acc})")
        self.lines.append(f"{self._indent()}acc_layout: gl.constexpr = {stmt.acc}.type.layout")
        layout_ctor = "gl.DotOperandLayout" if self.has_runtime_dot_operand_layout else "DotOperandLayout"
        k_width = self._dot_k_width_literal(A_var, B_var)
        self.lines.append(
            f"{self._indent()}a_layout: gl.constexpr = {layout_ctor}("
            f"parent=acc_layout, operand_index=0, k_width={k_width})"
        )
        self.lines.append(
            f"{self._indent()}b_layout: gl.constexpr = {layout_ctor}("
            f"parent=acc_layout, operand_index=1, k_width={k_width})"
        )
        self.lines.append(f"{self._indent()}{A_var}_dot = gl.convert_layout({A_var}, a_layout)")
        self.lines.append(f"{self._indent()}{B_var}_dot = gl.convert_layout({B_var}, b_layout)")
        if self.has_runtime_mma_v2:
            self.lines.append(f"{self._indent()}{stmt.acc} = mma_v2({A_var}_dot, {B_var}_dot, {stmt.acc})")
        else:
            self.lines.append(f"{self._indent()}{stmt.acc}_dot = tl.dot({A_var}_dot, {B_var}_dot, acc={stmt.acc})")
            self.lines.append(f"{self._indent()}{stmt.acc} = gl.convert_layout({stmt.acc}_dot, acc_layout)")

    def _fix_expr(self, expr):
        """Convert expression string from parser format to Python format.

        Parser returns 'by mult 128' but we need 'by * 128'.
        Also maps TileLang variable names to generated variable names.
        """
        if not isinstance(expr, str):
            return str(expr)
        # Replace operator names with actual operators
        replacements = {
            ' mult ': ' * ',
            ' add ': ' + ',
            ' sub ': ' - ',
            ' div ': ' // ',
            'Mult': '*',
            'Add': '+',
            'Sub': '-',
            'Div': '//',
        }
        result = expr
        for old, new in replacements.items():
            result = result.replace(old, new)
        return result

    def _generate_tma_load(self, stmt: GluonTmaLoad):
        """Generate pointer mode load using tl.load."""
        if stmt.src_tensor:
            src_idx = stmt.src_indices if stmt.src_indices else ["0", "0"]
            src_idx = [self._fix_expr(i) for i in src_idx]
            self.lines.append(f"{self._indent()}# Load {stmt.src_tensor} to {stmt.smem}")
            alloc = self.allocs.get(stmt.smem)
            shape = getattr(alloc, "shape", []) if alloc is not None else []
            if len(shape) >= 2 and len(src_idx) >= 2:
                rows = self._fix_expr(shape[0])
                cols = self._fix_expr(shape[1])
                row_base = src_idx[0]
                col_base = src_idx[1]
                stride = self._row_stride_expr(stmt.src_tensor)
                layout = self._blocked_layout_expr(rows, cols)
                self.lines.append(
                    f"{self._indent()}rows_{stmt.smem} = {row_base} + gl.arange(0, {rows}, layout=gl.SliceLayout(1, {layout}))"
                )
                self.lines.append(
                    f"{self._indent()}cols_{stmt.smem} = {col_base} + gl.arange(0, {cols}, layout=gl.SliceLayout(0, {layout}))"
                )
                self.lines.append(
                    f"{self._indent()}ptr_{stmt.smem} = {stmt.src_tensor} + rows_{stmt.smem}[:, None] * {stride} + cols_{stmt.smem}[None, :]"
                )
                self.lines.append(
                    f"{self._indent()}mask_{stmt.smem} = (rows_{stmt.smem}[:, None] < {self._dim_expr(stmt.src_tensor, 0)}) & "
                    f"(cols_{stmt.smem}[None, :] < {self._dim_expr(stmt.src_tensor, 1)})"
                )
                self.lines.append(
                    f"{self._indent()}{stmt.smem} = gl.load(ptr_{stmt.smem}, mask=mask_{stmt.smem})"
                )
            else:
                if len(src_idx) >= 2:
                    self.lines.append(
                        f"{self._indent()}ptr_{stmt.smem} = {stmt.src_tensor} + ({src_idx[0]}) * {self._row_stride_expr(stmt.src_tensor)} + ({src_idx[1]})"
                    )
                else:
                    self.lines.append(f"{self._indent()}ptr_{stmt.smem} = {stmt.src_tensor} + {src_idx[0] if src_idx else '0'}")
                self.lines.append(f"{self._indent()}{stmt.smem} = gl.load(ptr_{stmt.smem})")
        else:
            pass

    def _generate_tma_store(self, stmt: GluonTmaStore):
        """Generate pointer mode store using tl.store."""
        if stmt.dst_tensor:
            # Pointer mode: generate tl.store
            dst_idx = stmt.dst_indices if stmt.dst_indices else ["0", "0"]
            # Fix expressions (convert 'mult' to '*', etc.)
            dst_idx = [self._fix_expr(i) for i in dst_idx]
            self.lines.append(f"{self._indent()}# Store {stmt.smem} to {stmt.dst_tensor}")
            alloc = self.allocs.get(stmt.smem)
            shape = getattr(alloc, "shape", None) or []
            if len(dst_idx) >= 2 and len(shape) >= 2:
                rows = shape[0]
                cols = shape[1]
                layout = self._tensor_layout_expr(stmt.smem)
                self.lines.append(
                    f"{self._indent()}rows_out = ({dst_idx[0]}) + gl.arange(0, {rows}, layout=gl.SliceLayout(1, {layout}))"
                )
                self.lines.append(
                    f"{self._indent()}cols_out = ({dst_idx[1]}) + gl.arange(0, {cols}, layout=gl.SliceLayout(0, {layout}))"
                )
                self.lines.append(
                    f"{self._indent()}ptr_out = {stmt.dst_tensor} + (rows_out[:, None] * N + cols_out[None, :])"
                )
                self.lines.append(
                    f"{self._indent()}mask_out = (rows_out[:, None] < M) & (cols_out[None, :] < N)"
                )
                self.lines.append(f"{self._indent()}gl.store(ptr_out, {stmt.smem}, mask=mask_out)")
            else:
                if len(dst_idx) >= 2:
                    self.lines.append(f"{self._indent()}ptr_out = {stmt.dst_tensor} + ({dst_idx[0]}) * N + ({dst_idx[1]})")
                else:
                    self.lines.append(f"{self._indent()}ptr_out = {stmt.dst_tensor} + {dst_idx[0] if dst_idx else '0'}")
                self.lines.append(f"{self._indent()}gl.store(ptr_out, {stmt.smem})")
        else:
            # Fallback to simple store
            tensor_params = [p['name'] for p in self.kernel.params]
            C_name = tensor_params[-1] if tensor_params else "C"
            self.lines.append(f"{self._indent()}# Store result to global memory")
            self.lines.append(f"{self._indent()}c_ptrs = {C_name} + (offs_m[:, None] * N + offs_n[None, :])")
            self.lines.append(f"{self._indent()}c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)")
            self.lines.append(f"{self._indent()}gl.store(c_ptrs, {stmt.smem}, mask=c_mask)")

    def _generate_loop(self, stmt: GluonLoop):
        """Generate loop construct."""
        block_atomic = self._match_block_atomic_add(stmt)
        if block_atomic is not None:
            self._generate_block_atomic_add(*block_atomic)
            return

        start_str = str(stmt.start)
        end_str = str(stmt.end)
        step_str = str(stmt.step)

        # Replace ceildiv with inline computation
        import re
        def replace_ceildiv(expr):
            pattern = r'\bceildiv\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)'
            def replacer(match):
                a, b = match.group(1).strip(), match.group(2).strip()
                return f'(({a} + {b} - 1) // {b})'
            return re.sub(pattern, replacer, expr)

        end_str = replace_ceildiv(end_str)
        start_str = replace_ceildiv(start_str)
        step_str = replace_ceildiv(step_str)

        self.lines.append(
            f"{self._indent()}for {stmt.var} in range({start_str}, {end_str}, {step_str}):"
        )
        self.indent_level += 1

        for body_stmt in stmt.body:
            if body_stmt:
                if isinstance(body_stmt, ast.AST):
                    try:
                        if hasattr(ast, 'unparse'):
                            source = ast.unparse(body_stmt).strip()
                        else:
                            import astor
                            source = astor.to_source(body_stmt).strip()
                        self.lines.append(f"{self._indent()}{source}")
                    except Exception:
                        pass
                else:
                    self._generate_stmt(body_stmt)

        self.indent_level -= 1

    def _generate_raw_ast(self, stmt: ast.AST):
        """Emit preserved Python AST statements."""
        try:
            if hasattr(ast, "unparse"):
                source = ast.unparse(stmt).strip()
            else:
                import astor
                source = astor.to_source(stmt).strip()
            if source:
                self.lines.append(f"{self._indent()}{source}")
        except Exception:
            pass

    def _generate_clear(self, stmt: GluonClear):
        """Generate clear operation."""
        layout = self._tensor_layout_expr(stmt.buffer)
        self.lines.append(
            f"{self._indent()}{stmt.buffer} = gl.full({stmt.buffer}.shape, 0, {stmt.buffer}.dtype, layout={layout})"
        )

    def _generate_local_copy(self, stmt: GluonLocalCopy):
        """Generate in-kernel local/shared/register copy."""
        dst_layout = self._tensor_layout_expr(stmt.dst)
        src_layout = self._tensor_layout_expr(stmt.src)
        if dst_layout != f"{stmt.dst}.type.layout" and dst_layout != src_layout:
            self.lines.append(f"{self._indent()}{stmt.dst} = gl.convert_layout({stmt.src}, {dst_layout})")
        else:
            self.lines.append(f"{self._indent()}{stmt.dst} = {stmt.src}")

    def _generate_atomic_add(self, stmt: GluonAtomicAdd):
        """Generate atomic add into global memory."""
        target_indices = [self._fix_expr(i) for i in stmt.target_indices]
        if len(target_indices) >= 2:
            ptr_expr = f"{stmt.target} + ({target_indices[0]}) * {self._row_stride_expr(stmt.target)} + ({target_indices[1]})"
        elif target_indices:
            ptr_expr = f"{stmt.target} + ({target_indices[0]})"
        else:
            ptr_expr = stmt.target

        if stmt.value_indices:
            value_indices = ", ".join(self._fix_expr(i) for i in stmt.value_indices)
            value_expr = f"{stmt.value}[{value_indices}]"
        else:
            value_expr = self._fix_expr(stmt.value)

        self.lines.append(f"{self._indent()}gl.atomic_add({ptr_expr}, {value_expr})")

    def _match_block_atomic_add(self, stmt: GluonLoop) -> Optional[tuple[GluonAtomicAdd, List[str], List[str]]]:
        """Recognize parallel-loop atomic adds that can lower to a tile atomic op."""
        if stmt.start != 0 or stmt.step != 1 or len(stmt.body) != 1:
            return None

        body_stmt = stmt.body[0]
        if isinstance(body_stmt, GluonAtomicAdd):
            loop_vars = [stmt.var]
            extents = [self._fix_expr(stmt.end)]
            atomic = body_stmt
        elif (
            isinstance(body_stmt, GluonLoop)
            and body_stmt.start == 0
            and body_stmt.step == 1
            and len(body_stmt.body) == 1
            and isinstance(body_stmt.body[0], GluonAtomicAdd)
        ):
            loop_vars = [stmt.var, body_stmt.var]
            extents = [self._fix_expr(stmt.end), self._fix_expr(body_stmt.end)]
            atomic = body_stmt.body[0]
        else:
            return None

        value_indices = [self._fix_expr(i) for i in atomic.value_indices]
        if value_indices != loop_vars:
            return None

        return atomic, loop_vars, extents

    def _generate_block_atomic_add(self, stmt: GluonAtomicAdd, loop_vars: List[str], extents: List[str]):
        """Lower a Parallel-loop atomic add into a tile/vector atomic add."""
        suffix = stmt.value
        if len(loop_vars) == 1:
            layout = self._vector_layout_expr(extents[0])
            lane_name = f"{loop_vars[0]}_idx_{suffix}"
            self.lines.append(
                f"{self._indent()}{lane_name} = gl.arange(0, {extents[0]}, layout={layout})"
            )
            atomic_value = self._prepare_atomic_value(stmt.value, layout, suffix)
            index_expr = self._substitute_loop_vars(stmt.target_indices[0], {loop_vars[0]: lane_name})
            ptr_expr = f"{stmt.target} + ({index_expr})"
            mask_expr = f"({index_expr}) < {self._dim_expr(stmt.target, 0)}"
            self.lines.append(f"{self._indent()}gl.atomic_add({ptr_expr}, {atomic_value}, mask={mask_expr})")
            return

        rows, cols = extents
        layout = self._blocked_layout_expr(rows, cols)
        row_name = f"{loop_vars[0]}_idx_{suffix}"
        col_name = f"{loop_vars[1]}_idx_{suffix}"
        row_expr_name = f"{row_name}_expr"
        col_expr_name = f"{col_name}_expr"

        self.lines.append(
            f"{self._indent()}{row_name} = gl.arange(0, {rows}, layout=gl.SliceLayout(1, {layout}))"
        )
        self.lines.append(
            f"{self._indent()}{col_name} = gl.arange(0, {cols}, layout=gl.SliceLayout(0, {layout}))"
        )

        substitutions = {
            loop_vars[0]: f"{row_name}[:, None]",
            loop_vars[1]: f"{col_name}[None, :]",
        }
        row_expr = self._substitute_loop_vars(stmt.target_indices[0], substitutions)
        col_expr = self._substitute_loop_vars(stmt.target_indices[1], substitutions)

        self.lines.append(f"{self._indent()}{row_expr_name} = {row_expr}")
        self.lines.append(f"{self._indent()}{col_expr_name} = {col_expr}")
        self.lines.append(
            f"{self._indent()}ptr_{suffix} = {stmt.target} + ({row_expr_name}) * {self._row_stride_expr(stmt.target)} + ({col_expr_name})"
        )
        self.lines.append(
            f"{self._indent()}mask_{suffix} = (({row_expr_name}) < {self._dim_expr(stmt.target, 0)}) & "
            f"(({col_expr_name}) < {self._dim_expr(stmt.target, 1)})"
        )
        atomic_value = self._prepare_atomic_value(stmt.value, layout, suffix)
        self.lines.append(
            f"{self._indent()}gl.atomic_add(ptr_{suffix}, {atomic_value}, mask=mask_{suffix})"
        )

    def _generate_program_id(self, stmt: GluonProgramId):
        """Generate program ID access."""
        self.lines.append(
            f"{self._indent()}{stmt.var_name} = gl.program_id(axis={stmt.axis})"
        )

    def _generate_launcher(self, kernel: GluonKernel):
        """Generate host-side launcher function."""
        self.lines.append(f"# Kernel constants for {kernel.name}")
        self.lines.append(f"BLOCK_M = {self.block_M}")
        self.lines.append(f"BLOCK_N = {self.block_N}")
        self.lines.append(f"BLOCK_K = {self.block_K}")
        self.lines.append("")

        # Get tensor parameters (transformer sets 'type' field to 'tensor_descriptor')
        tensor_params = [p for p in kernel.params if p.get('type') == 'tensor_descriptor' or p.get('annotation', {}).get('type') == 'Tensor']

        if tensor_params:
            self.lines.append(f"def {kernel.name}(")
            self.lines.append("    " + ",\n    ".join([f"{p['name']}: torch.Tensor" for p in tensor_params]))
            self.lines.append("):")
        else:
            self.lines.append(f"def {kernel.name}():")

        self.indent_level += 1

        if tensor_params:
            first = tensor_params[0]['name']
            self.lines.append(f"{self._indent()}M = {first}.shape[0]")
            if len(tensor_params) >= 2:
                second = tensor_params[1]['name']
                self.lines.append(f"{self._indent()}K = {second}.shape[0]")
                self.lines.append(f"{self._indent()}N = {second}.shape[1]")
            else:
                self.lines.append(f"{self._indent()}N = {first}.shape[1]")
                self.lines.append(f"{self._indent()}K = {first}.shape[1]")
            if kernel.grid:
                self.lines.append(f"{self._indent()}grid = ({self._format_grid(kernel.grid)})")
            else:
                self.lines.append(f"{self._indent()}grid = (_ceildiv(N, BLOCK_N), _ceildiv(M, BLOCK_M))")

            # Launch kernel with dimensions as constexpr
            ptr_args = ", ".join([p['name'] for p in tensor_params])
            dim_args = "M=M, N=N, K=K"
            all_args = ", ".join([ptr_args, dim_args])
            self.lines.append(f"{self._indent()}{kernel.name}_kernel[grid]({all_args})")
            self.lines.append(f"{self._indent()}return {tensor_params[-1]['name']}")
        else:
            grid_str = ", ".join([str(g) for g in kernel.grid]) if kernel.grid else "1"
            self.lines.append(f"{self._indent()}grid = ({grid_str},)")
            self.lines.append(f"{self._indent()}{kernel.name}_kernel[grid]()")

        self.indent_level -= 1
        self.lines.append("")

    def _collect_block_symbols(self, kernel: GluonKernel) -> set:
        """Collect block_* symbols referenced by shapes."""
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

    def _row_stride_expr(self, tensor_name: str) -> str:
        """Return the contiguous row stride expression for a known tensor parameter."""
        tensor_names = [p["name"] for p in self.kernel.params if p.get("type") == "tensor_descriptor"]
        if tensor_name == tensor_names[0]:
            return "K"
        return "N"

    def _dim_expr(self, tensor_name: str, axis: int) -> str:
        """Return symbolic extents for known matrix operands."""
        tensor_names = [p["name"] for p in self.kernel.params if p.get("type") == "tensor_descriptor"]
        if tensor_name == tensor_names[0]:
            return "M" if axis == 0 else "K"
        if tensor_name == tensor_names[-1]:
            return "M" if axis == 0 else "N"
        return "K" if axis == 0 else "N"

    def _dot_k_width_literal(self, a_var: str, b_var: str) -> int:
        """Infer a constexpr k-width literal for DotOperandLayout."""
        a_bits = self._tensor_bitwidth(a_var)
        b_bits = self._tensor_bitwidth(b_var)
        return max(32 // min(a_bits, b_bits), 1)

    def _tensor_bitwidth(self, tensor_name: str) -> int:
        """Infer primitive bitwidth from transformed alloc/param metadata."""
        gluon_dtype = self._tensor_dtype(tensor_name)
        if gluon_dtype is None:
            return 32

        base = gluon_dtype.replace("gl.", "")
        bitwidth_map = {
            "float8e4nv": 8,
            "float8e5": 8,
            "float8e4b15": 8,
            "float16": 16,
            "bfloat16": 16,
            "int16": 16,
            "uint16": 16,
            "float32": 32,
            "int32": 32,
            "uint32": 32,
            "float64": 64,
            "int64": 64,
            "uint64": 64,
            "int8": 8,
            "uint8": 8,
            "int1": 1,
        }
        return bitwidth_map.get(base, 32)

    def _tensor_dtype(self, tensor_name: str) -> Optional[str]:
        """Resolve Gluon dtype for a known tensor/alloc name."""
        alloc = self.allocs.get(tensor_name)
        if alloc is not None and getattr(alloc, "dtype", None):
            return alloc.dtype

        for param in self.kernel.params:
            if param.get("name") == tensor_name and param.get("dtype"):
                return param["dtype"]

        return None

    def _tensor_layout_expr(self, tensor_name: str) -> str:
        """Resolve a stable Gluon layout expression for a known tensor/alloc name."""
        alloc = self.allocs.get(tensor_name)
        if alloc is not None and getattr(alloc, "layout", None):
            shape = getattr(alloc, "shape", None) or []
            # Pointer mode represents shared buffers as loaded/register tensors, not real shared-memory handles.
            if "NVMMASharedLayout" in alloc.layout:
                if len(shape) >= 2:
                    return self._blocked_layout_expr(shape[0], shape[1])
                if len(shape) == 1:
                    return self._vector_layout_expr(shape[0])
            return alloc.layout
        return f"{tensor_name}.type.layout"

    def _prepare_atomic_value(self, tensor_name: str, target_layout: str, suffix: str) -> str:
        """Convert a tensor value to the layout expected by a block atomic op when needed."""
        current_layout = self._tensor_layout_expr(tensor_name)
        if current_layout == target_layout:
            return tensor_name
        prepared_name = f"{tensor_name}_atomic_{suffix}"
        self.lines.append(f"{self._indent()}{prepared_name} = gl.convert_layout({tensor_name}, {target_layout})")
        return prepared_name

    def _blocked_layout_expr(self, rows: str, cols: str) -> str:
        """Construct a simple 2D blocked layout for block loads."""
        try:
            rows_i = int(str(rows))
            cols_i = int(str(cols))
            num_warps = int(self.kernel.num_warps)
            size_x = max(rows_i // max(num_warps, 1), 1)
            size_y = max(cols_i // 32, 1)
            return (
                f"gl.BlockedLayout([{size_x}, {size_y}], "
                f"[1, 32], [{num_warps}, 1], [1, 0])"
            )
        except Exception:
            pass
        return (
            "gl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])"
        )

    def _vector_layout_expr(self, extent: str) -> str:
        """Construct a simple 1D blocked layout."""
        try:
            extent_i = int(str(extent))
            num_warps = int(self.kernel.num_warps)
            size = max(extent_i // max(num_warps * 32, 1), 1)
            return f"gl.BlockedLayout([{size}], [32], [{num_warps}], [0])"
        except Exception:
            return "gl.BlockedLayout([1], [32], [4], [0])"

    def _substitute_loop_vars(self, expr: Any, replacements: dict[str, str]) -> str:
        """Substitute loop variables in an index expression with tensor expressions."""
        result = self._fix_expr(expr)
        for var, replacement in replacements.items():
            result = re.sub(rf"\b{re.escape(var)}\b", replacement, result)
        return result

    def _detect_dot_operand_layout(self) -> bool:
        """Return whether the active Gluon runtime exposes dot operand layouts."""
        try:
            from triton.experimental.gluon.language import _layouts  # type: ignore
        except Exception:
            return False
        return hasattr(_layouts, "DotOperandLayout")

    def _detect_mma_v2(self) -> bool:
        """Return whether the active Gluon runtime exposes the Ampere mma_v2 helper."""
        try:
            from triton.experimental.gluon.language.nvidia.ampere import mma_v2  # type: ignore
        except Exception:
            return False
        return callable(mma_v2)
