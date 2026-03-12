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
        self.dim_vars = {"M": "M", "N": "N", "K": "K"}
        self.vector_name_counter = 0
        self.has_runtime_dot_operand_layout = self._detect_dot_operand_layout()
        self.has_runtime_mma_v2 = self._detect_mma_v2()
        self.loop_constant_env = {}

    def generate(self, kernel: GluonKernel) -> str:
        """Generate Gluon kernel source code using pointer mode."""
        self.lines = []
        self.kernel = kernel
        self.vector_name_counter = 0
        self.loop_constant_env = {}
        self.store_passthrough = {}
        self.dim_vars = self._choose_dim_vars(kernel)
        self.symbol_aliases = self._infer_symbol_aliases()
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

    def _infer_symbol_aliases(self) -> dict[str, str]:
        """Infer dynamic TIR shape symbols that should map to launcher dims."""
        aliases = {}
        for param in getattr(self.kernel, "params", []):
            shape = param.get("shape", []) or []
            for dim in shape:
                if not isinstance(dim, str):
                    continue
                if dim == "m":
                    aliases[dim] = self._dim_var("M")
                elif dim == "n":
                    aliases[dim] = self._dim_var("N")
                elif dim == "k":
                    aliases[dim] = self._dim_var("K")
        return aliases

    def _choose_dim_vars(self, kernel: GluonKernel) -> dict[str, str]:
        """Choose launcher dimension names that do not collide with tensor params."""
        reserved = {
            p.get("name")
            for p in getattr(kernel, "params", [])
            if p.get("type") == "tensor_descriptor" or p.get("annotation", {}).get("type") == "Tensor"
        }
        chosen = {}
        for base in ("M", "N", "K"):
            candidate = base
            if candidate in reserved or candidate in chosen.values():
                candidate = f"{base}_DIM"
            suffix = 2
            while candidate in reserved or candidate in chosen.values():
                candidate = f"{base}_DIM_{suffix}"
                suffix += 1
            chosen[base] = candidate
        return chosen

    def _dim_var(self, base: str) -> str:
        return self.dim_vars.get(base, base)

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
            f"{self._dim_var('M')}: gl.constexpr",
            f"{self._dim_var('N')}: gl.constexpr",
            f"{self._dim_var('K')}: gl.constexpr",
            f"num_warps: gl.constexpr = {kernel.num_warps}",
            f"BLOCK_M: gl.constexpr = {self.block_M}",
            f"BLOCK_N: gl.constexpr = {self.block_N}",
            f"BLOCK_K: gl.constexpr = {self.block_K}",
        ]
        thread_dims = list(getattr(kernel, "thread_dims", [kernel.num_warps * 32, 1, 1]))
        thread_dims += [1] * (3 - len(thread_dims))
        constexpr_params.extend([
            f"THREADS_X: gl.constexpr = {thread_dims[0]}",
            f"THREADS_Y: gl.constexpr = {thread_dims[1]}",
            f"THREADS_Z: gl.constexpr = {thread_dims[2]}",
        ])

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
        for dim, var_name in enumerate(getattr(kernel, "thread_var_names", [None, None, None])):
            if var_name:
                self.lines.append(f"{self._indent()}{var_name} = {self._thread_binding_expr(dim)}")

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
        if getattr(stmt, "is_thread_local", False):
            extent = stmt.shape[0] if stmt.shape else 1
            self.lines.append(f"{self._indent()}# Initialize thread-local vector array")
            for idx in range(extent):
                self.lines.append(
                    f"{self._indent()}{self._thread_local_elem_name(stmt.name, idx)} = "
                    f"gl.full([THREADS_X], 0, {stmt.dtype}, layout={self._thread_binding_layout_expr()})"
                )
            return
        elif list(stmt.shape or []) == [1]:
            shape_str = "[THREADS_X]"
            layout = self._thread_binding_layout_expr()
        else:
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
            self.lines.append(
                f"{self._indent()}a_ptrs = {A_name} + "
                f"(offs_m[:, None] * {self._dim_var('K')} + (k + offs_k[None, :]))"
            )
            self.lines.append(
                f"{self._indent()}a_mask = (offs_m[:, None] < {self._dim_var('M')}) & "
                f"((k + offs_k[None, :]) < {self._dim_var('K')})"
            )
            self.lines.append(f"{self._indent()}a = gl.load(a_ptrs, mask=a_mask)")

            self.lines.append(
                f"{self._indent()}b_ptrs = {B_name} + "
                f"((k + offs_k[:, None]) * {self._dim_var('N')} + offs_n[None, :]))"
            )
            self.lines.append(
                f"{self._indent()}b_mask = ((k + offs_k[:, None]) < {self._dim_var('K')}) & "
                f"(offs_n[None, :] < {self._dim_var('N')})"
            )
            self.lines.append(f"{self._indent()}b = gl.load(b_ptrs, mask=b_mask)")

            A_var = "a"
            B_var = "b"

        self.lines.append(f"{self._indent()}# MMA: {stmt.acc} = dot({A_var}, {B_var}, acc={stmt.acc})")
        layout_id = self.vector_name_counter
        self.vector_name_counter += 1
        acc_layout_name = f"acc_layout_{layout_id}"
        a_layout_name = f"a_layout_{layout_id}"
        b_layout_name = f"b_layout_{layout_id}"
        self.lines.append(f"{self._indent()}{acc_layout_name}: gl.constexpr = {stmt.acc}.type.layout")
        layout_ctor = "gl.DotOperandLayout" if self.has_runtime_dot_operand_layout else "DotOperandLayout"
        k_width = self._dot_k_width_literal(A_var, B_var)
        self.lines.append(
            f"{self._indent()}{a_layout_name}: gl.constexpr = {layout_ctor}("
            f"parent={acc_layout_name}, operand_index=0, k_width={k_width})"
        )
        self.lines.append(
            f"{self._indent()}{b_layout_name}: gl.constexpr = {layout_ctor}("
            f"parent={acc_layout_name}, operand_index=1, k_width={k_width})"
        )
        self.lines.append(f"{self._indent()}{A_var}_dot = gl.convert_layout({A_var}, {a_layout_name})")
        self.lines.append(f"{self._indent()}{B_var}_dot = gl.convert_layout({B_var}, {b_layout_name})")
        if self.has_runtime_mma_v2:
            self.lines.append(f"{self._indent()}{stmt.acc} = mma_v2({A_var}_dot, {B_var}_dot, {stmt.acc})")
        else:
            self.lines.append(f"{self._indent()}{stmt.acc}_dot = tl.dot({A_var}_dot, {B_var}_dot, acc={stmt.acc})")
            self.lines.append(f"{self._indent()}{stmt.acc} = gl.convert_layout({stmt.acc}_dot, {acc_layout_name})")

    def _tensor_shape(self, tensor_name: str) -> list[str]:
        """Return known tensor/alloc shape expressions."""
        alloc = self.allocs.get(tensor_name)
        if alloc is not None and getattr(alloc, "shape", None):
            return [self._fix_expr(dim) for dim in getattr(alloc, "shape", [])]
        return self._shape_exprs(tensor_name)

    def _mma_lines(
        self,
        a_var: str,
        b_var: str,
        acc_var: str,
        *,
        trans_a: bool = False,
        trans_b: bool = False,
        m_dim: Optional[str] = None,
        n_dim: Optional[str] = None,
        k_dim: Optional[str] = None,
    ) -> list[str]:
        """Emit pointer-mode MMA lines for lowered TIR helper calls."""
        layout_ctor = "gl.DotOperandLayout" if self.has_runtime_dot_operand_layout else "DotOperandLayout"
        k_width = self._dot_k_width_literal(a_var, b_var)
        layout_id = self.vector_name_counter
        self.vector_name_counter += 1
        acc_layout_name = f"acc_layout_{layout_id}"
        a_layout_name = f"a_layout_{layout_id}"
        b_layout_name = f"b_layout_{layout_id}"
        a_input = a_var
        b_input = b_var
        a_shape = self._tensor_shape(a_var)
        b_shape = self._tensor_shape(b_var)
        if trans_a and len(a_shape) >= 2 and m_dim is not None and k_dim is not None:
            if a_shape[-2] == k_dim and a_shape[-1] == m_dim:
                a_input = f"{a_var}_trans"
        if trans_b and len(b_shape) >= 2 and n_dim is not None and k_dim is not None:
            if b_shape[-2] == n_dim and b_shape[-1] == k_dim:
                b_input = f"{b_var}_trans"
        lines = [
            f"{acc_layout_name}: gl.constexpr = {acc_var}.type.layout",
            f"{a_layout_name}: gl.constexpr = {layout_ctor}(parent={acc_layout_name}, operand_index=0, k_width={k_width})",
            f"{b_layout_name}: gl.constexpr = {layout_ctor}(parent={acc_layout_name}, operand_index=1, k_width={k_width})",
        ]
        if a_input != a_var:
            lines.append(f"{a_input} = gl.permute({a_var}, [1, 0])")
        if b_input != b_var:
            lines.append(f"{b_input} = gl.permute({b_var}, [1, 0])")
        lines.extend([
            f"{a_var}_dot = gl.convert_layout({a_input}, {a_layout_name})",
            f"{b_var}_dot = gl.convert_layout({b_input}, {b_layout_name})",
        ])
        if self.has_runtime_mma_v2:
            lines.append(f"{acc_var} = mma_v2({a_var}_dot, {b_var}_dot, {acc_var})")
        else:
            lines.append(f"{acc_var}_dot = tl.dot({a_var}_dot, {b_var}_dot, acc={acc_var})")
            lines.append(f"{acc_var} = gl.convert_layout({acc_var}_dot, {acc_layout_name})")
        return lines

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
        for symbol, alias in self.symbol_aliases.items():
            result = re.sub(rf"\b{re.escape(symbol)}\b", alias, result)
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
                region_extents = [self._fix_expr(e) for e in (stmt.src_extents or [])]
                row_axis, col_axis = self._region_matrix_axes(stmt.src_tensor, src_idx, region_extents)
                base_offset = self._region_base_offset_expr(stmt.src_tensor, src_idx, region_extents)
                row_stride = self._stride_expr(stmt.src_tensor, row_axis)
                col_stride = self._stride_expr(stmt.src_tensor, col_axis)
                layout = self._blocked_layout_expr(rows, cols)
                self.lines.append(
                    f"{self._indent()}rows_{stmt.smem} = gl.arange(0, {rows}, layout=gl.SliceLayout(1, {layout}))"
                )
                self.lines.append(
                    f"{self._indent()}cols_{stmt.smem} = gl.arange(0, {cols}, layout=gl.SliceLayout(0, {layout}))"
                )
                base_expr = f"{stmt.src_tensor} + {base_offset}" if base_offset != "0" else stmt.src_tensor
                row_term = (
                    f"rows_{stmt.smem}[:, None]"
                    if row_stride == "1"
                    else f"rows_{stmt.smem}[:, None] * {row_stride}"
                )
                col_term = (
                    f"cols_{stmt.smem}[None, :]"
                    if col_stride == "1"
                    else f"cols_{stmt.smem}[None, :] * {col_stride}"
                )
                self.lines.append(
                    f"{self._indent()}ptr_{stmt.smem} = {base_expr} + {row_term} + {col_term}"
                )
                mask_terms = self._region_mask_terms(
                    stmt.src_tensor,
                    src_idx,
                    region_extents,
                    row_axis,
                    col_axis,
                    f"rows_{stmt.smem}[:, None]",
                    f"cols_{stmt.smem}[None, :]",
                )
                self.lines.append(
                    f"{self._indent()}mask_{stmt.smem} = " + " & ".join(mask_terms)
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
            store_value = self.store_passthrough.get(stmt.smem, stmt.smem)
            self.lines.append(f"{self._indent()}# Store {stmt.smem} to {stmt.dst_tensor}")
            alloc = self.allocs.get(stmt.smem)
            shape = getattr(alloc, "shape", None) or []
            if len(dst_idx) >= 2 and len(shape) >= 2:
                rows = shape[0]
                cols = shape[1]
                region_extents = [self._fix_expr(e) for e in (stmt.dst_extents or [])]
                row_axis, col_axis = self._region_matrix_axes(stmt.dst_tensor, dst_idx, region_extents)
                base_offset = self._region_base_offset_expr(stmt.dst_tensor, dst_idx, region_extents)
                row_stride = self._stride_expr(stmt.dst_tensor, row_axis)
                col_stride = self._stride_expr(stmt.dst_tensor, col_axis)
                layout = self._tensor_layout_expr(stmt.smem)
                self.lines.append(
                    f"{self._indent()}rows_out = gl.arange(0, {rows}, layout=gl.SliceLayout(1, {layout}))"
                )
                self.lines.append(
                    f"{self._indent()}cols_out = gl.arange(0, {cols}, layout=gl.SliceLayout(0, {layout}))"
                )
                base_expr = f"{stmt.dst_tensor} + {base_offset}" if base_offset != "0" else stmt.dst_tensor
                row_term = "rows_out[:, None]" if row_stride == "1" else f"rows_out[:, None] * {row_stride}"
                col_term = "cols_out[None, :]" if col_stride == "1" else f"cols_out[None, :] * {col_stride}"
                self.lines.append(f"{self._indent()}ptr_out = {base_expr} + {row_term} + {col_term}")
                mask_terms = self._region_mask_terms(
                    stmt.dst_tensor,
                    dst_idx,
                    region_extents,
                    row_axis,
                    col_axis,
                    "rows_out[:, None]",
                    "cols_out[None, :]",
                )
                self.lines.append(f"{self._indent()}mask_out = " + " & ".join(mask_terms))
                self.lines.append(f"{self._indent()}gl.store(ptr_out, {store_value}, mask=mask_out)")
            else:
                if len(dst_idx) >= 2:
                    self.lines.append(
                        f"{self._indent()}ptr_out = {stmt.dst_tensor} + "
                        f"({dst_idx[0]}) * {self._dim_var('N')} + ({dst_idx[1]})"
                    )
                else:
                    self.lines.append(f"{self._indent()}ptr_out = {stmt.dst_tensor} + {dst_idx[0] if dst_idx else '0'}")
                self.lines.append(f"{self._indent()}gl.store(ptr_out, {store_value})")
        else:
            # Fallback to simple store
            tensor_params = [p['name'] for p in self.kernel.params]
            C_name = tensor_params[-1] if tensor_params else "C"
            self.lines.append(f"{self._indent()}# Store result to global memory")
            self.lines.append(
                f"{self._indent()}c_ptrs = {C_name} + "
                f"(offs_m[:, None] * {self._dim_var('N')} + offs_n[None, :])"
            )
            self.lines.append(
                f"{self._indent()}c_mask = (offs_m[:, None] < {self._dim_var('M')}) & "
                f"(offs_n[None, :] < {self._dim_var('N')})"
            )
            self.lines.append(f"{self._indent()}gl.store(c_ptrs, {stmt.smem}, mask=c_mask)")

    def _generate_loop(self, stmt: GluonLoop):
        """Generate loop construct."""
        block_atomic = self._match_block_atomic_add(stmt)
        if block_atomic is not None:
            self._generate_block_atomic_add(*block_atomic)
            return

        if self._generate_unrolled_loop(stmt):
            return

        checkpoint = len(self.lines)
        if self._try_generate_vectorized_loop(stmt):
            return
        self.lines = self.lines[:checkpoint]

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
        end_str = self._fix_expr(end_str)
        start_str = self._fix_expr(start_str)
        step_str = self._fix_expr(step_str)

        self.lines.append(
            f"{self._indent()}for {stmt.var} in range({start_str}, {end_str}, {step_str}):"
        )
        self.indent_level += 1

        for body_stmt in stmt.body:
            if body_stmt:
                if isinstance(body_stmt, ast.AST):
                    self._generate_raw_ast(body_stmt)
                else:
                    self._generate_stmt(body_stmt)

        self.indent_level -= 1

    def _generate_raw_ast(self, stmt: ast.AST):
        """Emit preserved Python AST statements."""
        lowered = self._lower_ast_stmt(stmt)
        if lowered is None:
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
            return
        for line in lowered:
            self.lines.append(f"{self._indent()}{line}")

    def _generate_clear(self, stmt: GluonClear):
        """Generate clear operation."""
        if self._is_thread_local_tensor(stmt.buffer):
            extent = (getattr(self.allocs.get(stmt.buffer), "shape", None) or [1])[0]
            for idx in range(extent):
                self.lines.append(
                    f"{self._indent()}{self._thread_local_elem_name(stmt.buffer, idx)} = "
                    f"gl.full([THREADS_X], 0, {self._tensor_dtype(stmt.buffer)}, layout={self._thread_binding_layout_expr()})"
                )
            return
        layout = self._tensor_layout_expr(stmt.buffer)
        self.lines.append(
            f"{self._indent()}{stmt.buffer} = gl.full({stmt.buffer}.shape, 0, {stmt.buffer}.dtype, layout={layout})"
        )

    def _generate_local_copy(self, stmt: GluonLocalCopy):
        """Generate in-kernel local/shared/register copy."""
        dst_layout = self._tensor_layout_expr(stmt.dst)
        src_layout = self._tensor_layout_expr(stmt.src)
        src_expr = stmt.src
        src_dtype = self._tensor_dtype(stmt.src)
        dst_dtype = self._tensor_dtype(stmt.dst)
        if src_dtype is not None and dst_dtype is not None and src_dtype != dst_dtype:
            src_expr = self._cast_expr(src_expr, src_dtype, dst_dtype)
        if dst_layout != f"{stmt.dst}.type.layout" and dst_layout != src_layout:
            self.lines.append(f"{self._indent()}{stmt.dst} = gl.convert_layout({src_expr}, {dst_layout})")
        else:
            self.lines.append(f"{self._indent()}{stmt.dst} = {src_expr}")

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
            self.lines.append(f"{self._indent()}{self._dim_var('M')} = {first}.shape[0]")
            if len(tensor_params) >= 2:
                second = tensor_params[1]['name']
                self.lines.append(f"{self._indent()}{self._dim_var('K')} = {second}.shape[0]")
                self.lines.append(f"{self._indent()}{self._dim_var('N')} = {second}.shape[1]")
            else:
                self.lines.append(f"{self._indent()}{self._dim_var('N')} = {first}.shape[1]")
                self.lines.append(f"{self._indent()}{self._dim_var('K')} = {first}.shape[1]")
            if kernel.grid:
                if len(kernel.grid) == 1:
                    self.lines.append(f"{self._indent()}grid = ({self._format_grid(kernel.grid)},)")
                else:
                    self.lines.append(f"{self._indent()}grid = ({self._format_grid(kernel.grid)})")
            else:
                self.lines.append(
                    f"{self._indent()}grid = (_ceildiv({self._dim_var('N')}, BLOCK_N), "
                    f"_ceildiv({self._dim_var('M')}, BLOCK_M))"
                )

            # Launch kernel with dimensions as constexpr
            ptr_args = ", ".join([p['name'] for p in tensor_params])
            dim_args = ", ".join(
                f"{self._dim_var(base)}={self._dim_var(base)}" for base in ("M", "N", "K")
            )
            all_args = ", ".join([ptr_args, dim_args])
            self.lines.append(f"{self._indent()}{kernel.name}_kernel[grid]({all_args})")
            param_names = {p['name'] for p in tensor_params}
            output_name_set = {name for name in getattr(kernel, "output_params", []) if name in param_names}
            output_names = [p['name'] for p in tensor_params if p['name'] in output_name_set]
            if len(output_names) > 1:
                self.lines.append(f"{self._indent()}return ({', '.join(output_names)})")
            elif len(output_names) == 1:
                self.lines.append(f"{self._indent()}return {output_names[0]}")
            else:
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
            for symbol, alias in self.symbol_aliases.items():
                expr = re.sub(rf"\b{re.escape(symbol)}\b", alias, expr)
            expr = re.sub(r"\bceildiv\s*\(", "_ceildiv(", expr)
            rendered.append(expr)
        return ", ".join(rendered)

    def _row_stride_expr(self, tensor_name: str) -> str:
        """Return the contiguous row stride expression for a known tensor parameter."""
        for param in self.kernel.params:
            if param.get("name") != tensor_name:
                continue
            shape = param.get("shape", []) or []
            if len(shape) >= 2:
                return self._stride_expr(tensor_name, len(shape) - 2)
        tensor_names = [p["name"] for p in self.kernel.params if p.get("type") == "tensor_descriptor"]
        if tensor_names and tensor_name == tensor_names[0]:
            return self._dim_var("K")
        return self._dim_var("N")

    def _shape_exprs(self, tensor_name: str) -> list[str]:
        """Return symbolic shape expressions for a known tensor parameter."""
        for param in self.kernel.params:
            if param.get("name") != tensor_name:
                continue
            return [self._fix_expr(str(dim)) for dim in (param.get("shape", []) or [])]
        return []

    def _stride_expr(self, tensor_name: str, axis: int) -> str:
        """Return contiguous row-major stride for a tensor axis."""
        shape = self._shape_exprs(tensor_name)
        if not shape:
            rank = self._tensor_rank(tensor_name)
            if rank == 0:
                return "1"
            if rank <= 2:
                return self._dim_var("K") if axis == 0 else "1"
            return "1"
        if axis >= len(shape) - 1:
            return "1"
        stride_terms = shape[axis + 1 :]
        if not stride_terms:
            return "1"
        if len(stride_terms) == 1:
            return stride_terms[0]
        return "(" + " * ".join(stride_terms) + ")"

    def _dim_expr(self, tensor_name: str, axis: int) -> str:
        """Return symbolic extents for known matrix operands."""
        for param in self.kernel.params:
            if param.get("name") != tensor_name:
                continue
            shape = param.get("shape", []) or []
            if len(shape) > axis:
                return self._fix_expr(str(shape[axis]))
        tensor_names = [p["name"] for p in self.kernel.params if p.get("type") == "tensor_descriptor"]
        if tensor_name == tensor_names[0]:
            return self._dim_var("M") if axis == 0 else self._dim_var("K")
        if tensor_name == tensor_names[-1]:
            return self._dim_var("M") if axis == 0 else self._dim_var("N")
        return self._dim_var("K") if axis == 0 else self._dim_var("N")

    def _linear_offset_expr(self, tensor_name: str, indices: list[str]) -> str:
        """Flatten row-major tensor indices into a linear pointer offset."""
        if not indices:
            return "0"
        terms = []
        for axis, idx in enumerate(indices):
            stride = self._stride_expr(tensor_name, axis)
            if stride == "1":
                terms.append(f"({idx})")
            else:
                terms.append(f"({idx}) * {stride}")
        return " + ".join(terms) if terms else "0"

    def _region_matrix_axes(self, tensor_name: str, indices: list[str], extents: list[str]) -> tuple[int, int]:
        """Infer which tensor axes form the row/col tile in a T.region copy."""
        if len(extents) == len(indices):
            varying_axes = [axis for axis, extent in enumerate(extents) if str(extent) != "1"]
            if len(varying_axes) >= 2:
                return varying_axes[0], varying_axes[1]
        tensor_rank = self._tensor_rank(tensor_name)
        return max(tensor_rank - 2, 0), max(tensor_rank - 1, 0)

    def _region_base_offset_expr(self, tensor_name: str, indices: list[str], extents: list[str]) -> str:
        """Compute the linearized base offset for a T.region copy."""
        terms = []
        for axis, idx in enumerate(indices):
            base_idx = idx
            stride = self._stride_expr(tensor_name, axis)
            if base_idx == "0":
                continue
            if stride == "1":
                terms.append(f"({base_idx})")
            else:
                terms.append(f"({base_idx}) * {stride}")
        return " + ".join(terms) if terms else "0"

    def _region_mask_terms(
        self,
        tensor_name: str,
        indices: list[str],
        extents: list[str],
        row_axis: int,
        col_axis: int,
        row_expr: str,
        col_expr: str,
    ) -> list[str]:
        """Build bounds checks for a T.region copy."""
        mask_terms = []
        for axis, idx in enumerate(indices):
            if axis == row_axis:
                mask_terms.append(f"(({idx}) + {row_expr} < {self._dim_expr(tensor_name, axis)})")
            elif axis == col_axis:
                mask_terms.append(f"(({idx}) + {col_expr} < {self._dim_expr(tensor_name, axis)})")
            elif len(extents) != len(indices) or str(extents[axis]) == "1":
                mask_terms.append(f"(({idx}) < {self._dim_expr(tensor_name, axis)})")
        return mask_terms or ["True"]

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

    def _dtype_bitwidth(self, dtype_expr: str) -> int:
        """Infer primitive bitwidth from a Gluon dtype expression."""
        base = dtype_expr.replace("gl.", "")
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

    def _is_floating_dtype(self, dtype_expr: str) -> bool:
        base = dtype_expr.replace("gl.", "")
        return base.startswith("float") or base == "bfloat16"

    def _cast_expr(self, expr: str, src_dtype: str, dst_dtype: str) -> str:
        """Emit explicit RTNE for floating downcasts to match Triton/Torch FP8 semantics."""
        if (
            self._is_floating_dtype(src_dtype)
            and self._is_floating_dtype(dst_dtype)
            and self._dtype_bitwidth(dst_dtype) < self._dtype_bitwidth(src_dtype)
        ):
            return f'tl.cast({expr}, {dst_dtype}, fp_downcast_rounding="rtne")'
        return f"{expr}.to({dst_dtype})"

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
            if getattr(alloc, "is_thread_local", False):
                return self._thread_binding_layout_expr()
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

    def _thread_binding_layout_expr(self) -> str:
        """Construct a simple 1D logical-thread layout."""
        return "gl.BlockedLayout([1], [32], [num_warps], [0])"

    def _thread_dim_symbol(self, dim: int) -> str:
        mapping = {0: "THREADS_X", 1: "THREADS_Y", 2: "THREADS_Z"}
        return mapping.get(dim, "THREADS_X")

    def _thread_binding_expr(self, dim: int) -> str:
        return (
            f"gl.arange(0, {self._thread_dim_symbol(dim)}, "
            f"layout={self._thread_binding_layout_expr()})"
        )

    def _is_global_tensor(self, name: str) -> bool:
        return any(param.get("name") == name for param in getattr(self, "tensor_params", []))

    def _tensor_rank(self, name: str) -> int:
        for param in self.kernel.params:
            if param.get("name") == name:
                shape = param.get("shape", []) or []
                return len(shape)
        return 0

    def _is_scalar_like_local(self, name: str) -> bool:
        alloc = self.allocs.get(name)
        if alloc is None:
            return False
        shape = getattr(alloc, "shape", None) or []
        return list(shape) == [1]

    def _is_thread_local_tensor(self, name: str) -> bool:
        alloc = self.allocs.get(name)
        return bool(alloc is not None and getattr(alloc, "is_thread_local", False))

    def _thread_local_elem_name(self, name: str, idx: int) -> str:
        return f"{name}_{idx}"

    def _thread_column_elem_name(self, name: str, idx: int) -> str:
        return f"{name}_{idx}"

    def _uses_thread_var(self, node: ast.AST) -> bool:
        thread_vars = {name for name in getattr(self.kernel, "thread_var_names", []) if name}
        return any(isinstance(child, ast.Name) and child.id in thread_vars for child in ast.walk(node))

    def _resolve_temp_indexed_name(self, node: ast.Subscript) -> Optional[str]:
        if not isinstance(node.value, ast.Name) or self._is_global_tensor(node.value.id):
            return None
        if self._is_scalar_like_local(node.value.id) or self._is_thread_local_tensor(node.value.id):
            return None
        indices = self._subscript_indices(node)
        if len(indices) == 1:
            idx_value = self._eval_static_int_ast(indices[0])
            if idx_value is not None:
                return self._thread_local_elem_name(node.value.id, idx_value)
            return None
        if len(indices) == 2 and self._uses_thread_var(indices[0]):
            idx_value = self._eval_static_int_ast(indices[1])
            if idx_value is not None:
                return self._thread_column_elem_name(node.value.id, idx_value)
        return None

    def _lower_global_access_mask(self, tensor_name: str, indices: list[ast.AST]) -> Optional[str]:
        if not any(self._uses_thread_var(idx) for idx in indices):
            return None
        lowered = [self._lower_ast_expr(idx) for idx in indices]
        rank = self._tensor_rank(tensor_name)
        if rank <= 1 or len(lowered) == 1:
            return f"({lowered[0]}) < {self._dim_expr(tensor_name, 0)}"
        if len(lowered) == 2:
            return (
                f"(({lowered[0]}) < {self._dim_expr(tensor_name, 0)}) & "
                f"(({lowered[1]}) < {self._dim_expr(tensor_name, 1)})"
            )
        if len(lowered) == 3:
            return (
                f"(({lowered[0]}) < {self._dim_expr(tensor_name, 0)}) & "
                f"(({lowered[1]}) < {self._dim_expr(tensor_name, 1)}) & "
                f"(({lowered[2]}) < {self._dim_expr(tensor_name, 2)})"
            )
        return None

    def _operator_str(self, op: ast.AST) -> str:
        mapping = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.FloorDiv: "//",
            ast.Mod: "%",
            ast.Pow: "**",
            ast.BitAnd: "&",
            ast.BitOr: "|",
            ast.BitXor: "^",
            ast.LShift: "<<",
            ast.RShift: ">>",
        }
        return mapping[type(op)]

    def _cmp_str(self, op: ast.AST) -> str:
        mapping = {
            ast.Eq: "==",
            ast.NotEq: "!=",
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Gt: ">",
            ast.GtE: ">=",
        }
        return mapping[type(op)]

    def _dtype_attr_expr(self, attr: str) -> str:
        mapping = {
            "float": "gl.float32",
            "float8_e4m3fn": "gl.float8e4nv",
            "float16": "gl.float16",
            "float32": "gl.float32",
            "float64": "gl.float64",
            "bfloat16": "gl.bfloat16",
            "int8": "gl.int8",
            "int16": "gl.int16",
            "int32": "gl.int32",
            "int64": "gl.int64",
            "uint8": "gl.uint8",
            "uint16": "gl.uint16",
            "uint32": "gl.uint32",
            "uint64": "gl.uint64",
        }
        return mapping.get(attr, f"gl.{attr}")

    def _subscript_indices(self, node: ast.Subscript) -> list[ast.AST]:
        if isinstance(node.slice, ast.Tuple):
            return list(node.slice.elts)
        return [node.slice]

    def _lower_global_ptr_expr(self, tensor_name: str, indices: list[ast.AST]) -> str:
        lowered = [self._lower_ast_expr(idx) for idx in indices]
        rank = self._tensor_rank(tensor_name)
        if rank <= 1 or len(lowered) == 1:
            return f"{tensor_name} + ({lowered[0]})"
        if len(lowered) == 2:
            return (
                f"{tensor_name} + ({lowered[0]}) * {self._row_stride_expr(tensor_name)} "
                f"+ ({lowered[1]})"
            )
        if len(lowered) == 3:
            batch_stride = f"({self._dim_expr(tensor_name, 1)} * {self._dim_expr(tensor_name, 2)})"
            return (
                f"{tensor_name} + ({lowered[0]}) * {batch_stride} "
                f"+ ({lowered[1]}) * {self._dim_expr(tensor_name, 2)} + ({lowered[2]})"
            )
        return f"{tensor_name} + ({lowered[0]})"

    def _lower_ast_expr(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return self.symbol_aliases.get(node.id, node.id)
        if isinstance(node, ast.Constant):
            return repr(node.value)
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "T":
                return self._dtype_attr_expr(node.attr)
            return f"{self._lower_ast_expr(node.value)}.{node.attr}"
        if isinstance(node, ast.BinOp):
            return f"({self._lower_ast_expr(node.left)} {self._operator_str(node.op)} {self._lower_ast_expr(node.right)})"
        if isinstance(node, ast.UnaryOp):
            unary = {
                ast.UAdd: "+",
                ast.USub: "-",
                ast.Not: "not ",
                ast.Invert: "~",
            }[type(node.op)]
            return f"({unary}{self._lower_ast_expr(node.operand)})"
        if isinstance(node, ast.BoolOp):
            joiner = " and " if isinstance(node.op, ast.And) else " or "
            return "(" + joiner.join(self._lower_ast_expr(v) for v in node.values) + ")"
        if isinstance(node, ast.Compare):
            left = self._lower_ast_expr(node.left)
            parts = []
            for op, comp in zip(node.ops, node.comparators):
                parts.append(f"{left} {self._cmp_str(op)} {self._lower_ast_expr(comp)}")
                left = self._lower_ast_expr(comp)
            return "(" + " and ".join(parts) + ")"
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "T":
                    if node.func.attr == "Cast":
                        dtype_arg = node.args[0]
                        if isinstance(dtype_arg, ast.Constant):
                            dtype_expr = self._dtype_attr_expr(str(dtype_arg.value))
                        else:
                            dtype_expr = self._lower_ast_expr(dtype_arg)
                        value_expr = self._lower_ast_expr(node.args[1])
                        return f"tl.cast({value_expr}, {dtype_expr})"
                    if node.func.attr == "get_thread_binding":
                        dim = int(node.args[0].value) if node.args else 0
                        return self._thread_binding_expr(dim)
                    if node.func.attr == "tvm_warp_shuffle":
                        value_expr = self._lower_ast_expr(node.args[1])
                        lane_expr = self._lower_ast_expr(node.args[2])
                        return self._warp_shuffle_expr(value_expr, lane_expr)
                    if node.func.attr == "shift_left":
                        return f"({self._lower_ast_expr(node.args[0])} << {self._lower_ast_expr(node.args[1])})"
                    if node.func.attr == "shift_right":
                        return f"({self._lower_ast_expr(node.args[0])} >> {self._lower_ast_expr(node.args[1])})"
                    if node.func.attr == "bitwise_xor":
                        return f"({self._lower_ast_expr(node.args[0])} ^ {self._lower_ast_expr(node.args[1])})"
                    if node.func.attr == "bitwise_and":
                        return f"({self._lower_ast_expr(node.args[0])} & {self._lower_ast_expr(node.args[1])})"
                    if node.func.attr == "bitwise_or":
                        return f"({self._lower_ast_expr(node.args[0])} | {self._lower_ast_expr(node.args[1])})"
                    if node.func.attr == "if_then_else":
                        cond = self._lower_ast_expr(node.args[0])
                        true_val = self._lower_ast_expr(node.args[1])
                        false_val = self._lower_ast_expr(node.args[2])
                        return f"tl.where({cond}, {true_val}, {false_val})"
                    if node.func.attr == "infinity":
                        return "float('inf')"
                    if node.func.attr == "exp2":
                        return f"tl.exp2({self._lower_ast_expr(node.args[0])})"
                    if node.func.attr == "log2":
                        return f"tl.log2({self._lower_ast_expr(node.args[0])})"
                    if node.func.attr == "max":
                        return f"tl.maximum({self._lower_ast_expr(node.args[0])}, {self._lower_ast_expr(node.args[1])})"
                    if node.func.attr == "min":
                        return f"tl.minimum({self._lower_ast_expr(node.args[0])}, {self._lower_ast_expr(node.args[1])})"
                    if node.func.attr == "bool":
                        return self._lower_ast_expr(node.args[0])
                    if node.func.attr in {
                        "float16", "float32", "float64",
                        "int8", "int16", "int32", "int64",
                        "uint8", "uint16", "uint32", "uint64",
                        "bfloat16",
                    }:
                        if not node.args:
                            return self._dtype_attr_expr(node.func.attr)
                        return self._lower_ast_expr(node.args[0])
                if node.func.attr == "astype":
                    base = self._lower_ast_expr(node.func.value)
                    dtype_expr = self._lower_ast_expr(node.args[0])
                    return f"{base}.to({dtype_expr})"
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                args = ", ".join(self._lower_ast_expr(arg) for arg in node.args)
                return f"{func_name}({args})"
            if hasattr(ast, "unparse"):
                return ast.unparse(node).strip()
        if isinstance(node, ast.Subscript):
            temp_name = self._resolve_temp_indexed_name(node)
            if temp_name is not None:
                return temp_name
            if isinstance(node.value, ast.Name) and self._is_thread_local_tensor(node.value.id):
                indices = self._subscript_indices(node)
                if len(indices) == 1:
                    idx_value = self._eval_static_int_ast(indices[0])
                    if idx_value is None:
                        raise NotImplementedError(f"Dynamic thread-local index is not supported: {ast.unparse(indices[0])}")
                    return self._thread_local_elem_name(node.value.id, idx_value)
            if isinstance(node.value, ast.Name) and self._is_global_tensor(node.value.id):
                ptr_expr = self._lower_global_ptr_expr(node.value.id, self._subscript_indices(node))
                mask_expr = self._lower_global_access_mask(node.value.id, self._subscript_indices(node))
                if mask_expr is not None:
                    return f"gl.load({ptr_expr}, mask={mask_expr})"
                return f"gl.load({ptr_expr})"
            if isinstance(node.value, ast.Name) and self._is_scalar_like_local(node.value.id):
                indices = self._subscript_indices(node)
                if len(indices) == 1 and isinstance(indices[0], ast.Constant) and indices[0].value == 0:
                    return node.value.id
            base = self._lower_ast_expr(node.value)
            indices = ", ".join(self._lower_ast_expr(idx) for idx in self._subscript_indices(node))
            return f"{base}[{indices}]"
        if hasattr(ast, "unparse"):
            return ast.unparse(node).strip()
        raise NotImplementedError(f"Unsupported AST expression: {ast.dump(node)}")

    def _lower_ast_store(self, target: ast.Subscript, value_expr: str) -> str:
        temp_name = self._resolve_temp_indexed_name(target)
        if temp_name is not None:
            return f"{temp_name} = {value_expr}"
        if isinstance(target.value, ast.Name) and self._is_thread_local_tensor(target.value.id):
            indices = self._subscript_indices(target)
            if len(indices) == 1:
                idx_value = self._eval_static_int_ast(indices[0])
                if idx_value is None:
                    raise NotImplementedError(f"Dynamic thread-local index is not supported: {ast.unparse(indices[0])}")
                return f"{self._thread_local_elem_name(target.value.id, idx_value)} = {value_expr}"
        if isinstance(target.value, ast.Name) and self._is_global_tensor(target.value.id):
            ptr_expr = self._lower_global_ptr_expr(target.value.id, self._subscript_indices(target))
            mask_expr = self._lower_global_access_mask(target.value.id, self._subscript_indices(target))
            if mask_expr is not None:
                return f"gl.store({ptr_expr}, {value_expr}, mask={mask_expr})"
            return f"gl.store({ptr_expr}, {value_expr})"
        if isinstance(target.value, ast.Name) and self._is_scalar_like_local(target.value.id):
            indices = self._subscript_indices(target)
            if len(indices) == 1 and isinstance(indices[0], ast.Constant) and indices[0].value == 0:
                return f"{target.value.id} = {value_expr}"
        base = self._lower_ast_expr(target.value)
        indices = ", ".join(self._lower_ast_expr(idx) for idx in self._subscript_indices(target))
        return f"{base}[{indices}] = {value_expr}"

    def _lower_ast_stmt(self, stmt: ast.AST) -> Optional[list[str]]:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            lowered = self._lower_ast_call_stmt(stmt.value)
            if lowered is not None:
                return lowered
        if isinstance(stmt, ast.Assign):
            if self._is_shape_symbol_declaration(stmt):
                return []
            if self._is_handle_declaration(stmt):
                return []
            value_expr = self._lower_ast_expr(stmt.value)
            lines = []
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    lines.append(f"{target.id} = {value_expr}")
                elif isinstance(target, ast.Subscript):
                    lines.append(self._lower_ast_store(target, value_expr))
                else:
                    return None
            return lines
        if isinstance(stmt, ast.AnnAssign):
            if stmt.value is None:
                return []
            if isinstance(stmt.target, ast.Name) and self._is_zero_arg_t_dtype_call(stmt.annotation):
                return [f"{stmt.target.id} = {self._lower_ast_expr(stmt.value)}"]
            if isinstance(stmt.target, ast.Name):
                return [f"{stmt.target.id} = {self._lower_ast_expr(stmt.value)}"]
            if isinstance(stmt.target, ast.Subscript):
                value_expr = self._lower_ast_expr(stmt.value)
                return [self._lower_ast_store(stmt.target, value_expr)]
        if isinstance(stmt, ast.AugAssign):
            if isinstance(stmt.target, ast.Subscript):
                target_expr = self._lower_ast_expr(stmt.target)
                value_expr = self._lower_ast_expr(stmt.value)
                op = self._operator_str(stmt.op)
                if isinstance(stmt.target.value, ast.Name) and self._is_global_tensor(stmt.target.value.id):
                    return [self._lower_ast_store(stmt.target, f"{target_expr} {op} {value_expr}")]
                return [f"{target_expr} = {target_expr} {op} {value_expr}"]
            if isinstance(stmt.target, ast.Name):
                value_expr = self._lower_ast_expr(stmt.value)
                op = self._operator_str(stmt.op)
                return [f"{stmt.target.id} = {stmt.target.id} {op} {value_expr}"]
        if isinstance(stmt, ast.If):
            lines = [f"if {self._lower_ast_expr(stmt.test)}:"]
            body_lines = []
            for inner in stmt.body:
                lowered = self._lower_ast_stmt(inner)
                if lowered is None:
                    return None
                body_lines.extend(f"    {line}" for line in lowered)
            if not body_lines:
                body_lines.append("    pass")
            lines.extend(body_lines)
            if stmt.orelse:
                lines.append("else:")
                else_lines = []
                for inner in stmt.orelse:
                    lowered = self._lower_ast_stmt(inner)
                    if lowered is None:
                        return None
                    else_lines.extend(f"    {line}" for line in lowered)
                if not else_lines:
                    else_lines.append("    pass")
                lines.extend(else_lines)
            return lines
        return None

    def _is_zero_arg_t_dtype_call(self, node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "T"
            and node.func.attr in {"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}
            and not node.args
        )

    def _is_shape_symbol_declaration(self, stmt: ast.Assign) -> bool:
        if len(stmt.targets) != 1:
            return False
        target = stmt.targets[0]
        if isinstance(target, ast.Tuple):
            if not isinstance(stmt.value, ast.Tuple) or len(target.elts) != len(stmt.value.elts):
                return False
            return all(isinstance(t, ast.Name) for t in target.elts) and all(
                self._is_zero_arg_t_dtype_call(v) for v in stmt.value.elts
            )
        return isinstance(target, ast.Name) and self._is_zero_arg_t_dtype_call(stmt.value)

    def _is_handle_declaration(self, stmt: ast.Assign) -> bool:
        if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
            return False
        value = stmt.value
        return (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and isinstance(value.func.value, ast.Name)
            and value.func.value.id == "T"
            and value.func.attr == "handle"
        )

    def _region_base_name(self, node: ast.AST) -> Optional[str]:
        """Resolve the underlying buffer name from a T.region call."""
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "region" and node.args:
                base = node.args[0]
                if isinstance(base, ast.Subscript) and isinstance(base.value, ast.Name):
                    return base.value.id
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            return node.value.id
        if isinstance(node, ast.Name):
            return node.id
        return None

    def _lower_ast_call_stmt(self, call: ast.Call) -> Optional[list[str]]:
        """Lower selected TileLang/TIR helper calls."""
        if not isinstance(call.func, ast.Attribute):
            return None
        if not isinstance(call.func.value, ast.Name) or call.func.value.id != "T":
            return None

        if call.func.attr == "fill" and len(call.args) >= 2:
            target = self._region_base_name(call.args[0])
            if target is None:
                return None
            value_expr = self._lower_ast_expr(call.args[1])
            layout_expr = self._tensor_layout_expr(target)
            return [f"{target} = gl.full({target}.shape, {value_expr}, {target}.dtype, layout={layout_expr})"]

        if call.func.attr == "block_attr":
            return []

        if call.func.attr == "reduce" and len(call.args) >= 4:
            src = self._region_base_name(call.args[0])
            dst = self._region_base_name(call.args[1])
            if src is None or dst is None:
                return None
            op_kind = call.args[2].value if isinstance(call.args[2], ast.Constant) else None
            axis_expr = self._lower_ast_expr(call.args[3])
            dst_layout = self._tensor_layout_expr(dst)
            if op_kind == "max":
                return [f"{dst} = gl.convert_layout(tl.max({src}, axis={axis_expr}), {dst_layout})"]
            if op_kind == "sum":
                return [f"{dst} = gl.convert_layout(tl.sum({src}, axis={axis_expr}), {dst_layout})"]
            if op_kind == "absmax":
                return [f"{dst} = gl.convert_layout(tl.max(tl.abs({src}), axis={axis_expr}), {dst_layout})"]

        if call.func.attr in {"gemm_py", "gemm"} and len(call.args) >= 3:
            a_var = self._region_base_name(call.args[0])
            b_var = self._region_base_name(call.args[1])
            acc_var = self._region_base_name(call.args[2])
            if a_var is None or b_var is None or acc_var is None:
                return None
            trans_a = False
            trans_b = False
            m_dim = None
            n_dim = None
            k_dim = None
            if call.func.attr == "gemm_py" and len(call.args) >= 8:
                trans_a = bool(self._eval_static_int_ast(call.args[3]))
                trans_b = bool(self._eval_static_int_ast(call.args[4]))
                m_val = self._eval_static_int_ast(call.args[5])
                n_val = self._eval_static_int_ast(call.args[6])
                k_val = self._eval_static_int_ast(call.args[7])
                if m_val is not None:
                    m_dim = str(m_val)
                if n_val is not None:
                    n_dim = str(n_val)
                if k_val is not None:
                    k_dim = str(k_val)
            return self._mma_lines(
                a_var,
                b_var,
                acc_var,
                trans_a=trans_a,
                trans_b=trans_b,
                m_dim=m_dim,
                n_dim=n_dim,
                k_dim=k_dim,
            )
        return None

    def _warp_shuffle_expr(self, value_expr: str, lane_expr: str) -> str:
        """Lower warp shuffle using Gluon inline PTX over a lane vector."""
        return (
            'gl.inline_asm_elementwise("mov.b32 $0, $1;", "=r,r", ['
            'gl.inline_asm_elementwise("shfl.sync.idx.b32 $0, $1, $2, 31, 0xffffffff;", "=r,r,r", ['
            f'gl.inline_asm_elementwise("mov.b32 $0, $1;", "=r,r", [{value_expr}], dtype=gl.int32, is_pure=True, pack=1), '
            f'{lane_expr}], dtype=gl.int32, is_pure=True, pack=1)'
            '], dtype=gl.float32, is_pure=True, pack=1)'
        )

    def _generate_unrolled_loop(self, stmt: GluonLoop) -> bool:
        if not any(self._is_thread_local_tensor(name) for name in self.allocs):
            return False
        start = self._eval_static_int(stmt.start)
        end = self._eval_static_int(stmt.end)
        step = self._eval_static_int(stmt.step)
        if start is None or end is None or step is None:
            return False

        saved_env = dict(self.loop_constant_env)
        for value in range(start, end, step):
            self.loop_constant_env[stmt.var] = value
            for body_stmt in stmt.body:
                self._generate_stmt_with_static_env(body_stmt)
            self.loop_constant_env = dict(saved_env)
        return True

    def _generate_stmt_with_static_env(self, stmt: Any):
        if isinstance(stmt, GluonLoop):
            self._generate_loop(stmt)
            return
        if isinstance(stmt, ast.AST):
            substituted = self._substitute_loop_constants(stmt)
            self._record_static_assignment(substituted)
            self._generate_raw_ast(substituted)
            return
        self._generate_stmt(stmt)

    def _substitute_loop_constants(self, stmt: ast.AST) -> ast.AST:
        env = dict(self.loop_constant_env)

        class ConstantSubstituter(ast.NodeTransformer):
            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Load) and node.id in env:
                    return ast.copy_location(ast.Constant(env[node.id]), node)
                return node

        return ast.fix_missing_locations(ConstantSubstituter().visit(ast.parse(ast.unparse(stmt)).body[0]))

    def _record_static_assignment(self, stmt: ast.AST):
        target_name = None
        value_node = None
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            target_name = stmt.targets[0].id
            value_node = stmt.value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            target_name = stmt.target.id
            value_node = stmt.value
        if target_name is None or value_node is None:
            return
        value = self._eval_static_int_ast(value_node)
        if value is not None:
            self.loop_constant_env[target_name] = value

    def _eval_static_int(self, value: Any) -> Optional[int]:
        if isinstance(value, int):
            return value
        if isinstance(value, ast.AST):
            return self._eval_static_int_ast(value)
        if isinstance(value, str):
            expr = self._fix_expr(str(value))
            try:
                return int(eval(expr, {"__builtins__": {}}, dict(self.loop_constant_env)))
            except Exception:
                return None
        return None

    def _eval_static_int_ast(self, node: ast.AST) -> Optional[int]:
        substituted = self._substitute_loop_constants(ast.Expr(value=node)).value
        expr = self._lower_ast_expr(substituted)
        try:
            return int(eval(expr, {"__builtins__": {}}, {}))
        except Exception:
            return None

    def _vector_axis_name(self, axis: int, var_name: str) -> str:
        suffix = "row" if axis == 0 else "col"
        name = f"{var_name}_{suffix}_idx_{self.vector_name_counter}"
        self.vector_name_counter += 1
        return name

    def _emit_vector_axis(self, axis: int, extent: str, layout: str, axis_name: str) -> None:
        if axis == 0:
            self.lines.append(
                f"{self._indent()}{axis_name} = gl.arange(0, {extent}, layout=gl.SliceLayout(1, {layout}))"
            )
        else:
            self.lines.append(
                f"{self._indent()}{axis_name} = gl.arange(0, {extent}, layout=gl.SliceLayout(0, {layout}))"
            )

    def _vector_index_expr(self, axis_info: dict[str, dict[str, Any]], var_name: str, *, for_tensor: bool) -> str:
        info = axis_info[var_name]
        if len(axis_info) == 1:
            return info["name"]
        if info["axis"] == 0:
            return f"{info['name']}[:, None]" if for_tensor else f"{info['name']}[:, None]"
        return f"{info['name']}[None, :]" if for_tensor else f"{info['name']}[None, :]"

    def _broadcast_to_target(
        self,
        value_expr: str,
        *,
        expanded_axis: int,
        target_shape: list[Any],
        target_layout: str,
    ) -> str:
        rows = str(target_shape[0])
        cols = str(target_shape[1])
        parent_layout = self._blocked_layout_expr(rows, cols)
        slice_layout = f"gl.SliceLayout({expanded_axis}, {parent_layout})"
        broadcasted = (
            f"gl.convert_layout({value_expr}, {slice_layout}).expand_dims({expanded_axis}).broadcast_to(({rows}, {cols}))"
        )
        return f"gl.convert_layout({broadcasted}, {target_layout})"

    def _lower_vectorized_index_expr(self, node: ast.AST, axis_info: dict[str, dict[str, Any]]) -> str:
        if isinstance(node, ast.Name) and node.id in axis_info:
            return self._vector_index_expr(axis_info, node.id, for_tensor=False)
        if isinstance(node, ast.Constant):
            return repr(node.value)
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "T":
                return self._dtype_attr_expr(node.attr)
            return f"{self._lower_vectorized_index_expr(node.value, axis_info)}.{node.attr}"
        if isinstance(node, ast.BinOp):
            return (
                f"({self._lower_vectorized_index_expr(node.left, axis_info)} "
                f"{self._operator_str(node.op)} "
                f"{self._lower_vectorized_index_expr(node.right, axis_info)})"
            )
        if isinstance(node, ast.UnaryOp):
            unary = {
                ast.UAdd: "+",
                ast.USub: "-",
                ast.Not: "not ",
                ast.Invert: "~",
            }[type(node.op)]
            return f"({unary}{self._lower_vectorized_index_expr(node.operand, axis_info)})"
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "T":
                    if node.func.attr == "bool":
                        return self._lower_vectorized_index_expr(node.args[0], axis_info)
                    if node.func.attr in {
                        "float16", "float32", "float64",
                        "int8", "int16", "int32", "int64",
                        "uint8", "uint16", "uint32", "uint64",
                        "bfloat16",
                    }:
                        return self._lower_vectorized_index_expr(node.args[0], axis_info)
            return self._lower_ast_expr(node)
        return self._lower_ast_expr(node)

    def _lower_vectorized_expr(
        self,
        node: ast.AST,
        axis_info: dict[str, dict[str, Any]],
        target_shape: Optional[list[Any]] = None,
        target_layout: Optional[str] = None,
    ) -> str:
        if isinstance(node, ast.Name):
            if node.id in axis_info:
                if target_shape is not None and target_layout is not None and len(axis_info) > 1:
                    expanded_axis = 1 if axis_info[node.id]["axis"] == 0 else 0
                    return self._broadcast_to_target(
                        axis_info[node.id]["name"],
                        expanded_axis=expanded_axis,
                        target_shape=target_shape,
                        target_layout=target_layout,
                    )
                return self._vector_index_expr(axis_info, node.id, for_tensor=True)
            return node.id
        if isinstance(node, ast.Constant):
            return repr(node.value)
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "T":
                return self._dtype_attr_expr(node.attr)
            return f"{self._lower_vectorized_expr(node.value, axis_info)}.{node.attr}"
        if isinstance(node, ast.BinOp):
            return (
                f"({self._lower_vectorized_expr(node.left, axis_info, target_shape, target_layout)} "
                f"{self._operator_str(node.op)} "
                f"{self._lower_vectorized_expr(node.right, axis_info, target_shape, target_layout)})"
            )
        if isinstance(node, ast.UnaryOp):
            unary = {
                ast.UAdd: "+",
                ast.USub: "-",
                ast.Not: "not ",
                ast.Invert: "~",
            }[type(node.op)]
            return f"({unary}{self._lower_vectorized_expr(node.operand, axis_info, target_shape, target_layout)})"
        if isinstance(node, ast.BoolOp):
            joiner = " and " if isinstance(node.op, ast.And) else " or "
            return "(" + joiner.join(self._lower_vectorized_expr(v, axis_info, target_shape, target_layout) for v in node.values) + ")"
        if isinstance(node, ast.Compare):
            left = self._lower_vectorized_expr(node.left, axis_info, target_shape, target_layout)
            parts = []
            for op, comp in zip(node.ops, node.comparators):
                parts.append(f"{left} {self._cmp_str(op)} {self._lower_vectorized_expr(comp, axis_info, target_shape, target_layout)}")
                left = self._lower_vectorized_expr(comp, axis_info, target_shape, target_layout)
            return "(" + " and ".join(parts) + ")"
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "T":
                    if node.func.attr == "if_then_else":
                        cond = self._lower_vectorized_expr(node.args[0], axis_info, target_shape, target_layout)
                        true_val = self._lower_vectorized_expr(node.args[1], axis_info, target_shape, target_layout)
                        false_val = self._lower_vectorized_expr(node.args[2], axis_info, target_shape, target_layout)
                        return f"tl.where({cond}, {true_val}, {false_val})"
                    if node.func.attr == "Cast":
                        dtype_arg = node.args[0]
                        if isinstance(dtype_arg, ast.Constant):
                            dtype_expr = self._dtype_attr_expr(str(dtype_arg.value))
                        else:
                            dtype_expr = self._lower_vectorized_expr(dtype_arg, axis_info, target_shape, target_layout)
                        value_expr = self._lower_vectorized_expr(node.args[1], axis_info, target_shape, target_layout)
                        return f"tl.cast({value_expr}, {dtype_expr})"
                    if node.func.attr == "infinity":
                        return "float('inf')"
                    if node.func.attr == "exp2":
                        return f"tl.exp2({self._lower_vectorized_expr(node.args[0], axis_info, target_shape, target_layout)})"
                    if node.func.attr == "log2":
                        return f"tl.log2({self._lower_vectorized_expr(node.args[0], axis_info, target_shape, target_layout)})"
                    if node.func.attr == "max":
                        return f"tl.maximum({self._lower_vectorized_expr(node.args[0], axis_info, target_shape, target_layout)}, {self._lower_vectorized_expr(node.args[1], axis_info, target_shape, target_layout)})"
                    if node.func.attr == "min":
                        return f"tl.minimum({self._lower_vectorized_expr(node.args[0], axis_info, target_shape, target_layout)}, {self._lower_vectorized_expr(node.args[1], axis_info, target_shape, target_layout)})"
                    if node.func.attr == "bool":
                        return self._lower_vectorized_expr(node.args[0], axis_info, target_shape, target_layout)
                    if node.func.attr in {
                        "float16", "float32", "float64",
                        "int8", "int16", "int32", "int64",
                        "uint8", "uint16", "uint32", "uint64",
                        "bfloat16",
                    }:
                        return self._lower_vectorized_expr(node.args[0], axis_info, target_shape, target_layout)
                if node.func.attr == "astype":
                    base = self._lower_vectorized_expr(node.func.value, axis_info, target_shape, target_layout)
                    dtype_expr = self._lower_vectorized_expr(node.args[0], axis_info, target_shape, target_layout)
                    return f"{base}.to({dtype_expr})"
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                args = ", ".join(self._lower_vectorized_expr(arg, axis_info, target_shape, target_layout) for arg in node.args)
                return f"{func_name}({args})"
            return self._lower_ast_expr(node)
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name):
                tensor_name = node.value.id
                indices = self._subscript_indices(node)
                alloc = self.allocs.get(tensor_name)
                if alloc is not None:
                    shape = list(getattr(alloc, "shape", None) or [])
                    axis_names = [idx.id if isinstance(idx, ast.Name) else None for idx in indices]
                    if len(shape) == 1 and len(indices) == 1 and axis_names[0] in axis_info:
                        if target_shape is not None and target_layout is not None and len(axis_info) > 1:
                            expanded_axis = 1 if axis_info[axis_names[0]]["axis"] == 0 else 0
                            return self._broadcast_to_target(
                                tensor_name,
                                expanded_axis=expanded_axis,
                                target_shape=target_shape,
                                target_layout=target_layout,
                            )
                        if len(axis_info) > 1 and axis_info[axis_names[0]]["axis"] == 0:
                            return f"{tensor_name}[:, None]"
                        return tensor_name
                    if len(shape) >= 2 and len(indices) >= 2:
                        if axis_names[0] in axis_info and axis_info[axis_names[0]]["axis"] == 0:
                            if axis_names[1] in axis_info and axis_info[axis_names[1]]["axis"] == 1:
                                return tensor_name
                if self._is_global_tensor(tensor_name):
                    ptr_expr = self._lower_vectorized_global_ptr_expr(tensor_name, indices, axis_info)
                    mask_expr = self._lower_vectorized_global_mask_expr(tensor_name, indices, axis_info)
                    return f"gl.load({ptr_expr}, mask={mask_expr})"
            return self._lower_ast_expr(node)
        return self._lower_ast_expr(node)

    def _lower_vectorized_global_ptr_expr(self, tensor_name: str, indices: list[ast.AST], axis_info: dict[str, dict[str, Any]]) -> str:
        lowered = [self._lower_vectorized_index_expr(idx, axis_info) for idx in indices]
        rank = self._tensor_rank(tensor_name)
        if rank <= 1 or len(lowered) == 1:
            return f"{tensor_name} + ({lowered[0]})"
        if len(lowered) == 2:
            return (
                f"{tensor_name} + ({lowered[0]}) * {self._row_stride_expr(tensor_name)} "
                f"+ ({lowered[1]})"
            )
        return self._lower_global_ptr_expr(tensor_name, indices)

    def _lower_vectorized_global_mask_expr(self, tensor_name: str, indices: list[ast.AST], axis_info: dict[str, dict[str, Any]]) -> str:
        lowered = [self._lower_vectorized_index_expr(idx, axis_info) for idx in indices]
        if self._tensor_rank(tensor_name) <= 1 or len(lowered) == 1:
            return f"({lowered[0]}) < {self._dim_expr(tensor_name, 0)}"
        if len(lowered) == 2:
            return (
                f"(({lowered[0]}) < {self._dim_expr(tensor_name, 0)}) & "
                f"(({lowered[1]}) < {self._dim_expr(tensor_name, 1)})"
            )
        return "True"

    def _emit_vectorized_assign(self, assign: ast.Assign, axis_info: dict[str, dict[str, Any]]) -> bool:
        if len(assign.targets) != 1 or not isinstance(assign.targets[0], ast.Subscript):
            return False
        target = assign.targets[0]
        if not isinstance(target.value, ast.Name):
            return False

        target_name = target.value.id
        target_shape = None
        target_layout = None
        alloc = self.allocs.get(target_name)
        if alloc is not None:
            shape = list(getattr(alloc, "shape", None) or [])
            target_shape = shape if len(shape) >= 2 else None
            target_layout = self._tensor_layout_expr(target_name) if target_shape is not None else None
        value_expr = self._lower_vectorized_expr(assign.value, axis_info, target_shape, target_layout)
        if alloc is not None:
            axis_names = [idx.id if isinstance(idx, ast.Name) else None for idx in self._subscript_indices(target)]
            if len(shape) == 1 and len(axis_names) == 1 and axis_names[0] in axis_info:
                self.lines.append(f"{self._indent()}{target_name} = {value_expr}")
                return True
            if len(shape) >= 2 and len(axis_names) >= 2:
                if axis_names[0] in axis_info and axis_names[1] in axis_info:
                    self.lines.append(f"{self._indent()}{target_name} = {value_expr}")
                    return True

        if self._is_global_tensor(target_name):
            if len(axis_info) == 1 and self._tensor_rank(target_name) > 1:
                return self._emit_vectorized_global_store(assign, axis_info, value_expr)
            ptr_expr = self._lower_vectorized_global_ptr_expr(target_name, self._subscript_indices(target), axis_info)
            mask_expr = self._lower_vectorized_global_mask_expr(target_name, self._subscript_indices(target), axis_info)
            self.lines.append(f"{self._indent()}gl.store({ptr_expr}, {value_expr}, mask={mask_expr})")
            return True

        return False

    def _contains_loop_var(self, node: ast.AST, loop_var: str) -> bool:
        return any(isinstance(child, ast.Name) and child.id == loop_var for child in ast.walk(node))

    def _emit_vectorized_global_store(
        self,
        assign: ast.Assign,
        axis_info: dict[str, dict[str, Any]],
        value_expr: str,
    ) -> bool:
        """Lower a 1D vector write into a higher-rank global tensor when only one index varies."""
        target = assign.targets[0]
        if not isinstance(target, ast.Subscript) or not isinstance(target.value, ast.Name):
            return False
        indices = self._subscript_indices(target)
        if len(indices) != 2:
            return False

        loop_var = next(iter(axis_info.keys()))
        first_depends = self._contains_loop_var(indices[0], loop_var)
        second_depends = self._contains_loop_var(indices[1], loop_var)
        if first_depends == second_depends:
            return False

        vector_name = axis_info[loop_var]["name"]
        lane_layout = self._vector_layout_expr(str(next(iter(axis_info.values()))["extent"]))

        varying_expr = self._lower_vectorized_index_expr(indices[0] if first_depends else indices[1], axis_info)
        scalar_expr = self._lower_ast_expr(indices[1] if first_depends else indices[0])
        scalar_name = f"{vector_name}_scalar"
        self.lines.append(
            f"{self._indent()}{scalar_name} = gl.full([{next(iter(axis_info.values()))['extent']}], {scalar_expr}, gl.int32, layout={lane_layout})"
        )

        if first_depends:
            row_expr = varying_expr
            col_expr = scalar_name
        else:
            row_expr = scalar_name
            col_expr = varying_expr

        ptr_expr = (
            f"{target.value.id} + ({row_expr}) * {self._row_stride_expr(target.value.id)} + ({col_expr})"
        )
        mask_expr = (
            f"(({row_expr}) < {self._dim_expr(target.value.id, 0)}) & "
            f"(({col_expr}) < {self._dim_expr(target.value.id, 1)})"
        )
        self.lines.append(f"{self._indent()}gl.store({ptr_expr}, {value_expr}, mask={mask_expr})")
        return True

    def _try_generate_vectorized_loop(self, stmt: GluonLoop) -> bool:
        if any(self._is_thread_local_tensor(name) for name in self.allocs):
            return False
        if stmt.start != 0 or stmt.step != 1:
            return False

        if len(stmt.body) == 1 and isinstance(stmt.body[0], GluonLoop):
            inner = stmt.body[0]
            if inner.start != 0 or inner.step != 1:
                return False
            if not inner.body or not all(isinstance(s, ast.Assign) for s in inner.body):
                return False

            row_extent = str(stmt.end)
            col_extent = str(inner.end)
            layout = self._blocked_layout_expr(row_extent, col_extent)
            axis_info = {
                stmt.var: {"name": self._vector_axis_name(0, stmt.var), "axis": 0, "extent": row_extent},
                inner.var: {"name": self._vector_axis_name(1, inner.var), "axis": 1, "extent": col_extent},
            }
            self._emit_vector_axis(0, row_extent, layout, axis_info[stmt.var]["name"])
            self._emit_vector_axis(1, col_extent, layout, axis_info[inner.var]["name"])
            for body_stmt in inner.body:
                if not self._emit_vectorized_assign(body_stmt, axis_info):
                    return False
            return True

        if not stmt.body or not all(isinstance(s, ast.Assign) for s in stmt.body):
            return False

        extent = str(stmt.end)
        layout = self._vector_layout_expr(extent)
        axis_info = {
            stmt.var: {"name": self._vector_axis_name(0, stmt.var), "axis": 0, "extent": extent},
        }
        self.lines.append(
            f"{self._indent()}{axis_info[stmt.var]['name']} = gl.arange(0, {extent}, layout={layout})"
        )
        for body_stmt in stmt.body:
            if not self._emit_vectorized_assign(body_stmt, axis_info):
                return False
        return True

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
