"""
Parse TileLang kernel code into an intermediate representation.
"""

import ast
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union


@dataclass
class AllocShared:
    """Shared memory allocation."""
    name: str
    shape: List[int]
    dtype: str
    scope: str = "shared"


@dataclass
class AllocFragment:
    """Fragment/accumulator allocation for tensor cores."""
    name: str
    shape: List[int]
    dtype: str


@dataclass
class AllocLocal:
    """Local/thread memory allocation."""
    name: str
    shape: List[int]
    dtype: str


@dataclass
class CopyOp:
    """Copy operation."""
    src: str
    dst: str
    src_indices: Optional[List] = None
    dst_indices: Optional[List] = None


@dataclass
class GemmOp:
    """GEMM operation."""
    A: str
    B: str
    C: str
    trans_A: bool = False
    trans_B: bool = False


@dataclass
class ClearOp:
    """Clear buffer operation."""
    buffer: str


@dataclass
class AtomicAddOp:
    """Atomic add operation."""
    target: str
    value: str
    target_indices: Optional[List] = None
    value_indices: Optional[List] = None


@dataclass
class ParallelLoop:
    """Parallel loop construct."""
    var: str
    extent: Any
    body: List[Any] = field(default_factory=list)
    extra_dims: List[Any] = field(default_factory=list)  # For multi-dimensional parallel loops
    all_vars: List[str] = field(default_factory=list)  # All loop variable names


@dataclass
class PipelinedLoop:
    """Pipelined loop construct."""
    var: str
    extent: Any
    num_stages: int
    body: List[Any] = field(default_factory=list)


@dataclass
class SerialLoop:
    """Serial loop construct."""
    var: str
    start: Any
    end: Any
    body: List[Any] = field(default_factory=list)


@dataclass
class TileLangKernel:
    """Represents a parsed TileLang kernel."""
    name: str
    params: List[Dict[str, Any]]
    block_dims: List[Any]
    block_vars: List[str]
    thread_count: int
    body: List[Any] = field(default_factory=list)


Stmt = Union[AllocShared, AllocFragment, AllocLocal, CopyOp, GemmOp,
             AtomicAddOp,
             ClearOp, ParallelLoop, PipelinedLoop, SerialLoop, ast.AST]


class TileLangParser:
    """Parser for TileLang kernel code."""

    def __init__(self):
        self.current_kernel = None

    def parse(self, source: str) -> TileLangKernel:
        """Parse TileLang kernel source code."""
        tree = ast.parse(source)
        return self._parse_module(tree)

    def _parse_module(self, tree: ast.Module) -> TileLangKernel:
        """Parse module to find kernel function."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check for T.prim_func decorator
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Attribute):
                        if decorator.attr == "prim_func":
                            return self._parse_kernel_function(node)
                    elif isinstance(decorator, ast.Name):
                        if decorator.id == "prim_func":
                            return self._parse_kernel_function(node)
        raise ValueError("No kernel function found with @T.prim_func decorator")

    def _parse_kernel_function(self, node: ast.FunctionDef) -> TileLangKernel:
        """Parse kernel function definition."""
        kernel = TileLangKernel(
            name=node.name,
            params=self._parse_params(node.args),
            block_dims=[],
            block_vars=[],
            thread_count=128,
            body=[]
        )
        self.current_kernel = kernel

        # Parse function body
        for stmt in node.body:
            parsed = self._parse_stmt(stmt)
            if parsed:
                if isinstance(parsed, list):
                    kernel.body.extend(parsed)
                else:
                    kernel.body.append(parsed)

        return kernel

    def _parse_params(self, args: ast.arguments) -> List[Dict[str, Any]]:
        """Parse function parameters."""
        params = []
        for arg in args.args:
            param = {"name": arg.arg}
            if arg.annotation:
                param["annotation"] = self._extract_annotation(arg.annotation)
            params.append(param)
        return params

    def _extract_annotation(self, annotation: ast.AST) -> Dict[str, Any]:
        """Extract type annotation information."""
        result = {}
        if isinstance(annotation, ast.Subscript):
            if isinstance(annotation.value, ast.Attribute):
                if annotation.value.attr == "Tensor":
                    result["type"] = "Tensor"
                    # Extract shape and dtype from annotation
                    if isinstance(annotation.slice, ast.Tuple):
                        elts = annotation.slice.elts
                        if len(elts) >= 2:
                            result["shape"] = self._extract_value(elts[0])
                            result["dtype"] = self._extract_dtype(elts[1])
        elif isinstance(annotation, ast.Call):
            # Handle T.Tensor((M, K), dtype) which is parsed as ast.Call
            if isinstance(annotation.func, ast.Attribute):
                if annotation.func.attr == "Tensor":
                    result["type"] = "Tensor"
                    # Extract shape and dtype from arguments
                    if annotation.args:
                        # First arg is shape
                        result["shape"] = self._extract_value(annotation.args[0])
                        # Second arg is dtype (if present)
                        if len(annotation.args) >= 2:
                            result["dtype"] = self._extract_dtype(annotation.args[1])
        return result

    def _extract_dtype(self, node: ast.AST) -> str:
        """Extract dtype from AST node."""
        if isinstance(node, ast.Attribute):
            return node.attr
        elif isinstance(node, ast.Name):
            return node.id
        return "float32"

    def _parse_stmt(self, stmt: ast.AST) -> Optional[Any]:
        """Parse a single statement."""
        if isinstance(stmt, ast.With):
            return self._parse_with_stmt(stmt)
        elif isinstance(stmt, ast.For):
            return self._parse_for_stmt(stmt)
        elif isinstance(stmt, ast.Expr):
            return self._parse_expr_stmt(stmt.value)
        elif isinstance(stmt, ast.Assign):
            return self._parse_assign_stmt(stmt)
        elif isinstance(stmt, ast.AnnAssign):
            return self._parse_ann_assign(stmt)
        elif isinstance(stmt, ast.Call):
            return self._parse_call_stmt(stmt)
        return None

    def _parse_with_stmt(self, node: ast.With) -> Optional[Any]:
        """Parse with statement (T.Kernel context)."""
        for item in node.items:
            if isinstance(item.context_expr, ast.Call):
                call = item.context_expr
                if isinstance(call.func, ast.Attribute):
                    if call.func.attr == "Kernel":
                        # Extract grid dimensions and thread count
                        self.current_kernel.block_dims = [
                            self._extract_value(arg) for arg in call.args
                        ]
                        if isinstance(item.optional_vars, ast.Tuple):
                            self.current_kernel.block_vars = [
                                elt.id for elt in item.optional_vars.elts if isinstance(elt, ast.Name)
                            ]
                        elif isinstance(item.optional_vars, ast.Name):
                            self.current_kernel.block_vars = [item.optional_vars.id]
                        # Extract thread count from keywords
                        for kw in call.keywords:
                            if kw.arg == "threads":
                                self.current_kernel.thread_count = self._extract_value(kw.value)

                        # Parse body
                        body_stmts = []
                        for stmt in node.body:
                            parsed = self._parse_stmt(stmt)
                            if parsed:
                                if isinstance(parsed, list):
                                    body_stmts.extend(parsed)
                                else:
                                    body_stmts.append(parsed)
                        return body_stmts
        return None

    def _parse_for_stmt(self, node: ast.For) -> Optional[Stmt]:
        """Parse for loop statement."""
        # Handle tuple unpacking like "for local_y, local_x in T.Parallel(...)"
        if isinstance(node.target, ast.Tuple):
            loop_vars = [elt.id for elt in node.target.elts if isinstance(elt, ast.Name)]
            loop_var = loop_vars[0] if loop_vars else "i"
        else:
            loop_var = node.target.id if isinstance(node.target, ast.Name) else "i"
            loop_vars = [loop_var]

        # Check if this is a TileLang loop construct
        if isinstance(node.iter, ast.Call):
            call = node.iter
            if isinstance(call.func, ast.Attribute):
                if call.func.attr == "Parallel":
                    # Handle 2D Parallel like T.Parallel(block_M, block_N)
                    if len(call.args) >= 2:
                        # Multi-dimensional parallel - create nested loops
                        dims = [self._extract_value(arg) for arg in call.args]
                        # For now, just use the first dimension for the outer loop
                        extent = dims[0]
                        # Store additional dimensions in a way the transformer can use
                        loop = ParallelLoop(var=loop_var, extent=extent, body=[], extra_dims=dims[1:], all_vars=loop_vars)
                    else:
                        extent = self._extract_value(call.args[0]) if call.args else 0
                        loop = ParallelLoop(var=loop_var, extent=extent, body=[])
                    for stmt in node.body:
                        parsed = self._parse_stmt(stmt)
                        if parsed:
                            loop.body.append(parsed)
                        else:
                            # Keep raw AST for statements we don't parse
                            loop.body.append(stmt)
                    return loop
                elif call.func.attr == "Pipelined":
                    extent = self._extract_value(call.args[0]) if call.args else 0
                    num_stages = 2  # default
                    for kw in call.keywords:
                        if kw.arg == "num_stages":
                            num_stages = self._extract_value(kw.value)
                    loop = PipelinedLoop(
                        var=loop_var,
                        extent=extent,
                        num_stages=num_stages,
                        body=[]
                    )
                    for stmt in node.body:
                        parsed = self._parse_stmt(stmt)
                        if parsed:
                            loop.body.append(parsed)
                    return loop
                elif call.func.attr == "serial":
                    start = self._extract_value(call.args[0]) if len(call.args) > 0 else 0
                    end = self._extract_value(call.args[1]) if len(call.args) > 1 else 0
                    loop = SerialLoop(var=loop_var, start=start, end=end, body=[])
                    for stmt in node.body:
                        parsed = self._parse_stmt(stmt)
                        if parsed:
                            loop.body.append(parsed)
                    return loop

        # Regular Python for loop
        loop = SerialLoop(
            var=loop_var,
            start=0,
            end=self._extract_value(node.iter),
            body=[]
        )
        for stmt in node.body:
            parsed = self._parse_stmt(stmt)
            if parsed:
                loop.body.append(parsed)
        return loop

    def _parse_expr_stmt(self, expr: ast.AST) -> Optional[Stmt]:
        """Parse expression statement."""
        if isinstance(expr, ast.Call):
            return self._parse_call_stmt(expr)
        return None

    def _parse_call_stmt(self, call: ast.Call) -> Optional[Stmt]:
        """Parse call statement."""
        if isinstance(call.func, ast.Attribute):
            func_name = call.func.attr

            if func_name == "alloc_shared":
                return self._parse_alloc_shared(call)
            elif func_name == "alloc_fragment":
                return self._parse_alloc_fragment(call)
            elif func_name == "alloc_local":
                return self._parse_alloc_local(call)
            elif func_name == "copy":
                return self._parse_copy(call)
            elif func_name == "gemm":
                return self._parse_gemm(call)
            elif func_name == "atomic_add":
                return self._parse_atomic_add(call)
            elif func_name == "clear":
                return self._parse_clear(call)

        return None

    def _parse_assign_stmt(self, node: ast.Assign) -> Optional[Stmt]:
        """Parse assignment statement."""
        # Check if RHS is a TileLang construct
        if isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Attribute):
                if call.func.attr == "alloc_shared":
                    result = self._parse_alloc_shared(call)
                    if node.targets and isinstance(node.targets[0], ast.Name):
                        result.name = node.targets[0].id
                    return result
                elif call.func.attr == "alloc_fragment":
                    result = self._parse_alloc_fragment(call)
                    if node.targets and isinstance(node.targets[0], ast.Name):
                        result.name = node.targets[0].id
                    return result
                elif call.func.attr == "alloc_local":
                    result = self._parse_alloc_local(call)
                    if node.targets and isinstance(node.targets[0], ast.Name):
                        result.name = node.targets[0].id
                    return result
        return node

    def _parse_ann_assign(self, node: ast.AnnAssign) -> Optional[Stmt]:
        """Parse annotated assignment."""
        if node.value and isinstance(node.value, ast.Call):
            return self._parse_call_stmt(node.value)
        return node

    def _parse_alloc_shared(self, call: ast.Call) -> AllocShared:
        """Parse T.alloc_shared() call."""
        shape = self._extract_value(call.args[0]) if call.args else [1]
        dtype = self._extract_value(call.args[1]) if len(call.args) > 1 else "float32"
        return AllocShared(name="", shape=shape, dtype=dtype)

    def _parse_alloc_fragment(self, call: ast.Call) -> AllocFragment:
        """Parse T.alloc_fragment() call."""
        shape = self._extract_value(call.args[0]) if call.args else [1]
        dtype = self._extract_value(call.args[1]) if len(call.args) > 1 else "float32"
        return AllocFragment(name="", shape=shape, dtype=dtype)

    def _parse_alloc_local(self, call: ast.Call) -> AllocLocal:
        """Parse T.alloc_local() call."""
        shape = self._extract_value(call.args[0]) if call.args else [1]
        dtype = self._extract_value(call.args[1]) if len(call.args) > 1 else "float32"
        return AllocLocal(name="", shape=shape, dtype=dtype)

    def _parse_copy(self, call: ast.Call) -> CopyOp:
        """Parse T.copy() call."""
        src = self._extract_buffer_ref(call.args[0]) if call.args else ""
        dst = self._extract_buffer_ref(call.args[1]) if len(call.args) > 1 else ""

        # Extract indices if present
        src_indices = None
        dst_indices = None
        if call.args and isinstance(call.args[0], ast.Subscript):
            src_indices = self._extract_indices(call.args[0])
        if len(call.args) > 1 and isinstance(call.args[1], ast.Subscript):
            dst_indices = self._extract_indices(call.args[1])

        return CopyOp(src=src, dst=dst, src_indices=src_indices, dst_indices=dst_indices)

    def _parse_gemm(self, call: ast.Call) -> GemmOp:
        """Parse T.gemm() call."""
        A = self._extract_value(call.args[0]) if call.args else ""
        B = self._extract_value(call.args[1]) if len(call.args) > 1 else ""
        C = self._extract_value(call.args[2]) if len(call.args) > 2 else ""

        trans_A = False
        trans_B = True  # default in TileLang

        for kw in call.keywords:
            if kw.arg == "trans_A":
                trans_A = self._extract_value(kw.value)
            elif kw.arg == "trans_B":
                trans_B = self._extract_value(kw.value)

        return GemmOp(A=A, B=B, C=C, trans_A=trans_A, trans_B=trans_B)

    def _parse_clear(self, call: ast.Call) -> ClearOp:
        """Parse T.clear() call."""
        buffer = self._extract_value(call.args[0]) if call.args else ""
        return ClearOp(buffer=buffer)

    def _parse_atomic_add(self, call: ast.Call) -> AtomicAddOp:
        """Parse T.atomic_add() call."""
        target = self._extract_buffer_ref(call.args[0]) if call.args else ""
        value = self._extract_buffer_ref(call.args[1]) if len(call.args) > 1 else ""

        target_indices = None
        value_indices = None
        if call.args and isinstance(call.args[0], ast.Subscript):
            target_indices = self._extract_indices(call.args[0])
        if len(call.args) > 1 and isinstance(call.args[1], ast.Subscript):
            value_indices = self._extract_indices(call.args[1])
        elif len(call.args) > 1:
            value = self._extract_value(call.args[1])

        return AtomicAddOp(
            target=target,
            value=value,
            target_indices=target_indices,
            value_indices=value_indices,
        )

    def _extract_buffer_ref(self, node: ast.AST) -> str:
        """Extract buffer reference from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Subscript):
            return self._extract_buffer_ref(node.value)
        elif isinstance(node, ast.Attribute):
            return node.attr
        return ""

    def _extract_indices(self, node: ast.Subscript) -> List[Any]:
        """Extract indices from subscript."""
        indices = []
        if isinstance(node.slice, ast.Tuple):
            for elt in node.slice.elts:
                indices.append(self._extract_value(elt))
        else:
            indices.append(self._extract_value(node.slice))
        return indices

    def _extract_value(self, node: ast.AST) -> Any:
        """Extract value from AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            # Handle T.float16, T.float32, etc.
            if isinstance(node.value, ast.Name) and node.value.id == "T":
                return node.attr
            return node.attr
        elif isinstance(node, ast.List):
            return [self._extract_value(elt) for elt in node.elts]
        elif isinstance(node, ast.Tuple):
            return tuple(self._extract_value(elt) for elt in node.elts)
        elif isinstance(node, ast.BinOp):
            left = self._extract_value(node.left)
            right = self._extract_value(node.right)
            op_map = {
                ast.Mult: "*",
                ast.Add: "+",
                ast.Sub: "-",
                ast.Div: "/",
                ast.FloorDiv: "//",
            }
            # Only evaluate if both operands are numeric
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                if isinstance(node.op, ast.Mult):
                    return left * right
                elif isinstance(node.op, ast.Add):
                    return left + right
                elif isinstance(node.op, ast.Sub):
                    return left - right
                elif isinstance(node.op, ast.Div):
                    return left / right
                elif isinstance(node.op, ast.FloorDiv):
                    return left // right
            # Return as string representation for symbolic expressions
            op = op_map.get(type(node.op), type(node.op).__name__.lower())
            return f"{left} {op} {right}"
        elif isinstance(node, ast.Slice):
            # Handle slice notation like 0:128
            lower = self._extract_value(node.lower) if node.lower else 0
            upper = self._extract_value(node.upper) if node.upper else None
            return slice(lower, upper)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == "ceildiv":
                    a = self._extract_value(node.args[0]) if node.args else 0
                    b = self._extract_value(node.args[1]) if len(node.args) > 1 else 1
                    # Only evaluate if both operands are numeric
                    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                        return (a + b - 1) // b
                    # Return as string for symbolic expressions
                    return f"ceildiv({a}, {b})"
        return None
