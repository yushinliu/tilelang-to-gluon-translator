"""
@to_gluon decorator for automatic TileLang to Gluon translation and execution.

This module provides a decorator that allows seamless replacement of @T.prim_func
with automatic translation to Gluon and JIT compilation.
"""

import ast
import hashlib
import inspect
import os
import re
import sys
import textwrap
from copy import deepcopy
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import torch

from .translator import TileLangToGluonTranslator


class GluonKernelCache:
    """
    Cache for compiled Gluon kernels.
    Supports both memory cache and disk cache.
    """

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        self.memory_cache: Dict[str, Any] = {}
        self.cache_version = "v2"
        if cache_dir is None:
            self.cache_dir = Path.home() / ".cache" / "tilelang-to-gluon"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_hash(self, source_code: str) -> str:
        """Generate hash for source code."""
        payload = f"{self.cache_version}:{source_code}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def get(self, source_code: str) -> Optional[Any]:
        """Get cached kernel if exists."""
        key = self._get_hash(source_code)

        # Check memory cache first
        if key in self.memory_cache:
            return self.memory_cache[key]

        # Check disk cache
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                import pickle
                with open(cache_file, 'rb') as f:
                    kernel = pickle.load(f)
                    self.memory_cache[key] = kernel
                    return kernel
            except Exception:
                # Disk cache corrupted, ignore
                pass

        return None

    def set(self, source_code: str, kernel: Any):
        """Cache compiled kernel."""
        key = self._get_hash(source_code)
        self.memory_cache[key] = kernel

        # Also save to disk
        cache_file = self.cache_dir / f"{key}.pkl"
        try:
            import pickle
            with open(cache_file, 'wb') as f:
                pickle.dump(kernel, f)
        except Exception:
            # Disk cache is optional
            pass


class TileLangGluonWrapper:
    """
    Wrapper class that handles TileLang kernel translation to Gluon.
    """

    def __init__(
        self,
        func: Callable,
        translator: Optional[TileLangToGluonTranslator] = None,
        cache: Optional[GluonKernelCache] = None,
        max_jobs: int = 8
    ):
        self.original_func = func
        self.translator = translator or TileLangToGluonTranslator(max_jobs=max_jobs)
        self.cache = cache or GluonKernelCache()
        self.max_jobs = max_jobs
        self._compiled_kernel = None
        self._gluon_source = None
        self._fallback_to_original = False

        # Extract source code
        self.source_code = self._extract_source(func)
        self._has_fragment_subscript_elementwise = self._detect_fragment_subscript_elementwise(
            self.source_code
        )

        # Set environment
        os.environ['MAX_JOBS'] = str(max_jobs)

    def _detect_fragment_subscript_elementwise(self, source: str) -> bool:
        """
        Detect elementwise-style fragment writes that are known to be problematic
        with Gluon shared memory descriptor semantics.
        """
        if "T.alloc_fragment" not in source or "T.Parallel(" not in source:
            return False
        # Example target: C_local[local_y, local_x] = ...
        return re.search(r"[A-Za-z_]\w*\s*\[[^\]]+\]\s*=", source) is not None

    def _should_silent_fallback(self, args, result) -> bool:
        """Detect silent wrong-result cases for known incompatible kernel pattern."""
        if not self._has_fragment_subscript_elementwise:
            return False

        out_tensor = None
        if isinstance(result, torch.Tensor):
            out_tensor = result
        elif args and isinstance(args[-1], torch.Tensor):
            out_tensor = args[-1]

        if out_tensor is None or out_tensor.numel() == 0:
            return False

        # Silent failure pattern observed in Gluon path: output remains all zeros.
        if torch.count_nonzero(out_tensor).item() != 0:
            return False

        # Avoid false positives when all inputs are also zeros.
        for inp in args[:-1]:
            if isinstance(inp, torch.Tensor) and inp.numel() > 0:
                if torch.count_nonzero(inp).item() != 0:
                    return True

        return False

    def _extract_source(self, func: Callable) -> str:
        """Extract source code from function."""
        try:
            # Check if this is a @tilelang.jit wrapped function (JITImpl)
            # JITImpl objects have func_source attribute with the original source
            if hasattr(func, 'func_source') and isinstance(func.func_source, str):
                source = func.func_source
                # Dedent first to remove any leading whitespace from nested definitions
                source = textwrap.dedent(source)
                # Extract the inner @T.prim_func kernel from @tilelang.jit wrapper
                inner_source = self._extract_inner_prim_func(source)
                if inner_source:
                    return self._inline_outer_constants(source, inner_source)
                # If extraction fails, return the full source
                return source

            # For regular functions, use inspect
            source = inspect.getsource(func)
            # Dedent to handle decorated functions
            source = textwrap.dedent(source)

            inner_source = self._extract_inner_prim_func(source)
            if inner_source:
                return self._inline_outer_constants(source, inner_source)

            # Check if this is a @tilelang.jit wrapped function (from source inspection)
            if self._is_tilelang_jit_wrapper(source):
                return source

            return source
        except (OSError, TypeError) as e:
            raise ValueError(f"Cannot extract source code: {e}")

    def _is_tilelang_jit_wrapper(self, source: str) -> bool:
        """Check if source is from a @tilelang.jit wrapped function."""
        # Check for patterns that indicate @tilelang.jit wrapper
        patterns = [
            "@tilelang.jit",
            "@tilelang.jit(out_idx",
            "@tilelang.jit(target",
            "return elem_add",  # Common pattern in examples
            "return gemm",      # Common pattern in examples
            "return kernel",    # Generic pattern
        ]
        source_lines = source.split('\n')
        first_lines = '\n'.join(source_lines[:10])  # Check first 10 lines

        # Check if it has tilelang.jit decorator
        has_jit_decorator = "@tilelang.jit" in first_lines or "@ tilelang.jit" in first_lines

        # Check if it returns an inner function
        has_return_inner = any(
            pattern in source for pattern in ["return elem_add", "return gemm", "return kernel"]
        )

        return has_jit_decorator or has_return_inner

    def _extract_inner_prim_func(self, source: str) -> Optional[str]:
        """Extract the inner @T.prim_func function from a @tilelang.jit wrapper."""
        try:
            tree = ast.parse(source)

            # Find the outer function definition
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Look for nested function definitions with @T.prim_func decorator
                    for inner_node in ast.walk(node):
                        if isinstance(inner_node, ast.FunctionDef) and inner_node != node:
                            # Check if it has @T.prim_func decorator
                            for decorator in inner_node.decorator_list:
                                if isinstance(decorator, ast.Attribute):
                                    if decorator.attr == "prim_func":
                                        # Extract this inner function
                                        return self._extract_function_source(source, inner_node)
                                elif isinstance(decorator, ast.Name):
                                    if decorator.id == "prim_func":
                                        return self._extract_function_source(source, inner_node)

            # Alternative: look for functions that use T.Kernel (TileLang kernel signature)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    for inner_node in ast.walk(node):
                        if isinstance(inner_node, ast.FunctionDef) and inner_node != node:
                            # Check if it uses T.Kernel (indicates TileLang kernel)
                            if self._has_t_kernel(inner_node):
                                return self._extract_function_source(source, inner_node)

        except SyntaxError:
            pass

        return None

    def _inline_outer_constants(self, outer_source: str, inner_source: str) -> str:
        """Inline simple outer-scope bindings into the extracted prim_func source."""
        try:
            outer_tree = ast.parse(outer_source)
            outer_func = next((node for node in ast.walk(outer_tree) if isinstance(node, ast.FunctionDef)), None)
            if outer_func is None:
                return inner_source

            bindings: Dict[str, ast.AST] = {}
            for stmt in outer_func.body:
                if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                    continue
                target = stmt.targets[0]
                if isinstance(target, ast.Name):
                    value = self._replace_bound_names(deepcopy(stmt.value), bindings)
                    bindings[target.id] = value
                    continue
                if isinstance(target, ast.Tuple) and isinstance(stmt.value, ast.Tuple):
                    values = [self._replace_bound_names(deepcopy(v), bindings) for v in stmt.value.elts]
                    for name_node, value in zip(target.elts, values):
                        if isinstance(name_node, ast.Name):
                            bindings[name_node.id] = value

            if not bindings:
                return inner_source

            inner_tree = ast.parse(inner_source)
            inliner = _OuterBindingInliner(bindings)
            updated_tree = inliner.visit(inner_tree)
            ast.fix_missing_locations(updated_tree)
            return ast.unparse(updated_tree)
        except Exception:
            return inner_source

    def _replace_bound_names(self, node: ast.AST, bindings: Dict[str, ast.AST]) -> ast.AST:
        """Recursively substitute previously collected outer bindings."""
        class _BindingResolver(ast.NodeTransformer):
            def visit_Name(self, inner_node: ast.Name):
                if isinstance(inner_node.ctx, ast.Load) and inner_node.id in bindings:
                    return deepcopy(bindings[inner_node.id])
                return inner_node

        return _BindingResolver().visit(node)

    def _has_t_kernel(self, node: ast.FunctionDef) -> bool:
        """Check if function body contains T.Kernel usage."""
        for child in ast.walk(node):
            if isinstance(child, ast.With):
                for item in child.items:
                    if isinstance(item.context_expr, ast.Call):
                        if isinstance(item.context_expr.func, ast.Attribute):
                            if item.context_expr.func.attr == "Kernel":
                                return True
        return False

    def _extract_function_source(self, source: str, node: ast.FunctionDef) -> str:
        """Extract source code for a specific AST node."""
        lines = source.split('\n')

        # Get line numbers (1-indexed in AST, 0-indexed in list)
        # Include decorators in the extraction
        start_line = (node.decorator_list[0].lineno - 1) if node.decorator_list else (node.lineno - 1)
        end_line = node.end_lineno

        # Extract the lines
        func_lines = lines[start_line:end_line]

        # Dedent to remove leading whitespace
        func_source = '\n'.join(func_lines)
        func_source = textwrap.dedent(func_source)

        return func_source

    def _compile_kernel(self) -> Callable:
        """Compile TileLang kernel to Gluon."""
        # Check cache first
        cached = self.cache.get(self.source_code)
        if cached is not None:
            return cached

        # Translate to Gluon
        try:
            gluon_code = self.translator.translate(self.source_code)
            self._gluon_source = gluon_code
        except ValueError:
            raise

        # Write generated code to a temporary file so Gluon can inspect it
        import tempfile
        import importlib.util
        import os

        # Create a temporary file with unique name based on source hash
        source_hash = hashlib.sha256(self.source_code.encode()).hexdigest()[:16]
        temp_dir = Path(tempfile.gettempdir()) / "tilelang_gluon_kernels"
        temp_dir.mkdir(parents=True, exist_ok=True)

        temp_file = temp_dir / f"kernel_{source_hash}.py"

        # Write the generated code to file
        with open(temp_file, 'w') as f:
            f.write(gluon_code)

        # Import the module from file
        spec = importlib.util.spec_from_file_location(f"kernel_{source_hash}", temp_file)
        module = importlib.util.module_from_spec(spec)

        # Need to add torch and triton to module globals
        module.__dict__['torch'] = torch
        try:
            import triton
            module.__dict__['triton'] = triton
        except ImportError:
            pass

        spec.loader.exec_module(module)

        # Find the launcher function (not the kernel itself)
        tree = ast.parse(gluon_code)
        launcher_name = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # The launcher doesn't have the _kernel suffix
                if not node.name.endswith('_kernel') and not node.name.startswith('_'):
                    launcher_name = node.name
                    break

        if launcher_name is None:
            raise ValueError("Could not find launcher function in generated code")

        compiled_kernel = getattr(module, launcher_name)

        # Cache the compiled kernel
        self.cache.set(self.source_code, compiled_kernel)

        return compiled_kernel

    def __call__(self, *args, **kwargs):
        """Execute the kernel with given arguments."""
        if self._compiled_kernel is None:
            self._compiled_kernel = self._compile_kernel()

        # Convert arguments to torch.Tensor if needed
        torch_args = []
        for arg in args:
            if hasattr(arg, 'data'):  # TileLang tensor wrapper
                torch_args.append(arg.data)
            elif isinstance(arg, torch.Tensor):
                torch_args.append(arg)
            else:
                torch_args.append(arg)

        # Strict behavior: if Gluon compilation/runtime fails, raise exception.
        return self._compiled_kernel(*torch_args, **kwargs)

    def _execute_original_fallback(self, args, kwargs):
        """Execute original TileLang function as a compatibility fallback."""
        # First try obtaining executable kernel (common @tilelang.jit wrapper path).
        kernel = None
        try:
            kernel = self.original_func()
        except Exception:
            kernel = None

        if callable(kernel):
            if args and isinstance(args[-1], torch.Tensor):
                # Common TileLang path: kernel(inputs...) -> output tensor
                try:
                    result = kernel(*args[:-1], **kwargs)
                    if isinstance(result, torch.Tensor) and result.shape == args[-1].shape:
                        args[-1].copy_(result)
                        return args[-1]
                except TypeError:
                    pass
            return kernel(*args, **kwargs)

        # Fallback to direct execution for non-factory callables.
        result = self.original_func(*args, **kwargs)
        if isinstance(result, torch.Tensor) and args and isinstance(args[-1], torch.Tensor):
            # If caller passed output tensor, write into it.
            if result.shape == args[-1].shape:
                args[-1].copy_(result)
                return args[-1]
        return result

    def get_gluon_source(self) -> Optional[str]:
        """Get the generated Gluon source code."""
        if self._gluon_source is None:
            self._gluon_source = self.translator.translate(self.source_code)
        return self._gluon_source


def to_gluon(
    func: Optional[Callable] = None,
    *,
    max_jobs: int = 8,
    verify: bool = True,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    cache_dir: Optional[Union[str, Path]] = None,
    use_pointer_mode: bool = False  # Default to TMA mode (pointer mode needs more work)
) -> Union[Callable, TileLangGluonWrapper]:
    """
    Decorator to automatically translate TileLang kernels to Gluon.

    This decorator replaces @T.prim_func and handles:
    - Automatic translation from TileLang to Gluon
    - JIT compilation using Gluon
    - Caching of translated and compiled kernels
    - Transparent execution with torch.Tensor arguments

    Args:
        func: The function to decorate (when used without parentheses)
        max_jobs: Maximum parallel compilation jobs (default: 8)
        verify: Whether to verify translation correctness
        atol: Absolute tolerance for verification
        rtol: Relative tolerance for verification
        cache_dir: Directory for caching compiled kernels

    Example:
        from tilelang import T
        from tilelang_to_gluon import to_gluon

        @to_gluon
        def matmul(A: T.Tensor, B: T.Tensor, C: T.Tensor):
            with T.Kernel(T.ceildiv(N, 128), T.ceildiv(M, 128), threads=128) as (bx, by):
                # ... TileLang code
                pass

        # Direct call - automatically translates and compiles
        matmul(a, b, c)
    """
    # Handle both @to_gluon and @to_gluon() syntax
    if func is not None and callable(func):
        # Used as @to_gluon without parentheses
        translator = TileLangToGluonTranslator(
            max_jobs=max_jobs,
            verify=verify,
            atol=atol,
            rtol=rtol,
            use_pointer_mode=use_pointer_mode
        )
        return TileLangGluonWrapper(func, translator=translator, max_jobs=max_jobs)

    # Used as @to_gluon(...) with parentheses
    def decorator(f: Callable) -> TileLangGluonWrapper:
        translator = TileLangToGluonTranslator(
            max_jobs=max_jobs,
            verify=verify,
            atol=atol,
            rtol=rtol,
            use_pointer_mode=use_pointer_mode
        )
        cache = GluonKernelCache(cache_dir) if cache_dir is not None else GluonKernelCache()
        return TileLangGluonWrapper(f, translator=translator, cache=cache, max_jobs=max_jobs)

    return decorator


class _OuterBindingInliner(ast.NodeTransformer):
    """Inline extracted outer-scope bindings into the prim_func AST."""

    def __init__(self, bindings: Dict[str, ast.AST]):
        self.bindings = bindings

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load) and node.id in self.bindings:
            return deepcopy(self.bindings[node.id])
        return node
