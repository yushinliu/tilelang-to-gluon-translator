"""
@to_gluon decorator for automatic TileLang to Gluon translation and execution.

This module provides a decorator that allows seamless replacement of @T.prim_func
with automatic translation to Gluon and JIT compilation.
"""

import ast
import hashlib
import inspect
import os
import sys
import textwrap
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
        if cache_dir is None:
            self.cache_dir = Path.home() / ".cache" / "tilelang-to-gluon"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_hash(self, source_code: str) -> str:
        """Generate hash for source code."""
        return hashlib.sha256(source_code.encode()).hexdigest()[:16]

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

        # Extract source code
        self.source_code = self._extract_source(func)

        # Set environment
        os.environ['MAX_JOBS'] = str(max_jobs)

    def _extract_source(self, func: Callable) -> str:
        """Extract source code from function."""
        try:
            source = inspect.getsource(func)
            # Dedent to handle decorated functions
            source = textwrap.dedent(source)
            return source
        except (OSError, TypeError) as e:
            raise ValueError(f"Cannot extract source code: {e}")

    def _compile_kernel(self) -> Callable:
        """Compile TileLang kernel to Gluon."""
        # Check cache first
        cached = self.cache.get(self.source_code)
        if cached is not None:
            return cached

        # Translate to Gluon
        gluon_code = self.translator.translate(self.source_code)
        self._gluon_source = gluon_code

        # Compile Gluon code at runtime
        # Create a new module namespace
        module_namespace = {
            'torch': torch,
        }

        # Add triton imports dynamically
        try:
            triton = __import__('triton')
            module_namespace['triton'] = triton
        except ImportError:
            raise ImportError("Triton is required for Gluon compilation")

        # Execute to define the kernel
        exec(gluon_code, module_namespace)

        # Extract the launcher function (not the kernel itself)
        # The generated code has both kernel_name_kernel and kernel_name
        tree = ast.parse(gluon_code)
        launcher_name = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # The launcher doesn't have the _kernel suffix
                if not node.name.endswith('_kernel'):
                    launcher_name = node.name
                    break

        if launcher_name is None:
            raise ValueError("Could not find launcher function in generated code")

        compiled_kernel = module_namespace[launcher_name]

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

        # Call the compiled Gluon kernel
        return self._compiled_kernel(*torch_args, **kwargs)

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
    cache_dir: Optional[Union[str, Path]] = None
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
        return TileLangGluonWrapper(func, max_jobs=max_jobs)

    # Used as @to_gluon(...) with parentheses
    def decorator(f: Callable) -> TileLangGluonWrapper:
        translator = TileLangToGluonTranslator(
            max_jobs=max_jobs,
            verify=verify,
            atol=atol,
            rtol=rtol
        )
        cache = GluonKernelCache(cache_dir) if cache_dir is not None else GluonKernelCache()
        return TileLangGluonWrapper(f, translator=translator, cache=cache, max_jobs=max_jobs)

    return decorator
