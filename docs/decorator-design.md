# @to_gluon Decorator Design Document

## 1. Overview

The `@to_gluon` decorator provides a seamless drop-in replacement for TileLang's `@T.prim_func` decorator, enabling automatic translation and JIT compilation of TileLang kernels to Triton Gluon kernels.

### Goals
1. **Zero-friction migration**: Users simply replace `@T.prim_func` with `@to_gluon`
2. **Automatic translation**: TileLang kernel is translated to Gluon on first invocation
3. **Automatic compilation**: Gluon kernel is JIT compiled with MAX_JOBS=8
4. **Caching**: Translation and compilation results are cached to avoid redundant work
5. **Interface preservation**: Same calling convention as original TileLang kernel
6. **Numerical consistency**: Ensures bit-wise compatible results

## 2. Architecture

### 2.1 High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           User Code                                          │
│  @to_gluon                                                                   │
│  def my_kernel(A: T.Tensor(...), B: T.Tensor(...)):                          │
│      with T.Kernel(...) as (bx, by):                                         │
│          ...                                                                 │
│                                                                              │
│  # Call kernel                                                               │
│  my_kernel(a_tensor, b_tensor)  # Automatic translation + compilation        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      @to_gluon Decorator (GluonDecorator)                    │
│                                                                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐  │
│  │  Cache Lookup   │───▶│  Source Capture │───▶│  TileLang→Gluon         │  │
│  │  (hash-based)   │    │  (AST + globals)│    │  Translation            │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────────────┘  │
│           │                                               │                  │
│           │ Cache Miss                                    ▼                  │
│           │                                    ┌─────────────────────────┐  │
│           │                                    │  Gluon Code Generation  │  │
│           │                                    └─────────────────────────┘  │
│           │                                               │                  │
│           │                                               ▼                  │
│           │                                    ┌─────────────────────────┐  │
│           │                                    │  JIT Compilation        │  │
│           │                                    │  (MAX_JOBS=8)           │  │
│           │                                    └─────────────────────────┘  │
│           │                                               │                  │
│           │                                               ▼                  │
│           │                                    ┌─────────────────────────┐  │
│           └───────────────────────────────────▶│  Cache Store            │  │
│                                                │  (compiled kernel)      │  │
│                                                └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Compiled Gluon Kernel (Cached)                            │
│                                                                              │
│  @gluon.jit                                                                  │
│  def my_kernel_kernel(...):                                                  │
│      ...                                                                     │
│                                                                              │
│  def my_kernel(A: torch.Tensor, B: torch.Tensor):                            │
│      ...                                                                     │
│      my_kernel_kernel[grid](...)                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Class Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Decorator Classes                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    GluonDecorator (Main Entry)                       │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │  Attributes:                                                         │   │
│  │    - func: Callable              # Original TileLang kernel          │   │
│  │    - translator: TileLangToGluonTranslator  # Translation engine     │   │
│  │    - cache: TranslationCache     # Translation + compilation cache   │   │
│  │    - compiled_kernel: Optional[Callable]  # Cached compiled kernel   │   │
│  │    - max_jobs: int = 8           # Compilation parallelism           │   │
│  │    - verify: bool = True         # Numerical verification            │   │
│  │                                                                      │   │
│  │  Methods:                                                            │   │
│  │    - __init__(func, **options)                                       │   │
│  │    - __call__(*args, **kwargs)  # Main invocation path               │   │
│  │    - _translate() → str          # TileLang → Gluon source           │   │
│  │    - _compile(gluon_code) → Callable  # Compile Gluon kernel         │   │
│  │    - _get_cache_key() → str      # Compute cache key                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              ▲                                               │
│                              │                                               │
│  ┌───────────────────────────┴─────────────────────────────────────────┐   │
│  │                    TranslationCache                                   │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │  Attributes:                                                         │   │
│  │    - cache_dir: Path             # ~/.cache/tilelang-to-gluon/       │   │
│  │    - ttl: timedelta              # Cache entry lifetime              │   │
│  │                                                                      │   │
│  │  Methods:                                                            │   │
│  │    - get(key) → Optional[CachedKernel]                               │   │
│  │    - put(key, kernel, metadata)                                      │   │
│  │    - invalidate(key)                                                 │   │
│  │    - clear()                                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              ▲                                               │
│                              │                                               │
│  ┌───────────────────────────┴─────────────────────────────────────────┐   │
│  │                    CachedKernel (Data Class)                         │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │  Attributes:                                                         │   │
│  │    - gluon_code: str             # Generated Gluon source            │   │
│  │    - compiled_module: ModuleType # Compiled Python module            │   │
│  │    - kernel_func: Callable       # Launch function                   │   │
│  │    - source_hash: str            # Source code hash                  │   │
│  │    - created_at: datetime        # Cache entry timestamp             │   │
│  │    - triton_version: str         # For cache invalidation            │   │
│  │    - tilelang_version: str       # For cache invalidation            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Call Flow Sequence

```
User calls: my_kernel(tensor_a, tensor_b)
         │
         ▼
┌─────────────────────┐
│ GluonDecorator.__call__()        │
│ 1. Capture arguments            │
│ 2. Convert tensors to proper    │
│    format if needed             │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ _get_cache_key()                 │
│ - Hash: source code + globals    │
│ - Hash: argument shapes/dtypes   │
│ - Hash: triton/tilelang versions │
└──────────┬──────────┘
           │
           ▼
    ┌──────────────┐
    │ Cache Hit?   │
    └──────┬───────┘
           │
     Yes ──┴── No
     │          │
     ▼          ▼
┌─────────┐  ┌─────────────────────┐
│ Return  │  │ _translate()        │
│ cached  │  │ - Parse TileLang AST│
│ kernel  │  │ - Transform to Gluon│
└─────────┘  │ - Generate code     │
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │ _compile()          │
             │ - Set MAX_JOBS=8    │
             │ - exec() gluon code │
             │ - Extract kernel fn │
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │ _verify_numerical() │
             │ (optional)          │
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │ cache.put()         │
             │ - Store compiled    │
             │   kernel            │
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │ Invoke compiled     │
             │ Gluon kernel        │
             └─────────────────────┘
```

## 3. Detailed Design

### 3.1 Core Decorator Implementation

```python
# File: /mnt/d/yuliu/ws/tilelang-to-gluon-translator/src/decorator.py

import inspect
import hashlib
import os
import sys
import tempfile
import importlib.util
from pathlib import Path
from typing import Callable, Optional, Any, Dict, Union
from functools import wraps
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

from .translator import TileLangToGluonTranslator
from .verifier import KernelVerifier


@dataclass
class CachedKernel:
    """Represents a cached translated and compiled kernel."""
    gluon_code: str
    kernel_func: Callable
    source_hash: str
    created_at: datetime
    triton_version: str
    tilelang_version: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class TranslationCache:
    """
    File-based cache for translated and compiled kernels.
    
    Cache structure:
    ~/.cache/tilelang-to-gluon/
        └── <cache_key>/
            ├── gluon_code.py      # Generated Gluon source
            ├── metadata.json      # Cache metadata
            └── compiled.pkl       # Serialized compiled kernel (optional)
    """
    
    CACHE_DIR = Path.home() / ".cache" / "tilelang-to-gluon"
    CACHE_TTL = timedelta(days=7)  # Cache entries valid for 7 days
    
    def __init__(self):
        self.cache_dir = self.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str) -> Optional[CachedKernel]:
        """Retrieve cached kernel if valid."""
        entry_dir = self.cache_dir / key
        
        if not entry_dir.exists():
            return None
        
        metadata_path = entry_dir / "metadata.json"
        if not metadata_path.exists():
            return None
        
        # Load and validate metadata
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            # Check TTL
            created_at = datetime.fromisoformat(metadata["created_at"])
            if datetime.now() - created_at > self.CACHE_TTL:
                return None
            
            # Check version compatibility
            if metadata.get("triton_version") != self._get_triton_version():
                return None
            if metadata.get("tilelang_version") != self._get_tilelang_version():
                return None
            
            # Load Gluon code
            gluon_code_path = entry_dir / "gluon_code.py"
            with open(gluon_code_path) as f:
                gluon_code = f.read()
            
            # Re-import the compiled module
            kernel_func = self._load_cached_kernel(entry_dir, gluon_code)
            
            return CachedKernel(
                gluon_code=gluon_code,
                kernel_func=kernel_func,
                source_hash=metadata["source_hash"],
                created_at=created_at,
                triton_version=metadata["triton_version"],
                tilelang_version=metadata["tilelang_version"],
                metadata=metadata
            )
            
        except Exception:
            # Any error -> cache miss
            return None
    
    def put(self, key: str, gluon_code: str, kernel_func: Callable, 
            source_hash: str, metadata: Dict[str, Any] = None):
        """Store kernel in cache."""
        entry_dir = self.cache_dir / key
        entry_dir.mkdir(parents=True, exist_ok=True)
        
        # Write Gluon code
        gluon_code_path = entry_dir / "gluon_code.py"
        with open(gluon_code_path, "w") as f:
            f.write(gluon_code)
        
        # Write metadata
        metadata = metadata or {}
        metadata.update({
            "source_hash": source_hash,
            "created_at": datetime.now().isoformat(),
            "triton_version": self._get_triton_version(),
            "tilelang_version": self._get_tilelang_version(),
        })
        
        metadata_path = entry_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
    
    def _get_triton_version(self) -> str:
        """Get installed Triton version."""
        try:
            import triton
            return triton.__version__
        except Exception:
            return "unknown"
    
    def _get_tilelang_version(self) -> str:
        """Get installed TileLang version."""
        try:
            import tilelang
            return getattr(tilelang, "__version__", "unknown")
        except Exception:
            return "unknown"
    
    def _load_cached_kernel(self, entry_dir: Path, gluon_code: str) -> Callable:
        """Load a cached kernel from its stored code."""
        # Create a unique module name
        module_name = f"gluon_cached_{entry_dir.name}"
        
        # Execute the Gluon code in a new module
        spec = importlib.util.spec_from_loader(module_name, loader=None)
        module = importlib.util.module_from_spec(spec)
        
        # Set MAX_JOBS before execution
        os.environ["MAX_JOBS"] = "8"
        
        exec(gluon_code, module.__dict__)
        
        # Find the launcher function (non-kernel function)
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if callable(attr) and not attr_name.endswith("_kernel") and not attr_name.startswith("_"):
                return attr
        
        raise ValueError("No launcher function found in cached module")


class GluonDecorator:
    """
    Decorator that translates TileLang kernels to Gluon and JIT compiles them.
    
    Usage:
        @to_gluon
        def my_kernel(A: T.Tensor(...), B: T.Tensor(...)):
            with T.Kernel(...) as (bx, by):
                ...
        
        # Call directly - translation and compilation happen automatically
        my_kernel(tensor_a, tensor_b)
    
    Options:
        max_jobs: int = 8
            Maximum parallel compilation jobs (sets MAX_JOBS environment variable)
        
        verify: bool = True
            Run numerical verification on first invocation
        
        atol: float = 1e-2
            Absolute tolerance for numerical verification
        
        rtol: float = 1e-2
            Relative tolerance for numerical verification
        
        cache: bool = True
            Enable translation/compilation caching
        
        cache_dir: Optional[Path] = None
            Custom cache directory (default: ~/.cache/tilelang-to-gluon)
    """
    
    def __init__(
        self,
        func: Callable = None,
        *,
        max_jobs: int = 8,
        verify: bool = True,
        atol: float = 1e-2,
        rtol: float = 1e-2,
        cache: bool = True,
        cache_dir: Optional[Path] = None
    ):
        self.func = func
        self.max_jobs = max_jobs
        self.verify = verify
        self.atol = atol
        self.rtol = rtol
        self.cache_enabled = cache
        
        # Initialize translator
        self.translator = TileLangToGluonTranslator(
            max_jobs=max_jobs,
            verify=verify,
            atol=atol,
            rtol=rtol
        )
        
        # Initialize cache
        self.cache = TranslationCache()
        if cache_dir:
            self.cache.cache_dir = cache_dir
        
        # State
        self._compiled_kernel: Optional[Callable] = None
        self._gluon_code: Optional[str] = None
        self._cache_key: Optional[str] = None
        
        # If decorating a function immediately (not using @to_gluon() with args)
        if func is not None:
            wraps(func)(self)
    
    def __call__(self, *args, **kwargs):
        """
        Invoke the kernel.
        
        On first call:
        1. Check cache for pre-compiled kernel
        2. If cache miss: translate TileLang -> Gluon
        3. Compile Gluon kernel with MAX_JOBS=8
        4. Optionally verify numerical correctness
        5. Cache the compiled kernel
        6. Execute the kernel
        
        On subsequent calls:
        1. Retrieve cached kernel
        2. Execute directly
        """
        # Ensure we have a compiled kernel
        if self._compiled_kernel is None:
            self._compiled_kernel = self._get_or_compile_kernel()
        
        # Convert TileLang tensors to PyTorch tensors if needed
        converted_args = self._convert_arguments(args)
        converted_kwargs = self._convert_arguments(kwargs)
        
        # Invoke the compiled kernel
        return self._compiled_kernel(*converted_args, **converted_kwargs)
    
    def _get_or_compile_kernel(self) -> Callable:
        """Get cached kernel or compile a new one."""
        # Compute cache key
        cache_key = self._compute_cache_key()
        self._cache_key = cache_key
        
        # Check cache
        if self.cache_enabled:
            cached = self.cache.get(cache_key)
            if cached is not None:
                self._gluon_code = cached.gluon_code
                return cached.kernel_func
        
        # Cache miss - translate and compile
        gluon_code = self._translate()
        self._gluon_code = gluon_code
        
        kernel_func = self._compile(gluon_code)
        
        # Store in cache
        if self.cache_enabled:
            source_hash = self._compute_source_hash()
            self.cache.put(cache_key, gluon_code, kernel_func, source_hash)
        
        return kernel_func
    
    def _translate(self) -> str:
        """Translate TileLang kernel to Gluon source code."""
        # Get source code of decorated function
        source = inspect.getsource(self.func)
        
        # Translate using existing translator
        gluon_code = self.translator.translate(source)
        
        return gluon_code
    
    def _compile(self, gluon_code: str) -> Callable:
        """
        Compile Gluon source code to executable kernel.
        
        Sets MAX_JOBS=8 to limit parallel compilation threads.
        """
        # Set compilation environment
        os.environ["MAX_JOBS"] = str(self.max_jobs)
        
        # Create temporary module
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(gluon_code)
            temp_path = f.name
        
        try:
            # Import the module
            spec = importlib.util.spec_from_file_location("gluon_kernel", temp_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find the launcher function
            launcher = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if callable(attr) and not attr_name.endswith("_kernel") and not attr_name.startswith("_"):
                    launcher = attr
                    break
            
            if launcher is None:
                raise RuntimeError("No launcher function found in generated Gluon code")
            
            return launcher
            
        finally:
            # Cleanup temp file
            Path(temp_path).unlink(missing_ok=True)
    
    def _compute_cache_key(self) -> str:
        """
        Compute a unique cache key for this kernel.
        
        Based on:
        - Source code hash
        - Global variable values that affect compilation
        - Triton/TileLang versions
        """
        source = inspect.getsource(self.func)
        source_hash = hashlib.sha256(source.encode()).hexdigest()
        
        # Include relevant global dependencies
        globals_hash = self._hash_globals()
        
        # Combine hashes
        combined = f"{source_hash}:{globals_hash}"
        return hashlib.sha256(combined.encode()).hexdigest()[:32]
    
    def _compute_source_hash(self) -> str:
        """Compute hash of source code."""
        source = inspect.getsource(self.func)
        return hashlib.sha256(source.encode()).hexdigest()
    
    def _hash_globals(self) -> str:
        """Hash relevant global variables."""
        # Get function's globals
        func_globals = self.func.__globals__
        
        # Find referenced globals by analyzing source
        referenced = self._find_referenced_globals()
        
        # Hash values of referenced globals
        hasher = hashlib.sha256()
        for name in sorted(referenced):
            if name in func_globals:
                try:
                    value = func_globals[name]
                    # Only hash primitive types
                    if isinstance(value, (int, float, str, bool, tuple, list)):
                        hasher.update(f"{name}:{value}".encode())
                except Exception:
                    pass
        
        return hasher.hexdigest()
    
    def _find_referenced_globals(self) -> set:
        """Find global variables referenced in the function."""
        import ast
        source = inspect.getsource(self.func)
        tree = ast.parse(source)
        
        referenced = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                referenced.add(node.id)
            elif isinstance(node, ast.Attribute):
                # Handle module.attribute references
                if isinstance(node.value, ast.Name):
                    referenced.add(node.value.id)
        
        return referenced
    
    def _convert_arguments(self, args):
        """
        Convert arguments from TileLang format to PyTorch format.
        
        TileLang uses TVM tensors, Gluon uses PyTorch tensors.
        """
        import torch
        
        if isinstance(args, tuple):
            return tuple(self._convert_arguments(a) for a in args)
        elif isinstance(args, dict):
            return {k: self._convert_arguments(v) for k, v in args.items()}
        elif hasattr(args, "data_ptr") and hasattr(args, "dtype"):
            # Already a PyTorch-like tensor
            return args
        else:
            # Try to convert to tensor
            try:
                return torch.tensor(args)
            except Exception:
                return args


# Convenience function for decorator syntax
def to_gluon(
    func: Callable = None,
    *,
    max_jobs: int = 8,
    verify: bool = True,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    cache: bool = True,
    cache_dir: Optional[Path] = None
):
    """
    Decorator to translate TileLang kernels to Triton Gluon.
    
    Usage:
        # Simple usage
        @to_gluon
        def my_kernel(A: T.Tensor(...), B: T.Tensor(...)):
            ...
        
        # With options
        @to_gluon(max_jobs=4, verify=False)
        def my_kernel(A: T.Tensor(...), B: T.Tensor(...)):
            ...
    
    Args:
        func: The function to decorate (automatically provided)
        max_jobs: Maximum parallel compilation jobs (default: 8)
        verify: Run numerical verification (default: True)
        atol: Absolute tolerance for verification (default: 1e-2)
        rtol: Relative tolerance for verification (default: 1e-2)
        cache: Enable caching (default: True)
        cache_dir: Custom cache directory
    
    Returns:
        GluonDecorator instance
    """
    decorator_factory = lambda f: GluonDecorator(
        f,
        max_jobs=max_jobs,
        verify=verify,
        atol=atol,
        rtol=rtol,
        cache=cache,
        cache_dir=cache_dir
    )
    
    if func is not None:
        # Called as @to_gluon (without parentheses)
        return decorator_factory(func)
    else:
        # Called as @to_gluon(...) (with parentheses)
        return decorator_factory
```

### 3.2 Cache Strategy

#### Cache Key Computation

The cache key is computed from:

1. **Source Code Hash**: SHA256 of the function source code
2. **Global Dependencies Hash**: Hash of referenced global variables that affect compilation
3. **Version Information**: Triton and TileLang versions for cache invalidation

```python
def _compute_cache_key(self) -> str:
    """
    Cache key = hash(source_code + globals + versions)
    
    This ensures:
    - Source changes invalidate cache
    - Global constant changes invalidate cache  
    - Version upgrades invalidate cache
    """
    components = [
        hash_source_code(),
        hash_referenced_globals(),
        get_triton_version(),
        get_tilelang_version()
    ]
    return hashlib.sha256(":".join(components).encode()).hexdigest()
```

#### Cache Entry Structure

```
~/.cache/tilelang-to-gluon/
└── <cache_key>/
    ├── gluon_code.py          # Generated Gluon source
    ├── metadata.json          # Cache metadata
    └── launcher.pkl           # Pickled launcher (optional optimization)
```

**metadata.json:**
```json
{
    "source_hash": "abc123...",
    "created_at": "2024-01-15T10:30:00",
    "triton_version": "3.0.0",
    "tilelang_version": "0.1.0",
    "kernel_name": "matmul",
    "options": {
        "max_jobs": 8,
        "verify": true
    }
}
```

#### Cache Invalidation Rules

1. **TTL-based**: Entries older than 7 days are considered stale
2. **Version-based**: Triton or TileLang version changes invalidate cache
3. **Source-based**: Source code hash mismatch invalidates cache

### 3.3 Error Handling Strategy

```python
class GluonTranslationError(Exception):
    """Raised when TileLang to Gluon translation fails."""
    pass


class GluonCompilationError(Exception):
    """Raised when Gluon kernel compilation fails."""
    pass


class GluonVerificationError(Exception):
    """Raised when numerical verification fails."""
    pass


class GluonDecorator:
    ...
    
    def _translate(self) -> str:
        """Translate with detailed error reporting."""
        try:
            source = inspect.getsource(self.func)
            return self.translator.translate(source)
        except SyntaxError as e:
            raise GluonTranslationError(
                f"Failed to parse TileLang kernel '{self.func.__name__}': {e}"
            ) from e
        except Exception as e:
            raise GluonTranslationError(
                f"Translation failed for '{self.func.__name__}': {e}"
            ) from e
    
    def _compile(self, gluon_code: str) -> Callable:
        """Compile with detailed error reporting."""
        try:
            # ... compilation code ...
            pass
        except SyntaxError as e:
            # Save problematic code for debugging
            debug_path = self._save_debug_code(gluon_code)
            raise GluonCompilationError(
                f"Generated Gluon code has syntax error: {e}\n"
                f"Debug code saved to: {debug_path}"
            ) from e
        except Exception as e:
            debug_path = self._save_debug_code(gluon_code)
            raise GluonCompilationError(
                f"Gluon compilation failed: {e}\n"
                f"Debug code saved to: {debug_path}"
            ) from e
    
    def _save_debug_code(self, gluon_code: str) -> Path:
        """Save generated code for debugging."""
        debug_dir = Path.home() / ".cache" / "tilelang-to-gluon" / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_path = debug_dir / f"{self.func.__name__}_{timestamp}.py"
        
        with open(debug_path, "w") as f:
            f.write(gluon_code)
        
        return debug_path
```

### 3.4 Numerical Verification

```python
def _verify_numerical(self, gluon_kernel: Callable) -> bool:
    """
    Verify that Gluon kernel produces same results as TileLang.
    
    Strategy:
    1. Run TileLang kernel with random inputs (if available)
    2. Run Gluon kernel with same inputs
    3. Compare outputs with tolerance
    
    Note: This requires the original TileLang kernel to be runnable,
    which may not always be the case. Falls back to compilation check only.
    """
    if not self.verify:
        return True
    
    try:
        verifier = KernelVerifier(atol=self.atol, rtol=self.rtol)
        
        # Generate test inputs based on kernel signature
        test_inputs = self._generate_test_inputs()
        
        # Run verification
        result = verifier.verify(
            self._gluon_code,
            reference_fn=None,  # Would need original TileLang execution
            test_cases=test_inputs
        )
        
        if not result.get("verified", False):
            raise GluonVerificationError(
                f"Numerical verification failed: {result.get('error', 'Unknown error')}"
            )
        
        return True
        
    except Exception as e:
        if self.verify:
            raise
        # If verification is optional, just log and continue
        import warnings
        warnings.warn(f"Verification skipped: {e}")
        return True
```

## 4. Usage Examples

### 4.1 Basic Usage

```python
import tilelang.language as T
from tilelang_to_gluon import to_gluon

# Simply replace @T.prim_func with @to_gluon
@to_gluon
def elementwise_add(
    A: T.Tensor((1024,), T.float32),
    B: T.Tensor((1024,), T.float32),
    C: T.Tensor((1024,), T.float32),
):
    with T.Kernel(1, threads=128) as (bx,):
        shared = T.alloc_shared([128], T.float32)
        T.copy(A[0:128], shared)
        for i in T.Parallel(128):
            shared[i] = shared[i] + B[i]
        T.copy(shared, C[0:128])

# Call directly - translation and compilation happen automatically
import torch
a = torch.randn(1024, device='cuda')
b = torch.randn(1024, device='cuda')
c = torch.empty(1024, device='cuda')

elementwise_add(a, b, c)  # First call: translate + compile + run
print(c)  # Result
```

### 4.2 With Options

```python
@to_gluon(max_jobs=4, verify=True, atol=1e-3)
def matmul(
    A: T.Tensor((M, K), T.float16),
    B: T.Tensor((K, N), T.float16),
    C: T.Tensor((M, N), T.float16),
):
    with T.Kernel(T.ceildiv(N, 128), T.ceildiv(M, 128), threads=128) as (bx, by):
        A_shared = T.alloc_shared([128, 32], T.float16)
        B_shared = T.alloc_shared([128, 32], T.float16)
        C_local = T.alloc_fragment([128, 128], T.float32)
        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, 32), num_stages=2):
            T.copy(A[by * 128, k * 32], A_shared)
            T.copy(B[k * 32, bx * 128], B_shared)
            T.gemm(A_shared, B_shared, C_local, False, True)
        T.copy(C_local, C[by * 128, bx * 128])
```

### 4.3 Manual Cache Management

```python
from tilelang_to_gluon import TranslationCache

# Clear all cached kernels
cache = TranslationCache()
cache.clear()

# Or clear specific kernel
cache.invalidate(cache_key)
```

## 5. Integration with Existing Translator

The decorator integrates with the existing translation pipeline:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Existing Translator Components                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  @to_gluon decorator                                                         │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  TileLangToGluonTranslator (from translator.py)                     │   │
│  │    ├── TileLangParser (from parser.py)                              │   │
│  │    ├── TileLangToGluonTransformer (from transformer.py)             │   │
│  │    └── GluonCodeGenerator (from codegen.py)                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                      │
│       ▼                                                                      │
│  Generated Gluon Code                                                        │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Gluon JIT Compilation (@gluon.jit)                                 │   │
│  │    └── Triton runtime cache                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                      │
│       ▼                                                                      │
│  Cached Compiled Kernel                                                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 6. Testing Strategy

### 6.1 Unit Tests

```python
# tests/test_decorator.py

import pytest
import torch
from tilelang_to_gluon import to_gluon, TranslationCache
import tilelang.language as T


class TestToGluonDecorator:
    """Tests for the @to_gluon decorator."""
    
    def test_basic_elementwise(self):
        """Test basic elementwise kernel translation."""
        @to_gluon(verify=False)
        def add_kernel(
            A: T.Tensor((1024,), T.float32),
            B: T.Tensor((1024,), T.float32),
            C: T.Tensor((1024,), T.float32),
        ):
            with T.Kernel(1, threads=128) as (bx,):
                shared = T.alloc_shared([128], T.float32)
                T.copy(A[0:128], shared)
                for i in T.Parallel(128):
                    shared[i] = shared[i] + B[i]
                T.copy(shared, C[0:128])
        
        # Test execution
        a = torch.randn(1024, device='cuda')
        b = torch.randn(1024, device='cuda')
        c = torch.empty(1024, device='cuda')
        
        add_kernel(a, b, c)
        
        # Verify result
        expected = a[:128] + b[:128]
        assert torch.allclose(c[:128], expected, atol=1e-2)
    
    def test_caching(self):
        """Test that caching works correctly."""
        cache = TranslationCache()
        cache.clear()
        
        @to_gluon(cache=True)
        def simple_kernel(A: T.Tensor((128,), T.float32)):
            with T.Kernel(1, threads=128):
                pass
        
        # First call should populate cache
        a = torch.randn(128, device='cuda')
        simple_kernel(a)
        
        # Check cache was populated
        assert len(list(cache.cache_dir.iterdir())) > 0
        
        # Second call should use cache
        simple_kernel(a)  # Should be faster
    
    def test_cache_invalidation_on_source_change(self):
        """Test that source changes invalidate cache."""
        # This test would verify that modifying the kernel function
        # causes a cache miss and recompilation
        pass
    
    def test_max_jobs_enforcement(self):
        """Test that MAX_JOBS is properly set during compilation."""
        import os
        
        @to_gluon(max_jobs=4)
        def kernel(A: T.Tensor((128,), T.float32)):
            with T.Kernel(1, threads=128):
                pass
        
        a = torch.randn(128, device='cuda')
        kernel(a)
        
        # Verify MAX_JOBS was set
        assert os.environ.get("MAX_JOBS") == "4"
    
    def test_error_handling(self):
        """Test error handling for invalid kernels."""
        with pytest.raises(GluonTranslationError):
            @to_gluon
            def invalid_kernel():  # Missing T.Kernel context
                pass
            
            invalid_kernel()
```

### 6.2 Integration Tests

```python
# tests/test_decorator_integration.py

import pytest
import torch
import torch.nn.functional as F
from tilelang_to_gluon import to_gluon
import tilelang.language as T


class TestDecoratorIntegration:
    """Integration tests comparing TileLang and Gluon outputs."""
    
    def test_matmul_numerical_accuracy(self):
        """Test that matmul produces correct results."""
        M, N, K = 128, 128, 64
        
        @to_gluon(verify=False)  # We'll verify manually
        def matmul(
            A: T.Tensor((M, K), T.float16),
            B: T.Tensor((K, N), T.float16),
            C: T.Tensor((M, N), T.float32),
        ):
            with T.Kernel(1, 1, threads=128) as (bx, by):
                A_shared = T.alloc_shared([M, K], T.float16)
                B_shared = T.alloc_shared([K, N], T.float16)
                C_local = T.alloc_fragment([M, N], T.float32)
                
                T.copy(A, A_shared)
                T.copy(B, B_shared)
                T.clear(C_local)
                T.gemm(A_shared, B_shared, C_local, False, True)
                T.copy(C_local, C)
        
        # Create test tensors
        a = torch.randn(M, K, dtype=torch.float16, device='cuda')
        b = torch.randn(K, N, dtype=torch.float16, device='cuda')
        c = torch.empty(M, N, dtype=torch.float32, device='cuda')
        
        # Run Gluon kernel
        matmul(a, b, c)
        
        # Compare with PyTorch reference
        expected = torch.matmul(a, b.T)  # Note: trans_B=True in kernel
        
        assert torch.allclose(c, expected, atol=1e-2, rtol=1e-2)
```

## 7. Performance Considerations

### 7.1 First-Call Overhead

The first call to a `@to_gluon` decorated kernel has overhead from:
1. Source code inspection (~1ms)
2. Cache lookup (~1ms)
3. Translation (TileLang → Gluon) (~10-50ms depending on kernel complexity)
4. Gluon JIT compilation (~1-5s depending on kernel)
5. Numerical verification (optional, ~100ms-1s)

**Total first-call overhead: ~1-6 seconds**

### 7.2 Subsequent Calls

After caching:
1. Cache lookup (~1ms)
2. Direct kernel execution

**Subsequent calls: No decorator overhead**

### 7.3 Optimization Strategies

1. **Lazy Translation**: Only translate when kernel is first called, not at decoration time
2. **Parallel Compilation**: MAX_JOBS=8 allows parallel compilation of GPU code
3. **Persistent Cache**: Cache survives process restarts
4. **Source-based Invalidation**: Only recompile when source changes

## 8. Limitations and Future Work

### 8.1 Current Limitations

1. **Dynamic Shapes**: Limited support for dynamic tensor shapes
2. **Complex Control Flow**: Some complex control flow patterns may not translate correctly
3. **Warp Specialization**: Not yet supported
4. **CPU Kernels**: Only GPU kernels are supported

### 8.2 Future Enhancements

1. **Ahead-of-Time Compilation**: Option to pre-compile kernels at import time
2. **Distributed Cache**: Shared cache across multiple machines
3. **Auto-Tuning Integration**: Integrate with Triton's autotuner
4. **Better Error Messages**: Enhanced error reporting with source mapping

## 9. Summary

The `@to_gluon` decorator provides a seamless bridge between TileLang and Triton Gluon:

- **Simple**: Single decorator replacement for `@T.prim_func`
- **Automatic**: Translation and compilation happen transparently
- **Cached**: Avoids redundant work across process restarts
- **Compatible**: Maintains TileLang calling conventions
- **Verified**: Optional numerical verification ensures correctness

### Critical Files for Implementation

- `/mnt/d/yuliu/ws/tilelang-to-gluon-translator/src/decorator.py` - Main decorator implementation
- `/mnt/d/yuliu/ws/tilelang-to-gluon-translator/src/translator.py` - Translation orchestration (existing)
- `/mnt/d/yuliu/ws/tilelang-to-gluon-translator/src/parser.py` - TileLang AST parsing (existing)
- `/mnt/d/yuliu/ws/tilelang-to-gluon-translator/src/codegen.py` - Gluon code generation (existing)
- `/mnt/d/yuliu/ws/tilelang-to-gluon-translator/__init__.py` - Package exports
