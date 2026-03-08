# TileLang to Gluon Translator

A translator that converts TileLang GPU kernels to Triton Gluon kernels.

## Overview

This project provides a translation pipeline that:
1. Parses TileLang Python kernels into an intermediate representation
2. Transforms TileLang IR to Triton Gluon IR
3. Generates Triton Gluon kernel source code
4. Verifies translation correctness

## Features

- **Decorator Support**: Use `@to_gluon` decorator to seamlessly replace `@T.prim_func`
- **Automatic Translation**: Automatically translates TileLang to Gluon at runtime
- **JIT Compilation**: Automatically compiles Gluon kernels (max 8 parallel jobs)
- **Smart Caching**: Two-level caching (memory + disk) to avoid recompilation
- **Accuracy Verification**: Optional numerical accuracy verification
- **Full Test Coverage**: 30+ test cases covering translation, decorator, and integration tests

## Project Structure

```
tilelang-to-gluon-translator/
├── src/
│   ├── __init__.py          # Package entry, exports to_gluon decorator
│   ├── decorator.py         # @to_gluon decorator implementation
│   ├── parser.py            # Parse TileLang AST
│   ├── transformer.py       # Transform to Gluon IR
│   ├── codegen.py           # Generate Gluon code
│   ├── translator.py        # Main orchestration
│   └── verifier.py          # Verification utilities
├── tests/
│   ├── test_decorator.py    # Decorator tests
│   ├── test_parser.py
│   ├── test_transformer.py
│   ├── test_codegen.py
│   └── test_integration.py
├── docs/
│   ├── design.md            # Architecture design
│   ├── mapping.md           # Primitive mapping table
│   ├── decorator-design.md  # Decorator design doc
│   └── code-review.md       # Code review report
└── examples/
    ├── example_matmul.py
    ├── example_elementwise.py
    └── verify_decorator.py  # Decorator verification script
```

## Installation

```bash
# Clone the repository
cd /mnt/d/yuliu/ws/tilelang-to-gluon-translator

# Install dependencies
pip install torch triton

# Ensure TileLang and Triton are available
export PYTHONPATH="/mnt/d/yuliu/ws/tilelang:$PYTHONPATH"
export PYTHONPATH="/mnt/d/yuliu/ws/triton/python:$PYTHONPATH"

# Install the package in editable mode
pip install -e .
```

### Build And Install A Wheel

```bash
# Build a wheel into ./dist
python -m pip wheel . -w dist --no-deps

# Install the built wheel
pip install dist/tilelang_to_gluon_translator-0.0.1-py3-none-any.whl
```

## Usage

### Method 1: Using @to_gluon Decorator (Recommended)

Simply replace `@T.prim_func` with `@to_gluon` and call directly:

```python
import tilelang.language as T
from tilelang_to_gluon_translator import to_gluon

@to_gluon  # Replace @T.prim_func
def matmul(
    A: T.Tensor((M, K), T.float16),
    B: T.Tensor((K, N), T.float16),
    C: T.Tensor((M, N), T.float32),
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

# Direct call - automatically translates, compiles, and runs
import torch
a = torch.randn(128, 64, device='cuda', dtype=torch.float16)
b = torch.randn(64, 128, device='cuda', dtype=torch.float16)
c = torch.empty(128, 128, device='cuda', dtype=torch.float32)

matmul(a, b, c)  # First call: translate + compile + run
matmul(a, b, c)  # Subsequent calls: use cache directly
```

#### Advanced Options

```python
@to_gluon(
    max_jobs=8,        # Maximum parallel compilation jobs
    verify=True,       # Enable numerical verification
    atol=1e-2,         # Absolute tolerance
    rtol=1e-2,         # Relative tolerance
    cache_dir=None     # Custom cache directory
)
def my_kernel(...):
    ...
```

#### Method 2: Using Translator API

```bash
# Translate a single file
python -m tilelang_to_gluon_translator.cli input_tilelang.py -o output_gluon.py

# Translate with verification
python -m src.translator input_tilelang.py --verify

# Translate directory
python -m src.translator input_dir/ -o output_dir/

# Control parallel compilation jobs (max 8)
python -m src.translator input.py --max-jobs 8
```

### Python API

```python
from translator import TileLangToGluonTranslator

translator = TileLangToGluonTranslator(max_jobs=8)

# Translate source code
gluon_code = translator.translate(tilelang_source)

# Translate file
output_path = translator.translate_file(
    Path("input.py"),
    Path("output/")
)

# Translate directory
output_paths = translator.translate_directory(
    Path("input_dir/"),
    Path("output_dir/")
)
```

## Supported TileLang Primitives

| TileLang | Gluon |
|----------|-------|
| `T.prim_func` | `@gluon.jit` |
| `T.Kernel()` | `gl.program_id()` |
| `T.alloc_shared()` | `gl.allocate_shared_memory()` |
| `T.alloc_fragment()` | Register tensor with `NVMMADistributedLayout` |
| `T.copy()` | `tma.async_copy_*()` |
| `T.gemm()` | `warpgroup_mma()` |
| `T.clear()` | `gl.zeros()` |
| `T.Parallel()` | Python range with layout |
| `T.Pipelined()` | Manual pipelining with barriers |

See `docs/mapping.md` for complete mapping table.

## Testing

```bash
# Run all tests
cd /mnt/d/yuliu/ws/tilelang-to-gluon-translator
pytest tests/ -v

# Run specific test file
pytest tests/test_decorator.py -v
pytest tests/test_parser.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Verify decorator
python examples/verify_decorator.py
```

## Verification

The translator includes verification capabilities to ensure:
1. Generated Gluon kernels compile successfully
2. Output matches reference implementation
3. Performance is comparable

```python
from verifier import KernelVerifier

verifier = KernelVerifier(atol=1e-2, rtol=1e-2)
result = verifier.verify(gluon_code, reference_fn, test_cases)

print(f"Verified: {result['verified']}")
print(f"Test cases passed: {result['test_cases_passed']}")
```

## Architecture

The translator follows a classic compiler pipeline:

```
TileLang Source
      ↓
  Parser (AST → TileLang IR)
      ↓
  Transformer (TileLang IR → Gluon IR)
      ↓
  Code Generator (Gluon IR → Gluon Source)
      ↓
  Verifier
```

See `docs/design.md` for detailed architecture documentation.

## Limitations

1. Layout inference uses simplified heuristics
2. Complex control flow may need manual adjustment
3. Warp specialization features not yet supported

## Contributing

1. Follow the existing code style
2. Add tests for new features
3. Update documentation
4. Ensure all tests pass

## License

Same as TileLang and Triton projects.
