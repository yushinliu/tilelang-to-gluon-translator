# TileLang to Gluon Translator

A translator that converts TileLang GPU kernels to Triton Gluon kernels.

## Overview

This project provides a translation pipeline that:
1. Parses TileLang Python kernels into an intermediate representation
2. Transforms TileLang IR to Triton Gluon IR
3. Generates Triton Gluon kernel source code
4. Verifies translation correctness

## Project Structure

```
tilelang-to-gluon-translator/
├── src/
│   ├── __init__.py
│   ├── parser.py          # Parse TileLang AST
│   ├── transformer.py     # Transform to Gluon IR
│   ├── codegen.py         # Generate Gluon code
│   ├── translator.py      # Main orchestration
│   └── verifier.py        # Verification utilities
├── tests/
│   ├── test_parser.py
│   ├── test_transformer.py
│   ├── test_codegen.py
│   └── test_integration.py
├── docs/
│   ├── design.md          # Architecture design
│   ├── mapping.md         # Primitive mapping table
│   └── code-review.md     # Code review report
└── examples/
    ├── example_matmul.py
    └── example_elementwise.py
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
```

## Usage

### Command Line

```bash
# Translate a single file
python -m src.translator input_tilelang.py -o output_gluon.py

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
pytest tests/test_parser.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
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
