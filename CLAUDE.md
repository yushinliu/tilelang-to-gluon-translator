# CLAUDE.md - AI Assistant Rules

This file contains rules and guidelines for AI assistants working on this project.

## Compilation Constraints

### Maximum Parallel Compilation Jobs

**Rule**: The number of parallel compilation processes MUST NOT exceed 8.

```python
# CORRECT - max_jobs <= 8
@to_gluon(max_jobs=8)
def kernel(...):
    ...

# INCORRECT - max_jobs > 8
@to_gluon(max_jobs=16)  # ❌ Violates constraint
def kernel(...):
    ...
```

**Rationale**:
- GPU resources are limited
- Excessive parallel compilation can cause OOM errors
- 8 jobs provides optimal throughput without resource contention

**Implementation**:
- Default value in `decorator.py`: `max_jobs=8`
- Translator class enforces this limit
- Command-line tool validates `--max-jobs` argument

## GPU Execution Requirements

### Kernel Verification Must Run on GPU

**Rule**: All kernel correctness verification MUST be performed on actual GPU hardware.

```python
# CORRECT - Verification runs on GPU
def verify_kernel():
    # Allocate tensors on GPU
    a = torch.randn(..., device='cuda')
    b = torch.randn(..., device='cuda')
    c = torch.empty(..., device='cuda')

    # Run kernel on GPU
    kernel(a, b, c)

    # Verify results on GPU
    expected = a @ b  # PyTorch GPU matmul
    assert torch.allclose(c, expected, atol=1e-2)

# INCORRECT - Verification on CPU
def verify_kernel_wrong():
    a = torch.randn(..., device='cpu')  # ❌ Wrong device
    ...
```

**Rationale**:
- TileLang and Gluon kernels are GPU-specific
- CPU execution does not test actual kernel behavior
- Numerical precision differs between CPU and GPU

**Implementation**:
- `verifier.py` automatically moves tensors to GPU
- Tests in `tests/` use `device='cuda'` for all tensors
- CI/CD must have GPU runners for verification tests

## Additional Guidelines

### Code Quality
- Follow existing code style in the project
- Maintain test coverage above 80%
- Update documentation when adding features

### Security
- No hardcoded secrets or API keys
- Validate all user inputs
- Use parameterized queries for any database operations

### Performance
- Prefer immutable data structures
- Use caching where appropriate
- Profile before optimizing
