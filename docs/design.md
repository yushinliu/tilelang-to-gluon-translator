# TileLang to Triton Gluon Translator - Design Document

## 1. Overview

This document describes the design of a translator that converts TileLang kernels to Triton Gluon kernels.

## 2. Language Analysis

### 2.1 TileLang Language Characteristics

- **Base**: Built on TVM TIR (Tensor Intermediate Representation)
- **Kernel Definition**: Uses `@T.prim_func` decorator with `T.Kernel()` context for thread/block binding
- **Memory Scopes**:
  - `T.alloc_shared()` - Shared memory
  - `T.alloc_local()` - Thread-local memory
  - `T.alloc_fragment()` - Fragment memory for tensor cores
- **Key Primitives**:
  - `T.copy()` - Memory copy with TMA/async support
  - `T.gemm()` - GEMM operation via tensor cores
  - `T.Parallel()` - Parallel loop nest
  - `T.Pipelined()` - Software pipelining
  - `T.serial()/T.unroll()` - Sequential/unrolled loops
- **Data Types**: `T.Tensor()`, `T.Buffer()` with shape and dtype

### 2.2 Triton Gluon Language Characteristics

- **Base**: Built on Triton compiler stack with MLIR-based IR
- **Kernel Definition**: Uses `@gluon.jit` decorator with explicit grid launch
- **Memory Scopes**:
  - `gl.allocate_shared_memory()` - Returns `shared_memory_descriptor`
  - Register tensors with `distributed_type` - Distributed across threads
- **Key Primitives**:
  - `gl.load()/gl.store()` - Global memory access
  - `shared_memory_descriptor.load()/store()` - Shared memory access
  - `warpgroup_mma()` - Warp-group MMA (WGMMA)
  - `tma.async_copy_global_to_shared()` - TMA async copy
  - `mbarrier.*` - Memory barriers
- **Layouts**: `BlockedLayout`, `NVMMADistributedLayout`, `NVMMASharedLayout`, `DotOperandLayout`
- **Threading**: Explicit `program_id`, `num_programs`

## 3. Translation Architecture

### Phase 1: AST Parsing and IR Extraction

```
TileLang Python AST -> TVM TIR AST -> TileLang IR
```

- Parse the `@T.prim_func` decorated function
- Extract TVM TIR representation using TVM's parser
- Build TileLang-specific semantic model:
  - Kernel launch configuration (blocks, threads)
  - Memory allocations with scopes
  - Loop structures (Parallel, Pipelined, serial)
  - Copy and GEMM operations

### Phase 2: Semantic Mapping

See `mapping.md` for detailed primitive mapping.

### Phase 3: Layout Inference and Conversion

Critical challenge: TileLang uses implicit layouts while Gluon requires explicit layouts.

**Layout Inference Strategy**:
1. Analyze memory access patterns in `T.copy()` operations
2. Infer `BlockedLayout` parameters from thread extents and shapes
3. For GEMM operations, construct `NVMMADistributedLayout` and `DotOperandLayout`
4. Map shared memory swizzling to `NVMMASharedLayout`

### Phase 4: Code Generation

Generate Gluon kernel with:
1. Kernel signature with `@gluon.jit`
2. Grid calculation from TileLang block dimensions
3. Shared memory allocation with inferred layouts
4. Register tensor allocation with distributed layouts
5. Copy operations mapped to TMA or async copies
6. GEMM operations mapped to WGMMA with proper layouts
7. Barrier synchronization (mbarrier vs TileLang barriers)

## 4. Translation Challenges and Solutions

### Challenge 1: Thread Binding Model
- **TileLang**: Implicit via `T.Kernel()` context with block/thread indices
- **Gluon**: Explicit `program_id()` calls
- **Solution**: Transform context managers to explicit grid calculations

### Challenge 2: Memory Layout Explicitness
- **TileLang**: Compiler infers optimal layouts
- **Gluon**: User must specify all layouts
- **Solution**: Implement layout inference pass analyzing access patterns

### Challenge 3: Pipelining Abstraction
- **TileLang**: High-level `T.Pipelined()` with automatic stage management
- **Gluon**: Manual barrier-based pipelining
- **Solution**: Pattern match pipelined loops and generate barrier synchronization

### Challenge 4: Barrier Synchronization
- **TileLang**: Implicit barriers in `T.copy()` and `T.gemm()`
- **Gluon**: Explicit `mbarrier` operations
- **Solution**: Insert barriers at copy/GEMM boundaries

## 5. Verification Strategy

1. **Numerical Correctness**: Compare outputs on same random inputs
2. **Performance Parity**: Benchmark against original TileLang kernel
3. **Layout Verification**: Check bank conflicts and memory coalescing
4. **Edge Cases**: Test various shapes, dtypes, and thread configurations

## 6. Implementation Phases

- **Phase 1**: Basic element-wise kernels (Parallel loops, simple copies)
- **Phase 2**: Shared memory kernels with explicit layouts
- **Phase 3**: GEMM kernels with tensor core operations
- **Phase 4**: Pipelined kernels with async operations
- **Phase 5**: Advanced features (warp specialization, TMEM)

## 7. Critical Files Reference

### TileLang Files
1. `/mnt/d/yuliu/ws/tilelang/tilelang/language/kernel.py` - Core TileLang kernel launching mechanism
2. `/mnt/d/yuliu/ws/tilelang/tilelang/language/proxy.py` - TileLang buffer/tensor abstractions
3. `/mnt/d/yuliu/ws/tilelang/tilelang/language/gemm_op.py` - TileLang GEMM operation implementation
4. `/mnt/d/yuliu/ws/tilelang/tilelang/language/copy_op.py` - TileLang copy operation with TMA/async support

### Triton Gluon Files
1. `/mnt/d/yuliu/ws/triton/python/triton/experimental/gluon/language/_layouts.py` - Gluon layout definitions
2. `/mnt/d/yuliu/ws/triton/python/triton/experimental/gluon/language/_semantic.py` - Gluon semantic operations
3. `/mnt/d/yuliu/ws/triton/python/triton/experimental/gluon/language/_core.py` - Gluon core types
4. `/mnt/d/yuliu/ws/triton/python/triton/experimental/gluon/_runtime.py` - Gluon JIT compilation infrastructure

### Reference Examples
1. `/mnt/d/yuliu/ws/tilelang/testing/python/kernel/test_tilelang_kernel_gemm.py` - Reference TileLang GEMM kernels
2. `/mnt/d/yuliu/ws/triton/python/tutorials/gluon/05-wgmma.py` - Reference Gluon WGMMA implementation
