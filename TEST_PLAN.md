# TileLang to Gluon 测试计划

## 目标
将 TileLang examples 中的测例添加到 tilelang-to-gluon-translator 测试套件中，转换为 Gluon 后验证精度一致性。

## 测例来源
`/mnt/d/yuliu/ws/tilelang/examples/`

## 测例分类与优先级

### P0 - 核心算子 (必须实现)
1. **gemm/example_gemm.py** - 矩阵乘法 (最基础)
2. **elementwise/example_elementwise_add.py** - 元素级加法
3. **norm/example_norm.py** - 归一化操作

### P1 - 重要算子 (高优先级)
4. **flash_attention/example_mha_fwd_bhsd.py** - Flash Attention 前向
5. **convolution/example_convolution.py** - 卷积
6. **gemm_splitk/example_gemm_splitk.py** - Split-K GEMM
7. **gemm_streamk/example_gemm_streamk.py** - Stream-K GEMM

### P2 - 扩展算子 (中优先级)
8. **gemm_fp8/example_gemm_fp8.py** - FP8 GEMM
9. **dequantize_gemm/example_dequantize_gemm.py** - 反量化 GEMM
10. **blocksparse_gemm/example_blocksparse_gemm.py** - 块稀疏 GEMM
11. **grouped_gemm/example_grouped_gemm.py** - 分组 GEMM

### P3 - 特殊算子 (低优先级)
12. **fusedmoe/** - MoE 融合算子
13. **deepseek_mla/** - DeepSeek MLA
14. **deepseek_deepgemm/** - DeepSeek DeepGEMM

## 测试架构

### 1. 测试基础设施 (tests/conftest.py)
- GPU 可用性检查
- TileLang kernel 加载器
- Gluon 转换器包装器
- 精度验证工具 (支持 rtol/atol 配置)
- 性能基准测试工具

### 2. 测试文件结构
```
tests/
├── conftest.py              # 测试基础设施
├── test_examples_p0.py      # 核心算子测试
├── test_examples_p1.py      # 重要算子测试
├── test_examples_p2.py      # 扩展算子测试
├── test_examples_p3.py      # 特殊算子测试
└── test_accuracy_regression.py  # 精度回归测试
```

### 3. 测试模板
每个测试遵循以下模式:
```python
def test_example_gemm():
    # 1. 加载 TileLang kernel
    tilelang_kernel = load_tilelang_kernel("gemm/example_gemm.py")

    # 2. 准备测试数据
    a, b, c_ref = prepare_test_data_gemm(M, N, K)

    # 3. 运行 TileLang kernel (参考结果)
    c_tilelang = tilelang_kernel(a, b)

    # 4. 转换为 Gluon kernel
    gluon_kernel = convert_to_gluon(tilelang_kernel)

    # 5. 运行 Gluon kernel
    c_gluon = gluon_kernel(a, b)

    # 6. 验证精度
    assert torch.allclose(c_tilelang, c_gluon, rtol=1e-2, atol=1e-2)
    assert torch.allclose(c_gluon, c_ref, rtol=1e-2, atol=1e-2)
```

## 关键约束
1. **max_jobs <= 8** - 并行编译限制
2. **GPU 必须可用** - 所有 kernel 测试必须在 GPU 上运行
3. **精度要求** - rtol=1e-2, atol=1e-2 (与原始 TileLang 测试一致)

## Task 分解

### Task 1: 增强测试基础设施
- 更新 tests/conftest.py
- 添加 TileLang kernel 动态加载功能
- 添加自动转换工具
- 添加精度验证工具

### Task 2: P0 核心算子测试
- test_examples_p0.py: gemm, elementwise, norm

### Task 3: P1 重要算子测试
- test_examples_p1.py: flash_attention, convolution, splitk, streamk

### Task 4: P2 扩展算子测试
- test_examples_p2.py: fp8, dequantize, blocksparse, grouped

### Task 5: P3 特殊算子测试
- test_examples_p3.py: fusedmoe, deepseek 系列

### Task 6: 精度回归测试
- test_accuracy_regression.py: 系统性的精度验证

### Task 7: 验证所有测试通过
- 运行完整测试套件
- 修复失败测试
- 生成测试报告

## SubAgent 分配策略

### Agent 1: 基础设施工程师
- 负责 Task 1
- 创建通用的 TileLang kernel 加载和转换工具

### Agent 2: 核心算子测试工程师
- 负责 Task 2
- 实现 P0 级别测试

### Agent 3: 扩展算子测试工程师
- 负责 Task 3 和 Task 4
- 实现 P1 和 P2 级别测试

### Agent 4: 集成验证工程师
- 负责 Task 5, 6, 7
- 运行完整验证并修复问题

## 成功标准
1. 所有 P0 测试通过
2. 90% 以上 P1 测试通过
3. 精度误差在 rtol=1e-2, atol=1e-2 范围内
4. 转换后的 Gluon kernel 输出与 TileLang kernel 输出一致
