# TileLang to Gluon 测试计划

## 2026-03-12 Flash/Attention 计划更新

### 已完成
- 已把真实 upstream `/mnt/d/yuliu/ws/tilelang/examples/flash_attention/example_mha_fwd_bhsd.py` 接入 `tests/test_accuracy_regression.py` 并转正
- 已把真实 upstream `/mnt/d/yuliu/ws/tilelang/examples/flash_attention/example_mha_fwd_bshd.py` 接入 `tests/test_accuracy_regression.py` 并转正
- 已把真实 upstream `/mnt/d/yuliu/ws/tilelang/examples/flash_attention/example_gqa_fwd_bshd.py` 接入 `tests/test_accuracy_regression.py` 并转正
- pointer-mode launcher 维度 constexpr 命名现在会避开 tensor 参数名冲突，例如 tensor `K` 不再和 shape constexpr `K` 冲突
- raw-AST `T.infinity(...)` 已在 pointer-mode lowering 中映射
- parser 现在同时支持 `trans_A/trans_B` 和 `transpose_A/transpose_B` 关键字别名
- `T.region(...)` 现在会按实际变化轴正确线性化到 pointer-mode `load/store` 地址，不再假设最后两轴总是 tile 行列
- lowered `T.gemm_py(...)` 的 `transpose_B=True` 语义已接通
- vectorized single-axis elementwise lowering 已补齐 `T.exp2/T.log2/T.Cast`
- lowered `T.reduce(..., "sum", ...)` 已接通

### 当前阻塞
1. **线性注意力目录暂时被外部依赖阻塞**
   - `linear_attention` 当前缺少本地 `fla` / `einops`，不适合优先继续推进
2. **flash_decoding 需要先筛选更稳定入口**
   - `example_mha_inference.py` 的小尺寸 probe 在 TileLang 上游 `LayoutInference` 就会失败

### 下一步拆分
1. **在 attention 族里继续筛下一个可落地入口**
   - `flash_attention` 目录内优先继续看 `varlen` / 其它 shape 变体
   - `flash_attention` 之外优先看不依赖 `fla` 的 `flash_decoding` 子入口
   - 暂时不把 `linear_attention` 作为下一优先项
2. **把这轮 flash-attention lowering 泛化给更多 attention 变体**
   - 复用按实际变化轴生成的 `region` 线性化
   - 复用 lowered `gemm_py transpose_B` 语义
   - 继续看 `flash_attention` 目录下 `varlen` 的最小真实入口

## 2026-03-12 SIMT/Hadamard 计划更新

### 已完成
- `gemv` 新增一条真实 GPU 回归并通过：`naive_gemv(128, 128, 128, 128)`
- `hadamard_transform` 已新增真实 GPU 回归并通过
- lowered TIR `threadIdx` 已接通到 pointer-mode codegen
- `T.tvm_warp_shuffle(...)` 已通过 Gluon `inline_asm_elementwise("shfl.sync.idx.b32")` 接通
- 对含 thread-local local array 的静态小循环，codegen 已支持常量传播和展开

### 当前收益
- `examples/gemv/example_gemv.py` 的基础 `naive_gemv` 路径已能真实转换到 Gluon 并在 GPU 上通过精度验证
- `examples/hadamard_transform/example_hadamard.py` 已能真实转换到 Gluon 并在 GPU 上通过精度验证
- 这条路径证明当前 translator 已具备一条最小可用的 SIMT local-array + warp-shuffle lowering 链

### 下一步拆分
1. **继续推进未覆盖 SIMT 类 examples**
   - `gemv` 里的 `splitk_gemv*` / `*_tvm` 变体
   - 选择一个不依赖额外上游布局修复的 `flash_decoding` / `attention_sink` 入口先做 probe
2. **收敛 thread-local lowering 的泛化边界**
   - 当前静态展开只覆盖“边界可静态求值”的 thread-local 小循环
   - 后续需要评估是否要把这条能力泛化到更多 SIMT TIR 模式
3. **继续保留 `cast` 为单独问题线**
   - `cast` 当前仍更像 Gluon 3.6 的 FP8 rounding 运行时缺口，不与 SIMT 线混做

## 2026-03-12 FP8 计划更新

### 已完成
- `topk` 已从 `xfail` 转正
- `dynamic_shape` 已新增真实 GPU 回归并通过
- `cast` 的 parser / transformer / vectorized fragment lowering / JITKernel-TIR 入口已经补齐
- 已通过独立 Gluon probe 证明 `cast` 当前不是广播或布局转换错误，而是 computed float32 value 到 FP8 的 rounding 偏差
- lowered TIR `dynamic m/n/k` 符号和 `T.gemm_py(...)` 已接通

### 当前唯一明确阻塞
1. **Gluon 3.6 computed SSA value -> FP8 rounding**
   - 现象：对计算得到的 float32 值执行 FP8 cast/store 时，结果会向零偏一档
   - 已排除：
     - 1D/2D blocked 广播错误
     - scale/reduce 错误
     - 直接 load 的 float32 向量 cast 到 FP8 的 dtype 映射错误
   - 当前判断：更像 Gluon 运行时在该路径上的 rounding bug，而不是 translator 的 AST/codegen bug

### 下一步拆分
1. **尝试纯 codegen workaround**
   - 评估是否能通过“先 materialize 再 cast”或显式 rounding-bias 修正来模拟 RTNE
   - 目标只先覆盖 `cast/example_per_token_cast_to_fp8.py`
2. **如果 workaround 不可靠，保留 `cast` 为已知运行时限制**
   - 在状态文档里明确这是 Triton/Gluon 3.6 的运行时缺口
   - 继续推进未覆盖 examples 的其它目录
3. **继续扩展 example 覆盖**
   - `gemv`
   - `flash_decoding`
   - `linear_attention`

## 2026-03-10 SIMT 计划更新

## 2026-03-10 JITKernel/TIR 计划更新

### 已完成
- `to_gluon` 现在能直接吃实例化后的 `JITKernel`
- 已打通 `prim_func.script()` 的最小 TIR 翻译入口
- 已新增真实 upstream `cast` / `topk` 回归并保留为 `xfail`

### 当前剩余缺口
1. **fragment 标量索引 -> 向量化表达式 lowering**
   - 典型模式：
     - `y_amax_local[i] = ...`
     - `expand_max_idx[i, j] = tl.where(...)`
     - `logits_frag[i, j] = ...`
   - Gluon 不接受当前这种 Python 标量循环 + fragment 下标写法
2. **TIR reduction / compare-select 的 block 级向量化**
   - `reduce(max/absmax)` 后续通常会紧跟 elementwise compare/select
   - 需要直接生成整块张量表达式，而不是逐元素 scalar loop
3. **继续扩展真实 example 覆盖**
   - `gemv`
   - `cast`
   - `topk`
   - 再推进 `flash_decoding` / `linear_attention`

### 当前进一步收敛的具体技术点
- `cast`
  - 非 GEMM TIR fragment 已切到 blocked layout
  - `absmax` / 广播主路径已接通
  - 下一步只剩 1D blocked 值安全写回 2D 全局 tensor
- `topk`
  - 需要让 `max_val` / `col_idx` 这类 1D 值在 `tl.where` 前先变成与 `logits_frag`/`expand_max_idx` 兼容的 distributed layout

### 已完成
- 已完成最小 SIMT lowering：
  - `get_thread_binding`
  - conservative `vectorized -> serial`
  - pointer mode 下的基础全局下标 load/store
  - local scalar accumulator
- 已有真实 GPU 回归覆盖这条路径

### 下一步拆分
1. **JITKernel 输入支持**
   - 让 `to_gluon` 能直接接收 TileLang example 返回的 `tilelang.jit.kernel.JITKernel`
   - 评估是恢复高层 TileLang 源，还是新增一条 TIR/PrimFunc 翻译入口
2. **复杂逻辑线程语义**
   - 处理像 `examples/gemv` 这类 TileLang 自身已出现 `tn >= block_N` 告警的 kernel
   - 需要区分“逻辑线程索引”和“实际 CTA 线程布局”
3. **继续回填未覆盖 examples**
   - 在最小 SIMT lowering 基础上，优先继续推进 `gemv`、`cast`、`topk`

## 2026-03-09 发布门槛更新

### 发布前置条件
- 在准备 `0.0.2` 之前，必须先验证 `/mnt/d/yuliu/ws/tilelang/examples` 下所有测试入口都能：
  - 成功转换到 Gluon
  - 在 GPU 上执行
  - 通过精度验证
- 如果任一 example 仍依赖“预期失败”或“手写 PyTorch 替身”路径，则不能视为达到发布条件

### 当前状态
- 当前已覆盖子集在 Triton `3.6.0` + GPU 上实测全绿：
  - `55 passed`
- 但 examples 全量覆盖远未完成，因此当前阶段仍然是“继续开发和验证”，不是“准备发布”

### 开发策略调整
- 原计划偏向按 example 目录推进
- 现在改为“先补通用 lowering 能力，再回填 examples 覆盖”，因为未覆盖 examples 中已经暴露出一批重复的共性缺口：
  - `T.get_thread_binding(...)`
  - `T.vectorized(...)`
  - `T.alloc_var(...)`
  - TileLang 标量表达式/赋值/增量赋值 lowering
  - `T.if_then_else(...)`
  - `T.reduce_absmax(...)`
  - `T.attr(...)` / `T.comm_reducer(...)` / `T.tvm_thread_allreduce(...)`
  - `warp_specialize`

### 当前分层优先级
1. **P0: 通用 lowering 补齐**
   - `get_thread_binding` / 线程维度语义
   - serial/vectorized loop lowering
   - 标量表达式、raw AST、dtype cast、下标读写
   - `alloc_var` / `if_then_else`
2. **P1: 轻量 SIMT / reduction examples 回填**
   - `gemv`
   - `cast`
   - `topk`
   - `hadamard_transform`
3. **P2: 复杂 reduction / allreduce / attention 族**
   - `linear_attention`
   - `flash_decoding`
   - `attention_sink`
   - `blocksparse_attention`
4. **P3: 高级调度 / warp-specialized / 大模型特化算子**
   - `warp_specialize`
   - `fusedmoe`
   - `deepseek_*`

### 近期已完成的基础修复
- Triton/Gluon 目标版本切换到 `3.6.0`
- 删除自定义 `NVMMADistributedLayout`
- GEMM 默认路径增加受限 pointer-mode fallback
- 修复 `T.serial(x)` 解析为 `range(0, x)`
- 修复 serial loop 与顶层 raw AST 语句丢失问题

### 近期执行顺序
1. 为 parser / transformer / codegen 新增回归测试并保持已覆盖 GPU 子集全绿
2. 补 `get_thread_binding` 与基础 SIMT lowering
3. 用 `gemv` 做第一批真实转换验证
4. 再扩展到 `cast` / `topk` 等轻量未覆盖目录
5. 最后再评估 attention / warp-specialize / deepseek 系列

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

## v0.0.2 发布修复计划

### 背景
- 已发布的 `v0.0.1` wheel 暴露了内部顶层包 `src`
- `tilelang_to_gluon_translator` 只是转发壳，公开对象的 `__module__` 仍指向 `src.*`
- 这会让发布 API 与仓库实现细节强耦合，也会增加与外部同名 `src` 包冲突的风险

### 修复目标
1. wheel 只发布 `tilelang_to_gluon_translator` 公共包
2. 所有公开导入、CLI 入口和运行时代码都不再依赖 `src.*`
3. 增加发布 smoke test，验证公开对象来自 `tilelang_to_gluon_translator.*`
4. 用隔离安装验证 wheel，确认安装后没有顶层 `src` 包

### 执行顺序
1. 迁移运行时代码到 `tilelang_to_gluon_translator/`
2. 更新测试、示例和 README 中的导入路径
3. 调整打包配置并发布 `0.0.2`
4. 构建 wheel 后做隔离安装验证

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
