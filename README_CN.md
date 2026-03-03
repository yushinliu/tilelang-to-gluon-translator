# TileLang 到 Gluon 转换器

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

一个将 TileLang GPU 内核自动转换为 Triton Gluon 内核的翻译器，支持装饰器方式无缝替换。

## 功能特性

- **装饰器支持**: 使用 `@to_gluon` 装饰器直接替换 `@T.prim_func`
- **自动翻译**: 运行时自动将 TileLang 代码翻译为 Gluon 代码
- **JIT 编译**: 自动编译 Gluon 内核（限制最大 8 个并行任务）
- **智能缓存**: 两级缓存机制（内存 + 磁盘），避免重复编译
- **精度验证**: 可选的数值精度验证，确保翻译正确性
- **完整测试**: 30+ 测试用例，覆盖翻译、装饰器和集成测试

## 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/yushinliu/tilelang-to-gluon-translator.git
cd tilelang-to-gluon-translator

# 安装依赖
pip install torch triton

# 确保 TileLang 和 Triton 可用
export PYTHONPATH="/path/to/tilelang:$PYTHONPATH"
export PYTHONPATH="/path/to/triton/python:$PYTHONPATH"
```

### 使用方法

#### 方式 1: 使用 @to_gluon 装饰器（推荐）

只需将 `@T.prim_func` 替换为 `@to_gluon`，然后直接调用：

```python
import tilelang.language as T
from tilelang_to_gluon import to_gluon

@to_gluon  # 替换 @T.prim_func
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

# 直接调用，自动完成翻译、编译和运行
import torch
a = torch.randn(128, 64, device='cuda', dtype=torch.float16)
b = torch.randn(64, 128, device='cuda', dtype=torch.float16)
c = torch.empty(128, 128, device='cuda', dtype=torch.float32)

matmul(a, b, c)  # 第一次调用: 翻译 + 编译 + 运行
matmul(a, b, c)  # 后续调用: 直接使用缓存
```

#### 高级选项

```python
@to_gluon(
    max_jobs=8,        # 最大并行编译任务数
    verify=True,       # 启用数值验证
    atol=1e-2,         # 绝对误差容限
    rtol=1e-2,         # 相对误差容限
    cache_dir=None     # 自定义缓存目录
)
def my_kernel(...):
    ...
```

#### 方式 2: 使用翻译器 API

```python
from tilelang_to_gluon import TileLangToGluonTranslator

translator = TileLangToGluonTranslator(max_jobs=8)

# 翻译源代码
gluon_code = translator.translate(tilelang_source_code)

# 翻译文件
translator.translate_file(
    Path("input_tilelang.py"),
    Path("output_gluon.py")
)

# 翻译目录
translator.translate_directory(
    Path("input_dir/"),
    Path("output_dir/")
)
```

#### 方式 3: 命令行

```bash
# 翻译单个文件
python -m src.translator input_tilelang.py -o output_gluon.py

# 翻译整个目录
python -m src.translator input_dir/ -o output_dir/

# 控制并行编译任务数
python -m src.translator input.py --max-jobs 8
```

## 项目结构

```
tilelang-to-gluon-translator/
├── src/
│   ├── __init__.py          # 包入口，导出 to_gluon 装饰器
│   ├── decorator.py         # @to_gluon 装饰器实现
│   ├── parser.py            # TileLang AST 解析
│   ├── transformer.py       # TileLang IR → Gluon IR 转换
│   ├── codegen.py           # Gluon 代码生成
│   ├── translator.py        # 翻译器主类
│   └── verifier.py          # 精度验证工具
├── tests/
│   ├── test_decorator.py    # 装饰器测试
│   ├── test_parser.py       # 解析器测试
│   ├── test_transformer.py  # 转换器测试
│   ├── test_codegen.py      # 代码生成测试
│   └── test_integration.py  # 集成测试
├── docs/
│   ├── design.md            # 架构设计文档
│   ├── mapping.md           # 原语映射表
│   ├── decorator-design.md  # 装饰器设计文档
│   └── code-review.md       # 代码审查报告
├── examples/
│   ├── example_matmul.py    # GEMM 示例
│   ├── example_elementwise.py  # Elementwise 示例
│   └── verify_decorator.py  # 装饰器验证脚本
├── README.md                # 英文文档
├── README_CN.md             # 中文文档（本文档）
└── pyproject.toml           # 项目配置
```

## 支持的 TileLang 原语

| TileLang | Gluon | 说明 |
|----------|-------|------|
| `@T.prim_func` | `@gluon.jit` | 内核装饰器 |
| `T.Kernel()` | `gl.program_id()` | 网格启动 |
| `T.alloc_shared()` | `gl.allocate_shared_memory()` | 共享内存分配 |
| `T.alloc_fragment()` | 寄存器张量 + `NVMMADistributedLayout` | 累加器布局 |
| `T.alloc_local()` | 寄存器张量 + `BlockedLayout` | 线程本地内存 |
| `T.copy()` | `tma.async_copy_*()` | 异步拷贝 |
| `T.gemm()` | `warpgroup_mma()` | 矩阵乘法 |
| `T.clear()` | `gl.zeros()` | 清零操作 |
| `T.Parallel()` | Python range + layout | 并行循环 |
| `T.Pipelined()` | 手动流水线 + barriers | 软件流水线 |

完整映射表见 `docs/mapping.md`

## 测试

```bash
# 运行所有测试
cd tilelang-to-gluon-translator
pytest tests/ -v

# 运行特定测试文件
pytest tests/test_decorator.py -v
pytest tests/test_parser.py -v

# 运行覆盖率测试
pytest tests/ --cov=src --cov-report=html

# 验证装饰器
python examples/verify_decorator.py
```

## 缓存机制

装饰器使用两级缓存：

1. **内存缓存**: 进程内缓存，访问速度极快
2. **磁盘缓存**: 持久化到 `~/.cache/tilelang-to-gluon/`，进程重启后仍然有效

缓存键基于源码内容的 SHA256 哈希，确保源码变更自动触发重新编译。

### 清除缓存

```python
from tilelang_to_gluon import GluonKernelCache

cache = GluonKernelCache()

# 清除所有缓存
import shutil
shutil.rmtree(cache.cache_dir)
```

## 性能

- **第一次调用**: ~1-6 秒（翻译 + 编译 + 可选验证）
- **缓存命中**: ~1 毫秒（仅缓存查找）
- **内存占用**: 每缓存内核约几 MB

## 架构

```
TileLang 源码
      ↓
  解析器 (Parser) - 解析 TileLang AST
      ↓
  转换器 (Transformer) - TileLang IR → Gluon IR
      ↓
  代码生成器 (Code Generator) - 生成 Gluon 源码
      ↓
  编译器 (JIT) - Gluon JIT 编译
      ↓
  执行
```

详细架构设计见 `docs/design.md` 和 `docs/decorator-design.md`

## 限制

1. 复杂切片表达式的 TMA 拷贝生成（需要符号表达式处理）
2. 某些复杂控制流可能需要手动调整
3. Warp specialization 特性暂不支持

## 贡献

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 许可证

与 TileLang 和 Triton 项目相同。

## 联系方式

- GitHub: https://github.com/yushinliu/tilelang-to-gluon-translator
- Issues: https://github.com/yushinliu/tilelang-to-gluon-translator/issues

## 致谢

- [TileLang](https://github.com/tile-ai/tilelang) - 优秀的 TileLang 项目
- [Triton](https://github.com/triton-lang/triton) - 强大的 Triton 编译器
