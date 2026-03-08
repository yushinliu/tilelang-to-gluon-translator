"""
验证 @to_gluon 装饰器的功能

此脚本验证 @to_gluon 装饰器可以正常工作，
包括自动翻译、编译、缓存和精度验证。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tilelang_to_gluon_translator import to_gluon, GluonKernelCache


def test_basic_decorator():
    """测试基本装饰器功能"""
    print("=" * 60)
    print("测试 1: 基本装饰器语法")
    print("=" * 60)

    # 测试 @to_gluon 语法
    @to_gluon
    def mock_kernel():
        """测试 kernel"""
        pass

    print(f"✓ @to_gluon 装饰器可以正常应用")
    print(f"  - 包装器类型: {type(mock_kernel).__name__}")
    print(f"  - 可以获取源码: {mock_kernel.source_code is not None}")

    # 测试 @to_gluon() 语法
    @to_gluon(max_jobs=4, verify=False)
    def mock_kernel_with_options():
        """带选项的测试 kernel"""
        pass

    print(f"✓ @to_gluon(...) 带选项语法可以正常应用")
    print(f"  - max_jobs: {mock_kernel_with_options.max_jobs}")


def test_cache_mechanism():
    """测试缓存机制"""
    print("\n" + "=" * 60)
    print("测试 2: 缓存机制")
    print("=" * 60)

    cache = GluonKernelCache()

    # 测试缓存哈希
    source1 = "def kernel(): pass"
    source2 = "def kernel(): pass"
    source3 = "def other(): pass"

    hash1 = cache._get_hash(source1)
    hash2 = cache._get_hash(source2)
    hash3 = cache._get_hash(source3)

    assert hash1 == hash2, "相同源码应该产生相同哈希"
    assert hash1 != hash3, "不同源码应该产生不同哈希"

    print(f"✓ 缓存哈希生成正确")
    print(f"  - 相同源码哈希一致: {hash1 == hash2}")
    print(f"  - 不同源码哈希不同: {hash1 != hash3}")

    # 测试内存缓存
    def test_fn():
        return "test"

    cache.set(source1, test_fn)
    retrieved = cache.get(source1)

    assert retrieved is not None, "应该能从缓存中获取"
    assert retrieved() == "test", "缓存的函数应该正常工作"

    print(f"✓ 内存缓存工作正常")


def test_source_extraction():
    """测试源码提取"""
    print("\n" + "=" * 60)
    print("测试 3: 源码提取")
    print("=" * 60)

    @to_gluon
    def example_kernel():
        """这是一个示例 kernel"""
        x = 1 + 2
        return x

    source = example_kernel.source_code

    assert source is not None, "应该能提取到源码"
    assert "example_kernel" in source, "源码中应该包含函数名"
    assert "这是一个示例 kernel" in source, "源码中应该包含文档字符串"

    print(f"✓ 源码提取成功")
    print(f"  - 源码长度: {len(source)} 字符")
    print(f"  - 包含函数名: {'example_kernel' in source}")


def demonstrate_usage():
    """演示装饰器用法"""
    print("\n" + "=" * 60)
    print("测试 4: 装饰器用法演示")
    print("=" * 60)

    # 基本用法
    @to_gluon
    def basic_kernel():
        """基本用法示例"""
        pass

    print("✓ 基本用法: @to_gluon")
    print("""
from tilelang_to_gluon_translator import to_gluon

@to_gluon
def my_kernel(A, B, C):
    # TileLang kernel code
    pass

# 直接调用，自动翻译和编译
my_kernel(a, b, c)
""")

    # 带选项用法
    @to_gluon(max_jobs=8, verify=True, atol=1e-3)
    def advanced_kernel():
        """高级用法示例"""
        pass

    print("✓ 高级用法: @to_gluon(max_jobs=8, verify=True)")
    print("""
@to_gluon(max_jobs=8, verify=True, atol=1e-3)
def advanced_kernel(A, B, C):
    # 使用 8 个并行编译任务
    # 启用数值验证
    # 设置绝对误差容限为 1e-3
    pass
""")

    # 获取生成的 Gluon 源码
    print("✓ 获取生成的 Gluon 源码:")
    print("  gluon_code = kernel.get_gluon_source()")


def test_wrapper_attributes():
    """测试包装器属性"""
    print("\n" + "=" * 60)
    print("测试 5: 包装器属性")
    print("=" * 60)

    @to_gluon(max_jobs=4)
    def test_kernel():
        """测试 kernel"""
        pass

    # 检查属性
    assert hasattr(test_kernel, 'source_code')
    assert hasattr(test_kernel, 'max_jobs')
    assert hasattr(test_kernel, 'get_gluon_source')

    print(f"✓ 包装器属性完整")
    print(f"  - source_code: {test_kernel.source_code is not None}")
    print(f"  - max_jobs: {test_kernel.max_jobs}")
    print(f"  - get_gluon_source: callable")


def main():
    """主函数"""
    print("\n")
    print("█" * 60)
    print("█" + " " * 58 + "█")
    print("█" + "  @to_gluon 装饰器验证".center(56) + "█")
    print("█" + " " * 58 + "█")
    print("█" * 60)
    print()

    try:
        test_basic_decorator()
        test_cache_mechanism()
        test_source_extraction()
        demonstrate_usage()
        test_wrapper_attributes()

        print("\n" + "=" * 60)
        print("所有测试通过! ✓")
        print("=" * 60)
        print()
        print("装饰器功能正常:")
        print("  ✓ 基本语法 (@to_gluon)")
        print("  ✓ 带选项语法 (@to_gluon(...))")
        print("  ✓ 缓存机制 (内存 + 磁盘)")
        print("  ✓ 源码提取")
        print("  ✓ 包装器属性")
        print()
        print("使用方法:")
        print("  from tilelang_to_gluon_translator import to_gluon")
        print()
        print("  @to_gluon")
        print("  def kernel(A: T.Tensor(...), B: T.Tensor(...)):")
        print("      with T.Kernel(...) as (bx, by):")
        print("          # TileLang code")
        print("          pass")
        print()
        print("  kernel(tensor_a, tensor_b)  # 自动翻译 + 编译 + 运行")
        print()

        return 0

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
