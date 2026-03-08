"""
Tests for the @to_gluon decorator.
"""

import pytest
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.decorator import to_gluon, GluonKernelCache, TileLangGluonWrapper


class TestToGluonDecorator:
    """Test cases for @to_gluon decorator."""

    def test_decorator_basic_syntax(self):
        """Test that decorator can be applied."""
        # This is a syntax test - we can't actually run without TileLang
        assert callable(to_gluon)

        # Test that it can be used as a decorator
        @to_gluon
        def mock_kernel():
            pass

        assert isinstance(mock_kernel, TileLangGluonWrapper)

    def test_decorator_with_parens(self):
        """Test decorator with parentheses."""
        @to_gluon(max_jobs=4, verify=False)
        def mock_kernel():
            pass

        assert isinstance(mock_kernel, TileLangGluonWrapper)
        assert mock_kernel.max_jobs == 4

    def test_cache_hash_generation(self):
        """Test that cache generates consistent hashes."""
        cache = GluonKernelCache()

        source1 = "@T.prim_func\ndef kernel(): pass"
        source2 = "@T.prim_func\ndef kernel(): pass"
        source3 = "@T.prim_func\ndef different(): pass"

        hash1 = cache._get_hash(source1)
        hash2 = cache._get_hash(source2)
        hash3 = cache._get_hash(source3)

        assert hash1 == hash2  # Same source, same hash
        assert hash1 != hash3  # Different source, different hash

    def test_cache_memory_storage(self):
        """Test memory caching."""
        cache = GluonKernelCache()

        source = "test source code"
        kernel = lambda x: x * 2  # Mock kernel

        # Store and retrieve
        cache.set(source, kernel)
        retrieved = cache.get(source)

        assert retrieved is not None
        assert retrieved(5) == 10

    def test_wrapper_get_source(self):
        """Test getting Gluon source from wrapper."""
        # Create a mock function
        def mock_kernel():
            """Mock kernel for testing."""
            pass

        wrapper = TileLangGluonWrapper(mock_kernel)

        # Should be able to extract source
        assert wrapper.source_code is not None
        assert "mock_kernel" in wrapper.source_code

    def test_wrapper_inlines_outer_constants_for_inner_prim_func(self):
        """Extracted prim_func source should inline simple outer-scope bindings."""
        def splitk_factory():
            M, N, K = 256, 256, 256
            block_M, block_N, block_K = 64, 64, 32
            split_k = 2
            splitK = K // split_k

            @T.prim_func
            def main(
                A: T.Tensor((M, K), T.float16),
                B: T.Tensor((N, K), T.float16),
                C: T.Tensor((M, N), T.float32),
            ):
                with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), split_k, threads=128) as (bx, by, bz):
                    for ko in T.Pipelined(T.ceildiv(splitK, block_K), num_stages=0):
                        pass

            return main

        wrapper = TileLangGluonWrapper(splitk_factory)
        assert "256" in wrapper.source_code
        assert "64" in wrapper.source_code
        assert "32" in wrapper.source_code
        assert "2" in wrapper.source_code
        assert "splitK" not in wrapper.source_code

    def test_cache_persistence(self):
        """Test that cache persists across calls."""
        cache = GluonKernelCache()

        source = "def kernel(): return 42"
        kernel = lambda: 42

        # First call stores
        cache.set(source, kernel)

        # Second call retrieves from memory
        cached = cache.get(source)
        assert cached is not None
        assert cached() == 42


class TestDecoratorEdgeCases:
    """Test edge cases and error handling."""

    def test_wrapper_with_lambda(self):
        """Test that lambda functions work (they can have source)."""
        # Lambdas assigned to variables can have source
        my_lambda = lambda x: x
        wrapper = TileLangGluonWrapper(my_lambda)
        assert wrapper.source_code is not None

    def test_cache_disk_storage(self):
        """Test disk caching."""
        import tempfile
        import pickle

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = GluonKernelCache(cache_dir=tmpdir)

            source = "test kernel"
            # Use a simple string that can be pickled
            kernel_data = "compiled_kernel_data"

            # Store
            cache.set(source, kernel_data)

            # Verify file was created
            cache_file = cache.cache_dir / f"{cache._get_hash(source)}.pkl"
            assert cache_file.exists()

            # Create new cache instance pointing to same dir
            cache2 = GluonKernelCache(cache_dir=tmpdir)

            # Should retrieve from disk
            retrieved = cache2.get(source)
            assert retrieved is not None
            assert retrieved == "compiled_kernel_data"


class TestNumericalAccuracy:
    """Tests for numerical accuracy (require full TileLang environment)."""

    @pytest.mark.skip(reason="Requires full TileLang and Gluon environment")
    def test_accuracy_vs_tilelang(self):
        """
        Compare @to_gluon output with original TileLang output.
        """
        pass

    @pytest.mark.skip(reason="Requires full TileLang and Gluon environment")
    def test_simple_elementwise(self):
        """Test simple elementwise kernel."""
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
