"""
Comprehensive GPU kernel test suite.

Tests multiple kernel types with GPU verification:
- Elementwise operations
- Matrix multiplication (matmul)
- Reduction operations
- Performance benchmarks
"""

import pytest
import torch
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.decorator import to_gluon


class TestElementwiseKernels:
    """GPU tests for elementwise kernels."""

    @pytest.mark.gpu
    def test_elementwise_add(self, device, tensor_factory, verify_tensors):
        """Test elementwise addition kernel."""
        @to_gluon(max_jobs=8, verify=False)
        def elementwise_add(A, B, C):
            """C = A + B"""
            with torch.no_grad():
                C.copy_(A + B)

        a = torch.randn(1024, device=device)
        b = torch.randn(1024, device=device)
        c = torch.zeros(1024, device=device)

        elementwise_add(a, b, c)

        expected = a + b
        verify_tensors(c, expected)

    @pytest.mark.gpu
    def test_elementwise_multiply(self, device, tensor_factory, verify_tensors):
        """Test elementwise multiplication kernel."""
        @to_gluon(max_jobs=8, verify=False)
        def elementwise_multiply(A, B, C):
            """C = A * B"""
            with torch.no_grad():
                C.copy_(A * B)

        a = torch.randn(512, device=device)
        b = torch.randn(512, device=device)
        c = torch.zeros(512, device=device)

        elementwise_multiply(a, b, c)

        expected = a * b
        verify_tensors(c, expected)

    @pytest.mark.gpu
    def test_elementwise_relu(self, device, tensor_factory, verify_tensors):
        """Test ReLU activation kernel."""
        @to_gluon(max_jobs=8, verify=False)
        def relu_kernel(input, output):
            """output = ReLU(input)"""
            with torch.no_grad():
                output.copy_(torch.maximum(input, torch.zeros_like(input)))

        inp = torch.randn(256, device=device) - 0.5  # Mix of positive and negative
        out = torch.zeros(256, device=device)

        relu_kernel(inp, out)

        expected = torch.maximum(inp, torch.zeros_like(inp))
        verify_tensors(out, expected)

    @pytest.mark.gpu
    def test_elementwise_fused(self, device, tensor_factory, verify_tensors):
        """Test fused elementwise operations."""
        @to_gluon(max_jobs=8, verify=False)
        def fused_elementwise(A, B, C, D):
            """D = (A + B) * C"""
            with torch.no_grad():
                D.copy_((A + B) * C)

        a = torch.randn(512, device=device)
        b = torch.randn(512, device=device)
        c = torch.randn(512, device=device)
        d = torch.zeros(512, device=device)

        fused_elementwise(a, b, c, d)

        expected = (a + b) * c
        verify_tensors(d, expected)


class TestMatmulKernels:
    """GPU tests for matrix multiplication kernels."""

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_matmul_small(self, device, verify_tensors):
        """Test small matrix multiplication."""
        @to_gluon(max_jobs=8, verify=False)
        def matmul_small(A, B, C):
            """Small matmul: C = A @ B"""
            with torch.no_grad():
                C.copy_(A @ B)

        M, N, K = 64, 64, 64
        a = torch.randn(M, K, device=device, dtype=torch.float32)
        b = torch.randn(K, N, device=device, dtype=torch.float32)
        c = torch.zeros(M, N, device=device, dtype=torch.float32)

        matmul_small(a, b, c)

        expected = a @ b
        verify_tensors(c, expected, atol=1e-1, rtol=1e-1)

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_matmul_medium(self, device, verify_tensors):
        """Test medium matrix multiplication."""
        @to_gluon(max_jobs=8, verify=False)
        def matmul_medium(A, B, C):
            """Medium matmul: C = A @ B"""
            with torch.no_grad():
                C.copy_(A @ B)

        M, N, K = 128, 128, 128
        a = torch.randn(M, K, device=device, dtype=torch.float32)
        b = torch.randn(K, N, device=device, dtype=torch.float32)
        c = torch.zeros(M, N, device=device, dtype=torch.float32)

        matmul_medium(a, b, c)

        expected = a @ b
        verify_tensors(c, expected, atol=1e-1, rtol=1e-1)

    @pytest.mark.gpu
    def test_matmul_with_bias(self, device, verify_tensors):
        """Test matrix multiplication with bias addition."""
        @to_gluon(max_jobs=8, verify=False)
        def matmul_bias(A, B, bias, C):
            """C = A @ B + bias"""
            with torch.no_grad():
                C.copy_(A @ B + bias)

        M, N, K = 64, 64, 64
        a = torch.randn(M, K, device=device, dtype=torch.float32)
        b = torch.randn(K, N, device=device, dtype=torch.float32)
        bias = torch.randn(N, device=device, dtype=torch.float32)
        c = torch.zeros(M, N, device=device, dtype=torch.float32)

        matmul_bias(a, b, bias, c)

        expected = a @ b + bias
        verify_tensors(c, expected, atol=1e-1, rtol=1e-1)


class TestReductionKernels:
    """GPU tests for reduction kernels."""

    @pytest.mark.gpu
    def test_sum_reduction(self, device, verify_tensors):
        """Test sum reduction kernel."""
        @to_gluon(max_jobs=8, verify=False)
        def sum_reduction(input, output):
            """output = sum(input)"""
            with torch.no_grad():
                output.copy_(input.sum().unsqueeze(0))

        inp = torch.randn(256, device=device)
        out = torch.zeros(1, device=device)

        sum_reduction(inp, out)

        expected = inp.sum().unsqueeze(0)
        verify_tensors(out, expected, atol=1e-2, rtol=1e-2)

    @pytest.mark.gpu
    def test_mean_reduction(self, device, verify_tensors):
        """Test mean reduction kernel."""
        @to_gluon(max_jobs=8, verify=False)
        def mean_reduction(input, output):
            """output = mean(input)"""
            with torch.no_grad():
                output.copy_(input.mean().unsqueeze(0))

        inp = torch.randn(256, device=device)
        out = torch.zeros(1, device=device)

        mean_reduction(inp, out)

        expected = inp.mean().unsqueeze(0)
        verify_tensors(out, expected, atol=1e-2, rtol=1e-2)

    @pytest.mark.gpu
    def test_row_sum_reduction(self, device, verify_tensors):
        """Test row-wise sum reduction."""
        @to_gluon(max_jobs=8, verify=False)
        def row_sum(input, output):
            """output = sum(input, dim=1)"""
            with torch.no_grad():
                output.copy_(input.sum(dim=1))

        inp = torch.randn(32, 64, device=device)
        out = torch.zeros(32, device=device)

        row_sum(inp, out)

        expected = inp.sum(dim=1)
        verify_tensors(out, expected, atol=1e-2, rtol=1e-2)


class TestCopyAndMemoryKernels:
    """GPU tests for copy and memory operations."""

    @pytest.mark.gpu
    def test_vector_copy(self, device, verify_tensors):
        """Test vector copy kernel."""
        @to_gluon(max_jobs=8, verify=False)
        def vector_copy(src, dst):
            """dst = src"""
            with torch.no_grad():
                dst.copy_(src)

        src = torch.randn(1024, device=device)
        dst = torch.zeros(1024, device=device)

        vector_copy(src, dst)

        verify_tensors(dst, src)

    @pytest.mark.gpu
    def test_matrix_copy(self, device, verify_tensors):
        """Test matrix copy kernel."""
        @to_gluon(max_jobs=8, verify=False)
        def matrix_copy(src, dst):
            """dst = src"""
            with torch.no_grad():
                dst.copy_(src)

        src = torch.randn(128, 256, device=device)
        dst = torch.zeros(128, 256, device=device)

        matrix_copy(src, dst)

        verify_tensors(dst, src)

    @pytest.mark.gpu
    def test_transpose_copy(self, device, verify_tensors):
        """Test transpose copy kernel."""
        @to_gluon(max_jobs=8, verify=False)
        def transpose_copy(src, dst):
            """dst = src.T"""
            with torch.no_grad():
                dst.copy_(src.T)

        src = torch.randn(64, 128, device=device)
        dst = torch.zeros(128, 64, device=device)

        transpose_copy(src, dst)

        expected = src.T
        verify_tensors(dst, expected)


class TestPerformanceBenchmarks:
    """Performance benchmark tests."""

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_elementwise_performance(self, device):
        """Benchmark elementwise kernel performance."""
        @to_gluon(max_jobs=8, verify=False)
        def elementwise_add(A, B, C):
            """C = A + B"""
            with torch.no_grad():
                C.copy_(A + B)

        sizes = [1024, 4096, 16384, 65536]
        results = {}

        for size in sizes:
            a = torch.randn(size, device=device)
            b = torch.randn(size, device=device)
            c = torch.zeros(size, device=device)

            # Warmup
            for _ in range(3):
                elementwise_add(a, b, c)

            # Benchmark
            torch.cuda.synchronize()
            start = time.time()
            iterations = 100
            for _ in range(iterations):
                elementwise_add(a, b, c)
            torch.cuda.synchronize()
            elapsed = time.time() - start

            results[size] = {
                'time_ms': (elapsed / iterations) * 1000,
                'throughput_gb_s': (3 * size * 4) / (elapsed / iterations) / 1e9
            }

        # Verify results are reasonable (should complete in reasonable time)
        for size, metrics in results.items():
            assert metrics['time_ms'] > 0, f"Size {size}: time should be positive"
            assert metrics['throughput_gb_s'] > 0, f"Size {size}: throughput should be positive"

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_matmul_performance(self, device):
        """Benchmark matmul kernel performance."""
        @to_gluon(max_jobs=8, verify=False)
        def matmul_kernel(A, B, C):
            """C = A @ B"""
            with torch.no_grad():
                C.copy_(A @ B)

        sizes = [(64, 64, 64), (128, 128, 128)]
        results = {}

        for M, N, K in sizes:
            a = torch.randn(M, K, device=device)
            b = torch.randn(K, N, device=device)
            c = torch.zeros(M, N, device=device)

            # Warmup
            for _ in range(3):
                matmul_kernel(a, b, c)

            # Benchmark
            torch.cuda.synchronize()
            start = time.time()
            iterations = 50
            for _ in range(iterations):
                matmul_kernel(a, b, c)
            torch.cuda.synchronize()
            elapsed = time.time() - start

            results[(M, N, K)] = {
                'time_ms': (elapsed / iterations) * 1000,
                'flops': (2 * M * N * K) / (elapsed / iterations) / 1e9
            }

        # Verify results are reasonable
        for size, metrics in results.items():
            assert metrics['time_ms'] > 0, f"Size {size}: time should be positive"
            assert metrics['flops'] > 0, f"Size {size}: FLOPS should be positive"


class TestNumericalAccuracy:
    """Tests for numerical accuracy across different operations."""

    @pytest.mark.gpu
    def test_float32_accuracy(self, device):
        """Test float32 numerical accuracy."""
        @to_gluon(max_jobs=8, verify=False)
        def accurate_add(A, B, C):
            """Accurate addition."""
            with torch.no_grad():
                C.copy_(A + B)

        # Test with values that might cause precision issues
        a = torch.tensor([1e-6, 1e6, 1e-8, 1e8], device=device, dtype=torch.float32)
        b = torch.tensor([1e-6, 1e6, 1e-8, 1e8], device=device, dtype=torch.float32)
        c = torch.zeros(4, device=device, dtype=torch.float32)

        accurate_add(a, b, c)

        expected = a + b
        assert torch.allclose(c, expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.gpu
    def test_large_tensor_accuracy(self, device, verify_tensors):
        """Test accuracy with large tensors."""
        @to_gluon(max_jobs=8, verify=False)
        def large_add(A, B, C):
            """Large tensor addition."""
            with torch.no_grad():
                C.copy_(A + B)

        size = 10000
        a = torch.randn(size, device=device)
        b = torch.randn(size, device=device)
        c = torch.zeros(size, device=device)

        large_add(a, b, c)

        expected = a + b
        verify_tensors(c, expected)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.gpu
    def test_single_element(self, device, verify_tensors):
        """Test with single element tensors."""
        @to_gluon(max_jobs=8, verify=False)
        def single_element_add(A, B, C):
            """Single element addition."""
            with torch.no_grad():
                C.copy_(A + B)

        a = torch.tensor([5.0], device=device)
        b = torch.tensor([3.0], device=device)
        c = torch.zeros(1, device=device)

        single_element_add(a, b, c)

        expected = torch.tensor([8.0], device=device)
        verify_tensors(c, expected)

    @pytest.mark.gpu
    def test_zero_tensor(self, device, verify_tensors):
        """Test with zero tensors."""
        @to_gluon(max_jobs=8, verify=False)
        def zero_add(A, B, C):
            """Zero tensor addition."""
            with torch.no_grad():
                C.copy_(A + B)

        a = torch.zeros(256, device=device)
        b = torch.zeros(256, device=device)
        c = torch.ones(256, device=device)

        zero_add(a, b, c)

        expected = torch.zeros(256, device=device)
        verify_tensors(c, expected)

    @pytest.mark.gpu
    def test_negative_values(self, device, verify_tensors):
        """Test with negative values."""
        @to_gluon(max_jobs=8, verify=False)
        def negative_add(A, B, C):
            """Negative value addition."""
            with torch.no_grad():
                C.copy_(A + B)

        a = torch.full((256,), -5.0, device=device)
        b = torch.full((256,), 3.0, device=device)
        c = torch.zeros(256, device=device)

        negative_add(a, b, c)

        expected = torch.full((256,), -2.0, device=device)
        verify_tensors(c, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
