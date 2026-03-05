"""
Strict-mode decorator behavior tests.

In strict mode, @to_gluon only accepts translatable TileLang @T.prim_func kernels.
Plain Python kernels must raise ValueError during translation/compilation.
"""

import pytest
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.decorator import to_gluon


class TestStrictDecoratorRejection:
    """Verify plain Python kernels are rejected in strict to_gluon mode."""

    @pytest.mark.parametrize("shape", [(64,), (1024,), (128, 256)])
    def test_elementwise_plain_python_rejected(self, shape):
        @to_gluon(max_jobs=8, verify=False)
        def elementwise_add(A, B, C):
            with torch.no_grad():
                C.copy_(A + B)

        a = torch.randn(*shape)
        b = torch.randn(*shape)
        c = torch.zeros(*shape)

        with pytest.raises(ValueError, match="No kernel function found with @T\\.prim_func decorator"):
            elementwise_add(a, b, c)

    @pytest.mark.parametrize("m,n,k", [(64, 64, 64), (128, 128, 128), (64, 128, 32)])
    def test_matmul_plain_python_rejected(self, m, n, k):
        @to_gluon(max_jobs=8, verify=False)
        def matmul_kernel(A, B, C):
            with torch.no_grad():
                C.copy_(A @ B)

        a = torch.randn(m, k)
        b = torch.randn(k, n)
        c = torch.zeros(m, n)

        with pytest.raises(ValueError, match="No kernel function found with @T\\.prim_func decorator"):
            matmul_kernel(a, b, c)

    @pytest.mark.parametrize("length", [32, 256, 1024])
    def test_reduction_plain_python_rejected(self, length):
        @to_gluon(max_jobs=8, verify=False)
        def sum_reduction(inp, out):
            with torch.no_grad():
                out.copy_(inp.sum().unsqueeze(0))

        inp = torch.randn(length)
        out = torch.zeros(1)

        with pytest.raises(ValueError, match="No kernel function found with @T\\.prim_func decorator"):
            sum_reduction(inp, out)

    def test_copy_plain_python_rejected(self):
        @to_gluon(max_jobs=8, verify=False)
        def copy_kernel(src, dst):
            with torch.no_grad():
                dst.copy_(src)

        src = torch.randn(128)
        dst = torch.zeros(128)

        with pytest.raises(ValueError, match="No kernel function found with @T\\.prim_func decorator"):
            copy_kernel(src, dst)


class TestStrictSourceAccess:
    """Verify source retrieval is strict for non-TileLang functions."""

    def test_get_gluon_source_rejects_plain_python(self):
        @to_gluon(max_jobs=8, verify=False)
        def plain_kernel(A, B, C):
            with torch.no_grad():
                C.copy_(A + B)

        with pytest.raises(ValueError, match="No kernel function found with @T\\.prim_func decorator"):
            _ = plain_kernel.get_gluon_source()
