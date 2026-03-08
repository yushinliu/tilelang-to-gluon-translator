"""
Pytest fixtures and utilities for GPU kernel testing.

This module provides:
- GPU availability checks
- Pytest fixtures for GPU tensors
- Common test utilities for kernel verification
- TileLang kernel loading and conversion utilities
- Precision verification between TileLang, Gluon, and PyTorch
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys
import importlib.util
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

from tilelang_to_gluon_translator import to_gluon, TileLangToGluonTranslator


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_RTOL = 1e-2
DEFAULT_ATOL = 1e-2
TILELANG_EXAMPLES_PATH = Path("/mnt/d/yuliu/ws/tilelang/examples")


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU hardware"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests that are slow to run"
    )
    config.addinivalue_line(
        "markers", "tilelang: marks tests that use real TileLang kernels"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip GPU tests if no GPU available."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


# =============================================================================
# GPU and Device Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def gpu_available():
    """Check if GPU is available."""
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def device(gpu_available):
    """Get the device to use for testing."""
    return "cuda" if gpu_available else "cpu"


@pytest.fixture(scope="session")
def cuda_device_count(gpu_available):
    """Get the number of CUDA devices available."""
    return torch.cuda.device_count() if gpu_available else 0


@pytest.fixture(scope="session")
def cuda_device_name(gpu_available):
    """Get the name of the first CUDA device."""
    if gpu_available:
        return torch.cuda.get_device_name(0)
    return None


# =============================================================================
# Tensor Factory Fixtures
# =============================================================================

@pytest.fixture
def tensor_factory(device):
    """Factory fixture for creating tensors on the correct device."""
    def _create_tensor(shape, dtype=torch.float32, fill_value=None):
        """Create a tensor on the test device."""
        if fill_value is not None:
            tensor = torch.full(shape, fill_value, dtype=dtype, device=device)
        else:
            tensor = torch.randn(shape, dtype=dtype, device=device)
        return tensor
    return _create_tensor


@pytest.fixture
def random_tensors(device):
    """Fixture providing a function to create random tensors."""
    def _create(*shapes, dtype=torch.float32):
        """Create multiple random tensors."""
        return tuple(torch.randn(s, dtype=dtype, device=device) for s in shapes)
    return _create


@pytest.fixture
def zero_tensors(device):
    """Fixture providing a function to create zero-initialized tensors."""
    def _create(*shapes, dtype=torch.float32):
        """Create multiple zero tensors."""
        return tuple(torch.zeros(s, dtype=dtype, device=device) for s in shapes)
    return _create


@pytest.fixture
def ones_tensors(device):
    """Fixture providing a function to create one-initialized tensors."""
    def _create(*shapes, dtype=torch.float32):
        """Create multiple tensors filled with ones."""
        return tuple(torch.ones(s, dtype=dtype, device=device) for s in shapes)
    return _create


# =============================================================================
# Verification Fixtures
# =============================================================================

@pytest.fixture
def verify_tensors(device):
    """Fixture for tensor verification utilities."""
    def _verify(output, expected, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL):
        """Verify output matches expected values within tolerance."""
        if device == "cuda":
            # Ensure both are on GPU
            if expected.device.type != "cuda":
                expected = expected.cuda()
            if output.device.type != "cuda":
                output = output.cuda()

        assert torch.allclose(output, expected, atol=atol, rtol=rtol), \
            f"Output does not match expected values. " \
            f"Max diff: {(output - expected).abs().max().item()}"

    return _verify


@pytest.fixture
def verify_precision(device):
    """
    Fixture for comprehensive precision verification between multiple implementations.

    Returns a function that compares outputs from TileLang, Gluon, and PyTorch reference.
    """
    def _verify(
        tilelang_output: torch.Tensor,
        gluon_output: torch.Tensor,
        pytorch_output: Optional[torch.Tensor] = None,
        atol: float = DEFAULT_ATOL,
        rtol: float = DEFAULT_RTOL
    ) -> Dict[str, Any]:
        """
        Verify precision between kernel outputs.

        Args:
            tilelang_output: Output from TileLang kernel
            gluon_output: Output from Gluon kernel
            pytorch_output: Optional output from PyTorch reference
            atol: Absolute tolerance
            rtol: Relative tolerance

        Returns:
            Dictionary with verification results
        """
        # Ensure all tensors are on the same device
        if device == "cuda":
            tilelang_output = tilelang_output.cuda() if tilelang_output.device.type != "cuda" else tilelang_output
            gluon_output = gluon_output.cuda() if gluon_output.device.type != "cuda" else gluon_output
            if pytorch_output is not None:
                pytorch_output = pytorch_output.cuda() if pytorch_output.device.type != "cuda" else pytorch_output

        results = {
            "tilelang_vs_gluon": False,
            "tilelang_vs_pytorch": False,
            "gluon_vs_pytorch": False,
            "max_diff_tilelang_gluon": 0.0,
            "max_diff_tilelang_pytorch": 0.0,
            "max_diff_gluon_pytorch": 0.0,
            "success": True
        }

        # Compare TileLang vs Gluon
        diff_tl_gl = (tilelang_output - gluon_output).abs()
        results["max_diff_tilelang_gluon"] = diff_tl_gl.max().item()
        results["tilelang_vs_gluon"] = torch.allclose(
            tilelang_output, gluon_output, atol=atol, rtol=rtol
        )

        # Compare with PyTorch reference if provided
        if pytorch_output is not None:
            diff_tl_pt = (tilelang_output - pytorch_output).abs()
            diff_gl_pt = (gluon_output - pytorch_output).abs()
            results["max_diff_tilelang_pytorch"] = diff_tl_pt.max().item()
            results["max_diff_gluon_pytorch"] = diff_gl_pt.max().item()
            results["tilelang_vs_pytorch"] = torch.allclose(
                tilelang_output, pytorch_output, atol=atol, rtol=rtol
            )
            results["gluon_vs_pytorch"] = torch.allclose(
                gluon_output, pytorch_output, atol=atol, rtol=rtol
            )

        return results

    return _verify


@pytest.fixture
def verify_kernels(device):
    """
    Fixture providing a comprehensive kernel verification function.

    This is the main verification utility that tests can use directly:

    Example:
        def test_gemm(tilelang_gemm_kernel, gluon_gemm_kernel, verify_kernels):
            a, b = prepare_data()
            c_tilelang = tilelang_gemm_kernel(a, b)
            c_gluon = gluon_gemm_kernel(a, b)
            verify_kernels(c_tilelang, c_gluon)
    """
    def _verify(
        tilelang_output: torch.Tensor,
        gluon_output: torch.Tensor,
        pytorch_output: Optional[torch.Tensor] = None,
        atol: float = DEFAULT_ATOL,
        rtol: float = DEFAULT_RTOL,
        assert_on_fail: bool = True
    ) -> Dict[str, Any]:
        """
        Verify kernel outputs match within tolerance.

        Args:
            tilelang_output: Output from TileLang kernel
            gluon_output: Output from Gluon kernel
            pytorch_output: Optional PyTorch reference output
            atol: Absolute tolerance (default: 1e-2)
            rtol: Relative tolerance (default: 1e-2)
            assert_on_fail: Whether to assert on verification failure

        Returns:
            Dictionary with verification results
        """
        # Ensure tensors are on the correct device
        if device == "cuda":
            tilelang_output = tilelang_output.cuda() if not tilelang_output.is_cuda else tilelang_output
            gluon_output = gluon_output.cuda() if not gluon_output.is_cuda else gluon_output
            if pytorch_output is not None:
                pytorch_output = pytorch_output.cuda() if not pytorch_output.is_cuda else pytorch_output

        results = {
            "tilelang_vs_gluon_passed": False,
            "tilelang_vs_pytorch_passed": False,
            "gluon_vs_pytorch_passed": False,
            "max_diff": 0.0,
            "errors": []
        }

        # Compare TileLang vs Gluon
        max_diff_tl_gl = (tilelang_output - gluon_output).abs().max().item()
        results["max_diff"] = max_diff_tl_gl

        if not torch.allclose(tilelang_output, gluon_output, atol=atol, rtol=rtol):
            error_msg = f"TileLang vs Gluon mismatch. Max diff: {max_diff_tl_gl}"
            results["errors"].append(error_msg)
            if assert_on_fail:
                assert False, error_msg
        else:
            results["tilelang_vs_gluon_passed"] = True

        # Compare with PyTorch reference if provided
        if pytorch_output is not None:
            max_diff_tl_pt = (tilelang_output - pytorch_output).abs().max().item()
            max_diff_gl_pt = (gluon_output - pytorch_output).abs().max().item()

            if not torch.allclose(tilelang_output, pytorch_output, atol=atol, rtol=rtol):
                error_msg = f"TileLang vs PyTorch mismatch. Max diff: {max_diff_tl_pt}"
                results["errors"].append(error_msg)
                if assert_on_fail:
                    assert False, error_msg
            else:
                results["tilelang_vs_pytorch_passed"] = True

            if not torch.allclose(gluon_output, pytorch_output, atol=atol, rtol=rtol):
                error_msg = f"Gluon vs PyTorch mismatch. Max diff: {max_diff_gl_pt}"
                results["errors"].append(error_msg)
                if assert_on_fail:
                    assert False, error_msg
            else:
                results["gluon_vs_pytorch_passed"] = True

        return results

    return _verify


# =============================================================================
# Reference Implementation Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def matmul_reference(device):
    """Reference implementation for matrix multiplication."""
    def _matmul(a, b, trans_a=False, trans_b=False):
        """Compute reference matmul result."""
        if trans_a:
            a = a.T
        if trans_b:
            b = b.T
        return a @ b
    return _matmul


@pytest.fixture(scope="session")
def elementwise_reference():
    """Reference implementations for elementwise operations."""
    def _add(a, b):
        return a + b

    def _multiply(a, b):
        return a * b

    def _relu(x):
        return torch.maximum(x, torch.zeros_like(x))

    def _subtract(a, b):
        return a - b

    return {
        "add": _add,
        "multiply": _multiply,
        "relu": _relu,
        "subtract": _subtract,
    }


# =============================================================================
# TileLang Kernel Loading Fixtures
# =============================================================================

@dataclass
class TileLangKernelInfo:
    """Container for TileLang kernel information."""
    name: str
    kernel: Callable
    source_file: Path
    category: str
    ref_program: Optional[Callable] = None


class TileLangKernelLoader:
    """
    Utility class to dynamically load TileLang kernels from examples.
    """

    def __init__(self, examples_path: Path = TILELANG_EXAMPLES_PATH):
        self.examples_path = examples_path
        self.loaded_kernels: Dict[str, TileLangKernelInfo] = {}

    def load_kernel_from_file(self, file_path: Union[str, Path]) -> Optional[TileLangKernelInfo]:
        """
        Load a TileLang kernel from a Python file.

        Args:
            file_path: Path to the Python file containing the kernel

        Returns:
            TileLangKernelInfo if successful, None otherwise
        """
        file_path = Path(file_path)
        if not file_path.exists():
            return None

        try:
            # Load the module
            spec = importlib.util.spec_from_file_location(
                f"tilelang_kernel_{file_path.stem}",
                file_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find kernel functions (decorated with @tilelang.jit or similar)
            kernel = None
            kernel_name = None
            ref_program = None

            for name, obj in inspect.getmembers(module):
                if callable(obj):
                    # Check if it's a tilelang kernel (has jit wrapper attributes)
                    if hasattr(obj, '__wrapped__') or hasattr(obj, 'get_kernel_source'):
                        kernel = obj
                        kernel_name = name
                    # Look for reference program
                    if name == 'ref_program' or name == 'reference':
                        ref_program = obj

            if kernel is not None:
                category = file_path.parent.name
                info = TileLangKernelInfo(
                    name=kernel_name,
                    kernel=kernel,
                    source_file=file_path,
                    category=category,
                    ref_program=ref_program
                )
                self.loaded_kernels[kernel_name] = info
                return info

        except Exception as e:
            print(f"Failed to load kernel from {file_path}: {e}")
            return None

    def load_kernels_from_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "example_*.py"
    ) -> List[TileLangKernelInfo]:
        """
        Load all TileLang kernels from a directory.

        Args:
            directory: Directory to search for kernels
            pattern: File pattern to match

        Returns:
            List of loaded kernel information
        """
        directory = Path(directory)
        kernels = []

        if not directory.exists():
            return kernels

        for file_path in directory.glob(pattern):
            info = self.load_kernel_from_file(file_path)
            if info is not None:
                kernels.append(info)

        return kernels

    def get_kernel(self, name: str) -> Optional[TileLangKernelInfo]:
        """Get a loaded kernel by name."""
        return self.loaded_kernels.get(name)

    def list_available_kernels(self) -> List[str]:
        """List all available kernel names."""
        return list(self.loaded_kernels.keys())


@pytest.fixture(scope="session")
def tilelang_kernel_loader():
    """Fixture providing a TileLang kernel loader."""
    return TileLangKernelLoader(TILELANG_EXAMPLES_PATH)


@pytest.fixture(scope="session")
def available_tilelang_kernels(tilelang_kernel_loader):
    """Fixture providing a list of available TileLang kernels."""
    # Pre-load common kernel categories
    categories = ["gemm", "elementwise", "flash_attention", "convolution"]
    kernels = []

    for category in categories:
        category_path = TILELANG_EXAMPLES_PATH / category
        if category_path.exists():
            kernels.extend(tilelang_kernel_loader.load_kernels_from_directory(category_path))

    return tilelang_kernel_loader


# =============================================================================
# TileLang to Gluon Conversion Fixtures
# =============================================================================

@pytest.fixture
def tilelang_to_gluon_converter(device):
    """
    Fixture providing a utility to convert TileLang kernels to Gluon.

    Returns a function that takes a TileLang kernel function and returns
    a Gluon-equivalent kernel using the @to_gluon decorator.
    """
    def _convert(
        kernel_func: Callable,
        max_jobs: int = 8,
        verify: bool = False,
        atol: float = DEFAULT_ATOL,
        rtol: float = DEFAULT_RTOL
    ) -> Callable:
        """
        Convert a TileLang kernel to Gluon.

        Args:
            kernel_func: The TileLang kernel function to convert
            max_jobs: Maximum parallel compilation jobs
            verify: Whether to verify translation correctness
            atol: Absolute tolerance for verification
            rtol: Relative tolerance for verification

        Returns:
            Gluon kernel function
        """
        # Apply the to_gluon decorator
        gluon_kernel = to_gluon(
            max_jobs=max_jobs,
            verify=verify,
            atol=atol,
            rtol=rtol
        )(kernel_func)

        return gluon_kernel

    return _convert


@pytest.fixture
def gluon_kernel_factory(tilelang_to_gluon_converter):
    """
    Fixture providing a factory for creating Gluon kernels from TileLang source.

    Example:
        def test_with_factory(gluon_kernel_factory):
            tilelang_source = '''...'''
            gluon_kernel = gluon_kernel_factory(tilelang_source)
            result = gluon_kernel(input_tensor)
    """
    def _create(
        source_code: str,
        max_jobs: int = 8,
        atol: float = DEFAULT_ATOL,
        rtol: float = DEFAULT_RTOL
    ) -> Callable:
        """
        Create a Gluon kernel from TileLang source code.

        Args:
            source_code: TileLang source code string
            max_jobs: Maximum parallel compilation jobs
            atol: Absolute tolerance
            rtol: Relative tolerance

        Returns:
            Callable Gluon kernel
        """
        translator = TileLangToGluonTranslator(
            max_jobs=max_jobs,
            verify=False,
            atol=atol,
            rtol=rtol
        )

        # Translate to Gluon
        gluon_code = translator.translate(source_code)

        # Compile and return executable kernel
        # Create a namespace with required imports
        namespace = {'torch': torch}
        try:
            triton = __import__('triton')
            namespace['triton'] = triton
        except ImportError:
            pass

        exec(gluon_code, namespace)

        # Find the launcher function
        for name, obj in namespace.items():
            if callable(obj) and not name.startswith('_') and not name.endswith('_kernel'):
                if name not in ['torch', 'triton']:
                    return obj

        raise ValueError("Could not find launcher function in generated Gluon code")

    return _create


# =============================================================================
# Pre-configured Kernel Fixtures
# =============================================================================

@pytest.fixture
def tilelang_gemm_kernel(available_tilelang_kernels):
    """Fixture providing a TileLang GEMM kernel."""
    kernel_info = available_tilelang_kernels.get_kernel("matmul")
    if kernel_info is None:
        pytest.skip("TileLang GEMM kernel not available")
    return kernel_info.kernel


@pytest.fixture
def gluon_gemm_kernel(tilelang_gemm_kernel, tilelang_to_gluon_converter):
    """Fixture providing a Gluon GEMM kernel converted from TileLang."""
    return tilelang_to_gluon_converter(tilelang_gemm_kernel)


@pytest.fixture
def tilelang_elementwise_kernel(available_tilelang_kernels):
    """Fixture providing a TileLang elementwise kernel."""
    kernel_info = available_tilelang_kernels.get_kernel("elementwise_add")
    if kernel_info is None:
        pytest.skip("TileLang elementwise kernel not available")
    return kernel_info.kernel


@pytest.fixture
def gluon_elementwise_kernel(tilelang_elementwise_kernel, tilelang_to_gluon_converter):
    """Fixture providing a Gluon elementwise kernel converted from TileLang."""
    return tilelang_to_gluon_converter(tilelang_elementwise_kernel)


# =============================================================================
# Benchmark and Performance Fixtures
# =============================================================================

@pytest.fixture
def benchmark_tensors(device):
    """Fixture providing benchmark tensor sizes."""
    return {
        "small": {
            "M": 128,
            "N": 128,
            "K": 64,
        },
        "medium": {
            "M": 512,
            "N": 512,
            "K": 256,
        },
        "large": {
            "M": 1024,
            "N": 1024,
            "K": 512,
        },
    }


@pytest.fixture
def benchmark_config():
    """Fixture providing benchmark configuration."""
    return {
        "warmup_iterations": 10,
        "benchmark_iterations": 100,
        "cuda_graph": True,
    }


# =============================================================================
# Utility Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def clear_cuda_cache():
    """Clear CUDA cache before each test to prevent OOM."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Provide a temporary cache directory for tests."""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@pytest.fixture
def precision_config():
    """Fixture providing configurable precision settings."""
    def _config(rtol: float = DEFAULT_RTOL, atol: float = DEFAULT_ATOL):
        return {"rtol": rtol, "atol": atol}
    return _config


# =============================================================================
# Skip Conditions
# =============================================================================

def skip_if_no_gpu():
    """Skip test if GPU is not available."""
    return pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="GPU not available"
    )


def skip_if_gpu():
    """Skip test if GPU is available (for CPU-only tests)."""
    return pytest.mark.skipif(
        torch.cuda.is_available(),
        reason="GPU is available (CPU-only test)"
    )


def skip_if_tilelang_not_available():
    """Skip test if TileLang is not installed."""
    try:
        import tilelang
        return pytest.mark.skipif(False, reason="")
    except ImportError:
        return pytest.mark.skipif(True, reason="TileLang not installed")


def skip_if_triton_not_available():
    """Skip test if Triton is not installed."""
    try:
        import triton
        return pytest.mark.skipif(False, reason="")
    except ImportError:
        return pytest.mark.skipif(True, reason="Triton not installed")


# =============================================================================
# Helper Functions for Tests
# =============================================================================

def create_test_tensors(
    shape: Union[Tuple[int, ...], List[int]],
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
    num_tensors: int = 2
) -> List[torch.Tensor]:
    """
    Create test tensors with random values.

    Args:
        shape: Shape of the tensors
        dtype: Data type
        device: Device to create tensors on
        num_tensors: Number of tensors to create

    Returns:
        List of random tensors
    """
    return [torch.randn(shape, dtype=dtype, device=device) for _ in range(num_tensors)]


def compare_tensors(
    a: torch.Tensor,
    b: torch.Tensor,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL
) -> Tuple[bool, float]:
    """
    Compare two tensors and return comparison result and max difference.

    Args:
        a: First tensor
        b: Second tensor
        atol: Absolute tolerance
        rtol: Relative tolerance

    Returns:
        Tuple of (is_close, max_difference)
    """
    max_diff = (a - b).abs().max().item()
    is_close = torch.allclose(a, b, atol=atol, rtol=rtol)
    return is_close, max_diff


def run_kernel_with_reference(
    kernel: Callable,
    ref_func: Callable,
    inputs: List[torch.Tensor],
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL
) -> Dict[str, Any]:
    """
    Run a kernel and compare with reference implementation.

    Args:
        kernel: Kernel function to test
        ref_func: Reference function for comparison
        inputs: Input tensors
        atol: Absolute tolerance
        rtol: Relative tolerance

    Returns:
        Dictionary with results and comparison info
    """
    # Run kernel
    kernel_output = kernel(*inputs)

    # Run reference
    ref_output = ref_func(*inputs)

    # Compare
    is_close, max_diff = compare_tensors(kernel_output, ref_output, atol, rtol)

    return {
        "kernel_output": kernel_output,
        "ref_output": ref_output,
        "is_close": is_close,
        "max_diff": max_diff
    }
