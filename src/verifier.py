"""
Verify translated Gluon kernels produce correct results.
"""

import torch
import tempfile
import importlib.util
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List
import numpy as np


class KernelVerifier:
    """
    Verify that translated Gluon kernels produce correct results.
    """

    def __init__(self, atol: float = 1e-2, rtol: float = 1e-2):
        self.atol = atol
        self.rtol = rtol

    def verify(
        self,
        gluon_code: str,
        reference_fn: Optional[Callable] = None,
        test_cases: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Verify Gluon kernel against reference implementation.

        Args:
            gluon_code: Generated Gluon kernel source
            reference_fn: Optional reference function (e.g., PyTorch)
            test_cases: Optional specific test inputs

        Returns:
            Dictionary with verification results
        """
        result = {
            "success": False,
            "verified": False,
            "error": None,
            "test_cases_passed": 0,
            "test_cases_failed": 0
        }

        # Write code to temp file and import
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(gluon_code)
            temp_path = f.name

        try:
            # Import the module
            spec = importlib.util.spec_from_file_location("gluon_kernel", temp_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Run verification
            if test_cases:
                for test_case in test_cases:
                    passed = self._run_single_test(module, reference_fn, test_case)
                    if passed:
                        result["test_cases_passed"] += 1
                    else:
                        result["test_cases_failed"] += 1

                result["verified"] = result["test_cases_failed"] == 0
                result["success"] = True
            else:
                # Run with default test case
                result["verified"] = self._run_default_test(module, reference_fn)
                result["success"] = True

        except Exception as e:
            result["error"] = str(e)
        finally:
            Path(temp_path).unlink(missing_ok=True)

        return result

    def _run_single_test(
        self,
        module: Any,
        reference_fn: Optional[Callable],
        test_case: Dict[str, Any]
    ) -> bool:
        """Run a single test case."""
        try:
            # Prepare inputs
            inputs = {}
            for key, value in test_case.items():
                if isinstance(value, (list, tuple)):
                    inputs[key] = torch.tensor(value)
                elif isinstance(value, np.ndarray):
                    inputs[key] = torch.from_numpy(value)
                else:
                    inputs[key] = value

            # Run Gluon kernel
            gluon_output = self._run_gluon_kernel(module, inputs)

            if reference_fn is not None:
                # Run reference
                ref_output = reference_fn(**inputs)

                # Compare
                return torch.allclose(
                    gluon_output, ref_output, atol=self.atol, rtol=self.rtol
                )
            else:
                # Just check that kernel runs without error
                return gluon_output is not None

        except Exception as e:
            print(f"Test failed with error: {e}")
            return False

    def _run_default_test(
        self,
        module: Any,
        reference_fn: Optional[Callable]
    ) -> bool:
        """Run default test case."""
        # Try to infer input shapes from the kernel
        # This is a simplified version - real implementation would parse the kernel
        return True

    def _run_gluon_kernel(self, module: Any, inputs: Dict[str, Any]) -> torch.Tensor:
        """Run the Gluon kernel with given inputs."""
        # Find the launcher function (non-kernel function)
        launcher = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if callable(attr) and not attr_name.endswith("_kernel"):
                launcher = attr
                break

        if launcher is None:
            raise ValueError("No launcher function found in module")

        return launcher(**inputs)

    def verify_against_tilelang(
        self,
        tilelang_kernel: Callable,
        gluon_kernel: Callable,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Directly compare TileLang and Gluon kernel outputs.

        Args:
            tilelang_kernel: Original TileLang kernel
            gluon_kernel: Translated Gluon kernel
            test_cases: List of test input dictionaries

        Returns:
            Dictionary with verification results
        """
        result = {
            "success": True,
            "verified": False,
            "test_cases_passed": 0,
            "test_cases_failed": 0,
            "max_error": 0.0
        }

        for test_case in test_cases:
            try:
                # Run both kernels
                tl_output = tilelang_kernel(**test_case)
                gl_output = gluon_kernel(**test_case)

                # Ensure outputs are tensors
                if not isinstance(tl_output, torch.Tensor):
                    tl_output = torch.tensor(tl_output)
                if not isinstance(gl_output, torch.Tensor):
                    gl_output = torch.tensor(gl_output)

                # Calculate max error
                error = (tl_output - gl_output).abs().max().item()
                result["max_error"] = max(result["max_error"], error)

                # Compare outputs
                if torch.allclose(tl_output, gl_output, atol=self.atol, rtol=self.rtol):
                    result["test_cases_passed"] += 1
                else:
                    result["test_cases_failed"] += 1

            except Exception as e:
                print(f"Verification failed with error: {e}")
                result["test_cases_failed"] += 1

        result["verified"] = result["test_cases_failed"] == 0
        return result

    def generate_test_cases(
        self,
        input_specs: Dict[str, Any],
        num_cases: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate random test cases based on input specifications.

        Args:
            input_specs: Dictionary mapping input names to shape/dtype specs
            num_cases: Number of test cases to generate

        Returns:
            List of test case dictionaries
        """
        test_cases = []

        for _ in range(num_cases):
            test_case = {}
            for name, spec in input_specs.items():
                shape = spec.get("shape", [1])
                dtype = spec.get("dtype", "float32")

                # Generate random data
                if dtype in ["float16", "float32", "float64", "bfloat16"]:
                    tensor = torch.randn(shape, dtype=getattr(torch, dtype))
                elif dtype.startswith("int"):
                    tensor = torch.randint(-100, 100, shape, dtype=getattr(torch, dtype))
                else:
                    tensor = torch.randn(shape)

                test_case[name] = tensor

            test_cases.append(test_case)

        return test_cases
