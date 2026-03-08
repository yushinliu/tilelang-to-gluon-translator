"""
Gluon version checking utilities.
"""

import warnings
from typing import Optional, Tuple

# Expected Gluon version for this translator
EXPECTED_GLUON_VERSION = "3.6.0"


def get_gluon_version() -> Optional[str]:
    """
    Get the installed Gluon version.

    Returns:
        Version string if Gluon is installed, None otherwise
    """
    try:
        import triton
        return triton.__version__
    except ImportError:
        return None


def check_gluon_version() -> Tuple[bool, str, str]:
    """
    Check if the installed Gluon version matches the expected version.

    Returns:
        Tuple of (is_compatible, installed_version, expected_version)
    """
    installed_version = get_gluon_version()

    if installed_version is None:
        warnings.warn(
            "Triton/Gluon is not installed. The generated code may not work correctly.",
            RuntimeWarning
        )
        return False, "not_installed", EXPECTED_GLUON_VERSION

    # Check major.minor version match
    try:
        installed_parts = installed_version.split('.')
        expected_parts = EXPECTED_GLUON_VERSION.split('.')

        # Compare major and minor version
        if len(installed_parts) >= 2 and len(expected_parts) >= 2:
            if installed_parts[0] != expected_parts[0] or installed_parts[1] != expected_parts[1]:
                warnings.warn(
                    f"Gluon version mismatch: installed {installed_version}, expected {EXPECTED_GLUON_VERSION}. "
                    f"The generated code is optimized for Gluon {EXPECTED_GLUON_VERSION} and may not work correctly "
                    f"with other versions.",
                    RuntimeWarning
                )
                return False, installed_version, EXPECTED_GLUON_VERSION
    except Exception:
        pass

    return True, installed_version, EXPECTED_GLUON_VERSION


def log_version_info():
    """Log version information for debugging."""
    is_compatible, installed, expected = check_gluon_version()

    if installed != "not_installed":
        print(f"[TileLang-to-Gluon] Gluon version: {installed} (expected: {expected})")
        if not is_compatible:
            print(f"[TileLang-to-Gluon] Warning: Version mismatch detected!")
    else:
        print(f"[TileLang-to-Gluon] Warning: Gluon not installed!")

    return is_compatible
