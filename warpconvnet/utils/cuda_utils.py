# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from functools import lru_cache
from pathlib import Path

import cupy as cp

logger = logging.getLogger(__name__)


def _get_warpconvnet_csrc_path() -> Path:
    """Get the warpconvnet csrc directory path."""
    # Get the warpconvnet package root by going up from utils directory
    utils_dir = Path(__file__).parent
    warpconvnet_root = utils_dir.parent
    return warpconvnet_root / "csrc"


@lru_cache
def _load_file(kernel_path: str) -> str:
    """Loads kernel source code from file."""
    try:
        with open(kernel_path) as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Kernel file not found: {kernel_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading kernel file {kernel_path}: {e}")
        raise


@lru_cache
def load_kernel(
    kernel_name: str,
    kernel_file: str,
    cache_dir: str | None = None,
    extra_options: tuple[str, ...] = (),
    use_standard_includes: bool = True,
) -> cp.RawKernel:
    """
    Loads and compiles a CuPy RawKernel from a .cu file.

    Args:
        kernel_name: The name of the kernel function in the CUDA code.
        kernel_file: The path to the .cu file containing the kernel. If just a filename
                     (no path separators), automatically looks in warpconvnet/csrc/.
        cache_dir: Directory to cache compiled kernels. Defaults to CuPy's default.
        extra_options: Tuple of extra compiler options (e.g., ('-std=c++17',)).
        use_standard_includes: If True, attempts to add standard CUDA include paths
                               if compilation fails initially.

    Returns:
        A compiled CuPy RawKernel object.
    """
    # If kernel_file is just a filename (no path separators), prepend csrc path
    if os.sep not in kernel_file and "/" not in kernel_file:
        csrc_path = _get_warpconvnet_csrc_path()
        kernel_file = str(csrc_path / kernel_file)
        logger.debug(f"Using centralized CUDA file: {kernel_file}")
    
    kernel_path = os.path.abspath(kernel_file)
    assert os.path.exists(kernel_path), f"Kernel file not found: {kernel_path}"

    options = tuple(extra_options)

    if cache_dir:
        os.environ["CUPY_CACHE_DIR"] = cache_dir
        logger.info(f"Set CUPY_CACHE_DIR to: {cache_dir}")

    try:
        kernel_code = _load_file(kernel_path)
        return cp.RawKernel(kernel_code, kernel_name, options=options)
    except cp.cuda.compiler.CompileException as e:
        logger.warning(
            f"Initial compilation failed for {kernel_name} from {kernel_path} with options {options}: {e}"
        )
        if use_standard_includes:
            # Try adding standard include path if compilation failed
            # Common CUDA include paths (adjust if needed for your system)
            include_paths = [
                "/usr/local/cuda/include",
                # Add other potential paths here if necessary
            ]
            found_path = None
            for p in include_paths:
                if os.path.isdir(p):
                    found_path = p
                    break

            if found_path:
                logger.info(f"Retrying compilation with include path: {found_path}")
                options_with_include = options + (f"-I{found_path}",)
                try:
                    # Must reload the code string if using NVRTC backend for RawKernel
                    kernel_code_reloaded = _load_file(kernel_path)
                    return cp.RawKernel(
                        kernel_code_reloaded,
                        kernel_name,
                        options=options_with_include,
                    )
                except cp.cuda.compiler.CompileException as e2:
                    logger.error(
                        f"Compilation failed even with include path {found_path} for {kernel_name}: {e2}"
                    )
                    raise e2 from e  # Raise the second exception, linked to the first
            else:
                logger.error("Could not find standard CUDA include directory. Compilation failed.")
                raise e  # Re-raise the original exception
        else:
            # If not using standard includes or path not found, re-raise original error
            raise e
    finally:
        if cache_dir:
            # It's generally better practice to let the environment variable persist
            # for the session if caching is desired, but uncomment if you need to reset it.
            # del os.environ['CUPY_CACHE_DIR']
            # logger.info("Unset CUPY_CACHE_DIR")
            pass
