# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import math
import os
import pickle
import threading
import time
import atexit
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Sequence
from dataclasses import dataclass

import torch

from warpconvnet.constants import (
    WARPCONVNET_BENCHMARK_CACHE_DIR,
    WARPCONVNET_BENCHMARK_CACHE_VERSION,
)
from warpconvnet.utils.logger import get_logger

logger = get_logger(__name__)


def _get_current_rank() -> int:
    """Get current process rank for distributed training."""
    # Check common distributed training environment variables
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    elif "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        return int(os.environ["SLURM_PROCID"])
    else:
        # Not in distributed mode or rank 0
        return 0


def _is_rank_zero() -> bool:
    """Check if current process is rank 0."""
    return _get_current_rank() == 0


def _int_sequence_hash(arr: Sequence[int]) -> int:  # noqa: F821
    """Hash a sequence of ints into a single 32‑bit value."""
    x = hash(arr[0])
    for i in range(1, len(arr)):
        x = (x * 31 + hash(arr[i])) & 0xFFFFFFFF  # Keep it within 32-bit range
    return x


_SPARSE_CONV_CONFIG_DTYPE_TO_INT = {
    torch.bfloat16: 0,
    torch.float16: 1,
    torch.float32: 2,
    torch.float64: 3,
}


@dataclass
class SpatiallySparseConvConfig:
    log_num_in_coords: int
    log_num_out_coords: int
    in_channels: int
    out_channels: int
    kernel_volume: int
    in_dtype: torch.dtype
    # explicit_matmul_batch_size: Optional[int] = None # TODO: Add if supporting batched explicit

    def __init__(
        self,
        num_in_coords: int,
        num_out_coords: int,
        in_channels: int,
        out_channels: int,
        kernel_volume: int,
        in_dtype: torch.dtype,
        # explicit_matmul_batch_size: Optional[int] = None, # TODO
    ):
        self.log_num_in_coords = math.ceil(math.log2(num_in_coords)) if num_in_coords > 0 else 0
        self.log_num_out_coords = math.ceil(math.log2(num_out_coords)) if num_out_coords > 0 else 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_volume = kernel_volume
        assert in_dtype in _SPARSE_CONV_CONFIG_DTYPE_TO_INT, f"Unsupported in_dtype: {in_dtype}"
        self.in_dtype = in_dtype
        # self.explicit_matmul_batch_size = explicit_matmul_batch_size # TODO

    def __hash__(self):
        return _int_sequence_hash(
            [
                # self.log_num_in_coords,
                # self.log_num_out_coords,
                self.in_channels,
                self.out_channels,
                self.kernel_volume,
                _SPARSE_CONV_CONFIG_DTYPE_TO_INT[self.in_dtype],
            ]
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SpatiallySparseConvConfig):
            return False
        return (
            # self.log_num_in_coords == other.log_num_in_coords
            # and self.log_num_out_coords == other.log_num_out_coords and
            self.in_channels == other.in_channels
            and self.out_channels == other.out_channels
            and self.kernel_volume == other.kernel_volume
            and self.in_dtype == other.in_dtype
        )


class BenchmarkCache:
    """
    Manages saving and loading of benchmark results to/from disk.
    Only rank 0 process saves to avoid conflicts in distributed training.
    Uses a background thread for periodic saving to avoid blocking the main computation.

    Version 2.0 supports multiple benchmark result types:
    - sparse_conv_forward_results
    - sparse_conv_backward_results
    - sparse_conv_depthwise_forward_results
    - sparse_conv_depthwise_backward_results
    """

    def __init__(self, cache_dir: str = WARPCONVNET_BENCHMARK_CACHE_DIR):
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_file = self.cache_dir / "benchmark_cache.pkl"
        self.lock = threading.Lock()

        # Periodic save settings
        self.save_interval = 60.0  # seconds - reduced from 300 to 60 for more frequent saves
        self.last_save_time = 0.0
        self.pending_changes = False
        self._shutdown_requested = False

        # Background thread for saving
        self._save_thread = None
        self._save_condition = threading.Condition(self.lock)

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Start background save thread only for rank 0
        if _is_rank_zero():
            self._start_background_saver()
            atexit.register(self._save_on_exit)

    def _start_background_saver(self) -> None:
        """Start the background thread for periodic saving."""
        if self._save_thread is not None:
            return

        self._save_thread = threading.Thread(
            target=self._background_save_worker, name="BenchmarkCacheSaver", daemon=True
        )
        self._save_thread.start()
        logger.debug("Started background benchmark cache saver thread")

    def _background_save_worker(self) -> None:
        """Background worker thread that periodically saves the cache."""
        while not self._shutdown_requested:
            with self._save_condition:
                # Wait for either pending changes or timeout
                self._save_condition.wait(timeout=self.save_interval)

                if self._shutdown_requested:
                    break

                # Check if we need to save
                current_time = time.time()
                if (
                    self.pending_changes
                    and (current_time - self.last_save_time) >= self.save_interval
                ):
                    self._do_save()

    def _do_save(self) -> None:
        """Internal method to perform the actual save (assumes lock is held)."""
        if not self.pending_changes:
            return

        try:
            # Import here to avoid circular imports
            from warpconvnet.nn.functional.sparse_conv import (
                _BENCHMARK_FORWARD_RESULTS,
                _BENCHMARK_BACKWARD_RESULTS,
            )

            # Try to import depthwise results, but don't fail if not available
            try:
                from warpconvnet.nn.functional.sparse_conv_depth import (
                    _BENCHMARK_DEPTHWISE_FORWARD_RESULTS,
                    _BENCHMARK_DEPTHWISE_BACKWARD_RESULTS,
                )
            except ImportError:
                _BENCHMARK_DEPTHWISE_FORWARD_RESULTS = {}
                _BENCHMARK_DEPTHWISE_BACKWARD_RESULTS = {}

            current_time = time.time()

            # Prepare cache data in version 2.0 format
            cache_data = {
                "sparse_conv_forward_results": _BENCHMARK_FORWARD_RESULTS,
                "sparse_conv_backward_results": _BENCHMARK_BACKWARD_RESULTS,
                "sparse_conv_depthwise_forward_results": _BENCHMARK_DEPTHWISE_FORWARD_RESULTS,
                "sparse_conv_depthwise_backward_results": _BENCHMARK_DEPTHWISE_BACKWARD_RESULTS,
                "timestamp": current_time,
                "version": WARPCONVNET_BENCHMARK_CACHE_VERSION,
            }

            # Atomic write: write to temp file then rename
            temp_file = self.cache_file.with_suffix(".tmp")
            with open(temp_file, "wb") as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Atomic move
            temp_file.replace(self.cache_file)

            self.last_save_time = current_time
            self.pending_changes = False

            logger.debug(
                f"Background saved benchmark cache: {len(_BENCHMARK_FORWARD_RESULTS)} forward, "
                f"{len(_BENCHMARK_BACKWARD_RESULTS)} backward, "
                f"{len(_BENCHMARK_DEPTHWISE_FORWARD_RESULTS)} depthwise forward, "
                f"{len(_BENCHMARK_DEPTHWISE_BACKWARD_RESULTS)} depthwise backward configurations"
            )

        except Exception as e:
            logger.warning(f"Failed to save benchmark cache in background: {e}")

    def load_cache(self) -> Dict[str, Dict]:
        """
        Load benchmark results from cache file.

        If the loaded version is less than the latest version, the cache will be reset.

        Returns a dictionary with all cached benchmark results.
        """
        # Default empty results for all supported cache types
        default_results = {
            "sparse_conv_forward_results": {},
            "sparse_conv_backward_results": {},
            "sparse_conv_depthwise_forward_results": {},
            "sparse_conv_depthwise_backward_results": {},
        }

        if not self.cache_file.exists():
            logger.debug(f"No benchmark cache file found at {self.cache_file}")
            return default_results

        try:
            with open(self.cache_file, "rb") as f:
                cache_data = pickle.load(f)

            if not isinstance(cache_data, dict):
                logger.warning("Invalid cache file format, starting with empty cache")
                return default_results

            # Determine cache version and handle accordingly
            version = cache_data.get("version", "1.0")
            # Remove any string before and after major.minor
            version = re.sub(r"[^0-9.]", "", str(version))
            major_version, minor_version = version.split(".")

            if int(major_version) == 3:
                # Version 3.0 format - direct mapping
                result = {
                    "sparse_conv_forward_results": cache_data.get(
                        "sparse_conv_forward_results", {}
                    ),
                    "sparse_conv_backward_results": cache_data.get(
                        "sparse_conv_backward_results", {}
                    ),
                    "sparse_conv_depthwise_forward_results": cache_data.get(
                        "sparse_conv_depthwise_forward_results", {}
                    ),
                    "sparse_conv_depthwise_backward_results": cache_data.get(
                        "sparse_conv_depthwise_backward_results", {}
                    ),
                }

                total_configs = sum(len(results) for results in result.values())
                logger.info(f"Loaded benchmark cache v3.0: {total_configs} total configurations")

            else:
                logger.warning(
                    f"Loaded benchmark cache v{version}, but expected v3.0. Resetting cache."
                )
                return default_results

            return result

        except Exception as e:
            logger.warning(f"Failed to load benchmark cache: {e}. Starting with empty cache.")
            return default_results

    def save_cache(self, cache_results: Dict[str, Dict], force: bool = False) -> None:
        """
        Save benchmark results to cache file in version 2.0 format.

        Args:
            cache_results: Dictionary mapping cache types to their results. Current supported types are:
                - sparse_conv_forward_results
                - sparse_conv_backward_results
                - sparse_conv_depthwise_forward_results
                - sparse_conv_depthwise_backward_results
            force: If True, save immediately. If False, schedule for background save.
        """
        if not _is_rank_zero():
            return

        if not force:
            # For non-forced saves, just mark dirty and let background thread handle it
            self.mark_dirty()
            return

        with self.lock:
            current_time = time.time()

            try:
                # Prepare cache data in the latest version
                cache_data = {
                    "sparse_conv_forward_results": cache_results.get(
                        "sparse_conv_forward_results", {}
                    ),
                    "sparse_conv_backward_results": cache_results.get(
                        "sparse_conv_backward_results", {}
                    ),
                    "sparse_conv_depthwise_forward_results": cache_results.get(
                        "sparse_conv_depthwise_forward_results", {}
                    ),
                    "sparse_conv_depthwise_backward_results": cache_results.get(
                        "sparse_conv_depthwise_backward_results", {}
                    ),
                    "timestamp": current_time,
                    "version": WARPCONVNET_BENCHMARK_CACHE_VERSION,
                }

                # Atomic write: write to temp file then rename
                temp_file = self.cache_file.with_suffix(".tmp")
                with open(temp_file, "wb") as f:
                    pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

                # Atomic move
                temp_file.replace(self.cache_file)

                self.last_save_time = current_time
                self.pending_changes = False

                total_configs = sum(len(results) for results in cache_results.values())
                logger.debug(
                    f"Force saved benchmark cache v{WARPCONVNET_BENCHMARK_CACHE_VERSION}: {total_configs} total configurations"
                )

            except Exception as e:
                logger.warning(
                    f"Failed to force save benchmark cache v{WARPCONVNET_BENCHMARK_CACHE_VERSION}: {e}"
                )

    def mark_dirty(self) -> None:
        """Mark that cache has pending changes that should be saved."""
        if not _is_rank_zero():
            return

        with self._save_condition:
            self.pending_changes = True
            # Notify the background thread that there are changes
            self._save_condition.notify()

    def _save_on_exit(self) -> None:
        """Save cache on program exit if there are pending changes."""
        # Signal shutdown to background thread
        self._shutdown_requested = True

        # Wake up the background thread
        with self._save_condition:
            self._save_condition.notify()

        # Wait for background thread to finish (with timeout)
        if self._save_thread is not None:
            self._save_thread.join(timeout=5.0)

        # Perform final save if there are still pending changes
        if self.pending_changes:
            from warpconvnet.nn.functional.sparse_conv import (
                _BENCHMARK_FORWARD_RESULTS,
                _BENCHMARK_BACKWARD_RESULTS,
            )

            from warpconvnet.nn.functional.sparse_conv_depth import (
                _BENCHMARK_DEPTHWISE_FORWARD_RESULTS,
                _BENCHMARK_DEPTHWISE_BACKWARD_RESULTS,
            )

            self.save_cache(
                {
                    "sparse_conv_forward_results": _BENCHMARK_FORWARD_RESULTS,
                    "sparse_conv_backward_results": _BENCHMARK_BACKWARD_RESULTS,
                    "sparse_conv_depthwise_forward_results": _BENCHMARK_DEPTHWISE_FORWARD_RESULTS,
                    "sparse_conv_depthwise_backward_results": _BENCHMARK_DEPTHWISE_BACKWARD_RESULTS,
                },
                force=True,
            )


# Global cache instance
_benchmark_cache: Optional[BenchmarkCache] = None


def get_benchmark_cache() -> BenchmarkCache:
    """Get the global benchmark cache instance."""
    global _benchmark_cache
    if _benchmark_cache is None:
        _benchmark_cache = BenchmarkCache()
    return _benchmark_cache


def load_sparse_conv_benchmark_cache() -> Tuple[Dict, Dict]:
    """
    Load benchmark cache and return (forward_results, backward_results) for backward compatibility.
    For full v2.0 results, use get_benchmark_cache().load_cache() directly.
    """
    cache = get_benchmark_cache()
    cache_results = cache.load_cache()

    # Return legacy format for backward compatibility
    forward_results = cache_results.get("sparse_conv_forward_results", {})
    backward_results = cache_results.get("sparse_conv_backward_results", {})

    return forward_results, backward_results


def load_dict_benchmark_cache() -> Dict[str, Dict]:
    """
    Load benchmark cache in version 2.0 format.
    Returns dictionary with all cached benchmark result types.
    """
    cache = get_benchmark_cache()
    return cache.load_cache()


def save_sparse_conv_benchmark_cache(
    forward_results: Dict, backward_results: Dict, force: bool = False
) -> None:
    """
    Save benchmark cache.

    Args:
        forward_results: Forward benchmark results
        backward_results: Backward benchmark results
        force: If True, save immediately. If False, schedule for background save.
    """
    cache = get_benchmark_cache()
    cache.save_cache(
        {
            "sparse_conv_forward_results": forward_results,
            "sparse_conv_backward_results": backward_results,
        },
        force=force,
    )


def save_dict_benchmark_cache(cache_results: Dict[str, Dict], force: bool = False) -> None:
    """
    Save benchmark cache in version 2.0 format.

    Args:
        cache_results: Dictionary mapping cache types to their results
        force: If True, save immediately. If False, schedule for background save.
    """
    cache = get_benchmark_cache()
    cache.save_cache(cache_results, force=force)


def mark_benchmark_cache_dirty() -> None:
    """Mark that benchmark cache has pending changes."""
    cache = get_benchmark_cache()
    cache.mark_dirty()
