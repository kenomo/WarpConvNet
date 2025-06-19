# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pickle
import threading
import time
import atexit
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging

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


class BenchmarkCache:
    """
    Manages saving and loading of benchmark results to/from disk.
    Only rank 0 process saves to avoid conflicts in distributed training.
    """

    def __init__(self, cache_dir: str = "~/.cache/warpconvnet"):
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_file = self.cache_dir / "benchmark_cache.pkl"
        self.lock = threading.Lock()

        # Periodic save settings
        self.save_interval = 300.0  # seconds
        self.last_save_time = 0.0
        self.pending_changes = False

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Register exit handler for final save
        if _is_rank_zero():
            atexit.register(self._save_on_exit)

    def load_cache(self) -> Tuple[Dict, Dict]:
        """
        Load benchmark results from cache file.
        Returns (forward_results, backward_results) dictionaries.
        """
        forward_results = {}
        backward_results = {}

        if not self.cache_file.exists():
            logger.debug(f"No benchmark cache file found at {self.cache_file}")
            return forward_results, backward_results

        try:
            with open(self.cache_file, "rb") as f:
                cache_data = pickle.load(f)

            # Validate cache structure
            if isinstance(cache_data, dict):
                forward_results = cache_data.get("forward_results", {})
                backward_results = cache_data.get("backward_results", {})

                logger.info(
                    f"Loaded benchmark cache: {len(forward_results)} forward, "
                    f"{len(backward_results)} backward configurations"
                )
            else:
                logger.warning("Invalid cache file format, starting with empty cache")

        except Exception as e:
            logger.warning(f"Failed to load benchmark cache: {e}. Starting with empty cache.")

        return forward_results, backward_results

    def save_cache(
        self, forward_results: Dict, backward_results: Dict, force: bool = False
    ) -> None:
        """
        Save benchmark results to cache file.
        Only saves if enough time has passed since last save (unless force=True).
        Only rank 0 process performs the actual save.
        """
        if not _is_rank_zero():
            return

        with self.lock:
            current_time = time.time()

            # Mark that we have pending changes
            self.pending_changes = True

            # Check if we should save now
            if not force and (current_time - self.last_save_time) < self.save_interval:
                return

            try:
                # Prepare cache data
                cache_data = {
                    "forward_results": forward_results,
                    "backward_results": backward_results,
                    "timestamp": current_time,
                    "version": "1.0",  # For future compatibility
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
                    f"Saved benchmark cache: {len(forward_results)} forward, "
                    f"{len(backward_results)} backward configurations"
                )

            except Exception as e:
                logger.warning(f"Failed to save benchmark cache: {e}")

    def mark_dirty(self) -> None:
        """Mark that cache has pending changes that should be saved."""
        self.pending_changes = True

    def _save_on_exit(self) -> None:
        """Save cache on program exit if there are pending changes."""
        if self.pending_changes:
            # Import here to avoid circular imports
            from warpconvnet.nn.functional.sparse_conv import (
                _BENCHMARK_FORWARD_RESULTS,
                _BENCHMARK_BACKWARD_RESULTS,
            )

            self.save_cache(_BENCHMARK_FORWARD_RESULTS, _BENCHMARK_BACKWARD_RESULTS, force=True)


# Global cache instance
_benchmark_cache: Optional[BenchmarkCache] = None


def get_benchmark_cache() -> BenchmarkCache:
    """Get the global benchmark cache instance."""
    global _benchmark_cache
    if _benchmark_cache is None:
        _benchmark_cache = BenchmarkCache()
    return _benchmark_cache


def load_benchmark_cache() -> Tuple[Dict, Dict]:
    """Load benchmark cache and return (forward_results, backward_results)."""
    cache = get_benchmark_cache()
    return cache.load_cache()


def save_benchmark_cache(
    forward_results: Dict, backward_results: Dict, force: bool = False
) -> None:
    """Save benchmark cache."""
    cache = get_benchmark_cache()
    cache.save_cache(forward_results, backward_results, force=force)


def mark_benchmark_cache_dirty() -> None:
    """Mark that benchmark cache has pending changes."""
    cache = get_benchmark_cache()
    cache.mark_dirty()
