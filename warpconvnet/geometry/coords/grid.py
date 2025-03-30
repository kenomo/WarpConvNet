# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, Union
from jaxtyping import Float, Int

import numpy as np
import torch
from torch import Tensor

from warpconvnet.geometry.base.coords import Coords
from warpconvnet.geometry.base.batched import BatchedTensor
from warpconvnet.geometry.coords.ops.grid import create_grid_coordinates


class GridCoords(Coords):
    """Grid coordinates representation with lazy tensor creation.

    This implementation only creates the full coordinate tensor when it's actually
    needed, improving memory efficiency for large grids.
    """

    def __init__(
        self,
        batched_tensor: Tensor,
        offsets: Tensor,
        grid_shape: Tuple[int, int, int],
        bounds: Optional[Tuple[Tensor, Tensor]] = None,
        device: Optional[torch.device] = None,
        is_lazy: bool = False,
        lazy_params: Optional[dict] = None,
    ):
        """Internal initialization method.

        Note: Users should prefer factory methods like from_shape(), from_tensor(),
        or create_regular_grid() instead of using this constructor directly.

        Args:
            batched_tensor: Coordinate tensor (real or dummy)
            offsets: Offset tensor (real or dummy)
            grid_shape: Shape of the grid (H, W, D)
            bounds: Min and max bounds (default: unit cube)
            device: Device to create tensors on
            is_lazy: Whether this is a lazily initialized instance
            lazy_params: Parameters for lazy initialization
        """
        # Set initialization flag first to avoid triggering lazy init
        # during parent class initialization
        self._is_initialized = not is_lazy
        self.grid_shape = grid_shape

        # Set bounds
        if bounds is None:
            # Default to unit cube
            min_bound = torch.zeros(3, device=device or batched_tensor.device)
            max_bound = torch.ones(3, device=device or batched_tensor.device)
        else:
            min_bound = bounds[0].to(device or batched_tensor.device)
            max_bound = bounds[1].to(device or batched_tensor.device)

        # Store lazy initialization parameters if needed
        if is_lazy and lazy_params:
            self._grid_shape = grid_shape
            self._batch_size = lazy_params.get("batch_size", 1)
            self._device = device
            self._flatten = lazy_params.get("flatten", True)
            self._bounds = (min_bound, max_bound)
        else:
            assert (
                batched_tensor.ndim == 5 or batched_tensor.ndim == 2
            ), f"Tensor must have shape (B,H,W,D,3) or (N,3). Got {batched_tensor.shape}"
            assert (
                batched_tensor.shape[-1] == 3
            ), f"Last dimension must be 3. Got {batched_tensor.shape}"
            if batched_tensor.ndim == 5:
                assert batched_tensor.shape[1:4] == grid_shape
            self._flatten = batched_tensor.ndim == 2

        # Call parent's __init__ directly to avoid attribute lookup problems
        BatchedTensor.__init__(self, batched_tensor, offsets)

    @classmethod
    def from_tensor(
        cls,
        tensor: Tensor,
        offsets: Tensor,
        grid_shape: Tuple[int, int, int],
        bounds: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> "GridCoords":
        """Create grid coordinates from an existing tensor.

        Args:
            tensor: Pre-created coordinate tensor of shape (B,H,W,D,3) or (N,3)
            offsets: Offset tensor for batched coordinates
            grid_shape: Shape of the grid (H, W, D)
            bounds: Optional min/max bounds

        Returns:
            GridCoords: Grid coordinates with eager initialization
        """
        return cls(
            batched_tensor=tensor,
            offsets=offsets,
            grid_shape=grid_shape,
            bounds=bounds,
            device=tensor.device,
            is_lazy=False,
        )

    @classmethod
    def from_shape(
        cls,
        grid_shape: Tuple[int, int, int],
        bounds: Optional[Tuple[Tensor, Tensor]] = None,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        flatten: bool = True,
    ) -> "GridCoords":
        """Create grid coordinates lazily from a shape.

        Coordinates will only be created when actually needed.

        Args:
            grid_shape: Grid resolution (H, W, D)
            bounds: Min and max bounds (default: unit cube)
            batch_size: Number of batches
            device: Device to create tensors on
            flatten: Whether to flatten the coordinates

        Returns:
            GridCoords: Lazily initialized grid coordinates
        """
        # Create minimal dummy tensors
        dummy_tensor = torch.zeros((1, 3), device=device)
        # Create the offset using the batch size and grid shape
        num_elements = np.prod(grid_shape)
        offsets = torch.tensor(
            [i * num_elements for i in range(batch_size + 1)], dtype=torch.long, device="cpu"
        )

        lazy_params = {
            "batch_size": batch_size,
            "flatten": flatten,
        }

        return cls(
            batched_tensor=dummy_tensor,
            offsets=offsets,
            grid_shape=grid_shape,
            bounds=bounds,
            device=device,
            is_lazy=True,
            lazy_params=lazy_params,
        )

    def _ensure_initialized(self):
        """Ensure coordinates are initialized."""
        if not self._is_initialized:
            # Create the actual coordinate tensor
            coords, offsets = create_grid_coordinates(
                self._grid_shape,
                self._bounds,
                self._batch_size,
                self._device,
                self._flatten,
            )

            # Replace dummy tensors
            # Need to use direct attribute access on parent class
            # to avoid triggering our custom __setattr__
            object.__setattr__(self, "batched_tensor", coords)
            object.__setattr__(self, "offsets", offsets)
            self._is_initialized = True

    @property
    def is_initialized(self):
        return hasattr(self, "_is_initialized") and self._is_initialized

    # Override __getattribute__ to intercept attribute access
    def __getattribute__(self, name):
        # First get the _is_initialized flag (if it exists)
        try:
            is_initialized = object.__getattribute__(self, "_is_initialized")
        except AttributeError:
            # During initialization, this attribute might not exist yet
            is_initialized = True

        # If accessing tensor attributes and not initialized, ensure initialization
        if not is_initialized and name in ("batched_tensor"):
            # Call _ensure_initialized through object.__getattribute__
            # to avoid recursion
            object.__getattribute__(self, "_ensure_initialized")()

        # Use default attribute lookup
        return object.__getattribute__(self, name)

    # Override other key methods
    def __getitem__(self, idx):
        # __getitem__ uses offsets, which will trigger lazy init via __getattribute__
        return super().__getitem__(idx)

    def to(self, device=None, dtype=None):
        """Handle device transfers while preserving lazy status."""
        if not self.is_initialized:
            # Create a new lazy GridCoords with updated device
            new_device = device if device is not None else self._device
            lazy_params = {
                "batch_size": self._batch_size,
                "flatten": self._flatten,
            }
            return self.__class__(
                batched_tensor=torch.zeros((1, 3), device=new_device),
                offsets=torch.tensor([0, 1], dtype=torch.long, device="cpu"),
                grid_shape=self._grid_shape,
                bounds=self._bounds,
                device=new_device,
                is_lazy=True,
                lazy_params=lazy_params,
            )

        # Otherwise use standard implementation
        return super().to(device, dtype)

    def check(self):
        """Override check to allow lazy initialization."""
        if not self.is_initialized:
            # Skip check for uninitialized tensors
            return
        super().check()

    @property
    def batch_size(self):
        """Override batch_size to avoid initialization."""
        if not self.is_initialized:
            return self._batch_size
        return super().batch_size

    @property
    def device(self):
        """Override device to avoid initialization."""
        if not self.is_initialized:
            return self._device
        return super().device

    @property
    def shape(self):
        """Override shape to avoid initialization."""
        if not self.is_initialized:
            # Return expected shape without initializing tensor
            H, W, D = self._grid_shape
            if self._flatten:
                # If flattened, shape is (N, 3) where N is batch_size * H * W * D
                return (self._batch_size * H * W * D, 3)
            else:
                # If not flattened, shape would depend on batch size
                return (self._batch_size, H, W, D, 3)
        return super().shape

    def half(self):
        self._ensure_initialized()
        return GridCoords.from_tensor(
            self.batched_tensor.half(), self.offsets, self.grid_shape, self._bounds
        )

    def float(self):
        self._ensure_initialized()
        return GridCoords.from_tensor(
            self.batched_tensor.float(), self.offsets, self.grid_shape, self._bounds
        )

    def double(self):
        self._ensure_initialized()
        return GridCoords.from_tensor(
            self.batched_tensor.double(), self.offsets, self.grid_shape, self._bounds
        )

    def numel(self):
        """Override numel to avoid initialization."""
        if not self.is_initialized:
            # Calculate expected number of elements
            H, W, D = self._grid_shape
            return self._batch_size * H * W * D * 3
        return super().numel()

    def __len__(self):
        """Override len to avoid initialization."""
        if not self.is_initialized:
            # Calculate expected length
            H, W, D = self._grid_shape
            return self._batch_size * H * W * D
        return super().__len__()

    # Methods specific to GridCoords that need to be preserved
    def get_spatial_indices(
        self, flat_indices: Int[Tensor, "M"]  # noqa: F821
    ) -> Tuple[Int[Tensor, "M"], Int[Tensor, "M"], Int[Tensor, "M"]]:  # noqa: F821
        """Convert flattened indices to 3D spatial indices."""
        H, W, D = self.grid_shape

        # Calculate indices for each dimension
        h_indices = flat_indices // (W * D)
        w_indices = (flat_indices % (W * D)) // D
        d_indices = flat_indices % D

        return h_indices, w_indices, d_indices

    def get_flattened_indices(
        self,
        h_indices: Int[Tensor, "M"],  # noqa: F821
        w_indices: Int[Tensor, "M"],  # noqa: F821
        d_indices: Int[Tensor, "M"],  # noqa: F821
    ) -> Int[Tensor, "M"]:  # noqa: F821
        """Convert 3D spatial indices to flattened indices."""
        H, W, D = self.grid_shape

        return h_indices * (W * D) + w_indices * D + d_indices

    # Prevent unwanted initialization
    def __repr__(self):
        if not self.is_initialized:
            return f"{self.__class__.__name__}(grid_shape={self.grid_shape}, lazy=True, device={self._device})"
        return super().__repr__()

    def __str__(self):
        if not self.is_initialized:
            return f"{self.__class__.__name__}(grid_shape={self.grid_shape}, lazy=True)"
        return super().__str__()
