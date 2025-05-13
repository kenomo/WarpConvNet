# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Optional, Tuple

import torch

# import warp as wp # Removed Warp import
import cupy as cp
import math
import os
from jaxtyping import Float, Int
from torch import Tensor

from warpconvnet.utils.argsort import argsort
from warpconvnet.geometry.coords.ops.batch_index import batch_indexed_coordinates
from warpconvnet.utils.cuda_utils import load_kernel


_kernel_dir = os.path.dirname(__file__)
_cuda_kernel_file = os.path.abspath(os.path.join(_kernel_dir, "cuda/morton_code.cu"))

_assign_order_16bit_kernel = load_kernel(
    kernel_file=_cuda_kernel_file, kernel_name="assign_order_discrete_16bit_kernel"
)
_assign_order_20bit_kernel = load_kernel(
    kernel_file=_cuda_kernel_file, kernel_name="assign_order_discrete_20bit_kernel"
)


class POINT_ORDERING(Enum):
    RANDOM = 0
    Z_ORDER = 1


@torch.inference_mode()
def morton_code(
    coords: Int[Tensor, "N 3"] | Int[Tensor, "N 4"],  # noqa: F821
    offsets: Optional[Int[Tensor, "B+1"]] = None,  # noqa: F821
    return_to_morton: bool = True,
    threads_per_block: int = 256,
) -> Tuple[Int[Tensor, "N"], Int[Tensor, "N"]]:  # noqa: F821
    """
    Returns the permutation of the input coordinates that sorts them according to the ordering.

    The coords must be in the range [0, 2^16 - 1] for 16-bit path (batched)
    or effectively [0, 2^20 - 1] for 20-bit path (single batch, after normalization)
    and the result_order will be the z-order number of the point.
    """
    min_coord = coords.min(0).values
    coords_normalized = (coords - min_coord).to(dtype=torch.int32).cuda()

    device = coords_normalized.device

    num_points = len(coords_normalized)
    result_code_cp = cp.empty(num_points, dtype=cp.int64)

    blocks_per_grid = math.ceil(num_points / threads_per_block)

    if coords_normalized.shape[1] == 3 and (offsets is None or len(offsets) < 2):
        # Single batch path (20-bit)
        # Ensure coords_cp is C-contiguous for the kernel
        coords_cp = cp.from_dlpack(coords_normalized.contiguous())

        # Kernel expects: const int* coords_data, int num_points, int64_t* result_order
        _assign_order_20bit_kernel(
            (blocks_per_grid,),
            (threads_per_block,),
            (coords_cp, num_points, result_code_cp),
        )
    else:
        # Multiple batches path (16-bit)
        bcoords = coords_normalized  # Already normalized and int32
        if bcoords.shape[1] == 3 and offsets is not None:
            # batch_indexed_coordinates returns a new tensor, ensure it's on the correct device and type
            bcoords = batch_indexed_coordinates(bcoords, offsets).to(
                device=device, dtype=torch.int32
            )

        # Ensure bcoords_cp is C-contiguous
        bcoords_cp = cp.from_dlpack(bcoords.contiguous())

        # Kernel expects: const int* bcoords_data, int num_points, int64_t* result_order
        _assign_order_16bit_kernel(
            (blocks_per_grid,),
            (threads_per_block,),
            (bcoords_cp, num_points, result_code_cp),
        )

    # Convert result from CuPy array back to PyTorch tensor on the original device
    result_code = torch.from_dlpack(result_code_cp).to(device)

    if return_to_morton:
        to_morton_order = argsort(result_code, backend="torch")
        return result_code, to_morton_order
    return result_code
