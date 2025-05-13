# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import logging
import math
import os
from enum import Enum
from pathlib import Path
from typing import Literal, Optional, Sequence, Tuple, Dict

import numpy as np
import torch

import cupy as cp
from jaxtyping import Int
from torch import Tensor

from warpconvnet.geometry.coords.search.torch_hashmap import TorchHashTable, HashMethod
from warpconvnet.geometry.coords.search.search_results import IntSearchResult
from warpconvnet.utils.cuda_utils import load_kernel
from warpconvnet.utils.ntuple import ntuple

logger = logging.getLogger(__name__)

# Path to the new CUDA kernel file
KERNEL_FILE = Path(__file__).parent / "cuda" / "discrete_kernels.cu"


def _get_kernel_map_offset_kernel(hash_method: HashMethod) -> cp.RawKernel:
    suffix = hash_method.kernel_suffix()
    return load_kernel(f"kernel_map_offset_{suffix}", str(KERNEL_FILE))


def _get_map_results_kernel() -> cp.RawKernel:
    return load_kernel("map_found_indices_to_maps_cuda", str(KERNEL_FILE))


def _get_kernel_map_size_4d_kernel(hash_method: HashMethod) -> cp.RawKernel:
    suffix = hash_method.kernel_suffix()
    return load_kernel(f"kernel_map_size_4d_{suffix}", str(KERNEL_FILE))


@torch.no_grad()
def kernel_offsets_from_size(
    kernel_size: Tuple[int, ...],
    kernel_dilation: Tuple[int, ...],
    center_offset: Optional[Tuple[int, ...]] = None,
    device: Optional[torch.device] = None,  # Added device argument
) -> Int[Tensor, "K D+1"]:
    """
    Generate the kernel offsets for the spatially sparse convolution.
    Supports arbitrary number of spatial dimensions.
    Returns a PyTorch Tensor.
    """
    assert len(kernel_size) == len(kernel_dilation)
    num_spatial_dims = len(kernel_size)

    # Create meshgrid for arbitrary dimensions
    ranges = [torch.arange(size, dtype=torch.int32, device="cpu") for size in kernel_size]
    grids = torch.meshgrid(*ranges, indexing="ij")
    flattened_grids = [grid.flatten() for grid in grids]

    if center_offset is None:
        # center odd-sized kernels and 0 for even-sized kernels
        center_offset = [(s - 1) // 2 if s % 2 == 1 else 0 for s in kernel_size]
    assert len(center_offset) == num_spatial_dims

    # Create offsets for each dimension
    offsets = [
        (grid - center_offset[i]) * kernel_dilation[i] for i, grid in enumerate(flattened_grids)
    ]

    # Add batch dimension (zeros)
    offsets = [torch.zeros_like(offsets[0])] + offsets

    return torch.stack(offsets, dim=1).contiguous().to(device)


@torch.no_grad()
def _kernel_map_search_to_result(
    found_in_coord_index: Int[Tensor, "K M"],
    return_type: Literal["indices", "offsets"] = "offsets",
    threads_per_block: int = 256,
) -> Int[Tensor, "K M"] | IntSearchResult:
    """Processes the raw found_in_coord_index tensor into the desired format."""
    # assert found_in_coord_index_wp.shape[0] == kernel_offsets.shape[0]
    # assert found_in_coord_index_wp.shape[1] == batched_query_coords.shape[0]
    target_device = found_in_coord_index.device
    K, M = found_in_coord_index.shape

    if return_type == "indices":
        return found_in_coord_index

    assert return_type == "offsets"

    found_in_coord_index_bool = found_in_coord_index >= 0

    # get the index of the non zero elements
    mapped_indices = (
        torch.cumsum(found_in_coord_index_bool.to(torch.int32), dim=1, dtype=torch.int32) - 1
    )
    # Need to handle rows with zero valid maps correctly (cumsum results in -1)
    # Clamp minimum value to 0 after subtracting 1
    mapped_indices = torch.clamp(mapped_indices, min=-1)  # Keep -1 for rows with no hits

    # Count valid maps per kernel offset row
    # If mapped_indices is -1 everywhere in a row, max will be -1, add 1 -> 0 count.
    num_valid_maps = mapped_indices.max(dim=1).values + 1

    # Calculate offsets
    offsets = torch.cumsum(num_valid_maps, dim=0, dtype=torch.int32)
    offsets = torch.cat([torch.zeros(1, dtype=torch.int32, device=target_device), offsets], dim=0)
    num_total_maps = offsets[-1].item()

    # Allocate output tensors
    in_maps = torch.empty(num_total_maps, device=target_device, dtype=torch.int32)
    out_maps = torch.empty(num_total_maps, device=target_device, dtype=torch.int32)

    if num_total_maps > 0:
        # Launch CUDA kernel to gather results
        map_results_kernel = _get_map_results_kernel()
        grid_size = math.ceil(found_in_coord_index.numel() / threads_per_block)

        # Ensure tensors are contiguous for kernel launch if necessary (CuPy might handle non-contiguous? Check docs)
        # Let's assume contiguous for safety for now.
        found_in_coord_index_cont = found_in_coord_index.contiguous()
        mapped_indices_cont = mapped_indices.contiguous()
        offsets_cont = offsets.contiguous()

        map_results_kernel(
            (grid_size,),
            (threads_per_block,),
            (
                found_in_coord_index_cont.data_ptr(),
                mapped_indices_cont.data_ptr(),
                offsets_cont.data_ptr(),
                in_maps.data_ptr(),
                out_maps.data_ptr(),
                K,  # num_kernel_offsets
                M,  # num_query_coords
            ),
        )
        # torch.cuda.synchronize(target_device) # Optional sync needed?

    return IntSearchResult(in_maps, out_maps, offsets)


@torch.no_grad()
def _kernel_map_from_offsets(
    hashtable: TorchHashTable,  # Use TorchHashTable
    batched_query_coords: Int[Tensor, "N D_1"],
    kernel_offsets: Int[Tensor, "K D_1"],
    return_type: Literal["indices", "offsets"] = "offsets",
) -> Int[Tensor, "K N"] | IntSearchResult:
    """
    Compute the kernel map (input index, output index) for each kernel offset using TorchHashTable.
    Assumes D_1 includes batch dimension (e.g., 4 for 3D spatial + batch).
    """
    target_device = hashtable.device
    assert (
        target_device == batched_query_coords.device
    ), f"{target_device} != {batched_query_coords.device}"
    assert target_device == kernel_offsets.device, f"{target_device} != {kernel_offsets.device}"
    assert batched_query_coords.shape[1] == kernel_offsets.shape[1]
    assert batched_query_coords.ndim == 2
    assert kernel_offsets.ndim == 2
    assert batched_query_coords.dtype == torch.int32
    assert kernel_offsets.dtype == torch.int32

    if hashtable._table_kvs is None or hashtable._vector_keys is None:
        raise RuntimeError(
            "Input TorchHashTable must be populated before calling kernel map functions."
        )

    num_query_coords = batched_query_coords.shape[0]
    key_dim = batched_query_coords.shape[1]
    num_kernel_offsets = kernel_offsets.shape[0]

    # Allocate output tensor
    found_in_coord_index = torch.empty(
        (num_kernel_offsets, num_query_coords),
        dtype=torch.int32,
        device=target_device,
    )

    # Get the appropriate kernel based on hash method
    kernel = _get_kernel_map_offset_kernel(hashtable.hash_method)
    threads_per_block = 256
    grid_size = math.ceil(num_query_coords / threads_per_block)

    # Ensure contiguous tensors for kernel launch
    table_kvs_cont = hashtable._table_kvs.contiguous()
    vector_keys_cont = hashtable._vector_keys.contiguous()
    query_coords_cont = batched_query_coords.contiguous()
    kernel_offsets_cont = kernel_offsets.contiguous()

    # Launch the kernel
    kernel(
        (grid_size,),
        (threads_per_block,),
        (
            table_kvs_cont.data_ptr(),
            vector_keys_cont.data_ptr(),
            query_coords_cont.data_ptr(),
            kernel_offsets_cont.data_ptr(),
            found_in_coord_index.data_ptr(),  # Output
            num_query_coords,  # N
            key_dim,  # D+1
            num_kernel_offsets,  # K
            hashtable.capacity,
        ),
    )
    # torch.cuda.synchronize(target_device) # Optional

    return _kernel_map_search_to_result(found_in_coord_index, return_type)


@torch.no_grad()
def _kernel_map_from_size(
    hashtable: TorchHashTable,  # Use TorchHashTable
    batched_query_coords: Int[Tensor, "N D_1"],
    kernel_sizes: Tuple[int, ...],
    return_type: Literal["indices", "offsets"] = "offsets",
) -> Int[Tensor, "K N"] | IntSearchResult:
    """
    Compute the kernel map using kernel_size. Uses _kernel_map_from_offsets internally,
    or a specialized kernel if coordinates are 4D.
    Assumes D_1 includes batch dimension.
    """
    target_device = hashtable.device
    assert str(target_device) == str(batched_query_coords.device)
    assert batched_query_coords.dtype == torch.int32

    if hashtable._table_kvs is None or hashtable._vector_keys is None:
        raise RuntimeError(
            "Input TorchHashTable must be populated before calling kernel map functions."
        )

    num_dims = batched_query_coords.shape[1]
    assert (
        len(kernel_sizes) == num_dims - 1
    ), f"kernel_size ({len(kernel_sizes)}) must match spatial dims ({num_dims - 1})"

    # --- Specialized 4D Case ---
    if num_dims == 4:
        num_query_coords = batched_query_coords.shape[0]
        num_kernels = kernel_sizes[0] * kernel_sizes[1] * kernel_sizes[2]

        # Allocate output tensor
        found_in_coord_index = torch.empty(
            (num_kernels, num_query_coords),  # Shape K x N
            dtype=torch.int32,
            device=target_device,
        )

        kernel = _get_kernel_map_size_4d_kernel(hashtable.hash_method)
        threads_per_block = 256
        grid_size = math.ceil(num_query_coords / threads_per_block)

        # Prepare kernel arguments
        table_kvs_cont = hashtable._table_kvs.contiguous()
        vector_keys_cont = hashtable._vector_keys.contiguous()
        query_coords_cont = batched_query_coords.contiguous()
        kernel_size_arg = cp.array(kernel_sizes, dtype=cp.int32)

        kernel(
            (grid_size,),
            (threads_per_block,),
            (
                table_kvs_cont.data_ptr(),
                vector_keys_cont.data_ptr(),
                query_coords_cont.data_ptr(),
                kernel_size_arg,  # Pass cp.int3
                found_in_coord_index.data_ptr(),
                num_query_coords,
                hashtable.capacity,
            ),
        )
        # torch.cuda.synchronize(target_device) # Optional

        return _kernel_map_search_to_result(found_in_coord_index, return_type)

    # --- Generic Case (Fallback to offset method) ---
    else:
        logger.warning(
            f"Using generic offset-based kernel map for {num_dims}D coords when method='size'. "
            f"Consider implementing a specialized kernel or using method='offset' directly for potential performance gains."
        )  # Log a warning for non-4D case using size method
        # Generate kernel offsets on the correct device
        kernel_offsets_tensor = kernel_offsets_from_size(
            kernel_sizes, (1,) * len(kernel_sizes), device=target_device
        )

        # Call the offset-based function
        return _kernel_map_from_offsets(
            hashtable, batched_query_coords, kernel_offsets_tensor, return_type=return_type
        )


def _kernel_map_from_direct_queries(
    hashtable: TorchHashTable,  # Use TorchHashTable
    batch_indexed_out_coords: Int[Tensor, "M D_1"],
    kernel_size: Tuple[int, ...],
    kernel_dilation: Optional[Tuple[int, ...]] = None,
    kernel_search_batch_size: Optional[int] = None,
    kernel_center_offset: Optional[Tuple[int, ...]] = None,
) -> IntSearchResult:
    """Computes kernel map by applying offsets and directly querying the hash table in batches."""
    target_device = hashtable.device
    num_spatial_dims = batch_indexed_out_coords.shape[1] - 1

    if kernel_dilation is None:
        kernel_dilation = (1,) * num_spatial_dims

    assert len(kernel_size) == num_spatial_dims
    assert len(kernel_dilation) == num_spatial_dims
    assert target_device == batch_indexed_out_coords.device
    assert batch_indexed_out_coords.dtype == torch.int32

    num_total_kernels = np.prod(kernel_size)
    if kernel_search_batch_size is None or kernel_search_batch_size <= 0:
        # Heuristic: aim for around 1M-4M queries per batch, avoid tiny/huge batches
        N_out = batch_indexed_out_coords.shape[0]
        approx_queries_per_batch = 2 * 1024 * 1024  # Aim for 2M
        kernel_search_batch_size = max(
            1, min(num_total_kernels, approx_queries_per_batch // N_out)
        )
    # Ensure it doesn't exceed total kernels
    kernel_search_batch_size = min(kernel_search_batch_size, num_total_kernels)

    N_out = batch_indexed_out_coords.shape[0]

    # Found indices and offsets for each kernel offset
    in_maps_list = []
    out_maps_list = []
    num_valid_maps_list = []

    # Generate all kernel offsets once
    all_offsets = kernel_offsets_from_size(
        kernel_size, kernel_dilation, center_offset=kernel_center_offset, device=target_device
    )

    for batch_start in range(0, num_total_kernels, kernel_search_batch_size):
        batch_end = min(batch_start + kernel_search_batch_size, num_total_kernels)
        num_kernels_in_batch = batch_end - batch_start
        curr_offsets = all_offsets[batch_start:batch_end]

        # Apply offsets and reshape for batch search
        # Unsqueeze dims: [K_batch, 1, D+1] + [1, M, D+1] -> [K_batch, M, D+1]
        coords_to_search = curr_offsets.unsqueeze(1) + batch_indexed_out_coords.unsqueeze(0)
        coords_to_search = coords_to_search.view(-1, num_spatial_dims + 1).contiguous()

        # Query the hashtable for all new coordinates at once
        # Note: This uses the TorchHashTable's search method, which launches its own kernel.
        in_indices = hashtable.search(coords_to_search)

        # Get the valid indices and offsets
        valid_in_indices_bool = in_indices >= 0
        num_valid_in_batch = valid_in_indices_bool.sum().item()

        if num_valid_in_batch > 0:
            # Calculate corresponding output indices (original query indices)
            # Create indices [0, 1, ..., N_out-1] repeated K_batch times
            out_indices_base = torch.arange(N_out, device=target_device)
            out_indices_expanded = out_indices_base.repeat(num_kernels_in_batch)

            # Filter based on valid search results
            valid_in_indices_int = in_indices[valid_in_indices_bool]
            valid_out_indices_int = out_indices_expanded[valid_in_indices_bool]

            in_maps_list.append(valid_in_indices_int)
            out_maps_list.append(valid_out_indices_int)
        else:
            # Append empty tensors if no hits in this batch for correct shape later
            in_maps_list.append(torch.empty((0,), dtype=torch.int32, device=target_device))
            out_maps_list.append(torch.empty((0,), dtype=torch.int32, device=target_device))

        # Count valid per kernel offset for the final offset calculation
        num_valid_per_kernel = valid_in_indices_bool.view(num_kernels_in_batch, N_out).sum(dim=1)
        num_valid_maps_list.append(num_valid_per_kernel)

    # Concatenate all the maps
    in_maps = torch.cat(in_maps_list, dim=0)
    out_maps = torch.cat(out_maps_list, dim=0)
    num_valid_maps = torch.cat(num_valid_maps_list, dim=0)

    # Convert the num_valid_maps to offsets
    offsets = torch.cumsum(num_valid_maps, dim=0, dtype=torch.int32)
    offsets = torch.cat([torch.zeros(1, dtype=torch.int32, device=target_device), offsets], dim=0)

    return IntSearchResult(in_maps, out_maps, offsets)


@torch.no_grad()
def generate_kernel_map(
    batch_indexed_in_coords: Int[Tensor, "N D_1"],
    batch_indexed_out_coords: Int[Tensor, "M D_1"],
    in_to_out_stride_ratio: Tuple[int, ...],
    kernel_size: Tuple[int, ...],
    kernel_dilation: Optional[Tuple[int, ...]] = None,
    kernel_search_batch_size: Optional[int] = None,
    kernel_center_offset: Optional[Tuple[int, ...]] = None,
    method: Literal["query", "offset", "size"] = "size",  # Added "size" back as an option
    hash_method: HashMethod = HashMethod.CITY,  # Allow selecting hash method
) -> IntSearchResult:
    """
    Generate the kernel map for the spatially sparse convolution using TorchHashTable.

    in_to_out_stride_ratio: the ratio of the input stride to the output stride. This will be multiplied to output coordinates to find matching input coordinates.
    method: 'query' directly queries the hash table for each offset point (can be slower for large kernels but flexible).
            'offset' pre-calculates all kernel offsets and uses a custom kernel to find matches (generally faster).
            'size' uses a specialized kernel for 4D coordinates if applicable, otherwise falls back to 'offset'.
    """
    target_device = batch_indexed_in_coords.device
    assert target_device == batch_indexed_out_coords.device
    assert batch_indexed_in_coords.dtype == torch.int32
    assert batch_indexed_out_coords.dtype == torch.int32

    # Create a TorchHashTable for the input coordinates
    hashtable = TorchHashTable.from_keys(
        batch_indexed_in_coords, hash_method=hash_method, device=target_device
    )

    num_spatial_dims = batch_indexed_out_coords.shape[1] - 1
    assert len(in_to_out_stride_ratio) == num_spatial_dims

    # Apply stride ratio to output coordinates
    if not all(s == 1 for s in in_to_out_stride_ratio):
        stride_tensor = torch.tensor(
            [1] + list(ntuple(in_to_out_stride_ratio, ndim=num_spatial_dims)),
            dtype=torch.int32,
            device=target_device,
        )
        # Ensure broadcasting works: coords [M, D+1], stride [D+1]
        strided_out_coords = batch_indexed_out_coords * stride_tensor
    else:
        strided_out_coords = batch_indexed_out_coords

    if method == "query":
        # This method applies offsets and calls hashtable.search directly
        return _kernel_map_from_direct_queries(
            hashtable,
            strided_out_coords,
            kernel_size,
            kernel_dilation=kernel_dilation,
            kernel_search_batch_size=kernel_search_batch_size,
            kernel_center_offset=kernel_center_offset,
        )
    elif method == "offset":
        # This method generates offsets and launches the custom kernel_map_offset kernel
        if kernel_dilation is None:
            kernel_dilation = (1,) * num_spatial_dims

        kernel_offsets_tensor = kernel_offsets_from_size(
            kernel_size, kernel_dilation, center_offset=kernel_center_offset, device=target_device
        )
        return _kernel_map_from_offsets(
            hashtable,
            strided_out_coords,  # Use strided coordinates
            kernel_offsets_tensor,
            return_type="offsets",
        )
    elif method == "size":
        # This method uses _kernel_map_from_size, which has the 4D specialization
        assert kernel_dilation is None or all(
            s == 1 for s in kernel_dilation
        ), "Kernel dilation is not supported with method='size'. Use method='offset' instead."
        assert (
            kernel_center_offset is None
        ), "Custom kernel_center_offset is not supported with method='size'. Use method='offset' instead."
        return _kernel_map_from_size(
            hashtable,
            strided_out_coords,
            kernel_size,
            return_type="offsets",
        )
    else:
        raise ValueError(f"Invalid method: {method}. Choose 'query', 'offset', or 'size'.")


def _int_sequence_hash(arr: Sequence[int]) -> int:  # noqa: F821
    x = hash(arr[0])
    for i in range(1, len(arr)):
        x = (x * 31 + hash(arr[i])) & 0xFFFFFFFF  # Keep it within 32-bit range
    return x


def string_hash(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16) & 0xFFFFFFFF
