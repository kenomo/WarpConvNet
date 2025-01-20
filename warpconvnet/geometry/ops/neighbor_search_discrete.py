import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
import warp as wp
import warp.utils
from jaxtyping import Int
from torch import Tensor

from warpconvnet.geometry.coords.spatial.hashmap import HashStruct, VectorHashTable, search_func
from warpconvnet.utils.batch_index import batch_indexed_coordinates
from warpconvnet.utils.ntuple import ntuple


class DISCRETE_NEIGHBOR_SEARCH_MODE(Enum):
    MANHATTAN_DISTANCE = "manhattan_distance"
    CUSTOM_OFFSETS = "custom_offsets"


class DiscreteNeighborSearchArgs:
    """
    Wrapper for the input of a neighbor search operation.
    """

    # The mode of the neighbor search
    _mode: DISCRETE_NEIGHBOR_SEARCH_MODE
    _distance_threshold: Optional[int]
    _offsets: Optional[Int[Tensor, "K 3"]]

    def __init__(
        self,
        mode: DISCRETE_NEIGHBOR_SEARCH_MODE,
        distance_threshold: Optional[int] = None,
        offsets: Optional[Int[Tensor, "K 3"]] = None,
    ):
        self._mode = mode
        if mode == DISCRETE_NEIGHBOR_SEARCH_MODE.MANHATTAN_DISTANCE:
            assert (
                distance_threshold is not None
            ), "Distance threshold must be provided for manhattan distance search"
            self._distance_threshold = distance_threshold
        elif mode == DISCRETE_NEIGHBOR_SEARCH_MODE.CUSTOM_OFFSETS:
            assert offsets is not None, "Offsets must be provided for custom offsets search"
            self._offsets = offsets
        else:
            raise ValueError(f"Invalid neighbor search mode: {mode}")


@dataclass
class DiscreteNeighborSearchResult:
    """
    Wrapper for the output of a neighbor search operation.
    """

    # The indices of the neighbors
    in_maps: Int[Tensor, "L"]  # noqa: F821
    out_maps: Int[Tensor, "L"]  # noqa: F821
    offsets: Int[Tensor, "K + 1"]  # noqa: F821

    def __init__(
        self,
        in_maps: Int[Tensor, "L"],  # noqa: F821
        out_maps: Int[Tensor, "L"],  # noqa: F821
        offsets: Int[Tensor, "K + 1"],  # noqa: F821
    ):
        assert len(in_maps) == len(out_maps) == offsets[-1].item()
        self.in_maps = in_maps
        self.out_maps = out_maps
        self.offsets = offsets.cpu()

    @torch.no_grad()
    def __getitem__(self, idx: int) -> Tuple[Int[Tensor, "N"], Int[Tensor, "N"]]:  # noqa: F821
        start, end = self.offsets[idx], self.offsets[idx + 1]
        return self.in_maps[start:end], self.out_maps[start:end]

    @torch.no_grad()
    def get_batch(
        self,
        start_idx: int,
        end_idx: int,
        out_format: Literal["list", "tensor"] = "list",
    ) -> Tuple[List[Int[Tensor, "N"]], List[Int[Tensor, "N"]]]:  # noqa: F821
        in_maps = []
        out_maps = []
        for i in range(start_idx, end_idx):
            in_maps.append(self.in_maps[self.offsets[i] : self.offsets[i + 1]])
            out_maps.append(self.out_maps[self.offsets[i] : self.offsets[i + 1]])
        if out_format == "list":
            return in_maps, out_maps
        elif out_format == "tensor":
            max_num_maps = max(len(in_map) for in_map in in_maps)
            in_maps_tensor = -1 * torch.ones(
                len(in_maps),
                max_num_maps,
                device=self.in_maps.device,
                dtype=torch.int64,
            )
            out_maps_tensor = -1 * torch.ones(
                len(out_maps),
                max_num_maps,
                device=self.out_maps.device,
                dtype=torch.int64,
            )
            for i, (in_map, out_map) in enumerate(zip(in_maps, out_maps)):
                in_maps_tensor[i, : len(in_map)] = in_map
                out_maps_tensor[i, : len(out_map)] = out_map
            return in_maps_tensor, out_maps_tensor
        else:
            raise ValueError(f"Invalid output format: {out_format}")

    def __len__(self):
        return len(self.offsets) - 1

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __repr__(self):
        return f"{self.__class__.__name__}(len={len(self)})"

    @torch.no_grad()
    def to_csr(
        self,
    ) -> Tuple[Int[Tensor, "L"], Int[Tensor, "K"], Int[Tensor, "K + 1"]]:  # noqa: F821
        """
        Convert the neighbor search result to a CSR format.

        in_maps to row indices
        out_maps to sort and use for columns
        ignore offsets
        """
        in_maps = self.in_maps
        out_maps = self.out_maps

        # Sort the out_maps and get the indices
        out_maps_sorted, out_maps_indices = torch.sort(out_maps)
        # cchoy: Could skip the sorting by implementing a custom warp kernel
        unique_out_maps_sorted, num_unique = torch.unique(
            out_maps_sorted, return_counts=True, sorted=True
        )

        # Get the in_maps from the indices
        in_maps_sorted = in_maps[out_maps_indices]

        # convert to offsets
        offsets = torch.cumsum(num_unique.cpu(), dim=0)
        offsets = torch.cat([torch.zeros(1, dtype=torch.int32), offsets], dim=0)
        return in_maps_sorted, unique_out_maps_sorted, offsets

    def clone(self):
        return DiscreteNeighborSearchResult(
            self.in_maps.clone(), self.out_maps.clone(), self.offsets.clone()
        )


@wp.kernel
def conv_kernel_map_arr(
    in_hashmap: HashStruct,
    query_coords: wp.array2d(dtype=int),
    scratch_coords: wp.array2d(dtype=int),
    kernel_offsets: wp.array2d(dtype=int),
    found_in_coord_index: wp.array2d(dtype=int),
):
    """
    Compute whether query + offset is in in_coords and return the index of the found input coordinate.

    For definitions, please refer to Sec. 4.2. of https://arxiv.org/pdf/1904.08755
    """
    idx = wp.tid()
    for k in range(kernel_offsets.shape[0]):
        # TODO(cchoy): Change this to shared memory operation.
        # Copy the query coordinate to the scratch coordinate.
        query_coord = scratch_coords[idx]
        for dim in range(kernel_offsets.shape[1]):
            query_coord[dim] = query_coords[idx][dim] + kernel_offsets[k][dim]
        index = search_func(
            in_hashmap.table_kvs,
            in_hashmap.vector_keys,
            query_coord,
            in_hashmap.capacity,
            in_hashmap.hash_method,
        )
        found_in_coord_index[k][idx] = index


@wp.kernel
def conv_kernel_map_vec4i(
    in_hashmap: HashStruct,
    query_coords: wp.array(dtype=wp.vec4i),
    kernel_size: wp.vec3i,
    found_in_coord_index: wp.array2d(dtype=int),
):
    """
    Compute whether query + offset is in in_coords and return the index of the found input coordinate.

    For definitions, please refer to Sec. 4.2. of https://arxiv.org/pdf/1904.08755
    """
    idx = wp.tid()

    # center to be 0 if kernel size is even
    center = wp.vec3i(0, 0, 0)
    if kernel_size[0] % 2 != 0:
        center[0] = kernel_size[0] // 2
    if kernel_size[1] % 2 != 0:
        center[1] = kernel_size[1] // 2
    if kernel_size[2] % 2 != 0:
        center[2] = kernel_size[2] // 2
    kernel_index = int(0)
    b = query_coords[idx][0]
    # Loop over the neighbors
    for i in range(kernel_size[0]):
        for j in range(kernel_size[1]):
            for k in range(kernel_size[2]):
                # Compute query coord
                coord = wp.vec4i(
                    b,
                    query_coords[idx][1] + i - center[0],
                    query_coords[idx][2] + j - center[1],
                    query_coords[idx][3] + k - center[2],
                )
                index = search_func(
                    in_hashmap.table_kvs,
                    in_hashmap.vector_keys,
                    coord,
                    in_hashmap.capacity,
                    in_hashmap.hash_method,
                )
                found_in_coord_index[kernel_index][idx] = index
                kernel_index += 1


@torch.no_grad()
def kernel_offsets_from_size(
    kernel_size: Tuple[int, ...],
    kernel_dilation: Tuple[int, ...],
    center_offset: Optional[Tuple[int, ...]] = None,
) -> Int[Tensor, "K D+1"]:
    """
    Generate the kernel offsets for the spatially sparse convolution.
    Supports arbitrary number of spatial dimensions.
    """
    assert len(kernel_size) == len(kernel_dilation)
    num_spatial_dims = len(kernel_size)

    # Create meshgrid for arbitrary dimensions
    ranges = [torch.arange(size, dtype=torch.int32) for size in kernel_size]
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

    return torch.stack(offsets, dim=1)


@wp.kernel
def map_found_indices_to_maps(
    found_in_coord_index: wp.array2d(dtype=int),
    mapped_indicies: wp.array2d(dtype=int),
    offsets: wp.array(dtype=int),
    in_maps: wp.array(dtype=int),
    out_maps: wp.array(dtype=int),
):
    idx = wp.tid()
    # K = found_in_coord_index.shape[0]
    M = found_in_coord_index.shape[1]

    k = idx // M
    m = idx % M
    if found_in_coord_index[k][m] >= 0:
        in_maps[mapped_indicies[k][m] + offsets[k]] = found_in_coord_index[k][m]
        out_maps[mapped_indicies[k][m] + offsets[k]] = m


@torch.no_grad()
def _kernel_map_search_to_result(
    found_in_coord_index_wp: wp.array2d(dtype=int),
    return_type: Literal["indices", "offsets"] = "offsets",
) -> Int[Tensor, "K M"] | DiscreteNeighborSearchResult:
    # Must have shape [K, M]
    # assert found_in_coord_index_wp.shape[0] == kernel_offsets.shape[0]
    # assert found_in_coord_index_wp.shape[1] == batched_query_coords.shape[0]
    found_in_coord_index = wp.to_torch(found_in_coord_index_wp)
    str_device = str(found_in_coord_index.device)
    K, M = found_in_coord_index.shape
    if return_type == "indices":
        return found_in_coord_index

    assert return_type == "offsets"
    found_in_coord_index_bool = found_in_coord_index >= 0
    # Debug only
    if False:
        num_valid_maps = found_in_coord_index_bool.sum(1)

        out_indices = torch.arange(M, device=str_device).repeat(K, 1)
        in_maps = found_in_coord_index[found_in_coord_index_bool]
        out_maps = out_indices[found_in_coord_index_bool]
        # convert the num_valid_maps to an offset
        offsets = torch.cumsum(num_valid_maps.cpu(), dim=0)
        # prepend 0 to the num_valid_maps
        offsets = torch.cat([torch.zeros(1, dtype=torch.int32), offsets], dim=0)
    else:
        # get the index of the non zero elements
        mapped_indices = (
            torch.cumsum(found_in_coord_index_bool.to(torch.int32), dim=1, dtype=torch.int32) - 1
        )
        num_valid_maps = mapped_indices[:, -1].cpu() + 1
        # convert the num_valid_maps to an offset
        offsets = torch.cumsum(num_valid_maps.cpu(), dim=0, dtype=torch.int32)
        # prepend 0 to the num_valid_maps
        offsets = torch.cat([torch.zeros(1, dtype=torch.int32), offsets], dim=0)
        num_total_maps = offsets[-1].item()
        in_maps_wp = wp.empty(num_total_maps, device=str_device, dtype=wp.int32)
        out_maps_wp = wp.empty(num_total_maps, device=str_device, dtype=wp.int32)
        mapped_indices_wp = wp.from_torch(mapped_indices)  # wp.array2d(dtype=int)
        found_in_coord_index_wp = wp.from_torch(found_in_coord_index)
        offsets_wp = wp.from_torch(offsets).to(device=str_device)
        wp.launch(
            kernel=map_found_indices_to_maps,
            dim=found_in_coord_index.numel(),
            inputs=[
                found_in_coord_index_wp,
                mapped_indices_wp,
                offsets_wp,
                in_maps_wp,
                out_maps_wp,
            ],
            device=str_device,
        )
        in_maps = wp.to_torch(in_maps_wp)
        out_maps = wp.to_torch(out_maps_wp)

    return DiscreteNeighborSearchResult(in_maps, out_maps, offsets)


@torch.no_grad()
def _kernel_map_from_offsets(
    in_hashmap: HashStruct,
    batched_query_coords: Int[Tensor, "N 4"],
    kernel_offsets: Int[Tensor, "K 4"],
    return_type: Literal["indices", "offsets"] = "offsets",
) -> Int[Tensor, "K N"] | DiscreteNeighborSearchResult:
    """
    Compute the kernel map (input index, output index) for each kernel offset using cached hashmap
    """
    device_wp = in_hashmap.table_kvs.device  # string device from warp array
    assert device_wp == str(
        batched_query_coords.device
    ), f"{device_wp} != {str(batched_query_coords.device)}"
    assert device_wp == str(kernel_offsets.device), f"{device_wp} != {kernel_offsets.device}"
    assert batched_query_coords.shape[1] == kernel_offsets.shape[1]

    # Allocate output of size K x N
    found_in_coord_index_wp = wp.empty(
        (len(kernel_offsets), len(batched_query_coords)),
        dtype=wp.int32,
        device=device_wp,
    )
    batched_query_coords_wp = wp.from_torch(batched_query_coords)
    kernel_offsets_wp = wp.from_torch(kernel_offsets)
    scratch_coords_wp = wp.empty_like(batched_query_coords_wp)

    # Launch the kernel
    wp.launch(
        kernel=conv_kernel_map_arr,
        dim=len(batched_query_coords),
        inputs=[
            in_hashmap,
            batched_query_coords_wp,
            scratch_coords_wp,
            kernel_offsets_wp,
            found_in_coord_index_wp,
        ],
        device=device_wp,
    )
    return _kernel_map_search_to_result(found_in_coord_index_wp, return_type)


@torch.no_grad()
def _kernel_map_from_size(
    in_hashmap: HashStruct,
    batched_query_coords: Int[Tensor, "N D"],
    kernel_sizes: Tuple[int, ...],
    return_type: Literal["indices", "offsets"] = "offsets",
) -> Int[Tensor, "K N"] | DiscreteNeighborSearchResult:
    """
    Compute the kernel map (input index, output index) for each kernel offset using cached hashmap.
    Supports arbitrary number of spatial dimensions.
    """
    device_wp = in_hashmap.table_kvs.device  # string device from warp array
    device_torch = batched_query_coords.device
    assert str(device_wp) == str(
        device_torch
    ), f"warp device {device_wp} != torch device {device_torch}"

    num_dims = batched_query_coords.shape[1]
    assert num_dims in (3, 4), f"Expected 3 or 4 dimensions, got {num_dims}"
    assert len(kernel_sizes) == num_dims - 1

    num_kernels = np.prod(kernel_sizes)
    # Allocate output of size K x N
    found_in_coord_index_wp = wp.empty(
        (num_kernels, len(batched_query_coords)),
        dtype=wp.int32,
        device=device_wp,
    )

    if num_dims == 4:
        # Use existing vec4i implementation for 4D coordinates
        batched_query_coords_wp = wp.from_torch(batched_query_coords, dtype=wp.vec4i)
        kernel_sizes_wp = wp.vec3i(kernel_sizes)

        wp.launch(
            kernel=conv_kernel_map_vec4i,
            dim=len(batched_query_coords),
            inputs=[
                in_hashmap,
                batched_query_coords_wp,
                kernel_sizes_wp,
                found_in_coord_index_wp,
            ],
            device=device_wp,
        )
    else:
        # Use array implementation for non 4D coordinates
        batched_query_coords_wp = wp.from_torch(batched_query_coords)
        # Generate kernel offsets
        offsets = kernel_offsets_from_size(kernel_sizes, (1,) * len(kernel_sizes)).to(device_torch)
        kernel_offsets_wp = wp.from_torch(offsets)
        scratch_coords_wp = wp.empty_like(batched_query_coords_wp)

        wp.launch(
            kernel=conv_kernel_map_arr,
            dim=len(batched_query_coords),
            inputs=[
                in_hashmap,
                batched_query_coords_wp,
                scratch_coords_wp,
                kernel_offsets_wp,
                found_in_coord_index_wp,
            ],
            device=device_wp,
        )

    return _kernel_map_search_to_result(found_in_coord_index_wp, return_type)


def _kernel_map_from_direct_queries(
    in_hashmap: HashStruct,
    batch_indexed_out_coords: Int[Tensor, "M 4"],
    kernel_size: Tuple[int, ...],
    kernel_dilation: Optional[Tuple[int, ...]] = None,
    kernel_search_batch_size: Optional[int] = None,
    kernel_center_offset: Optional[Tuple[int, ...]] = None,
) -> DiscreteNeighborSearchResult:
    num_spatial_dims = batch_indexed_out_coords.shape[1] - 1
    str_device = str(in_hashmap.table_kvs.device)
    if kernel_dilation is None:
        kernel_dilation = (1,) * num_spatial_dims

    assert len(kernel_size) == num_spatial_dims
    assert len(kernel_dilation) == num_spatial_dims
    assert str_device == str(batch_indexed_out_coords.device)

    num_total_kernels = np.prod(kernel_size)
    if kernel_search_batch_size is None:
        kernel_search_batch_size = num_total_kernels // kernel_size[0]

    N_out = batch_indexed_out_coords.shape[0]

    # Found indices and offsets for each kernel offset
    in_maps = []
    out_maps = []
    num_valid_maps = []

    # Query the hashtable for all kernel offsets
    all_out_indices = (
        torch.arange(N_out, device=str_device).repeat(kernel_search_batch_size, 1).view(-1)
    )

    # Generate kernel offsets
    offsets = kernel_offsets_from_size(
        kernel_size, kernel_dilation, center_offset=kernel_center_offset
    ).to(str_device)

    for batch_start in range(0, num_total_kernels, kernel_search_batch_size):
        batch_end = min(batch_start + kernel_search_batch_size, num_total_kernels)
        num_kernels_in_batch = batch_end - batch_start
        curr_offsets = offsets[batch_start:batch_end]

        # Apply offsets in batch and query output + offsets. Add the offsets in the expanded dimension
        # KND + K1D -> KND
        new_batch_indexed_out_coords = batch_indexed_out_coords.unsqueeze(
            0
        ) + curr_offsets.unsqueeze(1)
        new_batch_indexed_out_coords = new_batch_indexed_out_coords.view(-1, num_spatial_dims + 1)
        new_batch_indexed_out_coords_wp = wp.from_torch(new_batch_indexed_out_coords)

        # Query the hashtable for all new coordinates at once
        in_indices_wp = in_hashmap.search(new_batch_indexed_out_coords_wp)
        in_indices = wp.to_torch(in_indices_wp)

        # Get the valid indices and offsets.
        # valid indices are all >= 0 and offsets [0, N1, N1+N2, N1+N2+N3, ..., N1+...+N_kernel_batch] for N1, N2, N3 being the number of valid indices for each kernel offset
        valid_in_indices_bool = in_indices >= 0
        # Reshape valid indices to [kernel_batch, N_out] to get the number of valid indices for each kernel offset
        num_valid_in_indices = valid_in_indices_bool.view(num_kernels_in_batch, -1).sum(dim=1)
        # Compress indices to the valid indices
        valid_in_indices_int = in_indices[valid_in_indices_bool]
        if num_kernels_in_batch < kernel_search_batch_size:
            valid_out_indices_int = all_out_indices[: len(valid_in_indices_bool)][
                valid_in_indices_bool
            ]
        else:
            valid_out_indices_int = all_out_indices[valid_in_indices_bool]

        in_maps.append(valid_in_indices_int)
        out_maps.append(valid_out_indices_int)
        num_valid_maps.append(num_valid_in_indices)

    # Concatenate all the maps
    in_maps = torch.cat(in_maps, dim=0)
    out_maps = torch.cat(out_maps, dim=0)
    num_valid_maps = torch.cat(num_valid_maps, dim=0)
    # convert the num_valid_maps to an offset
    offsets = torch.cumsum(num_valid_maps, dim=0)
    # prepend 0 to the num_valid_maps
    offsets = torch.cat([torch.zeros(1, dtype=torch.int32, device=str_device), offsets], dim=0)

    return DiscreteNeighborSearchResult(in_maps, out_maps, offsets)


@torch.no_grad()
def generate_kernel_map(
    batch_indexed_in_coords: Int[Tensor, "N 4"],
    batch_indexed_out_coords: Int[Tensor, "M 4"],
    in_to_out_stride_ratio: Tuple[int, ...],
    kernel_size: Tuple[int, ...],
    kernel_dilation: Optional[Tuple[int, ...]] = None,
    kernel_search_batch_size: Optional[int] = None,
    kernel_center_offset: Optional[Tuple[int, ...]] = None,
    method: Literal["query", "size", "offset"] = "size",
) -> DiscreteNeighborSearchResult:
    """
    Generate the kernel map for the spatially sparse convolution.

    in_to_out_stride_ratio: the ratio of the input stride to the output stride. This will be multiplied to output coordinates to find matching input coordinates.
    """
    # Create a vector hashtable for the batched coordinates
    batch_indexed_in_coords_wp = wp.from_torch(batch_indexed_in_coords)
    hashtable = VectorHashTable.from_keys(batch_indexed_in_coords_wp)

    str_device = str(batch_indexed_out_coords.device)
    num_spatial_dims = batch_indexed_out_coords.shape[1] - 1

    # multiply output coordinates by in_to_out_stride_ratio if it is not all ones
    if not all(s == 1 for s in in_to_out_stride_ratio):
        batch_indexed_out_coords = batch_indexed_out_coords * torch.tensor(
            [1, *ntuple(in_to_out_stride_ratio, ndim=num_spatial_dims)],
            dtype=torch.int32,
            device=str_device,
        )
    else:
        batch_indexed_out_coords = batch_indexed_out_coords

    if method == "query":
        return _kernel_map_from_direct_queries(
            hashtable._hash_struct,
            batch_indexed_out_coords,
            kernel_size,
            kernel_dilation=kernel_dilation,
            kernel_search_batch_size=kernel_search_batch_size,
            kernel_center_offset=kernel_center_offset,
        )
    elif method == "size":
        assert kernel_dilation is None or all(s == 1 for s in kernel_dilation), "Not supported yet"
        assert kernel_center_offset is None, "Not supported yet"
        return _kernel_map_from_size(
            hashtable._hash_struct,
            batch_indexed_out_coords,
            kernel_size,
            return_type="offsets",
        )
    elif method == "offset":
        assert kernel_dilation is None or all(s == 1 for s in kernel_dilation), "Not supported yet"
        assert kernel_center_offset is None, "Not supported yet"
        kernel_offsets = kernel_offsets_from_size(
            kernel_size, kernel_dilation, center_offset=kernel_center_offset
        ).to(batch_indexed_out_coords.device)
        return _kernel_map_from_offsets(
            hashtable._hash_struct,
            batch_indexed_out_coords,
            kernel_offsets,
            return_type="offsets",
        )
    else:
        raise ValueError(f"Invalid method: {method}")


def _int_sequence_hash(arr: Sequence[int]) -> int:  # noqa: F821
    x = hash(arr[0])
    for i in range(1, len(arr)):
        x = x * 31 + hash(arr[i])
    return x


# Use a deterministic hash function for strings
def string_hash(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


class KernelMapCacheKey:
    """
    Key for kernel map cache.
    """

    kernel_size: Tuple[int, ...]
    kernel_dilation: Tuple[int, ...]
    transposed: bool
    generative: bool
    stride_mode: str
    in_offsets: Int[Tensor, "B+1"]  # noqa: F821
    out_offsets: Int[Tensor, "B+1"]  # noqa: F821

    def __init__(
        self,
        kernel_size,
        kernel_dilation,
        transposed,
        generative,
        stride_mode,
        in_offsets,
        out_offsets,
    ):
        self.kernel_size = kernel_size
        self.kernel_dilation = kernel_dilation
        self.transposed = transposed
        self.generative = generative
        self.stride_mode = stride_mode
        self.in_offsets = in_offsets.detach().cpu().int()
        self.out_offsets = out_offsets.detach().cpu().int()

    def __hash__(self):
        return int(
            _int_sequence_hash(self.kernel_size)
            ^ _int_sequence_hash(self.kernel_dilation)
            ^ hash(self.transposed)
            ^ hash(self.generative)
            ^ string_hash(self.stride_mode)  # Use string_hash for stride_mode
            ^ _int_sequence_hash(self.in_offsets.tolist())
            ^ _int_sequence_hash(self.out_offsets.tolist())
        )

    def __eq__(self, other: "KernelMapCacheKey"):
        return (
            self.kernel_size == other.kernel_size
            and self.kernel_dilation == other.kernel_dilation
            and self.transposed == other.transposed
            and self.generative == other.generative
            and self.stride_mode == other.stride_mode
            and self.in_offsets.equal(other.in_offsets)
            and self.out_offsets.equal(other.out_offsets)
        )

    def __repr__(self):
        return f"KernelMapCacheKey(kernel_size={self.kernel_size}, kernel_dilation={self.kernel_dilation}, transposed={self.transposed}, generative={self.generative}, stride_mode={self.stride_mode}, in_offsets={self.in_offsets}, out_offsets={self.out_offsets})"


class KernelMapCache:
    """
    Cache for kernel map.
    """

    def __init__(self):
        self.cache = {}

    def get(self, key: KernelMapCacheKey) -> Optional[DiscreteNeighborSearchResult]:
        return self.cache.get(key, None)

    def put(self, key: KernelMapCacheKey, value: DiscreteNeighborSearchResult):
        self.cache[key] = value

    def __repr__(self):
        return f"{self.__class__.__name__}({len(self.cache)} keys)"

    def __getstate__(self):
        return None

    def __setstate__(self, state):
        self.cache = {}
