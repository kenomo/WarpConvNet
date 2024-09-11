from dataclasses import dataclass
from enum import Enum
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
from jaxtyping import Int
from torch import Tensor

import warp as wp
import warp.utils
from warp.convnet.core.hashmap import HashStruct, VectorHashTable, search_func
from warp.convnet.utils.batch_index import batch_indexed_coordinates
from warp.convnet.utils.ntuple import ntuple


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
        self.offsets = offsets

    def __getitem__(self, idx: int) -> Tuple[Int[Tensor, "N"], Int[Tensor, "N"]]:  # noqa: F821
        start, end = self.offsets[idx], self.offsets[idx + 1]
        return self.in_maps[start:end], self.out_maps[start:end]

    def get_batch(
        self, start_idx: int, end_idx: int, out_format: Literal["list", "tensor"] = "list"
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
                len(in_maps), max_num_maps, device=self.in_maps.device, dtype=torch.int64
            )
            out_maps_tensor = -1 * torch.ones(
                len(out_maps), max_num_maps, device=self.out_maps.device, dtype=torch.int64
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


@wp.kernel
def conv_kernel_map(
    in_hashmap: HashStruct,
    query_coords: wp.array2d(dtype=int),
    N_query_coords: int,
    kernel_offsets: wp.array2d(dtype=int),
    N_kernel_offsets: int,
    found_in_coord_index: wp.array(dtype=int),
):
    """
    Compute whether query + offset is in in_coords and return the index of the found input coordinate.

    For definitions, please refer to Sec. 4.2. of https://arxiv.org/pdf/1904.08755
    """
    idx = wp.tid()
    for k in range(N_kernel_offsets):
        query_coord = query_coords[idx]
        for dim in range(query_coord.shape[0]):
            query_coord[dim] += kernel_offsets[k][dim]
        index = search_func(
            in_hashmap.table_kvs,
            in_hashmap.vector_keys,
            query_coord,
            in_hashmap.capacity,
            in_hashmap.hash_method,
        )
        found_in_coord_index[idx + N_query_coords * k] = index


@wp.kernel
def num_neighbors_kernel(
    in_hashmap: HashStruct,
    query_coords: wp.array2d(dtype=int),
    neighbor_distance_threshold: int,
    num_neighbors: wp.array(dtype=int),
):
    idx = wp.tid()

    curr_num_neighbors = int(0)
    center = neighbor_distance_threshold // 2
    # Loop over the neighbors
    for i in range(neighbor_distance_threshold):
        for j in range(neighbor_distance_threshold):
            for k in range(neighbor_distance_threshold):
                # Compute query coord
                query_coord = query_coords[idx]
                query_coord[1] += i - center
                query_coord[2] += j - center
                query_coord[3] += k - center
                index = search_func(
                    in_hashmap.table_kvs,
                    in_hashmap.vector_keys,
                    query_coord,
                    in_hashmap.capacity,
                    in_hashmap.hash_method,
                )
                if index >= 0:
                    curr_num_neighbors += 1

    num_neighbors[idx] = curr_num_neighbors


@wp.kernel
def fill_neighbors_kernel(
    in_hashmap: HashStruct,
    query_coords: wp.array2d(dtype=int),
    neighbor_distance_threshold: int,
    neighbor_offset_inclusive: wp.array(dtype=int),
    in_coords_index: wp.array(dtype=int),
    query_coords_index: wp.array(dtype=int),
):
    idx = wp.tid()
    if idx == 0:
        neighbor_offset = 0
    else:
        neighbor_offset = neighbor_offset_inclusive[idx - 1]
    curr_num_neighbors = int(neighbor_offset)
    center = neighbor_distance_threshold // 2
    # Loop over the neighbors
    for i in range(neighbor_distance_threshold):
        for j in range(neighbor_distance_threshold):
            for k in range(neighbor_distance_threshold):
                # Compute query coord
                query_coord = query_coords[idx]
                query_coord[1] += i - center
                query_coord[2] += j - center
                query_coord[3] += k - center
                index = search_func(
                    in_hashmap.table_kvs,
                    in_hashmap.vector_keys,
                    query_coord,
                    in_hashmap.capacity,
                    in_hashmap.hash_method,
                )
                if index >= 0:
                    in_coords_index[curr_num_neighbors] = index
                    query_coords_index[curr_num_neighbors] = idx
                    curr_num_neighbors += 1


def neighbor_search_hashmap(
    in_hashmap: HashStruct,
    batched_query_coords: wp.array2d(dtype=int),
    neighbor_distance_threshold: int,
):
    # device checks
    device = in_hashmap.table_kvs.device
    assert device == batched_query_coords.device, f"{device} != {batched_query_coords.device}"

    # Compute the number of neighbors for each query point
    num_neighbors = wp.empty(len(batched_query_coords), dtype=int, device=device)

    # Launch num neighbor kernel
    wp.launch(
        kernel=num_neighbors_kernel,
        dim=len(batched_query_coords),
        inputs=[
            in_hashmap,
            batched_query_coords,
            neighbor_distance_threshold,
            num_neighbors,
        ],
        device=device,
    )

    # array_scan to compute the total number and offsets of neighbors
    num_neighbors_scan_inclusive = wp.empty_like(num_neighbors)
    warp.utils.array_scan(num_neighbors, num_neighbors_scan_inclusive, inclusive=True)
    N = len(num_neighbors_scan_inclusive)
    tot = num_neighbors_scan_inclusive[N - 1 : N].numpy()

    # Allocate ouput
    in_coords_index = wp.empty(tot, dtype=int, device=device)
    query_coords_index = wp.empty(tot, dtype=int, device=device)

    # Launch the kernel
    wp.launch(
        kernel=fill_neighbors_kernel,
        dim=len(batched_query_coords),
        inputs=[
            in_hashmap,
            batched_query_coords,
            neighbor_distance_threshold,
            num_neighbors_scan_inclusive,
            in_coords_index,
            query_coords_index,
        ],
        device=device,
    )

    return in_coords_index, query_coords_index


def neighbor_search(
    in_coords: torch.Tensor,
    in_coords_offsets: torch.Tensor,
    query_coords: torch.Tensor,
    query_coords_offsets: torch.Tensor,
    neighbor_distance_threshold: int,
):
    # Convert the coordinates to batched coordinates
    in_bcoords = batch_indexed_coordinates(in_coords, in_coords_offsets)
    query_bcoords = batch_indexed_coordinates(query_coords, query_coords_offsets)
    query_bcoords = wp.from_torch(query_bcoords)

    # Create the hashmap for in_coords
    in_coords_hashmap = VectorHashTable.from_keys(in_bcoords)

    # Launch the kernel
    in_coords_index, query_coords_index = neighbor_search_hashmap(
        in_coords_hashmap, query_bcoords, neighbor_distance_threshold
    )

    return in_coords_index, query_coords_index


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


@torch.no_grad()
def kernel_map_from_offsets(
    in_hashmap: HashStruct,
    batched_query_coords: Int[Tensor, "M 4"],
    kernel_offsets: Int[Tensor, "K 4"],
    return_type: Literal["indices", "offsets"] = "offsets",
) -> Int[Tensor, "K M"] | DiscreteNeighborSearchResult:
    """
    Compute the kernel map (input index, output index) for each kernel offset using cached hashmap
    """
    device_wp = in_hashmap.table_kvs.device  # string device from warp array
    assert device_wp == str(
        batched_query_coords.device
    ), f"{device_wp} != {str(batched_query_coords.device)}"
    assert device_wp == str(kernel_offsets.device), f"{device_wp} != {kernel_offsets.device}"

    # Allocate output
    found_in_coord_index_wp = wp.empty(
        len(batched_query_coords) * len(kernel_offsets),
        dtype=wp.int32,
        device=device_wp,
    )

    # Launch the kernel
    wp.launch(
        kernel=conv_kernel_map,
        dim=len(batched_query_coords),
        inputs=[
            in_hashmap,
            wp.from_torch(batched_query_coords),
            len(batched_query_coords),
            wp.from_torch(kernel_offsets),
            len(kernel_offsets),
            found_in_coord_index_wp,
        ],
        device=device_wp,
    )

    found_in_coord_index = wp.to_torch(found_in_coord_index_wp).reshape(
        len(kernel_offsets), len(batched_query_coords)
    )

    if return_type == "indices":
        return found_in_coord_index

    assert return_type == "offsets"
    # Return the kernel map
    found_in_coord_index_bool = found_in_coord_index >= 0
    in_maps = found_in_coord_index[found_in_coord_index_bool]

    out_maps = []
    out_indices = torch.arange(len(batched_query_coords), device=str(device_wp))

    num_valid_maps = []
    for i in range(len(kernel_offsets)):
        out_maps.append(out_indices[found_in_coord_index_bool[i]])
        num_valid_maps.append(found_in_coord_index_bool[i].sum().item())

    out_maps = torch.cat(out_maps, dim=0)
    num_valid_maps = torch.tensor(num_valid_maps)
    # convert the num_valid_maps to an offset
    offsets = torch.cumsum(num_valid_maps, dim=0)
    # prepend 0 to the num_valid_maps
    offsets = torch.cat([torch.zeros(1, dtype=torch.int32), offsets], dim=0).to(str(device_wp))

    return DiscreteNeighborSearchResult(in_maps, out_maps, offsets)


@torch.no_grad()
def kernel_map_from_size(
    batch_indexed_in_coords: Int[Tensor, "N 4"],
    batch_indexed_out_coords: Int[Tensor, "M 4"],
    in_to_out_stride_ratio: Tuple[int, ...],
    kernel_size: Tuple[int, ...],
    kernel_dilation: Optional[Tuple[int, ...]] = None,
    kernel_search_batch_size: Optional[int] = None,
    kernel_center_offset: Optional[Tuple[int, ...]] = None,
) -> DiscreteNeighborSearchResult:
    """
    Generate the kernel map for the spatially sparse convolution.

    in_to_out_stride_ratio: the ratio of the input stride to the output stride. This will be multipled to output coordinates to find matching input coordinates.
    """
    num_spatial_dims = batch_indexed_in_coords.shape[1] - 1
    if kernel_dilation is None:
        kernel_dilation = (1,) * num_spatial_dims

    assert len(kernel_size) == num_spatial_dims
    assert len(kernel_dilation) == num_spatial_dims
    assert len(in_to_out_stride_ratio) == num_spatial_dims

    num_total_kernels = np.prod(kernel_size)
    if kernel_search_batch_size is None:
        kernel_search_batch_size = num_total_kernels // kernel_size[0]

    # convert to wp array
    device = batch_indexed_in_coords.device
    batch_indexed_in_coords_wp = wp.from_torch(batch_indexed_in_coords)
    # multiply output coordinates by in_to_out_stride_ratio if it is not all ones
    if not all(s == 1 for s in in_to_out_stride_ratio):
        batch_indexed_out_coords = batch_indexed_out_coords * torch.tensor(
            [1, *ntuple(in_to_out_stride_ratio, ndim=num_spatial_dims)],
            dtype=torch.int32,
            device=device,
        )
    else:
        batch_indexed_out_coords = batch_indexed_out_coords
    N_out = batch_indexed_out_coords.shape[0]

    # Create a vector hashtable for the batched coordinates
    hashtable = VectorHashTable.from_keys(batch_indexed_in_coords_wp)

    num_total_kernels = np.prod(kernel_size)

    # Found indices and offsets for each kernel offset
    in_maps = []
    out_maps = []
    num_valid_maps = []

    # Query the hashtable for all kernel offsets
    all_out_indices = (
        torch.arange(N_out, device=device).repeat(kernel_search_batch_size, 1).view(-1)
    )

    # Gerenate kernel offsets
    offsets = kernel_offsets_from_size(
        kernel_size, kernel_dilation, center_offset=kernel_center_offset
    ).to(device)

    # TODO(cchoy): replace the inner loop with the kernel_map_from_offsets
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
        in_indices_wp = hashtable.search(new_batch_indexed_out_coords_wp)
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
    offsets = torch.cat([torch.zeros(1, dtype=torch.int32, device=device), offsets], dim=0)

    return DiscreteNeighborSearchResult(in_maps, out_maps, offsets)
