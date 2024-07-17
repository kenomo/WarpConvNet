import torch
from jaxtyping import Int
from torch import Tensor

import warp as wp
import warp.utils
from warp.convnet.core.hashmap import HashStruct, VectorHashTable, search_func
from warp.convnet.utils.batch_index import batch_indexed_coordinates


@wp.kernel
def conv_kernel_map(
    in_hashmap: HashStruct,
    query_coords: wp.array(dtype=wp.vec4i),
    N_query_coords: int,
    kernel_offsets: wp.array(dtype=wp.vec4i),
    N_kernel_offsets: int,
    found_in_coord_index: wp.array(dtype=int),
):
    """
    Compute whether query + offset is in in_coords and return the index of the found input coordinate.

    For definitions, please refer to Sec. 4.2. of https://arxiv.org/pdf/1904.08755
    """
    idx = wp.tid()
    for k in range(N_kernel_offsets):
        query_coord = query_coords[idx] + kernel_offsets[k]
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
    query_coords: wp.array(dtype=wp.vec4i),
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
                query_coord = query_coords[idx] + wp.vec4i(0, i - center, j - center, k - center)
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
    query_coords: wp.array(dtype=wp.vec4i),
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
                query_coord = query_coords[idx] + wp.vec4i(0, i - center, j - center, k - center)
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
    batched_query_coords: wp.array(dtype=wp.vec4i),
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
    query_bcoords = wp.from_torch(query_bcoords, dtype=wp.vec4i)

    # Create the hashmap for in_coords
    in_coords_hashmap = VectorHashTable.from_keys(in_bcoords)

    # Launch the kernel
    in_coords_index, query_coords_index = neighbor_search_hashmap(
        in_coords_hashmap, query_bcoords, neighbor_distance_threshold
    )

    return in_coords_index, query_coords_index


def kernel_map_hashmap(
    in_hashmap: HashStruct,
    batched_query_coords: Int[Tensor, "M 4"],
    kernel_offsets: Int[Tensor, "K 4"],
) -> Int[Tensor, "K M"]:
    device = in_hashmap.table_kvs.device  # string device from warp array
    assert device == str(
        batched_query_coords.device
    ), f"{device} != {str(batched_query_coords.device)}"
    assert device == str(kernel_offsets.device), f"{device} != {kernel_offsets.device}"

    # Allocate output
    found_in_coord_index_wp = wp.empty(
        len(batched_query_coords) * len(kernel_offsets),
        dtype=wp.int32,
        device=device,
    )

    # Launch the kernel
    wp.launch(
        kernel=conv_kernel_map,
        dim=len(batched_query_coords),
        inputs=[
            in_hashmap,
            wp.from_torch(batched_query_coords, dtype=wp.vec4i),
            len(batched_query_coords),
            wp.from_torch(kernel_offsets, dtype=wp.vec4i),
            len(kernel_offsets),
            found_in_coord_index_wp,
        ],
    )

    found_in_coord_index = wp.to_torch(found_in_coord_index_wp).reshape(
        len(kernel_offsets), len(batched_query_coords)
    )
    return found_in_coord_index


def kernel_map(
    in_coords: Int[Tensor, "N 3"],
    in_coords_offsets: Int[Tensor, "B+1"],  # noqa: F821
    query_coords: Int[Tensor, "M 3"],
    query_coords_offsets: Int[Tensor, "B+1"],  # noqa: F821
    kernel_offsets: Int[Tensor, "K 4"],
):
    # device checks
    assert in_coords.device == query_coords.device

    # Convert the coordinates to batched coordinates
    in_bcoords = batch_indexed_coordinates(in_coords, in_coords_offsets)
    query_bcoords = batch_indexed_coordinates(query_coords, query_coords_offsets)

    # Create the hashmap for in_coords
    in_coords_hashmap = VectorHashTable.from_keys(in_bcoords)

    # Launch the kernel
    found_in_coord_index = kernel_map_hashmap(in_coords_hashmap, query_bcoords, kernel_offsets)

    return found_in_coord_index
