import torch

import warp as wp
from warp.convnet.core.hashmap import HashStruct, VectorHashTable, search_func
from warp.convnet.utils.batch_index import batch_indexed_coordinates


@wp.kernel
def conv_kernel_map(
    in_hashmap: HashStruct,
    query_coords: wp.array(dtype=wp.vec4i),
    kernel_offset: wp.vec4i,
    found_in_coord_index: wp.array(dtype=int),
):
    """
    Compute whether query + offset is in in_coords and return the index of the found input coordinate.

    For definitions, please refer to Sec. 4.2. of https://arxiv.org/pdf/1904.08755
    """
    idx = wp.tid()
    query_coord = query_coords[idx] + kernel_offset
    index = search_func(
        in_hashmap.table_kvs,
        in_hashmap.vector_keys,
        query_coord,
        in_hashmap.capacity,
        in_hashmap.hash_method,
    )
    found_in_coord_index[idx] = index


@wp.kernel
def num_neighbors_kernel(
    in_hashmap: HashStruct,
    query_coords: wp.array(dtype=wp.vec4i),
    neighbor_distance_threshold: int,
    num_neighbors: wp.array(dtype=int),
):
    idx = wp.tid()

    curr_num_neighbors = int(0)
    # Loop over the neighbors
    for i in range(neighbor_distance_threshold):
        for j in range(neighbor_distance_threshold):
            for k in range(neighbor_distance_threshold):
                # Compute query coord
                query_coord = query_coords[idx] + wp.vec4i(0, i, j, k)
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


def neighbor_search(
    in_coords: torch.Tensor,
    in_coords_offsets: torch.Tensor,
    query_coords: torch.Tensor,
    query_coords_offsets: torch.Tensor,
    neighbor_distance_threshold: int,
):
    # device checks
    str_device = str(in_coords.device)
    assert in_coords.device == query_coords.device

    # Convert the coordinates to batched coordinates
    in_bcoords = batch_indexed_coordinates(in_coords, in_coords_offsets)
    query_bcoords = batch_indexed_coordinates(query_coords, query_coords_offsets)

    # Create the hashmap for in_coords
    in_coords_hashmap = VectorHashTable.from_keys(in_bcoords)

    # Compute the number of neighbors for each query point
    num_neighbors = wp.empty(len(query_bcoords), dtype=int, device=str_device)

    # Launch num neighbor kernel
    wp.launch(
        kernel=num_neighbors_kernel,
        dim=len(query_bcoords),
        inputs=[
            in_coords_hashmap.table_kv,
            in_coords_hashmap.vector_keys,
            query_bcoords,
            num_neighbors,
        ],
    )
