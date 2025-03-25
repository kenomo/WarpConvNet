# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from jaxtyping import Float, Int
from typing import Optional, Tuple

import torch
from torch import Tensor
import warp as wp


@wp.kernel
def _radius_search_count(
    hashgrid: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    queries: wp.array(dtype=wp.vec3),
    result_count: wp.array(dtype=wp.int32),
    radius: wp.float32,
):
    tid = wp.tid()

    # create grid query around point
    qp = queries[tid]
    query = wp.hash_grid_query(hashgrid, qp, radius)
    index = int(0)
    result_count_tid = int(0)

    while wp.hash_grid_query_next(query, index):
        neighbor = points[index]

        # compute distance to neighbor point
        dist = wp.length(qp - neighbor)
        if dist <= radius:
            result_count_tid += 1

    result_count[tid] = result_count_tid


@wp.kernel
def _radius_search_query(
    hashgrid: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    queries: wp.array(dtype=wp.vec3),
    result_offset: wp.array(dtype=wp.int32),
    result_point_idx: wp.array(dtype=wp.int32),
    result_point_dist: wp.array(dtype=wp.float32),
    radius: wp.float32,
):
    tid = wp.tid()

    # create grid query around point
    qp = queries[tid]
    query = wp.hash_grid_query(hashgrid, qp, radius)
    index = int(0)
    result_count = int(0)
    offset_tid = result_offset[tid]

    while wp.hash_grid_query_next(query, index):
        neighbor = points[index]

        # compute distance to neighbor point
        dist = wp.length(qp - neighbor)
        if dist <= radius:
            result_point_idx[offset_tid + result_count] = index
            result_point_dist[offset_tid + result_count] = dist
            result_count += 1


def _radius_search(
    points: wp.array(dtype=wp.vec3),
    queries: wp.array(dtype=wp.vec3),
    radius: float,
    grid_dim: Optional[int | Tuple[int, int, int]] = None,
):
    if grid_dim is None:
        grid_dim = 128

    # convert grid_dim to Tuple if it is int
    if isinstance(grid_dim, int):
        grid_dim = (grid_dim, grid_dim, grid_dim)

    str_device = str(points.device)
    result_count = wp.zeros(shape=len(queries), dtype=wp.int32, device=str_device)
    grid = wp.HashGrid(
        dim_x=grid_dim[0],
        dim_y=grid_dim[1],
        dim_z=grid_dim[2],
        device=str_device,
    )
    grid.build(points=points, radius=2 * radius)

    # For 10M radius search, the result can overflow and fail
    wp.launch(
        kernel=_radius_search_count,
        dim=len(queries),
        inputs=[grid.id, points, queries, result_count, radius],
        device=str_device,
    )

    torch_offset = torch.zeros(len(result_count) + 1, device=str_device, dtype=torch.int32)
    result_count_torch = wp.to_torch(result_count)
    torch.cumsum(result_count_torch, dim=0, out=torch_offset[1:])
    total_count = torch_offset[-1].item()
    assert (
        0 < total_count and total_count < 2**31 - 1
    ), f"Invalid total count: {total_count}. Must be between 0 and 2**31 - 1"

    result_point_idx = wp.zeros(shape=(total_count,), dtype=wp.int32, device=str_device)
    result_point_dist = wp.zeros(shape=(total_count,), dtype=wp.float32, device=str_device)

    wp.launch(
        kernel=_radius_search_query,
        dim=len(queries),
        inputs=[
            grid.id,
            points,
            queries,
            wp.from_torch(torch_offset),
            result_point_idx,
            result_point_dist,
            radius,
        ],
        device=str_device,
    )

    return (result_point_idx, result_point_dist, torch_offset)


def radius_search(
    points: Float[Tensor, "N 3"],  # noqa: F821
    queries: Float[Tensor, "M 3"],  # noqa: F821
    radius: float,
    grid_dim: Optional[int | Tuple[int, int, int]] = None,
) -> Tuple[Float[Tensor, "Q"], Float[Tensor, "Q"], Float[Tensor, "M + 1"]]:  # noqa: F821
    """
    Args:
        points: [N, 3]
        queries: [M, 3]
        radius: float
        grid_dim: Union[int, Tuple[int, int, int]]
        device: str

    Returns:
        neighbor_index: [Q]
        neighbor_distance: [Q]
        neighbor_split: [M + 1]

    Warnings:
        The HashGrid supports a maximum of 4096^3 grid cells. The users must
        ensure that the points are bounded and 2 * radius * 4096 < max_bound.
    """
    # Convert from warp to torch
    assert points.is_contiguous(), "points must be contiguous"
    assert queries.is_contiguous(), "queries must be contiguous"
    points_wp = wp.from_torch(points, dtype=wp.vec3)
    queries_wp = wp.from_torch(queries, dtype=wp.vec3)

    result_point_idx, result_point_dist, torch_offset = _radius_search(
        points=points_wp,
        queries=queries_wp,
        radius=radius,
        grid_dim=grid_dim,
    )

    # Convert from warp to torch
    result_point_idx = wp.to_torch(result_point_idx)
    result_point_dist = wp.to_torch(result_point_dist)

    # Neighbor index, Neighbor Distance, Neighbor Split
    return result_point_idx, result_point_dist, torch_offset


def batched_radius_search(
    ref_positions: Float[Tensor, "N 3"],  # noqa: F821
    ref_offsets: Int[Tensor, "B + 1"],  # noqa: F821
    query_positions: Float[Tensor, "M 3"],  # noqa: F821
    query_offsets: Int[Tensor, "B + 1"],  # noqa: F821
    radius: float,
    grid_dim: Optional[int | Tuple[int, int, int]] = None,
) -> Tuple[Int[Tensor, "Q"], Float[Tensor, "Q"], Int[Tensor, "M + 1"]]:  # noqa: F821
    """
    Args:
        ref_positions: [N, 3]
        ref_offsets: [B + 1]
        query_positions: [M, 3]
        query_offsets: [B + 1]
        radius: float
        grid_dim: Union[int, Tuple[int, int, int]]

    Returns:
        neighbor_index: [Q]
        neighbor_distance: [Q]
        neighbor_split: [B + 1]
    """
    B = len(ref_offsets) - 1
    assert B == len(query_offsets) - 1
    assert (
        ref_offsets[-1] == ref_positions.shape[0]
    ), f"Last offset {ref_offsets[-1]} != {ref_positions.shape[0]}"
    assert (
        query_offsets[-1] == query_positions.shape[0]
    ), f"Last offset {query_offsets[-1]} != {query_positions.shape[0]}"
    neighbor_index_list = []
    neighbor_distance_list = []
    neighbor_split_list = []
    split_offset = 0
    # TODO(cchoy): optional parallelization for small point clouds
    for b in range(B):
        neighbor_index, neighbor_distance, neighbor_split = radius_search(
            points=ref_positions[ref_offsets[b] : ref_offsets[b + 1]],
            queries=query_positions[query_offsets[b] : query_offsets[b + 1]],
            radius=radius,
            grid_dim=grid_dim,
        )
        neighbor_index_list.append(neighbor_index + ref_offsets[b])
        neighbor_distance_list.append(neighbor_distance)
        # if b is last, append all neighbor_split since the last element is the total count
        if b == B - 1:
            neighbor_split_list.append(neighbor_split + split_offset)
        else:
            neighbor_split_list.append(neighbor_split[:-1] + split_offset)

        split_offset += len(neighbor_index)

    # Neighbor index, Neighbor Distance, Neighbor Split
    return (
        torch.cat(neighbor_index_list).long(),
        torch.cat(neighbor_distance_list),
        torch.cat(neighbor_split_list).long(),
    )
