from typing import Optional, Tuple

import torch

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
    assert total_count < 2**31 - 1, f"Total result count is too large: {total_count} > 2**31 - 1"

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
