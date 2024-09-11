from enum import Enum
from typing import Literal, Optional, Tuple

import torch
from jaxtyping import Float, Int
from torch import Tensor

import warp as wp


class CONTINUOUS_NEIGHBOR_SEARCH_MODE(Enum):
    RADIUS = "radius"
    KNN = "knn"
    SAME_VOXEL = "same_voxel"


class ContinuousNeighborSearchArgs:
    """
    Wrapper for the input of a neighbor search operation.
    """

    # The mode of the neighbor search
    _mode: CONTINUOUS_NEIGHBOR_SEARCH_MODE
    # The radius for radius search
    _radius: Optional[float]
    # The number of neighbors for knn search
    _k: Optional[int]
    # Grid dim
    _grid_dim: Optional[int | Tuple[int, int, int]]

    def __init__(
        self,
        mode: CONTINUOUS_NEIGHBOR_SEARCH_MODE,
        radius: Optional[float] = None,
        k: Optional[int] = None,
        grid_dim: Optional[int | Tuple[int, int, int]] = None,
    ):
        if isinstance(mode, str):
            mode = CONTINUOUS_NEIGHBOR_SEARCH_MODE(mode)

        self._mode = mode
        self._radius = radius
        self._k = k
        self._grid_dim = grid_dim

    @property
    def mode(self):
        return self._mode

    @property
    def radius(self):
        return self._radius

    @property
    def k(self):
        return self._k

    @property
    def grid_dim(self):
        return self._grid_dim

    def __repr__(self):
        if self._mode == CONTINUOUS_NEIGHBOR_SEARCH_MODE.RADIUS:
            out_str = f"{self._mode.name}({self._radius})"
        elif self._mode == CONTINUOUS_NEIGHBOR_SEARCH_MODE.KNN:
            out_str = f"{self._mode._name}({self._k})"
        elif self._mode == CONTINUOUS_NEIGHBOR_SEARCH_MODE.SAME_VOXEL:
            out_str = f"VOXEL({self._grid_dim})"
        return out_str

    def clone(
        self,
        mode: Optional[CONTINUOUS_NEIGHBOR_SEARCH_MODE] = None,
        radius: Optional[float] = None,
        k: Optional[int] = None,
        grid_dim: Optional[int | Tuple[int, int, int]] = None,
    ):
        return ContinuousNeighborSearchArgs(
            mode=mode if mode is not None else self._mode,
            radius=radius if radius is not None else self._radius,
            k=k if k is not None else self._k,
            grid_dim=grid_dim if grid_dim is not None else self._grid_dim,
        )


class NeighborSearchResult:
    """
    Wrapper for the output of a neighbor search operation.
    """

    # N is the total number of neighbors for all M queries
    _neighbors_index: Int[Tensor, "N"]  # noqa: F821
    # M is the number of queries
    _neighbors_row_splits: Int[Tensor, "M + 1"]  # noqa: F821
    # optional distance
    _neighbors_distance: Optional[Float[Tensor, "N"]]  # noqa: F821

    def __init__(self, *args):
        # If there are two args, assume they are neighbors_index and neighbors_row_splits
        # If there is one arg, assume it is a NeighborSearchReturnType
        if len(args) == 2:
            self._neighbors_index = args[0].long()
            self._neighbors_row_splits = args[1].long()
        elif len(args) == 1:
            # K-nn search result
            assert isinstance(args[0], torch.Tensor)
            # 2D tensor with shape (M, K)
            assert args[0].ndim == 2
            M, K = args[0].shape
            self._neighbors_index = args[0].long()
            self._neighbors_row_splits = torch.arange(
                0, M * K + 1, K, device=args[0].device, dtype=torch.long
            )
        else:
            raise ValueError("NeighborSearchReturn must be initialized with 1 or 2 arguments")

    @property
    def neighbors_index(self):
        return self._neighbors_index

    @property
    def neighbors_row_splits(self):
        return self._neighbors_row_splits

    def to(self, device: str | int | torch.device):
        self._neighbors_index.to(device)
        self._neighbors_row_splits.to(device)
        return self

    def __repr__(self):
        return f"{self.__class__.__name__}(neighbors_index={self._neighbors_index.shape}, neighbors_row_splits={self._neighbors_row_splits.shape})"


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


@torch.no_grad()
def _knn_search(
    ref_positions: Int[Tensor, "N 3"],  # noqa: F821
    query_positions: Int[Tensor, "M 3"],  # noqa: F821
    k: int,
) -> Int[Tensor, "M K"]:  # noqa: F821
    """Perform knn search using the open3d backend."""
    assert k > 0
    assert k < ref_positions.shape[0]
    assert ref_positions.device == query_positions.device
    # Critical for multi GPU
    if ref_positions.is_cuda:
        torch.cuda.set_device(ref_positions.device)
    # Use topk to get the top k indices from distances
    dists = torch.cdist(query_positions, ref_positions)
    _, neighbors_index = torch.topk(dists, k, dim=1, largest=False)
    return neighbors_index


@torch.no_grad()
def _chunked_knn_search(
    ref_positions: Int[Tensor, "N 3"],  # noqa: F821
    query_positions: Int[Tensor, "M 3"],  # noqa: F821
    k: int,
    chunk_size: int = 4096,
):
    """Divide the out_positions into chunks and perform knn search."""
    assert k > 0
    assert k < ref_positions.shape[0]
    assert chunk_size > 0
    neighbors_index = []
    for i in range(0, query_positions.shape[0], chunk_size):
        chunk_out_positions = query_positions[i : i + chunk_size]
        chunk_neighbors_index = _knn_search(ref_positions, chunk_out_positions, k)
        neighbors_index.append(chunk_neighbors_index)
    return torch.concatenate(neighbors_index, dim=0)


def _bvh_knn_search(
    ref_positions: Int[Tensor, "N 3"],  # noqa: F821
    query_positions: Int[Tensor, "M 3"],  # noqa: F821
    k: int,
) -> Int[Tensor, "M K"]:
    """Perform knn search using the open3d backend."""
    assert k > 0
    assert k < ref_positions.shape[0]
    assert ref_positions.device == query_positions.device
    # Compute min/max of ref
    min_ref = wp.from_torch(ref_positions.min(dim=0))
    max_ref = wp.from_torch(ref_positions.max(dim=0))
    # Convert to warp
    ref_positions_wp = wp.from_torch(ref_positions, dtype=wp.vec3)  # noqa: F841
    query_positions_wp = wp.from_torch(query_positions, dtype=wp.vec3)  # noqa: F841
    # Create bvh
    bvh = wp.Bvh(lowers=min_ref, uppers=max_ref, device=str(ref_positions.device))  # noqa: F841
    # TODO
    raise NotImplementedError


@torch.no_grad()
def knn_search(
    ref_positions: Int[Tensor, "N 3"],  # noqa: F821
    query_positions: Int[Tensor, "M 3"],  # noqa: F821
    k: int,
    search_method: Literal["chunk", "bvh"] = "chunk",  # noqa: F821
    chunk_size: int = 32768,  # 2^15
) -> Int[Tensor, "M K"]:
    """
    ref_positions: [N,3]
    query_positions: [M,3]
    k: int
    """
    assert 0 < k < ref_positions.shape[0]
    assert search_method in ["chunk"]
    # Critical for multi GPU
    if ref_positions.is_cuda:
        torch.cuda.set_device(ref_positions.device)
    assert ref_positions.device == query_positions.device
    if search_method == "chunk":
        if query_positions.shape[0] < chunk_size:
            neighbors_index = _knn_search(ref_positions, query_positions, k)
        else:
            neighbors_index = _chunked_knn_search(
                ref_positions, query_positions, k, chunk_size=chunk_size
            )
    else:
        raise ValueError(f"search_method {search_method} not supported.")
    return neighbors_index


@torch.no_grad()
def batched_knn_search(
    ref_positions: Int[Tensor, "N 3"],  # noqa: F821
    ref_offsets: Int[Tensor, "B + 1"],  # noqa: F821
    query_positions: Int[Tensor, "M 3"],  # noqa: F821
    query_offsets: Int[Tensor, "B + 1"],  # noqa: F821
    k: int,
    search_method: Literal["chunk", "bvh"] = "chunk",  # noqa: F821
    chunk_size: int = 4096,
) -> Int[Tensor, "MK"]:  # noqa: F821
    """
    ref_positions: [N,3]
    query_positions: [M,3]
    k: int
    """
    assert (
        ref_positions.shape[0] == query_positions.shape[0]
    ), f"Batch size mismatch, {ref_positions.shape[0]} != {query_positions.shape[0]}"
    neighbors = []
    B = len(ref_offsets) - 1
    for b in range(B):
        neighbor_index = knn_search(
            ref_positions[ref_offsets[b] : ref_offsets[b + 1],],
            query_positions[query_offsets[b] : query_offsets[b + 1],],
            k,
            search_method,
            chunk_size,
        )
        neighbors.append(neighbor_index + ref_offsets[b])
    return torch.cat(neighbors, dim=0).long()


def neighbor_search(
    ref_positions: Float[Tensor, "N 3"],  # noqa: F821
    ref_offsets: Int[Tensor, "B + 1"],  # noqa: F821
    query_positions: Float[Tensor, "M 3"],  # noqa: F821
    query_offsets: Int[Tensor, "B + 1"],  # noqa: F821
    search_args: ContinuousNeighborSearchArgs,
) -> NeighborSearchResult:
    """
    Args:
        ref_coords: BatchedCoordinates
        query_coords: BatchedCoordinates
        search_args: NeighborSearchArgs
        grid_dim: Union[int, Tuple[int, int, int]]

    Returns:
        NeighborSearchReturn
    """
    if search_args.mode == CONTINUOUS_NEIGHBOR_SEARCH_MODE.RADIUS:
        assert search_args.radius is not None, "Radius must be provided for radius search"
        neighbor_index, neighbor_distance, neighbor_split = batched_radius_search(
            ref_positions=ref_positions,
            ref_offsets=ref_offsets,
            query_positions=query_positions,
            query_offsets=query_offsets,
            radius=search_args.radius,
            grid_dim=search_args.grid_dim,
        )
        return NeighborSearchResult(
            neighbor_index,
            neighbor_split,
        )

    elif search_args.mode == CONTINUOUS_NEIGHBOR_SEARCH_MODE.KNN:
        assert search_args.k is not None, "knn_k must be provided for knn search"
        # M x K
        neighbor_index = batched_knn_search(
            ref_positions=ref_positions,
            ref_offsets=ref_offsets,
            query_positions=query_positions,
            query_offsets=query_offsets,
            k=search_args.k,
        )
        return NeighborSearchResult(neighbor_index)

    elif search_args.mode == CONTINUOUS_NEIGHBOR_SEARCH_MODE.SAME_VOXEL:
        raise NotImplementedError("Grid search not implemented yet")
