# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Literal
import torch
from jaxtyping import Int
from torch import Tensor


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
    _, neighbor_indices = torch.topk(dists, k, dim=1, largest=False)
    return neighbor_indices


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
    neighbor_indices = []
    for i in range(0, query_positions.shape[0], chunk_size):
        chunk_out_positions = query_positions[i : i + chunk_size]
        chunk_neighbor_indices = _knn_search(ref_positions, chunk_out_positions, k)
        neighbor_indices.append(chunk_neighbor_indices)
    return torch.concatenate(neighbor_indices, dim=0)


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
    assert (
        0 < k < ref_positions.shape[0]
    ), f"k must be greater than 0 and less than the number of reference points. K: {k}, ref_positions.shape[0]: {ref_positions.shape[0]}"
    assert search_method in ["chunk"]
    # Critical for multi GPU
    if ref_positions.is_cuda:
        torch.cuda.set_device(ref_positions.device)
    assert ref_positions.device == query_positions.device
    if search_method == "chunk":
        if query_positions.shape[0] < chunk_size:
            neighbor_indices = _knn_search(ref_positions, query_positions, k)
        else:
            neighbor_indices = _chunked_knn_search(
                ref_positions, query_positions, k, chunk_size=chunk_size
            )
    else:
        raise ValueError(f"search_method {search_method} not supported.")
    return neighbor_indices


@torch.no_grad()
def batch_list_knn_search(
    ref_positions: List[Int[Tensor, "N 3"]],
    query_positions: List[Int[Tensor, "M 3"]],
    k: int,
    search_method: Literal["chunk", "bvh"] = "chunk",  # noqa: F821
    chunk_size: int = 4096,
) -> List[Int[Tensor, "M K"]]:
    """
    ref_positions: List[Tensor[N, 3]]
    query_positions: List[Tensor[M, 3]]
    k: int
    """
    neighbors = []
    for ref_pos, query_pos in zip(ref_positions, query_positions):
        neighbor_index = knn_search(
            ref_pos,
            query_pos,
            k,
            search_method,
            chunk_size,
        )
        neighbors.append(neighbor_index)
    return neighbors


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
    neighbors = []
    # TODO(cchoy): warp kernel
    B = len(ref_offsets) - 1
    N_ref = len(ref_positions)
    N_query = len(query_positions)
    for b in range(B):
        assert (
            N_ref > ref_offsets[b]
        ), f"Invalid reference offsets: {ref_offsets}. Reference point index: {b}, ref_offsets[b]: {ref_offsets[b]}, N_ref: {N_ref}"
        assert (
            N_query > query_offsets[b]
        ), f"Invalid query offsets: {query_offsets}. Query point index: {b}, query_offsets[b]: {query_offsets[b]}, N_query: {N_query}"
        neighbor_index = knn_search(
            ref_positions[ref_offsets[b] : ref_offsets[b + 1],],
            query_positions[query_offsets[b] : query_offsets[b + 1],],
            k,
            search_method,
            chunk_size,
        )
        neighbors.append(neighbor_index + ref_offsets[b])
    return torch.cat(neighbors, dim=0).long()
