from typing import Any, Optional

import numpy as np
import torch
import warp as wp
from jaxtyping import Float, Int
from torch import Tensor

from warpconvnet.utils.batch_index import _find_bin

snippet = """
    __shared__ int shared_row_splits[256];

    int block_tid = threadIdx.x;
    int grid_tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load row_splits into shared memory
    if (block_tid < row_splits_len) {
        shared_row_splits[block_tid] = row_splits[block_tid];
    }
    __syncthreads();

    for (int j = 0; j < num_copy; j++) {
        int idx = grid_tid * num_copy + j;
        int in_row_idx = idx / C;
        int col_idx = idx % C;

        if (in_row_idx < in_features_rows && col_idx < C) {
            // Find bin using shared memory
            int out_batch_idx = -1;
            for (int i = 0; i < row_splits_len - 1; i++) {
                if (shared_row_splits[i] <= in_row_idx && in_row_idx < shared_row_splits[i + 1]) {
                    out_batch_idx = i;
                    break;
                }
            }

            int out_row_idx = in_row_idx - shared_row_splits[out_batch_idx];
            out_features[out_batch_idx * num_out_features_rows * C + out_row_idx * C + col_idx] =
                in_features[in_row_idx * C + col_idx];
        }
    }
"""


@wp.func_native(snippet)
def _copy_batch_kernel2_native(
    out_features: wp.array3d(dtype=Any),
    in_features: wp.array2d(dtype=Any),
    row_splits: wp.array(dtype=wp.int32),
    row_splits_len: int,
    in_features_rows: int,
    num_out_features_rows: int,
    C: int,
    num_copy: int,
): ...


@wp.kernel
def copy_batch_kernel2_native(
    out_features: wp.array3d(dtype=Any),
    in_features: wp.array2d(dtype=Any),
    row_splits: wp.array(dtype=wp.int32),
    num_copy: int,
):
    _copy_batch_kernel2_native(
        out_features,  # BxMxC
        in_features,  # NxC
        row_splits,  # B+1
        row_splits.shape[0],  # B+1
        in_features.shape[0],  # N
        out_features.shape[1],  # BxMxC
        in_features.shape[1],  # C
        num_copy,
    )


# Copy features from a batched tensor with NxC to BxMxC.
@wp.kernel
def copy_batch_kernel(
    out_features: wp.array3d(dtype=Any),  # mutable
    in_features: wp.array2d(dtype=Any),  # constant
    row_splits: wp.array(dtype=wp.int32),  # constant
):
    i = wp.tid()
    batch_idx = _find_bin(row_splits, i)
    row_idx = i - row_splits[batch_idx]
    for c in range(in_features.shape[1]):
        out_features[batch_idx, row_idx, c] = in_features[i, c]


def copy_batch_warp(
    in_features: Float[Tensor, "N F"],
    row_splits: Int[Tensor, "B+1"],  # noqa: F821
    num_copy_per_thread: Optional[int] = None,  # constant
    pad_multiple: Optional[int] = None,
) -> Float[Tensor, "B M F"]:
    num_points = row_splits.diff()
    batch_size = row_splits.shape[0] - 1
    device = str(in_features.device)
    out_num_points = (
        num_points.max()
        if pad_multiple is None
        else ((num_points.max() + pad_multiple - 1) // pad_multiple) * pad_multiple
    )
    out_features_wp = wp.zeros(
        (batch_size, out_num_points, in_features.shape[1]),
        device=device,
    )
    in_features_wp = wp.from_torch(in_features)
    row_splits_wp = wp.from_torch(row_splits.to(device))
    if num_copy_per_thread is None:
        wp.launch(
            copy_batch_kernel,
            dim=in_features.shape[0],
            inputs=[out_features_wp, in_features_wp, row_splits_wp],
        )
    else:
        wp.launch(
            copy_batch_kernel2_native,
            dim=int(np.ceil(in_features.numel() / num_copy_per_thread)),
            inputs=[
                out_features_wp,
                in_features_wp,
                row_splits_wp,
                num_copy_per_thread,
            ],
        )
    return wp.to_torch(out_features_wp)


def copy_batch_torch(
    in_features: Float[Tensor, "N F"],
    row_splits: Int[Tensor, "B+1"],  # noqa: F821
    pad_multiple: Optional[int] = None,
) -> Float[Tensor, "B M F"]:
    num_points = row_splits.diff()
    device = in_features.device
    out_num_points = (
        num_points.max()
        if pad_multiple is None
        else ((num_points.max() + pad_multiple - 1) // pad_multiple) * pad_multiple
    )
    out_features = torch.zeros(
        (row_splits.shape[0] - 1, out_num_points, in_features.shape[1]),
        dtype=in_features.dtype,
        device=device,
    )
    for batch_idx in range(row_splits.shape[0] - 1):
        out_features[batch_idx, : num_points[batch_idx]] = in_features[
            row_splits[batch_idx] : row_splits[batch_idx + 1]
        ]
    return out_features
