from typing import Literal, Optional

import numpy as np
import torch
import torch.bin
from jaxtyping import Float, Int
from torch import Tensor

import warp as wp

snippet = """
    __shared__ int shared_offsets[256];

    int block_tid = threadIdx.x;

    // Load offsets into shared memory.
    // Make sure that the last block loads the full offsets.
    if (block_tid < offsets_len) {
        shared_offsets[block_tid] = offsets[block_tid];
    }
    __syncthreads();

    if (tid < batch_index_len) {
        // Find bin
        int bin = -1;
        for (int i = 0; i < offsets_len - 1; i++) {
            int start = shared_offsets[i];
            int end = shared_offsets[i + 1];
            if (start <= tid && tid < end) {
                bin = i;
                break;
            }
        }

        batch_index_wp[tid] = bin;
    }
    """


@wp.func_native(snippet)
def _find_bin_native(
    offsets: wp.array(dtype=wp.int32),
    offsets_len: int,
    tid: int,
    batch_index_wp: wp.array(dtype=wp.int32),
    batch_index_len: int,
):
    ...


@wp.func
def _find_bin(offsets: wp.array(dtype=wp.int32), tid: int) -> int:
    N = offsets.shape[0] - 1
    bin_id = int(-1)
    for i in range(N):
        start = offsets[i]
        end = offsets[i + 1]
        if start <= tid < end:
            bin_id = i
            break
    return bin_id


@wp.kernel
def _batch_index(
    offsets: wp.array(dtype=wp.int32),
    batch_index_wp: wp.array(dtype=wp.int32),
) -> None:
    tid = wp.tid()
    if offsets.shape[0] > 256:
        batch_index_wp[tid] = _find_bin(offsets, tid)
    else:
        _find_bin_native(offsets, offsets.shape[0], tid, batch_index_wp, batch_index_wp.shape[0])


def batch_index_from_offset(
    offsets: Int[Tensor, "B+1"],  # noqa: F821
    device: Optional[str] = None,
    backend: Literal["torch", "warp"] = "warp",
) -> Int[Tensor, "N"]:  # noqa: F821
    assert isinstance(offsets, torch.Tensor), "offsets must be a torch.Tensor"
    assert backend in ["torch", "warp"], "backend must be either torch or warp"

    # offset to int
    offsets = offsets.int()

    if device is not None:
        offsets = offsets.to(device)

    if backend == "torch":
        batch_index = (
            torch.arange(len(offsets) - 1)
            .to(offsets)
            .repeat_interleave(offsets[1:] - offsets[:-1])
        )
        return batch_index

    if device is None:
        device = str(offsets.device)

    # Assert this is not cpu
    assert "cpu" not in device, "device must be a cuda device"

    N = offsets[-1].item()
    offsets_wp = wp.from_torch(offsets.int(), dtype=wp.int32).to(device)
    batch_index_wp = wp.zeros(shape=(N,), dtype=wp.int32, device=device)
    wp.launch(
        _batch_index,
        int(np.ceil(N / 256.0)) * 256,
        inputs=[offsets_wp, batch_index_wp],
        device=device,
    )
    return wp.to_torch(batch_index_wp)


def batch_indexed_coordinates(
    batched_coords: Float[Tensor, "N 3"],  # noqa: F821
    offsets: Int[Tensor, "B + 1"],  # noqa: F821
    backend: Literal["torch", "warp"] = "warp",
    return_type: Literal["torch", "warp"] = "torch",
) -> Float[Tensor, "N 4"]:  # noqa: F821
    device = str(batched_coords.device)
    batch_index = batch_index_from_offset(offsets, device=device, backend=backend)
    batched_coords = torch.cat([batch_index.unsqueeze(1), batched_coords], dim=1)
    if return_type == "torch":
        return batched_coords
    elif return_type == "warp":
        return wp.from_torch(batched_coords, dtype=wp.vec4i)
    else:
        raise ValueError("return_type must be either torch or warp")


def offsets_from_batch_index(
    batch_index: Int[Tensor, "N"],  # noqa: F821
    backend: Literal["torch", "warp"] = "torch",
) -> Int[Tensor, "B + 1"]:  # noqa: F821
    """
    Given a list of batch indices [0, 0, 1, 1, 2, 2, 2, 3, 3],
    return the offsets [0, 2, 4, 7, 9].
    """
    if backend == "torch":
        # Get unique elements
        _, counts = torch.unique(batch_index, return_counts=True)
        counts = counts.cpu()
        # Get the offsets by cumsum
        offsets = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32),
                counts.cumsum(dim=0),
            ],
            dim=0,
        ).to(batch_index.device)
        return offsets
    elif backend == "warp":
        raise NotImplementedError("warp backend not implemented")
    else:
        raise ValueError("backend must be torch")
