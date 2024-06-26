import warnings
from typing import Literal, Optional

import torch
import torch.bin
from jaxtyping import Float, Int
from torch import Tensor

import warp as wp

snippet = """
    __shared__ int shared_offsets[128];

    int curr_tid = threadIdx.x;

    // Load offsets into shared memory
    if (tid < offsets_len) {
        shared_offsets[tid] = offsets[tid];
    }
    __syncthreads();

    // Find bin
    int bin = -1;
    for (int i = 1; i < offsets_len; i++) {
        int start = shared_offsets[i - 1];
        int end = shared_offsets[i];
        if (start <= tid && tid < end) {
            bin = i;
            break;
        }
    }

    batch_index_wp[curr_tid] = bin;
    """


@wp.func_native(snippet)
def _find_bin_native(
    offsets: wp.array(dtype=wp.int32),
    offsets_len: int,
    tid: int,
    batch_index_wp: wp.array(dtype=wp.int32),
):
    ...


@wp.func
def _find_bin(offsets: wp.array(dtype=wp.int32), tid: int) -> int:
    N = offsets.shape[0] - 1
    for i in range(1, N):
        start = offsets[i - 1]
        end = offsets[i]
        if start <= tid < end:
            return i
    return -1


@wp.kernel
def _batch_index(
    offsets: wp.array(dtype=wp.int32),
    batch_index_wp: wp.array(dtype=wp.int32),
) -> None:
    tid = wp.tid()
    if offsets.shape[0] > 128:
        batch_index_wp[tid] = _find_bin(offsets, tid)
    else:
        _find_bin_native(offsets, offsets.shape[0], tid, batch_index_wp)


def batch_index_from_offset(
    offsets: Int[Tensor, "B+1"],  # noqa: F821
    backend: Literal["torch", "warp"] = "torch",
    device: Optional[str] = None,
) -> Int[Tensor, "N"]:  # noqa: F821
    assert isinstance(offsets, torch.Tensor), "offsets must be a torch.Tensor"

    if device is not None:
        offsets = offsets.to(device)

    if backend == "torch":
        batch_index = (
            torch.arange(len(offsets) - 1)
            .to(offsets)
            .repeat_interleave(offsets[1:] - offsets[:-1])
        )
        return batch_index

    warnings.warn(
        "Using Warp backend for batch_index_from_offset. This may be slower.", stacklevel=2
    )
    # cchoy: Probably slower due to copying back and forth between warp and torch
    N = offsets[-1].item()
    offsets_wp = wp.from_torch(offsets, dtype=wp.int32).to(device)
    batch_index_wp = wp.zeros(shape=(N,), dtype=wp.int32, device=device)
    wp.launch(
        _batch_index,
        N,
        inputs=[offsets_wp, batch_index_wp],
    )
    return wp.to_torch(batch_index_wp)


def batch_indexed_coordinates(
    batched_coords: Float[Tensor, "N 3"],  # noqa: F821
    offsets: Int[Tensor, "B + 1"],  # noqa: F821
    backend: Literal["torch", "warp"] = "torch",
) -> Float[Tensor, "N 4"]:  # noqa: F821
    device = str(batched_coords.device)
    batch_index = batch_index_from_offset(offsets, device=device, backend=backend)
    batched_coords = torch.cat([batch_index.unsqueeze(1), batched_coords], dim=1)
    return batched_coords
