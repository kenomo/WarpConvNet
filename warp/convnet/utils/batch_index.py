import torch
import torch.bin
from jaxtyping import Float, Int
from torch import Tensor

import warp as wp


@wp.func
def _find_bin(offsets: wp.array(dtype=wp.int32), tid: int) -> int:
    batch_id = int(0)
    N = offsets.shape[0] - 1
    for i in range(1, N):
        if tid >= offsets[i]:
            batch_id += 1
        else:
            break
    return batch_id


@wp.kernel
def _batch_index(
    offsets: wp.array(dtype=wp.int32),
    batch_index_wp: wp.array(dtype=wp.int32),
) -> None:
    tid = wp.tid()
    batch_index_wp[tid] = _find_bin(offsets, tid)


def batch_index_from_offset(
    offsets: Int[Tensor, "B+1"], device: str  # noqa: F821
) -> Int[Tensor, "N"]:  # noqa: F821
    N = offsets[-1].item() if isinstance(offsets, Tensor) else offsets[-1]
    if device == "cpu":
        batch_index = torch.zeros(N, dtype=torch.int32)
        for i in range(1, len(offsets) - 1):
            batch_index[offsets[i] : offsets[i + 1]] = i
        return batch_index

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
) -> Float[Tensor, "N 4"]:  # noqa: F821
    device = str(batched_coords.device)
    batch_index = batch_index_from_offset(offsets, device)
    batched_coords = torch.cat([batch_index.unsqueeze(1), batched_coords], dim=1)
    return batched_coords
