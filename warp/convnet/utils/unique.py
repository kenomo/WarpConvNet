import torch


def unique(x: torch.Tensor, dim: int = 0, stable: bool = False):
    """
    Get unique elements along a dimension.

    Args:
        x: Tensor
        dim: int
        stable: bool

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

    from https://github.com/pytorch/pytorch/issues/36748
    """
    unique, to_orig_indices, counts = torch.unique(
        x, dim=dim, sorted=True, return_inverse=True, return_counts=True
    )
    all_to_unique_indices = to_orig_indices.argsort(stable=stable)
    all_to_unique_offsets = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))

    dtype_ind, device = to_orig_indices.dtype, to_orig_indices.device
    perm = torch.arange(x.size(dim), dtype=dtype_ind, device=device)
    perm = torch.empty(unique.size(dim), dtype=dtype_ind, device=device).scatter_(
        dim, to_orig_indices, perm
    )

    return (
        unique,
        to_orig_indices,
        all_to_unique_indices,
        all_to_unique_offsets,
        perm,
    )
