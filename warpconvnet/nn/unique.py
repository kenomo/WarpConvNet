from dataclasses import dataclass
from typing import Literal, Optional, Tuple

from jaxtyping import Int
from torch import Tensor

from warpconvnet.geometry.base_geometry import SpatialFeatures
from warpconvnet.utils.ravel import ravel_multi_index
from warpconvnet.utils.unique import unique_torch


@dataclass
class UniqueInfo:
    to_orig_indices: Int[Tensor, "N"]  # noqa: F821
    to_csr_indices: Int[Tensor, "N"]  # noqa: F821
    to_csr_offsets: Int[Tensor, "M+1"]  # noqa: F821
    to_unique_indices: Optional[Int[Tensor, "M"]]  # noqa: F821


class ToUnique:
    unique_info: UniqueInfo

    def __init__(
        self,
        unique_method: Literal["torch", "ravel"] = "torch",
        return_to_unique_indices: bool = False,
    ):
        # Ravel can be used only when the raveled coordinates is less than 2**31
        self.unique_method = unique_method
        self.return_to_unique_indices = return_to_unique_indices or unique_method == "ravel"

    def to_unique(self, x: Int[Tensor, "N C"], dim: int = 0) -> Int[Tensor, "M C"]:
        if self.unique_method == "ravel":
            min_coords = x.min(dim=dim).values
            shifted_x = x - min_coords
            shape = shifted_x.max(dim=dim).values + 1
            unique_input = ravel_multi_index(shifted_x, shape)
        else:
            unique_input = x

        (
            unique,
            to_orig_indices,
            all_to_unique_indices,
            all_to_unique_offsets,
            to_unique_indices,
        ) = unique_torch(
            unique_input,
            dim=dim,
            stable=True,
            return_to_unique_indices=self.return_to_unique_indices,
        )
        self.unique_info = UniqueInfo(
            to_orig_indices=to_orig_indices,
            to_csr_indices=all_to_unique_indices,
            to_csr_offsets=all_to_unique_offsets,
            to_unique_indices=to_unique_indices,
        )
        if self.unique_method == "ravel":
            return x[self.unique_info.to_unique_indices]
        return unique

    def to_unique_csr(
        self, x: Int[Tensor, "N C"], dim: int = 0
    ) -> Tuple[Int[Tensor, "M C"], Int[Tensor, "N"], Int[Tensor, "M+1"]]:  # noqa: F821
        """
        Convert the the tensor to a unique tensor and return the indices and offsets that can be used for reduction.

        Returns:
            unique: M unique coordinates
            to_csr_indices: N indices to unique coordinates. x[to_csr_indices] == torch.repeat_interleave(unique, counts).
            to_csr_offsets: M+1 offsets to unique coordinates. counts = to_csr_offsets.diff()
        """
        unique = self.to_unique(x, dim=dim)
        return unique, self.unique_info.to_csr_indices, self.unique_info.to_csr_offsets

    def to_original(self, unique: Int[Tensor, "M C"]) -> Int[Tensor, "N C"]:
        return unique[self.unique_info.to_orig_indices]

    @property
    def to_unique_indices(self) -> Int[Tensor, "M"]:  # noqa: F821
        return self.unique_info.to_unique_indices

    @property
    def to_csr_indices(self) -> Int[Tensor, "N"]:  # noqa: F821
        return self.unique_info.to_csr_indices

    @property
    def to_csr_offsets(self) -> Int[Tensor, "M+1"]:  # noqa: F821
        return self.unique_info.to_csr_offsets

    @property
    def to_orig_indices(self) -> Int[Tensor, "N"]:  # noqa: F821
        return self.unique_info.to_orig_indices
