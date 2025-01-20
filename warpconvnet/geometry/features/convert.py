from typing import Optional, TYPE_CHECKING

from warpconvnet.ops.batch_copy import cat_to_pad_tensor, pad_to_cat_tensor

if TYPE_CHECKING:
    from .cat import CatFeatures
    from .pad import PadFeatures


def cat_to_pad(cat_features: "CatFeatures", pad_multiple: Optional[int] = None) -> "PadFeatures":
    """Convert concatenated features to padded format."""
    from .pad import PadFeatures

    batched_tensor = cat_to_pad_tensor(
        cat_features.batched_tensor,
        cat_features.offsets,
        backend="torch",
        pad_multiple=pad_multiple,
    )
    return PadFeatures(batched_tensor, cat_features.offsets, pad_multiple)


def pad_to_cat(pad_features: "PadFeatures") -> "CatFeatures":
    """Convert padded features to concatenated format."""
    from .cat import CatFeatures

    batched_tensor = pad_to_cat_tensor(pad_features.batched_tensor, pad_features.offsets)
    return CatFeatures(batched_tensor, pad_features.offsets)
