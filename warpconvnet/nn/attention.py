from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

from warpconvnet.geometry.base_geometry import SpatialFeatures
from warpconvnet.geometry.ops.neighbor_search_continuous import batched_knn_search
from warpconvnet.geometry.ops.warp_sort import POINT_ORDERING
from warpconvnet.geometry.point_collection import PointCollection
from warpconvnet.nn.base_module import BaseSpatialModule
from warpconvnet.nn.encodings import SinusoidalEncoding
from warpconvnet.nn.normalizations import _RMSNorm as RMSNorm
from warpconvnet.ops.batch_copy import cat_to_pad, pad_to_cat


def zero_out_points(
    x: Float[Tensor, "B N C"], num_points: Int[Tensor, "B"]  # noqa: F821
) -> Float[Tensor, "B N C"]:  # noqa: F821
    """
    Zero out the points in the batch.
    """
    for b in range(num_points.shape[0]):
        x[b, num_points[b] :] = 0
    return x


class ZeroOutPoints(nn.Module):
    def forward(
        self, x: Float[Tensor, "B N C"], num_points: Int[Tensor, "B"]  # noqa: F821
    ) -> Float[Tensor, "B N C"]:  # noqa: F821
        return zero_out_points(x, num_points)


def offset_to_mask(
    x: Float[Tensor, "B M C"],  # noqa: F821
    offsets: Float[Tensor, "B+1"],  # noqa: F821
    max_num_points: int,  # noqa: F821
) -> Float[Tensor, "B M M"]:  # noqa: F821
    """
    Create a mask for the points in the batch.
    """
    B = x.shape[0]
    assert B == offsets.shape[0] - 1
    mask = torch.zeros((B, max_num_points, max_num_points), dtype=torch.float32, device=x.device)
    for b in range(B):
        mask[b, : offsets[b], : offsets[b]] = -torch.inf
    return mask


class ToAttention(BaseSpatialModule):
    def __init__(
        self,
        out_channels: int,
        num_encoding_channels: Optional[int],
        encoding_range: Optional[float],
        num_heads: int = 1,
        concat_input: bool = True,
        num_spatial_features: int = 3,
    ):
        super().__init__()
        self.sinusoidal_encoding = nn.Sequential(
            SinusoidalEncoding(
                num_channels=num_encoding_channels,
                data_range=encoding_range,
                concat_input=concat_input,
            ),
            nn.Linear(
                num_encoding_channels * num_spatial_features
                + (num_spatial_features if concat_input else 0),
                out_channels // num_heads,
            ),
        )

    def forward(
        self, x: SpatialFeatures
    ) -> Tuple[Float[Tensor, "B M C"], Float[Tensor, "B M C"], Float[Tensor, "B M M"]]:
        features, offsets, num_points = x.feature_tensor, x.offsets, x.offsets.diff()
        features = cat_to_pad(features, offsets)
        pos_enc = self.sinusoidal_encoding(x.coordinate_tensor)
        pos_enc = cat_to_pad(pos_enc, offsets)
        mask = offset_to_mask(features, offsets, features.shape[1])
        return features, pos_enc, mask, num_points


class ToSpatialFeatures(nn.Module):
    def forward(self, x: Float[Tensor, "B N C"], target: SpatialFeatures) -> SpatialFeatures:
        feats = pad_to_cat(x, target.offsets)
        return target.replace(batched_features=feats)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: Float[Tensor, "B N C"],  # noqa: F821
        pos_enc: Optional[Float[Tensor, "B N C"]] = None,  # noqa: F821
        mask: Optional[Float[Tensor, "B N N"]] = None,  # noqa: F821
        num_points: Optional[Int[Tensor, "B"]] = None,  # noqa: F821
    ) -> Float[Tensor, "B N C"]:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # Apply positional encoding to the query and key
        if pos_enc is not None:
            q = q + pos_enc.unsqueeze(1)
            k = k + pos_enc.unsqueeze(1)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask.unsqueeze(1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if num_points is not None:
            x = zero_out_points(x, num_points)
        return x


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 2,  # make hidden_dim multiple of this
        ffn_dim_multiplier: Optional[float] = None,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # Custom dimension factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: Float[Tensor, "B N D"]) -> Float[Tensor, "B N D"]:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        hidden_dim_multiplier: int = 4,
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.dim = dim
        self.attention = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=hidden_dim_multiplier * dim,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        x: Float[Tensor, "B N D"],  # noqa: F821
        pos: Float[Tensor, "B N 3"],  # noqa: F821
        mask: Optional[Float[Tensor, "B N N"]] = None,  # noqa: F821
        num_points: Optional[Int[Tensor, "B"]] = None,  # noqa: F821
    ) -> Float[Tensor, "B N D"]:  # noqa: F821
        h = x + self.attention(self.attention_norm(x), pos, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        if num_points is not None:
            out = zero_out_points(out, num_points)
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        hidden_dim_multiplier: int = 4,
        ffn_dim_multiplier: Optional[float] = None,
        num_layers: int = 1,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(
                TransformerBlock(
                    dim,
                    num_heads,
                    qkv_bias,
                    qk_scale,
                    attn_drop,
                    proj_drop,
                    hidden_dim_multiplier,
                    ffn_dim_multiplier,
                    norm_eps,
                )
            )
        self.norm = RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        x: Float[Tensor, "B N D"],
        pos: Float[Tensor, "B N 3"],
        mask: Optional[Float[Tensor, "B N N"]] = None,
        num_points: Optional[Int[Tensor, "B"]] = None,  # noqa: F821
    ) -> Float[Tensor, "B N D"]:
        for block in self.blocks:
            x = block(x, pos, mask, num_points)
        return self.norm(x)


class SpatialFeaturesTransformer(Transformer):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_layers: int = 1,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        num_encoding_channels: int = 32,
        encoding_range: float = 1.0,
        hidden_dim_multiplier: int = 4,
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: float = 1e-5,
    ):
        super().__init__(
            dim,
            num_heads,
            qkv_bias,
            qk_scale,
            attn_drop,
            proj_drop,
            hidden_dim_multiplier,
            ffn_dim_multiplier,
            num_layers,
            norm_eps,
        )
        self.to_attn = ToAttention(
            dim,
            num_encoding_channels,
            encoding_range,
            num_heads,
            concat_input=True,
            num_spatial_features=3,
        )
        self.from_attn = ToSpatialFeatures()

    def forward(self, x: SpatialFeatures) -> SpatialFeatures:
        features, pos_enc, mask, num_points = self.to_attn(x)
        y = super().forward(features, pos_enc, mask, num_points)
        y = self.from_attn(y, x)
        return y
