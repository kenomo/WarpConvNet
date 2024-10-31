import functools
from typing import Any, Callable, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

from warpconvnet.geometry.base_geometry import (
    CatBatchedFeatures,
    CatPatchedFeatures,
    SpatialFeatures,
)
from warpconvnet.nn.base_module import BaseSpatialModule
from warpconvnet.nn.encodings import SinusoidalEncoding
from warpconvnet.nn.normalizations import LayerNorm, RMSNorm
from warpconvnet.ops.batch_copy import cat_to_pad, pad_to_cat
from warpconvnet.types import NestedTensor


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
) -> Float[Tensor, "B 1 M M"]:  # noqa: F821
    """
    Create a mask for the points in the batch.
    """
    B = x.shape[0]
    assert B == offsets.shape[0] - 1
    mask = torch.zeros(
        (B, 1, max_num_points, max_num_points), dtype=torch.float32, device=x.device
    )
    for b in range(B):
        mask[b, :, : offsets[b], : offsets[b]] = -torch.inf
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
        out_type: Literal["nested", "cat"] = "cat",
    ):
        super().__init__()
        self.out_type = out_type
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
        if self.out_type == "nested":
            features = x.nested_features
            coordinates = x.nested_coordinates
        else:
            features, offsets, num_points = (
                x.features,
                x.offsets,
                x.offsets.diff(),
            )
            features = cat_to_pad(features, offsets)
            coordinates = x.coordinate_tensor

        pos_enc = self.sinusoidal_encoding(coordinates)
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
        use_sdpa: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa

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
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        # Apply positional encoding to the query and key
        if pos_enc is not None:
            q = q + pos_enc.unsqueeze(1)
            k = k + pos_enc.unsqueeze(1)

        if self.use_sdpa:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.attn_drop.p,
                scale=self.scale,
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if mask is not None:
                attn = attn + mask

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if num_points is not None:
            x = zero_out_points(x, num_points)
        return x


class SpatialFeatureAttention(Attention):
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
    ):
        super().__init__(
            dim,
            num_heads,
            qkv_bias,
            qk_scale,
            attn_drop,
            proj_drop,
            use_sdpa=True,
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


class NestedAttention(BaseSpatialModule):
    """
    Warning: does not support gradient
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        pos_enc: Optional[nn.Module] = None,
        **kwargs,
    ):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.pos_enc = pos_enc

    def forward(
        self,
        query: Union[SpatialFeatures, Tensor, NestedTensor],
        key: Optional[SpatialFeatures] = None,
        value: Optional[SpatialFeatures] = None,
        query_pos: Optional[Tensor] = None,
        key_pos: Optional[Tensor] = None,
    ) -> Union[SpatialFeatures, Tensor, NestedTensor]:
        if isinstance(query, SpatialFeatures):
            query_feats = query.nested_features
        elif query.is_nested:
            query_feats = query
        elif isinstance(query, Tensor):
            query_feats = torch.nested.as_nested_tensor([q for q in query])

        # Assert query does not require gradient
        assert not query_feats.requires_grad, "NestedAttention does not support gradient"

        # All computations are done on nested tensors
        key_feats = key.nested_features if key is not None else query_feats
        value_feats = value.nested_features if value is not None else key_feats
        if self.pos_enc is not None:
            assert isinstance(query, SpatialFeatures)
            query_pos = self.pos_enc(query.nested_coordinates)
            key_pos = self.pos_enc(key.nested_coordinates) if key is not None else query_pos

        if query_pos is not None:
            if not query_pos.is_nested:
                query_pos = torch.nested.as_nested_tensor([q for q in query_pos])
            query_feats = query_feats + query_pos

        if key_pos is not None:
            if not key_pos.is_nested:
                key_pos = torch.nested.as_nested_tensor([k for k in key_pos])
            key_feats = key_feats + key_pos

        # Reshape for heads
        C = query_feats.size(-1)
        query_feats = torch.nested.nested_tensor(
            [
                q.reshape(-1, self.num_heads, C // self.num_heads).permute(1, 0, 2)
                for q in query_feats
            ]
        )
        key_feats = torch.nested.nested_tensor(
            [
                k.reshape(-1, self.num_heads, C // self.num_heads).permute(1, 0, 2)
                for k in key_feats
            ]
        )
        value_feats = torch.nested.nested_tensor(
            [
                v.reshape(-1, self.num_heads, C // self.num_heads).permute(1, 0, 2)
                for v in value_feats
            ]
        )
        # Reshape for heads
        x = F.scaled_dot_product_attention(
            query_feats,
            key_feats,
            value_feats,
            dropout_p=self.attn_drop.p,
            scale=self.scale,
        )
        x = torch.nested.nested_tensor([x_.permute(1, 0, 2).flatten(1) for x_ in x])

        # apply proj and proj_drop
        x = self.proj(x)
        x = self.proj_drop(x)

        # Return based on the type of query
        if isinstance(query, SpatialFeatures):
            return query.replace(batched_features=CatBatchedFeatures.from_nested(x))
        elif query.is_nested:
            return x
        else:
            return torch.stack([x_ for x_ in x])


class PatchAttention(BaseSpatialModule):
    def __init__(
        self,
        dim: int,
        patch_size: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_sdpa: bool = True,
        rand_perm_patch: bool = False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.use_sdpa = use_sdpa
        self.rand_perm_patch = rand_perm_patch
        if not use_sdpa:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: SpatialFeatures) -> SpatialFeatures:
        # Assert that x is serialized
        K = self.patch_size
        skip = x

        if self.rand_perm_patch:
            # Create permutation that preserves batch boundaries
            perm = []
            offsets = x.offsets
            for i in range(len(offsets) - 1):
                start, end = offsets[i], offsets[i + 1]
                perm.append(torch.randperm(end - start, device=x.device) + start)
            perm = torch.cat(perm)
            inverse_perm = torch.argsort(perm)
            x = x.replace(batched_features=x.feature_tensor[perm])

        patch_feats: CatPatchedFeatures = CatPatchedFeatures.from_cat(x.batched_features, K)
        feats = patch_feats.batched_tensor  # MxC
        M, C = feats.shape
        qkv = (
            self.qkv(feats)
            .reshape(M // K, K, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        # q: (M // K) x num_heads x K x C // num_heads
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        if self.use_sdpa:
            mask = torch.ones(M // K, 1, K, K, dtype=torch.bool, device=q.device)
            patch_offsets = patch_feats.patch_offsets
            num_points = patch_feats.offsets.diff()
            for i in range(patch_feats.batch_size):
                if num_points[i] % K != 0:
                    mask[patch_offsets[i + 1] // K - 1, :, :, num_points[i] % K :] = False
            out_feat = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.attn_drop if self.training else 0,
                scale=self.scale,
            )
            out_feat = out_feat.transpose(1, 2).reshape(M, C)
        else:
            # attn: (M // K) x num_heads x K x K
            attn = (q @ k.transpose(-2, -1)) * self.scale
            # mask out the attention weights for the padded points
            patch_offsets = patch_feats.patch_offsets
            num_points = patch_feats.offsets.diff()
            for i in range(patch_feats.batch_size):
                if num_points[i] % K != 0:
                    attn[patch_offsets[i + 1] // K - 1, :, :, num_points[i] % K :] = -torch.inf

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out_feat = (attn @ v).transpose(1, 2).reshape(M, C)

        out_feat = self.proj(out_feat)
        out_feat = self.proj_drop(out_feat)

        out_patch_feats: CatBatchedFeatures = patch_feats.replace(batched_tensor=out_feat).to_cat()

        if self.rand_perm_patch:
            out_patch_feats = out_patch_feats.batched_tensor[inverse_perm]

        return skip.replace(batched_features=out_patch_feats)


class FeedForward(BaseSpatialModule):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 2,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(
        self, x: Union[Float[Tensor, "B N D"], SpatialFeatures]
    ) -> Union[Float[Tensor, "B N D"], SpatialFeatures]:
        feat = x.features if isinstance(x, SpatialFeatures) else x
        # Apply feed forward
        feat = self.w2(F.silu(self.w1(feat)) * self.w3(feat))
        # Return based on the type of x
        return x.replace(batched_features=feat) if isinstance(x, SpatialFeatures) else feat


class TransformerBlock(BaseSpatialModule):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        ffn_multiplier: int = 4,
        norm_eps: float = 1e-5,
        attn_fn: Optional[Callable[..., nn.Module]] = None,
        norm_fn: Optional[Callable[..., nn.Module]] = LayerNorm,
    ):
        super().__init__()
        if attn_fn is None:
            attn_fn = functools.partial(PatchAttention, patch_size=1024)
        self.dim = dim
        self.attention = attn_fn(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=ffn_multiplier * dim,
        )
        self.attention_norm = norm_fn(dim, eps=norm_eps)
        self.ffn_norm = norm_fn(dim, eps=norm_eps)

    def forward(
        self,
        x: SpatialFeatures,
        *args: Any,
        **kwargs: Any,
    ) -> SpatialFeatures:
        h = x + self.attention(self.attention_norm(x), *args, **kwargs)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
