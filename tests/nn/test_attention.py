import functools
import unittest

import torch
import warp as wp

from warpconvnet.geometry.types.points import Points
from warpconvnet.nn.modules.attention import (
    Attention,
    NestedAttention,
    PatchAttention,
    SpatialFeatureAttention,
    ToAttention,
    TransformerBlock,
    ZeroOutPoints,
)
from warpconvnet.nn.encodings import FourierEncoding
from warpconvnet.nn.modules.mlp import Linear


class TestAttention(unittest.TestCase):
    def setUp(self):
        wp.init()
        self.B, min_N, max_N, self.C = 3, 1000, 10000, 128
        self.Ns = torch.randint(min_N, max_N, (self.B,))
        self.coords = [torch.rand((N, 3)) for N in self.Ns]
        self.features = [torch.rand((N, self.C)) for N in self.Ns]
        self.pc = Points(self.coords, self.features)

    def test_attention(self):
        device = torch.device("cuda:0")
        pc = self.pc.to(device)
        dim = self.C * 8
        num_heads = 8
        lift = Linear(self.C, dim).to(device)
        to_attn = ToAttention(
            out_channels=dim,
            num_heads=num_heads,
            num_encoding_channels=32,
            encoding_range=1.0,
            concat_input=True,
            num_spatial_features=3,
        ).to(device)
        attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=True,
        ).to(device)
        zero_out = ZeroOutPoints()
        pc = lift(pc)
        features, pos_enc, mask, num_points = to_attn(pc)
        x = attn(features, pos_enc, mask)
        x = zero_out(x, num_points)
        self.assertEqual(x.shape, (self.B, self.Ns.max(), dim))

    def test_to_attention(self):
        # Test the to_attention mask output
        device = torch.device("cuda:0")
        pc = self.pc.to(device)
        dim = self.C * 8
        num_heads = 8
        lift = Linear(self.C, dim).to(device)
        to_attn = ToAttention(
            out_channels=dim,
            use_encoding=True,
            num_heads=num_heads,
            num_encoding_channels=32,
            encoding_range=1.0,
            concat_input=True,
            num_spatial_features=3,
        ).to(device)
        pc = lift(pc)
        features, pos_enc, mask, num_points = to_attn(pc)
        # B, MaxN, C
        max_N = self.Ns.max()
        self.assertEqual(features.shape, (self.B, max_N, dim))
        # B, MaxN, dim / num_heads
        self.assertEqual(pos_enc.shape, (self.B, max_N, dim // num_heads))
        # B, 1, MaxN, MaxN
        self.assertEqual(mask.shape, (self.B, 1, max_N, max_N))
        for b in range(self.B):
            self.assertEqual(torch.all(mask[b, :, : num_points[b], : num_points[b]]).item(), True)
            # Rows
            # self.assertEqual(torch.any(mask[b, :, num_points[b] :]).item(), False)
            # Cols
            self.assertEqual(torch.any(mask[b, :, :, num_points[b] :]).item(), False)

    def test_patch_attention(self):
        device = torch.device("cuda:0")
        pc = self.pc.to(device)
        patch_size = 32
        dim = self.C * 8
        num_heads = 8
        lift = Linear(self.C, dim).to(device)
        patch_attn = PatchAttention(
            dim=dim,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=True,
            use_sdpa=False,
        ).to(device)

        pc = lift(pc)
        out = patch_attn(pc)

        self.assertIsInstance(out, Points)
        self.assertEqual(out.feature_tensor.shape[-1], dim)

        # Check that the number of points is preserved
        self.assertEqual(len(out), len(pc))

    def test_patch_transformer_block(self):
        device = torch.device("cuda:0")
        pc = self.pc.to(device)
        patch_size = 32
        dim = self.C * 8
        num_heads = 8
        lift = Linear(self.C, dim).to(device)
        patch_attn = TransformerBlock(
            dim=dim,
            num_heads=num_heads,
            ffn_multiplier=4,
            qkv_bias=True,
            attn_fn=functools.partial(PatchAttention, patch_size=patch_size),
        ).to(device)
        pc = lift(pc)
        st = pc.to_sparse(voxel_size=0.02)
        out = patch_attn(st)
        self.assertEqual(out.feature_tensor.shape[-1], dim)
        self.assertEqual(len(out), len(st))

        # Test backward pass and gradient
        pc = self.pc.to(device)
        pc.batched_features.batched_tensor.requires_grad_(True)
        pc = lift(pc)
        out = patch_attn(pc)
        loss = out.feature_tensor.sum()
        loss.backward()

        self.assertIsNotNone(patch_attn.attention.qkv.weight.grad)
        self.assertIsNotNone(patch_attn.attention_norm.norm.weight.grad)
        self.assertIsNotNone(lift.block.weight.grad)

    @torch.inference_mode()
    def test_nested_transformer_block(self):
        device = torch.device("cuda:0")
        dim = self.C * 8
        num_heads = 8
        lift = Linear(self.C, dim).to(device)
        transf = TransformerBlock(
            dim=dim,
            num_heads=num_heads,
            ffn_multiplier=4,
            qkv_bias=True,
            attn_fn=NestedAttention,
        ).to(device)

        pc = self.pc.to(device)
        pc = lift(pc)
        _ = transf(pc)

    @torch.inference_mode()
    def test_nested_attention(self):
        device = torch.device("cuda:0")
        pc = self.pc.to(device)
        dim = self.C * 8
        lift = Linear(self.C, dim).to(device)
        pc = lift(pc)
        attn = NestedAttention(
            dim=dim,
            num_heads=8,
            pos_enc=FourierEncoding(3, dim),
        ).to(device)
        out = attn(pc)
        self.assertEqual(out.features.shape, (self.Ns.sum(), dim))

    @torch.inference_mode()
    def test_cross_nested_attention(self):
        device = torch.device("cuda:0")
        pc = self.pc.to(device)
        dim = self.C * 8
        lift = Linear(self.C, dim).to(device)
        pc = lift(pc)
        # Initialize nested tensor with 100 queries per batch
        B, N = self.B, 100
        queries = torch.randn(B, N, dim).to(device)
        attn = NestedAttention(dim=dim, num_heads=8).to(device)
        out = attn(queries, pc)
        self.assertEqual(out.shape, (B, N, dim))

    def test_spatial_feature_attention(self):
        device = torch.device("cuda:0")
        lift = Linear(self.C, self.C * 8).to(device)
        attn = SpatialFeatureAttention(dim=self.C * 8, num_heads=8).to(device)
        pc = self.pc.to(device)
        pc = lift(pc)
        out = attn(pc)


if __name__ == "__main__":
    unittest.main()
