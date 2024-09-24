import unittest

import torch
import warp as wp

from warpconvnet.geometry.point_collection import PointCollection
from warpconvnet.nn.attention import (
    Attention,
    SpatialFeaturesTransformer,
    ToAttention,
    TransformerBlock,
    ZeroOutPoints,
)
from warpconvnet.nn.mlp import Linear


class TestAttention(unittest.TestCase):
    def setUp(self):
        wp.init()
        self.B, min_N, max_N, self.C = 3, 1000, 10000, 128
        self.Ns = torch.randint(min_N, max_N, (self.B,))
        self.coords = [torch.rand((N, 3)) for N in self.Ns]
        self.features = [torch.rand((N, self.C)) for N in self.Ns]
        self.pc = PointCollection(self.coords, self.features)

    # Test attention
    def test_spatial_attention(self):
        device = torch.device("cuda:0")
        pc = self.pc.to(device)
        attn = SpatialFeaturesTransformer(
            self.C,
            num_heads=8,
            num_encoding_channels=32,
            encoding_range=10.0,
        ).to(device)
        attn_out = attn(pc)

    def test_to_attention(self):
        device = torch.device("cuda:0")
        pc = self.pc.to(device)
        to_attn = ToAttention(
            out_channels=17,
            num_encoding_channels=32,
            encoding_range=1.0,
            concat_input=True,
            num_spatial_features=3,
        ).to(device)
        features, pos_enc, mask, num_points = to_attn(pc)
        self.assertEqual(features.shape, (self.B, self.Ns.max(), self.C))
        self.assertEqual(pos_enc.shape, (self.B, self.Ns.max(), 17))
        self.assertEqual(mask.shape, (self.B, self.Ns.max(), self.Ns.max()))
        self.assertEqual(len(num_points), self.B)

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

    def test_transformer_block(self):
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
        attn = TransformerBlock(
            dim,
            num_heads=num_heads,
            qkv_bias=True,
            hidden_dim_multiplier=4,
            ffn_dim_multiplier=4,
        ).to(device)
        pc = lift(pc)
        features, pos_enc, mask, num_points = to_attn(pc)
        x = attn(features, pos_enc, mask)
        self.assertEqual(x.shape, (self.B, self.Ns.max(), dim))


if __name__ == "__main__":
    unittest.main()
