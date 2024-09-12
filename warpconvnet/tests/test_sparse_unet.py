import unittest

import torch
import torch.nn as nn
import warp as wp

from warpconvnet.geometry.point_collection import PointCollection
from warpconvnet.geometry.spatially_sparse_tensor import SpatiallySparseTensor
from warpconvnet.models.sparse_conv_unet import SparseConvDecoder, SparseConvEncoder
from warpconvnet.nn.functional.transforms import apply_feature_transform


class TestSparseConvEncoder(unittest.TestCase):
    def setUp(self):
        wp.init()
        self.device = torch.device("cuda:0")
        self.B, min_N, max_N, self.C = 3, 1000, 10000, 7
        self.Ns = torch.randint(min_N, max_N, (self.B,))
        self.coords = [torch.rand((N, 3)) for N in self.Ns]
        self.features = [torch.rand((N, self.C)) for N in self.Ns]
        self.pc = PointCollection(self.coords, self.features).to(self.device)

    def test_sparse_conv_encoder(self):
        # Create conv layer
        voxel_size = 0.02
        pc: PointCollection = self.pc
        st: SpatiallySparseTensor = pc.to_sparse(voxel_size)
        enc_channels = [self.C, 32, 64]

        model = SparseConvEncoder(
            num_levels=2,
            kernel_sizes=3,
            encoder_channels=enc_channels,
            num_blocks_per_level=[1, 1],
        ).to(self.device)
        print(model)
        outs = model(st)
        # backward
        outs[-1].feature_tensor.mean().backward()

    def test_sparse_conv_decoder(self):
        # Create conv layer
        voxel_size = 0.02
        pc: PointCollection = self.pc
        st: SpatiallySparseTensor = pc.to_sparse(voxel_size)
        enc_channels = [self.C, 32, 64]
        dec_channels = [64, 32]

        encoder = SparseConvEncoder(
            num_levels=2,
            kernel_sizes=3,
            encoder_channels=enc_channels,
            num_blocks_per_level=[1, 1],
        ).to(self.device)

        decoder = SparseConvDecoder(
            num_levels=1,
            kernel_sizes=3,
            encoder_channels=enc_channels,
            decoder_channels=dec_channels,
            num_blocks_per_level=[1],
        ).to(self.device)

        encoder_outs = encoder(st)
        decoder_outs = decoder(encoder_outs)

        print(decoder_outs[0].feature_tensor.shape)


if __name__ == "__main__":
    unittest.main()
