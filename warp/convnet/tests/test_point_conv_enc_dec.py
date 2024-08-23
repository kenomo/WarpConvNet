import unittest

import torch

import warp as wp
from warp.convnet.geometry.ops.neighbor_search_continuous import (
    NEIGHBOR_SEARCH_MODE,
    NeighborSearchArgs,
)
from warp.convnet.geometry.ops.point_pool import (
    FEATURE_POOLING_MODE,
    FeaturePoolingArgs,
)
from warp.convnet.geometry.point_collection import PointCollection
from warp.convnet.models.point_conv_unet import PointConvEncoderDecoder
from warp.convnet.nn.point_conv import PointConv, PointConvDecoder, PointConvEncoder


class TestPointConvEncoder(unittest.TestCase):
    def setUp(self):
        wp.init()
        self.device = torch.device("cuda:0")
        self.B, min_N, max_N, self.C = 3, 1000, 10000, 7
        self.Ns = torch.randint(min_N, max_N, (self.B,))
        self.coords = [torch.rand((N, 3)) for N in self.Ns]
        self.features = [torch.rand((N, self.C)) for N in self.Ns]
        self.pc = PointCollection(self.coords, self.features).to(self.device)

    def test_point_conv_encoder(self):
        pc = self.pc
        # Create conv layer
        in_channels = self.C
        search_args = NeighborSearchArgs(
            mode=NEIGHBOR_SEARCH_MODE.RADIUS,
        )
        pool_args = FeaturePoolingArgs(
            pooling_mode=FEATURE_POOLING_MODE.REDUCTIONS,
            reductions=["mean"],
        )
        enc_channels = [16, 32, 64]
        dec_channels = [64, 48]
        neighbor_search_radii = [0.1, 0.2]
        downsample_voxel_sizes = [0.1, 0.2]
        first_search_arg = search_args.clone(radius=neighbor_search_radii[0])
        first_pool_arg = pool_args.clone(downsample_voxel_size=downsample_voxel_sizes[0] / 2)
        first_conv = PointConv(
            in_channels=in_channels,
            out_channels=enc_channels[0],
            neighbor_search_args=first_search_arg,
            pooling_args=first_pool_arg,
            out_point_type="downsample",
        ).to(self.device)
        encoder = PointConvEncoder(
            encoder_channels=enc_channels,
            num_blocks_per_level=[1, 1],
            neighbor_search_args=search_args,
            neighbor_search_radii=neighbor_search_radii,
            pooling_args=pool_args,
            downsample_voxel_sizes=downsample_voxel_sizes,
            num_levels=2,
        ).to(self.device)
        decoder = PointConvDecoder(
            decoder_channels=dec_channels,
            encoder_channels=enc_channels,
            num_blocks_per_level=[
                1,
            ],
            neighbor_search_args=search_args,
            neighbor_search_radii=neighbor_search_radii[::-1],
            num_levels=1,
        ).to(self.device)

        # Forward pass
        out = first_conv(pc)
        enc_outs = encoder(out)
        dec_outs = decoder(enc_outs[-1], enc_outs)
        # backward
        dec_outs[-1].feature_tensor.mean().backward()
        # print the conv param grads
        for name, param in decoder.named_parameters():
            if param.grad is not None:
                print(name, param.grad.shape)
            else:
                print(name, "has no grad")

    def test_point_conv_encdec(self):
        pc = self.pc
        # Create conv layer
        in_channels, out_channels = self.C, 16
        search_args = NeighborSearchArgs(
            mode=NEIGHBOR_SEARCH_MODE.RADIUS,
        )
        pool_args = FeaturePoolingArgs(
            pooling_mode=FEATURE_POOLING_MODE.REDUCTIONS,
            reductions=["mean"],
        )
        enc_channels = [16, 32, 64]
        dec_channels = [64, 48]
        neighbor_search_radii = [0.1, 0.2]
        downsample_voxel_sizes = [0.1, 0.2]

        model = PointConvEncoderDecoder(
            in_channels=in_channels,
            out_channels=out_channels,
            num_levels_enc=2,
            num_levels_dec=1,
            encoder_channels=enc_channels,
            decoder_channels=dec_channels,
            num_encoder_blocks_per_level=[1, 1],
            num_decoder_blocks_per_level=[1],
            neighbor_search_args=search_args,
            neighbor_search_radii=neighbor_search_radii,
            pooling_args=pool_args,
            downsample_voxel_sizes=downsample_voxel_sizes,
        ).to(self.device)
        out, out_dec, out_enc = model(pc)
        # backward
        out.feature_tensor.mean().backward()
