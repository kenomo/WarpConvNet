import unittest

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import warp as wp
from warpconvnet.geometry.ops.neighbor_search_continuous import (
    CONTINUOUS_NEIGHBOR_SEARCH_MODE,
    ContinuousNeighborSearchArgs,
)
from warpconvnet.geometry.point_collection import PointCollection
from warpconvnet.nn.functional.point_pool import (
    FEATURE_POOLING_MODE,
    FeaturePoolingArgs,
)
from warpconvnet.nn.mlp import MLPBlock
from warpconvnet.nn.point_conv import PointConv
from warpconvnet.nn.point_transform import PointCollectionTransform


class TestFSDP(unittest.TestCase):
    def setUp(self):
        self.B, min_N, max_N, self.C = 3, 1000, 10000, 7
        self.Ns = torch.randint(min_N, max_N, (self.B,))
        self.coords = [torch.rand((N, 3)) for N in self.Ns]
        self.features = [torch.rand((N, self.C), requires_grad=True) for N in self.Ns]
        self.pc = PointCollection(self.coords, self.features)

    def test_fsdp(self):
        device = dist.get_rank()
        print(f"Rank {device} is running test_fsdp")

        # Create conv layer
        in_channels, out_channels = self.C, 16
        search_arg = ContinuousNeighborSearchArgs(
            mode=CONTINUOUS_NEIGHBOR_SEARCH_MODE.RADIUS,
            radius=0.4,
        )
        pooling_arg = FeaturePoolingArgs(
            pooling_mode=FEATURE_POOLING_MODE.REDUCTIONS,
            reductions=["mean"],
            downsample_voxel_size=0.2,
        )
        torch.cuda.set_device(device)
        model = nn.Sequential(
            PointConv(
                in_channels,
                out_channels,
                neighbor_search_args=search_arg,
                pooling_args=pooling_arg,
                out_point_type="downsample",
            ),
            PointCollectionTransform(
                MLPBlock(out_channels, hidden_channels=32, out_channels=out_channels)
            ),
        ).to(device)

        fsdp_model = FSDP(model)
        # print the model only on rank 0
        if device == 0:
            print(fsdp_model)

        fsdp_model.train()
        optim = torch.optim.Adam(fsdp_model.parameters(), lr=0.0001)
        for _ in range(100):
            pc = self.pc.to(device)
            pc = pc.voxel_downsample(0.1)
            out = fsdp_model(pc)
            assert out.voxel_size is not None
            loss = out.feature_tensor.mean()
            loss.backward()
            optim.step()


if __name__ == "__main__":
    """
    Run with torch run

    torchrun --nproc_per_node=2 warp/convnet/tests/test_fsdp.py
    """
    dist.init_process_group(backend="nccl")
    wp.init()
    unittest.main()
