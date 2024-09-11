import unittest

import torch

import warp as wp
from warp.convnet.geometry.ops.voxel_ops import voxel_downsample_mapping
from warp.convnet.geometry.point_collection import PointCollection


class TestVoxelOps(unittest.TestCase):
    def setUp(self) -> None:
        # Set random seed
        wp.init()
        torch.manual_seed(0)

        self.B, min_N, max_N, self.C = 3, 100000, 1000000, 7
        self.Ns = torch.randint(min_N, max_N, (self.B,))
        self.voxel_size = 0.025

        self.coords = [torch.rand((N, 3)) for N in self.Ns]
        self.features = [torch.rand((N, self.C)) for N in self.Ns]
        self.pc = PointCollection(self.coords, self.features)

        return super().setUp()

    def test_voxel_down_mapping(self):
        device = "cuda:0"
        pc: PointCollection = self.pc.to(device)

        # Downsample the coordinates
        downsampled_pc = pc.voxel_downsample(self.voxel_size)

        # Find the mapping
        up_map, down_map, valid = voxel_downsample_mapping(
            pc.coordinate_tensor,
            pc.offsets,
            downsampled_pc.coordinate_tensor,
            downsampled_pc.offsets,
            self.voxel_size,
        )

        # Check the mapping
        up_coords = torch.floor(pc.coordinate_tensor[up_map] / self.voxel_size)
        down_coords = torch.floor(downsampled_pc.coordinate_tensor[down_map] / self.voxel_size)
        self.assertTrue(torch.allclose(up_coords, down_coords))


if __name__ == "__main__":
    unittest.main()
