import dataclasses
import unittest

import torch

import warp as wp
from warp.convnet.geometry.ops.neighbor_search_continuous import (
    CONTINUOUS_NEIGHBOR_SEARCH_MODE,
    ContinuousNeighborSearchArgs,
    NeighborSearchResult,
)
from warp.convnet.geometry.point_collection import PointCollection


class TestPointCollection(unittest.TestCase):
    def setUp(self):
        wp.init()
        self.B, min_N, max_N, self.C = 3, 1000, 10000, 7
        self.Ns = torch.randint(min_N, max_N, (self.B,))
        self.coords = [torch.rand((N, 3)) for N in self.Ns]
        self.features = [torch.rand((N, self.C)) for N in self.Ns]
        self.pc = PointCollection(self.coords, self.features, device="cuda:0")

    # Test indexing
    def test_point_collection_indexing(self):
        pc = self.pc
        Ns = self.Ns
        for i in range(self.B):
            self.assertTrue(pc[i].batched_coordinates.batch_size == 1)
            self.assertTrue(pc[i].batched_coordinates.batched_tensor.shape[0] == Ns[i])
            self.assertTrue(pc[i].batched_features.batch_size == 1)
            self.assertTrue(pc[i].batched_features.batched_tensor.shape[0] == Ns[i])

    # Test point collection construction
    def test_point_collection_construction(self):
        Ns_cumsum = self.Ns.cumsum(dim=0).tolist()
        self.assertTrue(self.pc.batched_coordinates.batch_size == self.B)
        self.assertTrue(
            (self.pc.batched_coordinates.offsets == torch.IntTensor([0] + Ns_cumsum)).all()
        )
        self.assertTrue(self.pc.batched_coordinates.batched_tensor.shape == (Ns_cumsum[-1], 3))
        self.assertTrue(self.pc.batched_features.batch_size == self.B)
        self.assertTrue(self.pc.batched_features.batched_tensor.shape == (Ns_cumsum[-1], self.C))

        device = torch.device("cuda:0")
        pc = self.pc.to(device)
        self.assertTrue(pc.batched_coordinates.batched_tensor.device == device)

        # Test point collection from concatenated tensors
        coords = torch.cat(self.coords, dim=0)
        features = torch.cat(self.features, dim=0)
        offsets = torch.IntTensor([0] + Ns_cumsum)
        pc = PointCollection(coords, features, offsets=offsets)
        self.assertTrue(pc.batched_coordinates.batch_size == self.B)

    def test_point_collection_dataclasses(self):
        """
        Test dataclass serialization
        """
        pc = self.pc
        pc = pc.voxel_downsample(0.1)
        pc_dict = dataclasses.asdict(pc)
        self.assertTrue("_extra_attributes" in pc_dict)
        self.assertTrue(pc_dict["_extra_attributes"]["voxel_size"] == 0.1)
        pc_replace = dataclasses.replace(pc)
        print(f"Dataclasses replace: {str(pc_replace)}")
        self.assertTrue("voxel_size" in pc_replace.extra_attributes)

    # Test point collection radius search
    def test_point_collection_radius_search(self):
        pc = self.pc
        radius = 0.1
        args = ContinuousNeighborSearchArgs(
            mode=CONTINUOUS_NEIGHBOR_SEARCH_MODE.RADIUS,
            radius=radius,
        )
        search_result = pc.batched_coordinates.neighbors(args)
        self.assertTrue(isinstance(search_result, NeighborSearchResult))
        self.assertTrue(sum(self.Ns) == search_result.neighbors_row_splits.shape[0] - 1)
        self.assertTrue(
            search_result.neighbors_row_splits[-1] == search_result.neighbors_index.numel()
        )

    def test_knn_search(self):
        pc = self.pc
        knn_k = 10
        args = ContinuousNeighborSearchArgs(
            mode=CONTINUOUS_NEIGHBOR_SEARCH_MODE.KNN,
            k=knn_k,
        )
        search_result = pc.batched_coordinates.neighbors(args)
        self.assertTrue(isinstance(search_result, NeighborSearchResult))
        self.assertTrue(sum(self.Ns) == search_result.neighbors_row_splits.shape[0] - 1)
        self.assertTrue(sum(self.Ns) * knn_k == search_result.neighbors_index.numel())
        self.assertTrue(
            search_result.neighbors_row_splits[-1] == search_result.neighbors_index.numel()
        )

    def test_voxel_downsample(self):
        pc = self.pc
        voxel_size = 0.1
        downsampled_pc = pc.voxel_downsample(voxel_size)
        self.assertTrue(downsampled_pc.batched_coordinates.batched_tensor.shape[1] == 3)
        self.assertTrue(
            downsampled_pc.batched_features.batched_tensor.shape[0]
            == downsampled_pc.batched_coordinates.batched_tensor.shape[0]
        )

    def test_binary_ops(self):
        pc = self.pc
        # Test addition and multiplication
        pc1 = pc + 1
        self.assertTrue(torch.allclose(pc.feature_tensor + 1, pc1.feature_tensor))
        pc2 = pc * 2
        self.assertTrue(torch.allclose(pc.feature_tensor * 2, pc2.feature_tensor))
        pc2 = pc**2
        self.assertTrue(torch.allclose(pc.feature_tensor**2, pc2.feature_tensor))
        pc3 = pc + pc2
        self.assertTrue(
            pc3.batched_coordinates.batched_tensor.shape[0]
            == pc.batched_coordinates.batched_tensor.shape[0]
        )
        self.assertTrue(
            pc3.batched_features.batched_tensor.shape[0]
            == pc.batched_features.batched_tensor.shape[0]
        )
        pc4 = pc * pc2
        self.assertTrue(
            pc4.batched_coordinates.batched_tensor.shape[0]
            == pc.batched_coordinates.batched_tensor.shape[0]
        )
        self.assertTrue(
            pc4.batched_features.batched_tensor.shape[0]
            == pc.batched_features.batched_tensor.shape[0]
        )


if __name__ == "__main__":
    wp.init()
    unittest.main()
