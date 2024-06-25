import unittest

import torch

from warp.convnet.geometry.spatially_sparse_tensor import SpatiallySparseTensor
from warp.convnet.utils.batch_index import batch_indexed_coordinates
from warp.convnet.utils.unique import unique_hashmap, unique_torch


class TestSpatiallySparseTensor(unittest.TestCase):
    def setUp(self) -> None:
        self.B, min_N, max_N, self.C = 3, 1000, 10000, 7
        self.Ns = torch.randint(min_N, max_N, (self.B,))
        self.voxel_size = 0.05
        self.coords = [(torch.rand((N, 3)) / self.voxel_size).int() for N in self.Ns]
        self.features = [torch.rand((N, self.C)) for N in self.Ns]
        self.st = SpatiallySparseTensor(self.coords, self.features)
        return super().setUp()

    def test_spatially_sparse_tensor(self):
        Ns_cumsum = self.Ns.cumsum(dim=0).tolist()
        self.assertTrue(self.st.batched_coordinates.batch_size == self.B)
        self.assertTrue(
            (self.st.batched_coordinates.offsets == torch.IntTensor([0] + Ns_cumsum)).all()
        )
        self.assertTrue(self.st.batched_coordinates.batched_tensor.shape == (Ns_cumsum[-1], 3))
        self.assertTrue(self.st.batched_features.batch_size == self.B)
        self.assertTrue(self.st.batched_features.batched_tensor.shape == (Ns_cumsum[-1], self.C))

        device = torch.device("cuda:0")
        st = self.st.to(device)
        self.assertTrue(st.batched_coordinates.batched_tensor.device == device)

    def test_unique_hashmap(self):
        device = "cuda:0"
        st = self.st.to(device)
        coords = st.batched_coordinates
        bcoords = batch_indexed_coordinates(coords.batched_tensor, coords.offsets)
        unique_index, hash_table = unique_hashmap(bcoords)
        (
            unique_coords,
            to_orig_indices,
            all_to_unique_indices,
            all_to_unique_offsets,
            perm,
        ) = unique_torch(bcoords, dim=0)
        self.assertTrue(len(unique_index) == len(perm), f"{len(unique_index)} != {len(perm)}")


if __name__ == "__main__":
    unittest.main()
