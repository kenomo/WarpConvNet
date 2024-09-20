import unittest

import torch
import warp as wp

from warpconvnet.core.hashmap import HashMethod
from warpconvnet.geometry.ops.warp_sort import POINT_ORDERING
from warpconvnet.geometry.spatially_sparse_tensor import SpatiallySparseTensor
from warpconvnet.utils.batch_index import batch_indexed_coordinates
from warpconvnet.utils.timer import Timer
from warpconvnet.utils.unique import unique_hashmap, unique_torch


class TestSpatiallySparseTensor(unittest.TestCase):
    def setUp(self) -> None:
        # Set random seed
        wp.init()
        torch.manual_seed(0)

        self.B, min_N, max_N, self.C = 3, 100000, 1000000, 7
        self.Ns = torch.randint(min_N, max_N, (self.B,))
        self.voxel_size = 0.01
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
        hash_timer, torch_timer = Timer(), Timer()
        bcoords = batch_indexed_coordinates(coords.batched_tensor, coords.offsets)
        for _ in range(20):
            with hash_timer:
                unique_index, hash_table = unique_hashmap(bcoords, HashMethod.CITY)

            with torch_timer:
                (
                    unique_coords,
                    to_orig_indices,
                    all_to_unique_indices,
                    all_to_unique_offsets,
                    perm,
                ) = unique_torch(bcoords, dim=0)

        self.assertTrue(len(unique_index) == len(perm), f"{len(unique_index)} != {len(perm)}")
        self.assertTrue((bcoords[unique_index].unique(dim=0) == unique_coords).all())

        print(f"Coord size: {coords.batched_tensor.shape}, N unique: {len(unique_index)}")
        print(f"Hashmap min time: {hash_timer.min_elapsed:.4e} s")
        print(f"Torch min time: {torch_timer.min_elapsed:.4e} s")
        print(f"Speedup: {torch_timer.min_elapsed / hash_timer.min_elapsed:.4e}")

    def test_sort(self):
        st = self.st
        device = torch.device("cuda:0")
        st = st.to(device)
        st = st.sort(ordering=POINT_ORDERING.Z_ORDER)

    def test_sparse_tensor_unique(self):
        device = torch.device("cuda:0")
        st = self.st.to(device)
        print(f"Number of coords: {len(st.batched_coordinates)}")
        unique_st = st.unique()
        print(f"Number of unique coords: {len(unique_st.batched_coordinates)}")
        bcoords = batch_indexed_coordinates(st.batched_coordinates.batched_tensor, st.offsets)
        self.assertTrue(
            torch.unique(bcoords, dim=0).shape[0] == len(unique_st.batched_coordinates)
        )

    def test_from_dense(self):
        dense_tensor = torch.rand(16, 3, 128, 128)
        # Empty out 80% of the elements
        dense_tensor[dense_tensor < 0.8] = 0
        st = SpatiallySparseTensor.from_dense(dense_tensor, dense_tensor_channel_dim=1)
        self.assertTrue(st.batch_size == 16)

        # test to_dense
        dense_tensor2 = st.to_dense(channel_dim=1)
        self.assertTrue((dense_tensor2 == dense_tensor).all())

    def test_sparse_to_dense_to_sparse(self):
        device = torch.device("cuda:0")
        st = self.st.to(device=device)
        dense_tensor = st.to_dense(channel_dim=1)
        # convolution on dense
        out_channels = 13
        conv = torch.nn.Conv3d(self.C, out_channels, 3, padding=1, bias=False).to(device=device)
        dense_tensor = conv(dense_tensor)

        # convert back to sparse
        st2 = SpatiallySparseTensor.from_dense(
            dense_tensor,
            to_spatial_sparse_tensor=st,
        )
        self.assertTrue(st2.num_channels == out_channels)


if __name__ == "__main__":
    unittest.main()
