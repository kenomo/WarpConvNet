import unittest

import torch

import warp as wp
from warp.convnet.geometry.ops.neighbor_search_discrete import num_neighbors_kernel
from warp.convnet.geometry.point_collection import PointCollection
from warp.convnet.geometry.spatially_sparse_tensor import SpatiallySparseTensor
from warp.convnet.utils.batch_index import batch_indexed_coordinates


class TestSorting(unittest.TestCase):
    def setUp(self):
        self.B, min_N, max_N, self.C = 3, 1000, 10000, 7
        self.Ns = torch.randint(min_N, max_N, (self.B,))
        self.coords = [torch.rand((N, 3)) for N in self.Ns]
        self.features = [torch.rand((N, self.C)) for N in self.Ns]
        self.pc = PointCollection(self.coords, self.features)

        self.voxel_size = 0.1
        self.st_coords = [torch.floor(coords / self.voxel_size).int() for coords in self.coords]
        self.st = SpatiallySparseTensor(self.st_coords, self.features)

    # Test point collection sorting
    def test_num_neighbor_kernel(self):
        device = torch.device("cuda:0")
        st: SpatiallySparseTensor = self.st.to(device)

        # Define the neighbor distance threshold
        neighbor_distance_threshold = 3
        hashmap = st.coordinate_hashmap

        # Use the st.coordinate_tensor as the query_coords
        batch_query_coords = batch_indexed_coordinates(st.coordinate_tensor, st.offsets)
        batch_query_coords = wp.from_torch(batch_query_coords, dtype=wp.vec4i)

        # Create the num_neighbors tensor
        num_neighbors = wp.zeros(len(st.coordinate_tensor), dtype=int, device=str(device))
        hashmap = st.coordinate_hashmap

        # Launch the kernel
        wp.launch(
            kernel=num_neighbors_kernel,
            dim=len(st.coordinate_tensor),
            inputs=[
                hashmap._hash_struct,
                batch_query_coords,
                neighbor_distance_threshold,
                num_neighbors,
            ],
        )

        print(num_neighbors.numpy())


if __name__ == "__main__":
    wp.init()
    unittest.main()
