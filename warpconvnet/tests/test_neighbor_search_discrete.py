import unittest

import torch

import warp as wp
import warp.utils
from warp.convnet.geometry.ops.neighbor_search_discrete import (
    DiscreteNeighborSearchResult,
    kernel_map_from_offsets,
    kernel_map_from_size,
    kernel_offsets_from_size,
    neighbor_search_hashmap,
    num_neighbors_kernel,
)
from warp.convnet.geometry.spatially_sparse_tensor import SpatiallySparseTensor
from warp.convnet.utils.batch_index import batch_indexed_coordinates
from warp.convnet.utils.timer import Timer


class TestNeighborSearchDiscrete(unittest.TestCase):
    def setUp(self):
        wp.init()
        device = torch.device("cuda:0")
        self.B, min_N, max_N, self.C = 3, 100000, 1000000, 7
        self.Ns = torch.randint(min_N, max_N, (self.B,))
        self.coords = [torch.rand((N, 3)) for N in self.Ns]
        self.features = [torch.rand((N, self.C)) for N in self.Ns]
        self.voxel_size = 0.025
        self.st_coords = [torch.floor(coords / self.voxel_size).int() for coords in self.coords]
        self.st = SpatiallySparseTensor(self.st_coords, self.features, device=device).unique()

    # Test point collection sorting
    def test_num_neighbor_kernel(self):
        device = torch.device("cuda:0")
        st: SpatiallySparseTensor = self.st.to(device)

        # Define the neighbor distance threshold
        neighbor_distance_threshold = 3
        hashmap = st.coordinate_hashmap

        # Use the st.coordinate_tensor as the query_coords
        batch_query_coords = batch_indexed_coordinates(st.coordinate_tensor, st.offsets)
        batch_query_coords = wp.from_torch(batch_query_coords)

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

        # cumsum
        cumsum_num_neighbors = wp.empty_like(num_neighbors)
        warp.utils.array_scan(num_neighbors, cumsum_num_neighbors)

    def test_neighbor_search(self):
        device = torch.device("cuda:0")
        st: SpatiallySparseTensor = self.st.to(device)
        batched_coords = batch_indexed_coordinates(st.coordinate_tensor, st.offsets)
        batched_coords_wp = wp.from_torch(batched_coords)

        # Define the neighbor distance threshold
        neighbor_distance_threshold = 3
        hashmap = st.coordinate_hashmap
        in_index, query_index = neighbor_search_hashmap(  # noqa: F841
            hashmap._hash_struct,
            batched_coords_wp,
            neighbor_distance_threshold,
        )

    def test_kernel_map(self):
        device = torch.device("cuda:0")
        st: SpatiallySparseTensor = self.st.to(device)

        kernel_offsets = torch.tensor(
            [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]],
            dtype=torch.int32,
            device=device,
        )

        bcoords = batch_indexed_coordinates(st.coordinate_tensor, st.offsets)
        st_hashmap = st.coordinate_hashmap

        kernel_map: DiscreteNeighborSearchResult = kernel_map_from_offsets(
            st_hashmap._hash_struct,
            bcoords,
            kernel_offsets,
        )

        tot_num_maps = kernel_map.offsets[-1].item()
        self.assertEqual(tot_num_maps, len(kernel_map.in_maps))
        self.assertEqual(tot_num_maps, len(kernel_map.out_maps))

    # Speed comparison between kernel_map_from_offsets and kernel_map_from_size
    def test_kernel_map_from_size(self):
        device = torch.device("cuda:0")
        st: SpatiallySparseTensor = self.st.to(device)
        bcoords = batch_indexed_coordinates(st.coordinate_tensor, st.offsets)
        st_hashmap = st.coordinate_hashmap

        kernel_offsets = kernel_offsets_from_size(
            (3, 3, 3),
            (1, 1, 1),
        ).to(device)

        kernel_map_size = kernel_map_from_size(
            batch_indexed_in_coords=bcoords,
            batch_indexed_out_coords=bcoords,
            in_to_out_stride_ratio=(1, 1, 1),
            kernel_size=(3, 3, 3),
            kernel_dilation=(1, 1, 1),
            kernel_search_batch_size=8,
        )

        kernel_map_offset = kernel_map_from_offsets(
            st_hashmap._hash_struct,
            bcoords,
            kernel_offsets,
        )

        for i, (in_map, out_map) in enumerate(kernel_map_size):
            # Check sizes
            in_map_o, out_map_o = kernel_map_offset[i]
            self.assertEqual(len(in_map), len(in_map_o))
            self.assertEqual(len(out_map), len(out_map_o))

            # Check unique values
            self.assertTrue(torch.all(in_map == in_map_o))
            self.assertTrue(torch.all(out_map == out_map_o))

    def test_kernel_map_speed(self):
        device = torch.device("cuda:0")
        st: SpatiallySparseTensor = self.st.to(device)
        bcoords = batch_indexed_coordinates(st.coordinate_tensor, st.offsets)
        st_hashmap = st.coordinate_hashmap

        print("N", len(bcoords))
        for kernel_size in [(3, 3, 3), (5, 5, 5)]:
            kernel_dilation = (1, 1, 1)
            in_to_out_stride_ratio = (1, 1, 1)

            kernel_offsets = kernel_offsets_from_size(
                kernel_size,
                kernel_dilation,
            ).to(device)

            backend_times = {backend: Timer() for backend in ["size", "offsets"]}

            for _ in range(4):
                with backend_times["size"]:
                    _ = kernel_map_from_size(
                        batch_indexed_in_coords=bcoords,
                        batch_indexed_out_coords=bcoords,
                        in_to_out_stride_ratio=in_to_out_stride_ratio,
                        kernel_size=kernel_size,
                        kernel_dilation=kernel_dilation,
                        kernel_search_batch_size=8,
                    )
                with backend_times["offsets"]:
                    _ = kernel_map_from_offsets(
                        st_hashmap._hash_struct,
                        bcoords,
                        kernel_offsets,
                    )

            print(f"kernel {kernel_size} size", backend_times["size"].min_elapsed)
            print(f"kernel {kernel_size} offsets", backend_times["offsets"].min_elapsed)


if __name__ == "__main__":
    unittest.main()
