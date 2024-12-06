import unittest

import torch
import warp as wp
import warp.utils

from warpconvnet.geometry.ops.neighbor_search_discrete import (
    DiscreteNeighborSearchResult,
    _kernel_map_from_direct_queries,
    _kernel_map_from_offsets,
    _kernel_map_from_size,
    kernel_map_from_size,
    kernel_offsets_from_size,
)
from warpconvnet.geometry.spatially_sparse_tensor import SpatiallySparseTensor
from warpconvnet.utils.batch_index import batch_indexed_coordinates
from warpconvnet.utils.timer import Timer


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

    def test_kernel_map_from_offset(self):
        device = torch.device("cuda:0")
        st: SpatiallySparseTensor = self.st.to(device)

        kernel_offsets = torch.tensor(
            [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]],
            dtype=torch.int32,
            device=device,
        )

        bcoords = batch_indexed_coordinates(st.coordinate_tensor, st.offsets)
        st_hashmap = st.coordinate_hashmap

        kernel_map: DiscreteNeighborSearchResult = _kernel_map_from_offsets(
            st_hashmap._hash_struct,
            bcoords,
            kernel_offsets,
        )

        tot_num_maps = kernel_map.offsets[-1].item()
        self.assertEqual(tot_num_maps, len(kernel_map.in_maps))
        self.assertEqual(tot_num_maps, len(kernel_map.out_maps))

    def test_kernel_map_from_size(self):
        device = torch.device("cuda:0")
        st: SpatiallySparseTensor = self.st.to(device)

        bcoords = batch_indexed_coordinates(st.coordinate_tensor, st.offsets)
        st_hashmap = st.coordinate_hashmap
        kernel_sizes = (3, 3, 3)

        kernel_map: DiscreteNeighborSearchResult = _kernel_map_from_size(
            st_hashmap._hash_struct,
            bcoords,
            kernel_sizes,
        )

        tot_num_maps = kernel_map.offsets[-1].item()
        self.assertEqual(tot_num_maps, len(kernel_map.in_maps))
        self.assertEqual(tot_num_maps, len(kernel_map.out_maps))

    def test_compare_all_kernel_maps(self):
        device = torch.device("cuda:0")
        st: SpatiallySparseTensor = self.st.to(device)
        bcoords = batch_indexed_coordinates(st.coordinate_tensor, st.offsets)
        st_hashmap = st.coordinate_hashmap

        # Test parameters
        kernel_size = (3, 3, 3)
        kernel_dilation = (1, 1, 1)
        in_to_out_stride_ratio = (1, 1, 1)

        # Generate kernel offsets
        kernel_offsets = kernel_offsets_from_size(
            kernel_size=kernel_size,
            kernel_dilation=kernel_dilation,
        ).to(device)

        # Get results from all three methods
        kernel_map_offsets = _kernel_map_from_offsets(
            st_hashmap._hash_struct,
            bcoords,
            kernel_offsets,
        )

        kernel_map_size = _kernel_map_from_size(
            st_hashmap._hash_struct,
            bcoords,
            kernel_size,
        )

        kernel_map_direct = _kernel_map_from_direct_queries(
            st_hashmap._hash_struct,
            bcoords,
            in_to_out_stride_ratio=in_to_out_stride_ratio,
            kernel_size=kernel_size,
            kernel_dilation=kernel_dilation,
        )

        # Compare results
        print("Comparing kernel maps:")
        print(f"Offsets method: {len(kernel_map_offsets)} batches")
        print(f"Size method: {len(kernel_map_size)} batches")
        print(f"Direct method: {len(kernel_map_direct)} batches")

        # All methods should produce same number of batches
        self.assertEqual(len(kernel_map_offsets), len(kernel_map_size))
        self.assertEqual(len(kernel_map_size), len(kernel_map_direct))

        for i in range(len(kernel_map_offsets)):
            # Get maps for each method
            in_map_o, out_map_o = kernel_map_offsets[i]
            in_map_s, out_map_s = kernel_map_size[i]
            in_map_d, out_map_d = kernel_map_direct[i]

            # Check sizes match
            self.assertEqual(len(in_map_o), len(in_map_s))
            self.assertEqual(len(in_map_s), len(in_map_d))
            self.assertEqual(len(out_map_o), len(out_map_s))
            self.assertEqual(len(out_map_s), len(out_map_d))

            # Check values match
            # Note: The order of indices within each batch might differ between methods,
            # so we need to sort before comparing
            self.assertTrue(torch.equal(torch.sort(in_map_o)[0], torch.sort(in_map_s)[0]))
            self.assertTrue(torch.equal(torch.sort(in_map_s)[0], torch.sort(in_map_d)[0]))
            self.assertTrue(torch.equal(torch.sort(out_map_o)[0], torch.sort(out_map_s)[0]))
            self.assertTrue(torch.equal(torch.sort(out_map_s)[0], torch.sort(out_map_d)[0]))

    def test_kernel_map_speed(self):
        device = torch.device("cuda:0")
        st: SpatiallySparseTensor = self.st.to(device)
        bcoords = batch_indexed_coordinates(st.coordinate_tensor, st.offsets)
        st_hashmap = st.coordinate_hashmap
        return_type = "offsets"

        print("N", len(bcoords))
        for kernel_size in [(3, 3, 3), (5, 5, 5), (7, 7, 7), (9, 9, 9)]:
            kernel_dilation = (1, 1, 1)
            in_to_out_stride_ratio = (1, 1, 1)

            kernel_offsets = kernel_offsets_from_size(
                kernel_size,
                kernel_dilation,
            ).to(device)

            backend_times = {backend: Timer() for backend in ["size", "offsets", "direct"]}

            for _ in range(4):
                with backend_times["offsets"]:
                    _ = _kernel_map_from_offsets(
                        st_hashmap._hash_struct,
                        bcoords,
                        kernel_offsets,
                        return_type=return_type,
                    )
                with backend_times["size"]:
                    _ = _kernel_map_from_size(
                        st_hashmap._hash_struct,
                        bcoords,
                        kernel_size,
                        return_type=return_type,
                    )
                with backend_times["direct"]:
                    _ = _kernel_map_from_direct_queries(
                        st_hashmap._hash_struct,
                        bcoords,
                        in_to_out_stride_ratio=in_to_out_stride_ratio,
                        kernel_size=kernel_size,
                        kernel_dilation=kernel_dilation,
                    )

            print(f"kernel {kernel_size} size", backend_times["size"].min_elapsed)
            print(f"kernel {kernel_size} offsets", backend_times["offsets"].min_elapsed)
            print(f"kernel {kernel_size} direct", backend_times["direct"].min_elapsed)


if __name__ == "__main__":
    unittest.main()
