import unittest

import torch

import warp as wp
from warp.convnet.core.hashmap import VectorHashTable
from warp.convnet.geometry.spatially_sparse_tensor import SpatiallySparseTensor
from warp.convnet.nn.sparse_conv import generate_kernel_map, generate_output_coords
from warp.convnet.utils.batch_index import batch_indexed_coordinates


class TestSparseConv(unittest.TestCase):
    def setUp(self) -> None:
        wp.init()
        # Set random seed
        torch.manual_seed(0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.B, min_N, max_N, self.C = 3, 100000, 1000000, 7
        self.Ns = torch.randint(min_N, max_N, (self.B,))
        self.voxel_size = 0.01
        self.coords = [(torch.rand((N, 3)) / self.voxel_size).int() for N in self.Ns]
        self.features = [torch.rand((N, self.C)) for N in self.Ns]
        self.st = SpatiallySparseTensor(self.coords, self.features, device=device)
        return super().setUp()

    def test_generate_output_coords(self):
        batch_indexed_coords = batch_indexed_coordinates(
            self.st.coordinate_tensor,
            self.st.offsets,
        )
        output_coords, offsets = generate_output_coords(batch_indexed_coords, stride=(2, 2, 2))
        self.assertTrue(output_coords.shape[0] < batch_indexed_coords.shape[0])
        self.assertTrue(offsets.shape == (self.B + 1,))

    def test_generate_kernel_map(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_indexed_in_coords = batch_indexed_coordinates(
            self.st.coordinate_tensor,
            self.st.offsets,
        )
        batch_indexed_output_coords, offsets = generate_output_coords(
            batch_indexed_in_coords, stride=(2, 2, 2)
        )
        self.assertTrue(batch_indexed_in_coords.dtype == torch.int32)
        in_to_out_stride_ratio = (2, 2, 2)
        kernel_size = (3, 3, 3)
        kernel_dilation = (1, 1, 1)
        kernel_batch = 8
        in_map, out_map, offsets = generate_kernel_map(  # noqa: F841
            batch_indexed_in_coords,
            batch_indexed_output_coords,
            in_to_out_stride_ratio,
            kernel_size,
            kernel_dilation,
            kernel_batch,
        )

        tot_kernel_map = offsets[-1].item()
        self.assertTrue(tot_kernel_map == in_map.shape[0])
        self.assertTrue(tot_kernel_map == out_map.shape[0])

        # Manually check the kernel map
        in_hashmap = VectorHashTable.from_keys(
            wp.from_torch(batch_indexed_in_coords, dtype=wp.vec4i)
        )
        i, j, k = torch.meshgrid(
            torch.arange(kernel_size[0], dtype=torch.int32),
            torch.arange(kernel_size[1], dtype=torch.int32),
            torch.arange(kernel_size[2], dtype=torch.int32),
            indexing="ij",
        )
        i, j, k = i.flatten(), j.flatten(), k.flatten()
        kernel_offsets = torch.stack(
            [
                torch.zeros_like(i),
                (i - kernel_size[0] // 2) * kernel_dilation[0] * in_to_out_stride_ratio[0],
                (j - kernel_size[1] // 2) * kernel_dilation[1] * in_to_out_stride_ratio[1],
                (k - kernel_size[2] // 2) * kernel_dilation[2] * in_to_out_stride_ratio[2],
            ],
            dim=1,
        ).to(device)

        batch_indexed_output_coords = batch_indexed_output_coords * torch.tensor(
            [1, *in_to_out_stride_ratio], dtype=torch.int32, device=device
        )
        for i, num_kernel_map in enumerate(offsets.diff()):
            offseted_out_coords = batch_indexed_output_coords + kernel_offsets[i]
            indices = in_hashmap.search(wp.from_torch(offseted_out_coords, dtype=wp.vec4i))
            indices = wp.to_torch(indices)
            num_valid = (indices > 0).sum().item()
            self.assertTrue(num_valid == num_kernel_map)


if __name__ == "__main__":
    unittest.main()
