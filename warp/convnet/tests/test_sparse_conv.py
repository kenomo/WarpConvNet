import unittest

import torch

import warp as wp
from warp.convnet.core.hashmap import VectorHashTable
from warp.convnet.geometry.ops.neighbor_search_discrete import kernel_map_from_size
from warp.convnet.geometry.spatially_sparse_tensor import SpatiallySparseTensor
from warp.convnet.nn.functional.sparse_conv import (
    SpatiallySparseConvExplicitGEMMFunction,
    generate_output_coords,
    spatially_sparse_conv,
)
from warp.convnet.nn.functional.sparse_ops import sparse_downsample_reduce
from warp.convnet.nn.sparse_conv import SpatiallySparseConv
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
        self.st = SpatiallySparseTensor(self.coords, self.features, device=device).unique()
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
        kernel_map = kernel_map_from_size(  # noqa: F841
            batch_indexed_in_coords,
            batch_indexed_output_coords,
            in_to_out_stride_ratio,
            kernel_size,
            kernel_dilation,
            kernel_batch,
        )

        tot_kernel_map = kernel_map.offsets[-1].item()
        self.assertTrue(tot_kernel_map == kernel_map.in_maps.shape[0])
        self.assertTrue(tot_kernel_map == kernel_map.out_maps.shape[0])

        for _, (in_map, out_map) in enumerate(kernel_map):
            self.assertTrue(in_map.shape[0] == out_map.shape[0])

        # Manually check the kernel map
        in_hashmap = VectorHashTable.from_keys(wp.from_torch(batch_indexed_in_coords))
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
                (i - kernel_size[0] // 2) * kernel_dilation[0],
                (j - kernel_size[1] // 2) * kernel_dilation[1],
                (k - kernel_size[2] // 2) * kernel_dilation[2],
            ],
            dim=1,
        ).to(device)

        batch_indexed_output_coords = batch_indexed_output_coords * torch.tensor(
            [1, *in_to_out_stride_ratio], dtype=torch.int32, device=device
        )

        N_in = batch_indexed_in_coords.shape[0]
        N_out = batch_indexed_output_coords.shape[0]
        for i, (in_map, out_map) in enumerate(kernel_map):
            offseted_out_coords = batch_indexed_output_coords + kernel_offsets[i]
            indices = in_hashmap.search(wp.from_torch(offseted_out_coords))
            indices = wp.to_torch(indices)
            valid_bool = (indices > 0).to(device)
            num_valid = valid_bool.sum().item()
            found_in_map = indices[valid_bool]

            self.assertTrue(num_valid == in_map.shape[0])
            self.assertTrue(in_map.max().item() < N_in)
            self.assertTrue(out_map.max().item() < N_out)
            self.assertTrue(found_in_map.max().item() <= N_in)
            unique_found_in_map = found_in_map.unique(sorted=True)
            unique_in_map = in_map.unique(sorted=True)
            self.assertTrue(torch.all(unique_found_in_map == unique_in_map))

    def test_sparse_conv(self):
        C_in, C_out = self.C, 13
        kernel_size = (3, 3, 3)
        stride = (2, 2, 2)
        num_kernels = kernel_size[0] * kernel_size[1] * kernel_size[2]
        weights = torch.randn(num_kernels, C_in, C_out).to(self.st.device)
        bias = torch.randn(C_out).to(self.st.device)
        out = spatially_sparse_conv(
            self.st,
            weight=weights,
            bias=bias,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.assertTrue(out.feature_tensor.shape[1] == C_out)

    def test_sparse_conv_backward(self):
        C_in, C_out = self.C, 13
        kernel_size = (3, 3, 3)
        stride = (2, 2, 2)
        num_kernels = kernel_size[0] * kernel_size[1] * kernel_size[2]
        weights = torch.randn(num_kernels, C_in, C_out).to(self.st.device)

        B, min_N, max_N, C = 3, 10, 20, 7
        Ns = torch.randint(min_N, max_N, (B,))
        voxel_size = 0.01
        coords = [(torch.rand((N, 3)) / voxel_size).int() for N in Ns]
        features = [torch.rand((N, C)) for N in Ns]
        st = SpatiallySparseTensor(coords, features, device="cuda:0").unique()

        batch_indexed_in_coords = batch_indexed_coordinates(st.coordinate_tensor, st.offsets)
        batch_indexed_out_coords, offsets = generate_output_coords(
            batch_indexed_in_coords, stride=stride
        )
        kernel_map = kernel_map_from_size(
            batch_indexed_in_coords,
            batch_indexed_out_coords,
            stride,
            kernel_size,
        )

        feature_tensor = st.feature_tensor.detach().requires_grad_(True)
        # torch gradcheck
        torch.autograd.gradcheck(
            SpatiallySparseConvExplicitGEMMFunction.apply,
            (feature_tensor, weights, kernel_map, batch_indexed_out_coords.shape[0]),
            eps=1e-3,
            atol=1e-3,
            rtol=1e-3,
        )

    def test_sparse_conv_module(self):
        C_in, C_out = self.C, 13
        kernel_size = (3, 3, 3)
        stride = (2, 2, 2)
        conv = SpatiallySparseConv(C_in, C_out, kernel_size, stride).to(self.st.device)
        out = conv(self.st)
        self.assertTrue(out.feature_tensor.shape[1] == C_out)

    def test_sparse_conv_module_transposed(self):
        C_in, C_out = self.C, 13
        kernel_size = (3, 3, 3)
        stride = (2, 2, 2)
        conv = SpatiallySparseConv(C_in, C_out, kernel_size, stride, transposed=True).to(
            self.st.device
        )
        st: SpatiallySparseTensor = self.st
        st_downsampled = sparse_downsample_reduce(st, (2, 2, 2))
        out = conv(st_downsampled, st)
        self.assertTrue(out.feature_tensor.shape[1] == C_out)

    def test_sparse_conv_generative(self):
        C_in, C_out = self.C, 13
        kernel_size = (3, 3, 3)
        stride = (1, 1, 1)
        conv = SpatiallySparseConv(C_in, C_out, kernel_size, stride, generative=True).to(
            self.st.device
        )
        out = conv(self.st)
        self.assertTrue(out.feature_tensor.shape[1] == C_out)
        # Check that the output is larger than the input
        self.assertTrue(out.coordinate_tensor.shape[0] > self.st.coordinate_tensor.shape[0])


if __name__ == "__main__":
    unittest.main()
