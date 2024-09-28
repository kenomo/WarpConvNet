import unittest

import torch
import warp as wp

from warpconvnet.geometry.point_collection import PointCollection
from warpconvnet.nn.unique import ToUnique
from warpconvnet.utils.argsort import argsort
from warpconvnet.utils.batch_index import (
    batch_index_from_indicies,
    batch_index_from_offset,
    batch_indexed_coordinates,
    offsets_from_batch_index,
)
from warpconvnet.utils.timer import Timer
from warpconvnet.utils.unique import unique_torch


class TestUtils(unittest.TestCase):
    def setUp(self) -> None:
        # Set random seed
        wp.init()
        torch.manual_seed(0)

        self.B, min_N, max_N, self.C = 3, 100000, 1000000, 7
        self.Ns = torch.randint(min_N, max_N, (self.B,))
        self.voxel_size = 0.01

        self.coords = [torch.rand((N, 3)) for N in self.Ns]
        self.features = [torch.rand((N, self.C)) for N in self.Ns]
        self.pc = PointCollection(self.coords, self.features)

        return super().setUp()

    def test_batch_index(self):
        device = "cuda:0"
        pc = self.pc.to(device)
        backends = ["torch", "warp"]
        backend_times = {backend: Timer() for backend in backends}
        offsets = pc.offsets.to(device)

        # Assert batch index are same for both backends
        wp_batch_index = batch_index_from_offset(offsets, backend="warp", device=device)
        torch_batch_index = batch_index_from_offset(offsets, backend="torch", device=device)
        self.assertTrue(torch.allclose(wp_batch_index, torch_batch_index))

        for backend in backends:
            for _ in range(20):
                with backend_times[backend]:
                    batch_index_from_offset(offsets, device=device, backend=backend)

        for backend in backends:
            print(f"Batch offsets {backend} time: {backend_times[backend].min_elapsed}")

        backend_times = {backend: Timer() for backend in backends}
        for backend in backends:
            for _ in range(20):
                with backend_times[backend]:
                    batch_indexed_coordinates(pc.coordinate_tensor, offsets, backend=backend)

        for backend in backends:
            print(
                f"Batch indexed coordinates {backend} time: {backend_times[backend].min_elapsed}"
            )

        # TITAN RTX (Warp 1.2.1 CUDA 12.1, Driver 12.3)
        #
        # batch_index_from_offset
        # torch time: 0.0001842975616455078
        # warp time: 0.00014734268188476562
        #
        # batch_indexed_coordinates
        # torch time: 0.00030159950256347656
        # warp time: 0.0011990070343017578

    def test_argsort(self):
        # test argsort torch/warp
        device = "cuda:0"
        backends = ["torch", "warp"]
        backend_times = {
            backend: {input_type: Timer() for input_type in ["torch", "warp"]}
            for backend in backends
        }
        rand_perm = torch.randperm(self.Ns.sum(), device=device).int()
        for backend in backends:
            for input_type in ["torch", "warp"]:
                if input_type == "warp":
                    rand_perm_in = wp.from_torch(rand_perm)
                else:
                    rand_perm_in = rand_perm
                for _ in range(20):
                    with backend_times[backend][input_type]:
                        argsort(rand_perm_in, backend=backend)

        for backend in backends:
            for input_type in ["torch", "warp"]:
                print(
                    f"Argsort sorting:{backend} input:{input_type} time: {backend_times[backend][input_type].min_elapsed}"
                )
        # Argsort sorting:torch input:torch time: 6.461143493652344e-05
        # Argsort sorting:torch input:warp time: 7.987022399902344e-05
        # Argsort sorting:warp input:torch time: 0.00016641616821289062
        # Argsort sorting:warp input:warp time: 0.00013446807861328125

    def test_offsets_from_batch_index(self):
        device = "cuda:0"
        backends = ["torch"]
        backend_times = {backend: Timer() for backend in backends}
        offsets = self.pc.offsets.to(device)
        batch_index = batch_index_from_offset(offsets, backend="torch", device=device)

        for backend in backends:
            gen_offsets = offsets_from_batch_index(batch_index, backend=backend).cpu()
            self.assertTrue(gen_offsets.equal(offsets.cpu()))

        for backend in backends:
            for _ in range(20):
                with backend_times[backend]:
                    gen_offsets = offsets_from_batch_index(batch_index, backend=backend)

        for backend in backends:
            print(f"Offsets from batch index {backend} time: {backend_times[backend].min_elapsed}")

    def test_batch_index_from_indicies(self):
        device = "cuda:0"
        backends = ["torch"]
        backend_times = {backend: Timer() for backend in backends}
        tot_N = self.Ns.sum()
        offsets = self.pc.offsets.to(device)
        batch_index = batch_index_from_offset(offsets)
        indices = torch.randint(0, tot_N, (100,))
        sel_batch_index = batch_index[indices]
        pred_batch_index = batch_index_from_indicies(indices, offsets, device=device)
        self.assertTrue(torch.allclose(sel_batch_index, pred_batch_index))


class TestToUnique(unittest.TestCase):
    def test_to_unique(self):
        x = torch.randint(0, 5, (10,))
        to_unique = ToUnique()

        unique, to_orig_indices, to_csr_indices, to_csr_offsets, _ = unique_torch(x)

        self.assertTrue(torch.allclose(x, unique[to_orig_indices]))
        # csr will sort the x values. Sorting will make no difference
        self.assertTrue(torch.allclose(torch.sort(x[to_csr_indices]).values, x[to_csr_indices]))

        unique = to_unique.to_unique(x)
        orig_x = to_unique.to_original(unique)

        self.assertTrue(torch.allclose(x, orig_x))


if __name__ == "__main__":
    unittest.main()
