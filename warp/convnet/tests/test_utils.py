import unittest

import torch

import warp as wp
from warp.convnet.geometry.point_collection import PointCollection
from warp.convnet.utils.argsort import argsort
from warp.convnet.utils.batch_index import (
    batch_index_from_offset,
    batch_indexed_coordinates,
)
from warp.convnet.utils.timer import Timer


class TestUtils(unittest.TestCase):
    def setUp(self) -> None:
        # Set random seed
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
                    batch_indexed_coordinates(pc.coords, offsets, backend=backend)

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


if __name__ == "__main__":
    unittest.main()
