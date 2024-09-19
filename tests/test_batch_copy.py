import unittest

import torch
import warp as wp

from warpconvnet.ops.batch_copy import cat_to_batch
from warpconvnet.utils.timer import Timer


class TestBatchCopy(unittest.TestCase):
    def setUp(self) -> None:
        # Set random seed
        wp.init()
        torch.manual_seed(0)

        self.B, min_N, max_N = 16, 1000, 100000
        self.Ns = torch.randint(min_N, max_N, (self.B,))
        self.offsets = torch.cat(
            [torch.zeros(1, dtype=torch.int32), torch.cumsum(torch.tensor(self.Ns), dim=0)], dim=0
        ).int()
        return super().setUp()

    def test_batch_copy(self):
        """
        Time all backends, num_copy_per_thread in [16, 32, 64, 128, 256], channels in [16, 32, 64, 128, 256, 512]

        Backend: torch, C: 16, time: 0.00016117095947265625
        Backend: warp, num_copy_per_thread: None, C: 16, time: 0.0005505084991455078
        Backend: warp, num_copy_per_thread: 16, C: 16, time: 0.0005505084991455078
        ...
        Backend: warp, num_copy_per_thread: 512, C: 16, time: 0.0005505084991455078
        ...
        Backend: torch, C: 512, time: 0.0001609325408935547
        Backend: warp, num_copy_per_thread: 16, C: 512, time: 0.04888772964477539
        ...
        Backend: warp, num_copy_per_thread: 512, C: 512, time: 0.04888772964477539
        """
        device = "cuda:0"
        for C in [16, 32, 64, 128, 256, 512]:
            features = torch.cat([torch.rand((N, C)) for N in self.Ns], dim=0).to(device)
            for backend in ["torch", "warp"]:
                timer = Timer()
                if backend == "warp":
                    for num_copy_per_thread in [None, 16, 32, 64, 128, 256, 512]:
                        for _ in range(10):
                            with timer:
                                out_features = cat_to_batch(
                                    features,
                                    self.offsets,
                                    backend=backend,
                                    num_copy_per_thread=num_copy_per_thread,
                                )
                        print(
                            f"Backend: {backend}, num_copy_per_thread: {num_copy_per_thread}, C: {C}, time: {timer.min_elapsed}"
                        )
                else:
                    for _ in range(10):
                        with timer:
                            out_features = cat_to_batch(features, self.offsets, backend=backend)
                    print(f"Backend: {backend}, C: {C}, time: {timer.min_elapsed}")


if __name__ == "__main__":
    unittest.main()
