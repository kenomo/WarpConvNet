import unittest

import torch
import warp as wp

from warpconvnet.geometry.features.convert import cat_to_pad_tensor
from warpconvnet.utils.timer import Timer


def run_batch_copy_test(B, min_N, max_N):
    # Set random seed
    wp.init()
    torch.manual_seed(0)

    Ns = torch.randint(min_N, max_N, (B,))
    offsets = torch.cat(
        [torch.zeros(1, dtype=torch.int32), torch.cumsum(torch.tensor(Ns), dim=0)], dim=0
    ).int()

    device = "cuda:0"
    for C in [16, 32, 64, 128, 256]:
        features = torch.cat([torch.rand((N, C)) for N in Ns], dim=0).to(device)
        for backend in ["torch", "warp"]:
            timer = Timer()
            if backend == "warp":
                for num_copy_per_thread in [None, 256]:
                    for _ in range(10):
                        with timer:
                            out_features = cat_to_pad_tensor(
                                features,
                                offsets,
                                backend=backend,
                                num_copy_per_thread=num_copy_per_thread,
                            )
                            torch.cuda.synchronize()
                    print(
                        f"B: {B}, min_N: {min_N}, max_N: {max_N}, Backend: {backend}, num_copy_per_thread: {num_copy_per_thread}, C: {C}, time: {timer.min_elapsed}"
                    )
            else:
                for _ in range(10):
                    with timer:
                        out_features = cat_to_pad_tensor(features, offsets, backend=backend)
                        # synchronize
                        torch.cuda.synchronize()
                print(
                    f"B: {B}, min_N: {min_N}, max_N: {max_N}, Backend: {backend}, C: {C}, time: {timer.min_elapsed}"
                )


class TestBatchCopy(unittest.TestCase):

    def test_batch_small(self):
        """
        B: 48, min_N: 1000, max_N: 2000, Backend: torch, C: 16, time: 0.0019376277923583984
        B: 48, min_N: 1000, max_N: 2000, Backend: warp, num_copy_per_thread: None, C: 16, time: 0.00028204917907714844
        B: 48, min_N: 1000, max_N: 2000, Backend: warp, num_copy_per_thread: 256, C: 16, time: 0.00028204917907714844
        B: 48, min_N: 1000, max_N: 2000, Backend: torch, C: 32, time: 0.0019314289093017578
        B: 48, min_N: 1000, max_N: 2000, Backend: warp, num_copy_per_thread: None, C: 32, time: 0.0008053779602050781
        B: 48, min_N: 1000, max_N: 2000, Backend: warp, num_copy_per_thread: 256, C: 32, time: 0.0008053779602050781
        B: 48, min_N: 1000, max_N: 2000, Backend: torch, C: 64, time: 0.0019297599792480469
        B: 48, min_N: 1000, max_N: 2000, Backend: warp, num_copy_per_thread: None, C: 64, time: 0.002134561538696289
        B: 48, min_N: 1000, max_N: 2000, Backend: warp, num_copy_per_thread: 256, C: 64, time: 0.0012700557708740234
        B: 48, min_N: 1000, max_N: 2000, Backend: torch, C: 128, time: 0.0019414424896240234
        B: 48, min_N: 1000, max_N: 2000, Backend: warp, num_copy_per_thread: None, C: 128, time: 0.004393339157104492
        B: 48, min_N: 1000, max_N: 2000, Backend: warp, num_copy_per_thread: 256, C: 128, time: 0.004393339157104492
        B: 48, min_N: 1000, max_N: 2000, Backend: torch, C: 256, time: 0.002007722854614258
        B: 48, min_N: 1000, max_N: 2000, Backend: warp, num_copy_per_thread: None, C: 256, time: 0.01200103759765625
        B: 48, min_N: 1000, max_N: 2000, Backend: warp, num_copy_per_thread: 256, C: 256, time: 0.01200103759765625
        """
        run_batch_copy_test(B=48, min_N=1_000, max_N=2_000)

    def test_batch_medium(self):
        """
        B: 12, min_N: 10000, max_N: 20000, Backend: torch, C: 16, time: 0.0005414485931396484
        B: 12, min_N: 10000, max_N: 20000, Backend: warp, num_copy_per_thread: None, C: 16, time: 0.0004763603210449219
        B: 12, min_N: 10000, max_N: 20000, Backend: warp, num_copy_per_thread: 256, C: 16, time: 0.0004763603210449219
        B: 12, min_N: 10000, max_N: 20000, Backend: torch, C: 32, time: 0.0005528926849365234
        B: 12, min_N: 10000, max_N: 20000, Backend: warp, num_copy_per_thread: None, C: 32, time: 0.0022835731506347656
        B: 12, min_N: 10000, max_N: 20000, Backend: warp, num_copy_per_thread: 256, C: 32, time: 0.0011730194091796875
        B: 12, min_N: 10000, max_N: 20000, Backend: torch, C: 64, time: 0.0005540847778320312
        B: 12, min_N: 10000, max_N: 20000, Backend: warp, num_copy_per_thread: None, C: 64, time: 0.006055355072021484
        B: 12, min_N: 10000, max_N: 20000, Backend: warp, num_copy_per_thread: 256, C: 64, time: 0.006055355072021484
        B: 12, min_N: 10000, max_N: 20000, Backend: torch, C: 128, time: 0.0006361007690429688
        B: 12, min_N: 10000, max_N: 20000, Backend: warp, num_copy_per_thread: None, C: 128, time: 0.012160062789916992
        B: 12, min_N: 10000, max_N: 20000, Backend: warp, num_copy_per_thread: 256, C: 128, time: 0.012160062789916992
        B: 12, min_N: 10000, max_N: 20000, Backend: torch, C: 256, time: 0.0011858940124511719
        B: 12, min_N: 10000, max_N: 20000, Backend: warp, num_copy_per_thread: None, C: 256, time: 0.03223538398742676
        B: 12, min_N: 10000, max_N: 20000, Backend: warp, num_copy_per_thread: 256, C: 256, time: 0.03223538398742676
        """
        run_batch_copy_test(B=12, min_N=10_000, max_N=20_000)

    def test_batch_large(self):
        """
        B: 4, min_N: 100000, max_N: 1000000, Backend: torch, C: 16, time: 0.0008306503295898438
        B: 4, min_N: 100000, max_N: 1000000, Backend: warp, num_copy_per_thread: None, C: 16, time: 0.007839202880859375
        B: 4, min_N: 100000, max_N: 1000000, Backend: warp, num_copy_per_thread: 256, C: 16, time: 0.007839202880859375
        B: 4, min_N: 100000, max_N: 1000000, Backend: torch, C: 32, time: 0.001584768295288086
        B: 4, min_N: 100000, max_N: 1000000, Backend: warp, num_copy_per_thread: None, C: 32, time: 0.01912522315979004
        B: 4, min_N: 100000, max_N: 1000000, Backend: warp, num_copy_per_thread: 256, C: 32, time: 0.01912522315979004
        B: 4, min_N: 100000, max_N: 1000000, Backend: torch, C: 64, time: 0.003114938735961914
        B: 4, min_N: 100000, max_N: 1000000, Backend: warp, num_copy_per_thread: None, C: 64, time: 0.04689621925354004
        B: 4, min_N: 100000, max_N: 1000000, Backend: warp, num_copy_per_thread: 256, C: 64, time: 0.04689621925354004
        B: 4, min_N: 100000, max_N: 1000000, Backend: torch, C: 128, time: 0.0061724185943603516
        B: 4, min_N: 100000, max_N: 1000000, Backend: warp, num_copy_per_thread: None, C: 128, time: 0.09880447387695312
        B: 4, min_N: 100000, max_N: 1000000, Backend: warp, num_copy_per_thread: 256, C: 128, time: 0.09880447387695312
        B: 4, min_N: 100000, max_N: 1000000, Backend: torch, C: 256, time: 0.012264251708984375
        B: 4, min_N: 100000, max_N: 1000000, Backend: warp, num_copy_per_thread: None, C: 256, time: 0.2668890953063965
        B: 4, min_N: 100000, max_N: 1000000, Backend: warp, num_copy_per_thread: 256, C: 256, time: 0.2668890953063965
        """
        run_batch_copy_test(B=4, min_N=100_000, max_N=1_000_000)


if __name__ == "__main__":
    unittest.main()
