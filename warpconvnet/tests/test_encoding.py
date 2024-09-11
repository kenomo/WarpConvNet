import unittest

import torch

import warp as wp
from warpconvnet.nn.encoding import SinusoidalEncoding


class TestSinusoidalEncoding(unittest.TestCase):
    def test_encoding(self):
        num_channels = 16
        data_range = 1.0
        N, C = 10, 3
        encoding = SinusoidalEncoding(num_channels, data_range)
        x = torch.rand((N, C))
        y = encoding(x)
        self.assertTrue(y.shape == (N, C * num_channels))


if __name__ == "__main__":
    wp.init()
    unittest.main()
