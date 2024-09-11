import unittest

import torch

from warp.convnet.nn.encodings import SinusoidalEncoding
from warp.convnet.nn.functional.encodings import sinusoidal_encoding


class TestSinusoidalEncoding(unittest.TestCase):
    def setUp(self):
        self.encoding = SinusoidalEncoding(num_channels=10, data_range=2.0)
        self.B, self.D, self.N = 7, 3, 100000

    def test_sinusoidal_encoding_function(self):
        x = torch.rand(self.B, self.N, self.D)
        num_channels = 4
        encoded = sinusoidal_encoding(
            x,
            num_channels=num_channels,
            data_range=1.0,
            encoding_dim=-1,
        )
        # Check the distributions of the output to be even between -1, to 1
        # Count the number of values in each bin evenly spaced between -1, to 1
        counts = torch.histc(encoded, bins=num_channels * self.D, min=-1.0, max=1.0)
        # Check if the bins are even
        print(counts)
        self.assertEqual(encoded.shape, (self.B, self.N, num_channels * self.D))

    def test_sinusoidal_encoding_module(self):
        x = torch.rand(self.B, self.N, self.D)
        encoded = self.encoding(x)
        self.assertEqual(encoded.shape, (self.B, self.N, self.D * 10))


if __name__ == "__main__":
    unittest.main()
