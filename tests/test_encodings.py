import unittest

import torch

from warpconvnet.nn.encodings import RelativeCoordsEncoding, SinusoidalEncoding
from warpconvnet.nn.functional.encodings import get_freqs, sinusoidal_encoding


class TestSinusoidalEncoding(unittest.TestCase):
    def setUp(self):
        self.B, self.D, self.N = 7, 3, 100000

    def test_sinusoidal_encoding_function(self):
        x = torch.rand(self.B, self.N, self.D)
        num_channels = 4
        encoded = sinusoidal_encoding(
            x,
            num_channels=num_channels,
            data_range=1.0,
            encoding_axis=-1,
            concat_input=False,
        )
        # Check the distributions of the output to be even between -1, to 1
        # Count the number of values in each bin evenly spaced between -1, to 1
        counts = torch.histc(encoded, bins=num_channels * self.D, min=-1.0, max=1.0)
        # Check if the bins are even
        print(counts)
        self.assertEqual(encoded.shape, (self.B, self.N, num_channels * self.D))

    def test_sinusoidal_encoding_module(self):
        x = torch.rand(self.B, self.N, self.D)
        encoding = SinusoidalEncoding(num_channels=10, data_range=2.0, concat_input=True)
        encoded = encoding(x)
        self.assertEqual(encoded.shape, (self.B, self.N, self.D * 10 + self.D))


if __name__ == "__main__":
    unittest.main()
