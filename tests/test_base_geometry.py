import unittest

import torch
import warp as wp

from warpconvnet.geometry.features.cat import CatFeatures
from warpconvnet.geometry.features.patch import CatPatchFeatures


class TestBaseGeometry(unittest.TestCase):
    def setUp(self):
        wp.init()
        self.B, min_N, max_N, self.C = 3, 1000, 10000, 7
        self.Ns = torch.randint(min_N, max_N, (self.B,))
        self.coords = [torch.rand((N, 3)) for N in self.Ns]
        self.features = [torch.rand((N, self.C)) for N in self.Ns]
        self.cat_features = CatFeatures(
            torch.cat(self.features, dim=0),
            torch.tensor([0] + list(torch.cumsum(self.Ns, dim=0).numpy())),
        )
        self.patch_size = 16

    def test_cat_to_patch(self):
        patch_features = CatPatchFeatures.from_cat(self.cat_features, self.patch_size)
        patch_features.clear_padding(-torch.inf)
        self.assertEqual(patch_features.num_channels, self.cat_features.num_channels)
        self.assertTrue((patch_features.offsets == self.cat_features.offsets).all())

        cat_features = patch_features.to_cat()
        self.assertTrue((cat_features.offsets == self.cat_features.offsets).all())
        self.assertTrue((cat_features.batched_tensor == self.cat_features.batched_tensor).all())


if __name__ == "__main__":
    unittest.main()
