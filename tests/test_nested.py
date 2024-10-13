import unittest

import torch
import warp as wp

from warpconvnet.geometry.base_geometry import (
    BatchedCoordinates,
    CatBatchedFeatures,
    NestBatchedObject,
)


class TestBaseGeometry(unittest.TestCase):
    def setUp(self):
        wp.init()
        self.B, min_N, max_N, self.C = 3, 1000, 10000, 7
        self.Ns = torch.randint(min_N, max_N, (self.B,))
        self.coords = [torch.rand((N, 3)) for N in self.Ns]
        self.features = [torch.rand((N, self.C)) for N in self.Ns]
        self.patch_size = 16

    def test_nested_coords(self):
        nested_coords = torch.nested.nested_tensor(self.coords)
        batched_coordinates = BatchedCoordinates.from_nested(nested_coords)
        for i in range(self.B):
            self.assertTrue((batched_coordinates[i] == self.coords[i]).all())

    def test_base_to_nested(self):
        cat_features = CatBatchedFeatures(self.features)
        nested_features = cat_features.to_nested()
        for i in range(self.B):
            self.assertTrue((nested_features[i] == self.features[i]).all())

    def test_nest_batched_object(self):
        cat_features = CatBatchedFeatures(self.features)
        nested_features = NestBatchedObject(cat_features.to_nested())
        for i in range(self.B):
            self.assertTrue((nested_features[i] == self.features[i]).all())


if __name__ == "__main__":
    unittest.main()
