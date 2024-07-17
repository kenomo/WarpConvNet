import unittest

# Layers
from warp.convnet.tests.test_encoding import TestSinusoidalEncoding
from warp.convnet.tests.test_neighbor_search_discrete import TestNeighborSearchDiscrete

# Points
from warp.convnet.tests.test_point_collection import TestPointCollection
from warp.convnet.tests.test_point_conv import TestPointConv

# Sparse Tensors
from warp.convnet.tests.test_spatially_sparse_tensor import TestSpatiallySparseTensor

# All utility
from warp.convnet.tests.test_utils import TestUtils


def test_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestUtils))

    suite.addTest(unittest.makeSuite(TestPointCollection))
    suite.addTest(unittest.makeSuite(TestPointConv))

    suite.addTest(unittest.makeSuite(TestSpatiallySparseTensor))
    suite.addTest(unittest.makeSuite(TestNeighborSearchDiscrete))

    suite.addTest(unittest.makeSuite(TestSinusoidalEncoding))
    return suite


if __name__ == "__main__":
    unittest.main(defaultTest="test_suite")
