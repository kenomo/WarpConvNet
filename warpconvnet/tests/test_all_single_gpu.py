import unittest

# Layers
from warp.convnet.tests.test_encoding import TestSinusoidalEncoding
from warp.convnet.tests.test_neighbor_search_discrete import TestNeighborSearchDiscrete

# Points
from warp.convnet.tests.test_point_collection import TestPointCollection
from warp.convnet.tests.test_point_conv import TestPointConv

# Network
from warp.convnet.tests.test_point_conv_enc_dec import TestPointConvEncoder
from warp.convnet.tests.test_point_conv_unet import TestPointConvUNet
from warp.convnet.tests.test_point_pool import TestPointPool
from warp.convnet.tests.test_sort import TestSorting

# Sparse Tensors
from warp.convnet.tests.test_spatially_sparse_tensor import TestSpatiallySparseTensor

# All utility
from warp.convnet.tests.test_utils import TestUtils

# Low-level ops
from warp.convnet.tests.test_voxel_ops import TestVoxelOps


def test_suite():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add tests using the loader
    suite.addTests(loader.loadTestsFromTestCase(TestUtils))

    suite.addTests(loader.loadTestsFromTestCase(TestPointCollection))
    suite.addTests(loader.loadTestsFromTestCase(TestPointConv))
    suite.addTests(loader.loadTestsFromTestCase(TestPointPool))

    suite.addTests(loader.loadTestsFromTestCase(TestSpatiallySparseTensor))
    suite.addTests(loader.loadTestsFromTestCase(TestNeighborSearchDiscrete))

    suite.addTests(loader.loadTestsFromTestCase(TestSinusoidalEncoding))

    suite.addTests(loader.loadTestsFromTestCase(TestPointConvEncoder))
    suite.addTests(loader.loadTestsFromTestCase(TestPointConvUNet))

    suite.addTests(loader.loadTestsFromTestCase(TestVoxelOps))
    suite.addTests(loader.loadTestsFromTestCase(TestSorting))

    return suite


if __name__ == "__main__":
    unittest.main(defaultTest="test_suite")
