import unittest

# Layers
from tests.test_encodings import TestSinusoidalEncoding

# Common
from tests.test_global_pool import TestGlobalPool
from tests.test_neighbor_search_discrete import TestNeighborSearchDiscrete

# Points
from tests.test_point_collection import TestPointCollection
from tests.test_point_conv import TestPointConv
from tests.test_point_pool import TestPointPool

# All utility
from tests.test_sort import TestSorting
from tests.test_sparse_conv import TestSparseConv
from tests.test_sparse_coords_ops import TestSparseOps
from tests.test_sparse_pool import TestSparsePool

# Sparse Tensors
from tests.test_spatially_sparse_tensor import TestSpatiallySparseTensor
from tests.test_utils import TestUtils

# Low-level ops
from tests.test_voxel_ops import TestVoxelOps


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
    suite.addTests(loader.loadTestsFromTestCase(TestSparseConv))
    suite.addTests(loader.loadTestsFromTestCase(TestSparseOps))
    suite.addTests(loader.loadTestsFromTestCase(TestSparsePool))

    suite.addTests(loader.loadTestsFromTestCase(TestSinusoidalEncoding))

    suite.addTests(loader.loadTestsFromTestCase(TestVoxelOps))
    suite.addTests(loader.loadTestsFromTestCase(TestSorting))
    suite.addTests(loader.loadTestsFromTestCase(TestGlobalPool))
    return suite


if __name__ == "__main__":
    unittest.main(defaultTest="test_suite")
