from dataclasses import dataclass
from warpconvnet.geometry.base.batched import BatchedTensor
from warpconvnet.geometry.base import Coords, Features, Geometry
from warpconvnet.geometry.features import (
    CatFeatures,
    PadFeatures,
    CatPatchFeatures,
    PadPatchFeatures,
)


def test_features_import():
    """Test that imports work. The file structure could cause circular import errors."""
    assert issubclass(Features, BatchedTensor)
    assert issubclass(CatFeatures, Features)
    assert issubclass(PadFeatures, Features)
    assert issubclass(CatPatchFeatures, Features)
    assert issubclass(PadPatchFeatures, Features)


def test_coords_import():
    assert issubclass(Coords, BatchedTensor)


def test_geometry_import():
    pass
