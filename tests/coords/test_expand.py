import pytest
import torch
import warp as wp

from warpconvnet.geometry.coords.ops.expand import expand_coords
from warpconvnet.geometry.types.voxels import Voxels


@pytest.fixture
def setup_voxels():
    """Setup test voxels with random coordinates."""
    wp.init()
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, min_N, max_N, C = 3, 100000, 1000000, 7
    Ns = torch.randint(min_N, max_N, (B,))
    voxel_size = 0.01
    coords = [(torch.rand((N, 3)) / voxel_size).int() for N in Ns]
    features = [torch.rand((N, C)) for N in Ns]
    voxels = Voxels(coords, features, device=device).unique()
    return voxels


def test_expand_coords(setup_voxels):
    """Test coordinate expansion functionality."""
    voxels = setup_voxels

    up_coords, offsets = expand_coords(
        voxels.batch_indexed_coordinates,
        kernel_size=(3, 3, 3),
        kernel_dilation=(1, 1, 1),
    )

    # Test output properties
    assert up_coords.shape[0] > voxels.coordinate_tensor.shape[0]
    assert offsets.shape == (voxels.batch_size + 1,)


@pytest.mark.parametrize("kernel_batch", [None, 9, 27])
def test_expand_kernel_batch(setup_voxels, kernel_batch):
    """Test expansion with different kernel batch sizes."""
    voxels = setup_voxels

    up_coords, offsets = expand_coords(
        voxels.batch_indexed_coordinates,
        kernel_size=(3, 3, 3),
        kernel_dilation=(1, 1, 1),
        kernel_batch=kernel_batch,
    )

    # Results should be the same regardless of batch size
    ref_coords, ref_offsets = expand_coords(
        voxels.batch_indexed_coordinates,
        kernel_size=(3, 3, 3),
        kernel_dilation=(1, 1, 1),
    )

    assert torch.equal(up_coords, ref_coords)
    assert torch.equal(offsets, ref_offsets)
