from typing import Callable

import torch.nn.functional as F

from warpconvnet.geometry.base_geometry import BatchedFeatures
from warpconvnet.geometry.point_collection import PointCollection
from warpconvnet.geometry.spatially_sparse_tensor import SpatiallySparseTensor


def apply_feature_transform(
    input: SpatiallySparseTensor | PointCollection,
    transform: Callable,
):
    assert isinstance(input, SpatiallySparseTensor) or isinstance(input, PointCollection)
    return input.replace(
        batched_features=BatchedFeatures(transform(input.feature_tensor), offsets=input.offsets),
    )


def create_activation_function(torch_func):
    def wrapper(input: SpatiallySparseTensor | PointCollection):
        return apply_feature_transform(input, torch_func)

    return wrapper


# Instantiate common activation functions
relu = create_activation_function(F.relu)
gelu = create_activation_function(F.gelu)
silu = create_activation_function(F.silu)
tanh = create_activation_function(F.tanh)
sigmoid = create_activation_function(F.sigmoid)
leaky_relu = create_activation_function(F.leaky_relu)
elu = create_activation_function(F.elu)
softmax = create_activation_function(F.softmax)
log_softmax = create_activation_function(F.log_softmax)


# Normalization functions
def create_norm_function(torch_norm_func):
    def wrapper(input: SpatiallySparseTensor | PointCollection, *args, **kwargs):
        return apply_feature_transform(input, lambda x: torch_norm_func(x, *args, **kwargs))

    return wrapper


# Instantiate common normalization functions
layer_norm = create_norm_function(F.layer_norm)
# layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5)
batch_norm = create_norm_function(F.batch_norm)
# batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5)
instance_norm = create_norm_function(F.instance_norm)
# instance_norm(input, running_mean=None, running_var=None, weight=None, bias=None, use_input_stats=True, momentum=0.1, eps=1e-5)
group_norm = create_norm_function(F.group_norm)
# group_norm(input, num_groups, weight=None, bias=None, eps=1e-5)
