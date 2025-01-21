# WarpConvNet

DISCLAIMER: THIS REPOSITORY IS NVIDIA INTERNAL/CONFIDENTIAL. DO NOT SHARE EXTERNALLY. IF YOU PLAN TO USE THIS CODEBASE FOR YOUR RESEARCH, PLEASE CONTACT CHRIS CHOY cchoy@nvidia.com

## Directory Structure

```
warpconvnet/
├── warpconvnet/
│   ├── geometry/           # Geometry implementations
│   │   ├── base/           # Core definitions
│   │   │   ├── batched.py  # Base batched tensor class
│   │   │   ├── coords.py   # Base coordinates class
│   │   │   ├── features.py # Base features class
│   │   │   └── geometry.py # Base geometry class
│   │   ├── coords/         # Coordinate implementations
│   │   │   ├── search/     # Search operations
│   │   │   ├── ops/        # Coordinate operations
│   │   │   │   ├── batch_index.py
│   │   │   │   ├── sample.py
│   │   │   │   └── voxel.py
│   │   │   ├── integer.py  # Integer coordinates
│   │   │   └── real.py     # Real coordinates
│   │   ├── features/       # Feature implementations
│   │   ├── types/          # Geometry types (Points, Voxels)
│   │   │   ├── points.py   # Point types (PointCloud)
│   │   │   └── voxels.py   # Voxel types (Spatially Sparse Tensor)
│   │   └── utils/          # Geometry utilities
│   │
│   ├── nn/                 # Neural network implementations
│   │   ├── functional/     # Functional implementations
│   │   │   ├── sparse_conv.py
│   │   │   ├── sparse_coords_ops.py
│   │   │   └── sparse_pool.py
│   │   ├── base_module.py  # Base neural network modules
│   │   └── sparse_conv.py  # Sparse convolution modules
│   │
│   ├── ops/                # General operations
│   │   ├── batch_copy.py
│   │   └── reductions.py
│   │
│   ├── utils/              # General utilities
│   │   ├── argsort.py
│   │   ├── ntuple.py
│   │   ├── ravel.py
│   │   └── unique.py
│   │
│   ├── dataset/            # Data loader implementations
│   │
│   └── models/             # Model implementations
│       ├── configs/        # Model configurations
│       └── examples/       # Usage examples
│
├── docker/                 # Docker build files
├── examples/               # Example scripts
├── tests/                  # Test suite
└── README.md
```

## Model Submodule

Model submodule contains popular models and configs. Please see the README.md in the subdirectory for more details.

## Installation

```bash
git clone --recurse-submodules https://gitlab-master.nvidia.com/3dmmllm/warp.git warpconvnet
cd warpconvnet
pip install -e .
```

## Example

### ScanNet Semantic Segmentation

```bash
cd warpconvnet/models
python examples/scannet.py train.batch_size=12 model=mink_unet
```

## Docker Usage

```bash
cd warpconvnet/docker
docker build \
    -t warpconvnet \
    .
docker run \
    --gpus all \
    --shm-size=32g \
    -it \
    -v "/home/${USER}:/root" \
    -v "$(pwd):/workspace" \
    warpconvnet:latest
```
