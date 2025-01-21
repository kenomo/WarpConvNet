# WarpConvNet

DISCLAIMER: THIS REPOSITORY IS NVIDIA INTERNAL/CONFIDENTIAL. DO NOT SHARE EXTERNALLY. IF YOU PLAN TO USE THIS CODEBASE FOR YOUR RESEARCH, PLEASE CONTACT CHRIS CHOY cchoy@nvidia.com

## Overview

WarpConvNet is a high-performance library for 3D deep learning, built on NVIDIA's Warp framework. It provides efficient implementations of:

- Point cloud processing
- Sparse voxel convolutions
- Attention mechanisms for 3D data
- Geometric operations and transformations

## Installation

```bash
# Clone repository with submodules
git clone --recurse-submodules https://gitlab-master.nvidia.com/3dmmllm/warp.git warpconvnet
cd warpconvnet

# Install package
pip install -e .
```

## Directory Structure

```
./
├── docker/             # Docker build files
│   ├── build.sh
│   └── Dockerfile
├── examples/           # Example applications
│   ├── mnist.py
│   └── modelnet.py
├── scripts/            # Development utilities
│   ├── build_docs.py
│   ├── dir_struct.sh
│   └── serve_docs.py
├── tests/              # Test suite
│   ├── base/           # Core functionality tests
│   ├── coords/         # Coordinate operation tests
│   ├── features/       # Feature processing tests
│   ├── nn/             # Neural network tests
│   └── types/          # Geometry type tests
└── warpconvnet/        # Main package
    ├── geometry/       # Geometric operations
    │   ├── base/       # Core definitions
    │   ├── coords/     # Coordinate operations
    │   ├── features/   # Feature operations
    │   └── types/      # Geometry types
    ├── nn/             # Neural networks
    │   ├── functional/ # Neural network functions
    │   └── modules/    # Neural network modules
    ├── ops/            # Basic operations
    └── utils/          # Utility functions
```

For complete directory structure, run `bash scripts/dir_struct.sh`.

## Quick Start

### ScanNet Semantic Segmentation

```bash
cd warpconvnet/models
python examples/scannet.py train.batch_size=12 model=mink_unet
```

### ModelNet Classification

```bash
python examples/modelnet.py
```

## Docker Usage

Build and run with GPU support:

```bash
# Build container
cd warpconvnet/docker
docker build -t warpconvnet .

# Run container
docker run --gpus all \
    --shm-size=32g \
    -it \
    -v "/home/${USER}:/root" \
    -v "$(pwd):/workspace" \
    warpconvnet:latest
```

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/nn/
pytest tests/coords/

# Run with benchmarks
pytest tests/ --benchmark-only
```

### Building Documentation

```bash
# Build docs
python scripts/build_docs.py

# Serve locally
python scripts/serve_docs.py
```

## License

NVIDIA Proprietary. See DISCLAIMER at top of README.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{warpconvnet2024,
  author = {Chris Choy and NVIDIA Research},
  title = {WarpConvNet: High-Performance 3D Deep Learning Library},
  year = {2024},
  publisher = {NVIDIA Corporation},
  howpublished = {\url{https://gitlab-master.nvidia.com/3dmmllm/warp}}
}
```
