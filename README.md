# WarpConvNet

DISCLAIMER: THIS REPOSITORY IS NVIDIA INTERNAL/CONFIDENTIAL. DO NOT SHARE EXTERNALLY. IF YOU PLAN TO USE THIS CODEBASE FOR YOUR RESEARCH, PLEASE CONTACT CHRIS CHOY cchoy@nvidia.com


## Directory Structure

```
warpconvnet/
├── warpconvnet/
│   ├── core/
│   ├── dataset/        # dataset classes (e.g. ModelNet, ScanNet)
│   ├── geometry/       # geometry classes
│   │   ├── ops/        # primitive operations on geometry classes
│   ├── models/         # model classes (e.g. DGCNN, MinkowskiNet)
│   │   ├── configs/    # model configs
│   │   ├── examples/   # model examples (e.g. ModelNet, ScanNet)
│   ├── nn/             # neural network modules
│   │   ├── functional/ # functions
│   ├── ops/            # non geometry related operations
│   ├── utils/          # utility functions
├── docker/             # docker files for building containers
├── examples/           # example usage of the library
├── README.md
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
