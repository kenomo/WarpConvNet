#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

DEFAULT_IMAGE_NAME="github.com/nvlabs/warpconvnet"
DEFAULT_IMAGE_TAG="latest"
DEFAULT_IMAGE_FULL_NAME="${DEFAULT_IMAGE_NAME}:${DEFAULT_IMAGE_TAG}"

# Use DEFAULT_IMAGE_FULL_NAME if no argument is given, otherwise use the argument
IMAGE_FULL_NAME="${1:-${DEFAULT_IMAGE_FULL_NAME}}"

DOCKER_DIR=$(dirname "$(realpath -s "$0")")
WARPCONVNET_DIR=$(realpath -s "${DOCKER_DIR}"/..)

echo -e "\e[0;32m"
echo "Building image: ${IMAGE_FULL_NAME}"
echo -e "\e[0m"

docker build \
    -t "${IMAGE_FULL_NAME}" \
    --network=host \
    -f "${DOCKER_DIR}"/Dockerfile \
    "${WARPCONVNET_DIR}"

# Run docker with all GPUs if available and test importing warpconvnet
if [ -z "$(nvidia-smi -L | wc -l)" ]; then
    echo -e "\e[0;32m"
    echo "No GPUs found. Test import only on CPU."
    echo -e "\e[0m"
    docker run \
        -it --rm \
        --name warpconvnet-test \
        "${IMAGE_FULL_NAME}" \
        bash -c 'python -c "import warpconvnet; import warpconvnet._C; print(dir(warpconvnet._C));"'
else
    echo -e "\e[0;32m"
    echo "Running test with all GPUs."
    echo -e "\e[0m"
    docker run \
        -it --rm --gpus all \
        --name warpconvnet-test \
        "${IMAGE_FULL_NAME}" \
        bash -c 'python -c "import warpconvnet; import warpconvnet._C; print(dir(warpconvnet._C));" && cd /opt/warpconvnet/WarpConvNet && pytest tests/csrc/test_cutlass_gemm.py'
fi
