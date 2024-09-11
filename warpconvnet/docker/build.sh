DEFAULT_IMAGE_NAME="gitlab-master.nvidia.com/3dmmllm/warp"
DEFAULT_IMAGE_TAG="latest"
DEFAULT_IMAGE_FULL_NAME="${DEFAULT_IMAGE_NAME}:${DEFAULT_IMAGE_TAG}"

# Use DEFAULT_IMAGE_FULL_NAME if no argument is given, otherwise use the argument
IMAGE_FULL_NAME="${1:-${DEFAULT_IMAGE_FULL_NAME}}"

DOCKER_DIR=$(dirname $(realpath -s $0))
WARP_DIR=$(realpath -s ${DOCKER_DIR}/../../..)

echo -e "\e[0;32m"
echo "Building image: ${IMAGE_FULL_NAME}"
echo -e "\e[0m"

docker build \
    -t ${IMAGE_FULL_NAME}   \
    --network=host          \
    -f ${DOCKER_DIR}/Dockerfile \
    ${WARP_DIR}

docker push ${DEFAULT_IMAGE_FULL_NAME}
