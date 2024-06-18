# Use dockerhub pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel


# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    vim \
    htop \
    mc \
    tmux \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
    "jaxtyping>=0.2"        \
    "torch_scatter>=2.1"    \
    "torchinfo>=1.8"        \
    "warp-lang>=1.2"        \
    "webdataset>=0.2"       \
    pre-commit              \
    black                   \
    isort                   \
    flake8

# Add a non-root user with a fixed UID and GID
ARG USERNAME=du
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN set -eux; \
    groupadd --gid $USER_GID $USERNAME; \
    useradd --uid $USER_UID --gid $USER_GID --no-log-init -m -G video $USERNAME

# Add sudo and allow the non-root user to execute commands as root
# without a password.
RUN apt-get update && apt-get install -y \
    sudo;
RUN echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME; \
    chmod 0440 /etc/sudoers.d/$USERNAME;
