# Warp ConvNet

DISCLAIMER: THIS REPOSITORY IS NVIDIA INTERNAL/CONFIDENTIAL. DO NOT SHARE EXTERNALLY. IF YOU PLAN TO USE THIS CODEBASE FOR YOUR RESEARCH, PLEASE CONTACT CHRIS CHOY cchoy@nvidia.com


## Directory Structure

- geometry: defines a set of geometry base classes
- geometry/ops: primitive operations on geometry base classes. To prevent circular dependencies, do not import geometry classes.
- nn: defines neural network modules that process geometry classes.
- ops: non geometry related operations