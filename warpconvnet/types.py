# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple, Union

import torch
from torch import Tensor

NestedTensor = torch.Tensor
# Float[IterableTensor, "N1 N2"] indicates a sequence of tensors with 2 dimensions
IterableTensor = Union[Tensor, List[Tensor], Tuple[Tensor, ...], NestedTensor]
