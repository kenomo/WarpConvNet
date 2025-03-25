# # SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# # SPDX-License-Identifier: Apache-2.0

#!/bin/bash

{
    echo "## Directory Tree"
    echo "\`\`\`"
    git ls-files | tree --fromfile -F --dirsfirst
    echo "\`\`\`"
} | tee DIR_STRUCTURE.md
