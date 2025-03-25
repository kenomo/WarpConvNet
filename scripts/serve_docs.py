# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""Serve documentation locally for WarpConvNet."""

import os
import subprocess
import sys
from pathlib import Path


def serve_docs():
    """Serve documentation using MkDocs development server."""
    try:
        # Ensure we're in the project root
        project_root = Path(__file__).parent.parent
        os.chdir(project_root)

        # Serve the documentation
        subprocess.run(["mkdocs", "serve"], check=True)
        return 0

    except subprocess.CalledProcessError as e:
        print("Error serving documentation:", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(serve_docs())
