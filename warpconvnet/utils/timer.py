# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

import numpy as np


class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.min_time = np.inf

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()
        if self.elapsed < self.min_time:
            self.min_time = self.elapsed

    @property
    def elapsed(self):
        return self.end_time - self.start_time

    @property
    def min_elapsed(self):
        return self.min_time

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        return False
