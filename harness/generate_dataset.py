#!/usr/bin/env python3
"""
If the datasets are too large to include, generate them here or pull them 
from a storage source.
"""
# Copyright 2025 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from pathlib import Path
from mnist import mnist

def main():
    """
    Usage:  python3 generate_dataset.py  <output_file>
    """

    if len(sys.argv) != 2:
        sys.exit("Usage: generate_dataset.py <output_file>")

    DATASET_PATH = Path(sys.argv[1])
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

    mnist.export_test_data(output_file=DATASET_PATH, num_samples=10000, seed=None)


if __name__ == "__main__":
    main()
