#!/usr/bin/env python3

# Copyright (c) 2025 HomomorphicEncryption.org
# All rights reserved.
#
# This software is licensed under the terms of the Apache v2 License.
# See the LICENSE.md file for details.

"""
If the datasets are too large to include, generate them here or pull them 
from a storage source.
"""
import numpy as np
from pathlib import Path
from utils import parse_submission_arguments

def main():
    """
    Generate random value representing the database in the workload.
    """
    __, params, seed, __, __ = parse_submission_arguments('Generate dataset for FHE benchmark.')
    DATASET_DB_PATH = params.datadir() / f"db.txt"
    DATASET_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    bound = params.get_db_bound()

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    db = round(np.random.uniform(-bound, bound), 2)
    DATASET_DB_PATH.write_text(f"{db}\n", encoding="utf-8")

if __name__ == "__main__":
    main()