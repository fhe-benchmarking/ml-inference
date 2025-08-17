#!/usr/bin/env python3

# Copyright (c) 2025 HomomorphicEncryption.org
# All rights reserved.
#
# This software is licensed under the terms of the Apache v2 License.
# See the LICENSE.md file for details.

"""
Generate a new query for each run.
"""
import numpy as np
from pathlib import Path
from utils import parse_submission_arguments

def main():
    """
    Generate random value representing the query in the workload.
    """
    __, params, seed, __, __ = parse_submission_arguments('Generate query for FHE benchmark.')
    DATASET_Q_PATH = params.datadir() / f"query.txt"
    DATASET_Q_PATH.parent.mkdir(parents=True, exist_ok=True)
    bound = params.get_query_bound()

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    q = round(np.random.uniform(-bound, bound), 2)
    DATASET_Q_PATH.write_text(f"{q}\n", encoding="utf-8")

if __name__ == "__main__":
    main()
