#!/usr/bin/env python3

# Copyright (c) 2025 HomomorphicEncryption.org
# All rights reserved.
#
# This software is licensed under the terms of the Apache v2 License.
# See the LICENSE.md file for details.
#============================================================================

"""
Cleartext reference for the “add” workload.
For each test case:
    - Reads the dataset and the query
    - Computes the sum between the two 
    - Writes the result to expected.txt for each test case (# datasets/xxx/expected.txt)
"""
import random
from pathlib import Path
from utils import parse_submission_arguments

def main():

    __, params, __, __, __ = parse_submission_arguments('Generate dataset for FHE benchmark.')
    DATASET_DB_PATH = params.datadir() / f"db.txt"
    DATASET_Q_PATH = params.datadir() / f"query.txt"

    # 1) load database and the query if not already stored
    db = float(DATASET_DB_PATH.read_text().strip())
    q = float(DATASET_Q_PATH.read_text().strip())

    # 2) compute reference result
    result = db + q

    # 3) write to expected.txt (overwrites if it already exists)
    OUT_PATH = params.datadir() / f"expected.txt"
    OUT_PATH.write_text(f"{result}\n", encoding="utf-8")

if __name__ == "__main__":
    main()