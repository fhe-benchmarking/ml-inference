#!/usr/bin/env python3
"""
verify_result.py - correctness oracle for ML Inference workload.
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

def main():

    """
    Usage:  python3 verify_result.py  <expected_file>  <result_file>
    Returns exit-code 0 if equal, 1 otherwise.
    Prints a message so the caller can log it.
    """

    if len(sys.argv) != 3:
        sys.exit("Usage: verify_result.py <expected> <result>")

    expected_file = Path(sys.argv[1])
    result_file   = Path(sys.argv[2])

    try:
        exp = int(expected_file.read_text().strip())
        res = int(result_file.read_text().strip())
    except Exception as e:
        print(f"[harness] failed to read files: {e}")
        sys.exit(1)

    if res == exp:
        print(f"[harness] PASS  (expected={exp}, got={res})")
        sys.exit(0)
    else:
        print(f"[harness] FAIL  (expected={exp}, got={res})")
        sys.exit(1)

if __name__ == "__main__":
    main()