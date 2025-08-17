#!/usr/bin/env bash

# Copyright (c) 2025 HomomorphicEncryption.org
# All rights reserved.
#
# This software is licensed under the terms of the Apache v2 License.
# See the LICENSE.md file for details.

# ------------------------------------------------------------
# Usage: ./scripts/build_task.sh 
# Compiles the files in the source directory.
# ------------------------------------------------------------
set -euo pipefail
ROOT="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )"
TASK_DIR="$1"
BUILD="$TASK_DIR/build"
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu || echo 4)

# By default, we assume the OpenFHE library is installed at the the local 
# directory /third_party/openfhe (the default location in get_openfhe.sh).
# If you want to use a different location, set the CMAKE_PREFIX_PATH variable
# accordingly.
cmake -S "$TASK_DIR" -B "$BUILD" \
      -DCMAKE_PREFIX_PATH="$ROOT/third_party/openfhe"
cd "$TASK_DIR/build"
make -j"$NPROC"