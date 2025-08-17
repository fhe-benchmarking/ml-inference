#!/usr/bin/env python3

# Copyright (c) 2025 HomomorphicEncryption.org
# All rights reserved.
#
# This software is licensed under the terms of the Apache v2 License.
# See the LICENSE.md file for details.

"""
params.py - Parameters and directory structure for the submission.
"""

from pathlib import Path

# Enum for benchmark size
TOY = 0
SMALL = 1
MEDIUM = 2
LARGE = 3

def instance_name(size):
    """Return the string name of the instance size."""
    if size > LARGE:
        return "unknown"
    names = ["toy", "small", "medium", "large"]
    return names[size]

class InstanceParams:
    """Parameters that differ for different instance sizes."""

    def __init__(self, size, rootdir=None):
        """Constructor."""
        self.size = size
        self.rootdir = Path(rootdir) if rootdir else Path.cwd()

        if size > LARGE:
            raise ValueError("Invalid instance size")

        # parameters for sizes:   toy  small   medium     large
        q_bound =              [ 1,   10,     100,      1000]
        db_bound =             [10, 100,    1000,   10000]

        self.query_bound = q_bound[size]
        self.db_bound = db_bound[size]

    def get_size(self):
        """Return the instance size."""
        return self.size

    def get_query_bound(self):
        """Return the dimension of the plaintext record."""
        return self.query_bound

    def get_db_bound(self):
        """Return the number of records in the dataset."""
        return self.db_bound

    # Directory structure methods
    def subdir(self):
        """Return the submission directory of this repository."""
        return self.rootdir

    def datadir(self):
        """Return the dataset directory path."""
        return self.rootdir / "datasets" / instance_name(self.size)

    def iodir(self):
        """Return the I/O directory path."""
        return self.rootdir / "io" / instance_name(self.size)
    
    def measuredir(self):
        """Return the measurements directory path."""
        return self.rootdir / "measurements" / instance_name(self.size)