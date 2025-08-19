## Workload implementation – ml inference
--------------------------------------

The submission is built with [HEIR](heir.dev) compiler.

## Model architecture changes
We assume that server has access to training data and the model architecture has been modified as follows
- The first FC layer, of size 1024x1024
- The activation layer, using Approx-RELU based on polynomial appox
- The second FC layer, of size 1024x1024
The input is padded and output is post porcessed to get the label.

This network is trained to get a plaintext accuracy of 9634/10000.

## Compilation details
HEIR compiler is used to compile the encrypted function with the following optimizations:
- halevi-shoup matrix multiplications.
- approximate sign and approximate relu as specified in the `docs/mlp.mlir` file.

See further details of compilation https://github.com/google/heir/issues/1232

## Build details
Weights have been inlined in the `mlp_openfhe.cpp` file. A precompiled library is provided as build takes ~3 hours.