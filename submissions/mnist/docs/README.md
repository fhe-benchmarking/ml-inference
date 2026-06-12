## Workload implementation – ml inference
--------------------------------------

The submission is built with [HEIR](heir.dev) compiler.

## Model architecture changes
We assume that server has access to training data and the model architecture has been modified as follows
- The first FC layer, of size 512x784
- The activation layer, using Approx-RELU based on polynomial appox

This network is trained to get a plaintext accuracy of approximately 95%.

## Compilation details
HEIR compiler is used to compile the encrypted function with the following optimizations:
- halevi-shoup matrix multiplications.
- approximate sign and approximate relu as specified in the `docs/mlp.mlir` file.

See further details of compilation [here](https://github.com/google/heir/blob/3ed0da33f81984b32a32f8490e2de1f07ed14c03/tests/Examples/openfhe/ckks/mnist/BUILD#L6)
