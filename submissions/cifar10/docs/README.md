# ResNet-20: Privacy-Preserving Encrypted Inference

This submission implements a ResNet-20 deep residual network using the FHEON framework for privacy-preserving machine learning inference on CIFAR-10 images.

## Overview

ResNet-20 is a deep residual network designed to classify 3×32×32 RGB images from the CIFAR-10 dataset using Fully Homomorphic Encryption (FHE). This implementation uses the FHEON framework built on top of OpenFHE.

## FHEON Framework

**FHEON** is a configurable framework designed to facilitate the implementation of privacy-preserving neural network inference using FHE. Built on top of OpenFHE, FHEON provides high-level abstractions for common deep learning components while optimizing for the unique constraints of homomorphic computation.

For more information, visit the [official website](https://fheon.pqcsecure.org/), explore the [source code](https://github.com/stamcenter/fheon), or read the [research paper](https://arxiv.org/abs/2510.03996).

## Model Architecture: ResNet-20 (diagonal primitives)

The ResNet-20 model of FHEON's `HEDiagonalSingleModels/ResNet20Diagonal`, built on the diagonal primitives (`FHEONANNDiagonalController`).

- **Architecture**: Initial convolution, three stages of three ResNet blocks (16, 32 and 64 channels; the first block of stages 2 and 3 has a stride-2 convolution and a 1x1 shortcut), global average pooling fused into the linear layer.
- **Dataset**: CIFAR-10
- **Layout**: channel-diagonal convolutions; every stage packs its activations in exactly channels x width^2 slots (16384, 8192 and 4096).
- **Bootstrapping**: before every ReLU (19 bootstrappings per image), on the slot count of the current stage, with a common correction factor.
- **ReLU**: Chebyshev approximation of degree 119 with **fixed scales**. The server never decrypts: every scale bounds the ReLU input of every image (the largest pre-activations of the plaintext network over the 60000 CIFAR-10 training and test images, +5%). These scales are larger than the per-image scales FHEON reads by decryption, which degree 27 cannot absorb. Plaintext simulation of the circuit on CIFAR-10 test images (exact ReLU 92.9%): degree 27 below 30%, degree 59 90.6%, degree 119 92.4%.
- **Keys**: one key file per stage (`rotation_keys_layer{1,2,3}.bin`): the bootstrapping keys of its slot count and the rotations of its layers. The context generates no other bootstrapping or EvalSum keys.
- **Server**: the batch runs stage by stage: the keys of a stage are loaded once and the stage runs for every image, so each key file is read once per batch.
- **Implementation**: `submissions/cifar10/src/resnet20_fheon.cpp` (shared by `client_key_generation`, `client_encode_encrypt_input` and `server_encrypted_compute`).

---

## Security Level

The security level is chosen when the submission is built: `cmake -DSECURITY_LEVEL=128` (the default) or `-DSECURITY_LEVEL=0` for the performance-efficient parameters. At 128 bits ResNet-20 satisfies the **128-bit security level** of the [Homomorphic Encryption Standard v1.1](https://homomorphicencryption.org/wp-content/uploads/2018/11/HomomorphicEncryptionStandardv1.1.pdf) (`HEStd_128_classic`; secret key SPARSE_ENCAPSULATED, a uniform ternary key whose sparse twin only serves the modulus raising of the bootstrapping): OpenFHE picks the ring dimension from log QP of the chain.

### CKKS Parameters, 128 bits
- **Ciphertexts depth**: 28 (10 levels after bootstrapping, level budget {4, 4})
- **Scale / first modulus**: 48 / 52 bits, 5 key-switching digits
- **log Q / log QP**: 1397 / 1697 (the standard allows log QP <= 1747 at this ring dimension)
- **Ring dimension**: 65536
- **Number of Slots**: 16384 (sparse packing in the 32768 slots of the ring)
- **Keys**: about 47 GB (three stage files); a stage needs about 20-25 GB of memory on the server

### Performance-efficient parameters (`-DSECURITY_LEVEL=0`)

Ring dimension 32768 (16384 slots), sparse ternary secret: 19.1 GB of keys. Not a standard security level.

---

### Execution Paths
The inference executables typically expect external weights and keys provided at runtime:
- **Weights**: `submissions/<model_name>/weights/`
- **Keys**: Generated and managed via model-specific `client_key_generation` utilities.