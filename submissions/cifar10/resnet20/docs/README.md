# ResNet-20: Privacy-Preserving Encrypted Inference

This submission implements a ResNet-20 deep residual network using the FHEON framework for privacy-preserving machine learning inference on CIFAR-10 images.

## Overview

ResNet-20 is a deep residual network designed to classify 3×32×32 RGB images from the CIFAR-10 dataset using Fully Homomorphic Encryption (FHE). This implementation uses the FHEON framework built on top of OpenFHE.

## FHEON Framework

**FHEON** is a configurable framework designed to facilitate the implementation of privacy-preserving neural network inference using FHE. Built on top of OpenFHE, FHEON provides high-level abstractions for common deep learning components while optimizing for the unique constraints of homomorphic computation.

For more information, visit the [official website](https://fheon.pqcsecure.org/), explore the [source code](https://github.com/stamcenter/fheon), or read the [research paper](https://arxiv.org/abs/2510.03996).

## Model Architecture: ResNet-20

A deep residual network targeting CIFAR-10 built using FHEON.

- **Architecture**: Initial convolution, three stages of ResNet blocks with shortcuts, and Global Average Pooling.
- **Dataset**: CIFAR-10
- **Bootstrapping**: Strategic integration of CKKS bootstrapping to maintain circuit depth.
- **Implementation**: `submissions/resnet20/src/resnet20_fheon.cpp`

---

## Security Level

ResNet-20 is configured to satisfy the **128-bit security level** using the standardized parameters for CKKS as defined in the [Homomorphic Encryption Standard v1.1](https://homomorphicencryption.org/wp-content/uploads/2018/11/HomomorphicEncryptionStandardv1.1.pdf).

### CKKS Parameters
- **Ciphertexts depth**: 29
- **log PQ**: 708
- **Cyclotomic Order**: 131072
- **Ring dimension**: 65536
- **Number of Slots**: 32768


## Performance Optimization

The `client_key_generation` utilities provide equivalent ring dimensions and slot counts that target smaller security levels. These configurations are designed to significantly improve computation speed and reduce memory overhead, consistent with the performance benchmarks presented in the FHEON research paper.

---

### Execution Paths
The inference executables typically expect external weights and keys provided at runtime:
- **Weights**: `submissions/<model_name>/weights/`
- **Keys**: Generated and managed via model-specific `client_key_generation` utilities.