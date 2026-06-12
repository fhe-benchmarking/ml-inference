// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#ifndef MLP_ENCRYPTION_UTILS_H_
#define MLP_ENCRYPTION_UTILS_H_

#include "openfhe.h"
#include "params.h"

using namespace lbcrypto;
using CiphertextT = Ciphertext<DCRTPoly>;
using ConstCiphertextT = ConstCiphertext<DCRTPoly>;
using CCParamsT = CCParams<CryptoContextCKKSRNS>;
using CryptoContextT = CryptoContext<DCRTPoly>;
using EvalKeyT = EvalKey<DCRTPoly>;
using PlaintextT = Plaintext;
using PrivateKeyT = PrivateKey<DCRTPoly>;
using PublicKeyT = PublicKey<DCRTPoly>;

#define MNIST_DIM 784
#define MNIST_LABEL_DIM 10

struct Sample {
  float image[MNIST_DIM];
};

struct Score {
  float score[MNIST_LABEL_DIM];
};

CryptoContextT mlp_generate_crypto_context();
CryptoContextT generate_mult_rot_key(CryptoContextT cc, PrivateKeyT sk);
CryptoContext<DCRTPoly> read_crypto_context(const InstanceParams& prms);
void read_eval_keys(const InstanceParams& prms, CryptoContextT cc);

std::vector<CiphertextT> mlp_encrypt(CryptoContextT cc, std::vector<float> v0, PublicKeyT pk);
std::vector<float> mlp_decrypt(CryptoContextT cc, std::vector<CiphertextT> v0, PrivateKeyT sk);
PublicKey<DCRTPoly> read_public_key(const InstanceParams& prms);
PrivateKey<DCRTPoly> read_secret_key(const InstanceParams& prms);

void load_dataset(std::vector<Sample> &dataset, const char *filename);
void write_dataset(const std::vector<Sample> &dataset, const char *filename);
void load_scores(std::vector<Score> &dataset, const char *filename);
void write_scores(const std::vector<Score> &dataset, const char *filename);
int argmax(float *A, int N);

#endif  // ifndef MLP_ENCRYPTION_UTILS_H_