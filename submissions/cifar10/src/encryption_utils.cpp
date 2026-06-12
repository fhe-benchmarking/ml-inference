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
#include "encryption_utils.h"
#include "utils.h"
#include <sstream>
#include <string>

PublicKey<DCRTPoly> read_public_key(const InstanceParams &prms) {
  PublicKey<DCRTPoly> pk;
  if (!Serial::DeserializeFromFile(prms.pubkeydir() / "pk.bin", pk,
                                   SerType::BINARY)) {
    throw std::runtime_error("Failed to get public key from  " +
                             prms.pubkeydir().string());
  }
  return pk;
}

PrivateKey<DCRTPoly> read_secret_key(const InstanceParams &prms) {
  PrivateKey<DCRTPoly> sk;
  if (!Serial::DeserializeFromFile(prms.seckeydir() / "sk.bin", sk,
                                   SerType::BINARY)) {
    throw std::runtime_error("Failed to get secret key from  " +
                             prms.seckeydir().string());
  }
  return sk;
}

CryptoContextT read_crypto_context(const InstanceParams &prms) {
  CryptoContextT cc;
  if (!Serial::DeserializeFromFile(prms.pubkeydir() / "cc.bin", cc,
                                   SerType::BINARY)) {
    throw std::runtime_error("Failed to get CryptoContext from " +
                             prms.pubkeydir().string());
  }
  return cc;
}

void read_eval_keys(const InstanceParams &prms, CryptoContextT cc) {
  std::ifstream emult_file(prms.pubkeydir() / "mk.bin",
                           std::ios::in | std::ios::binary);
  if (!emult_file.is_open() ||
      !cc->DeserializeEvalMultKey(emult_file, SerType::BINARY)) {
    throw std::runtime_error("Failed to get re-linearization key from " +
                             prms.pubkeydir().string());
  }

  std::ifstream erot_file(prms.pubkeydir() / "rk.bin",
                          std::ios::in | std::ios::binary);
  if (!erot_file.is_open() ||
      !cc->DeserializeEvalAutomorphismKey(erot_file, SerType::BINARY)) {
    throw std::runtime_error("Failed to get rotation keys from " +
                             prms.pubkeydir().string());
  }
}

ConstCiphertext<DCRTPoly> input_encrypt(CryptoContext<DCRTPoly> cc,
                                        std::vector<float> input,
                                        PublicKey<DCRTPoly> pk) {
  std::vector<double> v11340(std::begin(input), std::end(input));
  uint32_t v11340_filled_n =
      cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto v11340_filled = v11340;
  v11340_filled.clear();
  v11340_filled.reserve(v11340_filled_n);
  for (uint32_t i = 0; i < v11340_filled_n; ++i) {
    v11340_filled.push_back(v11340[i % v11340.size()]);
  }
  const auto &v11341 = cc->MakeCKKSPackedPlaintext(v11340_filled);
  const auto &v11342 = cc->Encrypt(pk, v11341);

  // auto plaintext = cc->MakeCKKSPackedPlaintext(v11340, 1, 1);
  // plaintext->SetLength(input.size());
  return v11342;
}

std::vector<float> input_decrypt(CryptoContextT v11343, CiphertextT v11344,
                                 PrivateKeyT v11345) {
  PlaintextT v11346;
  v11343->Decrypt(v11345, v11344, &v11346);
  // v11346->SetLength(1024);
  const auto &v11347_cast = v11346->GetCKKSPackedValue();
  std::vector<float> v11347(v11347_cast.size());
  std::transform(std::begin(v11347_cast), std::end(v11347_cast),
                 std::begin(v11347),
                 [](const std::complex<double> &c) { return c.real(); });
  return v11347;
}

void load_dataset(std::vector<Sample> &dataset, const char *filename, int dim,
                  int max_samples) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error(std::string("Failed to open file: ") + filename);
  }
  Sample sample;
  std::string line;
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    // Read dim values from file
    for (int i = 0; i < dim; i++) {
      if (!(iss >> sample.image[i])) {
        throw std::runtime_error("Error: Failed to read value at index " + std::to_string(i) + 
                                 " from line. Expected " + std::to_string(dim) + 
                                 " values, but got fewer. Please ensure you generated the "
                                 "correct dataset (e.g., using --dataset cifar10 instead of mnist).");
      }
    }
    // Pad remaining values with 0.0 if NORMALIZED_DIM > dim
    for (int i = dim; i < NORMALIZED_DIM; i++) {
      sample.image[i] = 0.0f;
    }

    dataset.push_back(sample);
    if (max_samples > 0 && static_cast<int>(dataset.size()) >= max_samples) {
      break;
    }
  }
}

int argmax(float *A, int N) {
  int max_idx = 0;
  for (int i = 1; i < N; i++) {
    if (A[i] > A[max_idx]) {
      max_idx = i;
    }
  }
  return max_idx;
}
