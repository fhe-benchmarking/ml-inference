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
#include "utils.h" 
#include "mlp_encryption_utils.h"
#include <sstream>
#include <string>

PublicKey<DCRTPoly> read_public_key(const InstanceParams& prms) {
    PublicKey<DCRTPoly> pk;
    if (!Serial::DeserializeFromFile(prms.pubkeydir()/"pk.bin", pk,
                                    SerType::BINARY)) {
        throw std::runtime_error("Failed to get public key from  " + prms.pubkeydir().string());
    }
    return pk;
}

PrivateKey<DCRTPoly> read_secret_key(const InstanceParams& prms) {
    PrivateKey<DCRTPoly> sk;
    if (!Serial::DeserializeFromFile(prms.seckeydir()/"sk.bin", sk,
                                    SerType::BINARY)) {
        throw std::runtime_error("Failed to get secret key from  " + prms.seckeydir().string());
    }
    return sk;
}

CryptoContextT read_crypto_context(const InstanceParams& prms) {
    CryptoContextT cc;
    if (!Serial::DeserializeFromFile(prms.pubkeydir()/"cc.bin", cc, SerType::BINARY)) {
        throw std::runtime_error("Failed to get CryptoContext from " + prms.pubkeydir().string());
    }
    return cc;
}

void read_eval_keys(const InstanceParams& prms, CryptoContextT cc) {
    std::ifstream emult_file(prms.pubkeydir()/"mk.bin", std::ios::in | std::ios::binary);
    if (!emult_file.is_open() ||
        !cc->DeserializeEvalMultKey(emult_file, SerType::BINARY)) {
      throw std::runtime_error(
        "Failed to get re-linearization key from " +prms.pubkeydir().string());
    }

    std::ifstream erot_file(prms.pubkeydir()/"rk.bin", std::ios::in | std::ios::binary);
    if (!erot_file.is_open() ||
        !cc->DeserializeEvalAutomorphismKey(erot_file, SerType::BINARY)) {
      throw std::runtime_error(
        "Failed to get rotation keys from " + prms.pubkeydir().string());
    }
}


std::vector<MutableCiphertextT> mlp_encrypt(CryptoContext<DCRTPoly> cc, std::vector<float> v0, PublicKey<DCRTPoly> pk) {
  [[maybe_unused]] size_t v1 = 0;
  std::vector<float> v2(1024, 0);
  [[maybe_unused]] int32_t v3 = 0;
  [[maybe_unused]] int32_t v4 = 1;
  [[maybe_unused]] int32_t v5 = 784;
  std::vector<float> v6 = v2;
  for (auto v7 = 0; v7 < 784; ++v7) {
    size_t v9 = static_cast<size_t>(v7);
    float v10 = v0[v9 + 784 * (0)];
    v6[v9 + 1024 * (0)] = v10;
  }
  std::vector<float> v12(1024);
  for (int64_t v12_i0 = 0; v12_i0 < 1; ++v12_i0) {
    for (int64_t v12_i1 = 0; v12_i1 < 1024; ++v12_i1) {
      v12[v12_i1 + 1024 * (v12_i0)] = v6[0 + v12_i1 * 1 + 1024 * (0 + v12_i0 * 1)];
    }
  }
  std::vector<double> v13(std::begin(v12), std::end(v12));
  auto pt_filled_n = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension() / 2;
  auto pt_filled = v13;
  pt_filled.clear();
  pt_filled.reserve(pt_filled_n);
  for (uint32_t i = 0; i < pt_filled_n; ++i) {
    pt_filled.push_back(v13[i % v13.size()]);
  }
  auto pt = cc->MakeCKKSPackedPlaintext(pt_filled);
  const auto& ct = cc->Encrypt(pk, pt);
  std::vector<MutableCiphertextT> ct_vec = {ct};
  return ct_vec;
}

std::vector<float> mlp_decrypt(CryptoContextT cc, std::vector<MutableCiphertextT> v0, PrivateKeyT sk) {
  [[maybe_unused]] size_t v1 = 0;
  [[maybe_unused]] int32_t v2 = 1024;
  [[maybe_unused]] int32_t v3 = 16;
  [[maybe_unused]] int32_t v4 = 6;
  [[maybe_unused]] int32_t v5 = 1;
  [[maybe_unused]] int32_t v6 = 0;
  std::vector<float> v7(10, 0);
  const auto& ct = v0[0];
  PlaintextT pt;
  cc->Decrypt(sk, ct, &pt);
  pt->SetLength(1024);
  const auto& v8_cast = pt->GetCKKSPackedValue();
  std::vector<float> v8(v8_cast.size());
  std::transform(std::begin(v8_cast), std::end(v8_cast), std::begin(v8), [](const std::complex<double>& c) { return c.real(); });
  std::vector<float> v9 = v7;
  for (auto v10 = 0; v10 < 1024; ++v10) {
    int32_t v12 = v10 + v4;
    int32_t v13 = v12 % v3;
    bool v14 = v13 >= v4;
    if (v14) {
      int32_t v16 = v10 % v3;
      size_t v17 = static_cast<size_t>(v10);
      float v18 = v8[v17 + 1024 * (0)];
      size_t v19 = static_cast<size_t>(v16);
      v9[v19 + 10 * (0)] = v18;
    } else {
    }
  }
  return v9;
}

void load_dataset(std::vector<Sample> &dataset, const char *filename) {
  std::ifstream file(filename);
  Sample sample;
  std::string line;
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    // Read MNIST_DIM values from file
    for (int i = 0; i < MNIST_DIM; i++) {
      iss >> sample.image[i];
    }
    dataset.push_back(sample);
  }
}

void write_dataset(const std::vector<Sample> &dataset, const char *filename) {
  std::ofstream file(filename);
  for (const auto &sample : dataset) {
    for (int i = 0; i < MNIST_DIM; i++) {
      file << sample.image[i];
      if (i < MNIST_DIM - 1) {
        file << " ";
      }
    }
    file << "\n";
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
