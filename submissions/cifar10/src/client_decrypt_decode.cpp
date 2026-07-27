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
#include "iomanip"
#include "limits"
#include "utils.h"

#include "encryption_utils.h"

using namespace lbcrypto;

int main(int argc, char *argv[]) {
  if (argc < 2 || !std::isdigit(argv[1][0])) {
    std::cout << "Usage: " << argv[0] << " instance-size [--count_only]\n";
    std::cout << "  Instance-size: 0-SINGLE, 1-SMALL, 2-MEDIUM, 3-LARGE\n";
    return 0;
  }
  auto size = static_cast<InstanceSize>(std::stoi(argv[1]));
  InstanceParams prms(size);

  CryptoContext<DCRTPoly> cc;
  if (!Serial::DeserializeFromFile(prms.pubkeydir() / "cc.bin", cc,
                                   SerType::BINARY)) {
    throw std::runtime_error("Failed to get CryptoContext from  " +
                             prms.pubkeydir().string());
  }
  PrivateKey<DCRTPoly> sk;
  if (!Serial::DeserializeFromFile(prms.seckeydir() / "sk.bin", sk,
                                   SerType::BINARY)) {
    throw std::runtime_error("Failed to get secret key from  " +
                             prms.seckeydir().string());
  }
  Ciphertext<DCRTPoly> ctxt;
  std::vector<float> output;
  auto result_path = prms.encrypted_model_predictions_file();
  std::ofstream out(result_path);
  for (size_t i = 0; i < prms.getBatchSize(); ++i) {
    auto ctxt_path =
        prms.ctxtdowndir() / ("cipher_result_" + std::to_string(i) + ".bin");
    if (!Serial::DeserializeFromFile(ctxt_path, ctxt, SerType::BINARY)) {
      throw std::runtime_error("Failed to get ciphertext from " +
                               ctxt_path.string());
    }
    output = input_decrypt(cc, ctxt, sk);
    auto max_id = argmax(output.data(), 1024);
    out << max_id << '\n';
  }

  return 0;
}
