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

CryptoContextT mlp_generate_crypto_context() {
  CCParamsT params;
  params.SetMultiplicativeDepth(8);
  params.SetKeySwitchTechnique(HYBRID);
  CryptoContextT cc = GenCryptoContext(params);
  cc->Enable(PKE);
  cc->Enable(KEYSWITCH);
  cc->Enable(LEVELEDSHE);
  return cc;
}
CryptoContextT generate_mult_rot_key(CryptoContextT cc, PrivateKeyT sk) {
  cc->EvalMultKeyGen(sk);
  cc->EvalRotateKeyGen(sk, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 32, 46, 64, 69, 92, 115, 128, 138, 161, 184, 207, 230, 253, 256, 276, 299, 322, 345, 368, 391, 414, 437, 460, 483, 506, 512});
  return cc;
}


int main(int argc, char* argv[]){

    if (argc < 2 || !std::isdigit(argv[1][0])) {
        std::cout << "Usage: " << argv[0] << " instance-size [--count_only]\n";
        std::cout << "  Instance-size: 0-SINGLE, 1-SMALL, 2-MEDIUM, 3-LARGE\n";
        return 0;
    }
    auto size = static_cast<InstanceSize>(std::stoi(argv[1]));
    InstanceParams prms(size);

    // Step 1: Setup CryptoContext
    auto cryptoContext = mlp_generate_crypto_context();

    // Step 2: Key Generation
    auto keyPair = cryptoContext->KeyGen();
    cryptoContext = generate_mult_rot_key(cryptoContext, keyPair.secretKey);

    // Step 3: Serialize cryptocontext and keys
    fs::create_directories(prms.pubkeydir());

    if (!Serial::SerializeToFile(prms.pubkeydir()/"cc.bin", cryptoContext,
                                SerType::BINARY) ||
        !Serial::SerializeToFile(prms.pubkeydir()/"pk.bin",
                                keyPair.publicKey, SerType::BINARY)) {
        throw std::runtime_error("Failed to write keys to " + prms.pubkeydir().string());
    }
    std::ofstream emult_file(prms.pubkeydir()/"mk.bin",
                           std::ios::out | std::ios::binary);
    std::ofstream erot_file(prms.pubkeydir()/"rk.bin",
                            std::ios::out | std::ios::binary);
    if (!emult_file.is_open() || !erot_file.is_open() ||
        !cryptoContext->SerializeEvalMultKey(emult_file, SerType::BINARY) ||
        !cryptoContext->SerializeEvalAutomorphismKey(erot_file, SerType::BINARY)) {
        throw std::runtime_error(
            "Failed to write eval keys to " + prms.pubkeydir().string());
    }

    fs::create_directories(prms.seckeydir());
    if (!Serial::SerializeToFile(prms.seckeydir()/"sk.bin",
                                keyPair.secretKey, SerType::BINARY)) {
        throw std::runtime_error("Failed to write keys to " + prms.seckeydir().string());
    }
    return 0;
}