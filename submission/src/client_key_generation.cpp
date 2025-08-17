// Copyright (c) 2025 HomomorphicEncryption.org
// All rights reserved.
//
// This software is licensed under the terms of the Apache v2 License.
// See the LICENSE.md file for details.
//============================================================================

#include "utils.h"
#include "params.h"

using namespace lbcrypto;

int main(int argc, char* argv[]){

    if (argc < 2 || !std::isdigit(argv[1][0])) {
        std::cout << "Usage: " << argv[0] << " instance-size [--count_only]\n";
        std::cout << "  Instance-size: 0-TOY, 1-SMALL, 2-MEDIUM, 3-LARGE\n";
        return 0;
    }
    auto size = static_cast<InstanceSize>(std::stoi(argv[1]));
    InstanceParams prms(size);

    // Step 1: Setup CryptoContext
    uint32_t multDepth = 0;
    uint32_t scaleModSize = 40;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(LEVELEDSHE);

    // Step 2: Key Generation
    auto keys = cc->KeyGen();

    // Step 3: Serialize cryptocontext and keys
    fs::create_directories(prms.pubkeydir());

    if (!Serial::SerializeToFile(prms.pubkeydir()/"cc.bin", cc,
                                SerType::BINARY) ||
        !Serial::SerializeToFile(prms.pubkeydir()/"pk.bin",
                                keys.publicKey, SerType::BINARY)) {
        throw std::runtime_error("Failed to write keys to " + prms.pubkeydir().string());
    }

    fs::create_directories(prms.seckeydir());
    if (!Serial::SerializeToFile(prms.seckeydir()/"sk.bin",
                                keys.secretKey, SerType::BINARY)) {
        throw std::runtime_error("Failed to write keys to " + prms.seckeydir().string());
    }
    return 0;
}