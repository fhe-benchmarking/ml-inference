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

    CryptoContext<DCRTPoly> cc;

    if (!Serial::DeserializeFromFile(prms.pubkeydir()/"cc.bin", cc,
                                    SerType::BINARY)) {
        throw std::runtime_error("Failed to get CryptoContext from  " + prms.pubkeydir().string());
    }
    PublicKey<DCRTPoly> pk;
    if (!Serial::DeserializeFromFile(prms.pubkeydir()/"pk.bin", pk,
                                    SerType::BINARY)) {
        throw std::runtime_error("Failed to get public key from  " + prms.pubkeydir().string());
    }

    Ciphertext<DCRTPoly> ctxt_db, ctxt_q;
    if (!Serial::DeserializeFromFile(prms.ctxtupdir()/"cipher_db.bin", ctxt_db, SerType::BINARY) || 
        !Serial::DeserializeFromFile(prms.ctxtupdir()/"cipher_query.bin", ctxt_q, SerType::BINARY)) {
        throw std::runtime_error("Failed to get ciphertexts from " + prms.ctxtupdir().string());
    }

    auto ctxtSum = cc->EvalAdd(ctxt_db, ctxt_q);
    fs::create_directories(prms.ctxtdowndir());
    Serial::SerializeToFile(prms.ctxtdowndir()/"cipher_sum.bin", ctxtSum, SerType::BINARY);

    return 0;
}