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

    std::string q_path = prms.intermdir()/"plain_q.bin";
    std::ifstream in(q_path, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("Cannot open " + q_path);
    double x; 
    in.read((char*)&x, sizeof(double));
    if (!in)
        throw std::runtime_error("Failed to read double from " + q_path);

    std::vector<double> q = {x};
    auto ptxt = cc->MakeCKKSPackedPlaintext({q});
    auto ctxt = cc->Encrypt(pk, ptxt);

    fs::create_directories(prms.ctxtupdir());
    Serial::SerializeToFile(prms.ctxtupdir()/"cipher_query.bin", ctxt, SerType::BINARY);

    return 0;
}