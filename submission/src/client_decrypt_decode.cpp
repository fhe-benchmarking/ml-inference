// Copyright (c) 2025 HomomorphicEncryption.org
// All rights reserved.
//
// This software is licensed under the terms of the Apache v2 License.
// See the LICENSE.md file for details.
//============================================================================

#include "utils.h"
#include "params.h"
#include "iomanip"
#include "limits"

using namespace lbcrypto;

int main(int argc, char* argv[]) {
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
    PrivateKey<DCRTPoly> sk;
    if (!Serial::DeserializeFromFile(prms.seckeydir()/"sk.bin", sk,
                                    SerType::BINARY)) {
        throw std::runtime_error("Failed to get secret key from  " + prms.seckeydir().string());
    }
    Ciphertext<DCRTPoly> ctxt;     
    if (!Serial::DeserializeFromFile(prms.ctxtdowndir()/"cipher_sum.bin", ctxt, SerType::BINARY)) {
      throw std::runtime_error("Failed to get ciphertext from " + prms.ctxtdowndir().string());
    }

    Plaintext ptxt; 
    cc->Decrypt(sk, ctxt, &ptxt);
    ptxt->SetLength(1);
    double res = ptxt->GetCKKSPackedValue()[0].real();

    // No post-processing, so write in the result file.
    std::ofstream out(prms.iodir() / "result.txt");
    out << std::setprecision(std::numeric_limits<double>::max_digits10) << std::defaultfloat << res << '\n';

    return 0;
}