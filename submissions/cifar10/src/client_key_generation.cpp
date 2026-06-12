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
#include "resnet20_fheon.h"

#include "utils.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
namespace fs = std::filesystem;

// Parameters are now defined in resnet20_fheon.h via config struct

CryptoContextT generate_crypto_context() {

    lbcrypto::SecretKeyDist secretKeyDist = lbcrypto::SPARSE_TERNARY;
    int circuitDepth = config.modelDepth +
                      lbcrypto::FHECKKSRNS::GetBootstrapDepth(config.levelBudget,
                                                              secretKeyDist);

    CCParamsT parameters;
    parameters.SetMultiplicativeDepth(circuitDepth);
    parameters.SetSecurityLevel(HEStd_128_classic);
    // parameters.SetSecurityLevel(HEStd_NotSet);
    // parameters.SetRingDim(config.ringDim);
    // parameters.SetBatchSize(config.numSlots);
    parameters.SetScalingModSize(config.dcrtBits);
    parameters.SetFirstModSize(config.firstMod);
    parameters.SetNumLargeDigits(config.digitSize);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetSecretKeyDist(secretKeyDist);

    CryptoContextT context = GenCryptoContext(parameters);
    context->Enable(PKE);
    context->Enable(KEYSWITCH);
    context->Enable(LEVELEDSHE);
    context->Enable(ADVANCEDSHE);
    context->Enable(FHE);

    cout << "Context built, generating keys..." << endl;
    cout << endl
        << "dcrtBits: " << config.dcrtBits << " -- firstMod: " << config.firstMod
        << endl
        << "Ciphertexts depth: " << circuitDepth
        << ", available multiplications: " << config.modelDepth - 2 << endl;

    return context;
}

CryptoContextT generate_mult_rot_key(CryptoContextT context, PrivateKeyT secretKey) {

    context->EvalMultKeyGen(secretKey);
    vector<int> rotPositions = {
        // -3276, // required for ringRim 1 << 16
        -15360, -14336, -13312, -12288, -11520, -11264, -10240, -9216, -8192,
        -7936,  -7680,  -7424,  -7168,  -6912,  -6656,  -6400,  -6144, -5952,
        -5888,  -5632,  -5376,  -5120,  -4864,  -4608,  -4352,  -4096, -4032,
        -3968,  -3904,  -3840,  -3776,  -3712,  -3648,  -3584,  -3520, -3456,
        -3392,  -3328,  -3264,  -3200,  -3136,  -3072,  -3008,  -2944, -2880,
        -2816,  -2752,  -2688,  -2624,  -2560,  -2496,  -2432,  -2368, -2304,
        -2240,  -2176,  -2112,  -2048,  -1984,  -1920,  -1856,  -1792, -1728,
        -1664,  -1600,  -1536,  -1472,  -1408,  -1344,  -1280,  -1216, -1152,
        -1088,  -1024,  -960,   -896,   -832,   -768,   -704,   -640,  -576,
        -512,   -448,   -384,   -320,   -256,   -192,   -128,   -64,   -48,
        -32,    -16,    -8,     -1,     1,      2,      3,      4,     5,
        6,      7,      8,      9,      10,     11,     12,     13,    14,
        15,     16,     24,     32,     48,     64,     256,    1024};
    context->EvalRotateKeyGen(secretKey, rotPositions);
    return context;
}

void generate_rotation_keys(FHEONHEController &fheonHEController,
                            CryptoContextT context, PrivateKeyT secretKey,
                            vector<int> channels, int dataset_size) {

    FHEONANNController fheonANNController(context);

    int img_depth = 3;
    int dataWidth = 32;
    int avgpoolSize = 8;
    int rotPositions = 16;
    auto size = static_cast<InstanceSize>(dataset_size);
    InstanceParams prms(size);

    /************************************** generate rotation keys for conv_layer
     * 1 **********************************8*/
    auto conv1_keys = fheonANNController.generate_optimized_convolution_rotation_positions(dataWidth, img_depth, channels[0]);
    auto conv2_keys = fheonANNController.generate_optimized_convolution_rotation_positions(dataWidth, channels[0], channels[0]);
    auto conv3_keys = fheonANNController.generate_optimized_convolution_rotation_positions(dataWidth, channels[0], channels[1], 2, "single_channel");
    dataWidth = dataWidth / 2;

    auto conv4_keys = fheonANNController.generate_optimized_convolution_rotation_positions(dataWidth, channels[1], channels[1]);
    auto conv5_keys = fheonANNController.generate_optimized_convolution_rotation_positions(dataWidth, channels[1], channels[2], 2, "single_channel");
    dataWidth = dataWidth / 2;

    auto conv6_keys = fheonANNController.generate_optimized_convolution_rotation_positions(dataWidth, channels[2], channels[2]);
    auto avgpool1_key = fheonANNController.generate_avgpool_optimized_rotation_positions(dataWidth, channels[2], avgpoolSize, avgpoolSize, true,
                                                        "single_channel", rotPositions);
    auto fc_keys = fheonANNController.generate_linear_rotation_positions(channels[3], rotPositions);

    /*************************************************************************************************/
    vector<vector<int>> rkeys_layer1, rkeys_layer2, rkeys_layer3, rkeys_layer4;
    rkeys_layer1.push_back(conv1_keys);
    rkeys_layer1.push_back(conv2_keys);

    rkeys_layer2.push_back(conv3_keys);
    rkeys_layer2.push_back(conv4_keys);

    rkeys_layer3.push_back(conv5_keys);
    rkeys_layer3.push_back(conv6_keys);

    rkeys_layer4.push_back(avgpool1_key);
    rkeys_layer4.push_back(fc_keys);
    //   rkeys_layer4.push_back({32, 64});

    /********************************************************************************************************************************************/
    /*** join all keys and generate unique values only */
    vector<int> serkeys_layer1 = serialize_rotation_keys(rkeys_layer1);
    vector<int> serkeys_layer2 = serialize_rotation_keys(rkeys_layer2);
    vector<int> serkeys_layer3 = serialize_rotation_keys(rkeys_layer3);
    vector<int> serkeys_layer4 = serialize_rotation_keys(rkeys_layer4);

    cout << "Layer 1 keys (" << serkeys_layer1.size() << ") " << serkeys_layer1 << endl;
    cout << "Layer 2 keys (" << serkeys_layer2.size() << ") " << serkeys_layer2 << endl;
    cout << "Layer 3 keys (" << serkeys_layer3.size() << ") " << serkeys_layer3 << endl;
    cout << "Layer 4 keys (" << serkeys_layer4.size() << ") " << serkeys_layer4 << endl;

    ofstream layer1_file(prms.pubkeydir() / "layer1_rk.bin", ios::out | ios::binary);
    ofstream layer2_file(prms.pubkeydir() / "layer2_rk.bin", ios::out | ios::binary);
    ofstream layer3_file(prms.pubkeydir() / "layer3_rk.bin", ios::out | ios::binary);
    ofstream layer4_file(prms.pubkeydir() / "layer4_rk.bin", ios::out | ios::binary);

    // Each layer's file must include bootstrap automorphism keys so the server
    // can call EvalBootstrap without sk. Pattern per layer:
    fheonHEController.harness_generate_bootstrapping_and_rotation_keys(context, secretKey, serkeys_layer1, layer1_file);
    fheonHEController.harness_clear_bootstrapping_and_rotation_keys(context);

    fheonHEController.harness_generate_bootstrapping_and_rotation_keys(context, secretKey, serkeys_layer2, layer2_file);
    fheonHEController.harness_clear_bootstrapping_and_rotation_keys(context);

    fheonHEController.harness_generate_bootstrapping_and_rotation_keys(context, secretKey, serkeys_layer3, layer3_file);
    fheonHEController.harness_clear_bootstrapping_and_rotation_keys(context);

    fheonHEController.harness_generate_bootstrapping_and_rotation_keys(context, secretKey, serkeys_layer4, layer4_file, true);
    fheonHEController.harness_clear_bootstrapping_and_rotation_keys(context);
    cout << "All keys generated" << endl;
  /********************************************************************************************************************************************/
}

int main(int argc, char *argv[]) {

    if (argc < 2 || !isdigit(argv[1][0])) {
        cout << "Usage: " << argv[0] << " instance-size [--count_only]\n";
        cout << "  Instance-size: 0-SINGLE, 1-SMALL, 2-MEDIUM, 3-LARGE\n";
        return 0;
    }

    int dataset_size = stoi(argv[1]);
    auto size = static_cast<InstanceSize>(dataset_size);
    InstanceParams prms(size);

    // Step 1: Setup CryptoContext
    auto cryptoContext = generate_crypto_context();

    // Step 2: Key Generation
    // cout << "Starting KeyGen..." << endl;
    auto keyPair = cryptoContext->KeyGen();
    cryptoContext->EvalMultKeyGen(keyPair.secretKey);
    cryptoContext->EvalSumKeyGen(keyPair.secretKey);

    FHEONHEController fheonHEController(cryptoContext);
    double logPQ = fheonHEController.getlogPQ(keyPair.publicKey->GetPublicElements()[0]);
    cout << "log PQ = " << logPQ << std::endl;
    cout << "Cyclotomic Order: " << cryptoContext->GetCyclotomicOrder() << endl;
    cout << "Ring dimension: " << (cryptoContext->GetCyclotomicOrder() / 2) << endl;
    cout << "Num Slots     : " << (cryptoContext->GetCyclotomicOrder() / 4) << endl;
    cout << endl;

    // Step 3: Serialize cryptocontext and keys
    fs::create_directories(prms.pubkeydir());
    // cout << "Serializing CC and PK..." << endl;

    if (!Serial::SerializeToFile(prms.pubkeydir() / "cc.bin", cryptoContext,
                                SerType::BINARY) ||
        !Serial::SerializeToFile(prms.pubkeydir() / "pk.bin", keyPair.publicKey,
                                SerType::BINARY)) {
        throw runtime_error("Failed to write keys to " + prms.pubkeydir().string());
    }
    // cout << "CC and PK serialized. Serializing Eval Keys..." << endl;
    ofstream emult_file(prms.pubkeydir() / "mk.bin", ios::out | ios::binary);
    // ofstream erot_file(prms.pubkeydir() / "rk.bin", ios::out | ios::binary);
    if (!emult_file.is_open() ||
        !cryptoContext->SerializeEvalMultKey(emult_file, SerType::BINARY)) {
        throw runtime_error("Failed to write mult keys to " +
                            prms.pubkeydir().string());
    }

    /*** work on rotation keys */
    vector<int> channels = {16, 32, 64, 10};
    generate_rotation_keys(fheonHEController, cryptoContext, keyPair.secretKey, channels, dataset_size);
    // cout << "Eval Keys serialized. Serializing Secret Key..." << endl;

    fs::create_directories(prms.seckeydir());
    if (!Serial::SerializeToFile(prms.seckeydir() / "sk.bin", keyPair.secretKey,
                                SerType::BINARY)) {
        throw runtime_error("Failed to write keys to " + prms.seckeydir().string());
    }
    // cout << "Secret Key serialized." << endl;

    return 0;
}
