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
#include "params.h"
#include "mlp_openfhe.h"
#include "mlp_encryption_utils.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <chrono>

using namespace lbcrypto;

using MutableCiphertextT = Ciphertext<DCRTPoly>;


int main(int argc, char* argv[]){

    if (argc < 2 || !std::isdigit(argv[1][0])) {
        std::cout << "Usage: " << argv[0] << " instance-size \n";
        std::cout << "  Instance-size: 0-SINGLE, 1-SMALL, 2-MEDIUM, 3-LARGE\n";
        return 0;
    }
    auto size = static_cast<InstanceSize>(std::stoi(argv[1]));
    InstanceParams prms(size);

    CryptoContext<DCRTPoly> cc = read_crypto_context(prms);
    read_eval_keys(prms, cc);
    PublicKey<DCRTPoly> pk = read_public_key(prms);
    
    std::cout << "         [server] Loading keys" << std::endl;

    const std::string model_path = "submission/data/traced_model.pt";
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(model_path);
        module.eval();
        std::cout << "         [server] PyTorch model weights successfully!" << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "         [server] Error loading PyTorch model: " << e.what() << std::endl;
        return 1;
    }

    // Extract model weights
    std::vector<float> fc1_weight, fc1_bias, fc2_weight, fc2_bias;
    try {
        // Get named parameters from the model
        auto named_params = module.named_parameters();
        
        for (const auto& param : named_params) {
            const std::string& name = param.name;
            const torch::Tensor& tensor = param.value;
            // Extract weights based on parameter name
            if (name.find("fc1.weight") != std::string::npos || name.find("0.weight") != std::string::npos) {
                auto flat_tensor = tensor.flatten().contiguous();
                fc1_weight.assign(flat_tensor.data_ptr<float>(), 
                                 flat_tensor.data_ptr<float>() + flat_tensor.numel());
            } else if (name.find("fc1.bias") != std::string::npos || name.find("0.bias") != std::string::npos) {
                auto flat_tensor = tensor.flatten().contiguous();
                fc1_bias.assign(flat_tensor.data_ptr<float>(), 
                               flat_tensor.data_ptr<float>() + flat_tensor.numel());
            } else if (name.find("fc2.weight") != std::string::npos || name.find("2.weight") != std::string::npos) {
                auto flat_tensor = tensor.flatten().contiguous();
                fc2_weight.assign(flat_tensor.data_ptr<float>(), 
                                 flat_tensor.data_ptr<float>() + flat_tensor.numel());
            } else if (name.find("fc2.bias") != std::string::npos || name.find("2.bias") != std::string::npos) {
                auto flat_tensor = tensor.flatten().contiguous();
                fc2_bias.assign(flat_tensor.data_ptr<float>(), 
                               flat_tensor.data_ptr<float>() + flat_tensor.numel());
            }
        }
        
        // Verify we have all required weights
        if (fc1_weight.empty() || fc1_bias.empty() || fc2_weight.empty() || fc2_bias.empty()) {
            std::cerr << "         [server] Error: Could not extract all required weights from model" << std::endl;
            return 1;
        }
        
    } catch (const c10::Error& e) {
        std::cerr << "         [server] Error extracting weights: " << e.what() << std::endl;
        return 1;
    }

    std::vector<MutableCiphertextT> ctxt;
    fs::create_directories(prms.ctxtdowndir());
    std::cout << "         [server] run encrypted MNIST inference" << std::endl;
    for (size_t i = 0; i < prms.getBatchSize(); ++i) {
        auto input_ctxt_path = prms.ctxtupdir()/("cipher_input_" + std::to_string(i) + ".bin");
        if (!Serial::DeserializeFromFile(input_ctxt_path, ctxt, SerType::BINARY)) {
            throw std::runtime_error("Failed to get ciphertexts from " + input_ctxt_path.string());
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        auto ctxtResults = mnist(cc, fc1_weight, fc1_bias, fc2_weight, fc2_bias, ctxt);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
        std::cout << "         [server] Execution time for ciphertext " << i << " : " 
                << duration.count() << " seconds" << std::endl;
        
        auto result_ctxt_path = prms.ctxtdowndir()/("cipher_result_" + std::to_string(i) + ".bin");
        Serial::SerializeToFile(result_ctxt_path, ctxtResults, SerType::BINARY);
    }

    return 0;
}
