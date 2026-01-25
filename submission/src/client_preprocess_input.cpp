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
#include "mlp_encryption_utils.h"
#include <fstream>
#include <sstream>
#include <vector>

using namespace lbcrypto;


int main(int argc, char* argv[]){

    if (argc < 2 || !std::isdigit(argv[1][0])) {
        std::cout << "Usage: " << argv[0] << " instance-size [--count_only]\n";
        std::cout << "  Instance-size: 0-SINGLE, 1-SMALL, 2-MEDIUM, 3-LARGE\n";
        return 0;
    }
    auto size = static_cast<InstanceSize>(std::stoi(argv[1]));
    InstanceParams prms(size);

    std::string test_pixels_path = prms.test_input_file().string();

    std::vector<Sample> dataset;
    load_dataset(dataset, prms.test_input_file().c_str());
    if (dataset.empty()) {
        throw std::runtime_error("No data found in " + prms.test_input_file().string());
    }

    if (dataset.size() != prms.getBatchSize()) {
        throw std::runtime_error("Dataset size does not match instance size");
    }

    // Normalize the inputs
    for (auto& sample : dataset) {
        for (int i = 0; i < MNIST_DIM; ++i) {
            sample.image[i] = (sample.image[i] - 0.1307) / 0.3081; // Normalization
        }
    }

    fs::create_directories(prms.iointermdir());
    write_dataset(dataset, prms.preprocessed_input_file().c_str());

    return 0;
}
