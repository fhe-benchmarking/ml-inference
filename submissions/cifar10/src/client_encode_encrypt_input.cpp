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
#include "utils.h"
#include "resnet20_fheon.h"

using namespace lbcrypto;

int main(int argc, char *argv[]) {

	if (argc < 2 || !std::isdigit(argv[1][0])) {
		std::cout << "Usage: " << argv[0] << " instance-size [--count_only]\n";
		std::cout << "  Instance-size: 0-SINGLE, 1-SMALL, 2-MEDIUM, 3-LARGE\n";
		return 0;
	}
	auto size = static_cast<InstanceSize>(std::stoi(argv[1]));
	InstanceParams prms(size);

	CryptoContext<DCRTPoly> cc = read_crypto_context(prms);

	// Step 2: Read public key
	PublicKey<DCRTPoly> pk = read_public_key(prms);

	std::vector<Sample> dataset;
	load_dataset(dataset, prms.test_input_file().c_str(), CIFAR_DIM, prms.getBatchSize());
	if (dataset.empty()) {
		throw std::runtime_error("No data found in " + prms.test_input_file().string());
	}
	// Step 2: Encrypt inputs
	if (dataset.size() != prms.getBatchSize()) {
		throw std::runtime_error("Dataset size does not match instance size");
	}

	std::shared_ptr<const CiphertextImpl<DCRTPoly>> ctxt;
	int input_level = resnet20_input_level(cc);
	fs::create_directories(prms.ctxtupdir());
	for (size_t i = 0; i < dataset.size(); ++i) {
	// for(size_t i = 0; i < 1; ++i) {
		auto *input = dataset[i].image;
		// std::vector<float> input_vector(input, input + NORMALIZED_DIM);
		std::vector<float> input_vector(input, input + CIFAR_DIM);
		// Apply CIFAR10 per-channel normalization: (x - mean) / std
		// CIFAR10: mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
		float cifar10_mean[] = {0.4914f, 0.4822f, 0.4465f};
		float cifar10_std[] = {0.2023f, 0.1994f, 0.2010f};
		int pixels_per_channel = CIFAR_DIM / 3;
		// std::cout << "Plaintext sample: " << input_vector << "..." << std::endl;

		for (int c = 0; c < 3; ++c) {
			for (int idx = c * pixels_per_channel; idx < (c + 1) * pixels_per_channel; ++idx) {
				input_vector[idx] = ((input_vector[idx]) - cifar10_mean[c]) / cifar10_std[c];
			}
		}

		// ResNet20Diagonal input: the 3072 values (channel-major) in the first slots of a 2^14-slot ciphertext, zeros in
		// the other slots, encrypted at the level of the circuit before its first bootstrapping
		std::vector<double> input_values(input_vector.begin(), input_vector.end());
		Plaintext plaintext = cc->MakeCKKSPackedPlaintext(input_values, 1, input_level);
		ctxt = cc->Encrypt(pk, plaintext);
		auto ctxt_path = prms.ctxtupdir() / ("cipher_input_" + std::to_string(i) + ".bin");
		Serial::SerializeToFile(ctxt_path, ctxt, SerType::BINARY);
	}

	return 0;
}
