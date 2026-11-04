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
#include "params.h"
#include "resnet20_fheon.h"
#include "utils.h"
#include <chrono>

using namespace lbcrypto;

int main(int argc, char *argv[]) {

	if (argc < 2 || !std::isdigit(argv[1][0])) {
		std::cout << "Usage: " << argv[0] << " instance-size \n";
		std::cout << "  Instance-size: 0-SINGLE, 1-SMALL, 2-MEDIUM, 3-LARGE\n";
		return 0;
	}
	auto size = static_cast<InstanceSize>(std::stoi(argv[1]));
	InstanceParams prms(size);

	// The context, the public key and the relinearization key of the client (no secret key: the server never decrypts)
	std::cout << "         [server] Loading keys" << std::endl;
	CryptoContext<DCRTPoly> context;
	FHEONHEController fheonHEController(context);
	fheonHEController.load_context(resnet20_he_config(prms.pubkeydir().string()), true);
	fheonHEController.set_bootstrap_correction_factor(bootstrapCorrectionFactor);
	context = fheonHEController.getContext();
	ANNController fheonANNController(context);

	// Every input is read before the first key file: loading the keys of a stage releases the crypto contexts of
	// OpenFHE, and a ciphertext deserialized after that would not belong to the context of the server
	std::vector<std::vector<Ctext>> states(prms.getBatchSize(), std::vector<Ctext>(1));
	for (size_t i = 0; i < prms.getBatchSize(); ++i) {
		auto input_ctxt_path = prms.ctxtupdir() / ("cipher_input_" + std::to_string(i) + ".bin");
		if (!Serial::DeserializeFromFile(input_ctxt_path, states[i][0], SerType::BINARY)) {
			throw std::runtime_error("Failed to get ciphertexts from " + input_ctxt_path.string());
		}
	}

	// Stage by stage: the keys of a stage are loaded once and the stage runs for every image
	std::cout << "         [server] run encrypted CIFAR10 inference" << std::endl;
	auto start = std::chrono::high_resolution_clock::now();
	for (int stage = 0; stage < numStages; stage++) {
		auto stage_start = std::chrono::high_resolution_clock::now();
		resnet20_use_stage_keys(fheonHEController, stage);
		auto keys_end = std::chrono::high_resolution_clock::now();
		for (size_t i = 0; i < prms.getBatchSize(); ++i) {
			resnet20_stage(fheonHEController, fheonANNController, stage, states[i]);
		}
		auto stage_end = std::chrono::high_resolution_clock::now();
		std::cout << "         [server] Stage " << stage+1 << ": keys loaded in "
				<< std::chrono::duration_cast<std::chrono::milliseconds>(keys_end - stage_start).count() / 1000.0 << " s, "
				<< prms.getBatchSize() << " ciphertext(s) in "
				<< std::chrono::duration_cast<std::chrono::milliseconds>(stage_end - keys_end).count() / 1000.0 << " s" << std::endl;
	}
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << "         [server] Execution time for " << prms.getBatchSize() << " ciphertext(s): "
			<< std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " seconds" << std::endl;

	fs::create_directories(prms.ctxtdowndir());
	for (size_t i = 0; i < prms.getBatchSize(); ++i) {
		auto result_ctxt_path = prms.ctxtdowndir() / ("cipher_result_" + std::to_string(i) + ".bin");
		Serial::SerializeToFile(result_ctxt_path, states[i][0], SerType::BINARY);
	}
	return 0;
}
