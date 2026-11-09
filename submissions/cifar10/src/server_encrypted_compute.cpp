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

#ifndef WEIGHTS_DIR
#define WEIGHTS_DIR "./submissions/cifar10/weights/resnet20/"
#endif

using namespace lbcrypto;

/** reads the trained weights from the CSV files of weightsDir (<layer>_weight.csv and <layer>_bias.csv of every layer) */
static ResNet20Weights load_resnet20_weights(const std::string &weightsDir) {
	ResNet20Weights weights;
	auto add_conv = [&](const std::string &layer, int outputChannels, int inputChannels) {
		weights.conv[layer] = load_weights(weightsDir + layer + "_weight.csv", outputChannels, inputChannels, kernelWidth, kernelWidth);
		weights.bias[layer] = load_bias(weightsDir + layer + "_bias.csv");
	};
	auto add_linear = [&](const std::string &layer, int outputs, int inputs) {
		weights.linear[layer] = load_fc_weights(weightsDir + layer + "_weight.csv", outputs, inputs);
		weights.bias[layer] = load_bias(weightsDir + layer + "_bias.csv");
	};
	add_conv("layer0_conv1", channelValues[0], imgDepth);
	for (int stage = 0; stage < numStages; stage++) {
		int channels = channelValues[stage];
		for (int block = 1; block <= 3; block++) {
			std::string name = "layer" + std::to_string(stage+1) + "_block" + std::to_string(block);
			/** the first block of stages 2 and 3 downsamples: its first convolution and its shortcut take the previous channels */
			int inputChannels = (stage > 0 && block == 1) ? channelValues[stage-1] : channels;
			add_conv(name + "_conv1", channels, inputChannels);
			add_conv(name + "_conv2", channels, channels);
			if (inputChannels != channels) {
				add_linear(name + "_shortcut", channels, inputChannels);
			}
		}
	}
	add_linear("layer_fc", channelValues[3], channelValues[numStages-1]);
	return weights;
}

int main(int argc, char *argv[]) {

	if (argc < 2 || !std::isdigit(argv[1][0])) {
		std::cout << "Usage: " << argv[0] << " instance-size \n";
		std::cout << "  Instance-size: 0-SINGLE, 1-SMALL, 2-MEDIUM, 3-LARGE\n";
		return 0;
	}
	resnet20_init_fheon();
	auto size = static_cast<InstanceSize>(std::stoi(argv[1]));
	InstanceParams prms(size);

	// The context, the public key and the relinearization key of the client (no secret key: the server never decrypts).
	// The bootstrapping and rotation keys of every stage are loaded by the model, stage by stage
	std::cout << "         [server] Loading keys" << std::endl;
	CryptoContext<DCRTPoly> context;
	FHEONHEController fheonHEController(context);
	FHEONHEController::HEConfig keysConfig;
	keysConfig.keysFolder = prms.pubkeydir().string();
	fheonHEController.load_context(keysConfig, true);
	fheonHEController.set_bootstrap_correction_factor(bootstrapCorrectionFactor);
	context = fheonHEController.getContext();
	ANNController fheonANNController(context);
	/** the trained weights, read once from the CSV files of the weights folder */
	ResNet20Weights weights = load_resnet20_weights(WEIGHTS_DIR);

	fs::create_directories(prms.ctxtdowndir());
	std::cout << "         [server] run encrypted CIFAR10 inference" << std::endl;
	Ctext ctxt;
	for (size_t i = 0; i < prms.getBatchSize(); ++i) {
		auto input_ctxt_path = prms.ctxtupdir() / ("cipher_input_" + std::to_string(i) + ".bin");
		if (!Serial::DeserializeFromFile(input_ctxt_path, ctxt, SerType::BINARY)) {
			throw std::runtime_error("Failed to get ciphertexts from " + input_ctxt_path.string());
		}
		auto start = std::chrono::high_resolution_clock::now();
		auto ctxtResult = resnet20_infer(fheonHEController, fheonANNController, weights, ctxt);
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		std::cout << "         [server] Execution time for ciphertext " << i
				<< " : " << duration.count() / 1000.0 << " seconds" << std::endl;
		auto result_ctxt_path = prms.ctxtdowndir() / ("cipher_result_" + std::to_string(i) + ".bin");
		Serial::SerializeToFile(result_ctxt_path, ctxtResult, SerType::BINARY);
	}
	return 0;
}
