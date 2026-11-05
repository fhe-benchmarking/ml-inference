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


#ifndef RESNET20_FHEON_H_
#define RESNET20_FHEON_H_

#include "FHEONANNDiagonalController.h"
#include "FHEONHEController.h"
#include "openfhe.h"

#include <map>

using namespace std;
using namespace lbcrypto;

/** ResNet-20 with the diagonal primitives of FHEON (FHEON's HEDiagonalSingleModels/ResNet20Diagonal): the architecture,
 *  the ReLU scales, the stages and their key files, the level of the input and the encrypted inference. The context and
 *  keys are generated in client_key_generation, the weights are read from the CSV files of the weights folder by
 *  server_encrypted_compute. */
using ANNController = ANNDiagonalController;

/** false: 128-bit security (HEStd_128_classic), the benchmark setting.
 *  true: performance-efficient parameters (HEStd_NotSet, sparse secret), not a standard security level.
 *  Rebuild and rerun key generation after changing it: the client and the server must agree on it */
const bool performanceMode = false;
/** security level of the CKKS context in bits (0 = performance efficient), see FHEONHEController::HEConfig */
const int resnet20SecurityLevel = performanceMode ? 0 : 128;

/** FHEON log verbosity: 0 silent, 1 layer names, 2 timings and memory of every operation */
const int fheonVerboseLevel = 0;

/** sets the FHEON verbosity and starts the clock of the "Total" timings: call once at the start of each stage */
inline void resnet20_init_fheon() {
    utils::verbose_level = fheonVerboseLevel;
    utils::program_start_time = std::chrono::steady_clock::now();
}


const int reluDegree = 119;
/** levels of a ReLU: the Chebyshev evaluation (4 at degree 13, 5 at 27, 6 at 59, 7 at 119) and the scaling of its input */
const int reluLevels = ((reluDegree > 59) ? 7 : (reluDegree > 27) ? 6 : (reluDegree > 13) ? 5 : 4) + 1;
const uint32_t bootstrapCorrectionFactor = 9;
const int bootstrapIterations = 1;

const int imgCols = 32;
const int imgDepth = 3;
const int kernelWidth = 3;
const int numStages = 3;
const vector<int> channelValues = {16, 32, 64, 10};

/** slot count of every stage: channels * width^2 (16384, 8192, 4096) */
vector<int> resnet20_stage_slots();
/** key file of a stage: its bootstrapping keys and the rotations of its layers */
string resnet20_stage_key_file(int stage);
/** layout of the stride-2 block that ends a stage (the client generates its rotations, the server runs it) */
FHEONLayout resnet20_stride2_layout(int inputWidth, int inputChannels);
/** level the input is encrypted at: the circuit before the first bootstrapping (3 convolutions and 2 ReLUs) ends with 3 towers */
int resnet20_input_level(const CryptoContext<DCRTPoly> &context);

/** the plaintext weights of ResNet-20, as trained (the CSV files of the weights folder), by layer name
 *  ("layer0_conv1", "layer2_block1_conv1", "layer2_block1_shortcut", "layer_fc", ...) */
struct ResNet20Weights {
    map<string, vector<vector<vector<vector<double>>>>> conv;
    map<string, vector<vector<double>>> linear;
    map<string, vector<double>> bias;
};

/** encrypted inference of one image, without decryption: the 10 logits in the first slots. Every stage has its own key
 *  file (its bootstrapping keys and the rotations of its layers): the model loads the keys of each stage before running it */
Ctext resnet20_infer(FHEONHEController &fheonHEController, ANNController &fheonANNController, ResNet20Weights &weights,
                     const Ctext &encryptedImage);

#endif // ifndef RESNET20_FHEON_H_
