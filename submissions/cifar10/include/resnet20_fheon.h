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

using namespace std;
using namespace lbcrypto;

/** ResNet-20 with the diagonal primitives of FHEON (FHEON's HEDiagonalSingleModels/ResNet20Diagonal).
 *  The client (key generation, encryption) and the server (inference) share the circuit below: the
 *  parameters of the context, the key file and slot count of every stage, and the level of the input. */
using ANNController = ANNDiagonalController;

/** degree of the ReLU approximation. FHEON's ResNet20Diagonal reads the scale of every ReLU from the decrypted
 *  activations and uses degree 27; here the server never decrypts and every ReLU has a fixed scale (see
 *  resnet20_fheon.cpp) that bounds the activations of every image. These scales are larger than the ones of a single
 *  image, and the error of the approximation grows with the scale. Plaintext simulation of the circuit (Chebyshev ReLU
 *  of the given degree, fixed scales) over CIFAR-10 test images, exact ReLU 92.9%: degree 27 below 30%, degree 59
 *  90.6%, degree 119 92.4% (97.9% of the predictions of the exact network, as degree 27 with per-image scales) */
#ifndef RESNET20_RELU_DEGREE
#define RESNET20_RELU_DEGREE 119
#endif
const int reluDegree = RESNET20_RELU_DEGREE;
/** levels of a ReLU: the Chebyshev evaluation (4 at degree 13, 5 at 27, 6 at 59, 7 at 119) and the scaling of its input */
const int reluLevels = ((reluDegree > 59) ? 7 : (reluDegree > 27) ? 6 : (reluDegree > 13) ? 5 : 4) + 1;
/** several slot counts are bootstrapped (one per stage): they must share the same correction factor */
const uint32_t bootstrapCorrectionFactor = 9;
/** a single bootstrapping iteration */
const int bootstrapIterations = 1;

const int imgCols = 32;
const int imgDepth = 3;
const int numStages = 3;
const vector<int> channelValues = {16, 32, 64, 10};

/** CKKS parameters of ResNet20Diagonal. The security level is FHEON_SECURITY_LEVEL (cmake -DSECURITY_LEVEL): at 128
 *  bits OpenFHE picks the ring dimension of the modulus chain and the 2^14 slots are sparsely packed in it */
FHEONHEController::HEConfig resnet20_he_config(const string &keysFolder);
/** slot count of every stage: channels * width^2 (16384, 8192, 4096) */
vector<int> resnet20_stage_slots();
/** key file of a stage: its bootstrapping keys and the rotations of its layers */
string resnet20_stage_key_file(int stage);
vector<int> resnet20_stage_rotation_keys(ANNController &fheonANNController, int stage);
/** level the input is encrypted at: the circuit before the first bootstrapping (3 convolutions and 2 ReLUs) ends with 3 towers */
int resnet20_input_level(const CryptoContext<DCRTPoly> &context);

/** the inference runs stage by stage, every stage with its own key file: the server loads the keys of a stage once and
 *  runs that stage for every image (the keys of a stage are several GB, too many to reload for every image).
 *  resnet20_use_stage_keys: the keys of a stage in place of the keys in memory, and the bootstrapping precomputations of
 *  its slot count.
 *  resnet20_stage: the layers of a stage for one image, without decryption. state is {input ciphertext} for stage 0;
 *  a stage that ends with a stride-2 block returns the convolution and shortcut outputs of that block (its first part runs
 *  with the keys of the stage it ends), the last stage returns {logits} (the 10 logits in the first slots) */
void resnet20_use_stage_keys(FHEONHEController &fheonHEController, int stage);
void resnet20_stage(FHEONHEController &fheonHEController, ANNController &fheonANNController, int stage, vector<Ctext> &state);

#endif // ifndef RESNET20_FHEON_H_
