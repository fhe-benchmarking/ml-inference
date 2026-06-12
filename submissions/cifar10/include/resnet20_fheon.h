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

#include "FHEONANNController.h"
#include "FHEONHEController.h"
#include "openfhe.h"

using namespace std;
using namespace lbcrypto;

struct ResNet20Config {
  vector<uint32_t> levelBudget = {3, 3};
  vector<uint32_t> bsgsDim = {0, 0};
  int ringDim = 1 << 15;
  int numSlots = 1 << 14;
  int dcrtBits = 48;
  int firstMod = 52;
  int modelDepth = 11;
  int digitSize = 4;
};

inline ResNet20Config config;

// using CiphertextT = ConstCiphertext<DCRTPoly>;

Ctext resnet20(FHEONHEController &fheonHEController, CryptoContext<DCRTPoly> &v0, Ctext &v1, string pubkey_dir);

#endif // ifndef RESNET20_FHEON_H_