
/***********************************************************************************************************************
*
* @author: Nges Brian, Njungle
*
* MIT License
* Copyright (c) 2025 Secure, Trusted and Assured Microelectronics, Arizona State
University

* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:

* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.

* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
********************************************************************************************************************/

#include "resnet20_fheon.h"
#include <iostream>
#include <sys/stat.h>

using namespace std;
using namespace lbcrypto;

#ifndef WEIGHTS_DIR
#define WEIGHTS_DIR "./submissions/cifar10/weights/resnet20/"
#endif

/** fixed scale of every ReLU (in the order of the network: layer0, then relu1 and relu2 of every block). The server
 *  never decrypts, so the scales cannot be read from the activations as FHEON's ResNet20Diagonal does: every scale bounds
 *  |x| at its ReLU for every image (the Chebyshev approximation diverges outside [-scale, scale]). They are the largest
 *  pre-activations of the plaintext network over the CIFAR-10 training and test sets (60000 images), +5%, rounded up */
static const vector<double> reluScales = {
    9,                                      // layer0
    19, 15,   11, 16,   7, 16,              // layer1 blocks 1-3 (relu1, relu2)
    8, 13,    5, 15,    6, 17,              // layer2 blocks 1-3
    7, 13,    8, 42,    3, 48};             // layer3 blocks 1-3

/** a bootstrapping is followed either by a ReLU and a convolution (reluLevels + 1 levels) or by a stride-2 block.
 *  When the two channel stages of the stride-2 downsampling do not fit in those levels, a single stage is used */
static FHEONLayout stride2Layout(int inputWidth, int inputChannels){
    FHEONLayout layout;
    int twoStageLevels = static_cast<int>(log2(inputWidth/2)) + 1 + 2 + 1;
    layout.channelGroup = (twoStageLevels > reluLevels + 1) ? inputChannels : 0;
    return layout;
}

FHEONHEController::HEConfig resnet20_he_config(const string &keysFolder) {
    FHEONHEController::HEConfig heinitConfig;
    heinitConfig.ringDim = 15;
    heinitConfig.numSlots = 14;
    heinitConfig.mlevelBootstrap = reluLevels + 2;
    heinitConfig.dcrtBits = 48;
    heinitConfig.firstMod = 52;
    heinitConfig.numDigits = 4;
    heinitConfig.levelBudget = {4, 4};
    heinitConfig.serialize = true;
    heinitConfig.keysFolder = keysFolder;
    /** no keys with the context: the stage key files hold exactly the bootstrapping and rotation keys of their layers
     *  (the diagonal primitives never call EvalSum) */
    heinitConfig.contextBootstrapKeys = false;
    heinitConfig.contextSumKeys = false;
    return heinitConfig;
}

vector<int> resnet20_stage_slots() {
    vector<int> stageSlots;
    for (int stage = 0; stage < numStages; stage++) {
        stageSlots.push_back(channelValues[stage]*(imgCols >> stage)*(imgCols >> stage));
    }
    return stageSlots;
}

string resnet20_stage_key_file(int stage) {
    return "layer" + to_string(stage+1) + ".bin";
}

/** rotations of a stage: its convolutions, the first layer (stage 1), and the stride-2 block that ends it or the pooling + linear layer */
vector<int> resnet20_stage_rotation_keys(ANNController &fheonANNController, int stage){
    int width = imgCols >> stage;
    int channels = channelValues[stage];
    vector<vector<int>> keys;
    keys.push_back(fheonANNController.generate_convolution_rotation_positions(width, channels, channels, 3, 1, 1));
    if (stage == 0) {
        keys.push_back(fheonANNController.generate_convolution_rotation_positions(width, imgDepth, channels, 3, 1, 1));
    }
    if (stage + 1 < numStages) {
        keys.push_back(fheonANNController.generate_convolution_rotation_positions(width, channels, channelValues[stage+1], 3, 1, 2, stride2Layout(width, channels)));
    } else {
        FHEONLayout layout;
        layout.inputSlots = channels * width * width;
        keys.push_back(fheonANNController.generate_linear_rotation_positions(channelValues[3], 0, layout));
    }
    return serialize_rotation_keys(keys);
}

int resnet20_input_level(const CryptoContext<DCRTPoly> &context) {
    int numTowers = context->GetCryptoParameters()->GetElementParams()->GetParams().size();
    return max(1, numTowers - (3 + 2*reluLevels + 3));
}

/************************************************* layers *************************************************/

static Ctext convolution_block(FHEONHEController &fheonHEController, ANNController &fheonANNController, string layer,
                               Ctext &encryptedInput, int dataWidth, int inputChannels, int outputChannels){
    string dataPath = string(WEIGHTS_DIR) + layer;
    int kernelWidth = 3;
    auto biasVector = load_bias(dataPath+"_bias.csv");
    auto rawKernelData = load_weights(dataPath+"_weight.csv", outputChannels, inputChannels, kernelWidth, kernelWidth);
    int numSlots = encryptedInput->GetSlots();
    int encode_level = fheonANNController.convolution_kernel_level(encryptedInput, dataWidth);
    auto kernelData = fheonANNController.encode_convolution_kernel(fheonHEController, rawKernelData, dataWidth, numSlots, encode_level);
    auto biasEncoded = fheonANNController.encode_convolution_bias(fheonHEController, biasVector, dataWidth, numSlots, encode_level);
    return fheonANNController.he_convolution(encryptedInput, kernelData, biasEncoded, dataWidth, inputChannels, outputChannels, 3, 1);
}

static vector<Ctext> double_shortcut_convolution_block(FHEONHEController &fheonHEController, ANNController &fheonANNController, string layer,
                                                       Ctext &encryptedInput, int dataWidth, int inputChannels, int outputChannels){
    string dataPath = string(WEIGHTS_DIR) + layer;
    int kernelWidth = 3;
    int widthOut = dataWidth/2;
    int inputSlots = encryptedInput->GetSlots();
    FHEONLayout layout = stride2Layout(dataWidth, inputChannels);
    int encode_level = fheonANNController.convolution_kernel_level(encryptedInput, dataWidth, 2, layout);

    auto biasVector = load_bias(dataPath+"_conv1_bias.csv");
    auto rawKernelData = load_weights(dataPath+"_conv1_weight.csv", outputChannels, inputChannels, kernelWidth, kernelWidth);
    auto kernelData = fheonANNController.encode_convolution_kernel(fheonHEController, rawKernelData, dataWidth, inputSlots, encode_level, 2, 1, layout);
    auto biasEncoded = fheonANNController.encode_convolution_bias(fheonHEController, biasVector, widthOut, inputSlots/2, encode_level);

    auto shortcutbiasVector = load_bias(dataPath+"_shortcut_bias.csv");
    auto shortcutrawKernelData = load_fc_weights(dataPath+"_shortcut_weight.csv",  outputChannels, inputChannels);
    auto shortcutKernelData = fheonANNController.encode_shortcut_kernel(fheonHEController, shortcutrawKernelData, dataWidth, inputSlots, encode_level);
    auto shortcutBiasEncoded = fheonANNController.encode_convolution_bias(fheonHEController, shortcutbiasVector, widthOut, inputSlots/2, encode_level);

    return fheonANNController.he_convolution_and_shortcut_optimized(encryptedInput, kernelData, shortcutKernelData, biasEncoded, shortcutBiasEncoded,
                                                                   dataWidth, inputChannels, outputChannels, layout);
}

void resnet20_use_stage_keys(FHEONHEController &fheonHEController, int stage){
    vector<int> stageSlots = resnet20_stage_slots();
    fheonHEController.clear_context(stageSlots[(stage + numStages - 1) % numStages]);
    fheonHEController.read_evaluation_keys(fheonHEController.getContext(), resnet20_stage_key_file(stage));
    fheonHEController.prepare_bootstrapping(stageSlots[stage]);
}

/** first part of a stride-2 block, with the keys of the stage it ends: bootstrapping, stride-2 convolution and shortcut.
 *  The two outputs have the slot count of the next stage */
static vector<Ctext> downsample_head(FHEONHEController &fheonHEController, ANNController &fheonANNController, string layer, Ctext encryptedInput,
                                     int dataWidth, int inputChannels, int outputChannels){
    printModelLayer(layer);
    encryptedInput = fheonHEController.bootstrap_function(encryptedInput, bootstrapIterations);
    return double_shortcut_convolution_block(fheonHEController, fheonANNController, layer, encryptedInput, dataWidth, inputChannels, outputChannels);
}

/** rest of a block from the output of its first convolution: ReLU, second convolution, shortcut sum, ReLU */
static Ctext block_tail(FHEONHEController &fheonHEController, ANNController &fheonANNController, string layer, Ctext convData,
                        Ctext shortcut_convData, int dataWidth, int channels, bool bootstrapState, int reluIndex){
    int dataSize = convData->GetSlots();
    if (bootstrapState) {
        convData = fheonHEController.bootstrap_function(convData, bootstrapIterations);
    }
    convData = fheonANNController.he_relu(convData, reluScales[reluIndex], dataSize, reluDegree);

    auto second_convData = convolution_block(fheonHEController, fheonANNController, layer+"_conv2", convData, dataWidth, channels, channels);
    Ctext sum_convData = fheonANNController.he_sum_two_ciphertexts(second_convData, shortcut_convData);
    sum_convData = fheonHEController.bootstrap_function(sum_convData, bootstrapIterations);
    return fheonANNController.he_relu(sum_convData, reluScales[reluIndex+1], dataSize, reluDegree);
}

/** a block with the identity shortcut */
static Ctext resnet_block(FHEONHEController &fheonHEController, ANNController &fheonANNController, string layer, Ctext encryptedInput,
                          int dataWidth, int channels, bool bootstrapState, int reluIndex){
    printModelLayer(layer);
    Ctext convData = convolution_block(fheonHEController, fheonANNController, layer+"_conv1", encryptedInput, dataWidth, channels, channels);
    return block_tail(fheonHEController, fheonANNController, layer, convData, encryptedInput, dataWidth, channels, bootstrapState, reluIndex);
}

static Ctext avgpool_fc_layer_block(FHEONHEController &fheonHEController, ANNController &fheonANNController, string layer,
                                    Ctext &encryptedInput, int dataWidth, int inputChannels, int outputChannels){
    printModelLayer(layer);
    string dataPath = string(WEIGHTS_DIR) + layer;
    auto fc_biasVector = load_bias(dataPath+"_bias.csv");
    auto fc_rawKernelData = load_fc_weights(dataPath+"_weight.csv", outputChannels, inputChannels);
    /** the global average pooling is part of the linear layer: every input is the average of the dataWidth^2 values of its channel */
    int channelSize = dataWidth*dataWidth;
    FHEONLayout layout;
    layout.weightScale = 1.0/channelSize;
    layout.inputPositions.resize(inputChannels);
    for (int ch = 0; ch < inputChannels; ch++) {
        for (int i = 0; i < channelSize; i++) { layout.inputPositions[ch].push_back(ch*channelSize + i); }
    }
    int numSlots = encryptedInput->GetSlots();
    int encode_level = fheonANNController.convolution_kernel_level(encryptedInput, 1);
    auto fc_kernelData = fheonANNController.encode_linear_weights(fheonHEController, fc_rawKernelData, numSlots, encode_level, layout);
    Ptext fc_biasEncoded = fheonANNController.encode_linear_bias(fheonHEController, fc_biasVector, numSlots, encode_level);
    return fheonANNController.he_linear(encryptedInput, fc_kernelData, fc_biasEncoded, inputChannels, outputChannels, 0, layout);
}

void resnet20_stage(FHEONHEController &fheonHEController, ANNController &fheonANNController, int stage, vector<Ctext> &state) {
    int dataWidth = imgCols >> stage;
    int channels = channelValues[stage];
    string stageName = "layer" + to_string(stage+1);
    Ctext convData;
    if (stage == 0) {
        printModelLayer("layer0");
        convData = convolution_block(fheonHEController, fheonANNController, "layer0_conv1", state[0], dataWidth, imgDepth, channels);
        convData = fheonANNController.he_relu(convData, reluScales[0], convData->GetSlots(), reluDegree);
        /** the first block runs on the fresh ciphertext */
        convData = resnet_block(fheonHEController, fheonANNController, stageName + "_block1", convData, dataWidth, channels, false, 1);
    } else {
        /** the stride-2 block that started with the keys of the previous stage */
        convData = block_tail(fheonHEController, fheonANNController, stageName + "_block1", state[0], state[1], dataWidth, channels, true,
                              1 + 6*stage);
    }
    for (int block = 2; block <= 3; block++) {
        convData = resnet_block(fheonHEController, fheonANNController, stageName + "_block" + to_string(block), convData, dataWidth, channels,
                                true, 1 + 6*stage + 2*(block-1));
    }
    if (stage + 1 < numStages) {
        state = downsample_head(fheonHEController, fheonANNController, "layer" + to_string(stage+2) + "_block1", convData, dataWidth,
                                channels, channelValues[stage+1]);
        return;
    }
    /** the pooling and the linear layer need one level: bootstrap only when it is not available */
    if (convData->GetElements()[0].GetNumOfElements() - (convData->GetNoiseScaleDeg() - 1) < 3) {
        convData = fheonHEController.bootstrap_function(convData, bootstrapIterations);
    }
    state = {avgpool_fc_layer_block(fheonHEController, fheonANNController, "layer_fc", convData, dataWidth, channels, channelValues[3])};
}
