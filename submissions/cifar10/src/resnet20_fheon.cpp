
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
#define WEIGHTS_DIR "./submissions/cifar10/resnet20/weights/resnet20/"
#endif

Ctext convolution_block(FHEONHEController &fheonHEController, FHEONANNController &fheonANNController, string layer, Ctext &encrytedInput, 
                        int &dataWidth, int kernelWidth, int striding, int inputChannels, int outputChannels);
vector<Ctext> double_shortcut_convolution_block(FHEONHEController &fheonHEController, FHEONANNController &fheonANNController, string layer, 
                        Ctext &encrytedInput, int &dataWidth, int inputChannels, int outputChannels);
Ctext resnet_block(FHEONHEController &fheonHEController, FHEONANNController &fheonANNController, string layer, Ctext &encrytedInput,
                        int &dataWidth, int &dataSize, int inputChannels, int outputChannels, int bootstrapState, bool shortcutConv);
Ctext fc_layer_block(FHEONHEController &fheonHEController, FHEONANNController &fheonANNController, string layer,
                     Ctext encryptedInput, int inputChannels, int outputChannels);

Ctext resnet20(FHEONHEController &fheonHEController, CryptoContext<DCRTPoly> &context, Ctext &encryptedInput, string pubkey_dir) {

    FHEONANNController fheonANNController(context);

    int img_depth = 3;
    int img_cols = 32;
    int kernelSize = 3;
    int striding = 1;
    int avgpoolSize = 8;
    vector<int> channelValues = {16, 32, 64, 10};
    int rotPositions = 16;
    int dataWidth = img_cols;
    int dataSize = img_depth * pow(img_cols, 2);
    int reluScale = 10;
    int polyDeg = 59;

    cout << "         [server] Starting encrypted ResNet20 inference" << endl;

    string mk_file = "mk.bin";
    cout << "         [server] Layer 0" << endl;
    string l1_rk = "layer1_rk.bin";
    fheonHEController.read_evaluation_keys(context, pubkey_dir, mk_file, l1_rk);
    context->EvalBootstrapSetup(config.levelBudget);
    Ctext convData = convolution_block(fheonHEController, fheonANNController, "layer0_conv1", encryptedInput,
                                    dataWidth, kernelSize, striding, img_depth, channelValues[0]);
    dataSize = channelValues[0] * pow(dataWidth, 2);
    convData = fheonANNController.he_relu(convData, reluScale, dataSize, polyDeg);

    cout << "         [server] Layer 1" << endl;
    cout << "                  [server] Block 1" << endl;
    convData = resnet_block(fheonHEController, fheonANNController,"layer1_block1", convData, dataWidth, dataSize,
                            channelValues[0], channelValues[0], false, false);
    cout << "                  [server] Block 2" << endl;
    convData = resnet_block(fheonHEController, fheonANNController, "layer1_block2", convData, dataWidth, dataSize,
                            channelValues[0], channelValues[0], true, false);
    cout << "                  [server] Block 3" << endl;
    convData = resnet_block(fheonHEController, fheonANNController, "layer1_block3", convData, dataWidth, dataSize,
                            channelValues[0], channelValues[0], true, false);

    string l2_rk = "layer2_rk.bin";
    fheonHEController.read_evaluation_keys(context, pubkey_dir, mk_file, l2_rk);
    cout << "         [server] Layer 2" << endl;
    cout << "                  [server] Block 1" << endl;
    convData = resnet_block(fheonHEController, fheonANNController, "layer2_block1", convData, dataWidth, dataSize,
                            channelValues[0], channelValues[1], true, true);
    cout << "                  [server] Block 2" << endl;
    convData = resnet_block(fheonHEController, fheonANNController, "layer2_block2", convData, dataWidth, dataSize,
                            channelValues[1], channelValues[1], true, false);
    cout << "                  [server] Block 3" << endl;
    convData = resnet_block(fheonHEController, fheonANNController, "layer2_block3", convData, dataWidth, dataSize,
                            channelValues[1], channelValues[1], true, false);

    string l3_rk = "layer3_rk.bin";
    fheonHEController.read_evaluation_keys(context, pubkey_dir, mk_file, l3_rk);
    cout << "         [server] Layer 3" << endl;
    cout << "                  [server] Block 1" << endl;
    convData = resnet_block(fheonHEController, fheonANNController, "layer3_block1", convData, dataWidth, dataSize,
                            channelValues[1], channelValues[2], true, true);
    cout << "                  [server] Block 2" << endl;
    convData = resnet_block(fheonHEController, fheonANNController, "layer3_block2", convData, dataWidth, dataSize,
                            channelValues[2], channelValues[2], true, false);
    cout << "                  [server] Block 3" << endl;
    convData = resnet_block(fheonHEController, fheonANNController, "layer3_block3", convData, dataWidth, dataSize,
                            channelValues[2], channelValues[2], true, false);

    string l4_rk = "layer4_rk.bin";
    fheonHEController.read_evaluation_keys(context, pubkey_dir, mk_file, l4_rk);
    cout << "         [server] Pool + Classifier" << endl;
    convData = fheonHEController.bootstrap_function(convData);
    convData = fheonANNController.he_globalavgpool( convData, dataWidth, channelValues[2], avgpoolSize, rotPositions);
    convData = fc_layer_block(fheonHEController, fheonANNController, "layer_fc", convData, channelValues[2], channelValues[3]);
    return convData;
}

Ctext convolution_block(FHEONHEController &fheonHEController, FHEONANNController &fheonANNController, string layer, Ctext &encrytedInput, 
                            int &dataWidth, int kernelWidth, int striding, int inputChannels, int outputChannels) {

    int widthSq = pow(dataWidth, 2);
    int encodeLevel = encrytedInput->GetLevel();
    vector<vector<Ptext>> kernelData;
    string dataPath = string(WEIGHTS_DIR) + layer;

    auto biasData = load_bias(dataPath + "_bias.csv");
    auto rawKernel = load_weights(dataPath + "_weight.csv", outputChannels,
                                    inputChannels, kernelWidth, kernelWidth);
    for (int i = 0; i < outputChannels; i++) {
        auto encodeKernel = fheonHEController.encode_kernel_optimized(rawKernel[i], widthSq, encodeLevel);
        kernelData.push_back(encodeKernel);
    }

    auto biasDataEncoded = fheonHEController.encode_bais_input(biasData, widthSq, encodeLevel);
    auto conv_data = fheonANNController.he_convolution_optimized(encrytedInput, kernelData, 
                                biasDataEncoded, dataWidth, inputChannels, outputChannels, striding);

    kernelData.clear();
    kernelData.shrink_to_fit();
    biasData.clear();
    rawKernel.clear();
    rawKernel.shrink_to_fit();

    return conv_data;
}

vector<Ctext> double_shortcut_convolution_block(FHEONHEController &fheonHEController, FHEONANNController &fheonANNController, 
                    string layer, Ctext &encrytedInput, int &dataWidth, int inputChannels, int outputChannels) {

    string dataPath = string(WEIGHTS_DIR) + layer;
    int widthSq = pow(dataWidth, 2);
    int widthOutSq = pow((dataWidth / 2), 2);
    int kernelWidth = 3;
    int encodeLevel = encrytedInput->GetLevel();
    vector<vector<Ptext>> kernelData;
    vector<Ptext> shortcutkernelData;

    /*** convolution and shortcut data */
    auto biasData = load_bias(dataPath + "_conv1_bias.csv");
    auto shortcutbiasData = load_bias(dataPath + "_shortcut_bias.csv");
    auto rawKernel = load_weights(dataPath + "_conv1_weight.csv", outputChannels,
                                    inputChannels, kernelWidth, kernelWidth);
    auto shortcutrawKernel = load_fc_weights(dataPath + "_shortcut_weight.csv",
                                            outputChannels, inputChannels);
    for (int i = 0; i < outputChannels; i++) {
        auto encodeKernel = fheonHEController.encode_kernel_optimized(rawKernel[i], widthSq, encodeLevel);
        auto encodeWeights = fheonHEController.encode_bais_input(shortcutrawKernel[i], widthSq);
        kernelData.push_back(encodeKernel);
        shortcutkernelData.push_back(encodeWeights);
    }
    auto biasDataEncoded = fheonHEController.encode_bais_input(biasData, widthOutSq);
    auto shortcutbiasDataEncoded = fheonHEController.encode_bais_input(shortcutbiasData, widthOutSq);

    auto returnedCiphers = fheonANNController.he_convolution_and_shortcut_optimized(
                                    encrytedInput, kernelData, shortcutkernelData, biasDataEncoded,
                                    shortcutbiasDataEncoded, dataWidth, inputChannels, outputChannels);

    kernelData.clear();
    kernelData.shrink_to_fit();
    rawKernel.clear();
    rawKernel.shrink_to_fit();
    biasData.clear();

    shortcutkernelData.clear();
    shortcutkernelData.shrink_to_fit();
    shortcutrawKernel.clear();
    shortcutrawKernel.shrink_to_fit();
    shortcutbiasData.clear();
    return returnedCiphers;
}

Ctext resnet_block(FHEONHEController &fheonHEController, FHEONANNController &fheonANNController, string layer, Ctext &encrytedInput, 
                    int &dataWidth, int &dataSize, int inputChannels, int outputChannels, int bootstrapState, bool shortcutConv) {

    int kernelWidth = 3;
    int striding = 1;
    int polyDeg = 59;
    int reluScale = 15;
    Ctext convData;
    Ctext shortcutConvData = encrytedInput->Clone();

    if (shortcutConv) {
        encrytedInput = fheonHEController.bootstrap_function(encrytedInput);
        auto doubleResults = double_shortcut_convolution_block(fheonHEController, fheonANNController, layer, encrytedInput, dataWidth,
                                    inputChannels, outputChannels);
        dataWidth = dataWidth / 2;
        dataSize = (outputChannels * pow(dataWidth, 2));
        convData = doubleResults[0]->Clone();
        shortcutConvData = doubleResults[1]->Clone();
    } else {
        convData = convolution_block(fheonHEController, fheonANNController, layer + "_conv1", encrytedInput,
                                    dataWidth, kernelWidth, striding, inputChannels, outputChannels);
    }
    if (bootstrapState) {
        convData = fheonHEController.bootstrap_function(convData);
    }
    
    convData = fheonANNController.he_relu(convData, reluScale, dataSize, polyDeg);
    Ctext secConvData = convolution_block(fheonHEController, fheonANNController, layer + "_conv2", convData,
                                    dataWidth, kernelWidth, striding, outputChannels, outputChannels);
    Ctext sumConvData = fheonANNController.he_sum_two_ciphertexts(secConvData, shortcutConvData);
    sumConvData = fheonHEController.bootstrap_function(sumConvData);
    if (layer == "layer3_block2" || layer == "layer3_block3") {
            reluScale = 25;
    }
    sumConvData = fheonANNController.he_relu(sumConvData, reluScale, dataSize, polyDeg);
    return sumConvData;
}

Ctext fc_layer_block(FHEONHEController &fheonHEController, FHEONANNController &fheonANNController, string layer,
                     Ctext encryptedInput, int inputChannels, int outputChannels) {

    string dataPath = string(WEIGHTS_DIR) + layer;
    auto fcBiasData = load_bias(dataPath + "_bias.csv");
    auto fc_rawKernelData = load_fc_weights(dataPath + "_weight.csv", outputChannels, inputChannels);
    vector<Ptext> fcKernelData;
    for (int i = 0; i < outputChannels; i++) {
            auto encodeWeights = fheonHEController.encode_input(fc_rawKernelData[i]);
            fcKernelData.push_back(encodeWeights);
    }
    Ptext encodedbaisData = fheonHEController.encode_input(fcBiasData);
    Ctext fcData = fheonANNController.he_linear_optimized(encryptedInput, fcKernelData, encodedbaisData, inputChannels, outputChannels);

    fcKernelData.clear();
    fcKernelData.shrink_to_fit();
    fcBiasData.clear();
    fc_rawKernelData.clear();
    fcBiasData.clear();
    return fcData;
}