/***********************************************************************************************************************
 *
 * @author: Nges Brian, Njungle
 *
 * MIT License
 * Copyright (c) 2025 Secure, Trusted and Assured Microelectronics, Arizona State University
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 **********************************************************************************************************************/

#include "resnet20_fheon.h"

/** convolutions: 3x3, padding 1; stride 1 except the first convolution of a stride-2 block */
const int convPadding = 1;

/** 
 * every scale bounds |x| at its ReLU for every image (the Chebyshev approximation diverges outside [-scale, scale]). They are the largest
 *  pre-activations of the plaintext network over the CIFAR-10 training and test sets (60000 images), +5%, rounded up */
static const vector<double> reluScales = {
    9,                                      // layer0
    19, 15,   11, 16,   7, 16,              // layer1 blocks 1-3 (relu1, relu2)
    8, 13,    5, 15,    6, 17,              // layer2 blocks 1-3
    7, 13,    8, 42,    3, 48};             // layer3 blocks 1-3

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

/** a bootstrapping is followed either by a ReLU and a convolution (reluLevels + 1 levels) or by a stride-2 block.
 *  When the two channel stages of the stride-2 downsampling do not fit in those levels, a single stage is used */
FHEONLayout resnet20_stride2_layout(int inputWidth, int inputChannels) {
    FHEONLayout layout;
    int twoStageLevels = static_cast<int>(log2(inputWidth/2)) + 1 + 2 + 1;
    layout.channelGroup = (twoStageLevels > reluLevels + 1) ? inputChannels : 0;
    return layout;
}

int resnet20_input_level(const CryptoContext<DCRTPoly> &context) {
    int numTowers = context->GetCryptoParameters()->GetElementParams()->GetParams().size();
    return max(1, numTowers - (3 + 2*reluLevels + 3));
}

/*************************************************** Layers *********************************************************/

/** the layer being run, in the log of the server (independent of the FHEON verbosity) */
static void print_layer(const string &layer) {
    cout << "         [server] " << layer << endl;
}

/** 3x3 convolution, stride 1, with the weights of layer encoded at the level of the input */
static Ctext convolution_block(FHEONHEController &fheonHEController, ANNController &fheonANNController, ResNet20Weights &weights,
                               const string &layer, Ctext &x, int dataWidth, int inputChannels, int outputChannels) {
    int numSlots = x->GetSlots();
    int level = fheonANNController.convolution_kernel_level(x, dataWidth);
    auto kernel = fheonANNController.encode_convolution_kernel(fheonHEController, weights.conv.at(layer), dataWidth, numSlots, level);
    auto bias = fheonANNController.encode_convolution_bias(fheonHEController, weights.bias.at(layer), dataWidth, numSlots, level);
    return fheonANNController.he_convolution(x, kernel, bias, dataWidth, inputChannels, outputChannels, kernelWidth, convPadding);
}

/** stride-2 3x3 convolution and 1x1 shortcut of a downsampling block: both outputs have half the slots */
static vector<Ctext> double_shortcut_convolution_block(FHEONHEController &fheonHEController, ANNController &fheonANNController,
                                                       ResNet20Weights &weights, const string &block, Ctext &x, int dataWidth,
                                                       int inputChannels, int outputChannels) {
    int widthOut = dataWidth/2;
    int inputSlots = x->GetSlots();
    FHEONLayout layout = resnet20_stride2_layout(dataWidth, inputChannels);
    int level = fheonANNController.convolution_kernel_level(x, dataWidth, 2, layout);

    string conv = block + "_conv1";
    auto kernel = fheonANNController.encode_convolution_kernel(fheonHEController, weights.conv.at(conv), dataWidth, inputSlots, level,
                                                               2, convPadding, layout);
    auto bias = fheonANNController.encode_convolution_bias(fheonHEController, weights.bias.at(conv), widthOut, inputSlots/2, level);

    string shortcut = block + "_shortcut";
    auto shortcutKernel = fheonANNController.encode_shortcut_kernel(fheonHEController, weights.linear.at(shortcut), dataWidth, inputSlots, level);
    auto shortcutBias = fheonANNController.encode_convolution_bias(fheonHEController, weights.bias.at(shortcut), widthOut, inputSlots/2, level);

    return fheonANNController.he_convolution_and_shortcut_optimized(x, kernel, shortcutKernel, bias, shortcutBias,
                                                                   dataWidth, inputChannels, outputChannels, layout);
}

/** first part of a stride-2 block, with the keys of the stage it ends: bootstrapping, stride-2 convolution and shortcut.
 *  The two outputs have the slot count of the next stage */
static vector<Ctext> downsample_head(FHEONHEController &fheonHEController, ANNController &fheonANNController, ResNet20Weights &weights,
                                     const string &block, Ctext x, int dataWidth, int inputChannels, int outputChannels) {
    print_layer(block);
    x = fheonHEController.bootstrap_function(x, bootstrapIterations);
    return double_shortcut_convolution_block(fheonHEController, fheonANNController, weights, block, x, dataWidth, inputChannels, outputChannels);
}

/** rest of a block from the output of its first convolution: ReLU, second convolution, shortcut sum, ReLU */
static Ctext block_tail(FHEONHEController &fheonHEController, ANNController &fheonANNController, ResNet20Weights &weights,
                        const string &block, Ctext x, Ctext shortcut, int dataWidth, int channels, bool bootstrapFirst, int reluIndex) {
    int dataSize = x->GetSlots();
    if (bootstrapFirst) {
        x = fheonHEController.bootstrap_function(x, bootstrapIterations);
    }
    x = fheonANNController.he_relu(x, reluScales[reluIndex], dataSize, reluDegree);

    x = convolution_block(fheonHEController, fheonANNController, weights, block + "_conv2", x, dataWidth, channels, channels);
    x = fheonANNController.he_sum_two_ciphertexts(x, shortcut);
    x = fheonHEController.bootstrap_function(x, bootstrapIterations);
    return fheonANNController.he_relu(x, reluScales[reluIndex+1], dataSize, reluDegree);
}

/** a block with the identity shortcut */
static Ctext resnet_block(FHEONHEController &fheonHEController, ANNController &fheonANNController, ResNet20Weights &weights,
                          const string &block, Ctext x, int dataWidth, int channels, bool bootstrapFirst, int reluIndex) {
    print_layer(block);
    Ctext convData = convolution_block(fheonHEController, fheonANNController, weights, block + "_conv1", x, dataWidth, channels, channels);
    return block_tail(fheonHEController, fheonANNController, weights, block, convData, x, dataWidth, channels, bootstrapFirst, reluIndex);
}

/** global average pooling and the linear layer, in one: every input is the average of the dataWidth^2 values of its channel */
static Ctext avgpool_fc_layer_block(FHEONHEController &fheonHEController, ANNController &fheonANNController, ResNet20Weights &weights,
                                    const string &layer, Ctext &x, int dataWidth, int inputChannels, int outputChannels) {
    print_layer(layer);
    int channelSize = dataWidth*dataWidth;
    FHEONLayout layout;
    layout.weightScale = 1.0/channelSize;
    layout.inputPositions.resize(inputChannels);
    for (int ch = 0; ch < inputChannels; ch++) {
        for (int i = 0; i < channelSize; i++) { layout.inputPositions[ch].push_back(ch*channelSize + i); }
    }
    int numSlots = x->GetSlots();
    int level = fheonANNController.convolution_kernel_level(x, 1);
    auto fcWeights = fheonANNController.encode_linear_weights(fheonHEController, weights.linear.at(layer), numSlots, level, layout);
    Ptext fcBias = fheonANNController.encode_linear_bias(fheonHEController, weights.bias.at(layer), numSlots, level);
    return fheonANNController.he_linear(x, fcWeights, fcBias, inputChannels, outputChannels, 0, layout);
}

/*************************************************** Encrypted inference *********************************************/

/** access to the context registry of OpenFHE (AddContext is protected) */
struct ContextRegistry : CryptoContextFactory<DCRTPoly> {
    static void register_only(const CryptoContext<DCRTPoly> &context) {
        ReleaseAllContexts();
        AddContext(context);
    }
};

/** the keys of a stage in place of the keys in memory, and the bootstrapping precomputations of its slot count */
static void use_stage_keys(FHEONHEController &fheonHEController, int stage) {
    vector<int> stageSlots = resnet20_stage_slots();
    fheonHEController.clear_context(stageSlots[(stage + numStages - 1) % numStages]);
    fheonHEController.read_evaluation_keys(fheonHEController.getContext(), resnet20_stage_key_file(stage));
    /** loading the keys releases the crypto contexts of OpenFHE: register the context of the server again, so that the
     *  next ciphertext read from a file belongs to it */
    ContextRegistry::register_only(fheonHEController.getContext());
    fheonHEController.prepare_bootstrapping(stageSlots[stage]);
}

/** the layers of a stage for one image. state is {input ciphertext} for stage 0; a stage that ends with a stride-2 block
 *  returns the convolution and shortcut outputs of that block (its first part runs with the keys of the stage it ends),
 *  the last stage returns {logits} */
static void resnet20_stage(FHEONHEController &fheonHEController, ANNController &fheonANNController, ResNet20Weights &weights,
                    int stage, vector<Ctext> &state) {
    int dataWidth = imgCols >> stage;
    int channels = channelValues[stage];
    string stageName = "layer" + to_string(stage+1);
    Ctext x;
    if (stage == 0) {
        print_layer("layer0");
        x = convolution_block(fheonHEController, fheonANNController, weights, "layer0_conv1", state[0], dataWidth, imgDepth, channels);
        x = fheonANNController.he_relu(x, reluScales[0], x->GetSlots(), reluDegree);
        /** the first block runs on the fresh ciphertext */
        x = resnet_block(fheonHEController, fheonANNController, weights, stageName + "_block1", x, dataWidth, channels, false, 1);
    } else {
        /** the stride-2 block that started with the keys of the previous stage */
        x = block_tail(fheonHEController, fheonANNController, weights, stageName + "_block1", state[0], state[1], dataWidth, channels,
                       true, 1 + 6*stage);
    }
    for (int block = 2; block <= 3; block++) {
        x = resnet_block(fheonHEController, fheonANNController, weights, stageName + "_block" + to_string(block), x, dataWidth,
                         channels, true, 1 + 6*stage + 2*(block-1));
    }
    if (stage + 1 < numStages) {
        state = downsample_head(fheonHEController, fheonANNController, weights, "layer" + to_string(stage+2) + "_block1", x,
                                dataWidth, channels, channelValues[stage+1]);
        return;
    }
    /** the pooling and the linear layer need one level: bootstrap only when it is not available */
    if (x->GetElements()[0].GetNumOfElements() - (x->GetNoiseScaleDeg() - 1) < 3) {
        x = fheonHEController.bootstrap_function(x, bootstrapIterations);
    }
    state = {avgpool_fc_layer_block(fheonHEController, fheonANNController, weights, "layer_fc", x, dataWidth, channels, channelValues[3])};
}

Ctext resnet20_infer(FHEONHEController &fheonHEController, ANNController &fheonANNController, ResNet20Weights &weights,
                     const Ctext &encryptedImage) {
    vector<Ctext> state = {encryptedImage};
    for (int stage = 0; stage < numStages; stage++) {
        use_stage_keys(fheonHEController, stage);
        resnet20_stage(fheonHEController, fheonANNController, weights, stage, state);
    }
    return state[0];
}
