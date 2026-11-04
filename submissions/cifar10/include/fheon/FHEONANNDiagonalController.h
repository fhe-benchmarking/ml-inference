
/*********************************************************************************************************************** 
*
* @author: Nges Brian, Njungle 
*
* MIT License
* Copyright (c) 2025 Secure, Trusted and Assured Microelectronics, Arizona State University

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
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
********************************************************************************************************************/

#ifndef FHEON_ANNDIAGONALCONTROLLER_H
#define FHEON_ANNDIAGONALCONTROLLER_H

#include "FHEONANNStandardController.h"

/**
 * @brief Diagonal ANN controller: the ANN layers evaluated over channel diagonals (FHEON-D).
 *
 * It offers only the optimized primitives, with the names and arguments they have in FHEONANNStandardController
 * (the standard controller keeps every primitive and declares these exactly the same way), so a model written with
 * them switches implementation by changing only the controller type:
 *   he_convolution:  3x3 "same" convolution over channel diagonals (padding 1), or "valid" kernelWidth x kernelWidth
 *                    convolution on a strided layout (padding 0, see FHEONLayout)
 *   he_convolution_and_shortcut_optimized: stride-2 3x3 convolution and 1x1 shortcut, outputs with half the slots
 *   he_avgpool:      2x2 pooling, compacted onto layout.outputSlots slots, or left in place on a strided layout
 *                    (layout.rowStride set); it returns window sums, avgpool_weight_scale() is the factor for the next weights
 *   he_globalavgpool: channel averages gathered into the first slots
 *   he_linear:       diagonals with baby-step giant-step rotations, inputs anywhere (layout.inputPositions)
 *   he_relu:         scalar scaling, optionally normalized output
 */
class FHEONANNDiagonalController : protected FHEONANNStandardController {

public:
    FHEONANNDiagonalController(CryptoContext<DCRTPoly>& ctx) : FHEONANNStandardController(ctx) {}
    using FHEONANNStandardController::setContext;
    using FHEONANNStandardController::setNumSlots;
    using FHEONANNStandardController::num_slots;

    /****************************************************************************************************************
     * Optimized primitives, declared exactly as in FHEONANNStandardController
     ****************************************************************************************************************/
    vector<int> generate_convolution_rotation_positions(int inputWidth, int inputChannels, int outputChannels,
                                                    int kernelWidth, int padding, int stride, const FHEONLayout& layout = FHEONLayout());
    vector<int> generate_avgpool_rotation_positions(int inputWidth, int kernelWidth, int stride, int inputChannels, const FHEONLayout& layout = FHEONLayout());
    vector<int> generate_linear_rotation_positions(int maxFCLayeroutputs, int rotationPosition, const FHEONLayout& layout = FHEONLayout());
    Ctext he_convolution(Ctext& encryptedInput, vector<vector<Ptext>>& kernelData, Ptext& biasInput,
                            int inputWidth, int inputChannels, int outputChannels, int kernelWidth, int padding=0, int stride=1,
                            const FHEONLayout& layout = FHEONLayout());
    vector<Ctext> he_convolution_and_shortcut_optimized(const Ctext& encryptedInput, const vector<vector<Ptext>>& kernelData,
                            const vector<Ptext>& shortcutKernelData, Ptext& biasVector,  Ptext& shortcutBiasVector,
                            int inputWidth, int inputChannels, int outputChannels, const FHEONLayout& layout = FHEONLayout());
    Ctext he_avgpool(Ctext encryptedInput, int imgCols, int outputChannels, int kernelWidth=2, int stride=2, const FHEONLayout& layout = FHEONLayout());
    Ctext he_globalavgpool(Ctext& encryptedInput, int inputWidth, int outputChannels, int kernelWidth, int rotatePositions, const FHEONLayout& layout = FHEONLayout());
    Ctext he_linear(Ctext& encryptedInput, vector<Ptext>& weightMatrix, Ptext& biasInput, int inputSize, int outputSize, int rotatePositions,
                            const FHEONLayout& layout = FHEONLayout());
    Ctext he_relu(Ctext& encryptedInput, double scale, int vectorSize, int polyDegree = 59, bool normalizedOutput = false);
    Ctext he_sum_two_ciphertexts(Ctext& firstInput, Ctext& secondInput);
    int convolution_kernel_level(const Ctext& encryptedInput, int inputWidth, int stride = 1, const FHEONLayout& layout = FHEONLayout());
    double avgpool_weight_scale(int kernelWidth);
    vector<vector<Ptext>> encode_convolution_kernel(FHEONHEController& heController, vector<vector<vector<vector<double>>>>& kernelData,
                            int inputWidth, int numSlots, int encodeLevel, int stride = 1, int padding = 1, const FHEONLayout& layout = FHEONLayout());
    Ptext encode_convolution_bias(FHEONHEController& heController, vector<double>& biasData, int outputWidth, int numSlots, int encodeLevel,
                            const FHEONLayout& layout = FHEONLayout());
    vector<Ptext> encode_shortcut_kernel(FHEONHEController& heController, vector<vector<double>>& kernelData, int inputWidth, int numSlots, int encodeLevel,
                            const FHEONLayout& layout = FHEONLayout());
    vector<Ptext> encode_linear_weights(FHEONHEController& heController, vector<vector<double>>& weightData, int numSlots, int encodeLevel,
                            const FHEONLayout& layout = FHEONLayout());
    Ptext encode_linear_bias(FHEONHEController& heController, vector<double>& biasData, int numSlots, int encodeLevel, const FHEONLayout& layout = FHEONLayout());

protected:
    /************************************ diagonal implementations ************************************/
    vector<int> generate_diagonal_convolution_rotation_positions(int inputWidth, int inputChannels, int stride = 1, int channelGroup = 0);
    vector<int> generate_avgpool_linear_rotation_positions(int inputWidth, int inputChannels, int outputs);
    vector<int> generate_convolution_layout_rotation_positions(int kernelWidth, int rowStride, int colStride, int channelStride, int numGroups, const vector<int>& subShifts);
    vector<int> generate_sumpool_layout_rotation_positions(int rowStride, int colStride);
    vector<int> generate_linear_diagonal_rotation_positions(int numSlots, int numDiagonals, int babySteps);
    vector<int> generate_avgpool_diagonal_rotation_positions(int inputWidth, int inputChannels, int outputSlots, int channelGroup = 0);
    int diagonal_kernel_level(const Ctext& encryptedInput, int inputWidth, int stride = 1, int channelGroup = 0);

    Ctext he_convolution_diagonal(const Ctext& encryptedInput, const vector<vector<Ptext>>& kernelData, Ptext& biasInput, int inputWidth);
    vector<Ctext> he_convolution_and_shortcut_diagonal(const Ctext& encryptedInput, const vector<vector<Ptext>>& kernelData,
                            const vector<Ptext>& shortcutKernelData, Ptext& biasInput, Ptext& shortcutBiasInput,
                            int inputWidth, int inputChannels, int channelGroup = 0);
    Ctext he_convolution_layout(const Ctext& encryptedInput, const vector<vector<vector<Ptext>>>& kernelData, Ptext& biasInput,
                            int kernelWidth, int rowStride, int colStride, int channelStride, int numGroups, const vector<int>& subShifts);
    Ctext he_sumpool_layout(const Ctext& encryptedInput, int rowStride, int colStride);
    Ctext he_avgpool_diagonal(const Ctext& encryptedInput, int inputWidth, int inputChannels, int outputSlots, int channelGroup = 0);
    Ctext he_linear_diagonal(const Ctext& encryptedInput, const vector<vector<Ptext>>& weightDiagonals, Ptext& biasInput, int numDiagonals);
    Ctext he_globalavgpool_linear(const Ctext& encryptedInput, const vector<Ptext>& weightMatrix, Ptext& biasInput, int inputWidth, int inputChannels);
    Ctext he_relu_scalar(const Ctext& encryptedInput, double scale, int polyDegree = 59);
    Ctext he_relu_normalized(const Ctext& encryptedInput, double scale, int polyDegree = 59);

    /** baby steps of a diagonal linear layer with numDiagonals diagonals (power of two >= sqrt(numDiagonals)) */
    static int linear_baby_steps(int numDiagonals);
    /** strided-layout arguments of he_convolution with the defaults filled in */
    FHEONLayout convolution_layout(const FHEONLayout& layout, int inputWidth, int outputChannels, int kernelWidth);

    Ctext diagonal_channel_sum(const vector<Ctext>& taps, const vector<vector<Ptext>>& kernelData, int blockSize);
    Ctext plain_inner_product(const vector<Ctext>& taps, const vector<Ptext>& kernelData);
    Ctext rescale_pending(const Ctext& input);
    vector<Ctext> downsample_channels(const vector<Ctext>& inputs, int inputWidth, int inputChannels, int channelGroup);
    Ctext masked_rotation_sum(Ctext input, const vector<Ptext>& masks, int rotation);
    int diagonal_channel_group(int numChannels);
};

/** the controller of the single-input models (sequential) */
using ANNDiagonalController = FHEONANNDiagonalController;

#endif // FHEON_ANNDIAGONALCONTROLLER_H
