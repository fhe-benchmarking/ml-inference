
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

/**
 * @brief FHE Controller for defining and managing homomorphic ANN functions.
 *
 * This class provides different methods for convolution, pooling, fully connected
 *  relu, etc for neural network development on encrypted data using FHE.
 */

#include <fstream>
#include <filesystem>
#include <iostream>
#include <cmath>
#include <functional>
#include <thread>
#include "FHEONANNDiagonalController.h"

/**
 * @brief Rotation positions needed by he_convolution_diagonal (stride 1) or
 *        he_convolution_and_shortcut_diagonal (stride 2).
 *
 * @param inputWidth Spatial width of one input channel.
 * @param inputChannels Number of channel blocks in the input ciphertext.
 * @param stride 1 or 2.
 * @param channelGroup Stride 2: channel blocks compacted together in the first channel stage of the downsampling (0 = default, diagonal_channel_group); unused for stride 1.
 * @return Rotation positions.
 */
vector<int> FHEONANNDiagonalController::generate_diagonal_convolution_rotation_positions(int inputWidth, int inputChannels, int stride, int channelGroup){
    int inputSize = inputWidth * inputWidth;
    if (stride == 1) {
        return {1, -1, inputWidth, -inputWidth, inputWidth + 1, -(inputWidth + 1), inputWidth - 1, -(inputWidth - 1), inputSize};
    }
    int outputWidth = inputWidth / 2;
    int outputSize = outputWidth * outputWidth;
    int inputSlots = inputChannels * inputSize;
    vector<int> keys_position = {1, inputWidth, inputWidth + 1, 2 * inputWidth - outputWidth, inputSize - outputSize,
                                 (channelGroup > 0 ? channelGroup : diagonal_channel_group(inputChannels)) * (inputSize - outputSize),
                                 -(inputSlots / 4), -(inputSlots / 2), -1, -outputWidth, -(outputWidth + 1), outputSize};
    for (int step = 1; step < outputWidth; step <<= 1) {
        keys_position.push_back(step);
    }
    return keys_position;
}

/**
 * @brief Rotation positions needed by he_globalavgpool_linear.
 *
 * @param inputWidth Spatial width of one input channel.
 * @param inputChannels Number of input channels.
 * @param outputs Number of outputs of the linear layer.
 * @return Rotation positions.
 */
vector<int> FHEONANNDiagonalController::generate_avgpool_linear_rotation_positions(int inputWidth, int inputChannels, int outputs){
    int inputSize = inputWidth * inputWidth;
    vector<int> keys_position;
    for (int step = 1; step < inputSize; step <<= 1) {
        keys_position.push_back(step);
    }
    for (int k = 1; k < outputs; k++) {
        keys_position.push_back(-k);
    }
    for (int step = 1; step < inputChannels; step <<= 1) {
        keys_position.push_back(step * inputSize);
    }
    return keys_position;
}

/**
 * @brief Level at which the kernel plaintexts of the diagonal convolutions are multiplied.
 *
 * Encoding the kernels at this level avoids any plaintext level adjustment during the
 * convolution. For stride 2 it accounts for the levels used by the downsampling.
 *
 * @param encryptedInput Input ciphertext of the convolution.
 * @param inputWidth Spatial width of one input channel.
 * @param stride 1 or 2.
 * @param channelGroup Stride 2: channel blocks compacted together in the first channel stage of the downsampling (0 = default, diagonal_channel_group); unused for stride 1.
 * @return Encoding level for the kernel plaintexts.
 */
int FHEONANNDiagonalController::diagonal_kernel_level(const Ctext& encryptedInput, int inputWidth, int stride, int channelGroup){
    int level = encryptedInput->GetLevel() + encryptedInput->GetNoiseScaleDeg() - 1;
    if (stride != 1) {
        int numChannels = encryptedInput->GetSlots() / (inputWidth * inputWidth);
        int group = channelGroup > 0 ? channelGroup : diagonal_channel_group(numChannels);
        int channelStages = (group < numChannels) ? 2 : 1;
        level += static_cast<int>(log2(inputWidth / stride)) + 1 + channelStages;
    }
    return level;
}

/**
 * @brief Number of channels compacted together in the first channel stage of the stride-2
 *        downsampling; when it is smaller than numChannels a second stage moves the groups.
 */
int FHEONANNDiagonalController::diagonal_channel_group(int numChannels){
    return (numChannels >= 16) ? 4 : numChannels;
}

/**
 * @brief Apply the rescaling that FLEXIBLEAUTO leaves pending after a multiplication.
 *
 * A ciphertext that is reused in many plaintext multiplications or rotations would otherwise
 * be rescaled again inside every multiplication, and rotated with one more tower than needed.
 */
Ctext FHEONANNDiagonalController::rescale_pending(const Ctext& input){
    if (input->GetNoiseScaleDeg() < 2) {
        return input;
    }
    Ctext result = input->Clone();
    context->GetScheme()->ModReduceInternalInPlace(result, BASE_NUM_LEVELS_TO_DROP);
    return result;
}

/**
 * @brief Evaluate sum_i rotate(input, i * rotation) * masks[i] with a single rotation key,
 *        and rescale the result.
 */
Ctext FHEONANNDiagonalController::masked_rotation_sum(Ctext input, const vector<Ptext>& masks, int rotation){
    Ctext result;
    for (size_t i = 0; i < masks.size(); i++) {
        if (i > 0) {
            input = context->EvalRotate(input, rotation);
        }
        Ctext masked = context->EvalMult(input, masks[i]);
        if (result) {
            context->EvalAddInPlace(result, masked);
        } else {
            result = masked;
        }
    }
    return rescale_pending(result);
}

/**
 * @brief Evaluate sum_j taps[j] * kernelData[j], skipping the null plaintexts.
 *
 * The first product goes through EvalMult, which sets the level and scaling metadata of the
 * result. When the other taps and plaintexts are at that same level and scale (the normal
 * case: rescaled taps, plaintexts encoded at the level of the taps), their products are
 * accumulated directly on the polynomials through one reused scratch polynomial. EvalMult
 * would allocate and copy a ciphertext and a plaintext for every product, and with hundreds
 * of products per layer those allocations dominate and make the timing erratic.
 * Anything that does not match falls back to EvalMult.
 *
 * @return The sum, or a null ciphertext when every plaintext is null.
 */
Ctext FHEONANNDiagonalController::plain_inner_product(const vector<Ctext>& taps, const vector<Ptext>& kernelData){
    Ctext result;
    size_t first = 0;
    DCRTPoly scratch;
    for (size_t j = 0; j < kernelData.size(); j++) {
        if (!kernelData[j]) continue;
        if (!result) {
            result = context->EvalMult(taps[j], kernelData[j]);
            first = j;
            continue;
        }
        const auto& tapElements = taps[j]->GetElements();
        const DCRTPoly& plainElement = kernelData[j]->GetElement<DCRTPoly>();
        auto& resultElements = result->GetElements();
        bool direct = taps[first]->GetNoiseScaleDeg() == 1 && taps[j]->GetNoiseScaleDeg() == 1
                      && kernelData[first]->GetNoiseScaleDeg() == 1 && kernelData[j]->GetNoiseScaleDeg() == 1
                      && taps[j]->GetLevel() == taps[first]->GetLevel() && kernelData[j]->GetLevel() == kernelData[first]->GetLevel()
                      && taps[j]->GetScalingFactor() == taps[first]->GetScalingFactor()
                      && kernelData[j]->GetScalingFactor() == kernelData[first]->GetScalingFactor()
                      && tapElements.size() == resultElements.size()
                      && plainElement.GetFormat() == Format::EVALUATION
                      && plainElement.GetNumOfElements() == tapElements[0].GetNumOfElements()
                      && plainElement.GetNumOfElements() == resultElements[0].GetNumOfElements();
        if (direct) {
            for (size_t i = 0; i < resultElements.size(); i++) {
                scratch = tapElements[i];
                scratch *= plainElement;
                resultElements[i] += scratch;
            }
        } else {
            context->EvalAddInPlace(result, context->EvalMult(taps[j], kernelData[j]));
        }
    }
    return result;
}

/**
 * @brief Evaluate sum_d rotate(sum_j taps[j] * kernelData[d][j], d * blockSize) with one
 *        rotation per diagonal (Horner form, single rotation key).
 */
Ctext FHEONANNDiagonalController::diagonal_channel_sum(const vector<Ctext>& taps, const vector<vector<Ptext>>& kernelData, int blockSize){
    Ctext result;
    for (int d = static_cast<int>(kernelData.size()) - 1; d >= 0; d--) {
        Ctext diagonal = plain_inner_product(taps, kernelData[d]);
        if (result) {
            result = context->EvalRotate(result, blockSize);
            if (diagonal) {
                context->EvalAddInPlace(result, diagonal);
            }
        } else {
            result = diagonal;
        }
    }
    return result;
}

/**
 * @brief 3x3, stride 1, "same" convolution evaluated over channel diagonals.
 *
 * Instead of folding the input channels once per output channel, the kernel is encoded as
 * channel diagonals (FHEONHEController::encode_kernel_diagonal), so the whole layer costs
 * 8 hoisted rotations for the taps, one rotation per diagonal and a single multiplicative
 * level; the padding masks are part of the kernel plaintexts.
 * The channels are stored in consecutive blocks of inputWidth^2 slots. When kernelData has as
 * many diagonals as the ciphertext has blocks, the input is used as is; when it has fewer
 * (encode_kernel_diagonal with an inputPeriod), the input must be zero beyond the first
 * inputPeriod blocks and is replicated first, which needs the rotations
 * -inputPeriod * inputWidth^2, -2 * inputPeriod * inputWidth^2, ... up to half the slots.
 *
 * @param encryptedInput Input ciphertext.
 * @param kernelData [diagonal][tap] plaintexts from encode_kernel_diagonal.
 * @param biasInput Bias plaintext from encode_bias_diagonal.
 * @param inputWidth Spatial width of one channel.
 * @return Encrypted convolution output, channels in consecutive blocks.
 */
Ctext FHEONANNDiagonalController::he_convolution_diagonal(const Ctext& encryptedInput, const vector<vector<Ptext>>& kernelData, Ptext& biasInput, int inputWidth){
    auto start_time = startTime();
    int inputSize = inputWidth * inputWidth;
    int numSlots = encryptedInput->GetSlots();
    int inputPeriod = kernelData.size() * inputSize;
    if (numSlots % inputPeriod != 0) {
        throw std::invalid_argument("he_convolution_diagonal: the ciphertext slots must be a multiple of numDiagonals * inputWidth^2");
    }

    Ctext input = rescale_pending(encryptedInput);
    /*** fewer diagonals than blocks: the input sits in the first blocks and is replicated ***/
    for (int period = inputPeriod; period < numSlots; period <<= 1) {
        input = context->EvalAdd(input, context->EvalRotate(input, -period));
    }
    auto cyclotomicOrder = context->GetCyclotomicOrder();
    auto digits = context->EvalFastRotationPrecompute(input);
    vector<Ctext> taps(9);
    for (int j = 0; j < 9; j++) {
        int shift = (j / 3 - 1) * inputWidth + (j % 3 - 1);
        taps[j] = (shift == 0) ? input : context->EvalFastRotation(input, shift, cyclotomicOrder, digits);
    }

    Ctext finalResults = context->EvalAdd(diagonal_channel_sum(taps, kernelData, inputSize), biasInput);
    taps.clear();
    printTiming("Diagonal convolution evaluated...", "Conv", start_time);
    return finalResults;
}

/**
 * @brief Stride-2 downsampling of several ciphertexts with the same layout, sharing the masks.
 *
 * Every input holds inputChannels channels of inputWidth x inputWidth values in consecutive
 * blocks and fills its slots. Each output keeps the values at the even rows and columns,
 * compacted: channel c of width inputWidth / 2 in block c of the first slots / 4 slots, zeros
 * elsewhere. The positions are moved within the rows (log2(outputWidth) levels), then the rows
 * of every channel (1 level), the channels of every group of channelGroup channels (1 level)
 * and the groups (1 more level when channelGroup < inputChannels). The masks are encoded once
 * at the right levels; the outputs are rescaled.
 *
 * @param inputs Ciphertexts to downsample, all at the same level.
 * @param inputWidth Spatial width of one input channel.
 * @param inputChannels Number of input channels.
 * @param channelGroup Channels compacted in the first channel stage (0 = default).
 * @return The downsampled ciphertexts, in the order of the inputs.
 */
vector<Ctext> FHEONANNDiagonalController::downsample_channels(const vector<Ctext>& inputs, int inputWidth, int inputChannels, int channelGroup){
    constexpr int stride = 2;
    int inputSize = inputWidth * inputWidth;
    int outputWidth = inputWidth / stride;
    int outputSize = outputWidth * outputWidth;
    int inputSlots = inputs[0]->GetSlots();
    if (inputSlots != inputChannels * inputSize) {
        throw std::invalid_argument("downsample_channels: the ciphertext slots must equal inputChannels * inputWidth^2");
    }
    Ctext input = rescale_pending(inputs[0]);
    int level = input->GetLevel();
    int columnSteps = static_cast<int>(log2(outputWidth));
    if (channelGroup <= 0) {
        channelGroup = diagonal_channel_group(inputChannels);
    }
    vector<double> values(inputSlots);
    auto encode_mask = [&](int maskLevel, const std::function<bool(int, int, int)>& selected) {
        for (int ch = 0; ch < inputChannels; ch++) {
            for (int y = 0; y < inputWidth; y++) {
                for (int x = 0; x < inputWidth; x++) {
                    values[ch * inputSize + y * inputWidth + x] = selected(ch, y, x) ? 1.0 : 0.0;
                }
            }
        }
        return context->MakeCKKSPackedPlaintext(values, 1, maskLevel, nullptr, inputSlots);
    };
    Ptext firstMask = encode_mask(level++, [&](int, int y, int x) { return y % stride == 0 && x % stride == 0; });
    vector<Ptext> columnMasks;
    for (int s = 1; s < columnSteps; s++) {
        int pattern = 1 << s;
        columnMasks.push_back(encode_mask(level++, [&](int, int, int x) { return x % (2 * pattern) < pattern; }));
    }
    vector<Ptext> rowMasks, groupMasks, channelMasks;
    for (int row = 0; row < outputWidth; row++) {
        rowMasks.push_back(encode_mask(level, [&](int, int y, int x) { int pos = y * inputWidth + x; return pos >= row * outputWidth && pos < (row + 1) * outputWidth; }));
    }
    level++;
    /** group masks select [r * outputSize, (r + 1) * outputSize) counted from the first channel of each group */
    for (int r = 0; r < channelGroup; r++) {
        std::fill(values.begin(), values.end(), 0.0);
        for (int g = 0; g < inputChannels / channelGroup; g++) {
            int start = g * channelGroup * inputSize + r * outputSize;
            std::fill(values.begin() + start, values.begin() + start + outputSize, 1.0);
        }
        groupMasks.push_back(context->MakeCKKSPackedPlaintext(values, 1, level, nullptr, inputSlots));
    }
    if (channelGroup < inputChannels) {
        level++;
        for (int g = 0; g < inputChannels / channelGroup; g++) {
            std::fill(values.begin(), values.end(), 0.0);
            int start = g * channelGroup * outputSize;
            std::fill(values.begin() + start, values.begin() + start + channelGroup * outputSize, 1.0);
            channelMasks.push_back(context->MakeCKKSPackedPlaintext(values, 1, level, nullptr, inputSlots));
        }
    }

    vector<Ctext> outputs;
    for (const auto &source : inputs) {
        /** keep the stride positions and juxtapose them inside every row */
        Ctext phase = rescale_pending(context->EvalMult(rescale_pending(source), firstMask));
        for (int s = 1; s < columnSteps; s++) {
            phase = context->EvalAdd(phase, context->EvalRotate(phase, 1 << (s - 1)));
            phase = rescale_pending(context->EvalMult(phase, columnMasks[s - 1]));
        }
        phase = context->EvalAdd(phase, context->EvalRotate(phase, outputWidth / 2));
        /** juxtapose the rows of every channel, the channels of every group, then the groups */
        phase = masked_rotation_sum(phase, rowMasks, stride * inputWidth - outputWidth);
        phase = masked_rotation_sum(phase, groupMasks, inputSize - outputSize);
        if (channelGroup < inputChannels) {
            phase = masked_rotation_sum(phase, channelMasks, channelGroup * (inputSize - outputSize));
        }
        outputs.push_back(phase);
    }
    return outputs;
}

/**
 * @brief 2x2 average pooling (stride 2) on the diagonal layout, with the output repacked for the next layer.
 *
 * The window sums take 3 hoisted rotations and no level; downsample_channels then compacts the
 * even positions. The 1/4 of the average is expected in the weights of the next layer. The
 * output (inputChannels channels of width inputWidth / 2 in consecutive blocks, zeros after
 * them) is replicated and returned with outputSlots slots, so that a following convolution with
 * more output channels can use it directly. outputSlots must be at least slots / 4.
 *
 * @param encryptedInput Input ciphertext, slots = inputChannels * inputWidth^2.
 * @param inputWidth Spatial width of one input channel.
 * @param inputChannels Number of channels.
 * @param outputSlots Slot count of the returned ciphertext.
 * @param channelGroup Channels compacted in the first channel stage (0 = default, see downsample_channels).
 * @return Ciphertext holding the 4x window sums.
 */
Ctext FHEONANNDiagonalController::he_avgpool_diagonal(const Ctext& encryptedInput, int inputWidth, int inputChannels, int outputSlots, int channelGroup){
    auto start_time = startTime();
    int inputSlots = encryptedInput->GetSlots();
    if (outputSlots < inputSlots / 4 || outputSlots > inputSlots) {
        throw std::invalid_argument("he_avgpool_diagonal: outputSlots must be between slots / 4 and slots");
    }
    Ctext input = rescale_pending(encryptedInput);
    auto cyclotomicOrder = context->GetCyclotomicOrder();
    auto digits = context->EvalFastRotationPrecompute(input);
    Ctext sums = context->EvalAdd(input, context->EvalFastRotation(input, 1, cyclotomicOrder, digits));
    context->EvalAddInPlace(sums, context->EvalFastRotation(input, inputWidth, cyclotomicOrder, digits));
    context->EvalAddInPlace(sums, context->EvalFastRotation(input, inputWidth + 1, cyclotomicOrder, digits));

    Ctext result = downsample_channels({sums}, inputWidth, inputChannels, channelGroup)[0];
    for (int step = outputSlots; step < inputSlots; step <<= 1) {
        result = context->EvalAdd(result, context->EvalRotate(result, -step));
    }
    result->SetSlots(outputSlots);
    printTiming("Diagonal average pooling evaluated...", "AvgPool", start_time);
    return result;
}

/**
 * @brief Rotation positions needed by he_avgpool_diagonal.
 */
vector<int> FHEONANNDiagonalController::generate_avgpool_diagonal_rotation_positions(int inputWidth, int inputChannels, int outputSlots, int channelGroup){
    int inputSize = inputWidth * inputWidth;
    int outputWidth = inputWidth / 2;
    int outputSize = outputWidth * outputWidth;
    int inputSlots = inputChannels * inputSize;
    if (channelGroup <= 0) {
        channelGroup = diagonal_channel_group(inputChannels);
    }
    vector<int> keys_position = {1, inputWidth, inputWidth + 1, 2 * inputWidth - outputWidth, inputSize - outputSize, channelGroup * (inputSize - outputSize)};
    for (int step = 1; step < outputWidth; step <<= 1) {
        keys_position.push_back(step);
    }
    for (int step = outputSlots; step < inputSlots; step <<= 1) {
        keys_position.push_back(-step);
    }
    return keys_position;
}

/**
 * @brief 3x3 stride-2 convolution and 1x1 stride-2 shortcut evaluated over channel diagonals.
 *
 * The input is downsampled once per stride phase (4 multi-channel downsamples instead of two
 * per output channel); the 9 stride-2 taps are rotations of those phases, and the channels are
 * then mixed as in he_convolution_diagonal. The input slot count must be
 * inputChannels * inputWidth^2. The outputs hold 2 * inputChannels channels of width
 * inputWidth / 2 and are returned with half the slots of the input, properly repacked, so no
 * further slot reduction is needed.
 *
 * @param encryptedInput Input ciphertext.
 * @param kernelData [diagonal][tap] plaintexts from encode_kernel_diagonal (strided = true).
 * @param shortcutKernelData Plaintexts from encode_shortcut_diagonal.
 * @param biasInput Convolution bias from encode_bias_diagonal.
 * @param shortcutBiasInput Shortcut bias from encode_bias_diagonal.
 * @param inputWidth Spatial width of one input channel.
 * @param inputChannels Number of input channels.
 * @param channelGroup Channels compacted together in the first channel stage of the downsampling
 *        (0 = default). Passing inputChannels compacts the channels in a single stage, which
 *        uses one level less and more rotations; the kernels must then be encoded at
 *        diagonal_kernel_level(..., channelGroup) and the keys generated with the same value.
 * @return Vector containing [convolution_output, shortcut_output].
 */
vector<Ctext> FHEONANNDiagonalController::he_convolution_and_shortcut_diagonal(const Ctext& encryptedInput, const vector<vector<Ptext>>& kernelData,
                const vector<Ptext>& shortcutKernelData, Ptext& biasInput, Ptext& shortcutBiasInput, int inputWidth, int inputChannels, int channelGroup){
    auto start_time = startTime();
    constexpr int stride = 2;
    int inputSize = inputWidth * inputWidth;
    int outputWidth = inputWidth / stride;
    int outputSize = outputWidth * outputWidth;
    int inputSlots = encryptedInput->GetSlots();
    int outputSlots = inputSlots / 2;
    if (inputSlots != inputChannels * inputSize) {
        throw std::invalid_argument("he_convolution_and_shortcut_diagonal: the ciphertext slots must equal inputChannels * inputWidth^2");
    }

    /*** STEP 1: one downsample per stride phase, phase[2*a+b](y, x) = input(2y+a, 2x+b) ***/
    Ctext input = rescale_pending(encryptedInput);
    auto cyclotomicOrder = context->GetCyclotomicOrder();
    auto digits = context->EvalFastRotationPrecompute(input);
    vector<Ctext> phases(4);
    for (int a = 0; a < 2; a++) {
        for (int b = 0; b < 2; b++) {
            int shift = a * inputWidth + b;
            phases[2 * a + b] = (shift == 0) ? input : context->EvalFastRotation(input, shift, cyclotomicOrder, digits);
        }
    }
    phases = downsample_channels(phases, inputWidth, inputChannels, channelGroup);
    /*** STEP 2: replicate the inputSlots/4 downsampled values and halve the slot count ***/
    for (auto &phase : phases) {
        phase = context->EvalAdd(phase, context->EvalRotate(phase, -(inputSlots / 4)));
        phase = context->EvalAdd(phase, context->EvalRotate(phase, -(inputSlots / 2)));
        phase->SetSlots(outputSlots);
    }

    /*** STEP 3: the stride-2 taps input(2y+dy, 2x+dx) are shifts of the phases ***/
    vector<Ctext> taps(9);
    taps[0] = context->EvalRotate(phases[3], -(outputWidth + 1));
    taps[1] = context->EvalRotate(phases[2], -outputWidth);
    taps[2] = context->EvalRotate(phases[3], -outputWidth);
    taps[3] = context->EvalRotate(phases[1], -1);
    taps[4] = phases[0];
    taps[5] = phases[1];
    taps[6] = context->EvalRotate(phases[3], -1);
    taps[7] = phases[2];
    taps[8] = phases[3];

    /*** STEP 4: mix the channels ***/
    vector<vector<Ptext>> shortcutDiagonals(shortcutKernelData.size());
    for (size_t d = 0; d < shortcutKernelData.size(); d++) {
        shortcutDiagonals[d] = {shortcutKernelData[d]};
    }
    vector<Ctext> shortcutTaps = {phases[0]};
    Ctext finalMainResult = context->EvalAdd(diagonal_channel_sum(taps, kernelData, outputSize), biasInput);
    Ctext finalShortcutResult = context->EvalAdd(diagonal_channel_sum(shortcutTaps, shortcutDiagonals, outputSize), shortcutBiasInput);

    taps.clear();
    phases.clear();
    printTiming("Diagonal convolution and shortcut evaluated...", "Conv", start_time);
    return {finalMainResult, finalShortcutResult};
}

/**
 * @brief Global average pooling followed by a linear layer, in one multiplicative level.
 *
 * The channel sums are computed with log2(inputWidth^2) rotations, the averaging factor is
 * part of the weights (FHEONHEController::encode_avgpool_linear), and the per-channel products
 * are folded with log2(inputChannels) rotations. Output k is in slot k.
 * The input slot count must be inputChannels * inputWidth^2.
 *
 * @param encryptedInput Input ciphertext.
 * @param weightMatrix Plaintexts from encode_avgpool_linear.
 * @param biasInput Bias plaintext with output k in slot k.
 * @param inputWidth Spatial width of one input channel.
 * @param inputChannels Number of input channels.
 * @return Encrypted outputs of the linear layer.
 */
Ctext FHEONANNDiagonalController::he_globalavgpool_linear(const Ctext& encryptedInput, const vector<Ptext>& weightMatrix, Ptext& biasInput, int inputWidth, int inputChannels){
    auto start_time = startTime();
    int inputSize = inputWidth * inputWidth;
    if (static_cast<int>(encryptedInput->GetSlots()) != inputChannels * inputSize) {
        throw std::invalid_argument("he_globalavgpool_linear: the ciphertext slots must equal inputChannels * inputWidth^2");
    }

    /*** slot c * inputSize receives the sum of channel c ***/
    Ctext channelSums = rescale_pending(encryptedInput);
    for (int step = 1; step < inputSize; step <<= 1) {
        channelSums = context->EvalAdd(channelSums, context->EvalRotate(channelSums, step));
    }

    /*** output k collects weight[k][c] * sum[c] at slot c * inputSize + k ***/
    auto cyclotomicOrder = context->GetCyclotomicOrder();
    auto digits = context->EvalFastRotationPrecompute(channelSums);
    Ctext result = context->EvalMult(channelSums, weightMatrix[0]);
    for (size_t k = 1; k < weightMatrix.size(); k++) {
        Ctext shifted = context->EvalFastRotation(channelSums, -static_cast<int>(k), cyclotomicOrder, digits);
        context->EvalAddInPlace(result, context->EvalMult(shifted, weightMatrix[k]));
    }

    /*** fold the channel blocks ***/
    for (int step = 1; step < inputChannels; step <<= 1) {
        result = context->EvalAdd(result, context->EvalRotate(result, step * inputSize));
    }
    result = context->EvalAdd(result, biasInput);
    printTiming("Global average pooling and linear evaluated...", "FC", start_time);
    return result;
}

/**
 * @brief Rotation positions needed by he_convolution_layout.
 */
vector<int> FHEONANNDiagonalController::generate_convolution_layout_rotation_positions(int kernelWidth, int rowStride, int colStride, int channelStride,
                            int numGroups, const vector<int>& subShifts){
    vector<int> keys_position = {channelStride};
    for (int j = 1; j < kernelWidth * kernelWidth; j++) {
        keys_position.push_back((j / kernelWidth) * rowStride + (j % kernelWidth) * colStride);
    }
    for (int subShift : subShifts) {
        keys_position.push_back(-(subShift + channelStride * (numGroups - 1)));
    }
    return keys_position;
}

/**
 * @brief Rotation positions needed by he_sumpool_layout.
 */
vector<int> FHEONANNDiagonalController::generate_sumpool_layout_rotation_positions(int rowStride, int colStride){
    return {colStride, rowStride, rowStride + colStride};
}

/**
 * @brief Rotation positions needed by he_linear_diagonal.
 *
 * @param numSlots Slot count of the input ciphertext.
 * @param numDiagonals Number of diagonals of the layer.
 * @param babySteps Number of baby steps the diagonals were encoded with.
 */
vector<int> FHEONANNDiagonalController::generate_linear_diagonal_rotation_positions(int numSlots, int numDiagonals, int babySteps){
    vector<int> keys_position;
    for (int b = 1; b <= babySteps; b++) {
        keys_position.push_back(b);
    }
    for (int step = numDiagonals; step < numSlots; step <<= 1) {
        keys_position.push_back(step);
    }
    return keys_position;
}

/**
 * @brief Stride-1 "valid" convolution on a strided layout, without compacting the output.
 *
 * See FHEONHEController::encode_kernel_layout for the layout. The cost is k^2 - 1 hoisted
 * rotations for the taps, one rotation per diagonal and per sub shift, and one multiplicative
 * level: the output keeps the row and column strides of the input, so no row has to be moved,
 * and a following pooling or fully connected layer reads it where it is.
 *
 * @param encryptedInput Input ciphertext.
 * @param kernelData [sub shift][diagonal][tap] plaintexts from encode_kernel_layout.
 * @param biasInput Bias plaintext from encode_bias_layout.
 * @param kernelWidth Width of the square kernel.
 * @param rowStride Slots between two rows.
 * @param colStride Slots between two columns.
 * @param channelStride Slots between two input channels, and between two output groups.
 * @param numGroups Number of output groups.
 * @param subShifts Slot offset of each channel inside its group.
 * @return Encrypted convolution output.
 */
Ctext FHEONANNDiagonalController::he_convolution_layout(const Ctext& encryptedInput, const vector<vector<vector<Ptext>>>& kernelData, Ptext& biasInput,
                            int kernelWidth, int rowStride, int colStride, int channelStride, int numGroups, const vector<int>& subShifts){
    auto start_time = startTime();

    Ctext input = rescale_pending(encryptedInput);
    auto cyclotomicOrder = context->GetCyclotomicOrder();
    auto digits = context->EvalFastRotationPrecompute(input);
    vector<Ctext> taps(kernelWidth * kernelWidth);
    for (int j = 0; j < kernelWidth * kernelWidth; j++) {
        int shift = (j / kernelWidth) * rowStride + (j % kernelWidth) * colStride;
        taps[j] = (shift == 0) ? input : context->EvalFastRotation(input, shift, cyclotomicOrder, digits);
    }

    Ctext finalResults;
    for (size_t e = 0; e < subShifts.size(); e++) {
        Ctext subResult = diagonal_channel_sum(taps, kernelData[e], channelStride);
        subResult = context->EvalRotate(subResult, -(subShifts[e] + channelStride * (numGroups - 1)));
        if (finalResults) {
            context->EvalAddInPlace(finalResults, subResult);
        } else {
            finalResults = subResult;
        }
    }
    finalResults = context->EvalAdd(finalResults, biasInput);
    taps.clear();
    printTiming("Layout convolution evaluated...", "Conv", start_time);
    return finalResults;
}

/**
 * @brief 2x2 sum pooling on a strided layout, without scaling or compacting.
 *
 * The sum of each 2x2 window is left at the position of its top-left value, so the pooled
 * values keep twice the row and column strides of the input. It costs 3 hoisted rotations and
 * no level; the 1/4 of the average is expected in the weights of the next layer.
 *
 * @param encryptedInput Input ciphertext.
 * @param rowStride Slots between two rows of the input.
 * @param colStride Slots between two columns of the input.
 * @return Ciphertext holding the window sums.
 */
Ctext FHEONANNDiagonalController::he_sumpool_layout(const Ctext& encryptedInput, int rowStride, int colStride){
    auto start_time = startTime();
    auto cyclotomicOrder = context->GetCyclotomicOrder();
    auto digits = context->EvalFastRotationPrecompute(encryptedInput);
    Ctext result = context->EvalAdd(encryptedInput, context->EvalFastRotation(encryptedInput, colStride, cyclotomicOrder, digits));
    context->EvalAddInPlace(result, context->EvalFastRotation(encryptedInput, rowStride, cyclotomicOrder, digits));
    context->EvalAddInPlace(result, context->EvalFastRotation(encryptedInput, rowStride + colStride, cyclotomicOrder, digits));
    printTiming("Layout sum pooling evaluated...", "AvgPool", start_time);
    return result;
}

/**
 * @brief Fully connected layer evaluated over diagonals with baby-step giant-step rotations.
 *
 * See FHEONHEController::encode_linear_diagonal. The inputs can be anywhere in the slots.
 * The cost is numDiagonals plaintext multiplications, babySteps - 1 hoisted rotations,
 * giantSteps - 1 rotations, log2(slots / numDiagonals) rotations to fold the partial sums,
 * and one multiplicative level. Output k is in slot k and the result is returned with
 * numDiagonals slots, so the following operations (and bootstrapping) work on a small
 * ciphertext.
 *
 * @param encryptedInput Input ciphertext.
 * @param weightDiagonals [giant step][baby step] plaintexts from encode_linear_diagonal.
 * @param biasInput Bias plaintext encoded with numDiagonals slots, output k in slot k.
 * @param numDiagonals Number of diagonals the layer was encoded with.
 * @return Encrypted outputs, with numDiagonals slots.
 */
Ctext FHEONANNDiagonalController::he_linear_diagonal(const Ctext& encryptedInput, const vector<vector<Ptext>>& weightDiagonals, Ptext& biasInput, int numDiagonals){
    auto start_time = startTime();
    int numSlots = encryptedInput->GetSlots();
    int babySteps = weightDiagonals[0].size();

    Ctext input = rescale_pending(encryptedInput);
    auto cyclotomicOrder = context->GetCyclotomicOrder();
    auto digits = context->EvalFastRotationPrecompute(input);
    vector<Ctext> taps(babySteps);
    taps[0] = input;
    for (int b = 1; b < babySteps; b++) {
        taps[b] = context->EvalFastRotation(input, b, cyclotomicOrder, digits);
    }

    Ctext result = diagonal_channel_sum(taps, weightDiagonals, babySteps);
    /*** fold the numSlots / numDiagonals partial sums: the result repeats every numDiagonals slots ***/
    for (int step = numDiagonals; step < numSlots; step <<= 1) {
        result = context->EvalAdd(result, context->EvalRotate(result, step));
    }
    result->SetSlots(numDiagonals);
    result = context->EvalAdd(result, biasInput);
    taps.clear();
    printTiming("Diagonal linear evaluated...", "FC", start_time);
    return result;
}

/**
 * @brief Homomorphic ReLU that scales the input with a scalar instead of an encoded mask.
 *
 * Same approximation and depth as he_relu; the 1/scale factor is applied with a scalar
 * multiplication, which avoids encoding a full-chain plaintext for every activation.
 *
 * @param encryptedInput Ciphertext to activate.
 * @param scaleValue Scaling factor.
 * @param polyDegree Degree of approximating polynomial.
 * @return Activated ciphertext.
 */
Ctext FHEONANNDiagonalController::he_relu_scalar(const Ctext& encryptedInput, double scaleValue, int polyDegree) {
    auto start_time = startTime();
    double lowerBound = -1;
    double upperBound = 1;

    Ctext encryptInn = encryptedInput;
    if(scaleValue > 1){
        encryptInn = context->EvalMult(encryptedInput, 1.0 / scaleValue);
    }
    else{
        scaleValue = 1;
    }

    Ctext relu_result = context->EvalChebyshevFunction(
        [scaleValue](double x) -> double { if (x < 0) return 0; else return scaleValue*x; },
                                            encryptInn,
                                            lowerBound,
                                            upperBound,
                                            polyDegree);
    printTiming("ReLU evaluated...", "ReLU", start_time);
    return relu_result;
}

/**
 * @brief Homomorphic ReLU of the input divided by a scale: returns ReLU(x / scale), within [0, 1].
 *
 * Unlike he_relu and he_relu_scalar, the output is not multiplied back by the scale (and the
 * scale can be below 1): the next layer is expected to multiply its weights by the scale. The
 * activations then stay within [0, 1], where the approximation and the bootstrapping are most
 * accurate. The depth is the same as he_relu_scalar.
 *
 * @param encryptedInput Ciphertext to activate.
 * @param scale Value with |x| <= scale for every slot.
 * @param polyDegree Degree of approximating polynomial.
 * @return Ciphertext of ReLU(x / scale).
 */
Ctext FHEONANNDiagonalController::he_relu_normalized(const Ctext& encryptedInput, double scale, int polyDegree) {
    auto start_time = startTime();
    Ctext encryptInn = context->EvalMult(encryptedInput, 1.0 / scale);
    Ctext relu_result = context->EvalChebyshevFunction(
        [](double x) -> double { return (x < 0) ? 0 : x; }, encryptInn, -1, 1, polyDegree);
    printTiming("ReLU evaluated...", "ReLU", start_time);
    return relu_result;
}




/****************************************************************************************************************
 * Shared interface: the declarations of FHEONANNStandardController, evaluated with the diagonal primitives
 ****************************************************************************************************************/

int FHEONANNDiagonalController::linear_baby_steps(int numDiagonals){
    int babySteps = 1;
    while (babySteps * babySteps < numDiagonals) babySteps <<= 1;
    return babySteps;
}

/**
 * @brief Strided-layout arguments of a "valid" convolution with the defaults filled in: rows of inputWidth slots,
 *        consecutive columns, output channels outputWidth * rowStride slots apart, one channel per group.
 */
FHEONLayout FHEONANNDiagonalController::convolution_layout(const FHEONLayout& layout, int inputWidth, int outputChannels, int kernelWidth){
    FHEONLayout filled = layout;
    int outputWidth = inputWidth - kernelWidth + 1;
    if (filled.rowStride <= 0) filled.rowStride = inputWidth;
    if (filled.channelStride <= 0) filled.channelStride = outputWidth * filled.rowStride;
    if (filled.subShifts.empty()) filled.subShifts = {0};
    if (filled.numGroups <= 0) filled.numGroups = outputChannels / filled.subShifts.size();
    return filled;
}


/**
 * @brief Keys of he_linear / he_linear_optimized for maxFCLayeroutputs outputs on a layout.inputSlots-slot input
 *        (num_slots when 0); rotationPosition is unused.
 */
vector<int> FHEONANNDiagonalController::generate_linear_rotation_positions(int maxFCLayeroutputs, int rotationPosition, const FHEONLayout& layout){
    int numSlots = (layout.inputSlots > 0) ? layout.inputSlots : num_slots;
    int numDiagonals = nextPowerOf2(maxFCLayeroutputs);
    return generate_linear_diagonal_rotation_positions(numSlots, numDiagonals, linear_baby_steps(numDiagonals));
}

/************************************************* convolutions *************************************************/

/**
 * @brief 3x3 stride-2 convolution and 1x1 stride-2 shortcut (see he_convolution_and_shortcut_diagonal); half the slots out.
 */
vector<Ctext> FHEONANNDiagonalController::he_convolution_and_shortcut_optimized(const Ctext& encryptedInput, const vector<vector<Ptext>>& kernelData,
                            const vector<Ptext>& shortcutKernelData, Ptext& biasVector, Ptext& shortcutBiasVector,
                            int inputWidth, int inputChannels, int outputChannels, const FHEONLayout& layout){
    return he_convolution_and_shortcut_diagonal(encryptedInput, kernelData, shortcutKernelData, biasVector, shortcutBiasVector,
                                                inputWidth, inputChannels, layout.channelGroup);
}

/************************************************* poolings *************************************************/

/**
 * @brief Global average pooling: the average of channel c in slot c (two levels). kernelWidth and rotatePositions are unused.
 */
Ctext FHEONANNDiagonalController::he_globalavgpool(Ctext& encryptedInput, int inputWidth, int outputChannels, int kernelWidth, int rotatePositions,
                            const FHEONLayout& layout){
    auto start_time = startTime();
    int inputSize = inputWidth * inputWidth;
    int numSlots = encryptedInput->GetSlots();
    Ctext sums = rescale_pending(encryptedInput);
    for (int step = 1; step < inputSize; step <<= 1) {
        sums = context->EvalAdd(sums, context->EvalRotate(sums, step));
    }
    /** keep the sum of every channel (slot c * inputSize), divided by the channel size, then gather slot c * inputSize into slot c */
    int level = sums->GetLevel();
    vector<double> values(numSlots, 0.0);
    for (int ch = 0; ch < outputChannels; ch++) values[ch * inputSize] = 1.0 / inputSize;
    Ctext averages = rescale_pending(context->EvalMult(sums, context->MakeCKKSPackedPlaintext(values, 1, level, nullptr, numSlots)));
    vector<Ptext> gatherMasks;
    for (int ch = 0; ch < outputChannels; ch++) {
        std::fill(values.begin(), values.end(), 0.0);
        values[ch] = 1.0;
        gatherMasks.push_back(context->MakeCKKSPackedPlaintext(values, 1, level + 1, nullptr, numSlots));
    }
    Ctext result = masked_rotation_sum(averages, gatherMasks, inputSize - 1);
    printTiming("Diagonal global average pooling evaluated...", "AvgPool", start_time);
    return result;
}

/************************************************* linear and activation *************************************************/

/**
 * @brief ReLU: scale * ReLU(x / scale) with a scalar scaling, or ReLU(x / scale) within [0, 1] with normalizedOutput.
 *        vectorSize is unused.
 */
Ctext FHEONANNDiagonalController::he_relu(Ctext& encryptedInput, double scale, int vectorSize, int polyDegree, bool normalizedOutput){
    return normalizedOutput ? he_relu_normalized(encryptedInput, scale, polyDegree) : he_relu_scalar(encryptedInput, scale, polyDegree);
}

Ctext FHEONANNDiagonalController::he_sum_two_ciphertexts(Ctext& firstInput, Ctext& secondInput){
    return FHEONANNStandardController::he_sum_two_ciphertexts(firstInput, secondInput);
}

/************************************************* rotation keys (optimized set) *************************************************/

/**
 * @brief Keys of he_convolution and he_convolution_and_shortcut_optimized. padding 0: strided "valid" layout. Otherwise
 *        channel-diagonal convolution (stride 1 or 2, layout.channelGroup for stride 2); a stride-1 layer with fewer input
 *        than output channels also gets the rotations replicating its input (on layout.inputSlots slots,
 *        outputChannels * inputWidth^2 when 0).
 */
vector<int> FHEONANNDiagonalController::generate_convolution_rotation_positions(int inputWidth, int inputChannels, int outputChannels,
                                                    int kernelWidth, int padding, int stride, const FHEONLayout& layout){
    if (padding == 0) {
        FHEONLayout filled = convolution_layout(layout, inputWidth, outputChannels, kernelWidth);
        return generate_convolution_layout_rotation_positions(kernelWidth, filled.rowStride, filled.colStride, filled.channelStride,
                                                              filled.numGroups, filled.subShifts);
    }
    vector<int> keys = generate_diagonal_convolution_rotation_positions(inputWidth, inputChannels, stride, layout.channelGroup);
    if (stride == 1 && inputChannels < outputChannels) {
        int numSlots = (layout.inputSlots > 0) ? layout.inputSlots : outputChannels * inputWidth * inputWidth;
        for (int step = nextPowerOf2(inputChannels) * inputWidth * inputWidth; step < numSlots; step <<= 1) {
            keys.push_back(-step);
        }
    }
    return keys;
}

/**
 * @brief Keys of he_avgpool (strided layout when layout.rowStride is set, compacted onto layout.outputSlots slots
 *        otherwise) or, when kernelWidth == inputWidth, of he_globalavgpool.
 */
vector<int> FHEONANNDiagonalController::generate_avgpool_rotation_positions(int inputWidth, int kernelWidth, int stride, int inputChannels, const FHEONLayout& layout){
    int inputSize = inputWidth * inputWidth;
    if (layout.rowStride > 0) {
        return generate_sumpool_layout_rotation_positions(layout.rowStride, layout.colStride);
    }
    if (kernelWidth == inputWidth) {
        vector<int> keys = {inputSize - 1};
        for (int step = 1; step < inputSize; step <<= 1) keys.push_back(step);
        return keys;
    }
    int outputSlots = (layout.outputSlots > 0) ? layout.outputSlots : inputChannels * inputSize / 4;
    return generate_avgpool_diagonal_rotation_positions(inputWidth, inputChannels, outputSlots, layout.channelGroup);
}

/************************************************* operations (optimized set) *************************************************/

/**
 * @brief Convolution. padding 1: 3x3 "same" stride-1 convolution over channel diagonals (stride 2 goes through
 *        he_convolution_and_shortcut_optimized). padding 0: "valid" kernelWidth x kernelWidth convolution on the strided
 *        layout of layout (see he_convolution_layout; kernel from encode_convolution_kernel with padding 0).
 */
Ctext FHEONANNDiagonalController::he_convolution(Ctext& encryptedInput, vector<vector<Ptext>>& kernelData, Ptext& biasInput,
                            int inputWidth, int inputChannels, int outputChannels, int kernelWidth, int padding, int stride,
                            const FHEONLayout& layout){
    if (stride != 1) {
        throw std::invalid_argument("FHEONANNDiagonalController::he_convolution: stride 1 only (use he_convolution_and_shortcut_optimized for stride 2)");
    }
    if (padding != 0) {
        return he_convolution_diagonal(encryptedInput, kernelData, biasInput, inputWidth);
    }
    FHEONLayout filled = convolution_layout(layout, inputWidth, outputChannels, kernelWidth);
    size_t numSubs = filled.subShifts.size();
    size_t numDiagonals = kernelData.size() / numSubs;
    vector<vector<vector<Ptext>>> layoutKernel(numSubs);
    for (size_t e = 0; e < numSubs; e++) {
        layoutKernel[e].assign(kernelData.begin() + e * numDiagonals, kernelData.begin() + (e + 1) * numDiagonals);
    }
    return he_convolution_layout(encryptedInput, layoutKernel, biasInput, kernelWidth, filled.rowStride, filled.colStride,
                                 filled.channelStride, filled.numGroups, filled.subShifts);
}

/**
 * @brief 2x2 stride-2 pooling returning window sums (avgpool_weight_scale gives the 1/4 for the next weights): left in
 *        place on the strided layout when layout.rowStride is set (rows layout.rowStride slots apart), compacted onto
 *        layout.outputSlots slots (a quarter of the input when 0) otherwise.
 */
Ctext FHEONANNDiagonalController::he_avgpool(Ctext encryptedInput, int imgCols, int outputChannels, int kernelWidth, int stride, const FHEONLayout& layout){
    if (kernelWidth != 2 || stride != 2) {
        throw std::invalid_argument("FHEONANNDiagonalController::he_avgpool: only 2x2 pooling with stride 2 is supported");
    }
    if (layout.rowStride > 0) {
        return he_sumpool_layout(encryptedInput, layout.rowStride, layout.colStride);
    }
    int outputSlots = (layout.outputSlots > 0) ? layout.outputSlots : static_cast<int>(encryptedInput->GetSlots()) / 4;
    return he_avgpool_diagonal(encryptedInput, imgCols, outputChannels, outputSlots, layout.channelGroup);
}

/**
 * @brief Linear layer over the flattened diagonals of encode_linear_weights (see he_linear_diagonal); the output has
 *        weightMatrix.size() slots. rotatePositions is unused.
 */
Ctext FHEONANNDiagonalController::he_linear(Ctext& encryptedInput, vector<Ptext>& weightMatrix, Ptext& biasInput, int inputSize, int outputSize, int rotatePositions,
                            const FHEONLayout& layout){
    int numDiagonals = weightMatrix.size();
    int babySteps = linear_baby_steps(numDiagonals);
    vector<vector<Ptext>> diagonals(numDiagonals / babySteps, vector<Ptext>(babySteps));
    for (int d = 0; d < numDiagonals; d++) {
        diagonals[d / babySteps][d % babySteps] = weightMatrix[d];
    }
    return he_linear_diagonal(encryptedInput, diagonals, biasInput, numDiagonals);
}

/************************************************* weight encoding *************************************************/

int FHEONANNDiagonalController::convolution_kernel_level(const Ctext& encryptedInput, int inputWidth, int stride, const FHEONLayout& layout){
    return diagonal_kernel_level(encryptedInput, inputWidth, stride, layout.channelGroup);
}

/**
 * @brief The poolings return window sums: the next layer multiplies its weights by 1 / kernelWidth^2.
 */
double FHEONANNDiagonalController::avgpool_weight_scale(int kernelWidth){
    return 1.0 / (kernelWidth * kernelWidth);
}

/**
 * @brief Encode a convolution kernel. padding 0: strided "valid" layout of he_convolution (flattened [sub shift][diagonal]).
 *        Otherwise channel diagonals: numSlots is the slot count of the input ciphertext; stride 2 encodes for the half-slot
 *        output of he_convolution_and_shortcut_optimized; a stride-1 layer with fewer input than output channels gets a
 *        replicated input.
 */
vector<vector<Ptext>> FHEONANNDiagonalController::encode_convolution_kernel(FHEONHEController& heController, vector<vector<vector<vector<double>>>>& kernelData,
                            int inputWidth, int numSlots, int encodeLevel, int stride, int padding, const FHEONLayout& layout){
    int inputChannels = kernelData[0].size();
    int outputChannels = kernelData.size();
    if (padding == 0) {
        int kernelWidth = kernelData[0][0].size();
        FHEONLayout filled = convolution_layout(layout, inputWidth, outputChannels, kernelWidth);
        auto layoutKernel = heController.encode_kernel_layout(kernelData, layout.weightScale, numSlots, encodeLevel, filled.channelStride,
                                                              filled.rowStride, filled.colStride, inputWidth - kernelWidth + 1,
                                                              filled.numGroups, filled.subShifts);
        vector<vector<Ptext>> flattened;
        for (auto &sub : layoutKernel) flattened.insert(flattened.end(), sub.begin(), sub.end());
        return flattened;
    }
    auto scaled = kernelData;
    if (layout.weightScale != 1.0) {
        for (auto &outChannel : scaled) for (auto &inChannel : outChannel) for (auto &row : inChannel) for (auto &value : row) { value *= layout.weightScale; }
    }
    if (stride == 2) {
        return heController.encode_kernel_diagonal(scaled, inputWidth / 2, numSlots / 2, encodeLevel, true);
    }
    int inputPeriod = (inputChannels < outputChannels) ? nextPowerOf2(inputChannels) : 0;
    return heController.encode_kernel_diagonal(scaled, inputWidth, numSlots, encodeLevel, false, inputPeriod);
}

/**
 * @brief Encode a convolution bias for an output of numSlots slots (the strided layout when layout.rowStride is set).
 */
Ptext FHEONANNDiagonalController::encode_convolution_bias(FHEONHEController& heController, vector<double>& biasData, int outputWidth, int numSlots, int encodeLevel,
                            const FHEONLayout& layout){
    if (layout.rowStride > 0) {
        int channelStride = (layout.channelStride > 0) ? layout.channelStride : outputWidth * layout.rowStride;
        vector<int> subShifts = layout.subShifts.empty() ? vector<int>{0} : layout.subShifts;
        int numGroups = (layout.numGroups > 0) ? layout.numGroups : biasData.size() / subShifts.size();
        return heController.encode_bias_layout(biasData, numSlots, encodeLevel, channelStride, layout.rowStride, layout.colStride,
                                               outputWidth, numGroups, subShifts);
    }
    return heController.encode_bias_diagonal(biasData, outputWidth, numSlots, encodeLevel);
}

/**
 * @brief Encode the 1x1 stride-2 shortcut kernel as diagonals; numSlots is the slot count of the input ciphertext.
 */
vector<Ptext> FHEONANNDiagonalController::encode_shortcut_kernel(FHEONHEController& heController, vector<vector<double>>& kernelData, int inputWidth, int numSlots, int encodeLevel,
                            const FHEONLayout& layout){
    auto scaled = kernelData;
    for (auto &row : scaled) for (auto &value : row) { value *= layout.weightScale; }
    return heController.encode_shortcut_diagonal(scaled, inputWidth / 2, numSlots / 2, encodeLevel);
}

/**
 * @brief Encode a linear layer as its diagonals, flattened for he_linear_optimized: input i in the slots
 *        layout.inputPositions[i] (slot i when empty) of a numSlots-slot ciphertext, weights times layout.weightScale.
 */
vector<Ptext> FHEONANNDiagonalController::encode_linear_weights(FHEONHEController& heController, vector<vector<double>>& weightData, int numSlots, int encodeLevel,
                            const FHEONLayout& layout){
    int numDiagonals = nextPowerOf2(weightData.size());
    vector<vector<int>> inputPositions = layout.inputPositions;
    if (inputPositions.empty()) {
        inputPositions.resize(weightData[0].size());
        for (size_t i = 0; i < inputPositions.size(); i++) { inputPositions[i] = {static_cast<int>(i)}; }
    }
    auto diagonals = heController.encode_linear_diagonal(weightData, inputPositions, layout.weightScale, numSlots, numDiagonals,
                                                         linear_baby_steps(numDiagonals), encodeLevel);
    vector<Ptext> flattened;
    for (auto &giant : diagonals) for (auto &plaintext : giant) { flattened.push_back(plaintext); }
    return flattened;
}

/**
 * @brief Encode the bias of he_linear_optimized (output k in slot k of the output ciphertext).
 */
Ptext FHEONANNDiagonalController::encode_linear_bias(FHEONHEController& heController, vector<double>& biasData, int numSlots, int encodeLevel, const FHEONLayout& layout){
    return heController.encode_input(biasData, nextPowerOf2(biasData.size()), encodeLevel);
}
