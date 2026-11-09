
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
#include <thread>
#include "FHEONANNStandardController.h"

namespace fs = std::filesystem;

/**
 * @brief Set the global crypto context.
 *
 * @param in_context The crypto context to set.
 */
void FHEONANNStandardController::setContext(CryptoContext<DCRTPoly>& in_context){
    context = in_context;
}

/** rotations of downsample(inputWidth, stride): the binary row juxtaposition (same expressions as downsample, so
 *  outputWidth/2 for a width that is not a power of two) and the row shift. downsample_with_multiple_channels adds the
 *  channel shift inputSize - outputSize. */
static void add_downsample_rotation_positions(vector<int>& keys_position, int inputWidth, int stride, int numChannels = 1) {
    int width_out = inputWidth/stride;
    for (int s = 1; s < log2(width_out); s++) {
        keys_position.push_back(pow(2, s-1));
    }
    keys_position.push_back(pow(2, (log2(width_out)-1)));
    if (width_out > 1) {
        keys_position.push_back(stride*inputWidth - width_out);
    }
    if (numChannels > 1) {
        keys_position.push_back(inputWidth*inputWidth - width_out*width_out);
    }
}

/** rotations of EvalMerge over count ciphertexts: ciphertext i is rotated by -i */
static void add_merge_rotation_positions(vector<int>& keys_position, int count) {
    for (int i = 1; i < count; i++) {
        keys_position.push_back(-i);
    }
}

/** sorted unique rotations without 0 */
static vector<int> unique_rotation_positions(vector<int> keys_position) {
    std::sort(keys_position.begin(), keys_position.end());
    keys_position.erase(std::unique(keys_position.begin(), keys_position.end()), keys_position.end());
    keys_position.erase(std::remove(keys_position.begin(), keys_position.end(), 0), keys_position.end());
    return keys_position;
}

/**
 * @brief Rotation positions of EvalSum(ciphertext, sumSize) as used by he_linear, he_linear_optimized and
 * he_globalavgpool: EvalSum adds the rotations by the powers of two below sumSize. They replace the EvalSum keys of the
 * full batch (EvalSumKeyGen), which are the same automorphisms for all powers of two below the number of slots.
 *
 * @param sumSize Largest number of slots summed by EvalSum.
 * @return List of rotation steps.
 */
vector <int> FHEONANNStandardController::generate_sum_rotation_positions(int sumSize){
    vector<int> keys_position;
    for (int r = 1; r < sumSize; r *= 2) {
        keys_position.push_back(r);
    }
    return keys_position;
}

/**
 * @brief Generate rotation index positions needed for standard convolution.
 *
 * @param inputWidth Dimensions of input channel.
 * @param inputChannels Number of input channels.
 * @param outputChannels Number of output channels.
 * @param kernelWidth Dimension of square kernel.
 * @param padding Size of zero padding.
 * @param stride Convolution stride.
 * @param layout Optional diagonal-layout arguments (FHEONLayout); unused by the standard controller, which only keeps the interface shared with FHEONANNDiagonalController.
 * @return List of unique rotation steps.
 */
vector <int> FHEONANNStandardController::generate_convolution_rotation_positions(int inputWidth, int inputChannels, int outputChannels,
                                                    int kernelWidth, int padding, int stride, const FHEONLayout& layout){
    /** a padded 3x3 convolution is he_convolution_optimized (see he_convolution) */
    if (padding > 0) {
        return generate_optimized_convolution_rotation_positions(inputWidth, inputChannels, outputChannels, stride);
    }

    /** he_convolution: kernel slices (1 ... kernelWidth-1 and inputWidth), input channel sum (inputSize), rows placed
     *  in Horner form (-width_out) or downsample when strided, output channels placed in Horner form (-width_out_sq) */
    vector<int> keys_position;
    int width_out = ((inputWidth - kernelWidth)/stride)+1;
    for(int i=1; i < kernelWidth;i++){
        keys_position.push_back(i);
    }
    if (kernelWidth > 1) {
        keys_position.push_back(inputWidth);
    }
    if (inputChannels > 1) {
        keys_position.push_back(inputWidth*inputWidth);
    }
    if (stride > 1) {
        add_downsample_rotation_positions(keys_position, inputWidth, stride);
    }
    else if (width_out > 1) {
        keys_position.push_back(inputWidth);
        keys_position.push_back(-width_out);
    }
    if (outputChannels > 1) {
        keys_position.push_back(-width_out*width_out);
    }
    return unique_rotation_positions(keys_position);
}

/**
 * @brief Generate rotation index positions needed for average pooling.
 *
 * @param inputWidth Dimension of input.
 * @param kernelWidth Dimension of pooling window.
 * @param stride Pooling stride.
 * @param inputChannels Number of channels.
 * @param layout Optional diagonal-layout arguments (FHEONLayout); unused by the standard controller, which only keeps the interface shared with FHEONANNDiagonalController.
 * @return List of unique rotation steps.
 */
vector <int> FHEONANNStandardController::generate_avgpool_rotation_positions(int inputWidth, int kernelWidth,
                int stride, int inputChannels, const FHEONLayout& layout){
    /** a kernel as wide as the input is the global pooling of he_globalavgpool */
    if (kernelWidth == inputWidth) {
        return generate_avgpool_optimized_rotation_positions(inputWidth, inputChannels, kernelWidth, stride, true);
    }
    
    /** he_avgpool: kernel slices (1 ... kernelWidth-1 and inputWidth), channel shifts (inputSize), downsample of every
     *  channel and channels placed in Horner form (-outputSize); a width of at most 2 merges the shifted channels */
    vector<int> keys_position;
    int width_avgpool_out = (inputWidth/stride);
    int width_sq = inputWidth*inputWidth;
    for(int i=1; i < kernelWidth;i++){
        keys_position.push_back(i);
    }
    if (kernelWidth > 1) {
        keys_position.push_back(inputWidth);
    }
    if (inputChannels > 1) {
        keys_position.push_back(width_sq);
    }
    if (inputWidth <= 2) {
        add_merge_rotation_positions(keys_position, inputChannels - 1);
        return unique_rotation_positions(keys_position);
    }
    add_downsample_rotation_positions(keys_position, inputWidth, stride);
    if (inputChannels > 1) {
        keys_position.push_back(-width_avgpool_out*width_avgpool_out);
    }
    return unique_rotation_positions(keys_position);
}

/**
 * @brief Generate rotation positions optimized for convolution.
 *
 * @param inputWidth Dimension of input.
 * @param inputChannels Number of input channels.
 * @param outputChannels Number of output channels.
 * @param stride Convolution stride.
 * @param stridingType Type of striding optimization.
 * @param layout Optional diagonal-layout arguments (FHEONLayout); unused by the standard controller, which only keeps the interface shared with FHEONANNDiagonalController.
 * @return List of unique rotation steps.
 */
vector <int> FHEONANNStandardController::generate_optimized_convolution_rotation_positions(
    int inputWidth,  int inputChannels, int outputChannels, int stride, string stridingType, const FHEONLayout& layout){
        
    /** 3x3 kernel slices (-1, 1, -inputWidth, inputWidth) and the input channel sum (inputSize) of
     *  he_convolution_optimized, he_convolution_and_shortcut_optimized and their _with_multiple_channels variants */
    vector<int> keys_position;
    int inputWidth_sq = pow(inputWidth, 2);
    int width_out = (inputWidth/stride);
    int width_out_sq = pow(width_out,2);
    keys_position.push_back(-1);
    keys_position.push_back(1);
    keys_position.push_back(inputWidth);
    keys_position.push_back(-inputWidth);
    if (inputChannels > 1) {
        keys_position.push_back(inputWidth_sq);
    }

    if(stride > 1){
        if(stridingType == "basic"){
            /**** USE THIS BLOCK FOR BASIC STRIDING */
            for(int i=1; i < inputChannels; i++){
                int rot_val = i*width_out;
                keys_position.push_back(-rot_val);

                rot_val = i*width_out_sq;
                keys_position.push_back(-rot_val);
            }
            for(int i =1; i< width_out; i++){
                int rot_val = i*width_out;
                keys_position.push_back(i);
                keys_position.push_back(-rot_val);
            }
        }
        else if(stridingType == "single_channel"){
            /** he_convolution_optimized / he_convolution_and_shortcut_optimized: downsample of every output channel,
             *  output channels placed in Horner form (-width_out_sq) */
            add_downsample_rotation_positions(keys_position, inputWidth, stride);
            if (outputChannels > 1) {
                keys_position.push_back(-width_out_sq);
            }
        }
        else if(stridingType == "multi_channels"){
            /** _with_multiple_channels: the outputs of a group of inputChannels are placed in Horner form (-inputSize),
             *  downsampled together and the groups shifted by -(group * inputChannels * width_out_sq) */
            if (inputChannels > 1) {
                keys_position.push_back(-inputWidth_sq);
            }
            add_downsample_rotation_positions(keys_position, inputWidth, stride, inputChannels);
            for (int group = 1; group < outputChannels / inputChannels; group++) {
                keys_position.push_back(-group * inputChannels * width_out_sq);
            }
        }
    }
    else if (outputChannels > 1) {
        /** output channels placed in Horner form: one key */
        keys_position.push_back(-width_out_sq);
    }
    return unique_rotation_positions(keys_position);
}


/**
 * @brief Generate rotation positions optimized for average pooling.
 *
 * @param inputWidth Dimension of input.
 * @param inputChannels Number of channels.
 * @param kernelWidth Dimension of pooling window.
 * @param stride Pooling stride.
 * @param globalPooling Whether it's global average pooling.
 * @param stridingType Type of striding optimization.
 * @param rotationIndex Base rotation index.
 * @param layout Optional diagonal-layout arguments (FHEONLayout); unused by the standard controller, which only keeps the interface shared with FHEONANNDiagonalController.
 * @return List of unique rotation steps.
 */
vector <int> FHEONANNStandardController::generate_avgpool_optimized_rotation_positions(int inputWidth,  int inputChannels, 
                                        int kernelWidth, int stride, bool globalPooling, string stridingType, int rotationIndex, const FHEONLayout& layout){
    
    vector<int> keys_position;
    int width_sq = pow(inputWidth, 2);
    if(globalPooling){
        /** he_globalavgpool: channel shifts (inputSize), EvalSum over a channel, merges of up to rotationIndex sums
         *  (-1 ... -(rotationIndex - 1)) and the Horner placement of the merged groups (-rotationIndex) */
        if (inputChannels > 1) {
            keys_position.push_back(width_sq);
        }
        auto sum_keys = generate_sum_rotation_positions(width_sq);
        keys_position.insert(keys_position.end(), sum_keys.begin(), sum_keys.end());
        add_merge_rotation_positions(keys_position, std::min(rotationIndex, inputChannels));
        if (inputChannels > rotationIndex) {
            keys_position.push_back(-rotationIndex);
        }
        return unique_rotation_positions(keys_position);
    }

    /** he_avgpool_optimzed / he_avgpool_optimzed_with_multiple_channels: the 2x2 window (1, inputWidth, inputWidth + 1
     *  as inputWidth then 1); a width of at most 2 merges the shifted channels */
    int width_avgpool_out = (inputWidth/stride);
    int width_avgpool_sq = pow(width_avgpool_out, 2);
    keys_position.push_back(1);
    keys_position.push_back(inputWidth);

    if(inputWidth<=2){
        if (inputChannels > 1) {
            keys_position.push_back(width_sq);
        }
        add_merge_rotation_positions(keys_position, inputChannels - 1);
        return unique_rotation_positions(keys_position);
    }

    if(stride>1){
        if(stridingType == "basic"){
            /**** USE THIS BLOCK FOR BASIC STRIDING */
            for(int i=1; i < inputChannels; i++){
                int rot_val = i*width_avgpool_out;
                keys_position.push_back(-rot_val);
            }
            /** channels placed in Horner form: one key */
            keys_position.push_back(-width_avgpool_sq);
            for(int i =1; i< width_avgpool_out; i++){
                int rot_val = i*width_avgpool_out;
                keys_position.push_back(i);
                keys_position.push_back(-rot_val);
            }
        }
        else if(stridingType == "single_channel"){
            /** he_avgpool_optimzed: channel shifts (inputSize), downsample of every channel, channels placed in Horner
             *  form (-width_avgpool_sq) */
            add_downsample_rotation_positions(keys_position, inputWidth, stride);
            if (inputChannels > 1) {
                keys_position.push_back(width_sq);
                keys_position.push_back(-width_avgpool_sq);
            }
        }
        else if(stridingType == "multi_channels"){
            /** he_avgpool_optimzed_with_multiple_channels: one downsample_with_multiple_channels */
            add_downsample_rotation_positions(keys_position, inputWidth, stride, inputChannels);
        }

    }
    return unique_rotation_positions(keys_position);
}

/**
 * @brief Generate rotation positions for fully connected linear layers.
 *
 * @param maxFCLayeroutputs Maximum output size.
 * @param rotationPositions Base rotation steps.
 * @param layout Optional diagonal-layout arguments (FHEONLayout); unused by the standard controller, which only keeps the interface shared with FHEONANNDiagonalController.
 * @return List of unique rotation steps.
 */
vector <int> FHEONANNStandardController::generate_linear_rotation_positions(int maxFCLayeroutputs, int rotationPositions, const FHEONLayout& layout){
    
    vector<int> keys_position;

    /** he_linear: merges of up to rotationPositions outputs (-1 ... -(rotationPositions - 1)) and, with more than one
     *  merged group, their Horner placement (-rotationPositions). he_linear_optimized merges all its outputs at once and
     *  passes the number of outputs as rotationPositions. The EvalSum rotations come from
     *  generate_sum_rotation_positions(inputSize). */
    add_merge_rotation_positions(keys_position, std::min(rotationPositions, maxFCLayeroutputs));
    if (maxFCLayeroutputs > rotationPositions) {
        keys_position.push_back(-rotationPositions);
    }
    return unique_rotation_positions(keys_position);
}

/**
 * @brief Homomorphic standard convolution layer.
 *
 * @param encryptedInput Encrypted input ciphertext.
 * @param kernelData Encoded weights filter matrix.
 * @param biasInput Encoded bias vector.
 * @param inputWidth Dimension of input.
 * @param inputChannels Number of input channels.
 * @param outputChannels Number of output channels.
 * @param kernelWidth Dimension of square kernel.
 * @param paddingLen Padding size.
 * @param stride Stride size.
 * @param layout Optional diagonal-layout arguments (FHEONLayout); unused by the standard controller, which only keeps the interface shared with FHEONANNDiagonalController.
 * @return Encrypted convolution output.
 */
Ctext FHEONANNStandardController::he_convolution(Ctext& encryptedInput, vector<vector<Ptext>>& kernelData, Ptext& biasInput,
        int inputWidth,  int inputChannels, int outputChannels,  int kernelWidth, int paddingLen, int stride, const FHEONLayout& layout) {
    /** the computation below is a "valid" convolution: a padded 3x3 convolution goes to he_convolution_optimized, so that
     *  he_convolution covers both cases like in FHEONANNDiagonalController */
    if (paddingLen > 0) {
        return he_convolution_optimized(encryptedInput, kernelData, biasInput, inputWidth, inputChannels, outputChannels, stride);
    }
    auto start_time = startTime();

    int kernelSq = kernelWidth * kernelWidth;
    int inputSize = inputWidth * inputWidth;
    int outputWidth = ((inputWidth - kernelWidth) / stride) + 1;
    int outputSize = outputWidth * outputWidth;
    int encode_level = encryptedInput->GetLevel();

    // STEP 1 - Generate mixed mask for cleaning multi-channel inputs
    int zero_elements = inputSize * (inputChannels - 1);
    if(inputChannels  < 2){
        zero_elements = inputSize;
    }
    vector<double> mixed_mask = generate_mixed_mask(inputSize, zero_elements);
    Ptext cleaning_mask = context->MakeCKKSPackedPlaintext(mixed_mask, 1, encode_level);

    vector<double> mixed_mask_out = generate_mixed_mask(outputWidth, zero_elements);
    Ptext cleaning_mask_out = context->MakeCKKSPackedPlaintext(mixed_mask_out, 1, encode_level);

    // STEP 2 - ROTATE INPUT TO FORM k^2 slices
    vector<Ctext> rotated_ciphertexts;
    for (int i = 0; i < kernelWidth; i++) {
        if(i >0){
            encryptedInput = context->EvalRotate(encryptedInput, inputWidth);
        }
        rotated_ciphertexts.push_back(encryptedInput);
        for (int j = 1; j < kernelWidth; j++) {
            rotated_ciphertexts.push_back(context->EvalRotate(encryptedInput, j));
        }
    }

    // STEP 3-6 - Convolution over all output channels. The channels are placed with a single rotation key:
    // they are processed from the last one and the result is shifted by one channel before each addition
    Ctext strided_cipher;
    Ctext placed;
    for (int out_ch = outputChannels - 1; out_ch >= 0; out_ch--) {
        vector<Ctext> mult_results;

        // Per-kernel value multiplies
        for (int k = 0; k < kernelSq; k++) {
            mult_results.push_back(context->EvalMult(rotated_ciphertexts[k], kernelData[out_ch][k]));
        }

        Ctext conv_sum = context->EvalAddMany(mult_results);

        // STEP 4 - Sum all input channels (rotating and adding)
        if (inputChannels > 1) {
            vector<Ctext> channel_sums = { conv_sum };
            for (int ch = 1; ch < inputChannels; ch++) {
                conv_sum = context->EvalRotate(conv_sum, inputSize);
                channel_sums.push_back(conv_sum);
            }
            conv_sum = context->EvalAddMany(channel_sums);
        }
        conv_sum = context->EvalMult(conv_sum, cleaning_mask);

        // STEP 5 - Striding
        if (stride > 1) {
            strided_cipher = downsample(conv_sum, inputWidth, stride);
        } 
        else {
            vector<Ctext> strided_vec(outputWidth);
            for (int l = 0; l < outputWidth; l++) {
                if (l > 0) {
                    conv_sum = context->EvalRotate(conv_sum, inputWidth);
                }
                strided_vec[l] = context->EvalMult(conv_sum, cleaning_mask_out);
            }
            // row l goes to -(outputWidth * l): Horner form with the single key -outputWidth
            strided_cipher = strided_vec[outputWidth - 1];
            for (int l = outputWidth - 2; l >= 0; l--) {
                strided_cipher = context->EvalAdd(context->EvalRotate(strided_cipher, -outputWidth), strided_vec[l]);
            }
        }

        // STEP 7 - output channel out_ch goes to -(out_ch * outputSize)
        placed = placed ? context->EvalAdd(context->EvalRotate(placed, -outputSize), strided_cipher) : strided_cipher;
    }
    Ctext res = context->EvalAdd(placed, biasInput);
    rotated_ciphertexts.clear();
    printTiming("Convolution evaluated...", "Conv", start_time);
    return res;
}

/**
 * @brief Homomorphic convolution with advanced padding and stride configurations.
 *
 * @param encryptedInput Encrypted input ciphertext.
 * @param kernelData Pre-encoded filter weight vectors.
 * @param biasInput Pre-encoded bias input vector.
 * @param inputWidth Dimensions of the input channels.
 * @param inputChannels Number of input channels.
 * @param outputChannels Number of output filter channels.
 * @param kernelWidth Dimension of the square convolution kernel.
 * @param padding Size of zero padding to apply.
 * @param stride Convolution stride.
 * @param layout Optional diagonal-layout arguments (FHEONLayout); unused by the standard controller, which only keeps the interface shared with FHEONANNDiagonalController.
 * @return Encrypted output of the convolution layer.
 */
Ctext FHEONANNStandardController::he_convolution_advanced(Ctext& encryptedInput, vector<vector<Ptext>>& kernelData, Ptext& biasInput, 
            int inputWidth,  int inputChannels, int outputChannels, int kernelWidth, int padding, int stride, const FHEONLayout& layout){
    auto start_time = startTime();

    /** If padding is 0 */    
    if(padding ==0){
        auto conv_basic_cipher = he_convolution(encryptedInput, kernelData, biasInput, inputWidth, 
                                            kernelWidth, inputChannels, outputChannels, padding, stride);
        return conv_basic_cipher;
    }
    int padded_width = inputWidth + (2*padding);
    int padded_width_sq = pow(padded_width, 2);
    int width_sq = pow(inputWidth, 2);
    int zeros_elements = ((inputChannels*width_sq) - inputWidth);
    int encode_level = encryptedInput->GetLevel();
    auto padding_mix_mask = generate_mixed_mask(inputWidth, zeros_elements);
    Ptext in_clean_mask =  context->MakeCKKSPackedPlaintext(padding_mix_mask, 1, encode_level);

    /** generate vector of padding width */
    Ctext channel_cipher = encryptedInput;
    vector<Ctext> channel_vector_ciphers;
    for(int i = 0; i< inputChannels; i++){
        if(i!=0){
            channel_cipher = context->EvalRotate(channel_cipher, width_sq);
        }
        vector<Ctext> in_chan_vec;
        Ctext in_chan_cipher = channel_cipher;
        for(int k=0; k<inputWidth; k++){
            Ctext in_clean_cipher = context->EvalMult(in_chan_cipher, in_clean_mask);
            in_chan_cipher =  context->EvalRotate(in_chan_cipher, inputWidth);
            if(k==0){
                in_chan_vec.push_back(in_clean_cipher);
            }
            else{
                int in_rot_position = k*padded_width;
                Ctext padded_cipher = context->EvalRotate(in_clean_cipher, -in_rot_position);
                in_chan_vec.push_back(padded_cipher);
            }
        }
        Ctext in_sum_cipher = context->EvalAddMany(in_chan_vec);
        if(i==0){
            channel_vector_ciphers.push_back(in_sum_cipher);
        }
        else{
            int in_rotate = i*padded_width_sq;
            Ctext in_rotate_cipher = context->EvalRotate(in_sum_cipher, -in_rotate);
            channel_vector_ciphers.push_back(in_rotate_cipher);
        }
    }
    int padd_extra = (padding*padded_width)+padding;
    Ctext padded_cipher = context->EvalAddMany(channel_vector_ciphers);
    if(padd_extra != 0){
        padded_cipher = context->EvalRotate(padded_cipher, -padd_extra);
    }
    Ctext conv_basic_cipher = he_convolution(padded_cipher, kernelData, biasInput, inputWidth, kernelWidth, 
                                             inputChannels, outputChannels, padding, stride); 
    printTiming("Advanced convolution evaluated...", "Conv", start_time);
    return conv_basic_cipher;
}

/**
 * @brief Optimized homomorphic convolution using parallel channels.
 *
 * @param encryptedInput Encrypted input ciphertext.
 * @param kernelData Pre-encoded filter weight vectors.
 * @param biasInput Pre-encoded bias input vector.
 * @param inputWidth Dimensions of the input channels.
 * @param inputChannels Number of input channels.
 * @param outputChannels Number of output filter channels.
 * @param stride Convolution stride.
 * @param index Optimization index indicator.
 * @param layout Optional diagonal-layout arguments (FHEONLayout); unused by the standard controller, which only keeps the interface shared with FHEONANNDiagonalController.
 * @return Encrypted convolution output.
 */
Ctext FHEONANNStandardController::he_convolution_optimized(Ctext& encryptedInput,  vector<vector<Ptext>>& kernelData, Ptext& biasInput, 
            int inputWidth, int inputChannels, int outputChannels, int stride, int index, const FHEONLayout& layout){
    return he_convolution_optimized(encryptedInput, [&kernelData](int outCh) { return kernelData[outCh]; }, biasInput,
                                    inputWidth, inputChannels, outputChannels, stride);
}

/**
 * @brief he_convolution_optimized with the 9 kernel plaintexts of an output channel produced on demand by kernelAt.
 */
Ctext FHEONANNStandardController::he_convolution_optimized(Ctext& encryptedInput, const KernelSource& kernelAt, Ptext& biasInput,
            int inputWidth, int inputChannels, int outputChannels, int stride){
    auto start_time = startTime();
    
    int kernelSq = 9;
    int inputSize = inputWidth*inputWidth;
    int widthOut = inputWidth/stride;
    int outputSize = widthOut*widthOut;
    int encode_level = encryptedInput->GetLevel();
    vector<Ctext> rotated_ciphertexts;
    
    auto digits = context->EvalFastRotationPrecompute(encryptedInput);
    Ctext first_shot = context->EvalFastRotation(encryptedInput, -1, context->GetCyclotomicOrder(), digits);
    Ctext second_shot = context->EvalFastRotation(encryptedInput, 1, context->GetCyclotomicOrder(), digits);
    rotated_ciphertexts.push_back(context->EvalRotate(first_shot, -inputWidth));
    rotated_ciphertexts.push_back(context->EvalFastRotation(encryptedInput, -inputWidth, context->GetCyclotomicOrder(), digits));
    rotated_ciphertexts.push_back(context->EvalRotate(second_shot, -inputWidth));
    rotated_ciphertexts.push_back(first_shot);
    rotated_ciphertexts.push_back(encryptedInput);
    rotated_ciphertexts.push_back(second_shot);
    rotated_ciphertexts.push_back(context->EvalRotate(first_shot, inputWidth));
    rotated_ciphertexts.push_back(context->EvalFastRotation(encryptedInput, inputWidth, context->GetCyclotomicOrder(), digits));
    rotated_ciphertexts.push_back(context->EvalRotate(second_shot, inputWidth));
    Ptext cleaning_mask =  context->MakeCKKSPackedPlaintext(generate_mixed_mask(inputSize, (inputChannels*inputSize)), 1, encode_level);
            
    vector<Ctext> kernelSum(kernelSq);
    vector<Ctext> sumVec(inputChannels);
    /** the output channels are processed from the last one and placed with the single rotation key -outputSize */
    Ctext placed;
    for(int outCh=outputChannels-1; outCh>=0; outCh--){
        vector<Ptext> kernel = kernelAt(outCh);
        for(int j=0; j<kernelSq; j++){
            kernelSum[j] = context->EvalMult(rotated_ciphertexts[j], kernel[j]);
        }
        sumVec[0] = context->EvalAddMany(kernelSum);
        
        /*** STEP 4: SUM RESULTS OF ALL INPUT CHANNELS ***/
        for(int k=1; k<inputChannels; k++){
            sumVec[k] = context->EvalRotate(sumVec[k-1], inputSize);
        }
        Ctext interCipher = context->EvalMult(context->EvalAddMany(sumVec), cleaning_mask);
        /**** STEP 5: THIS IS EXTRACTING DATA FROM CONVOLUTION WITH STRIDING > 1 */
        if(stride != 1){
            interCipher = downsample(interCipher, inputWidth, stride);
        }
        /** STEP 7: output channel outCh goes to -(outCh * outputSize) */
        placed = placed ? context->EvalAdd(context->EvalRotate(placed, -outputSize), interCipher) : interCipher;
    }

    Ctext finalResults  = context->EvalAdd(placed, biasInput);
    sumVec.clear();
    kernelSum.clear();
    rotated_ciphertexts.clear();
    printTiming("Optimized convolution evaluated...", "Conv", start_time);
    return finalResults;
}


/**
 * @brief Optimized convolution supporting multiple packed channels.
 *
 * @param encryptedInput Encrypted input ciphertext.
 * @param kernelData Pre-encoded filter weight vectors.
 * @param biasInput Pre-encoded bias input vector.
 * @param inputWidth Dimensions of the input channels.
 * @param inputChannels Number of input channels.
 * @param outputChannels Number of output filter channels.
 * @param layout Optional diagonal-layout arguments (FHEONLayout); unused by the standard controller, which only keeps the interface shared with FHEONANNDiagonalController.
 * @return Encrypted convolution output.
 */
Ctext  FHEONANNStandardController::he_convolution_optimized_with_multiple_channels(Ctext& encryptedInput, vector<vector<Ptext>>& kernelData, 
    Ptext& biasInput,  int inputWidth,  int inputChannels, int outputChannels, const FHEONLayout& layout){
    auto start_time = startTime();
    
    constexpr int stride = 2;
    constexpr int kernelSq = 9;  // Assuming 3x3 kernel (kernelWidthSq = 9)
    int outputWidth = inputWidth / stride;
    int inputSize = inputWidth * inputWidth;
    int outputSize = outputWidth * outputWidth;
    int encodeLevel = encryptedInput->GetLevel();

    // Precompute rotations only once with minimal rotation set
    vector<Ctext> rotatedInputs;
    int vectorSize = inputSize * inputChannels;
    Ptext cleaningMask = context->MakeCKKSPackedPlaintext(generate_mixed_mask(inputSize, vectorSize), 1, encodeLevel);

    Ptext cleaningoutputMask = context->MakeCKKSPackedPlaintext(generate_mixed_mask((inputChannels*outputSize), vectorSize), 1, encodeLevel);
    
    // Horizontal rotations
    auto digits = context->EvalFastRotationPrecompute(encryptedInput);
    Ctext first_shot = context->EvalFastRotation(encryptedInput, -1, context->GetCyclotomicOrder(), digits);
    Ctext second_shot = context->EvalFastRotation(encryptedInput, 1, context->GetCyclotomicOrder(), digits);
    rotatedInputs.push_back(context->EvalRotate(first_shot, -inputWidth));
    rotatedInputs.push_back(context->EvalFastRotation(encryptedInput, -inputWidth, context->GetCyclotomicOrder(), digits));
    rotatedInputs.push_back(context->EvalRotate(second_shot, -inputWidth));
    rotatedInputs.push_back(first_shot);
    rotatedInputs.push_back(encryptedInput);
    rotatedInputs.push_back(second_shot);
    rotatedInputs.push_back(context->EvalRotate(first_shot, inputWidth));
    rotatedInputs.push_back(context->EvalFastRotation(encryptedInput, inputWidth, context->GetCyclotomicOrder(), digits));
    rotatedInputs.push_back(context->EvalRotate(second_shot, inputWidth));

    // Create vectors to store results
    int outchanSize = outputChannels/inputChannels;
    Ctext mainResult;
    vector<Ctext> mainResults(outchanSize);
    vector<Ctext> convChannelSum(inputChannels);
    vector<Ctext> kernelSum(kernelSq);
    // Process output channels in groups of inputChannels: inside a group, output i goes to -(i * inputSize), placed from
    // the last output with the single rotation key -inputSize
    for (int outCount = 0; outCount < outchanSize; outCount++) {
        Ctext groupPlaced;
        for (int innerCount = inputChannels - 1; innerCount >= 0; innerCount--) {
            int outCh = outCount * inputChannels + innerCount;

            // Apply convolution with batch channel processing
            for (int j = 0; j<kernelSq; ++j) {
                kernelSum[j] = context->EvalMult(rotatedInputs[j], kernelData[outCh][j]);
            }
            convChannelSum[0] = context->EvalAddMany(kernelSum);
            for (int inCh = 1; inCh<inputChannels; inCh++) {
                convChannelSum[inCh] = context->EvalRotate(convChannelSum[inCh-1] , inputSize);
            }
            Ctext channelResult = context->EvalMult(context->EvalAddMany(convChannelSum), cleaningMask);
            groupPlaced = groupPlaced ? context->EvalAdd(context->EvalRotate(groupPlaced, -inputSize), channelResult) : channelResult;
        }

        mainResult = downsample_with_multiple_channels(groupPlaced, inputWidth, stride, inputChannels);
        mainResult = context->EvalMult(mainResult, cleaningoutputMask);
        if(outCount == 0){
            mainResults[outCount] = mainResult;
        }
        else{
            int rotateAmount = - outCount * (inputChannels * outputSize);
            mainResults[outCount] = context->EvalRotate(mainResult, rotateAmount);
        }
    }

    // Combine results and add biases
    Ctext finalMainResult = context->EvalAdd(context->EvalAddMany(mainResults), biasInput);
    rotatedInputs.clear();
    mainResults.clear();
    convChannelSum.clear();
    printTiming("Optimized convolution with multiple channels evaluated...", "Conv", start_time);
    return finalMainResult;
}

/**
 * @brief Shortcut connection convolution layer for ResNet skip connections.
 *
 * @param encryptedInput Encrypted input ciphertext.
 * @param kernelData Pre-encoded filter weight vectors.
 * @param biasInput Pre-encoded bias input vector.
 * @param inputWidth Dimensions of the input channels.
 * @param inputChannels Number of input channels.
 * @param outputChannels Number of output filter channels.
 * @param layout Optional diagonal-layout arguments (FHEONLayout); unused by the standard controller, which only keeps the interface shared with FHEONANNDiagonalController.
 * @return Encrypted convolution output.
 */
Ctext FHEONANNStandardController::he_shortcut_convolution(Ctext& encryptedInput,  vector<Ptext>& kernelData, Ptext& biasInput,  
            int inputWidth, int inputChannels, int outputChannels, const FHEONLayout& layout){
    auto start_time = startTime();

    int width_sq = pow(inputWidth, 2);
    int stride = 2;
    int width_out = (inputWidth/stride);
    int width_out_sq =  pow(width_out, 2);
    int encode_level = encryptedInput->GetLevel();
    int noSlots = inputChannels*width_sq;

    int zeros_elements = width_sq*(inputChannels-1);
    auto mixed_mask = generate_mixed_mask(width_sq, zeros_elements);
    Ptext cleaning_mask =  context->MakeCKKSPackedPlaintext(mixed_mask, 1, encode_level, nullptr, noSlots);

    /*** we do this in a loop contains the output layers since this is repeated for every output layer */
    vector<Ctext> sum_vec(inputChannels);
    Ctext interCipher;
    /** the output channels are processed from the last one and placed with the single rotation key -width_out_sq */
    Ctext placed;
    for(int i=outputChannels-1; i>=0; i--){
        sum_vec[0] = context->EvalMult(encryptedInput, kernelData[i]);

        /*** STEP 4: SUM RESULTS OF ALL INPUT CHANNELS **/
        for(int k=1; k<inputChannels; k++){
            sum_vec[k] = context->EvalRotate(sum_vec[k-1], width_sq);
        }
        interCipher = context->EvalMult(context->EvalAddMany(sum_vec), cleaning_mask);

        /**** STEP 5: THIS IS EXTRACTING DATA FROM CONVOLUTION WITH STRIDING > 1 */
        interCipher = downsample(interCipher, inputWidth, stride);

        /** STEP 7: output channel i goes to -(i * width_out_sq) */
        placed = placed ? context->EvalAdd(context->EvalRotate(placed, -width_out_sq), interCipher) : interCipher;
    }

    Ctext finalResults = context->EvalAdd(placed, biasInput);
    sum_vec.clear();
    printTiming("Shortcut convolution evaluated...", "Conv", start_time);
    return finalResults;
}

/**
 * @brief Combined optimized convolution and shortcut branch computation.
 *
 * @param encryptedInput Input ciphertext.
 * @param kernelData Convolution weights.
 * @param shortcutKernelData Shortcut weights.
 * @param biasInput Convolution bias.
 * @param shortcutBiasVector Shortcut bias.
 * @param inputWidth Input dimension.
 * @param inputChannels Input channels count.
 * @param outputChannels Output channels count.
 * @param layout Optional diagonal-layout arguments (FHEONLayout); unused by the standard controller, which only keeps the interface shared with FHEONANNDiagonalController.
 * @return Vector containing [convolution_output, shortcut_output].
 */
vector<Ctext>  FHEONANNStandardController::he_convolution_and_shortcut_optimized( const Ctext& encryptedInput, const vector<vector<Ptext>>& kernelData, 
    const vector<Ptext>& shortcutKernelData, Ptext& biasInput, Ptext& shortcutBiasVector,  
    int inputWidth,  int inputChannels, int outputChannels, const FHEONLayout& layout){
    return he_convolution_and_shortcut_optimized(encryptedInput, [&kernelData](int outCh) { return kernelData[outCh]; },
                                                 [&shortcutKernelData](int outCh) { return shortcutKernelData[outCh]; },
                                                 biasInput, shortcutBiasVector, inputWidth, inputChannels, outputChannels);
}

/**
 * @brief he_convolution_and_shortcut_optimized with the plaintexts of an output channel produced on demand.
 */
vector<Ctext>  FHEONANNStandardController::he_convolution_and_shortcut_optimized(const Ctext& encryptedInput, const KernelSource& kernelAt,
    const ShortcutSource& shortcutAt, Ptext& biasInput, Ptext& shortcutBiasVector,
    int inputWidth,  int inputChannels, int outputChannels){
    auto start_time = startTime();
    constexpr int stride = 2;
    constexpr int kernelSq = 9;  // Assuming 3x3 kernel (kernelWidthSq = 9)
    int outputWidth = inputWidth / stride;
    int inputSize = inputWidth * inputWidth;
    int outputSize = outputWidth * outputWidth;
    int encodeLevel = encryptedInput->GetLevel();

    // Precompute rotations only once with minimal rotation set
    vector<Ctext> rotatedInputs;
    int vectorSize = inputSize * inputChannels;
    Ptext cleaningMask = context->MakeCKKSPackedPlaintext(generate_mixed_mask(inputSize, vectorSize), 1, encodeLevel);
    
    // Horizontal rotations
    auto digits = context->EvalFastRotationPrecompute(encryptedInput);
    Ctext first_shot = context->EvalFastRotation(encryptedInput, -1, context->GetCyclotomicOrder(), digits);
    Ctext second_shot = context->EvalFastRotation(encryptedInput, 1, context->GetCyclotomicOrder(), digits);
    rotatedInputs.push_back(context->EvalRotate(first_shot, -inputWidth));
    rotatedInputs.push_back(context->EvalFastRotation(encryptedInput, -inputWidth, context->GetCyclotomicOrder(), digits));
    rotatedInputs.push_back(context->EvalRotate(second_shot, -inputWidth));
    rotatedInputs.push_back(first_shot);
    rotatedInputs.push_back(encryptedInput);
    rotatedInputs.push_back(second_shot);
    rotatedInputs.push_back(context->EvalRotate(first_shot, inputWidth));
    rotatedInputs.push_back(context->EvalFastRotation(encryptedInput, inputWidth, context->GetCyclotomicOrder(), digits));
    rotatedInputs.push_back(context->EvalRotate(second_shot, inputWidth));

    // Create vectors to store results
    vector<Ctext> convChannelSum(inputChannels), shortcutChannelSum(inputChannels);
    vector<Ctext> kernelSum(kernelSq);
    Ctext mainResult, shortcutResult;
    /** the output channels are processed from the last one and placed with the single rotation key -outputSize */
    Ctext mainPlaced, shortcutPlaced;
    
    // Process output channels with batch approach
    for (int outCh = outputChannels - 1; outCh >= 0; outCh--) {

        // Apply convolution with batch channel processing
        vector<Ptext> kernel = kernelAt(outCh);
        for (int j = 0; j < kernelSq; ++j) {
            kernelSum[j] = context->EvalMult(rotatedInputs[j], kernel[j]);
        }
        convChannelSum[0] = context->EvalAddMany(kernelSum);
        shortcutChannelSum[0] = context->EvalMult(encryptedInput, shortcutAt(outCh));

        for (int inCh = 1; inCh < inputChannels; ++inCh) {
            convChannelSum[inCh] = context->EvalRotate(convChannelSum[inCh-1] , inputSize);
            shortcutChannelSum[inCh] = context->EvalRotate(shortcutChannelSum[inCh-1], inputSize);
        }

        mainResult = context->EvalMult(context->EvalAddMany(convChannelSum), cleaningMask);
        shortcutResult = context->EvalMult(context->EvalAddMany(shortcutChannelSum), cleaningMask);

        /** Compute Striding */
        mainResult = downsample(mainResult, inputWidth, stride);
        shortcutResult = downsample(shortcutResult, inputWidth, stride);
        
        /** output channel outCh goes to -(outCh * outputSize) */
        mainPlaced = mainPlaced ? context->EvalAdd(context->EvalRotate(mainPlaced, -outputSize), mainResult) : mainResult;
        shortcutPlaced = shortcutPlaced ? context->EvalAdd(context->EvalRotate(shortcutPlaced, -outputSize), shortcutResult) : shortcutResult;
    }
    
    // Combine results and add biases
    Ctext finalMainResult = context->EvalAdd(mainPlaced, biasInput);
    Ctext finalShortcutResult = context->EvalAdd(shortcutPlaced, shortcutBiasVector);

    rotatedInputs.clear();
    convChannelSum.clear();
    shortcutChannelSum.clear();
    printTiming("Optimized convolution and shortcut evaluated...", "Conv", start_time);
    return {finalMainResult, finalShortcutResult};
}

/**
 * @brief Combined optimized convolution and shortcut branch for multiple channels.
 *
 * @param encryptedInput Input ciphertext.
 * @param kernelData Convolution weights.
 * @param shortcutKernelData Shortcut weights.
 * @param biasInput Convolution bias.
 * @param shortcutBiasInput Shortcut bias.
 * @param inputWidth Input dimension.
 * @param inputChannels Input channels count.
 * @param outputChannels Output channels count.
 * @param layout Optional diagonal-layout arguments (FHEONLayout); unused by the standard controller, which only keeps the interface shared with FHEONANNDiagonalController.
 * @return Vector containing [convolution_output, shortcut_output].
 */
vector<Ctext>  FHEONANNStandardController::he_convolution_and_shortcut_optimized_with_multiple_channels( const Ctext& encryptedInput, const vector<vector<Ptext>>& kernelData, 
    const vector<Ptext>& shortcutKernelData, Ptext& biasInput, Ptext& shortcutBiasInput,  
    int inputWidth,  int inputChannels, int outputChannels, const FHEONLayout& layout){
    return he_convolution_and_shortcut_optimized_with_multiple_channels(encryptedInput, [&kernelData](int outCh) { return kernelData[outCh]; },
                                                 [&shortcutKernelData](int outCh) { return shortcutKernelData[outCh]; },
                                                 biasInput, shortcutBiasInput, inputWidth, inputChannels, outputChannels);
}

/**
 * @brief he_convolution_and_shortcut_optimized_with_multiple_channels with the plaintexts of an output channel produced on demand.
 */
vector<Ctext>  FHEONANNStandardController::he_convolution_and_shortcut_optimized_with_multiple_channels(const Ctext& encryptedInput,
    const KernelSource& kernelAt, const ShortcutSource& shortcutAt, Ptext& biasInput, Ptext& shortcutBiasInput,
    int inputWidth,  int inputChannels, int outputChannels){
    auto start_time = startTime();
    constexpr int stride = 2;
    constexpr int kernelSq = 9;  // Assuming 3x3 kernel (kernelWidthSq = 9)
    int outputWidth = inputWidth / stride;
    int inputSize = inputWidth * inputWidth;
    int outputSize = outputWidth * outputWidth;
    int encodeLevel = encryptedInput->GetLevel();

    // Precompute rotations only once with minimal rotation set
    vector<Ctext> rotatedInputs;
    int vectorSize = inputSize * inputChannels;
    Ptext cleaningMask = context->MakeCKKSPackedPlaintext(generate_mixed_mask(inputSize, vectorSize), 1, encodeLevel);

    Ptext cleaningoutputMask = context->MakeCKKSPackedPlaintext(generate_mixed_mask((inputChannels*outputSize), vectorSize), 1, encodeLevel);
    
    // Horizontal rotations
    auto digits = context->EvalFastRotationPrecompute(encryptedInput);
    Ctext first_shot = context->EvalFastRotation(encryptedInput, -1, context->GetCyclotomicOrder(), digits);
    Ctext second_shot = context->EvalFastRotation(encryptedInput, 1, context->GetCyclotomicOrder(), digits);
    rotatedInputs.push_back(context->EvalRotate(first_shot, -inputWidth));
    rotatedInputs.push_back(context->EvalFastRotation(encryptedInput, -inputWidth, context->GetCyclotomicOrder(), digits));
    rotatedInputs.push_back(context->EvalRotate(second_shot, -inputWidth));
    rotatedInputs.push_back(first_shot);
    rotatedInputs.push_back(encryptedInput);
    rotatedInputs.push_back(second_shot);
    rotatedInputs.push_back(context->EvalRotate(first_shot, inputWidth));
    rotatedInputs.push_back(context->EvalFastRotation(encryptedInput, inputWidth, context->GetCyclotomicOrder(), digits));
    rotatedInputs.push_back(context->EvalRotate(second_shot, inputWidth));

    // Create vectors to store results
    int outchanSize = outputChannels/inputChannels;
    Ctext mainResult, shortcutResult;
    vector<Ctext> mainResults(outchanSize), shortcutResults(outchanSize);
    vector<Ctext> convChannelSum(inputChannels), shortcutChannelSum(inputChannels);
    vector<Ctext> kernelSum(kernelSq);

    // Process output channels in groups of inputChannels: inside a group, output i goes to -(i * inputSize), placed from
    // the last output with the single rotation key -inputSize
    for (int outCount = 0; outCount < outchanSize; outCount++) {
        Ctext mainPlaced, shortcutPlaced;
        for (int innerCount = inputChannels - 1; innerCount >= 0; innerCount--) {
            int outCh = outCount * inputChannels + innerCount;

            // Apply convolution with batch channel processing
            vector<Ptext> kernel = kernelAt(outCh);
            for (int j = 0; j<kernelSq; ++j) {
                kernelSum[j] = context->EvalMult(rotatedInputs[j], kernel[j]);
            }
            convChannelSum[0] = context->EvalAddMany(kernelSum);
            shortcutChannelSum[0] = context->EvalMult(encryptedInput, shortcutAt(outCh));

            for (int inCh = 1; inCh<inputChannels; inCh++) {
                convChannelSum[inCh] = context->EvalRotate(convChannelSum[inCh-1] , inputSize);
                shortcutChannelSum[inCh] = context->EvalRotate(shortcutChannelSum[inCh-1], inputSize);
            }
            Ctext convChannel = context->EvalMult(context->EvalAddMany(convChannelSum), cleaningMask);
            Ctext shortcutChannel = context->EvalMult(context->EvalAddMany(shortcutChannelSum), cleaningMask);
            mainPlaced = mainPlaced ? context->EvalAdd(context->EvalRotate(mainPlaced, -inputSize), convChannel) : convChannel;
            shortcutPlaced = shortcutPlaced ? context->EvalAdd(context->EvalRotate(shortcutPlaced, -inputSize), shortcutChannel) : shortcutChannel;
        }

        mainResult = downsample_with_multiple_channels(mainPlaced, inputWidth, stride, inputChannels);
        shortcutResult = downsample_with_multiple_channels(shortcutPlaced, inputWidth, stride, inputChannels);
        mainResult = context->EvalMult(mainResult, cleaningoutputMask);
        shortcutResult = context->EvalMult(shortcutResult, cleaningoutputMask);
        
        if(outCount == 0){
            mainResults[outCount] = mainResult;
            shortcutResults[outCount] = shortcutResult;
        }
        else{
            
            int rotateAmount = - outCount * (inputChannels * outputSize);
            mainResults[outCount] = context->EvalRotate(mainResult, rotateAmount);
            shortcutResults[outCount] = context->EvalRotate(shortcutResult, rotateAmount);   
        }
    }
    
    // Combine results and add biases
    Ctext finalMainResult = context->EvalAdd(context->EvalAddMany(mainResults), biasInput);
    Ctext finalShortcutResult = context->EvalAdd(context->EvalAddMany(shortcutResults), shortcutBiasInput);

    rotatedInputs.clear();
    mainResults.clear();
    shortcutResults.clear();
    convChannelSum.clear();
    shortcutChannelSum.clear();
    printTiming("Optimized convolution and shortcut with multiple channels evaluated...", "Conv", start_time);
    return {finalMainResult, finalShortcutResult};
}

/**
 * @brief Homomorphic standard average pooling layer.
 *
 * @param encryptedInput Encrypted input ciphertext.
 * @param inputWidth Dimension of input.
 * @param inputChannels Number of channels.
 * @param kernelWidth Dimension of pooling window.
 * @param stride Pooling stride.
 * @param layout Optional diagonal-layout arguments (FHEONLayout); unused by the standard controller, which only keeps the interface shared with FHEONANNDiagonalController.
 * @return Average-pooled encrypted output.
 */
Ctext FHEONANNStandardController::he_avgpool(Ctext encryptedInput,  int inputWidth, int inputChannels, int kernelWidth, int stride, const FHEONLayout& layout){
    auto start_time = startTime();

    int outputWidth = inputWidth/stride;
    int kernelSq = pow(kernelWidth, 2);
    int inputSize = pow(inputWidth, 2);
    int outputSize = pow(outputWidth, 2);
    int encode_level = encryptedInput->GetLevel();

    /*** STEP 1 - ROTATE THE CIPHERTEXT into by k^2-1 and create a k^2 rotated right positions ***/
    vector<Ctext> rotated_ciphertexts;
    for (int i = 0; i < kernelWidth; i++) {
        if(i >0){
            encryptedInput = context->EvalRotate(encryptedInput, inputWidth);
        }
        rotated_ciphertexts.push_back(encryptedInput);
        for (int j = 1; j < kernelWidth; j++) {
            rotated_ciphertexts.push_back(context->EvalRotate(encryptedInput, j));
        }
    }
    
    /**** STEP 2: Sum the rotated ciphertext */
    Ctext sum_cipher = context->EvalAddMany(rotated_ciphertexts);
    vector<Ctext> channel_ciphers;
    if(inputWidth <= 2){
        for(int i = 1; i<inputChannels; i++){
            sum_cipher = context->EvalRotate(sum_cipher, inputSize);
            channel_ciphers.push_back(sum_cipher);
        }
        return context->EvalMerge(channel_ciphers);
    }
    
    /*** STEP 3: Multiply the scale value with the sum cipher */
    int num_of_elements = inputChannels*inputSize;
    auto masked_data = generate_scale_mask(kernelSq, num_of_elements);
    auto masked_cipher =  context->MakeCKKSPackedPlaintext(masked_data, 1, encode_level);
    sum_cipher = context->EvalMult(sum_cipher, masked_cipher);

    /*** STEP 4: Extract the values needed in the ciphertext; channel i goes to -(i * outputSize), placed from the last
     *** channel with the single rotation key -outputSize */
    channel_ciphers.push_back(downsample(sum_cipher, inputWidth,  stride));
    for(int i = 1; i<inputChannels; i++){
        sum_cipher = context->EvalRotate(sum_cipher, inputSize);
        channel_ciphers.push_back(downsample(sum_cipher, inputWidth, stride));
    }
    Ctext finalResults = channel_ciphers[inputChannels - 1];
    for(int i = inputChannels - 2; i >= 0; i--){
        finalResults = context->EvalAdd(context->EvalRotate(finalResults, -outputSize), channel_ciphers[i]);
    }
    channel_ciphers.clear();
    rotated_ciphertexts.clear();
    printTiming("Average pooling evaluated...", "AvgPool", start_time);
    return finalResults;
}

/**
 * @brief Advanced homomorphic average pooling supporting padding.
 *
 * @param encryptedInput Encrypted input.
 * @param inputWidth Input dimension.
 * @param outputChannels Output channels count.
 * @param kernelWidth Pooling kernel size.
 * @param stride Pooling stride.
 * @param padding Pooling padding.
 * @param layout Optional diagonal-layout arguments (FHEONLayout); unused by the standard controller, which only keeps the interface shared with FHEONANNDiagonalController.
 * @return Average-pooled encrypted output.
 */
Ctext FHEONANNStandardController::he_avgpool_advanced(Ctext encryptedInput, int inputWidth, int outputChannels, 
    int kernelWidth, int stride, int padding, const FHEONLayout& layout){
    auto start_time = startTime();
 
    int encode_level = encryptedInput->GetLevel();
    if(padding ==0){
        auto avgpool_cipher = he_avgpool(encryptedInput, inputWidth, outputChannels, kernelWidth, stride);
        printTiming("Advanced average pooling evaluated...", "AvgPool", start_time);
        return avgpool_cipher;
    }
    int padded_width = inputWidth + (2*padding);
    int padded_width_sq = pow(padded_width, 2);
    int width_sq = pow(inputWidth, 2);
    int zeros_elements = ((outputChannels*width_sq) - inputWidth);
    auto padding_mix_mask = generate_mixed_mask(inputWidth, zeros_elements);
    Ptext in_clean_mask =  context->MakeCKKSPackedPlaintext(padding_mix_mask, 1, encode_level);

    /** generate vector of padding width */
    Ctext channel_cipher = encryptedInput;
    vector<Ctext> channel_vector_ciphers;
    for(int i = 0; i< outputChannels; i++){
        if(i!=0){
            channel_cipher = context->EvalRotate(channel_cipher, width_sq);
        }
        vector<Ctext> in_chan_vec;
        Ctext in_chan_cipher = channel_cipher;
        for(int k=0; k<inputWidth; k++){
            Ctext in_clean_cipher = context->EvalMult(in_chan_cipher, in_clean_mask);
            in_chan_cipher =  context->EvalRotate(in_chan_cipher, inputWidth);
            if(k==0){
                in_chan_vec.push_back(in_clean_cipher);
            }
            else{
                int in_rot_position = k*padded_width;
                Ctext padded_cipher = context->EvalRotate(in_clean_cipher, -in_rot_position);
                in_chan_vec.push_back(padded_cipher);
            }
        }
        Ctext in_sum_cipher = context->EvalAddMany(in_chan_vec);
        if(i==0){
            channel_vector_ciphers.push_back(in_sum_cipher);
        }
        else{
            int in_rotate = i*padded_width_sq;
            Ctext in_rotate_cipher = context->EvalRotate(in_sum_cipher, -in_rotate);
            channel_vector_ciphers.push_back(in_rotate_cipher);
        }
    }
    int padd_extra = (padding*padded_width)+padding;
    Ctext padded_cipher = context->EvalAddMany(channel_vector_ciphers);
    if(padd_extra != 0){
        padded_cipher = context->EvalRotate(padded_cipher, -padd_extra);
    }
    Ctext avgpool_cipher = he_avgpool(padded_cipher, inputWidth, outputChannels, kernelWidth, stride); 
    printTiming("Advanced average pooling evaluated...", "AvgPool", start_time);
    return avgpool_cipher;
}

/**
 * @brief Optimized average pooling using slot shift aggregation.
 *
 * @param encryptedInput Encrypted input.
 * @param inputWidth Input dimension.
 * @param inputChannels Input channels.
 * @param kernelWidth pooling kernel size.
 * @param stride pooling stride.
 * @param layout Optional diagonal-layout arguments (FHEONLayout); unused by the standard controller, which only keeps the interface shared with FHEONANNDiagonalController.
 * @return Average-pooled encrypted output.
 */
Ctext FHEONANNStandardController::he_avgpool_optimzed(Ctext& encryptedInput,  int inputWidth, int inputChannels, int kernelWidth, int stride, const FHEONLayout& layout){
    auto start_time = startTime();

    int kernelSq = pow(kernelWidth, 2);
    int inputSize = pow(inputWidth, 2);
    int outputWidth = inputWidth/stride;
    int outputSize = pow(outputWidth, 2);
    int encode_level = encryptedInput->GetLevel();
    
    /*** STEP 1 - ROTATE THE CIPHERTEXT into by k^2-1 and create a k^2 rotated right positions ***/
    vector<Ctext> rotated_ciphertexts;
    auto digits = context->EvalFastRotationPrecompute(encryptedInput);
    rotated_ciphertexts.push_back(encryptedInput);
    rotated_ciphertexts.push_back(context->EvalFastRotation(encryptedInput, 1, context->GetCyclotomicOrder(), digits));
    rotated_ciphertexts.push_back(context->EvalFastRotation(encryptedInput, inputWidth, context->GetCyclotomicOrder(), digits));
    rotated_ciphertexts.push_back(context->EvalRotate(context->EvalFastRotation(encryptedInput, inputWidth, context->GetCyclotomicOrder(), digits), 1));
    Ctext sum_cipher = context->EvalAddMany(rotated_ciphertexts);

    /*** STEP 3: Multiply the scale value with the sum cipher */
    int num_of_elements = inputChannels*inputSize;
    auto masked_data = generate_scale_mask(kernelSq, num_of_elements);
    auto masked_cipher =  context->MakeCKKSPackedPlaintext(masked_data, 1, encode_level);
    sum_cipher = context->EvalMult(sum_cipher, masked_cipher);
    
    vector<Ctext> channel_ciphers;
    
    /**** Caryout the average pooling ofif we have just 3 elements in a channel */
    if(inputWidth <= 2){
        for(int i = 1; i<inputChannels; i++){
            sum_cipher = context->EvalRotate(sum_cipher, inputSize);
            channel_ciphers.push_back(sum_cipher);
        }
        return context->EvalMerge(channel_ciphers);
    }

    /*** STEP 4: Extract the values needed in the ciphertext; channel i goes to -(i * outputSize), placed from the last
     *** channel with the single rotation key -outputSize */
    channel_ciphers.push_back(downsample(sum_cipher, inputWidth,  stride));
    for(int i = 1; i<inputChannels; i++){
        sum_cipher = context->EvalRotate(sum_cipher, inputSize);
        channel_ciphers.push_back(downsample(sum_cipher, inputWidth, stride));
    }
    Ctext finalResult = channel_ciphers[inputChannels - 1];
    for(int i = inputChannels - 2; i >= 0; i--){
        finalResult = context->EvalAdd(context->EvalRotate(finalResult, -outputSize), channel_ciphers[i]);
    }
    channel_ciphers.clear();
    printTiming("Optimized average pooling evaluated...", "AvgPool", start_time);
    return finalResult;
}


/**
 * @brief Optimized average pooling supporting multiple channels.
 *
 * @param encryptedInput Encrypted input.
 * @param inputWidth Input dimension.
 * @param inputChannels Input channels.
 * @param kernelWidth pooling kernel size.
 * @param stride pooling stride.
 * @param layout Optional diagonal-layout arguments (FHEONLayout); unused by the standard controller, which only keeps the interface shared with FHEONANNDiagonalController.
 * @return Average-pooled encrypted output.
 */
Ctext FHEONANNStandardController::he_avgpool_optimzed_with_multiple_channels(Ctext& encryptedInput,  int inputWidth, int inputChannels, int kernelWidth, int stride, const FHEONLayout& layout){
    auto start_time = startTime();

    int kernelSq = pow(kernelWidth, 2);
    int inputSize = pow(inputWidth, 2);
    int encode_level = encryptedInput->GetLevel();
    
    /*** STEP 1 - ROTATE THE CIPHERTEXT into by k^2-1 and create a k^2 rotated right positions ***/
    vector<Ctext> rotated_ciphertexts;
    auto digits = context->EvalFastRotationPrecompute(encryptedInput);
    rotated_ciphertexts.push_back(encryptedInput);
    rotated_ciphertexts.push_back(context->EvalFastRotation(encryptedInput, 1, context->GetCyclotomicOrder(), digits));
    rotated_ciphertexts.push_back(context->EvalFastRotation(encryptedInput, inputWidth, context->GetCyclotomicOrder(), digits));
    rotated_ciphertexts.push_back(context->EvalRotate(context->EvalFastRotation(encryptedInput, inputWidth, context->GetCyclotomicOrder(), digits), 1));
    Ctext sum_cipher = context->EvalAddMany(rotated_ciphertexts);

    /*** STEP 3: Multiply the scale value with the sum cipher */
    int num_of_elements = inputChannels*inputSize;
    auto masked_data = generate_scale_mask(kernelSq, num_of_elements);
    auto masked_cipher =  context->MakeCKKSPackedPlaintext(masked_data, 1, encode_level);
    sum_cipher = context->EvalMult(sum_cipher, masked_cipher);
    
    vector<Ctext> channel_ciphers;
    
    /**** Caryout the average pooling ofif we have just 3 elements in a channel */
    if(inputWidth <= 2){
        for(int i = 1; i<inputChannels; i++){
            sum_cipher = context->EvalRotate(sum_cipher, inputSize);
            channel_ciphers.push_back(sum_cipher);
        }
        Ctext finalResult = context->EvalMerge(channel_ciphers);
        printTiming("Optimized average pooling with multiple channels evaluated...", "AvgPool", start_time);
        return finalResult;
    }

    Ctext finalResult = downsample_with_multiple_channels(sum_cipher, inputWidth, stride, inputChannels);
    printTiming("Optimized average pooling with multiple channels evaluated...", "AvgPool", start_time);
    return finalResult; 
}

/**
 * @brief Homomorphically add two ciphertexts together.
 *
 * @param firstInput The first encrypted input.
 * @param secondInput The second encrypted input.
 * @return Encrypted sum of the inputs.
 */
Ctext FHEONANNStandardController::he_sum_two_ciphertexts(Ctext& firstInput, Ctext& secondInput){
    auto start_time = startTime();
    Ctext sumCipher = context->EvalAdd(firstInput, secondInput);
    printTiming("Addition evaluated...", "Add", start_time);
    return sumCipher;
}

/**
 * @brief Homomorphic global average pooling (reduces space dimension to 1).
 *
 * @param encryptedInput Input ciphertext.
 * @param inputWidth Input dimension.
 * @param outputChannels Output channels.
 * @param kernelWidth Pooling window dimension.
 * @param rotatePositions Rotation positions.
 * @param layout Optional diagonal-layout arguments (FHEONLayout); unused by the standard controller, which only keeps the interface shared with FHEONANNDiagonalController.
 * @return Encrypted global pooled output.
 */
Ctext FHEONANNStandardController::he_globalavgpool(Ctext& encryptedInput,  int inputWidth, 
                        int outputChannels, int kernelWidth, int rotatePositions, const FHEONLayout& layout){
    auto start_time = startTime();
    
    // int encode_level = encryptedInput->GetLevel();
    /**** STEP 2: Sum the rotated ciphertext */
    // Ctext sum_cipher = context->EvalAddMany(rotated_ciphertexts);
    int width_sq = inputWidth*inputWidth;
    // int zero_elements = outputChannels*pow(kernelWidth, 2);
    auto masked_cipher =  context->MakeCKKSPackedPlaintext(generate_scale_mask(width_sq, outputChannels), 1);
    // auto mixed_masked_cipher =  context->MakeCKKSPackedPlaintext(generate_mixed_mask(width_sq, zero_elements), 1);

    /*** STEP 4: Extract the values needed in the ciphertext
    ****        Generate  ciphers = number of output channels rotating at width^2 
    **** rotate at each cipher at w and extract values in w *********/
    vector<Ctext> channel_ciphers;
    vector<Ctext> result_ciphers;
    int j = 0;
    
    for(int i = 0; i< outputChannels; i++){
        if(i != 0){
            encryptedInput = context->EvalRotate(encryptedInput, width_sq);
        }
        // Ctext sumRe = context->EvalMult(encryptedInput, mixed_masked_cipher);
        result_ciphers.push_back(context->EvalSum(encryptedInput, width_sq));
        j+=1;

        /** check whether is equal to imgcols, merge them and rotate by imgCols. 
        * If i is equal to the outputSize, merge and rotate by imgCols */
        if(j == rotatePositions || i == (outputChannels-1)){
            channel_ciphers.push_back(context->EvalMerge(result_ciphers));
            result_ciphers.clear();
            j=0;
        }
    }

    /** merged group k goes to -(k * rotatePositions): Horner form with the single key -rotatePositions */
    Ctext placed = channel_ciphers.back();
    for(int k = static_cast<int>(channel_ciphers.size()) - 2; k >= 0; k--){
        placed = context->EvalAdd(context->EvalRotate(placed, -rotatePositions), channel_ciphers[k]);
    }
    Ctext res = context->EvalMult(placed, masked_cipher);
    channel_ciphers.clear();
    result_ciphers.clear();
    printTiming("Global average pooling evaluated...", "AvgPool", start_time);
    return res;
}

/**
 * @brief Homomorphic fully connected linear layer.
 *
 * @param encryptedInput Input ciphertext vector.
 * @param weightMatrix Weight plaintexts vector.
 * @param biasInput Bias plaintext.
 * @param inputSize Size of input dimension.
 * @param outputSize Size of output dimension.
 * @param rotatePositions Rotation positions.
 * @param layout Optional diagonal-layout arguments (FHEONLayout); unused by the standard controller, which only keeps the interface shared with FHEONANNDiagonalController.
 * @return Encrypted output of linear layer.
 */
Ctext FHEONANNStandardController::he_linear(Ctext& encryptedInput, vector<Ptext>& weightMatrix, Ptext& biasInput, 
                    int inputSize, int outputSize, int rotatePositions, const FHEONLayout& layout){
    auto start_time = startTime();

    int output_size = weightMatrix.size();
    if(outputSize > output_size){
        /** need to handle error here because outputSize should never be grater than output_size */
        cout << "There is an error: ouputsize cannot be larger than weightedMatrix" << endl;
        return encryptedInput;
    }
    /* calculate the results of weights * encrypted vector + bais. 
    Shit the results by number of elements in inputsize*iteration value */
    vector<Ctext> result_matrix;
    vector<Ctext> inner_matrix;
    int j = 0;
    for(int i = 0; i < outputSize; i++){
        inner_matrix.push_back(context->EvalSum(context->EvalMult(encryptedInput, weightMatrix[i]), inputSize));
        j+=1;
        /** merge every rotatePositions outputs (and the last ones) */
        if(j == rotatePositions || i == (outputSize-1)){
            result_matrix.push_back(context->EvalMerge(inner_matrix));
            inner_matrix.clear();
            j=0;
        }
    }

    /**** merged group k goes to -(k * rotatePositions): Horner form with the single key -rotatePositions, then add the biasInput ***/
    Ctext placed = result_matrix.back();
    for(int k = static_cast<int>(result_matrix.size()) - 2; k >= 0; k--){
        placed = context->EvalAdd(context->EvalRotate(placed, -rotatePositions), result_matrix[k]);
    }
    Ctext fResults = context->EvalAdd(placed, biasInput);
    inner_matrix.clear();
    result_matrix.clear();
    printTiming("Linear evaluated...", "FC", start_time);
    return fResults;
}

/**
 * @brief Optimized homomorphic fully connected linear layer.
 *
 * @param encryptedInput Input ciphertext vector.
 * @param weightMatrix Weight plaintexts vector.
 * @param biasInput Bias plaintext.
 * @param inputSize Size of input.
 * @param outputSize Size of output.
 * @param layout Optional diagonal-layout arguments (FHEONLayout); unused by the standard controller, which only keeps the interface shared with FHEONANNDiagonalController.
 * @return Encrypted output of linear layer.
 */
Ctext FHEONANNStandardController::he_linear_optimized(Ctext& encryptedInput, vector<Ptext>& weightMatrix, 
                Ptext& biasInput, int inputSize, int outputSize, const FHEONLayout& layout){
    auto start_time = startTime();

    int output_size = weightMatrix.size();
    if(outputSize > output_size){
        /** need to handle error here because outputSize should never be grater than output_size */
        cout << "There is an error: ouputsize cannot be larger than weightedMatrix" << endl;
        return encryptedInput;
    }
    /* calculate the results of weights * encrypted vector + bais. 
    Shit the results by number of elements in inputsize*iteration value */
    vector<Ctext> inner_matrix;
    for(int i = 0; i < outputSize; i++){
        inner_matrix.push_back(context->EvalSum(context->EvalMult(encryptedInput, weightMatrix[i]), inputSize));
    }
    Ctext res = context->EvalAdd(context->EvalMerge(inner_matrix), biasInput);
    printTiming("Optimized linear evaluated...", "FC", start_time);
    return res;
}

/**
 * @brief Homomorphic ReLU activation layer (CKKS polynomial approximation).
 *
 * @param encryptedInput Ciphertext to activate.
 * @param scaleValue Scaling factor.
 * @param vectorSize Number of elements in the vector.
 * @param polyDegree Degree of approximating polynomial.
 * @param normalizedOutput If true, returns ReLU(x / scaleValue) within [0, 1] (the next layer multiplies its weights by the scale); otherwise ReLU(x) itself, evaluated as scaleValue * ReLU(x / scaleValue).
 * @return Activated ciphertext.
 */
Ctext FHEONANNStandardController::he_relu(Ctext& encryptedInput, double scaleValue,  int vectorSize, int polyDegree, bool normalizedOutput) {
    auto start_time = startTime();
    double lowerBound = -1;
    double upperBound = 1;

    // scaleValue = 2*scaleValue;
    
    auto encryptInn = encryptedInput->Clone();
    if (normalizedOutput) {
        /** ReLU(x / scale) within [0, 1]: the next layer multiplies its weights by the scale (any scale, also below 1) */
        encryptInn = context->EvalMult(encryptedInput, 1.0 / scaleValue);
        Ctext normalized = context->EvalChebyshevFunction([](double x) -> double { return (x < 0) ? 0 : x; }, encryptInn, lowerBound, upperBound, polyDegree);
        printTiming("ReLU evaluated...", "ReLU", start_time);
        return normalized;
    }
    if(scaleValue > 1){
        // scaleValue = 2*scaleValue;
        auto mask_data = context->MakeCKKSPackedPlaintext(generate_scale_mask(scaleValue, vectorSize), 1, 0, nullptr, nextPowerOf2(vectorSize));
        encryptInn = context->EvalMult(encryptedInput, mask_data);
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
 * @brief Extract elements from ciphertext using basic striding.
 *
 * @param in_cipher Input ciphertext.
 * @param inputWidth Width of input.
 * @param widthOut Width of output.
 * @param stride Stride value.
 * @return Strided output ciphertext.
 */
Ctext  FHEONANNStandardController::basic_striding(Ctext in_cipher, int inputWidth, int widthOut,  int stride){
    
    auto in_digits = context->EvalFastRotationPrecompute(in_cipher);
    vector<Ctext> chan_vec(widthOut);
    vector<Ctext> rotated_ciphertexts(widthOut);
    int i_rot = stride*inputWidth;
    
    for (int k =0; k<widthOut; k++){
        if(k!=0){
            in_cipher = context->EvalFastRotation(in_cipher, i_rot, context->GetCyclotomicOrder(), in_digits);
            in_digits = context->EvalFastRotationPrecompute(in_cipher);
        }
        for(int t=0; t<widthOut; t++){
            if(t==0){
                rotated_ciphertexts[t]= in_cipher;
            }
            else{
                rotated_ciphertexts[t] = context->EvalFastRotation(in_cipher, t*stride, context->GetCyclotomicOrder(), in_digits);
            }
        }
        Ctext merged_cipher = context->EvalMerge(rotated_ciphertexts);
        if(k == 0){
            chan_vec[k] = merged_cipher;
        }
        else{
            chan_vec[k] = context->EvalRotate(merged_cipher, -k*widthOut);
        }
    }
    return context->EvalAddMany(chan_vec);
}


// Apply convolution with batch channel processing
/**
 * @brief Perform batch convolution operations on rotated ciphertexts.
 *
 * Multiplies rotated ciphertexts with kernel weight masks and sums them up.
 *
 * @param rotatedInputs Rotated input ciphertexts.
 * @param kernelData Encoded kernel weight masks.
 * @param kernelSq Squared dimension of kernel (k^2).
 * @param inputSize Size of each input channel.
 * @param inputChannels Number of input channels.
 * @return Aggregated batch convolution output.
 */
Ctext  FHEONANNStandardController::batch_convolution_operation(const vector<Ctext>& rotatedInputs, const vector<Ptext>& kernelData, int kernelSq, int inputSize,  int inputChannels) {
    
    // Apply kernel to each rotated cipher
    vector<Ctext> kernelSum(kernelSq);
    for (int j = 0; j < kernelSq; ++j) {
        kernelSum[j] = context->EvalMult(rotatedInputs[j], kernelData[j]);
    }
    return context->EvalAddMany(kernelSum);
}

/**
 * @brief Downsample ciphertext slots.
 *
 * @param input Input ciphertext.
 * @param inputWidth Width of input.
 * @param stride Downsample stride.
 * @return Downsampled ciphertext.
 */
Ctext FHEONANNStandardController::downsample(const Ctext& input, int inputWidth, int stride) {
    
    Ctext result = input->Clone();
    const int outputWidth = inputWidth / stride;
    const int inputSize = inputWidth * inputWidth;
    
    // Step 1: Binary decomposition for row juxtaposition
    result = context->EvalMult(result, first_mask(inputWidth, inputSize, stride, input->GetLevel()));
    for (int s = 1; s < log2(outputWidth); s++) {
        result = context->EvalMult(
            context->EvalAdd(result, context->EvalRotate(result, pow(2, s-1))),
            generate_binary_mask(pow(2,s), inputSize, stride, input->GetLevel())
        );
    }
    result = context->EvalAdd(result, context->EvalRotate(result, pow(2, (log2(outputWidth)-1))));
    // Step 2: Row processing with optimized rotations
    Ctext downsampledrows = context->EvalMult(input, generate_zero_mask(inputSize, input->GetLevel()));
    
    for (int row = 0; row < outputWidth; ++row) {
        Ctext masked = context->EvalMult(result, generate_row_mask(row, outputWidth, inputSize, stride, input->GetLevel()));
        downsampledrows = context->EvalAdd(downsampledrows, masked);
        if(row < outputWidth-1){
            result = context->EvalRotate(result, (stride*inputWidth - outputWidth));
        }
    }
    return downsampledrows;
}

/**
 * @brief Downsample multiple channels packed within a ciphertext.
 *
 * @param input Input ciphertext.
 * @param inputWidth Input dimension.
 * @param stride Downsample stride.
 * @param numChannels Number of channels.
 * @return Downsampled ciphertext.
 */
Ctext FHEONANNStandardController::downsample_with_multiple_channels(const Ctext& input, int inputWidth, int stride, int numChannels) {
    if (!context) {
        cerr << "ERROR: FHEONANNStandardController::context is not set before downsample_with_multiple_channels" << endl;
        std::abort();
    }
    
    const int inputSize   = inputWidth * inputWidth; 
    const int outputWidth  = inputWidth / stride;
    const int level        = input->GetLevel();
    // const int level        = 1;
    int outputSize = outputWidth * outputWidth;
    Ctext encryptedzeros =  context->EvalMult(input, generate_zero_mask_channels(inputSize, numChannels, input->GetLevel()));

    // 2) binary-row decomposition
    Ctext result = context->EvalMult(input, first_mask_with_channels(inputWidth, inputSize, stride, numChannels, level));

     // Step 1: Binary decomposition for row juxtaposition
    for (int s = 1; s < log2(outputWidth); s++) {
        result = context->EvalMult(
            context->EvalAdd(result, context->EvalRotate(result, pow(2, s-1))),
            generate_binary_mask_with_channels(pow(2,s), inputSize, stride,  numChannels, input->GetLevel())
        );
    }

    result = context->EvalAdd(result, context->EvalRotate(result, pow(2, (log2(outputWidth)-1))));

    // Step 2: Row processing with optimized rotations
    Ctext downsampledrows = encryptedzeros->Clone();
    for (int row = 0; row < outputWidth; row++) {
        Ctext masked = context->EvalMult(result, generate_row_mask_with_channels(row, outputWidth, inputSize, stride, numChannels, input->GetLevel()));
        downsampledrows = context->EvalAdd(downsampledrows, masked);
        if(row < outputWidth-1){
            result = context->EvalRotate(result, (stride*inputWidth - outputWidth));
        }
    }

    /***
    * step 3: process per channel
    ******/ 
    Ctext downsampledchannels =  encryptedzeros->Clone();
    for (int ch=0; ch < numChannels; ch++){
        Ctext masked = context->EvalMult(downsampledrows, generate_channel_mask_with_zeros(ch, outputSize, numChannels, input->GetLevel()));
        downsampledchannels = context->EvalAdd(downsampledchannels, masked);
        if(ch < numChannels-1){
            downsampledrows = context->EvalRotate(downsampledrows, (inputSize - outputSize));
        }
    }

    /*** step 3: process per channel */ 
    // int totalSize = numChannels * outputSize;
    // Ctext downsampledchannels = encryptedzeros->Clone();
    // for (int i = 0; i < numChannels; i++) {
    //     Ctext masked = context->EvalMult(downsampledrows, gen_channel_full_mask(i, inputSize, outputSize, numChannels, downsampledrows->GetLevel()));
    //     downsampledchannels = context->EvalAdd(downsampledchannels, masked);
    //     downsampledchannels = context->EvalRotate(downsampledchannels, -(inputSize - outputSize));
    // }

    // downsampledchannels = context->EvalRotate(downsampledchannels, (inputSize - outputSize) * numChannels);
    // downsampledchannels = context->EvalAdd(downsampledchannels, context->EvalRotate(downsampledchannels, -totalSize));

    return downsampledchannels;
}

/**
 * @brief Generate binary mask selecting elements after striding.
 *
 * @param width Width.
 * @param inputSize Input size.
 * @param stride Stride.
 * @param level Ciphertext level.
 * @return Plaintext mask.
 */
Ptext FHEONANNStandardController::first_mask(int width, int inputSize, int stride, int level) {
    vector<double> mask(inputSize, 0);
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            if (j % stride == 0 && i % stride == 0) {
                int index = i * width + j;
                mask[index] = 1.0;
            }
        }
    }
    return context->MakeCKKSPackedPlaintext(mask, 1.0, level);
}

/**
 * @brief Generate binary mask selecting elements at stride positions.
 *
 * @param pattern Stride pattern index.
 * @param inputSize Size of input.
 * @param stride Stride size.
 * @param level Ciphertext level.
 * @return Plaintext mask.
 */
Ptext FHEONANNStandardController::generate_binary_mask(int pattern, int inputSize, int stride, int level) {
    vector<double> mask;
    int copy_interval = pattern;
    for (int i = 0; i < inputSize; i++) {
        if (copy_interval > 0) {
            mask.push_back(1);
        } else {
            mask.push_back(0);
        }

        copy_interval--;

        if (copy_interval <= -pattern) {
            copy_interval = pattern;
        }
    }
    return context->MakeCKKSPackedPlaintext(mask, 1.0, level);
}

/**
 * @brief Generate mask for selecting specific row.
 *
 * @param row Row index to select.
 * @param width Width of the input.
 * @param inputSize Total number of elements in the input.
 * @param stride Stride size.
 * @param level Encryption level.
 * @return Plaintext mask.
 */
Ptext FHEONANNStandardController::generate_row_mask(int row, int width, int inputSize, int stride, int level) {
    vector<double> mask;

    for (int j = 0; j < (row * width); j++) {
        mask.push_back(0);
    }
    for (int j = 0; j < width; j++) {
        mask.push_back(1);
    }
    for (int j = 0; j < (inputSize - width - (row * width)); j++) {
        mask.push_back(0);
    }
    return context->MakeCKKSPackedPlaintext(mask, 1.0, level);
}

/**
 * @brief Generate zero-valued mask of specified size.
 *
 * @param size Mask size.
 * @param level Ciphertext level.
 * @return Plaintext mask.
 */
Ptext FHEONANNStandardController::generate_zero_mask(int size, int level) {
    vector<double> mask(size, 0.0);
    return context->MakeCKKSPackedPlaintext(mask, 1.0, level);
}

/**
 * @brief Generate full channel mask.
 *
 * @param n Mask index.
 * @param in_elements Input elements count.
 * @param out_elements Output elements count.
 * @param numChannels Channels count.
 * @param level Ciphertext level.
 * @return Plaintext mask.
 */
Ptext FHEONANNStandardController::generate_channel_full_mask(int n, int in_elements, int out_elements, int numChannels, int level) {

    const int totalSlots = in_elements * numChannels;
    std::vector<double> mask(totalSlots, 0.0);
    const int base = n * in_elements;
    for (int i = 0; i < out_elements; ++i){
        mask[base + i] = 1.0;
    } 
    return context->MakeCKKSPackedPlaintext(mask, 1.0, level);
}

/**
 * @brief Generate zero mask across multiple channels.
 *
 * @param inputSize Input size.
 * @param numChannels Channels count.
 * @param level Ciphertext level.
 * @return Plaintext mask.
 */
Ptext FHEONANNStandardController::generate_zero_mask_channels(int inputSize, int numChannels, int level) {
    
    int totalSlots = inputSize * numChannels;
    vector<double> mask(totalSlots, 0.0);
    return context->MakeCKKSPackedPlaintext(mask, 1.0, level);
}

/**
 * @brief Generate first mask for multiple channels.
 *
 * @param inputWidth Input width.
 * @param inputSize Input size.
 * @param stride Stride value.
 * @param numChannels Channels count.
 * @param level Ciphertext level.
 * @return Plaintext mask.
 */
Ptext FHEONANNStandardController::first_mask_with_channels(int inputWidth, int inputSize, int stride, int numChannels, int level) {
    // int outputWidth = inputWidth / stride;
    vector<double> mask;
    vector<double> baseMask(inputSize, 0.0);
    for (int i = 0; i < inputWidth; i++) {
        for (int j = 0; j < inputWidth; j++) {
            if (j % stride == 0 && i % stride == 0) {
                int index =(i * inputWidth + j);
                baseMask[index] = 1.0;
            }
        }
    }

    for (int ch = 0; ch < numChannels; ch++) {
        mask.insert(mask.end(), baseMask.begin(), baseMask.end());
    }
    return context->MakeCKKSPackedPlaintext(mask, 1.0, level);
}

/**
 * @brief Generate binary mask supporting multiple channels.
 *
 * @param pattern Pattern index.
 * @param inputSize Input size.
 * @param stride Stride value.
 * @param numChannels Channels count.
 * @param level Ciphertext level.
 * @return Plaintext mask.
 */
Ptext FHEONANNStandardController::generate_binary_mask_with_channels(int pattern, int inputSize, int stride,  int numChannels, int level) {
   
    vector<double> baseMask;
    int copy_interval = pattern;
    for (int i = 0; i < inputSize; i++) {
        if (copy_interval > 0) {
            baseMask.push_back(1);
        } else {
            baseMask.push_back(0);
        }

        copy_interval--;

        if (copy_interval <= -pattern) {
            copy_interval = pattern;
        }
    }

    // repeat baseMask n times
    vector<double> mask;
    mask.reserve(baseMask.size() * numChannels);
    for (int i = 0; i < numChannels; i++) {
        mask.insert(mask.end(), baseMask.begin(), baseMask.end());
    }

    return context->MakeCKKSPackedPlaintext(mask, 1.0, level);
}

/**
 * @brief Generate row mask supporting multiple channels.
 *
 * @param row Row index.
 * @param width Width.
 * @param inputSize Input size.
 * @param stride Stride value.
 * @param numChannels Channels count.
 * @param level Ciphertext level.
 * @return Plaintext mask.
 */
Ptext FHEONANNStandardController::generate_row_mask_with_channels(int row, int width, int inputSize, int stride, int numChannels, int level) {
    
    vector<double> baseMask;
    for (int j = 0; j < (row * width); j++) {
        baseMask.push_back(0);
    }
    for (int j = 0; j < width; j++) {
        baseMask.push_back(1);
    }
    for (int j = 0; j < (inputSize - width - (row * width)); j++) {
        baseMask.push_back(0);
    }

    // repeat baseMask n times
    vector<double> mask;
    mask.reserve(baseMask.size() * numChannels);
    for (int i = 0; i < numChannels; i++) {
        mask.insert(mask.end(), baseMask.begin(), baseMask.end());
    }

    return context->MakeCKKSPackedPlaintext(mask, 1.0, level);
}

/**
 * @brief Generate channel mask selecting one channel and zeroing others.
 *
 * @param channel Channel index.
 * @param outputSize Channel size.
 * @param numChannels Channels count.
 * @param level Ciphertext level.
 * @return Plaintext mask.
 */
Ptext FHEONANNStandardController::generate_channel_mask_with_zeros(int channel, int outputSize, int numChannels, int level ){
    
    int totalSlots = outputSize * numChannels;
    vector<double> mask(totalSlots, 0.0);

    int pos = channel * outputSize;
    for (int i = 0; i < outputSize; i++) {
        mask[pos + i] = 1.0;
    }
    return context->MakeCKKSPackedPlaintext(mask, 1.0, level);
}


/****************************************************************************************************************
 * Shared interface (same declarations in FHEONANNDiagonalController)
 ****************************************************************************************************************/

/**
 * @brief Level at which the kernels of he_convolution_optimized / he_convolution_and_shortcut_optimized are encoded.
 *
 * @param encryptedInput Input ciphertext of the convolution.
 * @param inputWidth Spatial width of one input channel (unused here).
 * @param stride Stride of the convolution (unused here).
 * @param layout Optional diagonal-layout arguments (FHEONLayout); unused by the standard controller, which only keeps the interface shared with FHEONANNDiagonalController.
 * @return The level of the input ciphertext, as the standard models encode their kernels.
 */
int FHEONANNStandardController::convolution_kernel_level(const Ctext& encryptedInput, int inputWidth, int stride, const FHEONLayout& layout){
    return encryptedInput->GetLevel();
}

/**
 * @brief Encode a 3x3 kernel for he_convolution_optimized (stride 1) or he_convolution_and_shortcut_optimized (stride 2).
 *
 * @param heController Controller holding the encoders.
 * @param kernelData Weights [outputChannels][inputChannels][3][3].
 * @param inputWidth Spatial width of one input channel.
 * @param numSlots Slot count of the input ciphertext (unused here).
 * @param encodeLevel Encoding level.
 * @param stride Stride of the convolution (unused here: both use the input width).
 * @param layout Optional layout arguments (FHEONLayout): only weightScale is used, multiplied into the weights (e.g. a factor a preceding pooling expects in the next layer); the other fields are for FHEONANNDiagonalController.
 * @param padding Zero padding of the convolution: > 0 ("same", he_convolution_optimized) uses the masked encoding (encode_kernel_optimized), 0 ("valid", he_convolution) the plain one.
 * @return One set of 9 plaintexts per output channel.
 */
vector<vector<Ptext>> FHEONANNStandardController::encode_convolution_kernel(FHEONHEController& heController, vector<vector<vector<vector<double>>>>& kernelData,
                            int inputWidth, int numSlots, int encodeLevel, int stride, int padding, const FHEONLayout& layout){
    vector<vector<Ptext>> encodedKernel;
    for (auto outChannel : kernelData) {
        if (layout.weightScale != 1.0) {
            for (auto &inChannel : outChannel) for (auto &row : inChannel) for (auto &value : row) { value *= layout.weightScale; }
        }
        /** "same" convolutions (he_convolution_optimized) use the masked encoding, "valid" ones (he_convolution) the plain one */
        encodedKernel.push_back(padding > 0 ? heController.encode_kernel_optimized(outChannel, inputWidth*inputWidth, encodeLevel)
                                            : heController.encode_kernel(outChannel, inputWidth*inputWidth));
    }
    return encodedKernel;
}

/**
 * @brief Factor the next layer's weights need after a pooling: 1, the standard pooling returns averages.
 */
double FHEONANNStandardController::avgpool_weight_scale(int kernelWidth){
    return 1.0;
}

/**
 * @brief Encode a convolution bias, one value per output channel block of outputWidth^2 slots.
 * @param layout Optional diagonal-layout arguments (FHEONLayout); unused by the standard controller, which only keeps the interface shared with FHEONANNDiagonalController.
 * @param heController Controller holding the encoders.
 * @param biasData One bias per output channel.
 * @param outputWidth Output width: every bias fills outputWidth^2 slots.
 * @param numSlots Slot count of the input ciphertext (unused by the standard encoding, kept for the shared interface).
 * @param encodeLevel Encoding level (the level of the ciphertexts the plaintexts multiply).
 */
Ptext FHEONANNStandardController::encode_convolution_bias(FHEONHEController& heController, vector<double>& biasData, int outputWidth, int numSlots, int encodeLevel, const FHEONLayout& layout){
    return heController.encode_bais_input(biasData, outputWidth*outputWidth, encodeLevel);
}

/**
 * @brief Encode the 1x1 shortcut kernel of he_convolution_and_shortcut_optimized (one plaintext per output channel).
 * @param layout Optional diagonal-layout arguments (FHEONLayout); unused by the standard controller, which only keeps the interface shared with FHEONANNDiagonalController.
 * @param heController Controller holding the encoders.
 * @param kernelData 1x1 weights [output channel][input channel].
 * @param inputWidth Input width: every weight fills inputWidth^2 slots.
 * @param numSlots Slot count of the input ciphertext (unused by the standard encoding, kept for the shared interface).
 * @param encodeLevel Encoding level (the level of the ciphertexts the plaintexts multiply).
 */
vector<Ptext> FHEONANNStandardController::encode_shortcut_kernel(FHEONHEController& heController, vector<vector<double>>& kernelData, int inputWidth, int numSlots, int encodeLevel, const FHEONLayout& layout){
    vector<Ptext> encodedKernel;
    for (auto &outChannel : kernelData) {
        encodedKernel.push_back(heController.encode_bais_input(outChannel, inputWidth*inputWidth, encodeLevel));
    }
    return encodedKernel;
}

/**
 * @brief Encode the weights of he_linear_optimized (one plaintext per output).
 * @param layout Optional layout arguments (FHEONLayout): only weightScale is used, multiplied into the weights (e.g. a factor a preceding pooling expects in the next layer); the other fields are for FHEONANNDiagonalController.
 * @param heController Controller holding the encoders.
 * @param weightData Weights [output][input].
 * @param numSlots Slot count of the input ciphertext (unused by the standard encoding, kept for the shared interface).
 * @param encodeLevel Encoding level (the level of the ciphertexts the plaintexts multiply).
 */
vector<Ptext> FHEONANNStandardController::encode_linear_weights(FHEONHEController& heController, vector<vector<double>>& weightData, int numSlots, int encodeLevel, const FHEONLayout& layout){
    vector<Ptext> encodedWeights;
    for (auto row : weightData) {
        for (auto &value : row) { value *= layout.weightScale; }
        encodedWeights.push_back(heController.encode_input(row, nextPowerOf2(row.size()), encodeLevel));
    }
    return encodedWeights;
}

/**
 * @brief Encode the bias of he_linear_optimized (output k in slot k).
 * @param layout Optional diagonal-layout arguments (FHEONLayout); unused by the standard controller, which only keeps the interface shared with FHEONANNDiagonalController.
 * @param heController Controller holding the encoders.
 * @param biasData One bias per output.
 * @param numSlots Slot count of the input ciphertext (unused by the standard encoding, kept for the shared interface).
 * @param encodeLevel Encoding level (the level of the ciphertexts the plaintexts multiply).
 */
Ptext FHEONANNStandardController::encode_linear_bias(FHEONHEController& heController, vector<double>& biasData, int numSlots, int encodeLevel, const FHEONLayout& layout){
    return heController.encode_input(biasData, nextPowerOf2(biasData.size()), encodeLevel);
}
