
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

#ifndef FHEON_ANNSTANDARDCONTROLLER_H
#define FHEON_ANNSTANDARDCONTROLLER_H

#include <openfhe.h>
#include <thread>
#include <functional>

#include "FHEONHEController.h"
#include "Utils.h"
#include "UtilsData.h"

using namespace lbcrypto;
using namespace std;

/** he_FHEONANNController defined utils */
using namespace utils;
using namespace utilsdata;


/**
 * Optional arguments of the shared ANN interface that only the diagonal primitives use (the standard controller
 * ignores them). Default values mean "derive from the other arguments".
 */
struct FHEONLayout {
    int rowStride = 0;                    /**< strided ("valid") layout: slots between two rows; 0 = compact diagonal layout */
    int colStride = 1;                    /**< strided layout: slots between two columns */
    int channelStride = 0;                /**< strided layout: slots between two input channels / output groups (0 = outputWidth * rowStride) */
    int numGroups = 0;                    /**< strided layout: output groups (0 = outputChannels) */
    vector<int> subShifts = {0};          /**< strided layout: offset of each channel inside its group */
    int inputSlots = 0;                   /**< slot count of the input ciphertext when it cannot be derived (linear keys, ...) */
    int outputSlots = 0;                  /**< slot count of a pooling output (0 = a quarter of the input slots) */
    int channelGroup = 0;                 /**< channels compacted together by a stride-2 downsampling (0 = default) */
    double weightScale = 1.0;             /**< factor multiplied into the encoded weights (e.g. the 1/4 of a sum pooling) */
    vector<vector<int>> inputPositions;   /**< linear layer: slots holding every input (empty = input i in slot i) */
};

/**
 * @brief Standard ANN controller: the ANN layers (convolution, average pooling, linear, ReLU) as originally
 * defined in FHEON. FHEONANNDiagonalController offers the same interface with the diagonal primitives,
 * so a model written against the shared interface (the primitives below and the encode_* /
 * convolution_kernel_level helpers) switches implementation by changing only the controller type.
 */
class FHEONANNStandardController{

protected:
    CryptoContext<DCRTPoly> context;

public:
    string public_data = "sskeys";
    int num_slots = 1 << 14;
    
    FHEONANNStandardController(CryptoContext<DCRTPoly>& ctx) : context(ctx) {}
    void setContext(CryptoContext<DCRTPoly>& in_context);
    void setNumSlots(int numSlots){
        num_slots = 1<< numSlots;
    }
   
    /****************************************************************************************************************
     * Shared interface: FHEONANNStandardController and FHEONANNDiagonalController declare exactly these primitives,
     * so a model switches implementation by changing only the controller type. The trailing FHEONLayout carries what
     * only the diagonal primitives need (strided layouts, slot counts, fused pooling); the standard controller
     * ignores it. Weights are encoded with the encode_* methods of the same controller.
     ****************************************************************************************************************/
    vector<int> generate_convolution_rotation_positions(int inputWidth, int inputChannels, int outputChannels,
                                                    int kernelWidth, int padding, int stride, const FHEONLayout& layout = FHEONLayout());
    vector<int> generate_linear_rotation_positions(int maxFCLayeroutputs, int rotationPosition, const FHEONLayout& layout = FHEONLayout());
    vector<int> generate_avgpool_rotation_positions(int inputWidth, int kernelWidth, int stride, int inputChannels, const FHEONLayout& layout = FHEONLayout());
    vector<int> generate_optimized_convolution_rotation_positions(int inputWidth,  int inputChannels,
                                            int outputChannels, int stride = 1, string stridingType="multi_channels", const FHEONLayout& layout = FHEONLayout());
    vector<int> generate_avgpool_optimized_rotation_positions(int inputWidth,  int inputChannels,
                                            int kernelWidth, int stride, bool globalPooling=false, string stridingType="multi_channels", int rotationIndex=16,
                                            const FHEONLayout& layout = FHEONLayout());
    /** rotations of EvalSum (he_linear, he_linear_optimized): replace the full-batch EvalSum keys */
    vector<int> generate_sum_rotation_positions(int sumSize);

    Ctext he_convolution(Ctext& encryptedInput, vector<vector<Ptext>>& kernelData, Ptext& biasInput,
                            int inputWidth, int inputChannels, int outputChannels, int kernelWidth, int padding=0, int stride=1,
                            const FHEONLayout& layout = FHEONLayout());
    Ctext he_convolution_advanced(Ctext& encryptedInput, vector<vector<Ptext>>& kernelData, Ptext& biasInput,
                            int inputWidth, int inputChannels, int outputChannels, int kernelWidth, int padding, int stride,
                            const FHEONLayout& layout = FHEONLayout());
    Ctext he_convolution_optimized(Ctext& encryptedInput,  vector<vector<Ptext>>& kernelData, Ptext& biasInput,
                            int inputWidth, int inputChannels, int outputChannels, int stride = 1, int index=0, const FHEONLayout& layout = FHEONLayout());
    Ctext he_convolution_optimized_with_multiple_channels(Ctext& encryptedInput, vector<vector<Ptext>>& kernelData, Ptext& biasInput,
                            int inputWidth,  int inputChannels, int outputChannels, const FHEONLayout& layout = FHEONLayout());
    Ctext he_shortcut_convolution(Ctext& encryptedInput,  vector<Ptext>& kernelData, Ptext& biasInput,
                            int inputWith, int inputChannels, int outputChannels, const FHEONLayout& layout = FHEONLayout());
    vector<Ctext> he_convolution_and_shortcut_optimized(const Ctext& encryptedInput, const vector<vector<Ptext>>& kernelData,
                            const vector<Ptext>& shortcutKernelData, Ptext& biasVector,  Ptext& shortcutBiasVector,
                            int inputWidth, int inputChannels, int outputChannels, const FHEONLayout& layout = FHEONLayout());
    vector<Ctext>  he_convolution_and_shortcut_optimized_with_multiple_channels(const Ctext& encryptedInput, const vector<vector<Ptext>>& kernelData,
                            const vector<Ptext>& shortcutKernelData, Ptext& biasInput, Ptext& shortcutBiasInput,
                            int inputWidth,  int inputChannels, int outputChannels, const FHEONLayout& layout = FHEONLayout());

    /** the same convolutions with the weights of one output channel produced on demand (kernelAt(outCh): its 9 kernel
     *  plaintexts, shortcutAt(outCh): its shortcut plaintext): only the plaintexts of the output channel being computed
     *  are kept in memory instead of those of the whole layer */
    using KernelSource = std::function<vector<Ptext>(int)>;
    using ShortcutSource = std::function<Ptext(int)>;
    Ctext he_convolution_optimized(Ctext& encryptedInput, const KernelSource& kernelAt, Ptext& biasInput,
                            int inputWidth, int inputChannels, int outputChannels, int stride = 1);
    vector<Ctext> he_convolution_and_shortcut_optimized(const Ctext& encryptedInput, const KernelSource& kernelAt,
                            const ShortcutSource& shortcutAt, Ptext& biasVector, Ptext& shortcutBiasVector,
                            int inputWidth, int inputChannels, int outputChannels);
    vector<Ctext> he_convolution_and_shortcut_optimized_with_multiple_channels(const Ctext& encryptedInput, const KernelSource& kernelAt,
                            const ShortcutSource& shortcutAt, Ptext& biasInput, Ptext& shortcutBiasInput,
                            int inputWidth, int inputChannels, int outputChannels);

    Ctext he_avgpool(Ctext encryptedInput, int imgCols, int outputChannels, int kernelWidth=2, int stride=2, const FHEONLayout& layout = FHEONLayout());
    Ctext he_avgpool_advanced(Ctext encryptedInput, int inputWidth, int outputChannels, int kernelWidth, int stride, int padding,
                            const FHEONLayout& layout = FHEONLayout());
    Ctext he_avgpool_optimzed(Ctext& encryptedInput,  int inputWidth, int outputChannels, int kernelWidth, int stride, const FHEONLayout& layout = FHEONLayout());
    Ctext he_avgpool_optimzed_with_multiple_channels(Ctext& encryptedInput,  int inputWidth, int inputChannels, int kernelWidth, int stride,
                            const FHEONLayout& layout = FHEONLayout());
    Ctext he_globalavgpool(Ctext& encryptedInput, int inputWidth, int outputChannels, int kernelWidth, int rotatePositions, const FHEONLayout& layout = FHEONLayout());

    Ctext he_linear(Ctext& encryptedInput, vector<Ptext>& weightMatrix, Ptext& biasInput, int inputSize, int outputSize, int rotatePositions,
                            const FHEONLayout& layout = FHEONLayout());
    Ctext he_linear_optimized(Ctext& encryptedInput, vector<Ptext>& weightMatrix, Ptext& biasInput, int inputSize, int outputSize,
                            const FHEONLayout& layout = FHEONLayout());

    Ctext he_relu(Ctext& encryptedInput, double scale, int vectorSize, int polyDegree = 59, bool normalizedOutput = false);
    Ctext he_sum_two_ciphertexts(Ctext& firstInput, Ctext& secondInput);

    /** weight encoding for the primitives above (layout.weightScale multiplies the weights, layout.inputPositions places the linear inputs) */
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
    Ctext downsample_with_multiple_channels(const Ctext& input, int inputWidth, int stride, int numChannels);
    
protected:
    /** helpers of the primitives, also used by FHEONANNStandardThreadedController */
    Ctext basic_striding(Ctext in_cipher, int inputWidth, int widthOut,  int stride);
    Ctext downsample(const Ctext& input, int inputWidth, int stride);

    Ctext batch_convolution_operation(const vector<Ctext>& rotatedInputs, const vector<Ptext>& kernelData, int kernelWidth, int inputSize,  int inputChannels);

    Ptext first_mask(int width, int inputSize, int stride, int level);
    Ptext first_mask_with_channels(int width, int inputSize, int stride, int numChannels, int level);
    
    Ptext generate_binary_mask(int pattern, int inputSize, int stride, int level);
    Ptext generate_binary_mask_with_channels(int pattern, int inputSize, int stride, int numChannels, int level);
   
    Ptext generate_row_mask(int row, int width, int inputSize, int stride, int level);
    Ptext generate_row_mask_with_channels(int row, int width, int inputSize, int stride, int numChannels,int level);

    Ptext generate_zero_mask(int size, int level);
    Ptext generate_zero_mask_channels(int size, int numChannels, int level);
    Ptext generate_channel_full_mask(int n, int in_elements, int out_elements, int numChannels, int level);
    Ptext generate_channel_mask_with_zeros(int channel, int outputSize, int numChannels, int level);

};

/** the controller of the single-input models (sequential) */
using ANNStandardController = FHEONANNStandardController;

#endif // FHEON_ANNSTANDARDCONTROLLER_H
