
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

/*******************************************************************************************************************
 * This ANN controller is used to define all ANN layers used in this project
 * such as; Convolution, avgpool, fclinear
 *******************************************************************************************************************/

#ifndef FHEON_ANNCONCROLLER_H
#define FHEON_ANNCONCROLLER_H

#include <openfhe.h>
#include <thread>

#include "FHEONHEController.h"

#include "Utils.h"
#include "UtilsData.h"

using namespace lbcrypto;
using namespace std;

/** he_FHEONANNController defined utils */
using namespace utils;
using namespace utilsdata;

class FHEONANNController {

private:
  CryptoContext<DCRTPoly> context;

public:
  string public_data = "sskeys";
  int num_slots = 1 << 14;

  FHEONANNController(CryptoContext<DCRTPoly> &ctx) : context(ctx) {}
  void setContext(CryptoContext<DCRTPoly> &in_context);
  void setNumSlots(int numSlots) { num_slots = 1 << numSlots; }

  vector<int> generate_convolution_rotation_positions(int inputWidth,
                                                      int inputChannels,
                                                      int outputChannels,
                                                      int kernelWidth,
                                                      int padding, int Stride);
  vector<int> generate_linear_rotation_positions(int maxFCLayeroutputs,
                                                 int rotationPosition);
  vector<int> generate_avgpool_rotation_positions(int inputWidth,
                                                  int kernelWidth, int Stride,
                                                  int inputChannels);

  vector<int> generate_optimized_convolution_rotation_positions(
      int inputWidth, int inputChannels, int outputChannels, int Stride = 1,
      string stridingType = "multi_channels");
  vector<int> generate_avgpool_optimized_rotation_positions(
      int inputWidth, int inputChannels, int kernelWidth, int Stride,
      bool globalPooling = false, string stridingType = "multi_channels",
      int rotationIndex = 16);

  Ctext he_convolution(Ctext &encryptedInput, vector<vector<Ptext>> &kernelData,
                       Ptext &biasInput, int inputWidth, int inputChannels,
                       int outputChannels, int kernelWidth, int padding = 0,
                       int stride = 1);
  Ctext he_convolution_advanced(Ctext &encryptedInput,
                                vector<vector<Ptext>> &kernelData,
                                Ptext &biasInput, int inputWidth,
                                int inputChannels, int outputChannels,
                                int kernelWidth, int padding, int stride);
  Ctext he_convolution_optimized(Ctext &encryptedInput,
                                 vector<vector<Ptext>> &kernelData,
                                 Ptext &biasInput, int inputWidth,
                                 int inputChannels, int outputChannels,
                                 int Stride = 1, int index = 0);
  Ctext he_convolution_optimized_with_multiple_channels(
      Ctext &encryptedInput, vector<vector<Ptext>> &kernelData,
      Ptext &biasInput, int inputWidth, int inputChannels, int outputChannels);
  Ctext he_shortcut_convolution(Ctext &encryptedInput,
                                vector<Ptext> &kernelData, Ptext &biasInput,
                                int inputWith, int inputChannels,
                                int outputChannels);
  vector<Ctext> he_convolution_and_shortcut_optimized(
      const Ctext &encryptedInput, const vector<vector<Ptext>> &kernelData,
      const vector<Ptext> &shortcutKernelData, Ptext &biasVector,
      Ptext &shortcutBiasVector, int inputWidth, int inputChannels,
      int outputChannels);
  vector<Ctext> he_convolution_and_shortcut_optimized_with_multiple_channels(
      const Ctext &encryptedInput, const vector<vector<Ptext>> &kernelData,
      const vector<Ptext> &shortcutKernelData, Ptext &biasInput,
      Ptext &shortcutBiasInput, int inputWidth, int inputChannels,
      int outputChannels);

  Ctext he_avgpool(Ctext encryptedInput, int imgCols, int outputChannels,
                   int kernelWidth = 2, int Stride = 2);
  Ctext he_avgpool_advanced(Ctext encryptedInput, int inputWidth,
                            int outputChannels, int kernelWidth, int stride,
                            int padding);
  Ctext he_avgpool_optimzed(Ctext &encryptedInput, int inputWidth,
                            int outputChannels, int kernelWidth, int Stride);
  Ctext he_avgpool_optimzed_with_multiple_channels(Ctext &encryptedInput,
                                                   int inputWidth,
                                                   int inputChannels,
                                                   int kernelWidth, int Stride);
  Ctext he_globalavgpool(Ctext &encryptedInput, int inputWidth,
                         int outputChannels, int kernelWidth,
                         int rotatePositions);

  Ctext he_linear(Ctext &encryptedInput, vector<Ptext> &weightMatrix,
                  Ptext &biasInput, int inputSize, int outputSize,
                  int rotatePositions);
  Ctext he_linear_optimized(Ctext &encryptedInput, vector<Ptext> &weightMatrix,
                            Ptext &biasInput, int inputSize, int outputSize);

  Ctext he_relu(Ctext &encryptedInput, double scale, int vectorSize,
                int polyDegree = 59);
  Ctext he_sum_two_ciphertexts(Ctext &firstInput, Ctext &secondInput);

private:
  Ctext basic_striding(Ctext in_cipher, int inputWidth, int widthOut,
                       int Stride);
  Ctext downsample(const Ctext &input, int inputWidth, int stride);
  Ctext downsample_with_multiple_channels(const Ctext &input, int inputWidth,
                                          int stride, int numChannels);
  Ctext batch_convolution_operation(const vector<Ctext> &rotatedInputs,
                                    const vector<Ptext> &kernelData,
                                    int kernelWidth, int inputSize,
                                    int inputChannels);

  Ptext first_mask(int width, int inputSize, int stride, int level);
  Ptext first_mask_with_channels(int width, int inputSize, int stride,
                                 int numChannels, int level);

  Ptext generate_binary_mask(int pattern, int inputSize, int stride, int level);
  Ptext generate_binary_mask_with_channels(int pattern, int inputSize,
                                           int stride, int numChannels,
                                           int level);

  Ptext generate_row_mask(int row, int width, int inputSize, int stride,
                          int level);
  Ptext generate_row_mask_with_channels(int row, int width, int inputSize,
                                        int stride, int numChannels, int level);

  Ptext generate_zero_mask(int size, int level);
  Ptext generate_zero_mask_channels(int size, int numChannels, int level);
  Ptext generate_channel_full_mask(int n, int in_elements, int out_elements,
                                   int numChannels, int level);
  Ptext generate_channel_mask_with_zeros(int channel, int outputSize,
                                         int numChannels, int level);
};

#endif // FHEON_ANNCONCROLLER_H