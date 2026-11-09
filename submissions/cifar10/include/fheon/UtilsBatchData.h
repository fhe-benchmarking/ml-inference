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
 * @file UtilsBatchData.h
 * @brief Manage weights and baises for batch neural networks
 *
 * This file provides functions to handle data preparation tasks, including
 * generating random datasets, reading data from files for different datasets, and organizing it for use
 * in HE-friendly neural networks.
 */

#ifndef FHEON_DATABATCHUTILS_H
#define FHEON_DATABATCHUTILS_H

#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <thread>
#include <functional>
#include <openfhe.h>
#include "Utils.h"

using namespace std;
using namespace std::chrono;
using namespace lbcrypto;

namespace utilsbatchdata {

    /*********************************************************************************************************************
     * Batches of any size: a batch program evaluates exactly the images it is given (--index-value, --test-size, or a
     * packed selection of any images). They are split into batches of the program's batch size; the last batch is
     * completed with blank (zero) images, which are evaluated but never reported.
     *********************************************************************************************************************/

    /** number of batches of batchSize images needed for numImages images */
    static inline int num_batches(int numImages, int batchSize) {
        return (numImages + batchSize - 1) / batchSize;
    }

    /**
     * @brief The batchSize images of a batch: images first, first + 1, ... of imagesData, blank from endImage on.
     *
     * @param imagesData Images read from the dataset (at least endImage of them).
     * @param first Index in imagesData of the first image of the batch.
     * @param batchSize Images per batch.
     * @param endImage One past the last image to evaluate: the positions from there on are blank.
     */
    static inline vector<vector<double>> batch_images(const vector<vector<double>>& imagesData, int first, int batchSize, int endImage) {
        size_t imageSize = imagesData.empty() ? 0 : imagesData[0].size();
        vector<vector<double>> batch;
        batch.reserve(batchSize);
        for (int b = 0; b < batchSize; b++) {
            int index = first + b;
            if (index < endImage && index < static_cast<int>(imagesData.size())) batch.push_back(imagesData[index]);
            else batch.push_back(vector<double>(imageSize, 0.0));
        }
        return batch;
    }

    /**
     * @brief Report the predictions of the real images of a batch.
     *
     * For i < numImages, the label of image firstIndex + i (dataset index, from 0) is the argmax of logitAt(i, k),
     * k < numClasses. Prints the lines of read_batch_inferenced_label ("Batch <index> -> Predicted Label: ..." and the
     * logits at verbose level 2), records the accuracy against the true labels and appends the labels to
     * predictions_file. Blank padding images are never passed here.
     *
     * @return The predicted labels.
     */
    static inline vector<int> report_batch_predictions(int firstIndex, int numImages, int numClasses,
                                                       const std::function<double(int, int)>& logitAt, const string& predictions_file) {
        vector<int> labels;
        ofstream predictionsOut;
        if (!predictions_file.empty()) predictionsOut.open(predictions_file, ios_base::app);
        vector<double> logits(numClasses);
        for (int i = 0; i < numImages; i++) {
            for (int k = 0; k < numClasses; k++) logits[k] = logitAt(i, k);
            int label = distance(logits.begin(), max_element(logits.begin(), logits.end()));
            labels.push_back(label);
            int index = firstIndex + i;
            if (utils::verbose_level == utils::level_zero) {
                cout << "Batch " << index << " -> Predicted Label: " << label << endl;
            } else {
                cout << "Batch " << index << " -> Predicted Label: " << label << " (Score: " << logits[label] << ")" << endl;
            }
            if (utils::verbose_level >= utils::level_two) {
                cout << "Weights for Batch " << index << ": [ ";
                for (double value : logits) cout << value << " ";
                cout << "]" << endl;
            }
            utils::recordPrediction(label);
            if (predictionsOut.is_open()) predictionsOut << label << endl;
        }
        return labels;
    }

    /**
     * @brief Load numeric data from a CSV file.
     * 
     * Reads a CSV file and converts each value into double. 
     * Invalid values are replaced with 0.0.
     *
     * @param fileName Path to the CSV file.
     * @return 2D vector of doubles with CSV contents.
     */
    static inline vector<vector<double>> loadCSV(const string& fileName) {
        std::vector<std::vector<double>> data;
        std::ifstream file(fileName);
        
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << fileName << std::endl;
            return data;
        }

        std::string line;
        while (std::getline(file, line)) {
            std::vector<double> row;
            std::stringstream ss(line);
            std::string cell;
            while (std::getline(ss, cell, ',')) {
                try {
                    row.push_back(std::stod(cell));
                } catch (const std::invalid_argument& e) {
                    std::cerr << "Invalid number: " << cell << std::endl;
                    row.push_back(0.0);
                }
            }
            data.push_back(row);
        }
        file.close();
        return data;
    }

    /**
     * @brief Flatten a 2D vector into a 1D vector.
     * 
     * @param data 2D vector to flatten.
     * @return 1D vector containing all elements in row-major order.
     */
    static inline vector<double> flatten_csv_data(const vector<vector<double>>& data) {
        vector<double> flattened;
        for (const auto& row : data) {
            flattened.insert(flattened.end(), row.begin(), row.end());
        }
        return flattened;
    }

    /**
     * @brief Load and replicate bias values for batched processing.
     *
     * Reads the first row of a CSV file as bias values and repeats each bias 
     * across all input positions for each batch. The final shape is 
     * [outputChannel][batch * inputSize], ready to be used in batched FHE operations.
     *
     * @param fileName       Path to the CSV file containing bias values.
     * @param outputChannels Number of output channels.
     * @param batchSize      Number of batches.
     * @param inputSize      Number of input positions per batch (input channels * feature map size).
     *
     * @return std::vector<std::vector<double>> 
     *         2D vector of shape [outputChannel][batchSize * inputSize].
     *
     * @throws std::runtime_error if the CSV is empty or does not contain enough bias values.
     */
    static inline vector<vector<double>> load_batch_bias(const string& fileName, int outputChannels,
        int batchSize, int inputSize) {
            
        vector<vector<double>> data = loadCSV(fileName);
        if (data.empty()) {
            throw runtime_error("CSV file is empty or failed to load.");
        }

        vector<double> raw_bias = flatten_csv_data(data);
        if (raw_bias.size() < static_cast<size_t>(outputChannels)) {
            throw runtime_error("Not enough bias values in CSV for the requested output channels.");
        }

        // Final output: [outputChannel][batchSize * inputSize]
        int totalSize = batchSize * inputSize;
        vector<vector<double>> reshapedBias(outputChannels, vector<double>(totalSize));

        for (int outCh = 0; outCh < outputChannels; outCh++) {
            double biasValue = raw_bias[outCh];
            if( abs(biasValue) < 1e-10){
                biasValue = 0.0;
            }
            fill(reshapedBias[outCh].begin(), reshapedBias[outCh].end(), biasValue);
        }

        return reshapedBias;
    }

    /**
     * @brief Load and reshape convolution kernel weights from a CSV file for batched processing.
     *
     * This function reads kernel weights from a CSV file and organizes them into a 5D vector
     * suitable for homomorphic batched convolution operations. The weights are organized
     * as [outputChannel][inputChannel][batch][rows][cols]. For each output channel and input
     * channel, the kernel values are repeated across the batch dimension.
     *
     * @param fileName       Path to the CSV file containing the kernel weights. The CSV should
     *                       contain all weights flattened in row-major order.
     * @param outputChannels Number of output channels (kernels).
     * @param inputChannels  Number of input channels.
     * @param batchSize      Number of batches; each input channel kernel is repeated this many times.
     * @param rowsWidth      Number of rows in each kernel.
     * @param imgCols        Number of columns in each kernel.
     *
     * @return std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>> 
     *         5D vector of weights with shape [outputChannel][inputChannel][batch][rows][cols].
     *
     * @note For each output channel, the function reads input channel weights from the CSV
     *       exactly once and repeats them across the batch dimension. This layout ensures
     *       compatibility with batched homomorphic convolution operations.
     *
     * @throws std::runtime_error if the CSV does not contain enough weights to fill the requested shape.
     */
    static inline vector<vector<vector<vector<vector<double>>>>> load_batch_weights(const string& fileName, int outputChannels,
        int inputChannels, int batchSize, int rowsWidth, int imgCols) {

        // Load CSV data
        vector<vector<double>> data = loadCSV(fileName);
        if (data.empty()) {
            throw runtime_error("CSV file is empty or failed to load.");
        }

        // Flatten raw weights from CSV
        vector<double> raw_weights = flatten_csv_data(data);
        size_t indexVal = 0;

        // Reshaped data: [outputChannel][inputChannel][batch][rows][cols]
        vector<vector<vector<vector<vector<double>>>>> reshapedData(
            outputChannels,
            vector<vector<vector<vector<double>>>>(
                inputChannels,
                vector<vector<vector<double>>>(
                    batchSize,
                    vector<vector<double>>(rowsWidth, vector<double>(imgCols))
                )
            )
        );

        for (int outCh = 0; outCh < outputChannels; outCh++) {
            for (int inCh = 0; inCh < inputChannels; inCh++) {
                // Fill single input channel [rows][cols]
                vector<vector<double>> singleInputChannel(rowsWidth, vector<double>(imgCols));
                for (int r = 0; r < rowsWidth; r++) {
                    for (int c = 0; c < imgCols; c++) {
                        if (indexVal >= raw_weights.size()) {
                            cout << "raw weights size: " <<  raw_weights.size() << " -- indexVal: " << indexVal << endl; 
                            throw runtime_error("Not enough weights in CSV to fill the requested shape.");
                        }
                        double cellVal = raw_weights[indexVal++];
                        if( abs(cellVal) < 1e-10){
                            cellVal = 0.0;
                        }
                        singleInputChannel[r][c] = cellVal;
                    }
                }

                // Repeat this singleInputChannel across batchSize
                for (int b = 0; b < batchSize; b++) {
                    reshapedData[outCh][inCh][b] = singleInputChannel;
                }
            }
        }

        return reshapedData;
    }


    /**
     * @brief Load shortcut batch weights and expand them for batching.
     *
     * Reads weights from a CSV file, then for each weight value corresponding to 
     * [outputChannel][inputChannel], repeats it batchSize * inputSize times. 
     * This produces a structure suitable for batched homomorphic operations.
     *
     * @param fileName       Path to the CSV file containing weights.
     * @param batchSize      Number of batches.
     * @param outputChannels Number of output channels.
     * @param inputChannels  Number of input channels.
     * @param inputSize      Size of each input vector.
     *
     * @return 3D vector of size [outputChannels][inputChannels][batchSize*inputSize].
     */
    static inline vector<vector<vector<double>>> load_shortcut_batch_weights(string fileName,  int batchSize,
        int outputChannels, int inputChannels, int inputSize) {

        vector<vector<double>> data = loadCSV(fileName);
        if (data.empty()) {
            throw runtime_error("CSV file is empty or failed to load.");
        }
        vector<double> raw_weights = flatten_csv_data(data);
        int totalInSize = batchSize * inputSize;

        // Allocate [outputChannels][inputChannels][totalInSize]
        vector<vector<vector<double>>> reshapedData(
            outputChannels, 
            vector<vector<double>>(inputChannels, vector<double>(totalInSize))
        );

        int indexVal = 0; 
        for (int i = 0; i < outputChannels; i++) {
            for (int j = 0; j < inputChannels; j++) {
                double cellVal = raw_weights[indexVal++];
                if( abs(cellVal) < 1e-10){
                    cellVal = 0.0;
                }
                std::fill(reshapedData[i][j].begin(), reshapedData[i][j].end(), cellVal);
            }
        }

        return reshapedData;
    }


    /**
     * @brief Load and reshape fully connected layer weights for batched processing.
     *
     * Reads a CSV file containing FC layer weights and organizes them into a 2D vector.
     * Each output channel contains a single vector where each input channel weight 
     * is repeated `batchSize` times consecutively.
     *
     * @param fileName       Path to the CSV file containing weights (flattened row-major).
     * @param outputChannels Number of output neurons (output channels).
     * @param batchSize      Number of batches; each input channel weight is repeated this many times.
     * @param inputChannels  Number of input features (input channels).
     *
     * @return std::vector<std::vector<double>> 
     *         2D vector of shape [outputChannel][inputChannel * batchSize].
     *
     * @throws std::runtime_error if CSV is empty or does not contain enough weights.
     */
    static inline vector<vector<double>> load_batch_fc_weights(const string& fileName, int outputChannels,
        int batchSize, int inputChannels) {

        vector<vector<double>> data = loadCSV(fileName);
        if (data.empty()) {
            throw runtime_error("CSV file is empty or failed to load.");
        }
        vector<double> raw_weights = flatten_csv_data(data);
        if (raw_weights.size() < static_cast<size_t>(outputChannels * inputChannels)) {
            throw runtime_error("Not enough weights in CSV for requested dimensions.");
        }

        vector<vector<double>> reshapedData(outputChannels, vector<double>(inputChannels * batchSize));
        int indexVal = 0;

        for (int outCh = 0; outCh < outputChannels; outCh++) {
            // take one vector of inputChannels weights
            vector<double> singleInputChannel(inputChannels);
            for (int inCh = 0; inCh < inputChannels; inCh++) {
                double cellVal =  raw_weights[indexVal++];
                if( abs(cellVal) < 1e-10){
                    cellVal = 0.0;
                }
                singleInputChannel[inCh] = cellVal;
            }

            // repeat this vector batchSize times
            for (int b = 0; b < batchSize; b++) {
                std::copy(singleInputChannel.begin(), singleInputChannel.end(),
                        reshapedData[outCh].begin() + b * inputChannels);
            }
        }

    return reshapedData;
}


    /**
     * @brief Load and expand fully connected (FC) layer biases for batched inputs.
     *
     * Reads a CSV containing bias values (one per output channel), and expands each bias
     * across the batch dimension so it can be added after matrix multiplication.
     *
     * @param fileName       Path to the CSV file containing bias values (1 row, outputChannels cols).
     * @param outputChannels Number of output channels (neurons).
     * @param batchSize      Number of inputs in the batch.
     *
     * @return Flattened vector<double> of size batchSize * outputChannels,
     *         ordered as [batch0: all biases][batch1: all biases]...
     */
    static inline vector<double> load_batch_fc_bias(const string& fileName,
        int outputChannels, int batchSize) {
    
        vector<vector<double>> data = loadCSV(fileName);
        if (data.empty()) {
            throw runtime_error("CSV file is empty or failed to load bias file.");
        }

        // Take the first row as the bias values
        vector<double> singleOutputChannels = flatten_csv_data(data);
        if (singleOutputChannels.size() < static_cast<size_t>(outputChannels)) {
            throw runtime_error("Not enough bias values in CSV for requested output channels.");
        }

        // Clean small values (treat near-zero as 0.0)
        for (double& val : singleOutputChannels) {
            if (abs(val) < 1e-10) {
                val = 0.0;
            }
        }

        // Allocate full bias (batchSize * outputChannels)
        vector<double> fullData(batchSize * outputChannels);

        // Repeat bias vector batchSize times
        for (int b = 0; b < batchSize; b++) {
            std::copy(singleOutputChannels.begin(),
                    singleOutputChannels.begin() + outputChannels,
                    fullData.begin() + b * outputChannels);
        }
        // cout << "Size of channels: " << fullData.size() << endl; 
        // cout << "Full Data: " << fullData << endl; 

        return fullData;
    }

    /**
     * @brief Reshape batched input data into channel-major format.
     *
     * Converts input data from [batch][inputChannels * inputSize] format into
     * [inputChannels][batchSize * inputSize], where each channel contains
     * all batches concatenated.
     *
     * @param inputDatas     Input data shaped as [batchSize][inputChannels * inputSize].
     * @param batchSize      Number of batches.
     * @param inputChannels  Number of input channels.
     * @param inputSize      Size of one input per channel (e.g., H*W).
     *
     * @return Reshaped data as [inputChannels][batchSize * inputSize].
     */
    static inline vector<vector<double>> convert_inputData(const vector<vector<double>>& inputDatas, 
        int batchSize, int inputChannels, int inputSize) {

        vector<vector<double>> reshapedData(inputChannels, vector<double>(batchSize * inputSize));

        for (int ch = 0; ch < inputChannels; ch++) {
            for (int b = 0; b < batchSize; b++) {
                // Offset in the input for this batch and channel
                int inputOffset = ch * inputSize;
                int outputOffset = b * inputSize;

                // Copy inputSize elements into correct spot
                for (int i = 0; i < inputSize; i++) {

                    double cellVal =  inputDatas[b][inputOffset + i];
                    if( abs(cellVal) < 1e-10){
                        cellVal = 0.0;
                    }
                    reshapedData[ch][outputOffset + i] = cellVal;
                }
            }
        }
        return reshapedData;
    }

    static inline vector<int> limit_to_physical_cores()
    {
        // Physical cores on Intel i9-14900K (Ubuntu)
        // const int physical_cores[] = {
        //     0,2,4,6,8,10,12,14,      // P-cores (1 thread each)
        //     16,17,18,19,20,21,22,23, // E-cores
        //     24,25,26,27,28,29,30,31  // E-cores
        // };

        const int physical_cores[] = {
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14,
            15, 16, 17, 18, 19, 20, 
            21, 22, 23
        };

        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);

        // Add only physical-core CPU IDs
        for (int cpu : physical_cores)
            CPU_SET(cpu, &cpuset);

        // Apply to the whole process (PID = 0)
        if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0){
            throw std::runtime_error("sched_setaffinity failed");
        }

        // Return vector of CPU IDs so threads can use it for affinity
        return std::vector<int>(std::begin(physical_cores), std::end(physical_cores));
    }


    // ---------------- reusable thread runner ----------------
    template<typename Func>
    void run_multithreaded(int total_work, const std::vector<int>& physical_cores, Func&& worker_func)
    {
        int numThreads = std::min(total_work, (int)physical_cores.size());
        std::vector<std::thread> threads(numThreads);

        int block = (total_work + numThreads - 1) / numThreads;

        for (int t = 0; t < numThreads; ++t) {
            int start = t * block;
            int end = std::min(start + block, total_work);
            if (start < end) {
                threads[t] = std::thread([&, start, end, t]() {
                    cpu_set_t cpuset;
                    CPU_ZERO(&cpuset);
                    CPU_SET(physical_cores[t], &cpuset);
                    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

                    worker_func(start, end);
                });
            }
        }

        for (auto &th : threads)
            if (th.joinable()) th.join();
    }

}

#endif //FHEON_DATABATCHUTILS_H