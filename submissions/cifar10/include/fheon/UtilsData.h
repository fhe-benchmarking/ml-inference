
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
 * @file UtilsData.h
 * @brief Manage data generation, reading from files, and arranging for use in networks.
 *
 * This file provides functions to handle data preparation tasks, including
 * generating random datasets, reading data from files for different datasets, and organizing it for use
 * in HE-friendly neural networks.
 */

#ifndef FHEON_DATAUTILS_H
#define FHEON_DATAUTILS_H

#include <iostream>
#include <cmath>
#include <openfhe.h>
#include "Utils.h"

using namespace std;
using namespace std::chrono;
using namespace lbcrypto;

namespace utilsdata {

    /**
     * @brief Print a 1D vector of doubles.
     * 
     * @param vecData Vector to print.
     */
    static inline void printVector(vector<double> &vecData) {
        cout << vecData <<endl;
        cout << endl;
    }

    /**
     * @brief Print a 2D matrix of doubles.
     * 
     * @param matrix2D Matrix to print.
     */
    static inline void print2DMatrix(vector<vector<double>> matrix2D){
        int rows = matrix2D.size();
        int cols = matrix2D[0].size();
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                cout << matrix2D[i][j] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }

    /**
     * @brief Print a 3D matrix of doubles.
     * 
     * @param matrix3D Matrix to print.
     */
    static inline void print3DMatrix(vector<vector<vector<double>>> matrix3D){
        int depth = matrix3D.size();
        int rows = matrix3D[0].size();
        int cols = matrix3D[0][0].size();
        for (int d = 0; d < depth; ++d) {
            cout << "Depth " << d << ":\n";
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    cout << matrix3D[d][i][j] << " ";
                }
               cout << endl;
            }
            cout << endl;
        }
    }
    /**
     * @brief Create a 1D vector with random values.
     * 
     * @param cols Number of elements.
     * @param minValue Minimum random value.
     * @param maxValue Maximum random value.
     * @return Random vector of doubles.
     */
    static inline vector<double> createVector(int cols, int minValue, int maxValue) {
        // Initialize random seed
        static bool seedInitialized = false;
        if (!seedInitialized) {
            srand(static_cast<unsigned int>(time(nullptr)));
            seedInitialized = true;
        }

        vector<double> vectorData(cols);
        for (int i = 0; i < cols; i++) {
            vectorData[i] = minValue + static_cast<double>(rand()) / RAND_MAX * (maxValue - minValue);
        }
        return vectorData;
    }

    /**
     * @brief Create a 2D matrix with random values.
     * 
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @param minValue Minimum random value.
     * @param maxValue Maximum random value.
     * @return Random 2D matrix.
     */
    static inline vector<vector<double>> create2DMatrix(int rows, int cols, int minValue, int maxValue) {
        vector<vector<double>> matrix2D;
        matrix2D.reserve(rows); // Reserve space to avoid multiple allocations

        for (int i = 0; i < rows; i++) {
            matrix2D.push_back(createVector(cols, minValue, maxValue));
        }
        return matrix2D;
    }
    
    /**
     * @brief Create a 3D matrix with random values.
     * 
     * @param depth Depth of matrix.
     * @param rows Number of rows per slice.
     * @param cols Number of columns per slice.
     * @param minValue Minimum random value.
     * @param maxValue Maximum random value.
     * @return Random 3D matrix.
     */
    static inline vector<vector<vector<double>>> create3DMatrix(int depth, int rows, int cols, int minValue, int maxValue) {
        vector<vector<vector<double>>> matrix3D;
        matrix3D.reserve(depth); // Reserve space to avoid multiple allocations

        for (int d = 0; d < depth; d++) {
            matrix3D.push_back(create2DMatrix(rows, cols, minValue, maxValue));
        }
        return matrix3D;
    }

    /**
     * @brief Flatten a 3D matrix into a 1D vector.
     * 
     * @param matrix3D Matrix to flatten.
     * @return Flattened vector.
     */
    static inline vector<double> flatten3DMatrix(const vector<vector<vector<double>>> matrix3D) {
        vector<double> flatVec;
        for (const auto& matrix : matrix3D) {
            for (const auto& row : matrix) {
                for (const auto& elem : row) {
                    flatVec.push_back(elem);
                }
            }
        }
        return flatVec;
    }
    
    /**
     * @brief Print a plaintext vector after decryption.
     * 
     * @param packedVec Plaintext vector to print.
     */
    static inline void printPtextVector(Plaintext packedVec) {
        if (utils::verbose_level < 1) return;
        vector<complex<double>> finalResult = packedVec->GetCKKSPackedValue();
        cout << finalResult << endl;
        cout << endl;
    }


    /**
     * @brief Generate a binary mask of ones followed by zeros.
     * 
     * @param ones_size Number of ones.
     * @param vector_size Total size of mask.
     * @return Mask vector.
    */
    static inline vector<double> generate_mixed_mask(int ones_size, int vector_size){
        vector<double> ones_vector(ones_size, 1.0);
        vector<double> zeros_vector((vector_size - ones_size), 0.0);
        ones_vector.insert(ones_vector.end(), zeros_vector.begin(), zeros_vector.end());
        return ones_vector;
    }

    static inline vector<double> generate_scale_and_clean_mask(double scale_value, int vector_size){
        vector<double> zeros_vector(vector_size, 0.0);
        zeros_vector[0] = scale_value;
        return zeros_vector;
    }

    /**
     * @brief Generate a scaled mask with uniform values.
     * 
     * @param scale_value Scaling factor.
     * @param vector_size Total size of mask.
     * @return Scaled mask vector.
     */
    static inline vector<double> generate_scale_mask(int scale_value, int vector_size){
        double scale_val = (1.0/scale_value);

        vector<double> scaled_vector(vector_size, scale_val);
        return scaled_vector;
    }

    /**
     * @brief Generate a value mask with a fixed value.
     * 
     * @param scale_value Value to assign.
     * @param vector_size Total size of mask.
     * @return Value mask vector.
     */
    static inline vector<double> generate_value_mask(double scale_value, int vector_size){

        vector<double> scaled_vector(vector_size, scale_value);
        return scaled_vector;
    }
   
    /**
     * @brief Approximate greater-than function for spiking.
     * 
     * @param x Input value.
     * @return Spike value if x > 0, else 0.
     */
    static inline int greaterFunction(double x) {
        double threshold_value = 0;
        double spike_value = 0;
        int scale_value = 10;
        if(x > threshold_value){
            spike_value = x*scale_value;
            return spike_value; 

        }
        else{
            return 0;
        }
    }


    /**
     * @brief Generate a one-hot mask.
     * 
     * @param position Index of the one-hot element.
     * @param vector_size Total size of the mask.
     * @return One-hot mask vector.
     */
    static inline vector<double> generate_one_hot_mask(int position, int vector_size){
        vector<double> one_hot_vector(vector_size, 0.0);
        if(position < vector_size){
            one_hot_vector[position] = 1.0;
        }
        return one_hot_vector;
    }

    /**
     * @brief Approximate smooth greater-than step function.
     * 
     * @param x Input value.
     * @return Smoothed spike value.
    */
    static inline double approximateGreaterFunction(double x){
        double threshold_value = 0.05;
        double steepness = 100.0;
        double spike_value = 0.5 * (1 + tanh(steepness * (x - threshold_value)));
        return spike_value; 
    }

    /**
     * @brief ReLU with scaling factor.
     * 
     * @param x Input value.
     * @param scale Scaling factor.
     * @return ReLU output.
     */
    static inline double innerRelu(double x, double scale){
        if (x < 0) return 0; else return (1 / scale) * x;
    }


    static inline double factorial(int n) {
        if (n == 0 || n == 1) return 1;
        double result = 1.0;
        for (int i = 2; i <= n; i++) {
            result *= i;
        }
        return result;
    }


    /**
     * @brief Create an average pooling filter.
     * 
     * @param kernel_width Width of pooling kernel.
     * @return Averaging filter vector.
     */
    static inline vector<double> avgpoolFilter(int kernel_width){
        int numVals = pow(kernel_width, 2);
        double scaled_value = (1.0/numVals);
        vector<double> avgpoolFilter(numVals, scaled_value);

        return avgpoolFilter;
    }
 

    /**
     * @brief Return the next power of 2 greater than or equal to n.
     *
     * If n is already a power of 2, return n. Otherwise, return the next higher power of 2.
     *
     * @param n Input integer.
     * @return Next power of 2 (>= n).
     */
    static inline unsigned int nextPowerOf2(unsigned int n) {
        if (n == 0) return 1;

        // Check if n is already a power of 2
        if ((n & (n - 1)) == 0) {
            return n;
        }

        // Otherwise, compute the next higher power of 2
        n--;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        n++;
        return n;
    }


    static inline unsigned int localLog2(unsigned int n) {
        unsigned int logValue = 0;
        while (n >>= 1) {
            logValue++;
        }
        return logValue;
    }

    /**
     * @brief Tile a ciphertext's real data across its full slot capacity so
     * that later EvalRotate/EvalFastRotation calls behave as true circular
     * rotation over exactly `elementCount` slots, instead of leaking through
     * the empty (zero) padding between `elementCount` and `totalSlots`.
     *
     * CKKS packs every ciphertext into a fixed `totalSlots`-wide ring shared
     * by the whole network, but a given layer's real data
     * (batchSize * width * width) is often smaller than that ring. Rotating
     * such a ciphertext directly wraps through true zeros near the start/end
     * of the real data instead of the intended circular neighbor (e.g. the
     * last row of the last batch image), corrupting any tap whose offset
     * would otherwise need that wraparound. Doubling the covered range
     * (add a copy of itself rotated by -coverage, repeatedly) until it fills
     * the ring fixes this: every slot in [0, elementCount) now correctly
     * reads its circular neighbor for any rotation used by 3x3-kernel
     * convolution taps. No-op (returns the input unchanged) when the ring is
     * already exactly (or not evenly divisible into) `elementCount` slots.
     *
     * @param context      CryptoContext used for the rotation/addition ops.
     * @param ciphertext   Ciphertext holding real data in its first elementCount slots.
     * @param elementCount Number of meaningful (real) slots, e.g. batchSize*width*width.
     * @param totalSlots   Total slot capacity of the ciphertext's packing (num_slots).
     * @return Ciphertext whose first elementCount slots are unchanged, with
     *         elementCount-periodic copies filling the rest of totalSlots.
     */
    static inline Ctext replicate_across_ring(CryptoContext<DCRTPoly>& context, Ctext ciphertext, int elementCount, int totalSlots) {
        int reps = totalSlots / elementCount;
        if (reps <= 1) {
            return ciphertext;
        }
        reps = static_cast<int>(nextPowerOf2(static_cast<unsigned int>(reps)));
        Ctext tiled = ciphertext;
        int coverage = elementCount;
        for (int r = 1; r < reps; r *= 2) {
            tiled = context->EvalAdd(tiled, context->EvalRotate(tiled, -coverage));
            coverage *= 2;
        }
        return tiled;
    }


    static inline double log2_approx(double x) {
        if (x <= 0.0) return 0.0;

        int k;
        double m = std::frexp(x, &k);  // x = m * 2^k, m in [0.5,1)

        m *= 2.0;
        k -= 1;

        // polynomial approximation of log2(m)
        double t = m - 1.0;

        double log2_m =
            1.442695 * (
                t
                - 0.5 * t * t
                + 0.333333 * t * t * t
            );

        return k + log2_m;
    }



    static inline double ln_approx(double x) {
        int terms = 20;
        if (x <= 0) {
            return 0;
        }

        double y = x - 1.0;
        double result = 0.0;
        double power = y;

        for (int n = 1; n <= terms; n++) {
            double term = power / n;

            if (n % 2 == 0)
                result -= term;
            else
                result += term;

            power *= y;
        }

        return result;
    }

    static inline double doerfler_ln(double x) {
        return 6.0 * (x - 1.0) /
            (x + 1.0 + 4.0 * sqrt(x));
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
     * @brief Load bias values from a CSV file.
     * 
     * Extracts the first row of the CSV file as bias values.
     *
     * @param fileName Path to the CSV file.
     * @return 1D vector containing bias values.
     */
    static inline vector<double> load_bias(string fileName){
        vector<vector<double>> data = loadCSV(fileName);
        vector<double> bias; 
        for (size_t i = 0; i< data.size(); i++) {
            
            bias = data[0];
            // cout << " bias data: "<< bias << endl;
        }
        return bias;
    }

    /**
     * @brief Load and reshape convolution weights from a CSV file.
     * 
     * Reads flat weight data and reshapes into a 4D structure:
     * [outputChannels][inputChannels][rowsWidth][imgCols].
     *
     * @param fileName Path to the CSV file.
     * @param outputChannels Number of output channels.
     * @param inputChannels Number of input channels.
     * @param rowsWidth Number of rows in each kernel.
     * @param imgCols Number of columns in each kernel.
     * @return 4D vector containing reshaped weights.
     */
    static inline vector<vector<vector<vector<double>>>> load_weights(string fileName, int outputChannels, int inputChannels, 
                int rowsWidth, int imgCols) {
        vector<vector<double>> data = loadCSV(fileName);
        vector<double> raw_weights;
        vector<vector<vector<vector<double>>>> reshapedData(outputChannels, 
                        vector<vector<vector<double>>>(inputChannels, 
                        vector<vector<double>>(rowsWidth, vector<double>(imgCols))));    
        int indexVal = 0; 

        for (const auto& row : data) {
            raw_weights.insert(raw_weights.end(), row.begin(), row.end());
        }
        for(int i = 0; i< outputChannels; i++){
            for(int j=0; j<inputChannels; j++){
                for(int k=0; k<rowsWidth; k++){
                    for(int l=0; l<imgCols; l++){

                        double cellVal = raw_weights[indexVal];
                        if( abs(cellVal) < 1e-10){
                            cellVal = 0.0;
                        }
                        reshapedData[i][j][k][l] = cellVal;
                        indexVal+=1;
                    }
                }
            }
        }
        data.clear();
        raw_weights.clear();
        return reshapedData;
    }
    

    /**
     * @brief Load and reshape fully connected layer weights.
     * 
     * Reads flat weight data and reshapes into 2D form:
     * [outputChannels][inputChannels].
     *
     * @param fileName Path to the CSV file.
     * @param outputChannels Number of output neurons.
     * @param inputChannels Number of input neurons.
     * @return 2D vector containing reshaped weights.
     */
    static inline vector<vector<double>> load_fc_weights(string fileName, int outputChannels, int inputChannels){
        vector<vector<double>> data = loadCSV(fileName);
        vector<double> raw_weights;
        for (const auto& row : data) {
            raw_weights.insert(raw_weights.end(), row.begin(), row.end());
        }

        vector<vector<double>> reshapedData(outputChannels, vector<double>(inputChannels));    
        int indexVal = 0; 
        for(int i = 0; i< outputChannels; i++){
            for(int j=0; j< inputChannels; j++){
                double cellVal = raw_weights[indexVal];
                if( abs(cellVal) < 1e-10){
                    cellVal = 0.0;
                }
                reshapedData[i][j] = cellVal;
                indexVal+=1;
            }
        }
        return reshapedData;
    }



    /**
     * @brief Write text content to a file.
     *
     * @param filename Path to the output file.
     * @param content String content to write.
     */
    static inline void write_to_file(string filename, string content) {
        ofstream file;
        file.open (filename);
        file << content.c_str();
        file.close();
    }

    
    /**
     * @brief Read the first line from a file.
     *
     * Reads only the first line and returns it as a string.
     *
     * @param filename Path to the input file.
     * @return First line from the file.
     */
    static inline string read_from_file(string filename) {
        //It reads only the first line!!
        string line;
        ifstream myfile (filename);
        if (myfile.is_open()) {
            if (getline(myfile, line)) {
                myfile.close();
                return line;
            } else {
                cerr << "Could not open " << filename << "." <<endl;
                exit(1);
            }
        } else {
            cerr << "Could not open " << filename << "." <<endl;
            exit(1);
        }
    }

    /**
     * @brief Flatten and deduplicate rotation keys.
     * 
     * Converts a 2D vector of rotation key sets into a unique, 
     * sorted 1D vector of non-zero rotation positions.
     *
     * @param rotation_keys 2D vector of rotation positions.
     * @return Unique sorted list of rotation positions.
     */
    static inline vector<int> serialize_rotation_keys( vector<vector<int>> rotation_keys){

        vector<int> rotation_positions;
        for (const auto& vec : rotation_keys) {
            rotation_positions.insert(rotation_positions.end(), vec.begin(), vec.end());
        }

        std::sort(rotation_positions.begin(), rotation_positions.end());
        rotation_positions.erase(std::unique(rotation_positions.begin(), rotation_positions.end()), rotation_positions.end());
        rotation_positions.erase(std::remove(rotation_positions.begin(), rotation_positions.end(), 0), rotation_positions.end());
        
        return rotation_positions;
    }


    /**
     * @brief Mask of windows of ones separated by a zero, with leading and trailing zero padding, repeated over tiles.
     *
     * One tile: starting_padding zeros, then windows of window_length ones each followed by one zero, until
     * max_length - ending_padding slots, trimmed or zero-padded to max_length. The tile is repeated tile_count times.
     *
     * @param starting_padding Leading zeros of every tile.
     * @param ending_padding   Slots left at the end of every tile (zeros).
     * @param window_length    Ones per window.
     * @param max_length       Slots of one tile.
     * @param tile_count       Number of tiles (e.g. channels).
     * @return Mask of tile_count * max_length values.
     */
    static inline vector<double> build_tile_mask(int starting_padding, int ending_padding, 
        int window_length, int max_length, int tile_count) {
    
        vector<double> mask;

        // Add starting padding
        for (int i = 0; i < starting_padding; ++i) {
            mask.push_back(0.0);
        }

        // Add windows of 1s and a trailing 0
        while (mask.size() < static_cast<size_t>(max_length - ending_padding)) {
            for (int j = 0; j < window_length; ++j) {
                mask.push_back(1.0);
            }
            mask.push_back(0.0);
        }

        // Trim or pad the mask to match max_length
        while (mask.size() > static_cast<size_t>(max_length)) {
            mask.pop_back();
        }
        while (mask.size() < static_cast<size_t>(max_length)) {
            mask.push_back(0.0);
        }

        // Add ending padding
        for (int i = 0; i < ending_padding; ++i) {
            mask[max_length - i - 1] = 0.0;
        }

        // Tile the mask
        std::vector<double> tiled_mask;
        for (int i = 0; i < tile_count; ++i) {
            tiled_mask.insert(tiled_mask.end(), mask.begin(), mask.end());
        }
        return tiled_mask;
    }

}

#endif //FHEON_DATAUTILS_H