
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
 * @file UtilsImage.h
 * @brief Utilities for inputting images into HE-friendly neural networks.
 *
 * Provides functions for handling datasets such as CIFAR-10 and MNIST,
 * preparing them for use in homomorphic encryption-based neural networks.
 */

#ifndef FHEON_IMAGESUTILS_H
#define FHEON_IMAGESUTILS_H

#include <iostream>
#include <openfhe.h>
#include "Utils.h"

using namespace std;

namespace utilsimages {

    /**
     * @brief Read CIFAR-10 image data and return multiple images as normalized vectors.
     *
     * @param full_path Path to binary file.
     * @param num_images Number of images to read.
     * @param img_size Size of one image (in bytes).
     * @return Vector of images, each image stored as a flattened [R,G,B] channel vector.
     */
    static inline vector<vector<double>> read_images(string full_path, int num_images, int img_size) {
        int img_cols = 32;
        vector<double> meanValues = {0.4914, 0.4822, 0.4465};
        vector<double> stdValues = {0.2023, 0.1994, 0.2010};
        ifstream file(full_path, ios::binary);
        if (!file.is_open()) {
            cerr << "Error opening CIFAR-10 file!" << endl;
            return {};
        }
        /*** keep the ground-truth labels in memory for the accuracy: each record is its label
         *   byte(s) followed by the image (CIFAR-100 has a coarse and a fine label, the fine one is kept) */
        int label_bytes = (full_path.find("cifar-100") != string::npos || full_path.find("cifar100") != string::npos) ? 2 : 1;
        utils::image_labels.clear();
        utils::predictions_made = 0;
        utils::labelled_predictions = 0;
        utils::correct_predictions = 0;
        for (int i = 0; i < num_images; ++i) {
            file.seekg(static_cast<std::streamoff>(i) * (label_bytes + img_size) + (label_bytes - 1), std::ios::beg);
            char label;
            if (!file.read(&label, 1)) break;
            utils::image_labels.push_back(static_cast<unsigned char>(label));
        }
        file.clear();

        // read images: every record carries its own label byte(s) in front of the pixels,
        // so each image is read from the start of its own record
        size_t total_size = static_cast<size_t>(img_size) * num_images;
        vector<uint8_t> imagePixels(total_size);
        for (int i = 0; i < num_images; ++i) {
            file.seekg(static_cast<std::streamoff>(i) * (label_bytes + img_size) + label_bytes, std::ios::beg);
            file.read(reinterpret_cast<char*>(imagePixels.data()) + static_cast<size_t>(i) * img_size, img_size);
        }
        vector<vector<double>> allImages;

        for (int i = 0; i < num_images; ++i) {
            int image_offset = i * img_size;
            // Separate channels (Red, Green, Blue) into individual vectors
            vector<double> redChannel(img_cols * img_cols);
            vector<double> greenChannel(img_cols * img_cols);
            vector<double> blueChannel(img_cols * img_cols);

            for (int j = 0; j < img_cols * img_cols; ++j) {
                double redValue = (static_cast<double>(imagePixels[image_offset + j]));
                double greenValue = (static_cast<double>(imagePixels[image_offset + (img_cols * img_cols) + j]));
                double blueValue = (static_cast<double>(imagePixels[image_offset + (2 * img_cols * img_cols) + j]));

                redChannel[j] = (((redValue/255.0)- meanValues[0])/stdValues[0]);
                greenChannel[j] =  (((greenValue/255.0)- meanValues[1])/stdValues[1]);  // Green channel
                blueChannel[j] = (((blueValue/255.0)- meanValues[2])/stdValues[2]);
            }
            // Create a single vector to hold all channels in sequence
            vector<double> allPixels;
            allPixels.insert(allPixels.end(), redChannel.begin(), redChannel.end());
            allPixels.insert(allPixels.end(), greenChannel.begin(), greenChannel.end());
            allPixels.insert(allPixels.end(), blueChannel.begin(), blueChannel.end());

            // Add the single image vector to the allImages vector
            allImages.push_back(allPixels);
        }

        file.close();
        return allImages;
    }

    /**
     * @brief Display pixel values of a single image in channel-wise 32x32 format.
     *
     * @param allPixels Flattened vector of one image (3 * 32 * 32).
     * @param imageSize Size of image vector.
     * @param pixelState If true, prints pixel values channel by channel.
     */
    static inline void display_image(vector<double> allPixels, int imageSize, bool pixelState){
        int img_cols = 32;
        cout << "Image pixel values (3*32x32):" << endl;
    
        if(pixelState){
            cout << "Image Red Channel:\n";
            for (int j = 0; j < img_cols * img_cols; ++j) {
                cout << allPixels[j] << " ";
                if ((j + 1) % img_cols == 0) cout << "\n";
            }

            cout << "\nImage Green Channel:\n";
            for (int j = 0; j < img_cols * img_cols; ++j) {
                cout << (allPixels[(img_cols * img_cols) + j]) << " ";
                if ((j + 1) % img_cols == 0) cout << "\n";
            }

            cout << "\nImage Blue Channel:\n";
            for (int j = 0; j < img_cols * img_cols; ++j) {
                cout << (allPixels[(2 * img_cols * img_cols) + j]) << " ";
                if ((j + 1) % img_cols == 0) cout << "\n";
            }
        }
        else{
            cout << endl << endl;
        }
        std::cout << "Total number of pixels in the combined vector: " << allPixels.size() << std::endl;
    }

    /**
     * @brief Load a batch of individual CIFAR-10 .bin images starting from a given index.
     *
     * Reads batchSize individual .bin files (e.g., 0000.bin, 0001.bin, ...) from the specified folder,
     * starting from startIndex. Each file contains one image in channel-first order (C, H, W).
     * If normalize is true, applies CIFAR-10 normalization; otherwise, returns raw values.
     *
     * @param folderPath Path to the folder containing .bin files.
     * @param startIndex Starting index for file names (e.g., 0 for 000.bin).
     * @param batchSize Number of images to load.
     * @param normalize If true, normalize pixel values with CIFAR-10 mean/std.
     * @return Vector of images, each as a flattened [R,G,B] channel vector.
     */
    static inline vector<vector<double>> readBatchCIFAR10Images(const string& folderPath, int startIndex, int batchSize, bool normalize) {
        int img_cols = 32;
        int img_size = 3 * img_cols * img_cols;  // 3072
        vector<double> meanValues = {0.4914, 0.4822, 0.4465};
        vector<double> stdValues = {0.2023, 0.1994, 0.2010};
        vector<vector<double>> allImages;

        for (int i = 0; i < batchSize; ++i) {
            int fileIndex = startIndex + i;
            string filename = (fileIndex < 10 ? "000" : (fileIndex < 100 ? "00" : (fileIndex < 1000 ? "0" : ""))) + to_string(fileIndex) + ".bin";
            string filepath = folderPath + "/" + filename;

            if (utils::verbose_level >= 1) {
                cout << "Loading image: " << filepath << endl;
            }

            ifstream file(filepath, ios::binary);
            if (!file.is_open()) {
                cerr << "Error opening file: " << filepath << endl;
                continue;  // Skip missing files or handle error
            }

            vector<double> allPixels;
            if (normalize) {
                // Read as float32 (normalized data)
                vector<float> floatPixels(img_size);
                file.read(reinterpret_cast<char*>(floatPixels.data()), img_size * sizeof(float));
                for (float val : floatPixels) {
                    allPixels.push_back(static_cast<double>(val));
                }
            } else {
                // Read as uint8 (raw data)
                vector<uint8_t> imagePixels(img_size);
                file.read(reinterpret_cast<char*>(imagePixels.data()), img_size);
                for (uint8_t val : imagePixels) {
                    allPixels.push_back(static_cast<double>(val));
                }
            }
            file.close();

            // The data is in channel-first order (C, H, W), but we need to convert to channel-last for consistency with read_images
            // read_images returns [R,G,B] flattened, so rearrange
            vector<double> redChannel(img_cols * img_cols);
            vector<double> greenChannel(img_cols * img_cols);
            vector<double> blueChannel(img_cols * img_cols);

            for (int j = 0; j < img_cols * img_cols; ++j) {
                redChannel[j] = allPixels[j];
                greenChannel[j] = allPixels[img_cols * img_cols + j];
                blueChannel[j] = allPixels[2 * img_cols * img_cols + j];
            }

            vector<double> reorderedPixels;
            reorderedPixels.insert(reorderedPixels.end(), redChannel.begin(), redChannel.end());
            reorderedPixels.insert(reorderedPixels.end(), greenChannel.begin(), greenChannel.end());
            reorderedPixels.insert(reorderedPixels.end(), blueChannel.begin(), blueChannel.end());

            if (normalize) {
                // Already normalized in the file
                allImages.push_back(reorderedPixels);
            } else {
                // Apply normalization here
                for (size_t k = 0; k < reorderedPixels.size(); ++k) {
                    int channel = k / (img_cols * img_cols);
                    double val = reorderedPixels[k] / 255.0;
                    val = (val - meanValues[channel]) / stdValues[channel];
                    reorderedPixels[k] = val;
                }
                allImages.push_back(reorderedPixels);
            }
        }

        return allImages;
    }

    /**
     * @brief Load a batch of CIFAR-10 labels from a CSV file starting from a given index.
     *
     * Reads batchSize labels from the specified CSV file, starting from startIndex.
     * Each line in the CSV file is expected to contain a single label corresponding to an image.
     *
     * @param folderPath Path to the folder containing labels.csv.
     * @param startIndex Starting index for reading labels (e.g., 0 for the first label).
     * @param batchSize Number of labels to load.
     * @return Vector of labels as doubles.
     */
    static inline vector<double> readBatchedCIFAR10Labels(const string& folderPath, int startIndex, int batchSize ) {
        vector<double> labels;
        //read from the labels.csv file
        string labelFilePath = folderPath + "/labels.csv";
        ifstream file(labelFilePath);
        if (!file.is_open()) {
            cerr << "Error opening label file: " << labelFilePath << endl;
            return {};
        }
        string line;
        int currentIndex = 0;
        while (getline(file, line)) {
            if (currentIndex >= startIndex && currentIndex < startIndex + batchSize) {
                try {                    
                    labels.push_back(stod(line));
                } catch (const std::invalid_argument& e) {
                    cerr << "Invalid label: " << line << endl;
                    labels.push_back(0.0);  // Default to 0.0 for invalid labels
                }
            }
            /** every line counts (incrementing only inside the range never reached a startIndex > 0) */
            currentIndex++;
            if (currentIndex >= startIndex + batchSize) {
                break;  // Stop reading after we have the required number of labels
            }
        }
        file.close();   
        return labels;
    }




     /**
     * @brief Read MNIST images from a binary file into a 2D unsigned char array.
     *
     * @param full_path Path to MNIST image file (e.g., train-images.idx3-ubyte).
     * @param number_of_images Output parameter: number of images read.
     * @param image_size Output parameter: number of pixels per image (rows × cols).
     * @return Pointer to 2D array of image data [num_images][image_size].
     * @throws runtime_error If file cannot be opened or is invalid.
     */
    static inline unsigned char** read_mnist_images(string full_path, int& number_of_images, int& image_size) {
        auto reverseInt = [](int i) {
            unsigned char c1, c2, c3, c4;
            c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
            return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
        };

        typedef unsigned char uchar;

        ifstream file(full_path, ios::binary);

        if(file.is_open()) {
            int magic_number = 0, n_rows = 0, n_cols = 0;

            file.read((char *)&magic_number, sizeof(magic_number));
            magic_number = reverseInt(magic_number);

            if(magic_number != 2051) throw runtime_error("Invalid MNIST image file!");

            file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
            file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
            file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

            image_size = n_rows * n_cols;

            /*** keep the ground-truth labels in memory for the accuracy: MNIST ships them in a
             *   separate file next to the images (…-images-idx3-ubyte -> …-labels-idx1-ubyte) */
            utils::image_labels.clear();
            utils::predictions_made = 0;
            utils::labelled_predictions = 0;
            utils::correct_predictions = 0;
            string labels_path = full_path;
            size_t name_pos = labels_path.rfind("images-idx3");
            if (name_pos != string::npos) {
                labels_path.replace(name_pos, strlen("images-idx3"), "labels-idx1");
                ifstream labels_file(labels_path, ios::binary);
                int labels_magic = 0, number_of_labels = 0;
                if (labels_file.is_open()) {
                    labels_file.read((char *)&labels_magic, sizeof(labels_magic)), labels_magic = reverseInt(labels_magic);
                    labels_file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);
                    if (labels_magic == 2049 && number_of_labels > 0) {
                        vector<uchar> raw_labels(number_of_labels);
                        labels_file.read((char *)raw_labels.data(), number_of_labels);
                        raw_labels.resize(labels_file.gcount());
                        utils::image_labels.assign(raw_labels.begin(), raw_labels.end());
                    }
                }
                if (utils::image_labels.empty()) {
                    cerr << "MNIST labels not found at " << labels_path << ": accuracy will not be reported." << endl;
                }
            }

            uchar** _dataset = new uchar*[number_of_images];
            for(int i = 0; i < number_of_images; i++) {
                _dataset[i] = new uchar[image_size];
                file.read((char *)_dataset[i], image_size);
            }
            return _dataset;
        } else {
            throw runtime_error("Cannot open file `" + full_path + "`!");
        }
    }

    /**
     * @brief Convert a single MNIST image to a normalized vector of doubles.
     *
     * Normalization: (pixel / 255.0 - mean) / std.
     * Uses mean = 0.1307, std = 0.3081.
     *
     * @param imageData Pointer to raw image data (unsigned char array).
     * @param imageSize Number of pixels (28 × 28).
     * @return Normalized image as a vector<double>.
     */
    static inline vector<double> read_single_mnist_image(unsigned char* imageData, int imageSize) {
        vector<double> imageVector; 
        double meanVal = 0.1307; // mean value
        double stdVal = 0.3081; // standard diviation value
        for (int i = 0; i < imageSize; i++) {
            auto rescaledVal = static_cast<double>(imageData[i])/ 255.0;
            double normalizedVal = (rescaledVal - meanVal) / stdVal;
            imageVector.push_back(normalizedVal); 
        }
        return imageVector;
    }

    /**
     * @brief Read and normalize multiple MNIST images into a single batched vector.
     *
     * Takes raw unsigned char pixel data for multiple MNIST images, normalizes each 
     * pixel value using the dataset's mean and standard deviation, and concatenates 
     * all images into one flattened vector.
     *
     * @param imageData Pointer to the raw pixel data (batchSize * imageSize bytes).
     * @param batchSize Number of images in the batch.
     * @param imageSize Number of pixels per image (e.g., 28*28 for MNIST).
     *
     * @return Flattened vector<double> of size batchSize * imageSize containing all 
     *         normalized images in order: [batch0 image][batch1 image]...
     */
    static inline vector<double> read_multi_mnist_images(unsigned char* imageData, 
        int batchSize, int imageSize) {

        vector<double> batchedImages;
        // batchedImages.reserve(batchSize * imageSize); 

        const double meanVal = 0.1307;
        const double stdVal  = 0.3081;

        for (int b = 0; b < batchSize; b++) {
            for (int i = 0; i < imageSize; i++) {
                int idx = b * imageSize + i;
                double rescaledVal   = static_cast<double>(imageData[idx]) / 255.0;
                double normalizedVal = (rescaledVal - meanVal) / stdVal;
                batchedImages.push_back(normalizedVal);
            }
        }

        return batchedImages;
    }


    /**
     * @brief Display a single MNIST image in 28×28 format.
     *
     * @param imageData Pointer to raw image data (unsigned char array).
     * @param imageSize Number of pixels (28 × 28).
     * @param pixelState If true, prints pixel intensity values. If false, prints ASCII visualization ("X" or ".").
     */
    static inline void display_mnist_image( unsigned char* imageData, int imageSize, bool pixelState){
        int height = 28;
        int width = 28;
        cout << "Image pixel values (28x28):" << endl;
        if(pixelState){
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    int pixel = imageData[i * width + j];
                    cout << (int)pixel << "\t";  
                }
                cout << endl;
            }
            cout << endl;  
        }
        else{
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    int pixel = imageData[i * width + j];
                    cout << (pixel > 0 ? "X" : ".") << " ";  
                }
                cout << endl;
            }
        }
    }

    /**
     * @brief Free memory allocated for MNIST image dataset.
     *
     * @param mnistData Pointer to 2D array of MNIST images.
     * @param numImages Number of images to free.
     */
    static inline void clear_mnist_images(unsigned char** mnistData, int numImages){
        for (int i = 0; i < numImages; i++) {
            delete[] mnistData[i];
        }
        delete[] mnistData;
        // the models call this once all the images are processed
        utils::printAccuracy();
    }



   /**
     * @brief Clear image data from memory.
     *
     * @param imagesData Vector of images.
     * @param numImages Number of images (unused, kept for consistency).
     */
    static inline void clear_images(vector<vector<double>> imagesData, int numImages){
        // free the memory
        imagesData.clear();
        // the models call this once all the images are processed
        utils::printAccuracy();
    }




}

#endif //FHEON_IMAGESUTILS_H