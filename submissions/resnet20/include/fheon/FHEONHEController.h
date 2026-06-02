
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

/********************************************************************
 * This FHE controller is used to define basic FHE functions used across the
 * project such as: Context generation, bootstrapping, encryption
 ********************************************************************/

#ifndef FHEON_FHEONHEController_H
#define FHEON_FHEONHEController_H

#include <openfhe.h>
#include <thread>

#include <ciphertext-ser.h>
#include <cryptocontext-ser.h>
#include <key/key-ser.h>
#include <scheme/ckksrns/ckksrns-ser.h>
// #include "schemeswitching-data-serializer.h"

#include "Utils.h"
#include "UtilsData.h"
#include "UtilsImage.h"

using namespace lbcrypto;
using namespace std;
using namespace std::chrono;

using namespace utils;
using namespace utilsdata;
using namespace utilsimages;

using Ptext = Plaintext;
using Ctext = Ciphertext<DCRTPoly>;

class FHEONHEController {

protected:
  CryptoContext<DCRTPoly> context;

public:
  int circuit_depth;
  int num_slots;
  int pLWE;
  int mult_depth = 10;
  string keys_folder = "./../../io/single/";
  string cc_prefix = "./secret_key/cc.bin";
  string pk_prefix = "./secret_key/sk.bin";
  string rotation_prefix = "./public_keys/rk.bin";
  string mult_prefix = "./public_keys/mt.bin";
  string sum_prefix = "./public_keys/sm.bin";
  string sk_prefix = "./secret_key/sk.bin";

  FHEONHEController(CryptoContext<DCRTPoly> ctx) : context(ctx) {}

  CryptoContext<DCRTPoly> getContext() const { return context; }
  void setContext(CryptoContext<DCRTPoly> ctx) { context = ctx; }

  double getlogPQ(const DCRTPoly &poly);

  /*
   * Generating context, bootstrapping keys, rotation keys and loading them */
  void generate_context(int ringDim = 14, int numSlots = 12,
                        int mlevelBootstrap = 10, bool serialize = true);
  void generate_context(int ringDim = 15, int numSlots = 14,
                        int mlevelBootstrap = 10, int dcrtBits = 55,
                        int firstMod = 56, int numDigits = 3,
                        vector<uint32_t> levelBudget = {4, 4},
                        bool serialize = true);

  void generate_bootstrapping_keys(int bootstrap_slots, string filename,
                                   bool serialize);
  void generate_rotation_keys(vector<int> rotations, string filename = "",
                              bool serialize = true);
  void generate_bootstrapping_and_rotation_keys(vector<int> rotations,
                                                int bootstrap_slots,
                                                const string &filename,
                                                bool serialize);

  void load_context(bool verbose = false);
  void load_rotation_keys(const string &filename, bool verbose = false);
  void load_bootstrapping_and_rotation_keys(int bootstrap_slots,
                                            const string &filename,
                                            bool verbose = false);

  void clear_rotation_keys();
  void clear_context(int bootstrapping_key_slots);
  void clear_bootstrapping_and_rotation_keys(int bootstrap_num_slots);
  Ctext bootstrap_function(Ctext &encryptedInput, int level = 2);
  vector<Ctext> batch_bootstrap_function(vector<Ctext>& encryptedInputs, int inputChannels, int encode_level = 2);

  /*** Encrypt and decrypt packed ciphertext. used to encrypt image and decrpt
   * the results ****/
  Ctext encrypt_input(vector<double> &inputData);
  Ctext reencrypt_data(Ptext plaintextInput);
  Ptext encode_input(vector<double> &inputData, int encode_level = 1);
  Ptext encode_input(vector<double> &inputData, int num_slots,
                     int encode_level = 1);
  Ptext decrypt_data(Ctext encryptedInput, int cols);

  vector<vector<Ctext>>
  encrypt_kernel(vector<vector<vector<double>>> &kernelData, int colsSquare);
  vector<Ptext> encode_kernel(vector<vector<vector<double>>> &kernelData,
                              int colsSquare);
  vector<Ptext> encode_kernel(vector<double> &kernelData, int colsSquare);
  vector<Ptext>
  encode_kernel_optimized(vector<vector<vector<double>>> &kernelData,
                          int colsSquare, int encode_levels = 1);
  Ptext encode_shortcut_kernel(vector<double> &inputData, int colsSquare);
  Ptext encode_bais_input(vector<double> &inputData, int colsSquare,
                          int encode_levels = 1);

  Ctext change_num_slots(Ctext &encryptedInput, uint32_t numSlots);

  int read_inferenced_label(Ctext encryptedInput, int noElements,
                            ofstream &outFile);
  int read_minmax(Ctext encryptedInput, int noElements);
  int read_scaling_value(Ctext encryptedInput, int noElements);

  vector<int> read_batch_inferenced_label(Ctext inferencedData, int batchSize,
                                          int numClasses, ofstream &outFile,
                                          int baseIndex);
  vector<int> read_batch_inferenced_label_multiple_outputs(vector<Ctext> inferencedData,
                                                           int batchSize, int numClasses,
                                                           ofstream &outFile, int baseIndex);
  vector<int> read_batch_scaling_values(vector<Ctext> &encryptedInputs,
                                        int inputChannels, int num_slots);
  int read_batch_minmax(vector<Ctext> &encryptedInputs, int inputChannels,
                        int num_slots);
  int decrypt_batch_data(vector<Ctext> &encryptedInputs, int inputChannels,
                         int num_slots);
  void decrypt_and_print(Ctext encryptedpackedVector, int num_slots);

  Ptext decrypt_data_with_key(PrivateKey<DCRTPoly> &sk,
                              Ctext encryptedinputData, int cols);
  int read_scaling_value_with_key(PrivateKey<DCRTPoly> &sk,
                                  Ctext encryptedInput, int noElements);
  int read_inferenced_label_with_key(PrivateKey<DCRTPoly> &sk,
                                     Ctext encryptedInput, int noElements,
                                     ofstream &outFile);

  void harness_generate_bootstrapping_and_rotation_keys(CryptoContext<DCRTPoly> &crypto_context, PrivateKey<DCRTPoly> &sk,
                            vector<int> rotations_positions, ofstream &key_file, bool sum_keys = false);
  void harness_clear_bootstrapping_and_rotation_keys(CryptoContext<DCRTPoly> &crypto_context);

  void read_evaluation_keys(CryptoContext<DCRTPoly> &crypto_context, const string &pubkey_dir,
                            const string &mult_file, const string &rot_file);
  void read_evaluation_keys(CryptoContext<DCRTPoly> &crypto_context, const string &pubkey_dir,
                            const string &mult_file, const string &rot_file, const string &sk_path);

  void harness_generate_rotation_keys(CryptoContext<DCRTPoly> &crypto_context, PrivateKey<DCRTPoly> &sk,
                            vector<int> rotations_positions, ofstream &key_file, bool sum_key);

private:
  KeyPair<DCRTPoly> keyPair;
  vector<uint32_t> level_budget = {4, 4};
  vector<uint32_t> bsgsDim = {0, 0};
  vector<double> build_tiled_mask(int starting_padding, int ending_padding,
                                  int window_length, int max_length, int tile_count);

  void keys_serialization();
};

#endif // FHEON_FHEONHEController_H
