
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

#ifndef FHEON_FHEONHEController_H
#define FHEON_FHEONHEController_H

#include <thread>
#include <openfhe.h>

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

/**
 * Security level of the CKKS contexts, chosen when FHEON is compiled (cmake -DSECURITY_LEVEL=...):
 *   0   : performance efficient (HEStd_NotSet). The ring dimension and slots of the model's HEConfig are used as they
 *         are, with a sparse ternary secret. Not a standard security level.
 *   128 : HEStd_128_classic    192 : HEStd_192_classic    256 : HEStd_256_classic
 *         OpenFHE estimates log QP of the modulus chain the model asks for (scale, first modulus, depth, digits) and
 *         takes the smallest ring dimension the HE security standard allows for that log QP (uniform ternary secret;
 *         the bootstrapping contexts use SPARSE_ENCAPSULATED, whose sparse key only serves the bootstrapping). The
 *         ring dimension of the HEConfig is ignored; its slot count stays the batch size, unless HEConfig.fillSlots
 *         asks for every slot of that ring (num_slots = ringDim / 2, see slot_scale).
 */
#ifndef FHEON_SECURITY_LEVEL
#define FHEON_SECURITY_LEVEL 0
#endif

/**
 * @brief CKKS context, keys and encryption services of FHEON (OpenFHE).
 *
 * Context generation and loading (HEConfig, generate_context / load_context), rotation, sum and bootstrapping key
 * generation, serialization and loading (one key file per stage or model), encryption and decryption, plaintext encoders of
 * the layer weights for the standard, diagonal and batch layouts, single and batched bootstrapping (batch_bootstrap_function,
 * with set_bootstrap_precision for precise 2-iteration bootstrapping), and helpers that read predictions and scaling values
 * from decrypted results.
 */
class FHEONHEController {

protected:
    CryptoContext<DCRTPoly> context;
    bool context_loaded = false;
    bool context_bootstrap_keys = true;
    bool context_sum_keys = true;

public:

    /** Create a config */
    struct HEConfig {
        int ringDim;
        int numSlots;
        int mlevelBootstrap;
        int dcrtBits;
        int firstMod;
        int numDigits;
        vector<uint32_t> levelBudget;
        bool serialize;
        // string keysFolder = "/scratch/nnjungle/AllHEkeys/32threads/";
        string keysFolder = "../AllHEKeys/";
        /** keys generated with the context. false leaves them to the per-layer key files: the bootstrapping keys of the
         *  slot counts actually bootstrapped (generate_bootstrapping_and_rotation_keys) and the EvalSum rotations
         *  (powers of two up to the summed size) */
        bool contextBootstrapKeys = true;   /** bootstrapping keys for numSlots, numSlots/2 and numSlots/4 */
        bool contextSumKeys = true;         /** EvalSum keys for the full batch */
        /** 0 (performance efficient), 128, 192 or 256 bits: see FHEON_SECURITY_LEVEL */
        int securityLevel = FHEON_SECURITY_LEVEL;
        /** at a standard security level the ring OpenFHE picks usually holds more slots than numSlots (2^15 slots for a
         *  2^14-slot model at 128 bits). true packs every slot of that ring: num_slots becomes ringDim / 2 and the model
         *  scales its packing by slot_scale(numSlots) (more images per ciphertext, more channel groups, ...). false
         *  keeps numSlots (sparse packing). No effect at the performance-efficient level, whose ring is 2 * numSlots */
        bool fillSlots = false;
        /** ring optimization at a standard security level. OpenFHE takes the smallest ring dimension the standard allows
         *  for log QP = log Q + log P of the chain; a chain a little too long for half that ring fits it with a
         *  smaller scale (log Q = firstMod + depth * dcrtBits) or more key-switching digits (smaller P). With
         *  minScaleBits > 0 the context tries every smaller ring, the largest scale down to minScaleBits (firstMod
         *  keeps its distance to the scale) and the fewest digits up to maxDigits, and keeps the smallest ring
         *  OpenFHE accepts. minScaleBits is the precision the model was validated with; 0 keeps dcrtBits. maxDigits
         *  0 = numDigits + 3. No effect at the performance-efficient level */
        int minScaleBits = 0;
        int maxDigits = 0;
    };

    int circuit_depth;
    int num_slots;
    int pLWE;
    int mult_depth = 10;
    int num_bootsraps = 0;
    /** precision (bits) of the first iteration when bootstrapping with 2 iterations (EvalBootstrap's precision argument).
     *  0 = OpenFHE's default, with which the second iteration brings nothing on ciphertexts at a deep level (measured at
     *  ring 2^15, firstMod 50, scale 46: ~8e-3 error on values of magnitude 3 with 1 or 2 iterations; 3e-7 with 2
     *  iterations and precision 15). Opt-in so the validated models keep their behaviour. */
    int bootstrap_precision = 0;
    /** security level of the contexts this controller generates: 0 (performance efficient), 128, 192 or 256 */
    int security_level = FHEON_SECURITY_LEVEL;
    /** HEConfig.fillSlots, minScaleBits and maxDigits of the context being generated */
    bool fill_slots = false;
    int min_scale_bits = 0;
    int max_digits = 0;
    /** scale and key-switching digits of the generated context (the HEConfig values unless the ring optimization or
     *  the digit search of a standard level changed them) */
    int scale_bits = 0;
    int num_digits = 0;

    /** how many times more slots the context has than 2^baseSlotsLog2, the slot count a model is written for
     *  (HEConfig.numSlots): 1 at the performance-efficient level or without fillSlots, 2, 4, ... when the ring of a
     *  standard security level is filled. A model multiplies its packing (images per ciphertext, channel groups) by it */
    int slot_scale(int baseSlotsLog2) const {
        int base = 1 << baseSlotsLog2;
        return (num_slots > base) ? num_slots / base : 1;
    }
    // string keys_folder = "/scratch/nnjungle/AllHEkeys/32threads/";
    string keys_folder = "../AllHEKeys/";
    string rotation_prefix = "rotation_keys_"; 
    string mult_prefix = "./mult_keys_";
    string sum_prefix = "./sum_keys_";
    string mult_file = "./mult-keys.bin";
    string sum_file = "./sum-keys.bin";
    string multi_key_file;
    string sum_key_file;
    string crypto_context_file;
    string public_key_file;
    string secret_key_file;

    void update_key_paths();

    FHEONHEController(CryptoContext<DCRTPoly> ctx) : context(ctx), context_loaded(false) {
        update_key_paths();
    }
    
    CryptoContext<DCRTPoly> getContext() const {
        return context;
    }

    PublicKey<DCRTPoly> getPublicKey() const {
        return keyPair.publicKey;
    }

    /*
     * Generating context, bootstrapping keys, rotation keys and loading them */
    void generate_context(int ringDim=14, int numSlots=12, int mlevelBootstrap=10, bool serialize = true);
    void generate_context(int ringDim=15, int numSlots=14, int mlevelBootstrap=10, int dcrtBits=55, int firstMod=56,   
                        int numDigits=3, vector<uint32_t> levelBudget ={4, 4}, bool serialize=true);

    void generate_context(HEConfig config);
    void generate_leveled_context(HEConfig config);

    /** the OpenFHE level of a security level in bits (0 -> HEStd_NotSet), and its name for the logs */
    static SecurityLevel to_openfhe_security_level(int bits);
    static string security_level_name(int bits);


    void generate_bootstrapping_keys(int bootstrap_slots, string filename, bool serialize);
    void generate_rotation_keys(vector<int> rotations, string filename = "", bool serialize = true, bool sum_key=true);
    void generate_bootstrapping_and_rotation_keys(vector<int> rotations, int bootstrap_slots, const string& filename, bool serialize, bool sum_key = true);

    // Generates rotation keys for indices not covered at initial key-gen time and
    // merges them into whatever automorphism keys are already loaded/generated,
    // then re-serializes the FULL merged set to the same rotation key file — unlike
    // generate_rotation_keys/generate_bootstrapping_and_rotation_keys, this always
    // actually generates (it does not early-return to a load when context_loaded is
    // true), since its purpose is specifically to add keys a prior generation/load
    // didn't have. Caller is responsible for backing up the existing key file first.
    void add_and_persist_rotation_keys(vector<int> rotations, const string& filename);

    void load_context();
    void load_context(HEConfig config, bool loadContext = true);

    void load_rotation_keys(const string& filename, bool verbose=false);
    void load_bootstrapping_and_rotation_keys(int bootstrap_slots, const string& filename,  bool verbose=false);

    void harness_clear_bootstrapping_and_rotation_keys(CryptoContext<DCRTPoly> &crypto_context);
    CryptoContext<DCRTPoly> read_evaluation_keys(CryptoContext<DCRTPoly> crypto_context, const string &rot_file);

    void clear_rotation_keys();
    void clear_context(int bootstrapping_key_slots);
    void clear_bootstrapping_and_rotation_keys(int bootstrap_num_slots);
    
    void set_bootstrap_count(int count);
    void set_bootstrap_precision(int bits) { bootstrap_precision = bits; }
    Ctext bootstrap_function(Ctext& encryptedInput, int level = 2);
    vector<Ctext> batch_bootstrap_function(vector<Ctext>& encryptedInput, int inputChannels, int encode_level=2);
    
    /*** Encrypt and decrypt packed ciphertext. used to encrypt image and decrpt the results ****/
    Ctext encrypt_input(vector<double>& inputData, bool isTimeMeasurement = true);
    Ctext encrypt_input_at_level(vector<double>& inputData, int encode_level, bool isTimeMeasurement = true);
    Ctext reencrypt_data(Ptext plaintextInput);
    Ptext encode_input(vector<double>& inputData, int encode_level = 1);
    Ptext encode_input(vector<double>& inputData, int num_slots, int encode_level);
    Ptext decrypt_data(Ctext encryptedInput, int cols);

    vector<Ctext> encrypt_multiple_inputs(vector<vector<double>>& inputDatas);
    
    vector<vector<Ctext>> encrypt_kernel(vector<vector<vector<double>>>& kernelData, int colsSquare);
    vector<Ptext> encode_kernel(vector<vector<vector<double>>>& kernelData, int colsSquare);
    vector<Ptext> encode_kernel(vector<double>& kernelData, int colsSquare);
    vector<Ptext> encode_kernel_optimized(vector<vector<vector<double>>>& kernelData, int colsSquare, int encode_levels = 1);
    Ptext encode_shortcut_kernel(vector<double>& inputData, int colsSquare);
    Ptext encode_bais_input(vector<double>& inputData, int colsSquare, int encode_levels=1);

    Ctext change_num_slots(Ctext& encryptedInput, uint32_t numSlots);
    Ctext reduce_num_slots(const Ctext& encryptedInput, uint32_t numSlots);
    void set_encoding_slots(uint32_t numSlots);

    vector<vector<Ptext>> encode_kernel_diagonal(const vector<vector<vector<vector<double>>>>& kernelData, int outputWidth, int numSlots, int encode_level, bool strided = false, int inputPeriod = 0);
    vector<Ptext> encode_shortcut_diagonal(const vector<vector<double>>& kernelData, int outputWidth, int numSlots, int encode_level);
    Ptext encode_bias_diagonal(const vector<double>& biasData, int outputWidth, int numSlots, int encode_level);
    vector<vector<vector<Ptext>>> encode_kernel_layout(const vector<vector<vector<vector<double>>>>& kernelData, double scale, int numSlots, int encode_level,
                            int channelStride, int rowStride, int colStride, int outputWidth, int numGroups, const vector<int>& subShifts);
    Ptext encode_bias_layout(const vector<double>& biasData, int numSlots, int encode_level, int channelStride, int rowStride, int colStride,
                            int outputWidth, int numGroups, const vector<int>& subShifts);
    vector<vector<Ptext>> encode_linear_diagonal(const vector<vector<double>>& weightData, const vector<vector<int>>& inputPositions, double scale,
                            int numSlots, int numDiagonals, int babySteps, int encode_level);
    void generate_bootstrapping_keys_for_slots(uint32_t numSlots);
    void set_bootstrap_correction_factor(uint32_t correctionFactor);
    void prepare_bootstrapping(uint32_t numSlots);
    vector<Ptext> encode_avgpool_linear(const vector<vector<double>>& weightData, int inputWidth, int numSlots, int encode_level);

    int read_inferenced_label(Ctext encryptedInput, int noElements,  const string& predictions_file_path = "");
    int read_minmax(Ctext encryptedInput, int noElements); 
    int read_scaling_value(Ctext encryptedInput, int noElements);
    double read_max_abs(Ctext encryptedInput, int noElements);

    vector<int> read_batch_inferenced_label(Ctext inferencedData, int batchSize, int numClasses,  const string& predictions_file_path = "", int baseIndex = 0);
    vector<int> read_batch_inferenced_label_multiple_outputs(vector<Ctext> inferencedData,  int batchSize, int numClasses,  const string& predictions_file_path = "", int baseIndex = 0);
    vector<int> read_forward_inferenced_label_multiple_outputs(vector<Ctext>& inferencedData,  int batchSize, int numClasses,  ofstream& outFile, int baseIndex = 0);
    /** as read_batch_inferenced_label(_multiple_outputs), for the real images of a batch completed with blank ones: only the first
     *  numImages images are reported, as dataset images firstIndex, firstIndex + 1, ..., and recorded for the accuracy */
    vector<int> report_batch_labels(Ctext inferencedData, int batchSize, int numClasses, int firstIndex, int numImages, const string& predictions_file_path = "");
    vector<int> report_batch_labels_multiple_outputs(vector<Ctext>& inferencedData, int numClasses, int firstIndex, int numImages, const string& predictions_file_path = "");

    vector<int> read_batch_scaling_values(vector<Ctext>& encryptedInputs, int inputChannels, int num_slots);
    int read_batch_minmax(vector<Ctext>& encryptedInputs, int inputChannels,  int num_slots);
    int decrypt_batch_data(vector<Ctext>& encryptedInputs, int inputChannels,  int num_slots);
    vector<double> decrypt_and_print(Ctext encryptedpackedVector, int num_slots);
    // Decrypts just slot 0 — used by the training backward controller to print
    // single scalar weight/bias/gradient values (each such ciphertext holds
    // the same scalar broadcast across every slot) without a full batch decrypt.
    double decrypt_scalar(Ctext& encryptedInput);


    Ptext decrypt_data_with_key(PrivateKey<DCRTPoly> &sk,
                              Ctext encryptedinputData, int cols);
    int read_scaling_value_with_key(PrivateKey<DCRTPoly> &sk,
                                    Ctext encryptedInput, int noElements);
    int read_inferenced_label_with_key(PrivateKey<DCRTPoly> &sk,
                                     Ctext encryptedInput, int noElements,
                                     const string& predictions_file_path = "");

private:
    KeyPair<DCRTPoly> keyPair;
    vector<uint32_t> level_budget = {3, 3};
    vector<uint32_t> bsgsDim = {0, 0};
    uint32_t bootstrap_correction_factor = 0;
    vector<double> build_tiled_mask(int starting_padding, int ending_padding, int window_length, int max_length, int tile_count) ;
    
    void keys_serialization();
};


#endif //FHEON_FHEONHEController_H
