
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
 * @brief FHE controller for defining basic FHE functions used across different neural networks.
 *
 * This class provides fundamental methods for context generation, encryption, encoding, 
 * bootstrapping, and other FHE operations that are utilized throughout the ANN development.
 *
 */

#include <fstream>
#include <filesystem> 
#include <iostream>
// #include <thread>
namespace fs = std::filesystem;

#include "FHEONHEController.h"

/**
 * @brief Compute the PQ value, which defines the application's security level.
 *
 * This function calculates the PQ value from the given polynomial, which is used 
 * to determine the security level of the application.
 *
 * @param poly  Input polynomial used to compute the PQ value.
 *
 * @return The computed PQ value as a double.
 */

double FHEONHEController::getlogPQ(const DCRTPoly &poly) {
  int n = poly.GetNumOfElements();
  double logPQ = 0;
  for (int i = 0; i < n; i++) {
    auto qi = poly.GetParams()->GetParams()[i]->GetModulus();
    logPQ += log(qi.ConvertToDouble()) / log(2);
  }
  return logPQ;
}


/**
 * @brief Generate the full FHE context for the project.
 *
 * This function sets up the FHE context with all specified parameters, allowing 
 * fine-grained control over scaling, decomposition, and level budgets. Keys can 
 * be optionally serialized and saved.
 *
 * @param ringDim          Ring dimension.
 * @param numSlots         Number of slots (batch size).
 * @param mlevelBootstrap  Multiplication level after bootstrapping.
 * @param dcrtBits         Scaling factor for DCRT representation.
 * @param firstMod         Scaling factor for the first coefficients.
 * @param numDigits        Number of digits used in key decomposition.
 * @param levelBudget      Vector of budget levels.
 * @param serialize        Whether to serialize and save keys.
 */
void FHEONHEController::generate_context(int ringDim, int numSlots, int mlevelBootstrap, 
                        int dcrtBits, int firstMod, int numDigits, vector<uint32_t> levelBudget,
                        bool serialize) {

    CCParams<CryptoContextCKKSRNS> parameters;
    auto secretKeyDist = SPARSE_TERNARY;

    ScalingTechnique rescaleTech = FLEXIBLEAUTO;
    level_budget = levelBudget;
    num_slots = 1 << numSlots;
    mult_depth = mlevelBootstrap;

    parameters.SetRingDim(1 << ringDim);
    parameters.SetBatchSize(num_slots);
    parameters.SetScalingModSize(dcrtBits);
    parameters.SetFirstModSize(firstMod);
    parameters.SetNumLargeDigits(numDigits);
    
    parameters.SetSecretKeyDist(secretKeyDist);
    parameters.SetSecurityLevel(lbcrypto::HEStd_NotSet);
    // parameters.SetSecurityLevel(lbcrypto::HEStd_128_classic);
    parameters.SetScalingTechnique(rescaleTech);
    
    circuit_depth = mult_depth + FHECKKSRNS::GetBootstrapDepth(level_budget, secretKeyDist);
    parameters.SetMultiplicativeDepth(circuit_depth);

    cout << "Building the FHE Context" << endl;
    cout << "dcrtBits: "<< dcrtBits << " -- firstMod: " << firstMod << endl << "Ciphertexts depth: " 
         << circuit_depth << ", available multiplications: " << circuit_depth - 2 << endl;
   
    context = GenCryptoContext(parameters);
    context->Enable(PKE);
    context->Enable(KEYSWITCH);
    context->Enable(LEVELEDSHE);
    context->Enable(ADVANCEDSHE);
    context->Enable(FHE);
    
    keyPair = context->KeyGen();
    context->EvalMultKeyGen(keyPair.secretKey);
    context->EvalSumKeyGen(keyPair.secretKey);

    ringDim = context->GetRingDimension();
    numSlots = num_slots;
    usint halfnumSlots = numSlots/2;
    context->EvalBootstrapSetup(level_budget, bsgsDim, numSlots);
    context->EvalBootstrapKeyGen(keyPair.secretKey, numSlots);
    
    auto sec_level = parameters.GetSecurityLevel();
    auto logq = context->GetModulus().GetMSB();
     double logPQ = getlogPQ(keyPair.publicKey->GetPublicElements()[0]);
    cout << "Keys Generated." << endl;
    cout << "Cyclotomic Order: " << context->GetCyclotomicOrder() << endl;
    cout << "CKKS scheme is using ring dimension: " << ringDim  << endl;
    cout << "Avaliable numSlots: " << numSlots << "  - halfnumSlots: " << halfnumSlots << endl;
    cout << "LogQ: "<< logq << " - Security Level: " << endl;
    cout << "Security Level: " << sec_level << endl;
    cout << "Ciphertexts depth: " << circuit_depth << endl; 
    cout << "Multiplication Depth: " << mult_depth - 2 << endl;
    cout << "log PQ = " << logPQ << std::endl << std::endl;
    cout << "-----------------------------------------------------------" << endl;
    
    if(serialize){
        write_to_file(keys_folder + "/mult_depth.txt", to_string(mult_depth));
        write_to_file(keys_folder + "/num_slots.txt", to_string(num_slots));
        write_to_file(keys_folder + "/level_budget.txt", to_string(level_budget[0]) + "," + to_string(level_budget[1]));
        keys_serialization();
    }


    #include <pthread.h>
#include <sched.h>


    return;
}

/**
 * @brief Simplified version of generate_context using standard values for unspecified parameters.
 *
 * This function sets up the FHE context with default parameters for all unspecified 
 * values, simplifying context generation for typical use cases.
 *
 * @param ringDim          Ring dimension.
 * @param numSlots         Number of slots (batch size).
 * @param mlevelBootstrap  Multiplication level after bootstrapping.
 * @param serialize        Whether to serialize and save keys.
 */
void FHEONHEController::generate_context(int ringDim, int numSlots, int mlevelBootstrap, bool serialize) {
    CCParams<CryptoContextCKKSRNS> parameters;

    num_slots = 1 << numSlots;
    int dcrtBits               = 46;
    int firstMod               = 50;

    auto secretKeyDist = SPARSE_TERNARY;
    parameters.SetSecretKeyDist(secretKeyDist);
    parameters.SetSecurityLevel(lbcrypto::HEStd_NotSet);
    parameters.SetNumLargeDigits(3);
    parameters.SetRingDim(1 << ringDim);
    parameters.SetBatchSize(num_slots);
    ScalingTechnique rescaleTech = FLEXIBLEAUTO; 
    parameters.SetScalingModSize(dcrtBits);
    parameters.SetFirstModSize(firstMod);
    parameters.SetScalingTechnique(rescaleTech);
    mult_depth = mlevelBootstrap;
    uint32_t levelsAvailableAfterBootstrap = mult_depth;

    circuit_depth = levelsAvailableAfterBootstrap + FHECKKSRNS::GetBootstrapDepth(level_budget, secretKeyDist);

    cout << "Context built, generating keys..." << endl;
    cout << endl << "dcrtBits: "<< dcrtBits << " -- firstMod: " << firstMod << endl << "Ciphertexts depth: " 
         << circuit_depth << ", available multiplications: " << levelsAvailableAfterBootstrap - 2 << endl;

    parameters.SetMultiplicativeDepth(circuit_depth);
    context = GenCryptoContext(parameters);

    context->Enable(PKE);
    context->Enable(KEYSWITCH);
    context->Enable(LEVELEDSHE);
    context->Enable(ADVANCEDSHE);
    context->Enable(FHE);

    keyPair = context->KeyGen();
    context->EvalMultKeyGen(keyPair.secretKey);
    context->EvalSumKeyGen(keyPair.secretKey);

    numSlots = num_slots;
    usint halfnumSlots = numSlots/2;
    cout << "numSlots: " << numSlots << "  - halfnumSlots: " << halfnumSlots << endl;
    context->EvalBootstrapSetup(level_budget, bsgsDim, numSlots);
    context->EvalBootstrapKeyGen(keyPair.secretKey, numSlots);

    cout << " Keys Generated." << endl;
    ringDim = context->GetRingDimension();
    cout << " CKKS scheme is using ring dimension: " << ringDim  << endl;
    cout << " Ciphertexts depth: " << circuit_depth << endl; 
    cout << " Multiplication Depth: " << levelsAvailableAfterBootstrap - 2 << endl;
    cout << " Cyclotomic Order: " << context->GetCyclotomicOrder() << endl;
    cout << " -----------------------------------------------------------" << endl;

   if(serialize){
        write_to_file(keys_folder + "/mult_depth.txt", to_string(mult_depth));
        write_to_file(keys_folder + "/level_budget.txt", to_string(level_budget[0]) + "," + to_string(level_budget[1]));
        keys_serialization();
    }
    return;
}


/**
 * @brief Generate all evaluation keys and save them to the keys folder.
 *
 * This function generates all necessary evaluation keys for the FHE context 
 * and serializes them into the designated keys folder for later use.
 */

void FHEONHEController::keys_serialization(){
    
    cout << "------------------------------------------------------------" << endl;
    cout << "Now serializing keys ..." << endl;

    if (!fs::exists(keys_folder)) {
        if (!fs::create_directory(keys_folder)) {
            std::cerr << "Failed to create directory: " << keys_folder << std::endl;
            return;
        }
    }

    if (!Serial::SerializeToFile(keys_folder + "/crypto-context.bin", context, SerType::BINARY)) {
        cerr << "Error writing serialization of the crypto context to crypto-context.bin" << endl;
    } else {
        cout << "Crypto Context have been serialized" << std::endl;
    }

    ofstream multKeyFile(keys_folder + "/mult-keys.bin", ios::out | ios::binary);
    if (multKeyFile.is_open()) {
        if (!context->SerializeEvalMultKey(multKeyFile, SerType::BINARY)) {
            cerr << "Error writing eval mult keys" << std::endl;
            exit(1);
        }
        cout << "Relinearization Keys have been serialized" << std::endl;
        multKeyFile.close();
    }
    else {
        cerr << "Error serializing EvalMult keys in \"" << keys_folder + "/mult-keys.bin" << "\"" << endl;
        exit(1);
    }

    ofstream sumKeysFile(keys_folder + "/sum-keys.bin", ios::out | ios::binary);
    if (sumKeysFile.is_open()) {
        if (!context->SerializeEvalSumKey(sumKeysFile, SerType::BINARY)) {
            cerr << "Error writing sum keys" << std::endl;
            exit(1);
        }
        cout << "sum keys have been serialized" << std::endl;
    } else {
        cerr << "Error serializing sum keys \"" << keys_folder + "/sum-keys" << "\"" << std::endl;
        exit(1);
    }

    if (!Serial::SerializeToFile(keys_folder + "/public-key.bin", keyPair.publicKey, SerType::BINARY)) {
        cerr << "Error writing serialization of public key to public-key.bin" << endl;
    } else {
        cout << "Public Key has been serialized" << std::endl;
    }

    if (!Serial::SerializeToFile(keys_folder + "/secret-key.bin", keyPair.secretKey, SerType::BINARY)) {
        cerr << "Error writing serialization of public key to secret-key.bin" << endl;
    } else {
        cout << "Secret Key has been serialized" << std::endl;
    }
    return;
}

/**
 * @brief Load all serialized keys from the storage folder.
 *
 * This function reads and loads all keys that were previously serialized 
 * and stored in files, typically from the "sskeys" folder.
 */
void FHEONHEController::load_context(bool verbose) {
    
    context->ClearEvalMultKeys();
    context->ClearEvalAutomorphismKeys();
    CryptoContextFactory<lbcrypto::DCRTPoly>::ReleaseAllContexts();

    cout << "------------------------------------------------------------" << endl;
    if (verbose) cout << "Reading serialized context..." << endl;

    if (!Serial::DeserializeFromFile(keys_folder + "/crypto-context.bin", context, SerType::BINARY)) {
        cerr << "I cannot read serialized data from: " << keys_folder + "/crypto-context.bin" << endl;
        exit(1);
    }

    PublicKey<DCRTPoly> clientPublicKey;
    if (!Serial::DeserializeFromFile(keys_folder + "/public-key.bin", clientPublicKey, SerType::BINARY)) {
        cerr << "I cannot read serialized data from public-key.bin" << endl;
        exit(1);
    }

    PrivateKey<DCRTPoly> serverSecretKey;
    if (!Serial::DeserializeFromFile(keys_folder + "/secret-key.bin", serverSecretKey, SerType::BINARY)) {
        cerr << "I cannot read serialized data from secret-key.bin" << endl;
        exit(1);
    }

    keyPair.publicKey = clientPublicKey;
    keyPair.secretKey = serverSecretKey;

    std::ifstream multKeyIStream(keys_folder + "/mult-keys.bin", ios::in | ios::binary);
    if (!multKeyIStream.is_open()) {
        cerr << "Cannot read serialization from " << "mult-keys.bin" << endl;
        exit(1);
    }
    if (!context->DeserializeEvalMultKey(multKeyIStream, SerType::BINARY)) {
        cerr << "Could not deserialize eval multkey file" << endl;
        exit(1);
    }

    ifstream sumKeyIStream(keys_folder + "/sum-keys.bin", ios::in | ios::binary);
    if (!sumKeyIStream.is_open()) {
        cerr << "Cannot read serialization from " << "sum-keys.bin" << std::endl;
        exit(1);
    }
    if (!context->DeserializeEvalSumKey(sumKeyIStream, SerType::BINARY)) {
        cerr << "Could not deserialize eval rot key file" << std::endl;
        exit(1);
    }

    mult_depth = stoi(read_from_file(keys_folder + "/mult_depth.txt"));
    level_budget[0] = read_from_file(keys_folder + "/level_budget.txt").at(0) - '0';
    level_budget[1] = read_from_file(keys_folder + "/level_budget.txt").at(2) - '0';

    uint32_t approxBootstrapDepth = 4 + 4;  
    uint32_t levelsUsedBeforeBootstrap = mult_depth;
    circuit_depth = levelsUsedBeforeBootstrap + FHECKKSRNS::GetBootstrapDepth(approxBootstrapDepth, level_budget, SPARSE_TERNARY);

    if (verbose) cout << "Circuit depth: " << circuit_depth << ", available multiplications: " << levelsUsedBeforeBootstrap - 2 << endl;

    cout << "Context Loaded" << endl;
    cout << "------------------------------------------------------------" << endl;
}

/**
 * @brief Generate the bootstrapping keys for the FHE context.
 *
 * This function generates bootstrapping keys for the specified number of slots. 
 * The generated keys can be optionally serialized and saved to a file.
 *
 * @param bootstrap_slots  Number of bootstrapping slots.
 * @param filename         Filename to use when saving the keys.
 * @param serialize        Whether to serialize and save the bootstrapping keys.
 */
void FHEONHEController::generate_bootstrapping_keys(int bootstrap_slots, string filename, bool serialize) {
    
    // int numSlots = 1<<bootstrap_slots;
    // context->EvalBootstrapSetup(level_budget, bsgsDim, numSlots);
    // context->EvalBootstrapKeyGen(keyPair.secretKey, numSlots);
    context->EvalMultKeyGen(keyPair.secretKey);

    if(serialize){
        ofstream multKeysFile(keys_folder + mult_prefix + filename, ios::out | ios::binary);
        if (multKeysFile.is_open()) {
            if (!context->SerializeEvalMultKey(multKeysFile, SerType::BINARY)) {
                cerr << "Error writing mult keys" << std::endl;
                exit(1);
            }
            cout << "mult keys \"" << filename << "\" have been serialized" << std::endl;
        } else {
            cerr << "Error serializing mult keys" << keys_folder + mult_prefix + filename << std::endl;
            exit(1);
        }
    }
}


/**
 * @brief Generate and serialize rotation keys for the FHE context.
 *
 * This function generates rotation keys for the specified rotation positions. 
 * The generated keys can be optionally serialized and saved to a file.
 *
 * @param rotations  Vector of rotation positions to generate keys for.
 * @param filename   Filename to use when saving the rotation keys.
 * @param serialize  Whether to serialize and save the rotation keys.
 */
void FHEONHEController::generate_rotation_keys(const vector<int> rotations, std::string filename, bool serialize) {
   
    if (serialize && filename.size() == 0) {
        cout << "Filename cannot be empty when serializing rotation keys." << endl;
        return;
    }
    context->EvalRotateKeyGen(keyPair.secretKey, rotations);
    if (serialize) {
        ofstream rotationKeyFile(keys_folder + rotation_prefix + filename, ios::out | ios::binary);
        if (rotationKeyFile.is_open()) {
            if (!context->SerializeEvalAutomorphismKey(rotationKeyFile, SerType::BINARY)) {
                cerr << "Error writing rotation keys" << std::endl;
                exit(1);
            }
            cout << "Rotation keys \"" << filename << "\" have been serialized" << std::endl;
        } else {
            cerr << "Error serializing Rotation keys" << keys_folder + rotation_prefix + filename << std::endl;
            exit(1);
        }
    }
}


/**
 * @brief Generate and serialize both rotation keys and bootstrapping keys for the FHE context.
 *
 * This function generates rotation keys for the specified rotation positions and 
 * bootstrapping keys for the given number of slots. The generated keys can be 
 * optionally serialized and saved to a file.
 *
 * @param rotations        Vector of rotation positions to generate keys for.
 * @param bootstrap_slots  Number of bootstrapping slots.
 * @param filename         Filename to use when saving the keys.
 * @param serialize        Whether to serialize and save the generated keys.
 */
void FHEONHEController::generate_bootstrapping_and_rotation_keys(vector<int> rotations, int bootstrap_slots, const string& filename,  bool serialize) {
    if (serialize && filename.empty()) {
        cout << "Filename cannot be empty when serializing bootstrapping and rotation keys." << endl;
        return;
    }

    generate_bootstrapping_keys(bootstrap_slots, filename, serialize);
    generate_rotation_keys(rotations, filename, serialize);
}


/**
 * @brief Load previously generated bootstrapping and rotation keys from storage.
 *
 * This function loads bootstrapping and rotation keys that were previously 
 * generated and serialized, using the specified filename. Verbose mode can 
 * be enabled to display loading details.
 *
 * @param bootstrap_slots  Number of bootstrapping slots.
 * @param filename         Filename from which to load the keys.
 * @param verbose          Whether to display detailed loading information.
 */
void FHEONHEController::load_bootstrapping_and_rotation_keys(int bootstrap_slots, const string& filename, bool verbose) {
    if (verbose) cout << endl << "Loading bootstrapping and rotations keys from " << filename << "..." << endl;

    int numSlots =  1 << bootstrap_slots;
    context->EvalBootstrapSetup(level_budget, bsgsDim, numSlots);
    context->EvalBootstrapKeyGen(keyPair.secretKey, numSlots);

    if (verbose)  cout << "(1/4) Bootstrapping precomputations completed!" << endl;
    
    ifstream multKeyIStream(keys_folder +   mult_prefix + filename, ios::in | ios::binary);
    if (!multKeyIStream.is_open()) {
        cerr << "Cannot read serialization from " << keys_folder+ "/" << mult_prefix << filename << std::endl;
        exit(1);
    }
    if (!context->DeserializeEvalMultKey(multKeyIStream, SerType::BINARY)) {
        cerr << "Could not deserialize eval rot key file" << std::endl;
        exit(1);
    }
    if (verbose)  cout << "(2/4) MultKey deserialized and loaded!" << endl;

    ifstream rotKeyIStream(keys_folder + rotation_prefix + filename, ios::in | ios::binary);
    if (!rotKeyIStream.is_open()) {
        cerr << "Cannot read serialization from " << keys_folder+ "/" << rotation_prefix << filename << std::endl;
        exit(1);
    }
    if (!context->DeserializeEvalAutomorphismKey(rotKeyIStream, SerType::BINARY)) {
        cerr << "Could not deserialize eval rot key file" << std::endl;
        exit(1);
    }
    if (verbose) cout << "(4/4) Rotation keys deserialized and loaded!" << endl;
    if (verbose) cout << endl;
}


/**
 * @brief Load rotation keys from a specified file.
 *
 * This function loads rotation keys that were previously generated and serialized 
 * from the given filename. Verbose mode can be enabled to display loading details.
 *
 * @param filename  Filename from which to load the rotation keys.
 * @param verbose   Whether to display detailed loading information.
 */
void FHEONHEController::load_rotation_keys(const string& filename, bool verbose) {

    if (verbose) cout << endl << "Loading rotations keys from " << filename << "..." << endl;
    
    ifstream rotKeyIStream(keys_folder + rotation_prefix + filename, ios::in | ios::binary);
    if (!rotKeyIStream.is_open()) {
        cerr << "Cannot read serialization from " <<keys_folder + "/" << rotation_prefix << filename << std::endl;
        exit(1);
    }
    if (!context->DeserializeEvalAutomorphismKey(rotKeyIStream, SerType::BINARY)) {
        cerr << "Could not deserialize eval rot key file" << std::endl;
        exit(1);
    }

    if (verbose) {
        cout << "(1/1) Rotation keys read!" << endl;
        cout << endl;
    }
}

/**
 * @brief Clear all rotation keys stored in the FHE context.
 *
 * This function removes all previously stored rotation keys from the context, 
 * allowing new rotation keys to be generated without conflicts.
 */
void FHEONHEController::clear_rotation_keys() {
    context->ClearEvalMultKeys();
    context->ClearEvalAutomorphismKeys();
    CryptoContextFactory<lbcrypto::DCRTPoly>::ReleaseAllContexts();
}

/**
 * @brief Clear all bootstrapping and rotation keys in the FHE context.
 *
 * This function removes all stored bootstrapping and rotation keys up to the 
 * specified number of bootstrapping slots, allowing new keys to be generated.
 *
 * @param bootstrap_num_slots  Number of bootstrapping slots to clear.
 */
void FHEONHEController::clear_bootstrapping_and_rotation_keys(int bootstrap_num_slots) {
    //This lines would free more or less 1GB or precomputations, but requires access to the GetFHE function
    // FHECKKSRNS* derivedPtr = dynamic_cast<FHECKKSRNS*>(context->GetScheme()->GetFHE().get());
    // derivedPtr->m_bootPrecomMap.erase(bootstrap_num_slots);

    context->ClearEvalMultKeys();
    context->ClearEvalAutomorphismKeys();
    CryptoContextFactory<lbcrypto::DCRTPoly>::ReleaseAllContexts();
}


/**
 * @brief Clear the entire FHE context, including multiplication, bootstrapping, and rotation keys.
 *
 * This function removes all keys stored in the context, allowing a fresh setup 
 * or reinitialization of the FHE environment.
 *
 * @param bootstrapping_key_slots  Number of bootstrapping slots to clear.
 */
void FHEONHEController::clear_context(int bootstrapping_key_slots) {
    
    if (bootstrapping_key_slots != 0)
        clear_bootstrapping_and_rotation_keys(bootstrapping_key_slots);
    else
        clear_rotation_keys();
}


/**
 * @brief Bootstrap a ciphertext to refresh its noise budget.
 *
 * This function applies bootstrapping to the input ciphertext, effectively 
 * reducing accumulated noise and enabling further homomorphic operations. 
 * The bootstrapping level controls the depth and parameters used.
 *
 * @param encryptedInput  Ciphertext to be bootstrapped.
 * @param encode_level    Bootstrapping level as defined in OpenFHE (e.g., 1 or 2).
 *
 * @return Refreshed ciphertext after bootstrapping.
 */
Ctext FHEONHEController::bootstrap_function(Ctext& encryptedInput, int encode_level){
    Ctext boots_ciphertext = context->EvalBootstrap(encryptedInput, encode_level);
    return boots_ciphertext;
}

/**
 * @brief Bootstrap batched ciphertext to refresh the noise budget in all of them.
 *
 * This function applies bootstrapping to the input ciphertexts over the batch, effectively 
 * reducing accumulated noise and enabling further homomorphic operations. 
 * The bootstrapping level controls the depth and parameters used.
 *
 * @param encryptedInput  Ciphertext to be bootstrapped.
 * @param encode_level    Bootstrapping level as defined in OpenFHE (e.g., 1 or 2).
 *
 * @return Refreshed ciphertext after bootstrapping.
 */

// vector<Ctext> FHEONHEController::batch_bootstrap_function(vector<Ctext>& encryptedInputs, int inputChannels, int encode_level){
//     vector<Ctext> batch_ciphertexts(inputChannels);
//     for(int b=0; b<inputChannels; b++){
//         batch_ciphertexts[b] = context->EvalBootstrap(encryptedInputs[b], encode_level);
//     }
//     return batch_ciphertexts;
// }

vector<Ctext> FHEONHEController::batch_bootstrap_function(vector<Ctext>& encryptedInputs, int inputChannels, int encode_level) {
    vector<Ctext> batch_ciphertexts(inputChannels);
    int numThreads = min(inputChannels, (int)thread::hardware_concurrency());
    vector<thread> threads(numThreads);

    // Worker function for a range of channels
    auto worker = [&](int start, int end) {
        for (int b = start; b < end; b++) {
            batch_ciphertexts[b] = context->EvalBootstrap(encryptedInputs[b], encode_level);
        }
    };

    // Divide channels evenly among threads
    int block = (inputChannels + numThreads - 1) / numThreads;
    for (int t = 0; t < numThreads; t++) {
        int start = t * block;
        int end = min(start + block, inputChannels);
        threads[t] = thread(worker, start, end);
    }
    for (auto &th : threads) th.join();

    return batch_ciphertexts;
}

/**
 * @brief Encrypt a vector of input data into a packed ciphertext.
 *
 * This function takes a vector of double-precision values and encrypts them 
 * into a single packed ciphertext suitable for homomorphic computations.
 *
 * @param inputData  Vector of data to be encrypted.
 *
 * @return Ciphertext containing the encrypted input data.
 */
Ctext FHEONHEController::encrypt_input(vector<double>& inputData) {
    Ptext plaintext = context->MakeCKKSPackedPlaintext(inputData, 1, 1);
    plaintext->SetLength(inputData.size());
    auto encryptImage = context->Encrypt(keyPair.publicKey, plaintext);
    return encryptImage;
}


/**
 * @brief Encode a vector of input data into a packed plaintext.
 *
 * This function encodes a vector of double-precision values into a plaintext 
 * suitable for homomorphic encryption, using the specified encoding level.
 *
 * @param inputData     Vector of data to be encoded.
 * @param encode_level  Encoding level to use for the plaintext.
 *
 * @return Plaintext containing the encoded input data.
 */
Ptext FHEONHEController::encode_input(vector<double>& inputData, int encode_level) {
    Ptext plaintext = context->MakeCKKSPackedPlaintext(inputData, 1, encode_level);
    return plaintext;
}

/**
 * @brief Encode a vector of input data into a packed plaintext with a specified number of slots.
 *
 * This function encodes a vector of double-precision values into a plaintext 
 * suitable for homomorphic encryption, using the specified number of slots 
 * and encoding level.
 *
 * @param inputData     Vector of data to be encoded.
 * @param num_slots     Number of slots to use in the plaintext.
 * @param encode_level  Encoding level to use for the plaintext.
 *
 * @return Plaintext containing the encoded input data.
 */

Ptext FHEONHEController::encode_input(vector<double>& inputData, int num_slots, int encode_level) {
    int numElements = nextPowerOf2(num_slots);
    Ptext plaintext = context->MakeCKKSPackedPlaintext(inputData, 1, encode_level, nullptr, numElements);
    return plaintext;
}

/**
 * @brief Encode a vector for use in a shortcut layer.
 *
 * This function encodes a vector of double-precision values into a plaintext 
 * suitable for the shortcut layer in homomorphic neural network operations, 
 * using the specified column square size.
 *
 * @param inputData    Vector of data to be encoded.
 * @param cols_square  Size of the column square for the encoding.
 *
 * @return Plaintext containing the encoded shortcut kernel data.
 */
Ptext FHEONHEController::encode_shortcut_kernel(vector<double>& inputData, int cols_square) {
    int dim1 = inputData.size();
    vector<double> main_kernel;
    for(int t =0; t< dim1; t++){
        double cell_value = inputData[t];
        vector<double> repeated(cols_square, cell_value);
        main_kernel.insert(main_kernel.end(), repeated.begin(), repeated.end());
    }
    Ptext plaintext = context->MakeCKKSPackedPlaintext(main_kernel, 1, 1);
    return plaintext;
}

/**
 * @brief Encode a vector of bias data into a plaintext.
 *
 * This function encodes a vector of double-precision bias values into a 
 * plaintext suitable for homomorphic neural network operations, using the 
 * specified column square size and encoding level.
 *
 * @param inputData     Vector of bias data to be encoded.
 * @param cols_square   Size of the column square for the encoding.
 * @param encode_level  Encoding level to use for the plaintext.
 *
 * @return Plaintext containing the encoded bias data.
 */
Ptext FHEONHEController::encode_bais_input(vector<double>& inputData, int cols_square, int encode_level) {
    int dim1 = inputData.size();
    vector<double> main_kernel;
    for(int t =0; t< dim1; t++){
        double cell_value = inputData[t];
        vector<double> repeated(cols_square, cell_value);
        main_kernel.insert(main_kernel.end(), repeated.begin(), repeated.end());
    }

    Ptext plaintext = context->MakeCKKSPackedPlaintext(main_kernel, 1, encode_level);
    return plaintext;
}

/**
 * @brief Re-encrypt a plaintext vector into a ciphertext.
 *
 * This function takes a plaintext containing encoded data and encrypts it 
 * into a ciphertext suitable for homomorphic computations.
 *
 * @param plaintextData  Plaintext data to be re-encrypted.
 *
 * @return Ciphertext containing the encrypted data.
 */
Ctext FHEONHEController::reencrypt_data(Ptext plaintextData) {
    
    auto encryptedData = context->Encrypt(keyPair.publicKey, plaintextData);
    return encryptedData;
}

/**
 * @brief Decrypt a ciphertext into a plaintext vector.
 *
 * This function takes an encrypted ciphertext and decrypts it into a plaintext 
 * vector suitable for further processing or inspection.
 *
 * @param encryptedinputData  Ciphertext to be decrypted.
 * @param cols                Number of elements in the decrypted vector.
 *
 * @return Plaintext containing the decrypted data.
 */
Ptext FHEONHEController::decrypt_data(Ctext encryptedinputData, int cols) {
    
    Ptext plaintextDec;
    context->Decrypt(keyPair.secretKey, encryptedinputData, &plaintextDec);
    plaintextDec->SetLength(cols);
    return plaintextDec;
}

/**
 * @brief Encrypt a 3D kernel matrix into a 2D vector of ciphertexts.
 *
 * This function takes a 3D vector of double-precision kernel values and encrypts 
 * each 2D slice into a vector of ciphertexts suitable for homomorphic convolution 
 * operations.
 *
 * @param kernelData   3D vector containing kernel values to be encrypted.
 * @param cols_square  Size of the column square for the encryption.
 *
 * @return 2D vector of ciphertexts representing the encrypted kernel.
 */
vector<vector<Ctext>> FHEONHEController::encrypt_kernel(vector<vector<vector<double>>>& kernelData, int cols_square){
    size_t dim1 = kernelData.size();
    if (dim1 == 0) return {};
    size_t dim2 = kernelData[0].size();
    if (dim2 == 0) return {};
    size_t dim3 = kernelData[0][0].size();
    if (dim3 == 0) return {}; 

    vector<vector<Ctext>> encrypt_kernel; 
    for (size_t k=0; k< dim1; k++){
        vector<Ctext> filters;
        for (size_t i=0; i< dim2; i++){
            for (size_t j=0; j< dim3 ; j++){
                double cell_value  = kernelData[k][i][j];
                vector<double> repeated(cols_square, cell_value);
                Ctext encrypted_val = encrypt_input(repeated);
                filters.push_back(encrypted_val);
            }
        }
        encrypt_kernel.push_back(filters);
    }
    return encrypt_kernel;
}

/**
 * @brief Encode kernel data for fully connected layers.
 *
 * This function takes a vector of double-precision kernel values and encodes 
 * them into plaintexts suitable for homomorphic operations in fully connected layers.
 *
 * @param kernelData   Vector containing kernel values to be encoded.
 * @param cols_square  Size of the column square for the encoding.
 *
 * @return Vector of plaintexts representing the encoded kernel data.
 */
vector<Ptext> FHEONHEController::encode_kernel(vector<double>& kernelData, int cols_square){
    size_t dim1 = kernelData.size();
    if (dim1 == 0) return {};

    vector<Ptext> encrypt_kernel; 
     for (size_t j=0; j< dim1 ; j++){
        double cell_value  = kernelData[j];
        vector<double> repeated(cols_square, cell_value);
        Ptext encrypted_val = encode_input(repeated);
        encrypt_kernel.push_back(encrypted_val);
    }
    return encrypt_kernel;
}

/**
 * @brief Encode and replicate kernel values for homomorphic convolution.
 *
 * This function selects values from the kernel positions for all kernels, 
 * repeats them by the square of the width, concatenates them into a single 
 * long vector, and encodes the result. The output is a k^2 vector of repeated 
 * kernel values suitable for homomorphic convolution operations.
 *
 * @param kernelData   Vector containing kernel values to be encoded and replicated.
 * @param cols_square  Size of the column square for encoding.
 *
 * @return Vector of plaintexts representing the encoded and replicated kernel values.
 */
vector<Ptext> FHEONHEController::encode_kernel(vector<vector<vector<double>>>& kernelData, int cols_square){
    size_t dim1 = kernelData.size();
    if (dim1 == 0) return {};
    size_t dim2 = kernelData[0].size();
    if (dim2 == 0) return {};
    size_t dim3 = kernelData[0][0].size();
    if (dim3 == 0) return {}; 
    // cout <<"input kernel shape: " << dim1 << "*" << dim2 << "*" << dim3 <<endl;

    int kernelWidth_sq = pow(dim2, 2);
    vector<vector<double>> main_kernel(kernelWidth_sq, vector<double>());
    for (size_t k=0; k< dim1; k++){
        vector<vector<double>> filters;
        for (size_t i=0; i< dim2; i++){
            for (size_t j=0; j< dim3 ; j++){
                double cell_value  = kernelData[k][i][j];
                vector<double> repeated(cols_square, cell_value);
                filters.push_back(repeated);
            }
        }
        for(int t =0; t< kernelWidth_sq; t++){   
            main_kernel[t].insert(main_kernel[t].end(), filters[t].begin(), filters[t].end());
        }
    }
    vector<Ptext> encoded_kernel;
    for( int s =0; s< kernelWidth_sq; s++){
        // cout << "Kernel size: " << main_kernel[s].size() << endl;
        Ptext encoded_val = encode_input(main_kernel[s]);
        encoded_kernel.push_back(encoded_val);
    }
    return encoded_kernel;
}

/**
 * @brief Adjust the number of slots in a ciphertext after downsampling.
 *
 * This function modifies the number of slots in the given ciphertext to improve 
 * performance by reducing the size of the polynomial being processed.
 *
 * @param encryptedInput  Ciphertext whose number of slots will be adjusted.
 * @param num_slots       Desired number of slots in the ciphertext.
 *
 * @return Ciphertext with the updated number of slots.
 */

Ctext FHEONHEController::change_num_slots(Ctext& encryptedInput, uint32_t num_slots){
    encryptedInput->SetSlots(1 << num_slots);
    return encryptedInput;
}

/**
 * @brief Encode kernel data optimized for 3x3 kernels with padding of 1.
 *
 * This function takes 3D kernel data and encodes it into plaintexts, optimized 
 * for kernels of size 3x3 and padding of 1, using the specified encoding level.
 *
 * @param kernelData    3D vector containing kernel values to be encoded.
 * @param cols_square   Size of the column square for encoding.
 * @param encode_level  Encoding level to use for the plaintexts.
 *
 * @return Vector of plaintexts representing the encoded and optimized kernel data.
 */
vector<Ptext> FHEONHEController::encode_kernel_optimized(vector<vector<vector<double>>>& kernelData, int cols_square, int encode_level) {
    size_t dim1 = kernelData.size();
    if (dim1 == 0) return {};
    size_t dim2 = kernelData[0].size();
    if (dim2 == 0) return {};
    size_t dim3 = kernelData[0][0].size();
    if (dim3 == 0) return {};

    int kernelWidth_sq = pow(dim2, 2);
    vector<vector<double>> main_kernel(kernelWidth_sq, vector<double>());
    for (size_t k=0; k< dim1; k++){
        vector<vector<double>> filters;
        for (size_t i=0; i< dim2; i++){
            for (size_t j=0; j< dim3 ; j++){
                double cell_value  = kernelData[k][i][j];
                // if(cell_value == 0)
                //     cell_value = 1e-40;
                vector<double> repeated(cols_square, cell_value);
                filters.push_back(repeated);
            }
        }
        for(int t =0; t< kernelWidth_sq; t++){   
            main_kernel[t].insert(main_kernel[t].end(), filters[t].begin(), filters[t].end());
        }
    }

    int vector_width = sqrt(cols_square);
    vector<vector<double>> bin_masks = {
        build_tiled_mask(vector_width + 1, 0, vector_width - 1, cols_square, dim1),
        build_tiled_mask(vector_width, 0, cols_square, cols_square, dim1),
        build_tiled_mask(vector_width, 0, vector_width - 1, cols_square, dim1),
        build_tiled_mask(1, 0, vector_width - 1, cols_square, dim1),
        build_tiled_mask(0, 0, cols_square, cols_square, dim1),
        build_tiled_mask(0, 1, vector_width - 1, cols_square, dim1),
        build_tiled_mask(1, vector_width - 1, vector_width - 1, cols_square, dim1),
        build_tiled_mask(0, vector_width, cols_square, cols_square, dim1),
        build_tiled_mask(0, vector_width + 1, vector_width - 1, cols_square, dim1)
    };
        
    vector<Ptext> encoded_kernel;
    for (int s = 0; s < kernelWidth_sq; ++s) {
        if (s >= static_cast<int>(bin_masks.size())) {
            std::cerr << "Error: bin_mask index out of range!" << std::endl;
            return encoded_kernel;
        }

        // Multiply main_kernel[s] element-wise with bin_masks[s]
        
        vector<double> cleaned_kernel(main_kernel[s].size());
        for (size_t i = 0; i < main_kernel[s].size(); ++i) {
            cleaned_kernel[i] = main_kernel[s][i] * bin_masks[s][i];
        }

        // Encode the cleaned kernel
        int numElements = nextPowerOf2(main_kernel[s].size());
        Ptext encoded_val = encode_input(cleaned_kernel, numElements, encode_level);
        encoded_kernel.push_back(encoded_val);
    }

    return encoded_kernel;
}

/**
 * @brief Read the predicted label from encrypted inference data.
 *
 * This function decrypts and reads the predicted label from the given encrypted 
 * inference data. The result can be written to an output file.
 *
 * @param inferencedData  Ciphertext containing the inference results.
 * @param num_slots        Number of elements in the ciphertext.
 * @param outFile          Output file stream to write the predicted label.
 *
 * @return The predicted label as an integer.
 */
int FHEONHEController::read_inferenced_label(Ctext inferencedData, int num_slots,  ofstream& outFile){
    auto decryptedValue = decrypt_data(inferencedData, num_slots);
    auto decryptedVector = decryptedValue->GetRealPackedValue();
    auto maxElementIt = max_element(decryptedVector.begin(), decryptedVector.end());
    int maxIndex = distance(decryptedVector.begin(), maxElementIt);
    cout << "Predicted Value : " << maxIndex << " Weight:  " << decryptedVector[maxIndex] << endl;
    
    if (outFile.is_open()) {
        outFile << maxIndex << endl;
    } else {
        cout << "Unable to open file." << endl;
    }
    return 0;
}

/**
 * @brief Determine the minimum and maximum values from encrypted data.
 *
 * This helper function decrypts the inference data and computes the minimum 
 * and maximum values across all elements.
 *
 * @param inferencedData  Ciphertext containing the inference data.
 * @param num_slots       Number of elements in the ciphertext.
 *
 * @return An integer representing the computed min or max value, depending on implementation.
 */
int FHEONHEController::read_minmax(Ctext inferencedData, int num_slots) {
    auto decryptedValue = decrypt_data(inferencedData, num_slots);
    auto decryptedVector = decryptedValue->GetRealPackedValue();

    // cout << "Decrypted Vector " << decryptedVector << endl;
    auto maxElementIt = max_element(decryptedVector.begin(), decryptedVector.end());
    int maxIndex = distance(decryptedVector.begin(), maxElementIt);
    auto minElementIt = min_element(decryptedVector.begin(), decryptedVector.end());
    int minIndex = distance(decryptedVector.begin(), minElementIt);
    cout << "------------------------------------------------------------------ " << endl;
    cout << "Range [ " << decryptedVector[minIndex] << " , " << decryptedVector[maxIndex] <<" ]" << endl;
    cout << "Index: " << maxIndex << endl;
    cout << "------------------------------------------------------------------ " << endl;
    return 0;
}

/**
 * @brief Retrieve the maximum value from encrypted convolution data for ReLU scaling.
 *
 * This temporary function decrypts the inference data and returns the maximum 
 * value, which can be used for scaling in the ReLU operation.
 *
 * @param inferencedData  Ciphertext containing the inference data.
 * @param num_slots       Number of elements in the ciphertext.
 *
 * @return Maximum value as an integer.
 */
int FHEONHEController::read_scaling_value(Ctext inferencedData, int num_slots){
    auto decryptedValue = decrypt_data(inferencedData, num_slots);
    auto decryptedVector = decryptedValue->GetRealPackedValue();

    double maxAbsValue = *std::max_element(decryptedVector.begin(), decryptedVector.end(), [](int a, int b) {
        return std::abs(a) < std::abs(b);
    });
    int roundedMaxAbsValue = static_cast<int>(std::ceil(std::abs(maxAbsValue)));
    return roundedMaxAbsValue;
}

/**
 * @brief Build a tiled mask for optimized convolution operations.
 *
 * This function generates a mask of 0s and 1s to be element-wise multiplied 
 * with repeated kernel values in optimized convolution operations.
 *
 * @param starting_padding  Number of zeros to pad at the start of the mask.
 * @param ending_padding    Number of zeros to pad at the end of the mask.
 * @param window_length     Length of the convolution window.
 * @param max_length        Maximum length of the mask.
 * @param tile_count        Number of times the pattern should be repeated.
 *
 * @return Vector of doubles representing the tiled mask.
 */
vector<double> FHEONHEController::build_tiled_mask(int starting_padding, int ending_padding, 
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


/**
 * @brief Read the predicted label from encrypted inference data.
 *
 * This function decrypts and reads the predicted label from the given encrypted 
 * inference data. The result can be written to an output file.
 *
 * @param inferencedData  Ciphertext containing the inference results.
 * @param num_slots        Number of elements in the ciphertext.
 * @param outFile          Output file stream to write the predicted label.
 *
 * @return The predicted label as an integer.
 */
vector<int> FHEONHEController::read_batch_inferenced_label(Ctext inferencedData,  int batchSize, int numClasses,  ofstream& outFile, int baseIndex){
     int totalElements = batchSize * numClasses;

    // Decrypt
    auto decryptedVal = decrypt_data(inferencedData, totalElements);
    auto decryptedVec = decryptedVal->GetRealPackedValue();
    cout << "DecryptedVec: " << decryptedVec << endl;

    vector<int> predictedLabels(batchSize);

    // Process each batch separately
    for (int b = 0; b < batchSize; b++) {
        int startIdx = b * numClasses;
        int endIdx   = startIdx + numClasses;

        // Find max in this batch slice
        
        auto maxIt = max_element(
            decryptedVec.begin() + startIdx, 
            decryptedVec.begin() + endIdx
        );
        int label = distance(decryptedVec.begin() + startIdx, maxIt);
        
        // cout << "startIdx: " << startIdx
        //     << "  endIdx: " << endIdx
        //     << "  Elements: [ ";
        // for (int i = startIdx; i < endIdx; i++) {
        //     cout << decryptedVec[i] << " ";
        // }
        // cout << "]\n";

        predictedLabels[b] = label;

        cout << "Batch " << baseIndex+b 
             << " -> Predicted Label: " << label 
             << " (Score: " << *maxIt << ")" << endl;

        if (outFile.is_open()) {
            outFile << label << endl;
        }
    }

    return predictedLabels;
}


vector<int> FHEONHEController::read_batch_inferenced_label_multiple_outputs(vector<Ctext> inferencedData,  int batchSize, int numClasses,  ofstream& outFile, int baseIndex){
    
    vector<int> predictedLabels(batchSize);
    // Process each batch separately
    for (int b = 0; b < batchSize; b++) {
        auto decryptedVal = decrypt_data(inferencedData[b], numClasses);
        auto decryptedVec = decryptedVal->GetRealPackedValue();
        int startIdx = 0;
        int endIdx   = startIdx + numClasses;

        // Find max in this batch slice
        auto maxIt = max_element(
            decryptedVec.begin() + startIdx, 
            decryptedVec.begin() + endIdx
        );
        int label = distance(decryptedVec.begin() + startIdx, maxIt);
        predictedLabels[b] = label;

        cout << "Batch " << baseIndex+b 
             << " -> Predicted Label: " << label 
             << " (Score: " << *maxIt << ")" << endl;

        if (outFile.is_open()) {
            outFile << label << endl;
        }
    }
    return predictedLabels;
}


vector<int> FHEONHEController::read_batch_scaling_values(vector<Ctext>& encryptedInputs, int inputChannels, int num_slots){
    vector<int> maxVec;
    for(int i=0; i<inputChannels; i++){
        auto decryptedVal = decrypt_data(encryptedInputs[i], num_slots);
        auto decryptedVec = decryptedVal->GetRealPackedValue();
        double maxAbsVal = *std::max_element(decryptedVec.begin(), decryptedVec.end(), [](double a, double b) {
            return std::abs(a) < std::abs(b);
        });
        int roundedMax = static_cast<int>(std::ceil(std::abs(maxAbsVal)));
        maxVec.push_back(roundedMax);
    }
    return maxVec;
}

int FHEONHEController::read_batch_minmax(vector<Ctext>& encryptedInputs, int inputChannels,  int num_slots) {
    
    vector<int> maxVec;
    for(int i=0; i<inputChannels; i++){
        auto decryptedVal = decrypt_data(encryptedInputs[i], num_slots);
        auto decryptedVec = decryptedVal->GetRealPackedValue();

        auto maxElementIt = max_element(decryptedVec.begin(), decryptedVec.end());
        int maxIndex = distance(decryptedVec.begin(), maxElementIt);
        auto minElementIt = min_element(decryptedVec.begin(), decryptedVec.end());
        int minIndex = distance(decryptedVec.begin(), minElementIt);
        cout << "------------------------------------------------------------------ " << endl;
        cout << "Range [ " << decryptedVec[minIndex] << " , " << decryptedVec[maxIndex] <<" ]" << endl;
        cout << "Index: " << maxIndex << endl;
        cout << "------------------------------------------------------------------ " << endl;
    }
    return 0;
}

/**
 * @brief Decrypts and prints a batch of encrypted inputs.
 *
 * This function iterates through a vector of ciphertexts, decrypts each
 * encrypted input using the internally stored secret key, and prints
 * the real packed values to stdout.
 *
 * @param encryptedInputs Vector of encrypted ciphertext inputs.
 * @param inputChannels Number of encrypted inputs (channels) to decrypt.
 * @param num_slots Number of CKKS slots to decode.
 *
 * @return Returns 0 on successful execution.
 */
int FHEONHEController::decrypt_batch_data(vector<Ctext>& encryptedInputs, int inputChannels,  int num_slots) {
    
    vector<int> maxVec;
    for(int i=0; i<inputChannels; i++){
        auto decryptedVal = decrypt_data(encryptedInputs[i], num_slots);
        auto decryptedVec = decryptedVal->GetRealPackedValue();
        cout << "------------------------------------------------------------------ " << endl;
        cout << decryptedVec << endl;
        cout << "------------------------------------------------------------------ " << endl;
    }
    return 0;
}


/**
 * @brief Decrypts a packed CKKS ciphertext and prints the result.
 *
 * This function decrypts the provided ciphertext using the controller's
 * secret key, sets the plaintext length to the specified number of slots,
 * extracts the complex CKKS packed values, and prints them.
 *
 * @param encryptedpackedVector Encrypted CKKS ciphertext.
 * @param num_slots Number of slots to extract from the decrypted plaintext.
 */
void FHEONHEController::decrypt_and_print(Ctext encryptedpackedVector, int num_slots) {
    
    Ptext plaintextDec;
    context->Decrypt(keyPair.secretKey, encryptedpackedVector, &plaintextDec);
    plaintextDec->SetLength(num_slots);
    vector<complex<double>> finalResult = plaintextDec->GetCKKSPackedValue();
    cout << finalResult << endl;
    cout << endl;
}

/**
 * @brief Decrypts ciphertext using a provided private key.
 *
 * This function decrypts the given ciphertext using the specified secret key,
 * sets the plaintext length, and returns the decrypted plaintext object.
 *
 * @param sk Private key used for decryption.
 * @param encryptedinputData Ciphertext to decrypt.
 * @param cols Number of plaintext slots to retain.
 *
 * @return Decrypted plaintext (Ptext).
 */
Ptext FHEONHEController::decrypt_data_with_key(PrivateKey<DCRTPoly> &sk,
                                               Ctext encryptedinputData,
                                               int cols) {

    Ptext plaintextDec;
    context->Decrypt(sk, encryptedinputData, &plaintextDec);
    plaintextDec->SetLength(cols);
    return plaintextDec;
}

/**
 * @brief Reads and computes the scaling value from decrypted inference data.
 *
 * This function decrypts inference output, retrieves the real packed values,
 * and computes the maximum absolute value across all slots. The result is
 * rounded up to the nearest integer and returned as the scaling factor.
 *
 * @param sk Private key used for decryption.
 * @param inferencedData Encrypted inference output.
 * @param num_slots Number of slots to decode.
 *
 * @return Integer representing the ceiling of the maximum absolute value.
 */
int FHEONHEController::read_scaling_value_with_key(PrivateKey<DCRTPoly> &sk,
                                                   Ctext inferencedData,
                                                   int num_slots) {
    // int roundedMaxAbsValue = 10; // Temporary hardcoded value for testing
    auto decryptedValue = decrypt_data_with_key(sk, inferencedData, num_slots);
    auto decryptedVector = decryptedValue->GetRealPackedValue();

    // cout << endl
    //     << "--------------------------------------------------- " << endl
    //     << endl;
    // cout << "Decrypted Vector for Scaling Value: " << decryptedVector << endl;
    // cout << endl
    //     << "--------------------------------------------------- " << endl;

    double maxAbsValue =
        *std::max_element(decryptedVector.begin(), decryptedVector.end(),
                            [](int a, int b) { return std::abs(a) < std::abs(b); });
    int roundedMaxAbsValue = static_cast<int>(std::ceil(std::abs(maxAbsValue)));
    return roundedMaxAbsValue;
}


/**
 * @brief Decrypts inference output and determines the predicted label.
 *
 * This function decrypts the inference ciphertext, extracts the real packed
 * values, determines the index of the maximum value (argmax), and prints the
 * predicted label and its corresponding weight. If the output file stream is
 * open, the predicted label is written to the file.
 *
 * @param sk Private key used for decryption.
 * @param inferencedData Encrypted inference result.
 * @param num_slots Number of slots to decode.
 * @param outFile Output file stream to write predicted label.
 *
 * @return Index corresponding to the predicted label.
 */
int FHEONHEController::read_inferenced_label_with_key(PrivateKey<DCRTPoly> &sk,
                                                      Ctext inferencedData,
                                                      int num_slots,
                                                      ofstream &outFile) {
    auto decryptedValue = decrypt_data_with_key(sk, inferencedData, num_slots);
    auto decryptedVector = decryptedValue->GetRealPackedValue();
    auto maxElementIt =
        max_element(decryptedVector.begin(), decryptedVector.end());
    int maxIndex = distance(decryptedVector.begin(), maxElementIt);
    cout << "Predicted Value : " << maxIndex
        << " Weight:  " << decryptedVector[maxIndex] << endl;
    cout << "Decrypted Vector: " << decryptedVector << endl;

    if (outFile.is_open()) {
        outFile << maxIndex << endl;
    } else {
        cout << "Unable to open file." << endl;
    }
    return maxIndex;
}

void FHEONHEController::harness_generate_rotation_keys( CryptoContext<DCRTPoly> &crypto_context, PrivateKey<DCRTPoly> &sk,
          vector<int> rotations_positions, ofstream &key_file, bool sum_key) {

  if(sum_key){
    crypto_context->EvalSumKeyGen(sk);
  }
  crypto_context->EvalRotateKeyGen(sk, rotations_positions);
  
  if (key_file.is_open()) {
    if (!crypto_context->SerializeEvalAutomorphismKey(key_file, SerType::BINARY)) {
      cerr << "Error writing rotation keys" << std::endl;
      exit(1);
    }
  } else {
    cerr << "Error serializing Rotation keys" << endl;
    exit(1);
  }
}

void FHEONHEController::harness_generate_bootstrapping_and_rotation_keys( CryptoContext<DCRTPoly> &crypto_context, PrivateKey<DCRTPoly> &sk,
          vector<int> rotations_positions, ofstream &key_file, bool sum_key) {

  uint32_t numSlots = crypto_context->GetRingDimension() / 2;
  std::vector<uint32_t> levelBudget = {3, 3};
  crypto_context->EvalBootstrapSetup(levelBudget);
  crypto_context->EvalBootstrapKeyGen(sk, numSlots);
  crypto_context->EvalRotateKeyGen(sk, rotations_positions);

  if(sum_key){
    crypto_context->EvalSumKeyGen(sk);
  }
  
  if (key_file.is_open()) {
    if (!crypto_context->SerializeEvalAutomorphismKey(key_file, SerType::BINARY)) {
      cerr << "Error writing rotation keys" << std::endl;
      exit(1);
    }
  } else {
    cerr << "Error serializing Rotation keys" << endl;
    exit(1);
  }
}

void FHEONHEController::harness_clear_bootstrapping_and_rotation_keys(CryptoContext<DCRTPoly> &crypto_context) {
  // crypto_context->ClearEvalMultKeys();
  crypto_context->ClearEvalAutomorphismKeys();
  // CryptoContextFactory<lbcrypto::DCRTPoly>::ReleaseAllContexts();
}

void FHEONHEController::read_evaluation_keys(CryptoContext<DCRTPoly> &crypto_context, const string &pubkey_dir,
    const string &mult_file, const string &rot_file) {

	crypto_context->ClearEvalMultKeys();
	crypto_context->ClearEvalAutomorphismKeys();
	// DO NOT call ReleaseAllContexts() as it breaks existing ciphertexts

	// Open files in READ mode
	ifstream multkey_file(pubkey_dir + mult_file, ios::in | ios::binary);
	ifstream rotkey_file(pubkey_dir + rot_file, ios::in | ios::binary);

	// Deserialize Multiplication key into EXISTING context
	if (!multkey_file.is_open() ||
		!crypto_context->DeserializeEvalMultKey(multkey_file, SerType::BINARY)) {
		throw std::runtime_error("Failed to load relinearization key from " +
								pubkey_dir + mult_file);
	}

	// Deserialize Automorphism keys into EXISTING context
	if (!rotkey_file.is_open() || !crypto_context->DeserializeEvalAutomorphismKey(
										rotkey_file, SerType::BINARY)) {
		throw std::runtime_error("Failed to load rotation keys from " + pubkey_dir +
								rot_file);
	}
	// cout << "Successfully loaded evaluation keys" << endl;
}

void FHEONHEController::read_evaluation_keys(CryptoContext<DCRTPoly> &crypto_context, const string &pubkey_dir,
    const string &mult_file, const string &rot_file, const string &sk_path) {

  // Clear existing evaluation keys from the context
  crypto_context->ClearEvalMultKeys();
  crypto_context->ClearEvalAutomorphismKeys();
  // DO NOT call ReleaseAllContexts() as it breaks existing ciphertexts

  // Open files in READ mode
  ifstream multkey_file(pubkey_dir + mult_file, ios::in | ios::binary);
  ifstream rotkey_file(pubkey_dir + rot_file, ios::in | ios::binary);

  // Deserialize Multiplication key into EXISTING context
  if (!multkey_file.is_open() ||
      !crypto_context->DeserializeEvalMultKey(multkey_file, SerType::BINARY)) {
    throw std::runtime_error("Failed to load relinearization key from " +
                             pubkey_dir + mult_file);
  }

  // Deserialize Automorphism keys into EXISTING context
  if (!rotkey_file.is_open() || !crypto_context->DeserializeEvalAutomorphismKey(
                                    rotkey_file, SerType::BINARY)) {
    throw std::runtime_error("Failed to load rotation keys from " + pubkey_dir +
                             rot_file);
  }
}