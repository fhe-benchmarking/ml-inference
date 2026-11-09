
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
#include <limits>
#include <cmath>
#include <sstream>
#include <malloc.h>
// #include <thread>
namespace fs = std::filesystem;

#include "FHEONHEController.h"
#include "UtilsBatchData.h"

/**
 * @brief Update absolute file paths for FHE context and serialization keys.
 *
 * Appends trailing slashes to keys_folder if necessary and constructs full paths
 * for public, secret, multiplication, and sum keys.
 */
void FHEONHEController::update_key_paths() {
    fs::path base(keys_folder);
    if (!base.empty() && base.filename() == ".") {
        base = base.parent_path();
    }
    if (!base.empty() && base.string().back() == '/') {
        keys_folder = base.string();
    } else if (!base.empty()) {
        keys_folder = base.string() + "/";
    }

    crypto_context_file = keys_folder + "crypto-context.bin";
    multi_key_file = keys_folder + "mult-keys.bin";
    sum_key_file = keys_folder + "sum-keys.bin";
    public_key_file = keys_folder + "public-key.bin";
    secret_key_file = keys_folder + "secret-key.bin";
}

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
double getlogPQ(const DCRTPoly& poly) {
    int n = poly.GetNumOfElements();
    double logPQ = 0;
    for (int i = 0; i < n; i++) {
        auto qi = poly.GetParams()->GetParams()[i]->GetModulus(); 
        logPQ += log(qi.ConvertToDouble()) / log(2);
    }
    return logPQ;
}

/***************************************************************************************************
 * Security levels (FHEON_SECURITY_LEVEL, cmake -DSECURITY_LEVEL)
 ***************************************************************************************************/

SecurityLevel FHEONHEController::to_openfhe_security_level(int bits) {
    switch (bits) {
        case 0:   return HEStd_NotSet;
        case 128: return HEStd_128_classic;
        case 192: return HEStd_192_classic;
        case 256: return HEStd_256_classic;
        default:
            throw std::invalid_argument("FHEON security level must be 0 (performance efficient), 128, 192 or 256, not "
                                        + to_string(bits));
    }
}

string FHEONHEController::security_level_name(int bits) {
    return bits == 0 ? string("performance efficient (HEStd_NotSet)")
                     : "standard " + to_string(bits) + "-bit (HEStd_" + to_string(bits) + "_classic)";
}

/**
 * Sets the ring dimension, batch size, secret distribution and security level of a CKKS context.
 *
 * Performance efficient (0): the ring dimension and slots of the model, a sparse ternary secret and HEStd_NotSet.
 * Standard levels: no ring dimension. OpenFHE estimates log QP of the chain (first modulus, scale, depth and, with
 * hybrid key switching, the P moduli of the digits) and picks the smallest ring dimension the HE standard allows for
 * that log QP at the level, then checks the actual log QP of the moduli it generated. The secret is uniform ternary,
 * which the standard tables cover; contexts with bootstrapping use SPARSE_ENCAPSULATED, a uniform ternary key whose
 * sparse twin only serves the modulus raising of bootstrapping. The batch size stays the model's slot count: the
 * models and controllers encode and rotate within num_slots, which the larger ring replicates (sparse packing).
 */
static SecretKeyDist fheon_apply_security(CCParams<CryptoContextCKKSRNS>& parameters, int bits, int ringDim,
                                          uint32_t numSlots, bool bootstrapping) {
    SecurityLevel level = FHEONHEController::to_openfhe_security_level(bits);
    SecretKeyDist dist;
    if (level == HEStd_NotSet) {
        dist = SPARSE_TERNARY;
        parameters.SetRingDim(1 << ringDim);
    } else {
        dist = bootstrapping ? SPARSE_ENCAPSULATED : UNIFORM_TERNARY;
    }
    parameters.SetBatchSize(numSlots);
    parameters.SetSecretKeyDist(dist);
    parameters.SetSecurityLevel(level);
    return dist;
}

/** GenCryptoContext at a standard level; a ring holding fewer than numSlots slots is raised to 2 * numSlots, which is
 *  secure as well (a larger ring at the same modulus), so the model keeps its packing. */
static CryptoContext<DCRTPoly> fheon_gen_standard(CCParams<CryptoContextCKKSRNS>& parameters, uint32_t numSlots) {
    try {
        return GenCryptoContext(parameters);
    } catch (const lbcrypto::OpenFHEException& e) {
        if (string(e.what()).find("batch size") == string::npos) {
            throw;
        }
        parameters.SetRingDim(2 * numSlots);
        return GenCryptoContext(parameters);
    }
}

/**
 * GenCryptoContext for the parameters of fheon_apply_security.
 *
 * At a standard level the ring dimension follows log QP = log Q + log P. With hybrid key switching P shrinks as the
 * number of digits grows (log P ~ log Q / digits), so when log QP lands just above the bound of a ring dimension, one
 * or two more digits bring it back under the bound of half that ring: the context is then twice as fast and its keys
 * about half the size. The digits of the model are kept unless more digits give a smaller ring.
 */
/** log Q + log P of a chain as OpenFHE builds it (FLEXIBLEAUTO: firstMod + depth * scale; hybrid key switching:
 *  P made of 60-bit primes covering the largest digit). Only used to pick the candidates of the ring optimization:
 *  GenCryptoContext checks every candidate against the standard. */
static uint32_t fheon_estimate_log_qp(uint32_t depth, uint32_t firstMod, uint32_t scale, uint32_t digits) {
    uint32_t towers = depth + 1;
    uint32_t perDigit = (towers + digits - 1) / digits;
    uint32_t maxBits = std::max(firstMod + (perDigit - 1) * scale, perDigit * scale);
    return firstMod + scale * depth + 60 * ((maxBits + 59) / 60);
}

/**
 * Ring optimization (HEConfig.minScaleBits): every ring dimension below the one of `best`, down to 2 * numSlots, is
 * tried with the fewest digits (up to maxDigits) and, with them, the largest scale (down to minScaleBits) whose log QP
 * fits it.
 * The smallest ring OpenFHE accepts is kept and `parameters` takes its scale, first modulus and digits.
 */
static CryptoContext<DCRTPoly> fheon_optimize_ring(CCParams<CryptoContextCKKSRNS>& parameters, CryptoContext<DCRTPoly> best,
                                                   uint32_t numSlots, int minScaleBits, int maxDigits) {
    uint32_t scale0 = parameters.GetScalingModSize();
    uint32_t first0 = parameters.GetFirstModSize();
    if (minScaleBits <= 0 || static_cast<uint32_t>(minScaleBits) > scale0) {
        return best;
    }
    uint32_t gap = first0 > scale0 ? first0 - scale0 : 0;
    uint32_t depth = parameters.GetMultiplicativeDepth();
    uint32_t d0 = parameters.GetNumLargeDigits() ? parameters.GetNumLargeDigits() : 3;
    uint32_t dMax = std::min(maxDigits > 0 ? static_cast<uint32_t>(maxDigits) : d0 + 3, depth + 1);
    SecurityLevel level = parameters.GetSecurityLevel();
    uint32_t ring0 = best->GetRingDimension();
    for (uint32_t ring = ring0 / 2; ring >= 2 * numSlots; ring /= 2) {
        uint32_t bound = StdLatticeParm::FindMaxQ(HEStd_ternary, level, ring);
        CryptoContext<DCRTPoly> found;
        uint32_t foundScale = 0, foundDigits = 0;
        /** the fewest digits first (key switching, i.e. every rotation, costs in proportion to the digits), then the
         *  largest scale with them */
        for (uint32_t d = d0; d <= dMax && !found; d++) {
            for (uint32_t s = scale0; s >= static_cast<uint32_t>(minScaleBits) && !found; s--) {
                if (fheon_estimate_log_qp(depth, s + gap, s, d) > bound) continue;
                CCParams<CryptoContextCKKSRNS> trial = parameters;
                trial.SetScalingModSize(s);
                trial.SetFirstModSize(s + gap);
                trial.SetNumLargeDigits(d);
                trial.SetRingDim(ring);
                try {
                    CryptoContext<DCRTPoly> cc = GenCryptoContext(trial);
                    if (cc->GetRingDimension() == ring) {
                        found = cc;
                        foundScale = s;
                        foundDigits = d;
                    }
                } catch (const lbcrypto::OpenFHEException&) {
                    // log QP above the bound of this ring (the estimate was optimistic): next candidate
                }
            }
        }
        if (!found) break;
        best = found;
        parameters.SetScalingModSize(foundScale);
        parameters.SetFirstModSize(foundScale + gap);
        parameters.SetNumLargeDigits(foundDigits);
        parameters.SetRingDim(ring);
    }
    if (best->GetRingDimension() != ring0) {
        cout << "Ring optimization: ring dimension " << ring0 << " -> " << best->GetRingDimension() << " (scale "
             << scale0 << " -> " << parameters.GetScalingModSize() << " bits, first modulus " << first0 << " -> "
             << parameters.GetFirstModSize() << " bits, " << parameters.GetNumLargeDigits() << " key-switching digits)" << endl;
    }
    return best;
}

static CryptoContext<DCRTPoly> fheon_gen_context(CCParams<CryptoContextCKKSRNS>& parameters, uint32_t numSlots,
                                                 int minScaleBits = 0, int maxDigits = 0) {
    if (parameters.GetSecurityLevel() == HEStd_NotSet) {
        return GenCryptoContext(parameters);
    }
    uint32_t digits = parameters.GetNumLargeDigits() ? parameters.GetNumLargeDigits() : 3;
    CryptoContext<DCRTPoly> best = fheon_gen_standard(parameters, numSlots);
    uint32_t bestDigits = digits;
    uint32_t primes = parameters.GetMultiplicativeDepth() + 1;
    for (uint32_t d = digits + 1; d <= std::min(digits + 3, primes); d++) {
        CCParams<CryptoContextCKKSRNS> trial = parameters;
        trial.SetNumLargeDigits(d);
        CryptoContext<DCRTPoly> cc;
        try {
            cc = fheon_gen_standard(trial, numSlots);
        } catch (const lbcrypto::OpenFHEException&) {
            continue;  // OpenFHE cannot spread the towers over d digits
        }
        if (cc->GetRingDimension() < best->GetRingDimension()) {
            best = cc;
            bestDigits = d;
        }
    }
    if (bestDigits != digits) {
        cout << "Key switching digits raised from " << digits << " to " << bestDigits << ": log QP then fits ring dimension "
             << best->GetRingDimension() << " at this security level" << endl;
        parameters.SetNumLargeDigits(bestDigits);
    }
    return fheon_optimize_ring(parameters, best, numSlots, minScaleBits, maxDigits);
}

/**
 * Full packing at a standard level (HEConfig.fillSlots): the context generated for numSlots slots is generated again
 * with the same ring dimension (and the digits fheon_gen_context chose) and a batch size of ringDim / 2. The security of
 * the context does not change (same ring, same moduli). numSlots is updated; nothing changes when the ring already
 * holds numSlots slots or at the performance-efficient level.
 */
static CryptoContext<DCRTPoly> fheon_fill_slots(CCParams<CryptoContextCKKSRNS>& parameters, CryptoContext<DCRTPoly> context,
                                                int& numSlots) {
    if (parameters.GetSecurityLevel() == HEStd_NotSet) {
        return context;
    }
    uint32_t ringDim = context->GetRingDimension();
    uint32_t fullSlots = ringDim / 2;
    if (fullSlots <= static_cast<uint32_t>(numSlots)) {
        return context;
    }
    parameters.SetRingDim(ringDim);
    parameters.SetBatchSize(fullSlots);
    CryptoContext<DCRTPoly> filled = GenCryptoContext(parameters);
    cout << "Packing: the ring dimension " << ringDim << " of this security level holds " << fullSlots
         << " slots; the model packs " << fullSlots / numSlots << "x more data (" << numSlots << " -> " << fullSlots
         << " slots)" << endl;
    numSlots = static_cast<int>(fullSlots);
    return filled;
}

/** log Q, log QP and the bound of the standard for the ring dimension, written to the log and security.txt */
static string fheon_security_report(const CryptoContext<DCRTPoly>& context, int bits, uint32_t numSlots,
                                    const CCParams<CryptoContextCKKSRNS>& parameters) {
    auto params = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(context->GetCryptoParameters());
    uint32_t n = context->GetRingDimension();
    uint32_t logQ = context->GetModulus().GetMSB();
    uint32_t logQP = params->GetParamsQP() ? params->GetParamsQP()->GetModulus().GetMSB() : logQ;
    std::ostringstream out;
    out << "Security level: " << FHEONHEController::security_level_name(bits) << endl;
    out << "Ring dimension: " << n << " (2^" << static_cast<int>(std::log2(n)) << "), slots: " << numSlots
        << " of " << n / 2 << endl;
    out << "Scale: " << parameters.GetScalingModSize() << " bits, key-switching digits: " << params->GetNumPartQ() << endl;
    out << "log Q = " << logQ << ", log QP = " << logQP;
    if (bits != 0) {
        SecurityLevel level = FHEONHEController::to_openfhe_security_level(bits);
        out << " (the standard allows log QP <= " << StdLatticeParm::FindMaxQ(HEStd_ternary, level, n)
            << " at this ring dimension)";
    } else {
        uint32_t n128 = StdLatticeParm::FindRingDim(HEStd_ternary, HEStd_128_classic, logQP);
        out << " (128-bit security would need ring dimension " << n128 << " and a non-sparse secret)";
    }
    out << endl;
    return out.str();
}

/**
 * @brief Set the count of bootstrapping operations performed.
 *
 * @param count The number of bootstrapping operations to set.
 */
void FHEONHEController::set_bootstrap_count(int count){
    num_bootsraps = count;
}


/**
 * @brief Generate FHE crypto context using an HEConfig structure.
 *
 * @param config Configuration structure containing ring dimension, slots, etc.
 */
void FHEONHEController::generate_context(HEConfig config) {
    if (!config.keysFolder.empty()) {
        keys_folder = config.keysFolder;
    }
    update_key_paths();
    context_bootstrap_keys = config.contextBootstrapKeys;
    context_sum_keys = config.contextSumKeys;
    security_level = config.securityLevel;
    fill_slots = config.fillSlots;
    min_scale_bits = config.minScaleBits;
    max_digits = config.maxDigits;
    generate_context(config.ringDim, config.numSlots, config.mlevelBootstrap, config.dcrtBits,
                    config.firstMod, config.numDigits, config.levelBudget, config.serialize);
    return;
}

/**
 * @brief Generate the full FHE context with advanced parameters.
 *
 * @param ringDim Ring dimension for CKKS context.
 * @param numSlots Number of slots for packing.
 * @param mlevelBootstrap Multiplication depth/level for bootstrapping.
 * @param dcrtBits Bit size of the DCRT moduli.
 * @param firstMod Scaling factor/modulus for the first level.
 * @param numDigits Number of digits for key switching.
 * @param levelBudget Level budget for bootstrapping.
 * @param serialize Whether to serialize context to files.
 */
void FHEONHEController::generate_context(int ringDim, int numSlots, int mlevelBootstrap, 
                        int dcrtBits, int firstMod, int numDigits, vector<uint32_t> levelBudget,
                        bool serialize) {

    CCParams<CryptoContextCKKSRNS> parameters;

    ScalingTechnique rescaleTech = FLEXIBLEAUTO;
    // ScalingTechnique rescaleTech = FLEXIBLEAUTOEXT;
    level_budget = levelBudget;
    num_slots = 1 << numSlots;
    mult_depth = mlevelBootstrap;

    // ring dimension, batch size, secret distribution and security level (performance efficient or a standard level)
    auto secretKeyDist = fheon_apply_security(parameters, security_level, ringDim, num_slots, true);
    parameters.SetScalingModSize(dcrtBits);
    parameters.SetFirstModSize(firstMod);
    parameters.SetNumLargeDigits(numDigits);
    parameters.SetScalingTechnique(rescaleTech);

    circuit_depth = mult_depth + FHECKKSRNS::GetBootstrapDepth(level_budget, secretKeyDist);
    parameters.SetMultiplicativeDepth(circuit_depth);

    cout << "Building the FHE Context, security level: " << security_level_name(security_level) << endl;
    cout << "dcrtBits: "<< dcrtBits << " -- firstMod: " << firstMod << endl << "Ciphertexts depth: " 
         << circuit_depth << ", available multiplications: " << circuit_depth - 2 << endl;
   
    context = fheon_gen_context(parameters, num_slots, min_scale_bits, max_digits);
    if (fill_slots) {
        context = fheon_fill_slots(parameters, context, num_slots);
    }
    scale_bits = parameters.GetScalingModSize();
    num_digits = parameters.GetNumLargeDigits();
    context->Enable(PKE);
    context->Enable(KEYSWITCH);
    context->Enable(LEVELEDSHE);
    context->Enable(ADVANCEDSHE);
    context->Enable(FHE);
    
    keyPair = context->KeyGen();
    context->EvalMultKeyGen(keyPair.secretKey);
    if (context_sum_keys) {
        context->EvalSumKeyGen(keyPair.secretKey);
    }

    ringDim = context->GetRingDimension();
    numSlots = num_slots;
    usint halfnumSlots = numSlots/2;
    if (context_bootstrap_keys) {
        context->EvalBootstrapSetup(level_budget, bsgsDim, numSlots);
        context->EvalBootstrapKeyGen(keyPair.secretKey, numSlots);

        context->EvalBootstrapSetup(level_budget, bsgsDim, halfnumSlots);
        context->EvalBootstrapKeyGen(keyPair.secretKey, halfnumSlots);

        usint quadnumSlots = halfnumSlots/2;
        context->EvalBootstrapSetup(level_budget, bsgsDim, quadnumSlots);
        context->EvalBootstrapKeyGen(keyPair.secretKey, quadnumSlots);
    }

    string security = fheon_security_report(context, security_level, num_slots, parameters);
    cout << "Keys Generated." << endl;
    cout << "Cyclotomic Order: " << context->GetCyclotomicOrder() << endl;
    cout << "CKKS scheme is using ring dimension: " << ringDim  << endl;
    cout << "Avaliable numSlots: " << numSlots << "  - halfnumSlots: " << halfnumSlots << endl;
    cout << security;
    cout << "Ciphertexts depth: " << circuit_depth << endl; 
    cout << "Multiplication Depth: " << mult_depth - 2 << endl;
    cout << "log PQ = " << getlogPQ(keyPair.publicKey->GetPublicElements()[0]) << std::endl << std::endl;
    cout << "-----------------------------------------------------------" << endl;
    
    if(serialize){
        write_to_file(keys_folder + "/security.txt", security);
        write_to_file(keys_folder + "/mult_depth.txt", to_string(mult_depth));
        write_to_file(keys_folder + "/num_slots.txt", to_string(num_slots));
        write_to_file(keys_folder + "/level_budget.txt", to_string(level_budget[0]) + "," + to_string(level_budget[1]));
        keys_serialization();
    }

    context_loaded = false;
    return;
}

/**
 * @brief Generate a standard FHE context with defaults.
 *
 * @param ringDim Ring dimension for CKKS context.
 * @param numSlots Number of slots for packing.
 * @param mlevelBootstrap Multiplication depth/level for bootstrapping.
 * @param serialize Whether to serialize context to files.
 */
void FHEONHEController::generate_context(int ringDim, int numSlots, int mlevelBootstrap, bool serialize) {
    CCParams<CryptoContextCKKSRNS> parameters;

    num_slots = 1 << numSlots;
    int dcrtBits               = 46;
    int firstMod               = 50;

    auto secretKeyDist = fheon_apply_security(parameters, security_level, ringDim, num_slots, true);
    parameters.SetNumLargeDigits(3);
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
    context = fheon_gen_context(parameters, num_slots);

    context->Enable(PKE);
    context->Enable(KEYSWITCH);
    context->Enable(LEVELEDSHE);
    context->Enable(ADVANCEDSHE);
    context->Enable(FHE);

    keyPair = context->KeyGen();
    context->EvalMultKeyGen(keyPair.secretKey);
    context->EvalSumKeyGen(keyPair.secretKey);

    numSlots = num_slots;
    context->EvalBootstrapSetup(level_budget, bsgsDim, numSlots);
    context->EvalBootstrapKeyGen(keyPair.secretKey, numSlots);

    // usint halfnumSlots = numSlots/2;
    // context->EvalBootstrapSetup(level_budget, bsgsDim, halfnumSlots);
    // context->EvalBootstrapKeyGen(keyPair.secretKey, halfnumSlots);

    // usint quadnumSlots = numSlots/2;
    // context->EvalBootstrapSetup(level_budget, bsgsDim, quadnumSlots);
    // context->EvalBootstrapKeyGen(keyPair.secretKey, quadnumSlots);

    cout << " Keys Generated." << endl;
    ringDim = context->GetRingDimension();
    cout << " CKKS scheme is using ring dimension: " << ringDim  << endl;
    cout << "numSlots: " << numSlots << endl;
    cout << " Ciphertexts depth: " << circuit_depth << endl; 
    cout << " Multiplication Depth: " << levelsAvailableAfterBootstrap - 2 << endl;
    cout << " Cyclotomic Order: " << context->GetCyclotomicOrder() << endl;
    cout << fheon_security_report(context, security_level, num_slots, parameters);
    cout << " -----------------------------------------------------------" << endl;

   if(serialize){
        write_to_file(keys_folder + "/mult_depth.txt", to_string(mult_depth));
        write_to_file(keys_folder + "/num_slots.txt", to_string(num_slots));
        write_to_file(keys_folder + "/level_budget.txt", to_string(level_budget[0]) + "," + to_string(level_budget[1]));
        keys_serialization();
    }
    context_loaded = false;
    return;
}


/**
 * @brief Serialize generated FHE keys (public, secret, mult, sum) to files.
 */

void FHEONHEController::keys_serialization(){
    
    cout << "------------------------------------------------------------" << endl;
    cout << "Now serializing keys ..." << endl;

    fs::path keysDir = keys_folder;
    if (!fs::exists(keysDir)) {
        if (!fs::create_directories(keysDir)) {
            std::cerr << "Failed to create directory: " << keys_folder << std::endl;
            return;
        }
    }

    if (!Serial::SerializeToFile(crypto_context_file, context, SerType::BINARY)) {
        cerr << "Error writing serialization of the crypto context to crypto-context.bin" << endl;
    } else {
        cout << "Crypto Context have been serialized" << std::endl;
    }

    ofstream multKeyFile(multi_key_file, ios::out | ios::binary);
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

    ofstream sumKeysFile(sum_key_file, ios::out | ios::binary);
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

    if (!Serial::SerializeToFile(public_key_file, keyPair.publicKey, SerType::BINARY)) {
        cerr << "Error writing serialization of public key to public-key.bin" << endl;
    } else {
        cout << "Public Key has been serialized" << std::endl;
    }

    if (!Serial::SerializeToFile(secret_key_file, keyPair.secretKey, SerType::BINARY)) {
        cerr << "Error writing serialization of public key to secret-key.bin" << endl;
    } else {
        cout << "Secret Key has been serialized" << std::endl;
    }
    return;
}


/**
 * @brief Load or generate FHE context based on configuration.
 *
 * @param config Configuration parameters for initialization.
 * @param loadContext If true, load context and keys from files; if false, generate new ones.
 */
void FHEONHEController::load_context(HEConfig config, bool loadContext){
    if (!config.keysFolder.empty()) {
        keys_folder = config.keysFolder;
    }
    update_key_paths();
    if (loadContext) {
        load_context();
    } else {
        generate_context(config);
    }
}


/**
 * @brief Load serialized FHE context and keys from default paths.
 */
void FHEONHEController::load_context() {
    
    context->ClearEvalMultKeys();
    context->ClearEvalAutomorphismKeys();
    CryptoContextFactory<lbcrypto::DCRTPoly>::ReleaseAllContexts();

    cout << "------------------------------------------------------------" << endl;
    cout << "Loading context from: " << keys_folder << endl;
    cout << "------------------------------------------------------------" << endl;

    if (utils::verbose_level >= 1) {
        cout << "------------------------------------------------------------" << endl;
        cout << "Reading serialized context..." << endl;
    }

    if (!Serial::DeserializeFromFile(crypto_context_file, context, SerType::BINARY)) {
        cerr << "I cannot read serialized data from: " << crypto_context_file << endl;
        exit(1);
    }

    PublicKey<DCRTPoly> clientPublicKey;
    if (!Serial::DeserializeFromFile(public_key_file, clientPublicKey, SerType::BINARY)) {
        cerr << "I cannot read serialized data from public-key.bin" << endl;
        exit(1);
    }

    PrivateKey<DCRTPoly> serverSecretKey;
    if (Serial::DeserializeFromFile(secret_key_file, serverSecretKey, SerType::BINARY)) {
        keyPair.secretKey = serverSecretKey;
    } else {
        if (utils::verbose_level >= 1) {
            cout << "Warning: Could not read secret-key.bin. Private key not loaded." << endl;
        }
    }

    keyPair.publicKey = clientPublicKey;

    std::ifstream multKeyIStream(multi_key_file, ios::in | ios::binary);
    if (!multKeyIStream.is_open()) {
        cerr << "Cannot read serialization from " << multi_key_file << endl;
        exit(1);
    }
    if (!context->DeserializeEvalMultKey(multKeyIStream, SerType::BINARY)) {
        cerr << "Could not deserialize eval multkey file" << endl;
        exit(1);
    }

    ifstream sumKeyIStream(sum_key_file, ios::in | ios::binary);
    if (!sumKeyIStream.is_open()) {
        cerr << "Cannot read serialization from " << sum_key_file << std::endl;
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

    num_slots = stoi(read_from_file(keys_folder + "/num_slots.txt"));
    if (utils::verbose_level >= 1) {
        cout << "Setting up bootstrapping for " << num_slots << " slots..." << endl;
    }
    /** a leveled context (generate_leveled_context: MLPDiagonal, BasicMLP) has no bootstrapping (FHE is not enabled) */
    try {
        context->EvalBootstrapSetup(level_budget, bsgsDim, num_slots);
    } catch (const lbcrypto::OpenFHEException&) {
        if (utils::verbose_level >= 1) {
            cout << "Leveled context: no bootstrapping" << endl;
        }
    }

    if (utils::verbose_level >= 1) {
        cout << "Circuit depth: " << circuit_depth << ", available multiplications: " << levelsUsedBeforeBootstrap - 2 << endl;
        cout << "Context Loaded" << endl;
        cout << "------------------------------------------------------------" << endl;
    }
    context_loaded = true;
}

/**
 * @brief Generate evaluation keys required for bootstrapping.
 *
 * @param bootstrap_slots Number of slots reserved for bootstrapping.
 * @param filename Base filename/directory path for serialization.
 * @param serialize If true, serialize and save keys to disk.
 */
void FHEONHEController::generate_bootstrapping_keys(int bootstrap_slots, string filename, bool serialize) {
    // Instead, (re-)run the bootstrap setup for the specific slot count used at this
    // layer boundary so the precomputations are valid when EvalBootstrap is called.

    /** values up to 16 are log2 of the slot count (the slotsValues of the standard models, e.g. 14), larger values are
     *  slot counts (the diagonal models pass e.g. 8192): a log2 value taken as a slot count (14 slots) crashed EvalBootstrapSetup */
    if(bootstrap_slots <= 16){
        bootstrap_slots = 1 << bootstrap_slots;
    }


    // usint quadnumSlots = numSlots/2;
    // context->EvalBootstrapSetup(level_budget, bsgsDim, quadnumSlots);
    // context->EvalBootstrapKeyGen(keyPair.secretKey, quadnumSlots);
    
    // uint32_t numSlots = context->GetRingDimension() / 2;
    // Setup bootstrap precomputations for the exact slot count
    context->EvalBootstrapSetup(level_budget, bsgsDim, bootstrap_slots, bootstrap_correction_factor);
    context->EvalBootstrapKeyGen(keyPair.secretKey, bootstrap_slots);
    context->EvalMultKeyGen(keyPair.secretKey);
}


/**
 * @brief Generate evaluation keys for homomorphic rotation operations.
 *
 * @param rotations List of rotation step sizes to generate keys for.
 * @param filename Base filename/directory path for serialization.
 * @param serialize If true, serialize and save keys to disk.
 * @param sum_key If true, generate additional keys required for evaluation sum.
 */
void FHEONHEController::generate_rotation_keys(const vector<int> rotations, std::string filename, bool serialize, bool sum_key) {
    if (context_loaded) {
        load_rotation_keys(filename, false);
        return;
    }
    
    if (serialize && filename.size() == 0) {
        cout << "Filename cannot be empty when serializing rotation keys." << endl;
        return;
    }
  
    context->EvalRotateKeyGen(keyPair.secretKey, rotations);
    if (sum_key) {
        // Generate sum keys when requested (useful for fully-connected layers)
        context->EvalSumKeyGen(keyPair.secretKey);
    }
    
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
 * @brief Add rotation keys for indices missing from a prior generation/load, and
 * persist the result.
 *
 * EvalRotateKeyGen adds to the context's existing automorphism key map rather than
 * replacing it, so calling this after load_bootstrapping_and_rotation_keys (or after
 * a fresh generate_bootstrapping_and_rotation_keys) leaves all previously-available
 * rotation keys intact and simply adds the requested new ones. The re-serialized
 * file (same path convention as generate_rotation_keys: keys_folder + rotation_prefix
 * + filename) therefore contains the full merged set, so subsequent
 * load_bootstrapping_and_rotation_keys calls pick up the new indices too without
 * needing this step repeated. Requires the secret key to be available (loaded via
 * load_context() or generated via generate_context()).
 *
 * @param rotations New rotation step sizes to add.
 * @param filename  Same rotation-key filename used at initial generation (e.g. "layer1.bin").
 */
void FHEONHEController::add_and_persist_rotation_keys(vector<int> rotations, const string& filename) {
    if (!keyPair.secretKey) {
        cerr << "[add_and_persist_rotation_keys] ERROR: secret key not available — cannot generate new rotation keys." << endl;
        exit(1);
    }

    context->EvalRotateKeyGen(keyPair.secretKey, rotations);

    ofstream rotationKeyFile(keys_folder + rotation_prefix + filename, ios::out | ios::binary);
    if (rotationKeyFile.is_open()) {
        if (!context->SerializeEvalAutomorphismKey(rotationKeyFile, SerType::BINARY)) {
            cerr << "Error writing merged rotation keys" << std::endl;
            exit(1);
        }
        cout << "Added " << rotations.size() << " rotation key(s) and re-serialized the merged set to \"" << filename << "\"" << std::endl;
    } else {
        cerr << "Error serializing merged rotation keys to " << keys_folder + rotation_prefix + filename << std::endl;
        exit(1);
    }
}


/**
 * @brief Generate both bootstrapping and rotation keys.
 *
 * @param rotations List of rotation step sizes.
 * @param bootstrap_slots Number of slots for bootstrapping.
 * @param filename Base filename/directory path.
 * @param serialize If true, serialize keys to disk.
 * @param sum_key If true, generate evaluation sum keys.
 */
void FHEONHEController::generate_bootstrapping_and_rotation_keys(vector<int> rotations, int bootstrap_slots, const string& filename,  bool serialize, bool sum_key) {
    if (context_loaded) {
        load_bootstrapping_and_rotation_keys(bootstrap_slots, filename, false);
        return;
    }

    generate_bootstrapping_keys(bootstrap_slots, filename, serialize);
    if(sum_key){
        context->EvalSumKeyGen(keyPair.secretKey);
    }
    generate_rotation_keys(rotations, filename, serialize, sum_key);
}


/**
 * @brief Load both bootstrapping and rotation evaluation keys from files.
 *
 * @param bootstrap_slots Number of slots for bootstrapping.
 * @param filename Base filename/path.
 * @param verbose If true, print loading progress.
 */
void FHEONHEController::load_bootstrapping_and_rotation_keys(int bootstrap_slots, const string& filename, bool verbose) {
    if (utils::verbose_level >= 1) cout << endl << "    Loading bootstrapping and rotations keys from " << filename << "..." << endl;

    // int numSlots =  1 << bootstrap_slots;
    // context->EvalBootstrapSetup(level_budget, bsgsDim, numSlots);

    context->ClearEvalMultKeys();
	context->ClearEvalAutomorphismKeys();

    if (utils::verbose_level >= 1)  cout << "    (1/4) Bootstrapping precomputations completed!" << endl;
    
    ifstream multKeyIStream(multi_key_file, ios::in | ios::binary);
    if (!multKeyIStream.is_open()) {
        cerr << "Cannot read serialization from " << keys_folder+ "/" << mult_prefix << filename << std::endl;
        exit(1);
    }
    if (!context->DeserializeEvalMultKey(multKeyIStream, SerType::BINARY)) {
        cerr << "Could not deserialize eval rot key file" << std::endl;
        exit(1);
    }
    if (utils::verbose_level >= 1)  cout << "    (2/4) MultKey deserialized and loaded!" << endl;

    ifstream rotKeyIStream(keys_folder + rotation_prefix + filename, ios::in | ios::binary);
    if (!rotKeyIStream.is_open()) {
        cerr << "Cannot read serialization from " << keys_folder+ "/" << rotation_prefix << filename << std::endl;
        exit(1);
    }
    if (!context->DeserializeEvalAutomorphismKey(rotKeyIStream, SerType::BINARY)) {
        cerr << "Could not deserialize eval rot key file" << std::endl;
        exit(1);
    }
    if (utils::verbose_level >= 1) cout << "    (3/4) Rotation keys deserialized and loaded!" << endl;
    if (utils::verbose_level >= 1) cout << "    (4/4) Loaded keys for " << filename  << endl;
}

/**
 * @brief Helper function to read evaluation keys from file.
 *
 * @param crypto_context Context to associate keys with.
 * @param rot_file Path to rotation keys file.
 * @return Configured crypto context.
 */
CryptoContext<DCRTPoly> FHEONHEController::read_evaluation_keys(CryptoContext<DCRTPoly> crypto_context, const string &rot_file) {

    // Clear any existing eval keys in the provided context to avoid conflicts
    crypto_context->ClearEvalMultKeys();
    crypto_context->ClearEvalAutomorphismKeys();
    CryptoContextFactory<lbcrypto::DCRTPoly>::ReleaseAllContexts();
    /** return the memory of the cleared keys to the system before the next key set is loaded */
    malloc_trim(0);
    if (utils::verbose_level >= 1) cout << "    (1/4) Clear previous Keys and context!" << endl;

    // Open files in READ mode
    ifstream multkey_file(multi_key_file, ios::in | ios::binary);
    if (!multkey_file.is_open()) {
        throw std::runtime_error("Failed to open mult key file: " + multi_key_file);
    }
    if (!crypto_context->DeserializeEvalMultKey(multkey_file, SerType::BINARY)) {
        throw std::runtime_error("Failed to deserialize relinearization key from " + multi_key_file);
    }
    if (utils::verbose_level >= 1) cout << "    (2/4) MultKey deserialized and loaded!" << endl;

    ifstream rotkey_file(keys_folder + rotation_prefix + rot_file, ios::in | ios::binary);
    if (!rotkey_file.is_open()) {
        throw std::runtime_error("Failed to open rotation key file: " + keys_folder + rotation_prefix + rot_file);
    }
    if (!crypto_context->DeserializeEvalAutomorphismKey(rotkey_file, SerType::BINARY)) {
        throw std::runtime_error("Failed to deserialize rotation keys from " + keys_folder + rotation_prefix + rot_file);
    }
    if (utils::verbose_level >= 1) cout << "    (3/4) Rotation keys deserialized and loaded!" << endl;
    if (utils::verbose_level >= 1) cout << "    (4/4) Loaded context for " << rot_file  << endl;

    return crypto_context;
}


/**
 * @brief Load rotation evaluation keys from file.
 *
 * @param filename Path to rotation keys file.
 * @param verbose If true, print loading progress.
 */
void FHEONHEController::load_rotation_keys(const string& filename, bool verbose) {

    if (utils::verbose_level >= 1) cout << endl << "Loading rotations keys from " << filename << "..." << endl;
    
    ifstream rotKeyIStream(keys_folder + rotation_prefix + filename, ios::in | ios::binary);
    if (!rotKeyIStream.is_open()) {
        cerr << "Cannot read serialization from " <<keys_folder + "/" << rotation_prefix << filename << std::endl;
        exit(1);
    }
    if (!context->DeserializeEvalAutomorphismKey(rotKeyIStream, SerType::BINARY)) {
        cerr << "Could not deserialize eval rot key file" << std::endl;
        exit(1);
    }

    if (utils::verbose_level >= 1) {
        cout << "(1/1) Rotation keys read!" << endl;
        cout << endl;
    }
}

/**
 * @brief Clear rotation evaluation keys from current context.
 */
void FHEONHEController::clear_rotation_keys() {
    context->ClearEvalMultKeys();
    context->ClearEvalAutomorphismKeys();
    CryptoContextFactory<lbcrypto::DCRTPoly>::ReleaseAllContexts();
    /** return the memory of the cleared keys to the system before the next key set is loaded */
    malloc_trim(0);
}

/**
 * @brief Clear bootstrapping and rotation keys from current context.
 *
 * @param bootstrap_num_slots Number of bootstrapping slots.
 */
void FHEONHEController::clear_bootstrapping_and_rotation_keys(int bootstrap_num_slots) {
    //This lines would free more or less 1GB or precomputations, but requires access to the GetFHE function
    // FHECKKSRNS* derivedPtr = dynamic_cast<FHECKKSRNS*>(context->GetScheme()->GetFHE().get());
    // derivedPtr->m_bootPrecomMap.erase(bootstrap_num_slots);

    context->ClearEvalMultKeys();
    context->ClearEvalAutomorphismKeys();
    CryptoContextFactory<lbcrypto::DCRTPoly>::ReleaseAllContexts();
    /** return the memory of the cleared keys to the system before the next key set is loaded */
    malloc_trim(0);
}


/**
 * @brief Clear keys and release crypto context.
 *
 * @param bootstrapping_key_slots Number of bootstrapping slots.
 */
void FHEONHEController::clear_context(int bootstrapping_key_slots) {
    
    if (bootstrapping_key_slots != 0)
        clear_bootstrapping_and_rotation_keys(bootstrapping_key_slots);
    else
        clear_rotation_keys();
}



/**
 * @brief Perform homomorphic bootstrapping (refreshing noise level) on ciphertext.
 *
 * @param encryptedInput Ciphertext to bootstrap.
 * @param encode_level Level budget for encoding.
 * @return Refreshed ciphertext.
 */
Ctext FHEONHEController::bootstrap_function(Ctext& encryptedInput, int encode_level){
    num_bootsraps++;
    auto start_bootstrap = startTime();
    Ctext boots_ciphertext = context->EvalBootstrap(encryptedInput, encode_level, encode_level > 1 ? bootstrap_precision : 0);
    printBootstrapTiming("    Bootstrapping ciphertext... " + to_string(num_bootsraps), start_bootstrap);
    return boots_ciphertext;
}

/**
 * @brief Perform parallel bootstrapping on a batch of ciphertexts.
 *
 * @param encryptedInputs Vector of ciphertexts to bootstrap.
 * @param inputChannels Number of channels/ciphertexts.
 * @param encode_level Level budget.
 * @return Vector of refreshed ciphertexts.
 */

vector<Ctext> FHEONHEController::batch_bootstrap_function(vector<Ctext>& encryptedInputs, int inputChannels, int encode_level) {

    // Derive the true channel count from the actual vector to prevent out-of-bounds
    // heap writes when the caller-supplied inputChannels mismatches encryptedInputs.size().
    int actualChannels = static_cast<int>(encryptedInputs.size());
    if (actualChannels != inputChannels) {
        cerr << "[batch_bootstrap_function] WARNING: inputChannels=" << inputChannels
             << " does not match encryptedInputs.size()=" << actualChannels
             << ". Using actual size to avoid heap corruption." << endl;
        inputChannels = actualChannels;
    }
    if (inputChannels == 0) {
        cerr << "[batch_bootstrap_function] ERROR: encryptedInputs is empty." << endl;
        return {};
    }

    auto start_bootstrap = startTime();
    vector<Ctext> batch_ciphertexts(inputChannels);
    // Bootstrapping in OpenFHE requires a large amount of memory per ciphertext (typically 4-8 GB),
    // so running inputChannels-many in parallel can OOM on memory-constrained machines -- see git
    // history for this line. Sized by worker_threads() like the conv/FC worker pools: all hardware
    // threads by default, capped with --threads on memory-constrained machines.
    int numThreads = worker_threads(inputChannels);
    vector<thread> threads(numThreads);

    // Worker function for a range of channels
    auto worker = [&](int start, int end) {
        /** each worker bootstraps with its share of the OpenMP threads */
        use_worker_openmp_share(numThreads);
        for (int b = start; b < end; b++) {
            batch_ciphertexts[b] = context->EvalBootstrap(encryptedInputs[b], encode_level, encode_level > 1 ? bootstrap_precision : 0);
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

    printBootstrapTiming("  Batch Bootstrapping " + to_string(inputChannels) + " ciphertexts...", start_bootstrap);
    return batch_ciphertexts;
}

/**
 * @brief Encrypt a vector of double values into a packed CKKS ciphertext.
 *
 * @param inputData Data vector to be homomorphically encrypted.
 * @param isTimeMeasurement If true, initialize timing statistics for the image run.
 * @return homomorphically encrypted ciphertext.
 */
Ctext FHEONHEController::encrypt_input(vector<double>& inputData, bool isTimeMeasurement) {
   
    Ptext plaintext = context->MakeCKKSPackedPlaintext(inputData, 1, 1);
    plaintext->SetLength(inputData.size());
    auto encryptImage = context->Encrypt(keyPair.publicKey, plaintext);

     if(isTimeMeasurement) {
        image_start_time = std::chrono::steady_clock::now();
        program_start_time = std::chrono::steady_clock::now();
        image_circuit_time_secs = 0;
        image_bootstrap_time_ms = 0.0;
        num_bootsraps = 0;
        operation_count = 0;
        accumulated_circuit_time_ms = 0.0;
    }
    return encryptImage;
}


/**
 * @brief Encrypt a vector of double values at a chosen level of the modulus chain.
 *
 * Same as encrypt_input, but the ciphertext starts at encode_level instead of level 1. When
 * the circuit before the first bootstrapping does not need the whole chain, starting lower
 * makes every operation up to that bootstrapping work on fewer towers.
 *
 * @param inputData Data vector to be homomorphically encrypted.
 * @param encode_level Level of the fresh ciphertext.
 * @param isTimeMeasurement If true, initialize timing statistics for the image run.
 * @return homomorphically encrypted ciphertext.
 */
Ctext FHEONHEController::encrypt_input_at_level(vector<double>& inputData, int encode_level, bool isTimeMeasurement) {
    Ptext plaintext = context->MakeCKKSPackedPlaintext(inputData, 1, encode_level);
    plaintext->SetLength(inputData.size());
    auto encryptImage = context->Encrypt(keyPair.publicKey, plaintext);

    if(isTimeMeasurement) {
        image_start_time = std::chrono::steady_clock::now();
        program_start_time = std::chrono::steady_clock::now();
        image_circuit_time_secs = 0;
        image_bootstrap_time_ms = 0.0;
        num_bootsraps = 0;
        operation_count = 0;
        accumulated_circuit_time_ms = 0.0;
    }
    return encryptImage;
}

vector<Ctext> FHEONHEController::encrypt_multiple_inputs(vector<vector<double>>& inputDatas){
    
    vector<Ctext> encryptedInputs(inputDatas.size());
    for(size_t i=0; i<inputDatas.size(); i++){
        if(i == (inputDatas.size() - 1)){
            encryptedInputs[i] = encrypt_input(inputDatas[i], true);
        } else
        {
            encryptedInputs[i] = encrypt_input(inputDatas[i], false);
        }
    }
    return encryptedInputs;
}


/**
 * @brief Encode a double vector into packed CKKS plaintext.
 *
 * @param inputData Data vector to encode.
 * @param encode_level Level budget.
 * @return Encoded plaintext.
 */
Ptext FHEONHEController::encode_input(vector<double>& inputData, int encode_level) {
    Ptext plaintext = context->MakeCKKSPackedPlaintext(inputData, 1, encode_level);
    return plaintext;
}

/**
 * @brief Encode a double vector with explicit slot count.
 *
 * @param inputData Data vector to encode.
 * @param num_slots Number of slots to use.
 * @param encode_level Level budget.
 * @return Encoded plaintext.
 */

Ptext FHEONHEController::encode_input(vector<double>& inputData, int num_slots, int encode_level) {
    int numElements = nextPowerOf2(num_slots);
    Ptext plaintext = context->MakeCKKSPackedPlaintext(inputData, 1, encode_level, nullptr, numElements);
    return plaintext;
}

/**
 * @brief Encode kernel weights for ResNet shortcut connections.
 *
 * @param inputData Flattened weight vector.
 * @param cols_square Square dimension of output features.
 * @return Encoded plaintext.
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
 * @brief Encode bias inputs for convolution or linear layers.
 *
 * @param inputData Bias values vector.
 * @param cols_square Square dimension of output features.
 * @param encode_level Level budget.
 * @return Encoded plaintext.
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
 * @brief Re-encrypt plaintext data to ciphertext.
 *
 * @param plaintextData Plaintext to encrypt.
 * @return Encrypted ciphertext.
 */
Ctext FHEONHEController::reencrypt_data(Ptext plaintextData) {
    
    auto encryptedData = context->Encrypt(keyPair.publicKey, plaintextData);
    return encryptedData;
}

/**
 * @brief Decrypt ciphertext into packed plaintext values.
 *
 * @param encryptedinputData Ciphertext to decrypt.
 * @param cols Number of slots to retrieve.
 * @return Decrypted plaintext.
 */
Ptext FHEONHEController::decrypt_data(Ctext encryptedinputData, int cols) {
    
    Ptext plaintextDec;
    context->Decrypt(keyPair.secretKey, encryptedinputData, &plaintextDec);
    plaintextDec->SetLength(cols);
    return plaintextDec;
}

/**
 * @brief Encrypt a 3D weight kernel matrix.
 *
 * @param kernelData 3D raw weights vector.
 * @param cols_square Square dimension of features.
 * @return Encrypted weights matrix.
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
 * @brief Encode a 1D fully connected weight vector.
 *
 * @param kernelData Weight values vector.
 * @param cols_square Square dimension.
 * @return Vector of encoded plaintexts.
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
 * @brief Encode a 3D kernel weight matrix.
 *
 * @param kernelData 3D raw weights vector.
 * @param cols_square Square dimension.
 * @return Vector of encoded plaintexts.
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
 * @brief Dynamically change the number of active slots in a ciphertext.
 *
 * Useful for reducing ring dimension/size in pooled layers.
 *
 * @param encryptedInput The ciphertext to modify.
 * @param num_slots The new number of slots.
 * @return Modifed ciphertext with changed slots.
 */

Ctext FHEONHEController::change_num_slots(Ctext& encryptedInput, uint32_t num_slots){
    encryptedInput->SetSlots(1 << num_slots);
    return encryptedInput;
}

/**
 * @brief Repack a ciphertext into a sparse encoding with fewer slots, preserving its values.
 *
 * A CKKS ciphertext with n slots is one whose full-slot vector is periodic with period n.
 * Calling SetSlots(n) alone on data that sits in the first n slots (zeros elsewhere) makes
 * the decoder average the copies, scaling the values by n/currentSlots. The data is therefore
 * first replicated with rotations by -n, -2n, ... and only then relabelled.
 * Slots beyond numSlots must be zero, and rotation keys for -numSlots, -2*numSlots, ...
 * (up to half the current slot count) must be available.
 *
 * @param encryptedInput Ciphertext whose data occupies the first numSlots slots.
 * @param numSlots The new number of slots (power of two, not log2).
 * @return New ciphertext with numSlots slots decrypting to the same values.
 */
Ctext FHEONHEController::reduce_num_slots(const Ctext& encryptedInput, uint32_t numSlots){
    uint32_t currentSlots = encryptedInput->GetSlots();
    if (numSlots == 0 || (numSlots & (numSlots - 1)) != 0 || numSlots > currentSlots) {
        throw std::invalid_argument("reduce_num_slots: numSlots must be a power of two <= " + to_string(currentSlots));
    }
    Ctext result = encryptedInput->Clone();
    for (uint32_t s = numSlots; s < currentSlots; s <<= 1) {
        result = context->EvalAdd(result, context->EvalRotate(result, -static_cast<int>(s)));
    }
    result->SetSlots(numSlots);
    return result;
}

/**
 * @brief Set the slot count used by default when encoding plaintexts.
 *
 * Plaintexts combined with a sparse ciphertext must be encoded with the same number of
 * slots; a full-slot, zero-padded plaintext is seen scaled down by the sparse ciphertext.
 * Call this alongside reduce_num_slots, and restore the full slot count before a new input
 * is encrypted.
 *
 * @param numSlots Slot count for subsequent default encodings (power of two, not log2).
 */
void FHEONHEController::set_encoding_slots(uint32_t numSlots){
    uint32_t currentSlots = context->GetEncodingParams()->GetBatchSize();
    if (numSlots < currentSlots && verbose_level >= level_two) {
        cout << "    Data Size: " << numSlots << " from " << currentSlots << endl;
    }
    context->GetEncodingParams()->SetBatchSize(numSlots);
}

/**
 * @brief Encode a 3x3 convolution kernel as channel diagonals for he_convolution_diagonal
 *        (strided = false) or he_convolution_and_shortcut_diagonal (strided = true).
 *
 * The ciphertext is seen as numBlocks = numSlots / outputWidth^2 channel blocks. Diagonal d
 * holds, in block b, the weight that maps the input channel sitting in block b to output
 * channel (b - d) mod numBlocks, so that sum_d rotate(sum_j tap_j * diag[d][j], d * blockSize)
 * is the full convolution. The zero-padding masks are folded into the plaintexts.
 * For strided = false the mask is the usual "same" padding mask; for strided = true the taps
 * are the stride-2 samples of the input. The input is expected to repeat every inputPeriod
 * blocks, which is also the number of diagonals: by default all the blocks for strided = false
 * and inputChannels for strided = true. A layer with few input channels can pass a smaller
 * power-of-two period (>= inputChannels) to get fewer diagonals; he_convolution_diagonal then
 * replicates its input accordingly.
 *
 * @param kernelData Weights [outputChannels][inputChannels][3][3].
 * @param outputWidth Spatial width of one output channel.
 * @param numSlots Slot count of the ciphertext the plaintexts are multiplied with.
 * @param encode_level Level the plaintexts are encoded at.
 * @param strided True for the stride-2 layout.
 * @param inputPeriod Number of blocks after which the input repeats (0 = default).
 * @return [diagonal][tap] plaintexts; entries whose values are all zero are null.
 */
vector<vector<Ptext>> FHEONHEController::encode_kernel_diagonal(const vector<vector<vector<vector<double>>>>& kernelData,
                            int outputWidth, int numSlots, int encode_level, bool strided, int inputPeriod) {
    int outputChannels = kernelData.size();
    int inputChannels = kernelData[0].size();
    int blockSize = outputWidth * outputWidth;
    int numBlocks = numSlots / blockSize;
    if (inputPeriod == 0) {
        inputPeriod = strided ? inputChannels : numBlocks;
    }
    if (numBlocks * blockSize != numSlots || outputChannels > numBlocks || inputChannels > inputPeriod || numBlocks % inputPeriod != 0) {
        throw std::invalid_argument("encode_kernel_diagonal: channels do not fit the " + to_string(numSlots) + " slots");
    }

    vector<vector<Ptext>> encoded_kernel(inputPeriod, vector<Ptext>(9));
    vector<double> values(numSlots);
    for (int d = 0; d < inputPeriod; d++) {
        for (int j = 0; j < 9; j++) {
            int dy = j / 3 - 1;
            int dx = j % 3 - 1;
            bool nonZero = false;
            std::fill(values.begin(), values.end(), 0.0);
            for (int b = 0; b < numBlocks; b++) {
                int inCh = b % inputPeriod;
                int outCh = ((b - d) % numBlocks + numBlocks) % numBlocks;
                if (inCh >= inputChannels || outCh >= outputChannels) continue;
                double weight = kernelData[outCh][inCh][j / 3][j % 3];
                if (weight == 0.0) continue;
                nonZero = true;
                for (int y = 0; y < outputWidth; y++) {
                    if (strided ? (dy < 0 && y == 0) : (y + dy < 0 || y + dy >= outputWidth)) continue;
                    for (int x = 0; x < outputWidth; x++) {
                        if (strided ? (dx < 0 && x == 0) : (x + dx < 0 || x + dx >= outputWidth)) continue;
                        values[b * blockSize + y * outputWidth + x] = weight;
                    }
                }
            }
            if (nonZero) {
                encoded_kernel[d][j] = context->MakeCKKSPackedPlaintext(values, 1, encode_level, nullptr, numSlots);
            }
        }
    }
    return encoded_kernel;
}

/**
 * @brief Encode a 1x1 stride-2 shortcut kernel as channel diagonals (see encode_kernel_diagonal
 *        with strided = true).
 *
 * @param kernelData Weights [outputChannels][inputChannels].
 * @param outputWidth Spatial width of one output channel.
 * @param numSlots Slot count of the ciphertext the plaintexts are multiplied with.
 * @param encode_level Level the plaintexts are encoded at.
 * @return One plaintext per diagonal.
 */
vector<Ptext> FHEONHEController::encode_shortcut_diagonal(const vector<vector<double>>& kernelData, int outputWidth, int numSlots, int encode_level) {
    int outputChannels = kernelData.size();
    int inputChannels = kernelData[0].size();
    int blockSize = outputWidth * outputWidth;
    int numBlocks = numSlots / blockSize;
    if (numBlocks * blockSize != numSlots || outputChannels > numBlocks || numBlocks % inputChannels != 0) {
        throw std::invalid_argument("encode_shortcut_diagonal: channels do not fit the " + to_string(numSlots) + " slots");
    }

    vector<Ptext> encoded_kernel(inputChannels);
    vector<double> values(numSlots);
    for (int d = 0; d < inputChannels; d++) {
        std::fill(values.begin(), values.end(), 0.0);
        for (int b = 0; b < numBlocks; b++) {
            int outCh = ((b - d) % numBlocks + numBlocks) % numBlocks;
            if (outCh >= outputChannels) continue;
            std::fill(values.begin() + b * blockSize, values.begin() + (b + 1) * blockSize, kernelData[outCh][b % inputChannels]);
        }
        encoded_kernel[d] = context->MakeCKKSPackedPlaintext(values, 1, encode_level, nullptr, numSlots);
    }
    return encoded_kernel;
}

/**
 * @brief Encode a bias vector, one value per channel block, with an explicit slot count.
 *
 * @param biasData Bias values, one per output channel.
 * @param outputWidth Spatial width of one output channel.
 * @param numSlots Slot count of the ciphertext the bias is added to.
 * @param encode_level Level the plaintext is encoded at.
 * @return Encoded plaintext.
 */
Ptext FHEONHEController::encode_bias_diagonal(const vector<double>& biasData, int outputWidth, int numSlots, int encode_level) {
    int blockSize = outputWidth * outputWidth;
    vector<double> values(numSlots, 0.0);
    for (size_t ch = 0; ch < biasData.size(); ch++) {
        std::fill(values.begin() + ch * blockSize, values.begin() + (ch + 1) * blockSize, biasData[ch]);
    }
    return context->MakeCKKSPackedPlaintext(values, 1, encode_level, nullptr, numSlots);
}

/**
 * @brief Encode a stride-1 "valid" convolution kernel for he_convolution_layout.
 *
 * Works on strided layouts, so that neither the convolution nor a preceding sum pooling has
 * to compact its output. Input value (c, y, x) sits at slot
 * channelStride * c + rowStride * y + colStride * x. Output channel o = g * E + e (E =
 * subShifts.size()) is written at channelStride * g + rowStride * y + colStride * x +
 * subShifts[e]: the E channels of a group share a block and are interleaved through their sub
 * shifts, which is how the gaps left by a pooling are reused. Positions that hold no output
 * value are zero in every plaintext, so whatever the input holds outside its valid positions
 * is never read.
 * For sub shift e, diagonal t gathers the weights from input channel c to group
 * g = c - t + numGroups - 1; he_convolution_layout sums the diagonals with one rotation each.
 *
 * @param kernelData Weights [outputChannels][inputChannels][k][k].
 * @param scale Factor multiplied into every weight (e.g. 0.25 after a 2x2 sum pooling).
 * @param numSlots Slot count of the ciphertext.
 * @param encode_level Level the plaintexts are encoded at.
 * @param channelStride Slots between two input channels, and between two output groups.
 * @param rowStride Slots between two rows.
 * @param colStride Slots between two columns.
 * @param outputWidth Spatial width of one output channel.
 * @param numGroups Number of output groups (outputChannels / subShifts.size()).
 * @param subShifts Slot offset of each channel inside its group.
 * @return [sub shift][diagonal][tap] plaintexts; entries whose values are all zero are null.
 */
vector<vector<vector<Ptext>>> FHEONHEController::encode_kernel_layout(const vector<vector<vector<vector<double>>>>& kernelData, double scale,
                            int numSlots, int encode_level, int channelStride, int rowStride, int colStride, int outputWidth,
                            int numGroups, const vector<int>& subShifts) {
    int outputChannels = kernelData.size();
    int inputChannels = kernelData[0].size();
    int kernelWidth = kernelData[0][0].size();
    int numSubs = subShifts.size();
    int numDiagonals = inputChannels + numGroups - 1;
    int lastTap = (kernelWidth - 1) * (rowStride + colStride);
    int lastOutput = (outputWidth - 1) * (rowStride + colStride);
    if (numGroups * numSubs != outputChannels || (inputChannels - 1) * channelStride + lastOutput + lastTap >= numSlots
        || lastOutput >= channelStride) {
        throw std::invalid_argument("encode_kernel_layout: the layout does not fit the " + to_string(numSlots) + " slots");
    }

    vector<vector<vector<Ptext>>> encoded_kernel(numSubs, vector<vector<Ptext>>(numDiagonals, vector<Ptext>(kernelWidth * kernelWidth)));
    vector<double> values(numSlots);
    for (int e = 0; e < numSubs; e++) {
        for (int t = 0; t < numDiagonals; t++) {
            for (int j = 0; j < kernelWidth * kernelWidth; j++) {
                bool nonZero = false;
                std::fill(values.begin(), values.end(), 0.0);
                for (int c = 0; c < inputChannels; c++) {
                    int group = c - t + numGroups - 1;
                    if (group < 0 || group >= numGroups) continue;
                    double weight = scale * kernelData[group * numSubs + e][c][j / kernelWidth][j % kernelWidth];
                    if (weight == 0.0) continue;
                    nonZero = true;
                    for (int y = 0; y < outputWidth; y++) {
                        for (int x = 0; x < outputWidth; x++) {
                            values[c * channelStride + y * rowStride + x * colStride] = weight;
                        }
                    }
                }
                if (nonZero) {
                    encoded_kernel[e][t][j] = context->MakeCKKSPackedPlaintext(values, 1, encode_level, nullptr, numSlots);
                }
            }
        }
    }
    return encoded_kernel;
}

/**
 * @brief Encode the bias of a convolution in the output layout of encode_kernel_layout.
 *
 * @param biasData Bias values, one per output channel.
 * @param numSlots Slot count of the ciphertext.
 * @param encode_level Level the plaintext is encoded at.
 * @param channelStride Slots between two output groups.
 * @param rowStride Slots between two rows.
 * @param colStride Slots between two columns.
 * @param outputWidth Spatial width of one output channel.
 * @param numGroups Number of output groups.
 * @param subShifts Slot offset of each channel inside its group.
 * @return Encoded plaintext.
 */
Ptext FHEONHEController::encode_bias_layout(const vector<double>& biasData, int numSlots, int encode_level, int channelStride, int rowStride,
                            int colStride, int outputWidth, int numGroups, const vector<int>& subShifts) {
    int numSubs = subShifts.size();
    vector<double> values(numSlots, 0.0);
    for (int g = 0; g < numGroups; g++) {
        for (int e = 0; e < numSubs; e++) {
            for (int y = 0; y < outputWidth; y++) {
                for (int x = 0; x < outputWidth; x++) {
                    values[g * channelStride + y * rowStride + x * colStride + subShifts[e]] = biasData[g * numSubs + e];
                }
            }
        }
    }
    return context->MakeCKKSPackedPlaintext(values, 1, encode_level, nullptr, numSlots);
}

/**
 * @brief Encode a fully connected layer as diagonals for he_linear_diagonal.
 *
 * The layer is seen as an outputs x numSlots matrix whose column q holds the weights of the
 * input stored in slot q, wherever the inputs are: inputPositions lists, for every input, the
 * slot(s) holding it. An input listed with several slots is the sum of those slots, which
 * folds a sum pooling into the layer. Only numDiagonals diagonals (the outputs rounded up to
 * a power of two) are needed whatever the slot count: diagonal d is
 * D_d[p] = W[p mod numDiagonals][input in slot p + d], and he_linear_diagonal folds the
 * numSlots / numDiagonals partial sums afterwards. The diagonals are split in baby and giant
 * steps and pre-rotated for the giant steps.
 *
 * @param weightData Weights [outputs][inputs].
 * @param inputPositions For each input, the slots that hold it.
 * @param scale Factor multiplied into every weight (e.g. 0.25 for a fused 2x2 pooling).
 * @param numSlots Slot count of the input ciphertext.
 * @param numDiagonals Power of two >= outputs that divides numSlots.
 * @param babySteps Power of two that divides numDiagonals.
 * @param encode_level Level the plaintexts are encoded at.
 * @return [giant step][baby step] plaintexts; entries whose values are all zero are null.
 */
vector<vector<Ptext>> FHEONHEController::encode_linear_diagonal(const vector<vector<double>>& weightData, const vector<vector<int>>& inputPositions,
                            double scale, int numSlots, int numDiagonals, int babySteps, int encode_level) {
    int outputs = weightData.size();
    int inputs = weightData[0].size();
    if (outputs > numDiagonals || numSlots % numDiagonals != 0 || numDiagonals % babySteps != 0 || static_cast<int>(inputPositions.size()) != inputs) {
        throw std::invalid_argument("encode_linear_diagonal: the layer does not fit " + to_string(numDiagonals) + " diagonals of " + to_string(numSlots) + " slots");
    }
    vector<int> slotInput(numSlots, -1);
    for (int i = 0; i < inputs; i++) {
        for (int position : inputPositions[i]) {
            slotInput[position] = i;
        }
    }

    int giantSteps = numDiagonals / babySteps;
    vector<vector<Ptext>> encoded_diagonals(giantSteps, vector<Ptext>(babySteps));
    vector<double> values(numSlots);
    for (int g = 0; g < giantSteps; g++) {
        for (int b = 0; b < babySteps; b++) {
            int d = g * babySteps + b;
            bool nonZero = false;
            for (int q = 0; q < numSlots; q++) {
                /** value of diagonal d at slot p, stored pre-rotated at slot q = p + g * babySteps */
                int p = ((q - g * babySteps) % numSlots + numSlots) % numSlots;
                int output = p % numDiagonals;
                int input = slotInput[(p + d) % numSlots];
                values[q] = (output < outputs && input >= 0) ? scale * weightData[output][input] : 0.0;
                nonZero = nonZero || values[q] != 0.0;
            }
            if (nonZero) {
                encoded_diagonals[g][b] = context->MakeCKKSPackedPlaintext(values, 1, encode_level, nullptr, numSlots);
            }
        }
    }
    return encoded_diagonals;
}

/**
 * @brief Generate a CKKS context without bootstrapping, for circuits that fit the modulus chain.
 *
 * generate_context always adds the bootstrapping depth to the chain and generates the
 * bootstrapping keys. A network shallow enough to be evaluated on the fresh ciphertext needs
 * neither: here config.mlevelBootstrap is the whole multiplicative depth of the chain, and
 * config.levelBudget is unused. The shorter chain makes the keys smaller and the key
 * switching (rotations, ciphertext multiplications) faster.
 *
 * @param config Ring dimension, slots, depth (mlevelBootstrap), moduli sizes and digits.
 */
void FHEONHEController::generate_leveled_context(HEConfig config) {
    if (!config.keysFolder.empty()) {
        keys_folder = config.keysFolder;
    }
    update_key_paths();

    CCParams<CryptoContextCKKSRNS> parameters;
    level_budget = config.levelBudget;
    num_slots = 1 << config.numSlots;
    mult_depth = config.mlevelBootstrap;
    circuit_depth = config.mlevelBootstrap;

    security_level = config.securityLevel;
    fheon_apply_security(parameters, security_level, config.ringDim, num_slots, false);
    parameters.SetScalingModSize(config.dcrtBits);
    parameters.SetFirstModSize(config.firstMod);
    parameters.SetNumLargeDigits(config.numDigits);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetMultiplicativeDepth(circuit_depth);

    cout << "Building the leveled FHE Context (no bootstrapping), security level: "
         << security_level_name(security_level) << endl;
    cout << "dcrtBits: "<< config.dcrtBits << " -- firstMod: " << config.firstMod << endl << "Ciphertexts depth: " << circuit_depth << endl;

    context = fheon_gen_context(parameters, num_slots, config.minScaleBits, config.maxDigits);
    if (config.fillSlots) {
        context = fheon_fill_slots(parameters, context, num_slots);
    }
    scale_bits = parameters.GetScalingModSize();
    num_digits = parameters.GetNumLargeDigits();
    context->Enable(PKE);
    context->Enable(KEYSWITCH);
    context->Enable(LEVELEDSHE);
    context->Enable(ADVANCEDSHE);

    keyPair = context->KeyGen();
    context->EvalMultKeyGen(keyPair.secretKey);
    if (config.contextSumKeys) {
        context->EvalSumKeyGen(keyPair.secretKey);
    }

    string security = fheon_security_report(context, security_level, num_slots, parameters);
    cout << "Keys Generated." << endl;
    cout << "CKKS scheme is using ring dimension: " << context->GetRingDimension() << endl;
    cout << "Avaliable numSlots: " << num_slots << endl;
    cout << security << endl;
    cout << "-----------------------------------------------------------" << endl;

    if(config.serialize){
        write_to_file(keys_folder + "/security.txt", security);
        write_to_file(keys_folder + "/mult_depth.txt", to_string(mult_depth));
        write_to_file(keys_folder + "/num_slots.txt", to_string(num_slots));
        write_to_file(keys_folder + "/level_budget.txt", to_string(level_budget[0]) + "," + to_string(level_budget[1]));
        keys_serialization();
    }
}

/**
 * @brief Set up bootstrapping for a slot count that generate_context did not prepare.
 *
 * generate_context prepares the context slot count, its half and its quarter. A ciphertext
 * repacked to fewer slots (e.g. the 128 slots of a fully connected layer) bootstraps much
 * faster, but needs its own precomputations and keys.
 *
 * @param numSlots Slot count of the ciphertexts to bootstrap (power of two, not log2).
 */
void FHEONHEController::generate_bootstrapping_keys_for_slots(uint32_t numSlots){
    context->EvalBootstrapSetup(level_budget, bsgsDim, numSlots, bootstrap_correction_factor);
    context->EvalBootstrapKeyGen(keyPair.secretKey, numSlots);
}

/**
 * @brief Use the same bootstrapping correction factor for every slot count set up from now on.
 *
 * OpenFHE keeps a single correction factor for the context, and by default each
 * EvalBootstrapSetup overwrites it with a value that depends on the slot count being set up.
 * Once several slot counts are set up (one per stage of a network), the precomputations of the
 * earlier ones no longer match the factor in use, and bootstrapping them can fail ("approximation
 * error is too high"). With a fixed factor, generate_bootstrapping_keys and
 * generate_bootstrapping_keys_for_slots set every slot count up consistently. 0 restores the
 * OpenFHE default.
 *
 * @param correctionFactor Correction factor for all following bootstrapping setups.
 */
void FHEONHEController::set_bootstrap_correction_factor(uint32_t correctionFactor){
    bootstrap_correction_factor = correctionFactor;
}

/**
 * @brief Redo the bootstrapping precomputations for one slot count, without generating keys.
 *
 * When several slot counts are set up in one context, OpenFHE did not bootstrap all of them
 * reliably: depending on the order of the setups, the first bootstrapping at one of the earlier
 * slot counts failed ("approximation error is too high"), and redoing the setup for that slot
 * count right before using it always fixed it. Call this when a stage with a new slot count
 * starts (e.g. right after loading its keys with read_evaluation_keys).
 *
 * @param numSlots Slot count of the ciphertexts bootstrapped next (power of two, not log2).
 */
void FHEONHEController::prepare_bootstrapping(uint32_t numSlots){
    context->EvalBootstrapSetup(level_budget, bsgsDim, numSlots, bootstrap_correction_factor);
}

/**
 * @brief Encode fully connected weights for he_globalavgpool_linear.
 *
 * Plaintext k holds weight[k][c] / inputWidth^2 at slot c * inputWidth^2 + k, which folds the
 * averaging factor of the global pooling into the linear layer.
 *
 * @param weightData Weights [outputs][inputChannels].
 * @param inputWidth Spatial width of one input channel.
 * @param numSlots Slot count of the ciphertext (inputChannels * inputWidth^2).
 * @param encode_level Level the plaintexts are encoded at.
 * @return One plaintext per output.
 */
vector<Ptext> FHEONHEController::encode_avgpool_linear(const vector<vector<double>>& weightData, int inputWidth, int numSlots, int encode_level) {
    int blockSize = inputWidth * inputWidth;
    int outputs = weightData.size();
    int inputChannels = weightData[0].size();
    if (inputChannels * blockSize != numSlots || outputs > blockSize) {
        throw std::invalid_argument("encode_avgpool_linear: channels do not fit the " + to_string(numSlots) + " slots");
    }
    vector<Ptext> encoded_weights(outputs);
    for (int k = 0; k < outputs; k++) {
        vector<double> values(numSlots, 0.0);
        for (int ch = 0; ch < inputChannels; ch++) {
            values[ch * blockSize + k] = weightData[k][ch] / blockSize;
        }
        encoded_weights[k] = context->MakeCKKSPackedPlaintext(values, 1, encode_level, nullptr, numSlots);
    }
    return encoded_weights;
}

/**
 * @brief Encode a 3D weight kernel using optimized masking for 3x3 convolution.
 *
 * @param kernelData 3D raw weights vector.
 * @param cols_square Square dimension.
 * @param encode_level Level budget.
 * @return Vector of encoded plaintexts.
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
 * @brief Decrypt and read the predicted label (argmax) from encrypted output.
 *
 * Prints prediction info based on verbosity levels. Optionally appends result to predictions file.
 *
 * @param inferencedData Ciphertext of inference results.
 * @param num_slots Number of slots to decode.
 * @param predictions_file_path File path to write the predicted label to.
 * @return 0 on success.
 */
int FHEONHEController::read_inferenced_label(Ctext inferencedData, int num_slots,  const string& predictions_file_path){
    auto decryptedValue = decrypt_data(inferencedData, num_slots);
    auto decryptedVector = decryptedValue->GetRealPackedValue();
    auto maxElementIt = max_element(decryptedVector.begin(), decryptedVector.end());
    int maxIndex = distance(decryptedVector.begin(), maxElementIt);
    
    if (verbose_level == level_zero) {
        cout << "Predicted Value : " << maxIndex << endl;
    } else if (verbose_level == level_one) {
        cout << "Predicted Value : " << maxIndex << " Weight:  " << decryptedVector[maxIndex] << endl;
    } else if (verbose_level >= level_two) {
        cout << "Predicted Value : " << maxIndex << " Weight:  " << decryptedVector[maxIndex] << endl;
        cout << "Weights for the entire vector: " << decryptedVector << endl;
    }
    recordPrediction(maxIndex);

    double total_bts_secs = image_bootstrap_time_ms / 1000.0;
    double image_crt_time_secs = accumulated_circuit_time_ms / 1000.0;
    image_runtime_secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - image_start_time).count();
    cout << "Total Bootstrapping Time: " << total_bts_secs << "s" << endl;
    cout << "Total Circuit Time: " << image_crt_time_secs << "s" << endl;
    cout << "Total Runtime: " << image_runtime_secs << "s" << endl;
    printPeakMemory();

    if (!predictions_file_path.empty()) {
        ofstream outFile(predictions_file_path, ios_base::app);
        if (outFile.is_open()) {
            outFile << maxIndex << endl;
        }
    }
    return 0;
}

/**
 * @brief Decrypt and print minimum and maximum values of ciphertext slots.
 *
 * @param inferencedData Ciphertext to analyze.
 * @param num_slots Number of slots to decrypt.
 * @return 0 on success.
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
 * @brief Decrypt and read scale value (max absolute ceiling) for ReLU activation.
 *
 * @param inferencedData Ciphertext to analyze.
 * @param num_slots Number of slots to decrypt.
 * @return Max absolute value rounded up.
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
 * @brief Decrypt and read the largest absolute value, without rounding.
 *
 * Same purpose as read_scaling_value, which rounds up to an integer (at least 1): for
 * activations kept within [-1, 1] the exact maximum lets the ReLU approximation use its whole
 * input range.
 *
 * @param inferencedData Ciphertext to analyze.
 * @param num_slots Number of slots to decrypt.
 * @return Largest absolute value of the first num_slots slots.
 */
double FHEONHEController::read_max_abs(Ctext inferencedData, int num_slots){
    auto decryptedVector = decrypt_data(inferencedData, num_slots)->GetRealPackedValue();
    double maxAbsValue = 0.0;
    for (double value : decryptedVector) {
        maxAbsValue = std::max(maxAbsValue, std::abs(value));
    }
    return maxAbsValue;
}

/**
 * @brief Generate a tiled binary mask for convolution operations.
 *
 * @param starting_padding Zeros at the start.
 * @param ending_padding Zeros at the end.
 * @param window_length Length of the convolution window of 1s.
 * @param max_length Total slots per channel.
 * @param tile_count Number of channels.
 * @return Tiled mask data.
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
 * @brief Decrypt and read predicted labels for a batch of inferences.
 *
 * Extracts the argmax class index for each sample in the packed batch.
 *
 * @param inferencedData Ciphertext containing packed batch results.
 * @param batchSize Number of samples in the batch.
 * @param numClasses Number of output classes.
 * @param predictions_file_path Path to write the predicted labels.
 * @param baseIndex Index offset for batch tracking.
 * @return Vector of predicted class labels.
 */
vector<int> FHEONHEController::read_batch_inferenced_label(Ctext inferencedData,  int batchSize, int numClasses,  const string& predictions_file_path, int baseIndex){
     int totalElements = batchSize * numClasses;

    // Decrypt
    auto decryptedVal = decrypt_data(inferencedData, totalElements);
    auto decryptedVec = decryptedVal->GetRealPackedValue();
    if (verbose_level >= level_two) {
        cout << "DecryptedVec: " << decryptedVec << endl;
    }

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
        
        predictedLabels[b] = label;

        if (verbose_level == level_zero) {
            cout << "Batch " << baseIndex+b << " -> Predicted Label: " << label << endl;
        } else if (verbose_level == level_one) {
            cout << "Batch " << baseIndex+b << " -> Predicted Label: " << label << " (Score: " << *maxIt << ")" << endl;
        } else if (verbose_level >= level_two) {
            cout << "Batch " << baseIndex+b << " -> Predicted Label: " << label << " (Score: " << *maxIt << ")" << endl;
            cout << "Weights for Batch " << baseIndex+b << ": [ ";
            for (int i = startIdx; i < endIdx; i++) {
                cout << decryptedVec[i] << " ";
            }
            cout << "]" << endl;
        }

        if (!predictions_file_path.empty()) {
            ofstream outFile(predictions_file_path, ios_base::app);
            if (outFile.is_open()) {
                outFile << label << endl;
            }
        }
    }

    double total_bts_secs = image_bootstrap_time_ms / 1000.0;
    double image_crt_time_secs = accumulated_circuit_time_ms / 1000.0;
    image_runtime_secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - image_start_time).count();
    cout << "Total Bootstrapping Time: " << total_bts_secs << "s" << endl;
    cout << "Total Circuit Time: " << image_crt_time_secs << "s" << endl;
    cout << "Total Runtime: " << image_runtime_secs << "s" << endl;
    printPeakMemory();

    return predictedLabels;
}


/**
 * @brief Decrypt and read predicted labels from multiple independent ciphertexts.
 *
 * Each sample is represented by a separate ciphertext.
 *
 * @param inferencedData Vector of ciphertexts, one per batch sample.
 * @param batchSize Number of samples in the batch.
 * @param numClasses Number of output classes.
 * @param predictions_file_path Path to write the predicted labels.
 * @param baseIndex Index offset for batch tracking.
 * @return Vector of predicted class labels.
 */
vector<int> FHEONHEController::read_batch_inferenced_label_multiple_outputs(vector<Ctext> inferencedData,  int batchSize, int numClasses,  const string& predictions_file_path, int baseIndex){
    
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

        if (verbose_level == level_zero) {
            cout << "Batch " << baseIndex+b << " -> Predicted Label: " << label << endl;
        } else if (verbose_level == level_one) {
            cout << "Batch " << baseIndex+b << " -> Predicted Label: " << label << " (Score: " << *maxIt << ")" << endl;
        } else if (verbose_level >= level_two) {
            cout << "Batch " << baseIndex+b << " -> Predicted Label: " << label << " (Score: " << *maxIt << ")" << endl;
            cout << "Weights for Batch " << baseIndex+b << ": " << decryptedVec << endl;
        }

        if (!predictions_file_path.empty()) {
            ofstream outFile(predictions_file_path, ios_base::app);
            if (outFile.is_open()) {
                outFile << label << endl;
            }
        }
    }

    double total_bts_secs = image_bootstrap_time_ms / 1000.0;
    double image_crt_time_secs = accumulated_circuit_time_ms / 1000.0;
    image_runtime_secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - image_start_time).count();
    cout << "Total Bootstrapping Time: " << total_bts_secs << "s" << endl;
    cout << "Total Circuit Time: " << image_crt_time_secs << "s" << endl;
    cout << "Total Runtime: " << image_runtime_secs << "s" << endl;
    printPeakMemory();

    return predictedLabels;
}

/**
 * @brief Report the real images of a batch whose logits are in one ciphertext (image b in slots b * numClasses ...).
 *
 * Same output as read_batch_inferenced_label, for a batch completed with blank images: only the first numImages images
 * are reported (report_batch_predictions: dataset images firstIndex, firstIndex + 1, ..., with their accuracy), followed
 * by the totals of the run.
 */
vector<int> FHEONHEController::report_batch_labels(Ctext inferencedData, int batchSize, int numClasses, int firstIndex, int numImages,
                                                   const string& predictions_file_path){
    auto decryptedVec = decrypt_data(inferencedData, batchSize * numClasses)->GetRealPackedValue();
    auto labels = utilsbatchdata::report_batch_predictions(firstIndex, numImages, numClasses,
                                                           [&](int img, int k) { return decryptedVec[img * numClasses + k]; }, predictions_file_path);
    printTotalDurations();
    return labels;
}

/**
 * @brief Report the real images of a batch with one logit ciphertext per image (he_batch_linear_multiple_outputs).
 */
vector<int> FHEONHEController::report_batch_labels_multiple_outputs(vector<Ctext>& inferencedData, int numClasses, int firstIndex, int numImages,
                                                                    const string& predictions_file_path){
    vector<vector<double>> decrypted;
    for (int b = 0; b < numImages; b++) decrypted.push_back(decrypt_data(inferencedData[b], numClasses)->GetRealPackedValue());
    auto labels = utilsbatchdata::report_batch_predictions(firstIndex, numImages, numClasses,
                                                           [&](int img, int k) { return decrypted[img][k]; }, predictions_file_path);
    printTotalDurations();
    return labels;
}

//read the predicted label
// read predictions across the same index across multiple ciphertexts (e.g., for multiple outputs of a model) and determine the predicted label for each batch by finding the maximum value across the corresponding decrypted vectors.
vector<int> FHEONHEController::read_forward_inferenced_label_multiple_outputs(vector<Ctext>& inferencedData,  int batchSize, int numClasses,  ofstream& outFile, int baseIndex){
    vector<int> predictedLabels(batchSize, -1);
    vector<double> bestScores(batchSize, -std::numeric_limits<double>::infinity());

    if ((int)inferencedData.size() != numClasses) {
        cout << "Warning: expected " << numClasses << " ciphertexts, but received " << inferencedData.size() << "." << endl;
    }

    int availableClasses = std::min((int)inferencedData.size(), numClasses);
    for (int c = 0; c < availableClasses; ++c) {
        auto decryptedVal = decrypt_data(inferencedData[c], batchSize);
        auto decryptedVec = decryptedVal->GetRealPackedValue();
        int readSize = std::min(batchSize, (int)decryptedVec.size());

        for (int b = 0; b < readSize; ++b) {
            if (decryptedVec[b] > bestScores[b]) {
                bestScores[b] = decryptedVec[b];
                predictedLabels[b] = c;
            }
        }
    }

    for (int b = 0; b < batchSize; ++b) {
        int label = predictedLabels[b];
        double score = bestScores[b];

        cout << "Batch " << baseIndex + b
             << " -> Predicted Label: " << label
             << " (Score: " << score << ")" << endl;

        if (outFile.is_open()) {
            outFile << label << endl;
        }
    }

    return predictedLabels;
}


/**
 * @brief Decrypt and read scaling values for batch ReLU operations.
 *
 * @param encryptedInputs Vector of ciphertexts.
 * @param inputChannels Number of channels.
 * @param num_slots Number of slots per channel.
 * @return Vector of scaling factors.
 */
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

/**
 * @brief Decrypt and print min/max values for batch ciphertexts.
 *
 * @param encryptedInputs Vector of ciphertexts.
 * @param inputChannels Number of channels.
 * @param num_slots Number of slots per channel.
 * @return 0 on success.
 */
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
 * @brief Decrypt and print multiple ciphertexts representing channels or batches.
 *
 * @param encryptedInputs Vector of ciphertexts to decrypt.
 * @param inputChannels Number of ciphertexts/channels.
 * @param num_slots Number of slots to decode per ciphertext.
 * @return 0 on success.
 */
int FHEONHEController::decrypt_batch_data(vector<Ctext>& encryptedInputs, int inputChannels,  int num_slots) {
    
    vector<int> maxVec;
    for(int i=0; i<inputChannels; i++){
        auto decryptedVal = decrypt_data(encryptedInputs[i], num_slots);
        auto decryptedVec = decryptedVal->GetRealPackedValue();
        // double val = accumulate(decryptedVec.begin(), decryptedVec.end(), 0.0);
        cout << "------------------------------------------------------------------ " << endl;
        cout << decryptedVec << endl;
        cout << "------------------------------------------------------------------ " << endl;
        // cout << "Accumulated: " << val << endl; 
    }
    return 0;
}


/**
 * @brief Decrypt and print complex packed values of a ciphertext.
 *
 * @param encryptedpackedVector Ciphertext to print.
 * @param num_slots Number of slots to print.
 * @return Vector of real decrypted values.
 */
vector<double> FHEONHEController::decrypt_and_print(Ctext encryptedpackedVector, int num_slots) {
    
    Ptext plaintextDec;
    context->Decrypt(keyPair.secretKey, encryptedpackedVector, &plaintextDec);
    plaintextDec->SetLength(num_slots);
    vector<complex<double>> finalResult = plaintextDec->GetCKKSPackedValue();
    cout << finalResult << endl;
    vector<double> realResult;
    // add only the number of slots specified by num_slots
    for (int i = 0; i < num_slots; i++) {
        realResult.push_back(finalResult[i].real());
    }
    cout << endl;
    return realResult;
}

double FHEONHEController::decrypt_scalar(Ctext& encryptedInput) {
    Ptext plaintextDec;
    context->Decrypt(keyPair.secretKey, encryptedInput, &plaintextDec);
    plaintextDec->SetLength(1);
    return plaintextDec->GetCKKSPackedValue()[0].real();
}

/**
 * @brief Decrypt ciphertext using an explicit private key.
 *
 * @param sk Explicit private key for decryption.
 * @param encryptedinputData Ciphertext to decrypt.
 * @param cols Number of slots to retrieve.
 * @return Decrypted plaintext object.
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
 * @brief Decrypt and read scaling value using an explicit private key.
 *
 * @param sk Explicit private key.
 * @param inferencedData Ciphertext to analyze.
 * @param num_slots Number of slots.
 * @return Max absolute value rounded up.
 */
int FHEONHEController::read_scaling_value_with_key(PrivateKey<DCRTPoly> &sk,
                                                   Ctext inferencedData,
                                                   int num_slots) {
    // int roundedMaxAbsValue = 10; // Temporary hardcoded value for testing
    auto decryptedValue = decrypt_data_with_key(sk, inferencedData, num_slots);
    auto decryptedVector = decryptedValue->GetRealPackedValue();

    cout << endl
        << "--------------------------------------------------- " << endl
        << endl;
    cout << "Decrypted Vector for Scaling Value: " << decryptedVector << endl;
    cout << endl
        << "--------------------------------------------------- " << endl;

    double maxAbsValue =
        *std::max_element(decryptedVector.begin(), decryptedVector.end(),
                            [](int a, int b) { return std::abs(a) < std::abs(b); });
    int roundedMaxAbsValue = static_cast<int>(std::ceil(std::abs(maxAbsValue)));
    return roundedMaxAbsValue;
}


/**
 * @brief Decrypt and read prediction index using an explicit private key.
 *
 * @param sk Explicit private key for decryption.
 * @param inferencedData Ciphertext of inference results.
 * @param num_slots Number of slots to decode.
 * @param predictions_file_path File path to write the predicted label to.
 * @return Index corresponding to the predicted label.
 */
int FHEONHEController::read_inferenced_label_with_key(PrivateKey<DCRTPoly> &sk,
                                                      Ctext inferencedData,
                                                      int num_slots,
                                                      const string& predictions_file_path) {
    auto decryptedValue = decrypt_data_with_key(sk, inferencedData, num_slots);
    auto decryptedVector = decryptedValue->GetRealPackedValue();
    auto maxElementIt =
        max_element(decryptedVector.begin(), decryptedVector.end());
    int maxIndex = distance(decryptedVector.begin(), maxElementIt);

    if (verbose_level == level_zero) {
        cout << "Predicted Value : " << maxIndex << endl;
    } else if (verbose_level == level_one) {
        cout << "Predicted Value : " << maxIndex << " Weight:  " << decryptedVector[maxIndex] << endl;
    } else if (verbose_level >= level_two) {
        cout << "Predicted Value : " << maxIndex << " Weight:  " << decryptedVector[maxIndex] << endl;
        cout << "Decrypted Vector: " << decryptedVector << endl;
    }
    recordPrediction(maxIndex);

    double total_bts_secs = image_bootstrap_time_ms / 1000.0;
    double total_run_secs = duration_cast<milliseconds>(steady_clock::now() - image_start_time).count() / 1000.0;
    cout << "Total Bootstrapping Time: " << total_bts_secs << "s" << endl;
    cout << "Total Circuit Time: " << image_circuit_time_secs << "s" << endl;
    cout << "Total Runtime: " << total_run_secs << "s" << endl;
    printPeakMemory();

    if (!predictions_file_path.empty()) {
        ofstream outFile(predictions_file_path, ios_base::app);
        if (outFile.is_open()) {
            outFile << maxIndex << endl;
        }
    }
    return maxIndex;
}

/**
 * @brief Clear automorphism keys from the given crypto context.
 *
 * @param crypto_context Context to clear keys from.
 */
void FHEONHEController::harness_clear_bootstrapping_and_rotation_keys(CryptoContext<DCRTPoly> &crypto_context) {
  // crypto_context->ClearEvalMultKeys();
  crypto_context->ClearEvalAutomorphismKeys();
  // CryptoContextFactory<lbcrypto::DCRTPoly>::ReleaseAllContexts();
}

