
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
 * @file Utils.h
 * @brief Plain secure utility functions used as general helpers throughout the project.
 *
 * This file defines various helper functions for secure computations and 
 * homomorphic operations, providing reusable utilities across the project.
 */

#ifndef FHEON_UTILS_H
#define FHEON_UTILS_H

#include <iostream>
#include <openfhe.h>
#include <iomanip>
#include <sstream>
#include <mutex>
#include <thread>
#include <exception>
#include <atomic>
#include <condition_variable>
#include <omp.h>
#include <sys/resource.h>

using namespace std;
using namespace std::chrono;
using namespace lbcrypto;
using Ctext = Ciphertext<DCRTPoly>;

inline int verbose_level = 2;
inline int level_zero = 0;
inline int level_one = 1;
inline int level_two = 2;

inline double image_circuit_time_secs = 0.0;
inline double image_runtime_secs = 0.0;
inline double image_bootstrap_time_ms = 0.0;
inline double accumulated_circuit_time_ms = 0.0;
inline std::mutex timing_mutex;
inline std::chrono::time_point<std::chrono::steady_clock, std::chrono::nanoseconds> image_start_time;
inline std::chrono::time_point<std::chrono::steady_clock, std::chrono::nanoseconds> program_start_time;
inline int operation_count = 0;
inline bool measure_peak_memory = true;
/** largest number of worker threads of the multithreaded primitives: the system threads by default, set with --threads (0 also uses
 *  hardware_concurrency()). OpenFHE already parallelizes every operation with OpenMP, and each worker thread holds its own
 *  ciphertexts and bootstrapping temporaries, so fewer workers trade little speed for much less memory. */
inline int max_threads = std::max(1, (int)std::thread::hardware_concurrency());

/** number of worker threads for totalWork independent tasks: at most max_threads (hardware_concurrency() when 0) */
static inline int worker_threads(int totalWork) {
    int limit = max_threads > 0 ? max_threads : (int)std::thread::hardware_concurrency();
    return std::max(1, std::min(totalWork, limit > 0 ? limit : 4));
}
inline std::string last_printed_peak_memory = "";

/** true inside a worker of run_parallel_for: a nested parallel loop runs in its calling thread instead of starting
 *  threads of its own (OpenFHE already uses OpenMP inside every operation) */
inline thread_local bool in_parallel_region = false;

/** OpenMP threads of the OpenFHE operations run by one of numWorkers worker threads: the hardware threads are shared
 *  among the workers (OpenFHE parallelizes every operation with OpenMP, a team per worker would oversubscribe the
 *  cores); the calling thread keeps the default OpenMP setting */
static inline void use_worker_openmp_share(int numWorkers) {
    int hardware = static_cast<int>(std::thread::hardware_concurrency());
    omp_set_num_threads(std::max(1, hardware / std::max(1, numWorkers)));
}

/** number of workers for totalWork tasks under maxThreads: 1 runs in the calling thread, 0 uses worker_threads()
 *  (--threads or all hardware threads), a larger value caps worker_threads() */
static inline int parallel_worker_count(int totalWork, int maxThreads) {
    if (totalWork <= 1 || maxThreads == 1 || in_parallel_region) return 1;
    int numThreads = worker_threads(totalWork);
    return maxThreads > 1 ? std::min(numThreads, maxThreads) : numThreads;
}

/**
 * @brief Split [0, totalWork) into contiguous blocks and run worker(start, end) on each block in its own thread.
 *
 * With one worker (maxThreads == 1, a single task, or a call from inside another parallel loop) worker(0, totalWork)
 * runs in the calling thread, so a sequential controller executes exactly its original loop. An exception thrown in a
 * worker is rethrown in the caller once all workers have joined.
 */
template <typename Func>
static inline void run_parallel_for(int totalWork, Func worker, int maxThreads = 0) {
    if (totalWork <= 0) return;
    int numThreads = parallel_worker_count(totalWork, maxThreads);
    if (numThreads == 1) {
        worker(0, totalWork);
        return;
    }
    std::vector<std::thread> threads;
    std::vector<std::exception_ptr> errors(numThreads);
    int block = (totalWork + numThreads - 1) / numThreads;
    for (int t = 0; t < numThreads; t++) {
        int start = t * block;
        int end = std::min(start + block, totalWork);
        if (start >= end) break;
        threads.emplace_back([&worker, &errors, t, start, end, numThreads]() {
            in_parallel_region = true;
            use_worker_openmp_share(numThreads);
            try {
                worker(start, end);
            } catch (...) {
                errors[t] = std::current_exception();
            }
        });
    }
    for (auto &th : threads) th.join();
    for (auto &error : errors) {
        if (error) std::rethrow_exception(error);
    }
}

/**
 * @brief Ordered fold with parallel terms: compute(i) for the indices of order, in blocks of one result per worker,
 * then consume(i, result) for every index in the order of order.
 *
 * For folds that depend on the order (Horner placements, merges): the independent terms are computed in parallel while
 * at most one block of results is held. With one worker it is the plain loop compute-then-consume per index.
 */
template <typename T, typename Compute, typename Consume>
static inline void run_parallel_ordered(const std::vector<int>& order, Compute compute, Consume consume, int maxThreads = 0) {
    int total = static_cast<int>(order.size());
    int block = parallel_worker_count(total, maxThreads);
    std::vector<T> results(block);
    for (int first = 0; first < total; first += block) {
        int count = std::min(block, total - first);
        run_parallel_for(count, [&](int start, int end) {
            for (int k = start; k < end; k++) {
                results[k] = compute(order[first + k]);
            }
        }, maxThreads);
        for (int k = 0; k < count; k++) {
            consume(order[first + k], results[k]);
            results[k] = T();
        }
    }
}

/**
 * @brief Pipeline over a sequential chain: produce(i) in the order of order (e.g. the next rotation of a running
 * ciphertext), compute(i, input) for a block of indices in parallel, then consume(i, result) in order.
 *
 * Holds at most one block of inputs and results; with one worker it is the plain loop produce-compute-consume.
 */
template <typename In, typename Out, typename Produce, typename Compute, typename Consume>
static inline void run_parallel_pipeline(const std::vector<int>& order, Produce produce, Compute compute, Consume consume, int maxThreads = 0) {
    int total = static_cast<int>(order.size());
    int block = parallel_worker_count(total, maxThreads);
    std::vector<In> inputs(block);
    std::vector<Out> results(block);
    for (int first = 0; first < total; first += block) {
        int count = std::min(block, total - first);
        for (int k = 0; k < count; k++) {
            inputs[k] = produce(order[first + k]);
        }
        run_parallel_for(count, [&](int start, int end) {
            for (int k = start; k < end; k++) {
                results[k] = compute(order[first + k], inputs[k]);
            }
        }, maxThreads);
        for (int k = 0; k < count; k++) {
            inputs[k] = In();
            consume(order[first + k], results[k]);
            results[k] = Out();
        }
    }
}

/**
 * @brief Streaming ordered fold: worker threads compute(i) for the indices of order, taken in that order from a shared
 * counter, while the calling thread runs consume(i, result) in the order of order as soon as each result is ready.
 *
 * There is no barrier between blocks: a fold that must keep its order (a Horner placement with a single rotation key,
 * merges) overlaps with the parallel computation of its terms. With one worker it is the plain loop compute-then-consume
 * per index. An exception in compute or consume stops the workers and is rethrown in the caller.
 */
template <typename T, typename Compute, typename Consume>
static inline void run_parallel_stream(const std::vector<int>& order, Compute compute, Consume consume, int maxThreads = 0) {
    int total = static_cast<int>(order.size());
    int numThreads = parallel_worker_count(total, maxThreads);
    if (numThreads == 1) {
        for (int k = 0; k < total; k++) {
            T result = compute(order[k]);
            consume(order[k], result);
        }
        return;
    }
    std::vector<T> results(total);
    std::vector<char> state(total, 0);      /** 0 pending, 1 ready, 2 failed */
    std::mutex lock;
    std::condition_variable readyCondition;
    std::atomic<int> next{0};
    std::exception_ptr error;
    auto worker = [&]() {
        in_parallel_region = true;
        use_worker_openmp_share(numThreads);
        while (true) {
            int k = next.fetch_add(1);
            if (k >= total) break;
            try {
                T result = compute(order[k]);
                std::lock_guard<std::mutex> guard(lock);
                results[k] = std::move(result);
                state[k] = 1;
            } catch (...) {
                std::lock_guard<std::mutex> guard(lock);
                if (!error) error = std::current_exception();
                state[k] = 2;
                next = total;
            }
            readyCondition.notify_all();
        }
    };
    std::vector<std::thread> threads;
    for (int t = 0; t < numThreads; t++) threads.emplace_back(worker);
    try {
        for (int k = 0; k < total; k++) {
            T result;
            {
                std::unique_lock<std::mutex> guard(lock);
                readyCondition.wait(guard, [&] { return state[k] != 0 || error; });
                if (state[k] != 1) break;
                result = std::move(results[k]);
                results[k] = T();
            }
            consume(order[k], result);
        }
    } catch (...) {
        std::lock_guard<std::mutex> guard(lock);
        if (!error) error = std::current_exception();
        next = total;
    }
    for (auto &th : threads) th.join();
    if (error) std::rethrow_exception(error);
}

/** indices n-1, n-2, ..., 0 (the order of the Horner placements) */
static inline std::vector<int> descending_indices(int n) {
    std::vector<int> order(std::max(n, 0));
    for (int i = 0; i < n; i++) order[i] = n - 1 - i;
    return order;
}

/** indices 0, 1, ..., n-1 */
static inline std::vector<int> ascending_indices(int n) {
    std::vector<int> order(std::max(n, 0));
    for (int i = 0; i < n; i++) order[i] = i;
    return order;
}



/**
 * @brief Peak resident memory of the process so far, formatted in GB.
 *
 * Reads the high-water mark kept by the kernel, so it covers everything since the program
 * started (context and key generation included).
 */
static inline std::string peakMemoryString() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    std::stringstream ss;
    ss << std::fixed << std::setprecision(1) << usage.ru_maxrss / (1024.0 * 1024.0) << " GB";
    return ss.str();
}

/**
 * @brief Print the peak memory next to the total timings. Does nothing when --peak-memory is false.
 */
static inline void printPeakMemory() {
    if (!measure_peak_memory) return;
    std::cout << "Peak Memory Used: " << peakMemoryString() << std::endl;
}

static inline void printTiming(const std::string &prefix, const std::string &op_tag, std::chrono::time_point<std::chrono::steady_clock, std::chrono::nanoseconds> start_time) {
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time);
    std::lock_guard<std::mutex> timingLock(timing_mutex);

    std::string print_prefix = prefix;
    if (op_tag != "Bts") {
        operation_count++;
        print_prefix = std::to_string(operation_count) + ". " + prefix;
        accumulated_circuit_time_ms += duration.count();
    }

    if (verbose_level == level_one) {
        std::cout << print_prefix << std::endl;
    } else if (verbose_level >= level_two) {
        auto total_prog_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - program_start_time);
        auto total_prog_secs = std::chrono::duration_cast<std::chrono::seconds>(total_prog_ms);
        auto dur_secs = std::chrono::duration_cast<std::chrono::seconds>(duration);
        auto dur_ms_rem = duration - std::chrono::duration_cast<std::chrono::milliseconds>(dur_secs);

        std::stringstream ss;
        ss << std::fixed << std::setprecision(3);
        if (op_tag == "Bts") {
            ss << print_prefix << " (Op: " << dur_secs.count() << ":" << dur_ms_rem.count() << "s (Bts:" << (image_bootstrap_time_ms / 1000.0) << "s -- Total: " << total_prog_secs.count() << "s)";
        } else {
            ss << print_prefix << " (Op: " << dur_secs.count() << ":" << dur_ms_rem.count() << "s (Crts:" << (accumulated_circuit_time_ms / 1000.0) << "s Total : " << total_prog_secs.count() << "s)";
        }
        if (measure_peak_memory) {
            // only shown when the peak has changed since the last timing line
            std::string peak_memory = peakMemoryString();
            if (peak_memory != last_printed_peak_memory) {
                ss << ", Mem: " << peak_memory;
                last_printed_peak_memory = peak_memory;
            }
        }
        ss << ")";
        std::cout << ss.str() << std::endl;
    }
}


static inline void measureTiming(const std::string &prefix, const std::string &op_tag, std::chrono::time_point<std::chrono::steady_clock, std::chrono::nanoseconds> start_time) {
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time);
    std::lock_guard<std::mutex> timingLock(timing_mutex);

    if (op_tag != "Bts") {
        accumulated_circuit_time_ms += duration.count();
    }
    else{
        image_bootstrap_time_ms += duration.count();
    }
}

static inline void printBootstrapTiming(const std::string &prefix, std::chrono::time_point<std::chrono::steady_clock, std::chrono::nanoseconds> start_bootstrap) {
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_bootstrap);
    {
        std::lock_guard<std::mutex> timingLock(timing_mutex);
        image_bootstrap_time_ms += duration.count();
    }
    printTiming(prefix, "Bts", start_bootstrap);
}

static inline void printModelLayer(const std::string &layer_name, int layer_num, int block_num = -1) {
    if (verbose_level == level_zero) return;
    if (verbose_level == level_one) {
        std::cout << layer_name << "    " << layer_num << std::endl;
    } else if (verbose_level >= level_two) {
        if (block_num != -1) {
            std::cout << layer_name << "    " << layer_num << " Block " << block_num << std::endl;
        } else {
            std::cout << layer_name << "    " << layer_num << std::endl;
        }
    }
}

static inline void printModelLayer(const std::string &layer_string) {
    if (verbose_level == level_zero) return;
    std::string name = "    Layer";
    int num = -1;
    int block = -1;
    size_t layer_pos = layer_string.find("layer");
    if (layer_pos != std::string::npos) {
        size_t num_start = layer_pos + 5;
        if (num_start < layer_string.size() && std::isdigit(layer_string[num_start])) {
            num = layer_string[num_start] - '0';
        }
    } else if (layer_string.find("conv") != std::string::npos) {
        size_t conv_pos = layer_string.find("conv");
        size_t num_start = conv_pos + 4;
        if (num_start < layer_string.size() && std::isdigit(layer_string[num_start])) {
            num = 0;
            while (num_start < layer_string.size() && std::isdigit(layer_string[num_start])) {
                num = num * 10 + (layer_string[num_start] - '0');
                num_start++;
            }
        }
    } else if (layer_string.find("fc") != std::string::npos) {
        size_t fc_pos = layer_string.find("fc");
        size_t num_start = fc_pos + 2;
        if (num_start < layer_string.size() && std::isdigit(layer_string[num_start])) {
            num = 0;
            while (num_start < layer_string.size() && std::isdigit(layer_string[num_start])) {
                num = num * 10 + (layer_string[num_start] - '0');
                num_start++;
            }
        }
    }
    size_t block_pos = layer_string.find("block");
    if (block_pos != std::string::npos) {
        size_t block_num_start = block_pos + 5;
        if (block_num_start < layer_string.size() && std::isdigit(layer_string[block_num_start])) {
            block = layer_string[block_num_start] - '0';
        }
    }
    if (layer_string == "layer_fc" || layer_string == "Classification") {
        name = "Classification";
        num = -1;
    }
    if (verbose_level == level_one) {
        if (name == "Classification") {
            std::cout << name << std::endl;
        } else if (num != -1) {
            std::cout << name << " " << num << std::endl;
        } else {
            std::cout << layer_string << std::endl;
        }
    } else if (verbose_level >= level_two) {
        if (name == "Classification") {
            std::cout << name << std::endl;
        } else if (num != -1) {
            if (block != -1) {
                std::cout << name << " " << num << " Block " << block << std::endl;
            } else {
                std::cout << name << " " << num << std::endl;
            }
        } else {
            std::cout << layer_string << std::endl;
        }
    }
}

namespace utils {

    using ::verbose_level;
    using ::level_zero;
    using ::level_one;
    using ::level_two;
    using ::image_circuit_time_secs;
    using ::image_bootstrap_time_ms;
    using ::accumulated_circuit_time_ms;
    using ::image_start_time;
    using ::program_start_time;
    using ::operation_count;
    using ::measure_peak_memory;
    using ::printPeakMemory;
    using ::printBootstrapTiming;
    using ::printTiming;
    using ::printModelLayer;

    static duration<long long, ratio<1, 1000>> total_time;

    inline string weights_folder = "./../weights/";
    inline bool has_custom_weights_folder = false;
    inline string cifar10tPath = "./../images/cifar-10-batches-bin/test_batch.bin";
    inline string cifar100tPath = "./../images/cifar-100-binary/test.bin";
    inline string mnistPath = "./../images/mnist_images/raw/t10k-images-idx3-ubyte";
    inline string predictions_file = "./fhepredictions.txt";
    inline int test_size = 250;
    inline int index_value = 0;

    /** ground-truth labels of the loaded images (label of image i at position i) and the running accuracy counters */
    inline vector<int> image_labels;
    inline int predictions_made = 0;
    inline int labelled_predictions = 0;
    inline int correct_predictions = 0;

    /**
     * @brief Compare a predicted class with the ground-truth label of the current image.
     *
     * Predictions are expected in dataset order starting at index_value, one call per image,
     * which is how the models loop. Prints whether the prediction is correct and updates the
     * counters used by printAccuracy. Does nothing more than count when no label is available.
     *
     * @param predicted Predicted class index.
     */
    static inline void recordPrediction(int predicted) {
        size_t image_index = index_value + predictions_made;
        predictions_made++;
        if (image_index >= image_labels.size()) return;
        int label = image_labels[image_index];
        bool correct = (predicted == label);
        labelled_predictions++;
        if (correct) correct_predictions++;
        cout << "True Label      : " << label << " -- " << (correct ? "Correct" : "Incorrect") << endl;
    }

    /**
     * @brief Print the accuracy over all the predictions recorded so far.
     */
    static inline void printAccuracy() {
        if (labelled_predictions == 0) return;
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << (100.0 * correct_predictions / labelled_predictions);
        cout << "Accuracy: " << correct_predictions << "/" << labelled_predictions << " (" << ss.str() << "%)" << endl;
    }
    
    inline string keys_folder = "./../AllHEKeys/";
    inline string dataset_path = "";
    inline string training_data_dir = "./../encrypted_training_data/";

    /**
     * @brief Print the welcome message with project and author details, and parsed runtime configuration.
     */
    static inline void printWelcomeMessage(){
        if (verbose_level == level_zero) return;
        cout<< "----------------------------------------------------------------------------------- " << endl;
        cout<< "-------------------------------- WELCOME TO FHEON --------------------------------- " << endl;
        cout<< "------------------ Nges Brian, Eric Jahns, Michel A. Kinsy ------------------------ " << endl;
        cout<< "---------- Secure, Trusted and Assured Microelectronics (STAM) CENTER ------------- " << endl;
        cout<< "---------------------------- Arizona State University ----------------------------- " << endl; 
        cout<< "----------------------------------------------------------------------------------- " << endl;
        cout<< "Active Configuration:" << endl;
        cout<< "  - Verbose Level:      " << verbose_level << endl;
        cout<< "  - Test Size:          " << utils::test_size << endl;
        cout<< "  - Start Index:        " << utils::index_value << endl;
        cout<< "  - Keys Folder:        " << utils::keys_folder << endl;
        cout<< "  - Weights Folder:     " << utils::weights_folder << endl;
        cout<< "  - Dataset Path:       " << utils::dataset_path << endl;
        cout<< "  - Training Data Dir:  " << utils::training_data_dir << endl;
        cout<< "  - Peak Memory:        " << (measure_peak_memory ? "true" : "false") << endl;
        cout<< "  - Worker Threads:     " << (max_threads > 0 ? to_string(max_threads) : "all (" + to_string(std::thread::hardware_concurrency()) + ")") << endl;
        if (!utils::predictions_file.empty()) {
            cout<< "  - Predictions File:   " << utils::predictions_file << endl;
        }
        cout<< "----------------------------------------------------------------------------------- " << endl;
        cout << endl;
    }

   /**
     * @brief Get the current time point for timing measurements.
     */
    static inline chrono::time_point<steady_clock, nanoseconds> startTime() {
        return steady_clock::now();
    }

    /**
     * @brief Print duration since start time, optionally tracking global execution time.
     *
     * @param start Start time point.
     * @param caption Caption for printed output (default: "Time Taken is:").
     * @param global_time If true, accumulates total runtime across calls.
     */
    static inline void printDuration(chrono::time_point<steady_clock, nanoseconds> start, const string &caption="Time Taken is: ", bool global_time=false) {
        if (verbose_level < level_two) return;
        auto ms = duration_cast<milliseconds>(steady_clock::now() - start);

        static duration<long long, ratio<1, 1000>> total_duration; 
        if(global_time){
            total_duration  = total_time + ms;
            total_time = total_duration;
        }
        else{
            total_duration = ms;
        }

        auto secs = duration_cast<seconds>(ms);
        ms -= duration_cast<milliseconds>(secs);
        auto mins = duration_cast<minutes>(secs);
        secs -= duration_cast<seconds>(mins);

        // cout<< endl;
        if (mins.count() < 1) {
            cout << "--- " << caption << ": " << secs.count() << ":" << ms.count() << "s" << " (Total: " << duration_cast<seconds>(total_duration).count() << "s)" << " --- " << endl;
        } else {
            cout << "--- " << caption << ": " << mins.count() << "." << secs.count() << ":" << ms.count() << " (Total: " << duration_cast<minutes>(total_duration).count() << "mins)" << " --- " << endl;
        }
        cout<< endl;
    }

    /**
     * print the duration of the image processing steps.
    */
    static inline void printTotalDurations() {
        std::lock_guard<std::mutex> timingLock(timing_mutex);
        double total_bts_secs = image_bootstrap_time_ms / 1000.0;
        double image_crt_time_secs = accumulated_circuit_time_ms / 1000.0;
        image_runtime_secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - image_start_time).count();
        cout << "Total Bootstrapping Time: " << total_bts_secs << "s" << endl;
        cout << "Total Circuit Time: " << image_crt_time_secs << "s" << endl;
        cout << "Total Runtime: " << image_runtime_secs << "s" << endl;
        printPeakMemory();
    }

    /**
     * print the duration of the processing steps of a batch of images: the totals and their share per image.
    */
    static inline void printBatchDurations(int numImages) {
        printTotalDurations();
        std::lock_guard<std::mutex> timingLock(timing_mutex);
        cout << "Bootstrapping Time per Image: " << image_bootstrap_time_ms / 1000.0 / numImages << "s" << endl;
        cout << "Circuit Time per Image: " << accumulated_circuit_time_ms / 1000.0 / numImages << "s" << endl;
        cout << "Runtime per Image: " << image_runtime_secs / numImages << "s" << endl;
    }

    /**
     * @brief Print bootstrapping metadata for a ciphertext.
     *
     * @param ciphertextIn Input ciphertext.
     * @param depth Maximum depth available.
     */
    static inline void printBootsrappingData(Ctext ciphertextIn, int depth){
        if (verbose_level == level_zero) return;
        std::cout << "Number of levels remaining: "
              << depth - ciphertextIn->GetLevel() - (ciphertextIn->GetNoiseScaleDeg() - 1) 
              << " ***Level: " << ciphertextIn->GetLevel() << " ***noiseScaleDeg: " 
              << ciphertextIn->GetNoiseScaleDeg() << std::endl;
    }

    /**
     * @brief Measure elapsed time between two time points in seconds.
     *
     * @param start Start time.
     * @param end End time.
     * @return Elapsed time in seconds.
     */
    static inline int measureTime(const time_point<high_resolution_clock>& start, const time_point<high_resolution_clock>& end) {
        auto duration = duration_cast<seconds>(end - start);
        int secs = duration.count();
        image_circuit_time_secs += secs;
        return secs;
    }

    /**
     * @brief Get current time point (high resolution).
     *
     * @return Current time point.
     */
    static inline time_point<high_resolution_clock> get_current_time() {
        return high_resolution_clock::now();
    }

    /**
     * @brief Compute total time from a vector of durations.
     *
     * @param measuring Vector of time values (seconds).
     * @return Total time in seconds.
     */
    static inline int totalTime(vector<int> measuring){
        int total = accumulate(measuring.begin(), measuring.end(), 0);
        if (verbose_level >= level_two) {
            cout << "------- Circuit Total Time: " << total << endl;
        }
        return total;
    }

    static inline int printTimeWithMessage(string cirStr, vector<int> measuring){
        int total = accumulate(measuring.begin(), measuring.end(), 0);
        if (verbose_level >= level_two) {
            cout << "------- " << cirStr <<" Time: " << total << endl;
        }
        return total;
    }

    class NullBuffer : public std::streambuf {
    public:
        int overflow(int c) override {
            return c;
        }
    };

    inline NullBuffer null_buffer;
    inline std::streambuf* original_cout_rdbuf = nullptr;

    static inline void restore_cout() {
        if (original_cout_rdbuf) {
            std::cout.rdbuf(original_cout_rdbuf);
            original_cout_rdbuf = nullptr;
        }
    }

    static inline void suppress_cout() {
        if (!original_cout_rdbuf) {
            original_cout_rdbuf = std::cout.rdbuf();
            std::cout.rdbuf(&null_buffer);
            std::atexit(restore_cout);
        }
    }

    struct RuntimeConfig {
        int test_size;
        int index_value;
        int verbose;
        bool load_context;
        string keys_folder;
        string weights_folder;
        string cifar10tPath;
        string cifar100tPath;
        string mnistPath;
        string predictions_file;
        string training_data_dir;
        bool peak_memory = true;
        int max_threads = std::max(1, (int)std::thread::hardware_concurrency());
    };

    static inline RuntimeConfig parse_runtime_args(int argc, char* argv[], int default_test_size, int default_index_value, int default_verbose = 2) {
        RuntimeConfig config{
            default_test_size, 
            default_index_value, 
            default_verbose, 
            false, 
            utils::keys_folder, 
            utils::weights_folder, 
            utils::cifar10tPath,
            utils::cifar100tPath,
            utils::mnistPath,
            utils::predictions_file,
            utils::training_data_dir
        };

        bool has_custom_cifar10 = false;
        bool has_custom_cifar100 = false;
        bool has_custom_mnist = false;
        bool has_custom_images = false;

        for (int i = 1; i < argc; ++i) {
            const string arg = argv[i];
            if (arg == "--threads" && i + 1 < argc) {
                config.max_threads = atoi(argv[++i]);
            } else if (arg == "--test-size" && i + 1 < argc) {
                config.test_size = atoi(argv[++i]);
            } else if (arg == "--index-value" && i + 1 < argc) {
                config.index_value = atoi(argv[++i]);
            } else if (arg == "--verbose" && i + 1 < argc) {
                config.verbose = atoi(argv[++i]);
            } else if (arg == "--keys-folder" && i + 1 < argc) {
                config.keys_folder = argv[++i];
            } else if (arg.rfind("--keys-folder=", 0) == 0) {
                config.keys_folder = arg.substr(strlen("--keys-folder="));
            } else if (arg == "--weights-folder" && i + 1 < argc) {
                config.weights_folder = argv[++i];
                utils::has_custom_weights_folder = true;
            } else if (arg.rfind("--weights-folder=", 0) == 0) {
                config.weights_folder = arg.substr(strlen("--weights-folder="));
                utils::has_custom_weights_folder = true;
            } else if (arg == "--training-data-dir" && i + 1 < argc) {
                config.training_data_dir = argv[++i];
            } else if (arg.rfind("--training-data-dir=", 0) == 0) {
                config.training_data_dir = arg.substr(strlen("--training-data-dir="));
            } else if (arg == "--cifar10-path" && i + 1 < argc) {
                config.cifar10tPath = argv[++i];
                has_custom_cifar10 = true;
            } else if (arg.rfind("--cifar10-path=", 0) == 0) {
                config.cifar10tPath = arg.substr(strlen("--cifar10-path="));
                has_custom_cifar10 = true;
            } else if (arg == "--cifar100-path" && i + 1 < argc) {
                config.cifar100tPath = argv[++i];
                has_custom_cifar100 = true;
            } else if (arg.rfind("--cifar100-path=", 0) == 0) {
                config.cifar100tPath = arg.substr(strlen("--cifar100-path="));
                has_custom_cifar100 = true;
            } else if (arg == "--mnist-path" && i + 1 < argc) {
                config.mnistPath = argv[++i];
                has_custom_mnist = true;
            } else if (arg.rfind("--mnist-path=", 0) == 0) {
                config.mnistPath = arg.substr(strlen("--mnist-path="));
                has_custom_mnist = true;
            } else if ((arg == "--images-path" || arg == "--images-folder") && i + 1 < argc) {
                string p = argv[++i];
                config.cifar10tPath = p;
                config.cifar100tPath = p;
                config.mnistPath = p;
                has_custom_images = true;
            } else if (arg.rfind("--images-path=", 0) == 0) {
                string p = arg.substr(strlen("--images-path="));
                config.cifar10tPath = p;
                config.cifar100tPath = p;
                config.mnistPath = p;
                has_custom_images = true;
            } else if ((arg == "--prediction-file" || arg == "--predictions-file" || arg == "--output-file" || arg == "--output") && i + 1 < argc) {
                config.predictions_file = argv[++i];
            } else if (arg.rfind("--prediction-file=", 0) == 0) {
                config.predictions_file = arg.substr(strlen("--prediction-file="));
            } else if (arg.rfind("--predictions-file=", 0) == 0) {
                config.predictions_file = arg.substr(strlen("--predictions-file="));
            } else if (arg.rfind("--output-file=", 0) == 0) {
                config.predictions_file = arg.substr(strlen("--output-file="));
            } else if (arg.rfind("--output=", 0) == 0) {
                config.predictions_file = arg.substr(strlen("--output="));
            } else if (arg.rfind("--peak-memory=", 0) == 0) {
                string val = arg.substr(strlen("--peak-memory="));
                config.peak_memory = (val == "true" || val == "1");
            } else if (arg == "--peak-memory") {
                if (i + 1 < argc && string(argv[i + 1]).rfind("--", 0) != 0) {
                    string val = argv[++i];
                    config.peak_memory = (val == "true" || val == "1");
                } else {
                    config.peak_memory = true;
                }
            } else if (arg == "--load-context") {
                if (i + 1 < argc && string(argv[i + 1]).rfind("--", 0) != 0) {
                    string val = argv[++i];
                    config.load_context = (val == "true" || val == "1");
                } else {
                    config.load_context = true;
                }
            }
        }

        if (!config.weights_folder.empty() && config.weights_folder.back() != '/') {
            config.weights_folder += "/";
        }

        // Truncate/initialize predictions file
        if (!config.predictions_file.empty()) {
            std::ofstream ofs(config.predictions_file, std::ios::trunc);
        }

        verbose_level = config.verbose;
        measure_peak_memory = config.peak_memory;
        max_threads = config.max_threads;
        utils::weights_folder = config.weights_folder;
        utils::cifar10tPath = config.cifar10tPath;
        utils::cifar100tPath = config.cifar100tPath;
        utils::mnistPath = config.mnistPath;
        utils::predictions_file = config.predictions_file;
        utils::test_size = config.test_size;
        utils::index_value = config.index_value;
        utils::keys_folder = config.keys_folder;
        utils::training_data_dir = config.training_data_dir;

        // Resolve active dataset path
        string active_dataset_path = "";
        if (has_custom_images) {
            active_dataset_path = config.cifar10tPath;
        } else if (has_custom_mnist) {
            active_dataset_path = config.mnistPath;
        } else if (has_custom_cifar100) {
            active_dataset_path = config.cifar100tPath;
        } else if (has_custom_cifar10) {
            active_dataset_path = config.cifar10tPath;
        } else {
            string exec_name = argv[0];
            for (auto &c : exec_name) c = tolower(c);
            if (exec_name.find("lenet") != string::npos || exec_name.find("mnist") != string::npos) {
                active_dataset_path = config.mnistPath;
            } else if (exec_name.find("resnet34") != string::npos) {  // ResNet34 models (BResNet34*, ResNet34Diagonal) use CIFAR-100
                active_dataset_path = config.cifar100tPath;
            } else {
                active_dataset_path = config.cifar10tPath;
            }
        }
        utils::dataset_path = active_dataset_path;

        program_start_time = steady_clock::now();
        return config;
    }
  
}

#endif //FHEON_UTILS_H