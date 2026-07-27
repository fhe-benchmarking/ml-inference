
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
 * @file utils
 * @brief Plain secure utility functions used as general helpers throughout the project.
 *
 * This file defines various helper functions for secure computations and 
 * homomorphic operations, providing reusable utilities across the project.
 */

#ifndef FHEON_UTILS_H
#define FHEON_UTILS_H

#include <iostream>
#include <openfhe.h>

using namespace std;
using namespace std::chrono;
using namespace lbcrypto;
using Ctext = Ciphertext<DCRTPoly>;

namespace utils {

    static duration<long long, ratio<1, 1000>> total_time;

    /**
     * @brief Print the welcome message with project and author details.
     */
    static inline void printWelcomeMessage(){
        cout<< "----------------------------------------------------------------------------------- " << endl;
        cout<< "-------------------------------- WELCOME TO FHEON --------------------------------- " << endl;
        cout<< "------------------ Nges Brian, Eric Jahns, Michel A. Kinsy ------------------------ " << endl;
        cout<< "---------- Secure, Trusted and Assured Microelectronics (STAM) CENTER ------------- " << endl;
        cout<< "---------------------------- Arizona State University ----------------------------- " << endl; 
        cout << endl;
    }

   /**
     * @brief Get the current time point for timing measurements.
     *
     * @return Current time point.
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

        cout<< endl;
        if (mins.count() < 1) {
            cout << "------- " << caption << ": " << secs.count() << ":" << ms.count() << "s" << " (Total: " << duration_cast<seconds>(total_duration).count() << "s)" << " -------- " << endl;
        } else {
            cout << "-------- " << caption << ": " << mins.count() << "." << secs.count() << ":" << ms.count() << " (Total: " << duration_cast<minutes>(total_duration).count() << "mins)" << " -------- " << endl;
        }
        cout<< endl;
    }

    /**
     * @brief Print bootstrapping metadata for a ciphertext.
     *
     * @param ciphertextIn Input ciphertext.
     * @param depth Maximum depth available.
     */
    static inline void printBootsrappingData(Ctext ciphertextIn, int depth){
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
        return duration.count();
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
        cout << "------- Circuit Total Time: " << total << endl;
        return total;
    }
  
}

#endif //FHEON_UTILS_H