// Copyright (c) 2025 HomomorphicEncryption.org
// All rights reserved.
//
// This software is licensed under the terms of the Apache v2 License.
// See the LICENSE.md file for details.
//============================================================================

#include <fstream>
#include <iostream>
#include "params.h"

using namespace std;

int main(int argc, char* argv[]){
    if (argc < 2 || !std::isdigit(argv[1][0])) {
        std::cout << "Usage: " << argv[0] << " instance-size [--count_only]\n";
        std::cout << "  Instance-size: 0-TOY, 1-SMALL, 2-MEDIUM, 3-LARGE\n";
        return 0;
    }
    auto size = static_cast<InstanceSize>(std::stoi(argv[1]));
    InstanceParams prms(size);

    std::string db_path = prms.datadir()/"db.txt";
    std::string interm_path = prms.intermdir()/"plain_db.bin";

    double d;
    std::ifstream db(db_path);
    if (!db.is_open())
        throw std::runtime_error("Cannot open " + db_path);
    if (!(db >> d))
        throw std::runtime_error("Failed to read double from " + db_path);

    fs::create_directories(prms.intermdir());
    std::ofstream out(interm_path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Could not open " + interm_path);
    }
    out.write((char*)&d, sizeof(double));
    out.close();

    return 0;
}