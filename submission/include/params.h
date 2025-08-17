// Copyright (c) 2025 HomomorphicEncryption.org
// All rights reserved.
//
// This software is licensed under the terms of the Apache v2 License.
// See the LICENSE.md file for details.
//============================================================================

#ifndef PARAMS_H_
#define PARAMS_H_
// params.h - parameters and directory structure for the workload

#include <vector>
#include <string>
#include <stdexcept>
#include <filesystem>
namespace fs = std::filesystem;

// an enum for benchmark size
enum InstanceSize {
    TOY = 0,
    SMALL = 1,
    MEDIUM = 2,
    LARGE = 3
};
inline std::string instance_name(const InstanceSize size) {
    if (unsigned(size) > unsigned(InstanceSize::LARGE)) {
        return "unknown";
    }
    static const std::string names[] = {"toy", "small", "medium", "large"};
    return names[int(size)];
}

// Parameters that differ for different instance sizes
class InstanceParams {
    const InstanceSize size;
    // Add any parameters necessary
    fs::path rootdir; // root of the submission dir structure (see below)

public:
    // Constructor
    explicit InstanceParams(InstanceSize _size,
                            fs::path _rootdir = fs::current_path())
                            : size(_size), rootdir(_rootdir)
    {
        if (unsigned(_size) > unsigned(InstanceSize::LARGE)) {
            throw std::invalid_argument("Invalid instance size");
        }
    }

    // Getters for all the parameters. There are no setters, once
    // an object is constructed these parameters cannot be modified.
    const InstanceSize getSize() const { return size; }

    // The relevant directories where things are found
    fs::path rtdir() const  { return rootdir; }
    fs::path iodir() const  { return rootdir/"io"/instance_name(size); }
    fs::path pubkeydir() const { return iodir() / "public_keys"; }
    fs::path seckeydir() const { return iodir() / "secret_key"; }
    fs::path ctxtupdir() const { return iodir() / "ciphertexts_upload"; }
    fs::path ctxtdowndir() const { return iodir() / "ciphertexts_download"; }
    fs::path intermdir() const { return iodir() / "intermediate"; }
    fs::path datadir() const { 
        return rootdir/"datasets"/instance_name(size);
    }
};

#endif  // ifndef PARAMS_H_