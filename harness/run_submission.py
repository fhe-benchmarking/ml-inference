#!/usr/bin/env python3
"""
run_submission.py - run the entire submission process, from build to verify
"""
# Copyright 2025 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import subprocess
import sys
import numpy as np
import utils
from params import instance_name

def main():
    """
    Run the entire submission process, from build to verify
    """
    
    # 0. Prepare running
    # Get the arguments
    size, params, seed, num_runs, clrtxt = utils.parse_submission_arguments('Run ML Inference FHE benchmark.')
    test = instance_name(size)
    print(f"\n[harness] Running submission for {test} inference")

    # Ensure the required directories exist
    utils.ensure_directories(params.rootdir)

    # Build the submission if not built already
    utils.build_submission(params.rootdir/"scripts")

    # The harness scripts are in the 'harness' directory,
    # the executables are in the directory submission/build
    harness_dir = params.rootdir/"harness"
    exec_dir = params.rootdir/"submission"/"build"

    # Remove and re-create IO directory
    io_dir = params.iodir()
    if io_dir.exists():
        subprocess.run(["rm", "-rf", str(io_dir)], check=True)
    io_dir.mkdir(parents=True)
    utils.log_step(0, "Init", True)

    # 1. Client-side: Generate the test datasets
    dataset_path = params.datadir() / f"dataset.txt"
    cmd = ["python3", harness_dir/"generate_dataset.py", str(dataset_path)]
    subprocess.run(cmd, check=True)
    utils.log_step(1, "Harness: MNIST Test dataset generation")

    # 2. Client-side: Generate the cryptographic keys 
    # Note: this does not use the rng seed above, it lets the implementation
    #   handle its own prg needs. It means that even if called with the same
    #   seed multiple times, the keys and ciphertexts will still be different.
    subprocess.run([exec_dir/"client_key_generation", str(size)], check=True)
    utils.log_step(2 , "Client: Key Generation")
    # Report size of keys and encrypted data
    utils.log_size(io_dir / "public_keys", "Client: Public and evaluation keys")

    # 3. Server-side: Preprocess the (encrypted) dataset using exec_dir/server_preprocess_model
    subprocess.run(exec_dir/"server_preprocess_model", check=True)
    utils.log_step(3, "Server: (Encrypted) model preprocessing")

    # Run steps 4-10 multiple times if requested
    for run in range(num_runs):
        run_path = params.measuredir() / f"results-{run+1}.json"
        if num_runs > 1:
            print(f"\n         [harness] Run {run+1} of {num_runs}")

        # 4. Client-side: Generate a new random input using harness/generate_input.py
        cmd = ["python3", harness_dir/"generate_input.py", str(size)]
        if seed is not None:
            # Use a different seed for each run but derived from the base seed
            rng = np.random.default_rng(seed)
            genqry_seed = rng.integers(0,0x7fffffff)
            cmd.extend(["--seed", str(genqry_seed)])
        subprocess.run(cmd, check=True)
        utils.log_step(4, "Harness: Input generation for MNIST")

        # 5. Client-side: Preprocess input using exec_dir/client_preprocess_input
        subprocess.run([exec_dir/"client_preprocess_input", str(size)], check=True)
        utils.log_step(5, "Client: Input preprocessing")

        # 6. Client-side: Encrypt the input
        subprocess.run([exec_dir/"client_encode_encrypt_input", str(size)], check=True)
        utils.log_step(6, "Client: Input encryption")
        utils.log_size(io_dir / "ciphertexts_upload", "Client: Encrypted input")

        # 7. Server side: Run the encrypted processing run exec_dir/server_encrypted_compute
        subprocess.run([exec_dir/"server_encrypted_compute", str(size)], check=True)
        utils.log_step(7, "Server: Encrypted ML Inference computation")
        # Report size of encrypted results
        utils.log_size(io_dir / "ciphertexts_download", "Client: Encrypted results")

        # 8. Client-side: decrypt
        subprocess.run([exec_dir/"client_decrypt_decode", str(size)], check=True)
        utils.log_step(8, "Client: Result decryption")

        # 9. Client-side: post-process
        subprocess.run([exec_dir/"client_postprocess", str(size)], check=True)
        utils.log_step(9, "Client: Result postprocessing")

        # 10 Verify the result for single inference or calculate quality for batch inference.
        encrypted_model_preds = params.get_encrypted_model_predictions_file()
        ground_truth_labels = params.get_ground_truth_labels_file()
        if not encrypted_model_preds.exists():
            print(f"Error: Result file {encrypted_model_preds} not found")
            sys.exit(1)

        if (size == utils.SINGLE):
            subprocess.run(["python3", harness_dir/"verify_result.py",
                str(ground_truth_labels), str(encrypted_model_preds)], check=False)
        else:
            # 10.1 Run the cleartext computation in cleartext_impl.py
            test_pixels = params.get_test_input_file()
            harness_model_preds = params.get_harness_model_predictions_file()
            subprocess.run(["python3", harness_dir/"cleartext_impl.py", str(test_pixels), str(harness_model_preds)], check=True)
            utils.log_step(10.1, "Harness: Run inference for harness plaintext model")

            # 10.2 Run the quality calculation
            utils.calculate_quality(ground_truth_labels, encrypted_model_preds, "Encrypted model")
            utils.calculate_quality(ground_truth_labels, harness_model_preds, "Harness model")
            utils.log_step(10.2, "Harness: Run quality check")

        # 11. Store measurements
        run_path.parent.mkdir(parents=True, exist_ok=True)
        utils.save_run(run_path, size)

    print(f"\nAll steps completed for the {instance_name(size)} inference!")

if __name__ == "__main__":
    main()
