# FHE Benchmarking template

## Cloning the workload

```console
git clone https://github.com/andreea-alexandru/fhe-bench
cd fhe-bench
```

## Dependencies

The harness requires python and some corresponding packages specified in `requirements.txt`. 
```console
python3 -m venv virtualenv
source ./virtualenv/bin/activate
pip3 install -r requirements.txt
```

In this template, the library for the encrypted computations is OpenFHE v1.3.1. 
If other libraries are required, the developer should include the relevant instructions for installing.

This template assumes the correct version of OpenFHE is installed locally at `/third-party/openfhe`. 

### Installing OpenFHE

The installation steps for OpenFHE are described [here](https://openfhe-development.readthedocs.io/en/latest/sphinx_rsts/intro/installation/installation.html).  

If OpenFHE
is installed at a different location, that location should be specified using the `-CMAKE_PREFIX_PATH` variable in `build_task.sh`.
(In the case of a system-wide installation at `/usr/local/`, unset the `-CMAKE_PREFIX_PATH` variable.)

For users who want to do a local fresh install, they should run `get_openfhe.sh`, which 
is designed to install the specified version of OpenFHE at the `third-party` subdirectory in the current directory.
By default, `build_task.sh` looks for the library at this location. See more instructions in `submission/CMakeLists.txt` if 
`build_task.sh` does not succeed.

```console
./scripts/get_openfhe.sh
```

## Running the "add two numbers" workload

An example run is provided below.

```console
$ python3 harness/run_submission.py -h
usage: run_submission.py [-h] [--num_runs NUM_RUNS] [--seed SEED] [--clrtxt CLRTXT] {0,1,2,3}

Run the add-two-values FHE benchmark.

positional arguments:
  {0,1,2,3}            Instance size (0-toy/1-small/2-medium/3-large)

options:
  -h, --help           show this help message and exit
  --num_runs NUM_RUNS  Number of times to run steps 4-9 (default: 1)
  --seed SEED          Random seed for dataset and query generation
  --clrtxt CLRTXT      Specify with 1 if to rerun the cleartext computation

$ python3 ./harness/run_submission.py 2 --seed 3 --num_runs 2

[harness] Running submission for medium dataset
-- The CXX compiler identification is GNU 13.1.0
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- FOUND PACKAGE OpenFHE
-- OpenFHE Version: 1.3.1
-- OpenFHE installed as shared libraries: ON
-- OpenFHE include files location: /home/aandr/fhe-bench/third_party/openfhe/include/openfhe
-- OpenFHE lib files location: /home/aandr/fhe-bench/third_party/openfhe/lib
-- OpenFHE Native Backend size: 64
-- Configuring done
-- Generating done
-- Build files have been written to: /home/aandr/fhe-bench/submission/build
[  5%] Building CXX object CMakeFiles/client_postprocess.dir/src/client_postprocess.cpp.o
[...]
[100%] Built target client_decrypt_decode
15:58:45 [harness] 1: Dataset generation completed (elapsed: 0.4333s)
15:58:45 [harness] 2: Dataset preprocessing completed (elapsed: 0.1514s)
15:58:45 [harness] 3: Key Generation completed (elapsed: 0.2111s)
15:58:45 [harness] 4: Dataset encoding and encryption completed (elapsed: 0.1522s)
         [harness] Public and evaluation keys size: 517.8K
         [harness] Encrypted database size: 261.2K
15:58:46 [harness] 5: (Encrypted) dataset preprocessing completed (elapsed: 0.3696s)

         [harness] Run 1 of 2
15:58:46 [harness] 6: Query generation completed (elapsed: 0.3635s)
15:58:46 [harness] 7: Query preprocessing completed (elapsed: 0.017s)
15:58:46 [harness] 8: Query encryption completed (elapsed: 0.0955s)
         [harness] Encrypted query size: 257.2K
15:58:46 [harness] 9: Encrypted computation completed (elapsed: 0.0979s)
         [harness] Encrypted results size: 261.2K
15:58:47 [harness] 10: Result decryption completed (elapsed: 0.2023s)
15:58:47 [harness] 11: Result postprocessing completed (elapsed: 0.1582s)
         [harness] Wrote expected result to:  /home/aandr/fhe-bench/datasets/medium/expected.txt
[harness] PASS  (expected=13.89, got=13.889999999042985)
[total latency] 2.252s

         [harness] Run 2 of 2
15:58:47 [harness] 6: Query generation completed (elapsed: 0.7585s)
15:58:48 [harness] 7: Query preprocessing completed (elapsed: 0.071s)
15:58:48 [harness] 8: Query encryption completed (elapsed: 0.0339s)
         [harness] Encrypted query size: 257.2K
15:58:48 [harness] 9: Encrypted computation completed (elapsed: 0.121s)
         [harness] Encrypted results size: 261.2K
15:58:48 [harness] 10: Result decryption completed (elapsed: 0.1141s)
15:58:48 [harness] 11: Result postprocessing completed (elapsed: 0.1326s)
         [harness] Wrote expected result to:  /home/aandr/fhe-bench/datasets/medium/expected.txt
[harness] PASS  (expected=123.58, got=123.58000000003665)
[total latency] 2.5488s

All steps completed for the medium dataset!
```

After finishing the run, deactivate the virtual environment.
```console
deactivate
```

## Directory structure

```bash
Each submission to the workload in the FHE benchmarking should have the following directory structure:
[root] /
| ├─datasets/       # Holds cleartext data 
| |  ├─ toy/        # each instance-size in in a separate subdirectory
| |  ├─ small/
| |  ├─ medium/
| |  ├─ large/
| ├─docs/           # Documentation (beyond the top-level README.md)
| ├─harness/        # Scripts to generate data, run workload, check results
| ├─build/          # Handle installing dependencies and building the project
| ├─submission/     # The implementation, this is what the submitters modify
| |  └─ README.md   # likely also a src/, include/ subdirectories, CMakeLists.txt, etc.
| ├─io/             # Directory to hold the I/O between client & server parts
| |  ├─ toy/        # The reference implementation has subdirectories
| |     ├─ public_keys/             # holds the public evaluation keys
| |     ├─ ciphertexts_download/    # holds the ciphertexts to be downloaded by the client
| |     ├─ ciphertexts_upload/      # holds the ciphertexts (or other data except keys) to be uploaded by the client    
| |     ├─ intermediate/            # internal information to be passed around the functions
| |     └─ secret_key/              # holds the secret key
| |  ├─ small/
| |     …
| |  ├─ medium/
| |     …
| |  ├─ large/
| |     …
| ├─measurements/   # Holds json files with the results for each run
| |  ├─ toy/        # each instance-size in in a separate subdirectory
| |  ├─ small/
| |  ├─ medium/
| |  ├─ large/
```

## Description of stages

A submitter can edit any of the `client_*` / `server_*` sources in `/submission`. 
Moreover, for the particular parameters related to a workload, the submitter can modify the params files.
If the current description of the files are inaccurate, the stage names in `run_submission` can be also 
modified.

The current stages are the following, targeted to a client-server scenario.
The order in which they are happening in `run_submission` assumes an initialization step which is 
database-dependent and run only once, and potentially multiple runs for multiple queries.
Each file can take as argument the test case size.


| Stage executables                | Description |
|----------------------------------|-------------|
| `client_key_generation`          | Generate all key material and cryptographic context at the client.           
| `client_preprocess_dataset`      | (Optional) Any in the clear computations the client wants to apply over the dataset/model.
| `client_preprocess_query`        | (Optional) Any in the clear computations the client wants to apply over the query/input.
| `client_encode_encrypt_db`       | (Optional) Plaintext encoding and encryption of the dataset/model at the client.
| `client_encode_encrypt_query`    | Plaintext encoding and encryption of the query/input at the client.
| `server_preprocess_dataset`      | (Optional) Any in the clear or encrypted computations the server wants to apply over the dataset/model.
| `server_encrypted_compute`       | The computation the server applies to achieve the workload solution over encrypted daa.
| `client_decrypt_decode`          | Decryption and plaintext decoding of the result at the client.
| `client_postprocess`:            | Any in the clear computation that the client wants to apply on the decrypted result.


The outer python script measures the runtime of each stage.
The current stage separation structure requires reading and writing to files more times than minimally necessary.
For a more granular runtime measuring, which would account for the extra overhead described above, we encourage
submitters to separate and print in a log the individual times for reads/writes and computations inside each stage. 
