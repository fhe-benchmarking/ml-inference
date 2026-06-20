# Remote Submission Example

This directory contains a remote submission example for the benchmarking suite.

As part of the benchmarking process, all remote backends are expected to make their homomorphic encryption parameters explicit and reviewable, since these parameters directly affect security, performance, and comparability across submissions.
Ideally, such parameters should be reported automatically by the backend.

This doc includes a static description of the FHE parameters used by the remote-backend example, along with a brief description of the model architecture. While helpful for context, model architecture disclosure is not a requirement of the benchmarking process.

The parameters shown here are chosen for simplicity and reproducibility and are not intended to represent a security-hardened or final challenge submission.

---

## FHE parameters

The example uses a single set of homomorphic encryption parameters for all supported workload sizes:
```python
homomorphic_params = {
    "full_q_list_precision": (
        (61,),
        (61,),
    ),                    # modulus chain: two 61-bit moduli (~122 bits total)
    "n": 2 ** 10,         # polynomial degree (chosen for simplicity, should use 2**13 for 128-bit security)
    "err_std": 1,         # standard deviation of the encryption noise
    "sk_hw": 0,           # secret key distribution: uniform in {-1, 0, 1}
    "g_base_bits": 4,     # decomposition base in evaluation key
    "pt_scale": 2 ** 15,  # initial plaintext scaling factor
}
```

---

## Model weights

The model weights are stored in:

- `digits_recognizer.pth`

They correspond to a simple two-layer fully connected network and are loaded as a standard PyTorch state dict:

```python
model = torch.load(
    f"{Path(__file__).parent}/digits_recognizer.pth",
    weights_only=True,
    map_location="cpu",
)

l1_weight = model["fc1.weight"]
l1_bias   = model["fc1.bias"]
l2_weight = model["fc2.weight"]
l2_bias   = model["fc2.bias"]
```

---

## Model architecture

The remote homomorphic inference follows the pipeline below:

```python
hom_pipeline = SequentialHomOp(
    ClientReshape((BATCH_SIZE, 28 * 28,)),      # preprocess: flatten 28x28 input image to a 784-dimensional vector
    HomLinear(l1_weight.shape),                 # first linear layer
    HomSquare(),                                # square activation
    HomLinear(l2_weight.shape),                 # second linear layer
    ClientReshape((BATCH_SIZE, 10,)),           # postprocess: reshape output to 10-class vector
)
```

Note that `SequentialHomOp`, `HomLinear`, etc., are internal classes. They are shown here to document the structure of the computation, not as runnable public code.

---

## Packing strategy and ring dimension

This example uses different ciphertext packing strategies depending on the batch size, which affects the effective ciphertext and evaluation key sizes.

**BATCH_SIZE == 1:**

  The ring dimension corresponds to the *feature dimension*.

  - The input vector of size 784 is packed into 2 ciphertexts, each with 512 slots (zero-padded).
  - Inner products with the weights matrices are computed via slot summation using log n rotation keys.
  - The activation requires a single relinearization key.

\
**BATCH_SIZE > 1:**

  The ring dimension corresponds to the *batch dimension*.

  - An input of shape `(B, 784)` is represented by 784 ciphertexts, each packing up to 512 batch elements (remaining slots are zero-padded).
  - Summation is performed across ciphertexts (non-ring dimension), so no rotation keys are required.
  - The activation still requires a single relinearization key.

---

## Example execution logs

### `batch_size = 1`

```
python3 harness/run_submission.py --remote 0

13:27:10 [harness] 1: Harness: MNIST Test dataset generation completed (elapsed: 9.8773s)
13:27:15 [harness] 2.1: Communication: Get cryptographic context completed (elapsed: 5.0613s)
         [harness] Cryptographic Context size: 283.3K
13:27:17 [harness] 2.2: Client: Key Generation completed (elapsed: 2.5725s)
         [harness] Client: Public and evaluation keys size: 11.1M
13:27:28 [harness] 2.3: Communication: Upload evaluation key completed (elapsed: 10.6825s)
13:27:28 [harness] 3: Server: (Encrypted) model preprocessing completed (elapsed: 0.0002s)
13:27:32 [harness] 4: Harness: Input generation for MNIST completed (elapsed: 3.7074s)
13:27:32 [harness] 5: Client: Input preprocessing completed (elapsed: 0.0004s)
13:27:34 [harness] 6: Client: Input encryption completed (elapsed: 2.353s)
         [harness] Client: Encrypted input size: 64.1K
ct size: 0.1MB
apply_hom_pipeline timing: network;dur=1130, logic;dur=137, instance;dur=106, worker;dur=33
13:27:38 [harness] 7: Server: Encrypted ML Inference computation completed (elapsed: 4.1393s)
         [harness] Client: Encrypted results size: 32.1K
13:27:41 [harness] 8: Client: Result decryption completed (elapsed: 2.5242s)
13:27:41 [harness] 9: Client: Result postprocessing completed (elapsed: 0.0005s)
[harness] PASS  (expected=5, got=5)
[total latency] 40.9184s
```
- Public + evaluation keys size: \~11.1 MB
- Encrypted input size: \~64 KB
- Total inference latency: 4.1393
- Compute inference latency: 33 ms
-----

### `batch_size = 15`

```
python3 harness/run_submission.py --remote 1

13:29:38 [harness] 1: Harness: MNIST Test dataset generation completed (elapsed: 8.9621s)
13:29:44 [harness] 2.1: Communication: Get cryptographic context completed (elapsed: 5.822s)
         [harness] Cryptographic Context size: 211.1K
13:29:47 [harness] 2.2: Client: Key Generation completed (elapsed: 2.7623s)
         [harness] Client: Public and evaluation keys size: 2.1M
13:29:57 [harness] 2.3: Communication: Upload evaluation key completed (elapsed: 10.491s)
13:29:57 [harness] 3: Server: (Encrypted) model preprocessing completed (elapsed: 0.0004s)
13:30:02 [harness] 4: Harness: Input generation for MNIST completed (elapsed: 5.0357s)
13:30:02 [harness] 5: Client: Input preprocessing completed (elapsed: 0.0006s)
13:30:05 [harness] 6: Client: Input encryption completed (elapsed: 3.0495s)
         [harness] Client: Encrypted input size: 24.5M
ct size: 24.5MB
apply_hom_pipeline timing: network;dur=13920, logic;dur=224, instance;dur=153, worker;dur=68
13:30:22 [harness] 7: Server: Encrypted ML Inference computation completed (elapsed: 16.2854s)
         [harness] Client: Encrypted results size: 320.1K
13:30:24 [harness] 8: Client: Result decryption completed (elapsed: 2.5512s)
13:30:24 [harness] 9: Client: Result postprocessing completed (elapsed: 0.0003s)
13:30:28 [harness] 10.1: Harness: Run inference for harness plaintext model completed (elapsed: 3.856s)
[harness] Encrypted model: 0.8667 (13/15 correct)
[harness] Harness model: 0.8667 (13/15 correct)
13:30:28 [harness] 10.2: Harness: Run quality check completed (elapsed: 0.0024s)
[total latency] 58.8189s
```
- Public + evaluation keys size: \~2.1 MB
- Encrypted input size: \~24.5 MB
- Total inference latency: 16.2854
- Compute inference latency: 68 ms
-----

### `batch_size = 1000`

```
python3 harness/run_submission.py --remote 2

13:32:07 [harness] 1: Harness: MNIST Test dataset generation completed (elapsed: 9.0411s)
13:32:12 [harness] 2.1: Communication: Get cryptographic context completed (elapsed: 5.5564s)
         [harness] Cryptographic Context size: 211.1K
13:32:15 [harness] 2.2: Client: Key Generation completed (elapsed: 2.5813s)
         [harness] Client: Public and evaluation keys size: 2.1M
13:32:26 [harness] 2.3: Communication: Upload evaluation key completed (elapsed: 10.6807s)
13:32:26 [harness] 3: Server: (Encrypted) model preprocessing completed (elapsed: 0.0004s)
13:32:30 [harness] 4: Harness: Input generation for MNIST completed (elapsed: 4.6718s)
13:32:30 [harness] 5: Client: Input preprocessing completed (elapsed: 0.0005s)
13:32:35 [harness] 6: Client: Input encryption completed (elapsed: 4.1143s)
         [harness] Client: Encrypted input size: 49.0M
ct size: 49.0MB
apply_hom_pipeline timing: network;dur=16404, logic;dur=439, instance;dur=381, worker;dur=142
13:32:54 [harness] 7: Server: Encrypted ML Inference computation completed (elapsed: 19.0744s)
         [harness] Client: Encrypted results size: 640.1K
13:32:56 [harness] 8: Client: Result decryption completed (elapsed: 2.6214s)
13:32:56 [harness] 9: Client: Result postprocessing completed (elapsed: 0.0002s)
13:33:00 [harness] 10.1: Harness: Run inference for harness plaintext model completed (elapsed: 4.2081s)
[harness] Encrypted model: 0.8820 (882/1000 correct)
[harness] Harness model: 0.8820 (882/1000 correct)
13:33:00 [harness] 10.2: Harness: Run quality check completed (elapsed: 0.0029s)
[total latency] 62.5536s
```
- Public + evaluation keys size: \~2.1 MB
- Encrypted input size: \~49.0 MB
- Total inference latency: 19.0744s
- Compute inference latency: 142 ms

