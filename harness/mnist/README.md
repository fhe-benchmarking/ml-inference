## Simple Plaintext Feed-Forward Model for MNIST Classification

The directory `harness/mnist` contains PyTorch code to train and test a basic feed-forward neural network for the MNIST digit classification task. The model is designed to be relatively small but achieves an accuracy above 95% on the test dataset.

### Getting Started

To run the code, you'll first need to install the necessary Python packages. It uses the `torch` and `torchvision` modules, which are listed in the `requirements.txt` file.

1. **Install dependencies:** It is recommended to use a virtual environment to manage project dependencies.

```
pip install -r requirements.txt
```

2. **Run the script:**

```
python3 harness/mnist/mnist.py
```

**How it works**

The `mnist.py` script automatically handles model training and testing:

* **First Run:** When you run the script for the first time, it will train a new model on the MNIST dataset, save the trained weights to `mnist_ffnn_model.pth`, and evaluate it on the test data.

* **Subsequent Runs:** If `mnist_ffnn_model.pth` already exists in the directory, the script will skip the training phase and instead load the pre-trained model directly from the file.

**Example output**

```
❯ python3 harness/mnist/mnist.py
Using device: cuda

Model 'mnist_ffnn_model.pth' not found. Starting training...
Epoch 1/15, Loss: 0.2979, Train Accuracy: 91.04%
Validation Loss: 0.1664, Validation Accuracy: 94.78%
Model saved to mnist_ffnn_model.pth with validation accuracy: 94.78%
Epoch 2/15, Loss: 0.1268, Train Accuracy: 96.22%
Validation Loss: 0.1297, Validation Accuracy: 95.89%
Model saved to mnist_ffnn_model.pth with validation accuracy: 95.89%
Epoch 3/15, Loss: 0.0878, Train Accuracy: 97.31%
Validation Loss: 0.1125, Validation Accuracy: 96.52%
Model saved to mnist_ffnn_model.pth with validation accuracy: 96.52%
Epoch 4/15, Loss: 0.0665, Train Accuracy: 97.92%
Validation Loss: 0.1047, Validation Accuracy: 96.83%
Model saved to mnist_ffnn_model.pth with validation accuracy: 96.83%
Epoch 5/15, Loss: 0.0532, Train Accuracy: 98.26%
Validation Loss: 0.0988, Validation Accuracy: 97.11%
Model saved to mnist_ffnn_model.pth with validation accuracy: 97.11%
Epoch 6/15, Loss: 0.0437, Train Accuracy: 98.49%
Validation Loss: 0.1149, Validation Accuracy: 96.71%
Epoch 7/15, Loss: 0.0352, Train Accuracy: 98.81%
Validation Loss: 0.1128, Validation Accuracy: 96.91%
Epoch 8/15, Loss: 0.0312, Train Accuracy: 98.95%
Validation Loss: 0.1089, Validation Accuracy: 97.12%
Model saved to mnist_ffnn_model.pth with validation accuracy: 97.12%
Epoch 9/15, Loss: 0.0274, Train Accuracy: 99.06%
Validation Loss: 0.1038, Validation Accuracy: 97.26%
Model saved to mnist_ffnn_model.pth with validation accuracy: 97.26%
Epoch 10/15, Loss: 0.0238, Train Accuracy: 99.17%
Validation Loss: 0.1308, Validation Accuracy: 96.75%
Epoch 11/15, Loss: 0.0232, Train Accuracy: 99.20%
Validation Loss: 0.1140, Validation Accuracy: 97.58%
Model saved to mnist_ffnn_model.pth with validation accuracy: 97.58%
Epoch 12/15, Loss: 0.0207, Train Accuracy: 99.29%
Validation Loss: 0.1451, Validation Accuracy: 96.78%
Epoch 13/15, Loss: 0.0157, Train Accuracy: 99.47%
Validation Loss: 0.1111, Validation Accuracy: 97.57%
Epoch 14/15, Loss: 0.0173, Train Accuracy: 99.43%
Validation Loss: 0.1129, Validation Accuracy: 97.52%
Epoch 15/15, Loss: 0.0171, Train Accuracy: 99.42%
Validation Loss: 0.1318, Validation Accuracy: 97.10%
Training finished.

Evaluating model on test data...
Accuracy on test data: 97.37%
```

**Rebuilding the Model**

If you wish to re-train the model from scratch (e.g., after making changes to the network architecture or training parameters), simply delete the `mnist_ffnn_model.pth` file and run the script again.
