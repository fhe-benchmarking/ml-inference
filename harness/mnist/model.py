import torch
import torch.nn as nn


class SimpleFFNN(nn.Module):

  def __init__(self):
    super(SimpleFFNN, self).__init__()
    # MNIST images are 28x28 = 784 pixels
    self.fc1 = nn.Linear(28 * 28, 128)  # First hidden layer
    self.relu = nn.ReLU()  # Activation function
    self.fc2 = nn.Linear(128, 64)  # Second hidden layer
    self.fc3 = nn.Linear(64, 10)  # Output layer (10 classes for digits 0-9)

  def forward(self, x):
    x = x.view(
        -1, 28 * 28
    )  # Flatten the 28x28 image into a 784-dimensional vector
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    x = self.relu(x)
    x = self.fc3(x)
    return x  # No softmax here, as CrossEntropyLoss will apply it internally
