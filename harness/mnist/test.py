
import os
import torch
import model as simple_ffn

def test_model(model, test_loader, device):
    model.eval() # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad(): # Disable gradient calculation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy on test data: {accuracy:.2f}%')
    return accuracy


def predict(pixels_file, model_path="mnist_ffnn_model.pth", predictions_file='predictions.txt', device='cpu'):
    """
    Load a trained model and make predictions on pixel data from a file.
    
    Args:
        pixels_file (str): Path to file containing pixel data (one sample per line, 784 values per line)
        model_path (str): Path to the trained model file
        predictions_file (str): Path to save the predictions
        device (str): Device to run inference on ('cpu' or 'cuda')
    
    Returns:
        list: List of predicted class labels
    """
    # Set device
    device = torch.device(device)
    
    # Load the trained model
    model = simple_ffn.SimpleFFNN().to(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set model to evaluation mode
    
    # Read pixel data from file
    pixel_data = []
    with open(pixels_file, 'r') as f:
        for line in f:
            # Parse the line and convert to float values
            pixel_values = [float(x) for x in line.strip().split()]
            if len(pixel_values) != 784:
                raise ValueError(f"Each line must contain exactly 784 pixel values, got {len(pixel_values)}")
            pixel_data.append(pixel_values)
    
    # Convert to tensor and apply normalization (same as training)
    # The pixel values should already be normalized (0-1), but we need to apply MNIST normalization
    pixel_tensor = torch.tensor(pixel_data, dtype=torch.float32).to(device)
    # Apply MNIST normalization: (pixel - mean) / std
    pixel_tensor = (pixel_tensor - 0.1307) / 0.3081
    
    predictions = []
    
    # Make predictions
    with torch.no_grad():
        outputs = model(pixel_tensor)
        _, predicted = torch.max(outputs, 1)
        predictions = predicted.cpu().numpy().tolist()
    
    # Save predictions to file
    with open(predictions_file, 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    
    return predictions