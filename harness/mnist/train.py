import torch

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device, model_path):
    best_accuracy = 0.0
    for epoch in range(epochs):
        model.train() # Set model to training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad() # Zero the gradients
            output = model(data)  # Forward pass
            loss = criterion(output, target) # Calculate loss
            loss.backward()       # Backward pass
            optimizer.step()      # Update weights

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()

        train_accuracy = 100 * correct_train / total_train
        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%')

        # Validation phase
        model.eval() # Set model to evaluation mode
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        with torch.no_grad(): # Disable gradient calculation during validation
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total_val += target.size(0)
                correct_val += (predicted == target).sum().item()

        val_accuracy = 100 * correct_val / total_val
        val_loss /= len(val_loader)
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        # Save the model if it's the best so far
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), model_path)
            print(f'Model saved to {model_path} with validation accuracy: {best_accuracy:.2f}%')