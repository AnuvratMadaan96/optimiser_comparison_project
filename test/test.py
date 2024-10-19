import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.feedforward_nn import FeedForwardNN 
from data.kminst import get_kmnist_data

def test_model(model_path):
    # Load the trained model
    model = torch.load(model_path) # Load the entire model
    model.eval()  # Set model to evaluation mode

    # Load the KMNIST test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Same normalization as training
    ])
    
    _, test_loader = get_kmnist_data()  
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation for testing
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total * 100
    return accuracy

if __name__ == "__main__":

    optimizers = ['RMSprop', 'Adam', 'AdamW']
    for optimizer in optimizers:
        model_path = f'optimiser_comparison_project/models/model_{optimizer}.pth'
        print(f'Accuracy on the test set for {optimizer}: {test_model(model_path):.2f}%')
