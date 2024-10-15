import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_kmnist_data(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_data = datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.KMNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader