from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import torch
from models.feedforward_nn import FeedForwardNN
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Tunable hyperparameters for the model
TUNABLE_HYPERPARAMETERS = {
    "learning_rate": {
        "RMSprop": [0.001, 0.0001],
        "Adam": [0.001, 0.0001],
        "AdamW": [0.001, 0.0001],
    },
    "weight_initialization": ['xavier_uniform', 'he_uniform'],
    "momentum": {
        "RMSprop": [0.9, 0.95, 0.99],
        "Adam": [0.9, 0.95, 0.99],
        "AdamW": [0.9, 0.95, 0.99],
    },
    "beta1": {
        "Adam": [0.9, 0.95, 0.99],
        "AdamW": [0.9, 0.95, 0.99],
    },
    "beta2": {
        "Adam": [0.999, 0.9995, 0.9999],
        "AdamW": [0.999, 0.9995, 0.9999],
    },
    "epsilon": {
        "RMSprop": [1e-8, 1e-7, 1e-6],
        "Adam": [1e-8, 1e-7, 1e-6],
        "AdamW": [1e-8, 1e-7, 1e-6],
    },
    "weight_decay": {
        "AdamW": [1e-5, 1e-4, 1e-3],
    },
}

# Dataset loading
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# Function to initialize weights
def initialize_weights(m, init_type):
    if isinstance(m, nn.Linear):
        if init_type == 'xavier_uniform':
            nn.init.xavier_uniform_(m.weight)
        elif init_type == 'he_uniform':
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

# Training function
def train_model(model, train_loader, optimizer, criterion, epochs=2):
    model.train()
    total_loss = 0.0
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        total_loss += running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader)}")

    return total_loss / epochs

# Grid Search implementation
def perform_grid_search():
    # Prepare to store the best parameters
    best_params = {
        'RMSprop': None,
        'Adam': None,
        'AdamW': None
    }
    best_loss = {
        'RMSprop': float('inf'),
        'Adam': float('inf'),
        'AdamW': float('inf')
    }

    # Iterate over optimizers
    for optimizer_name in ['RMSprop']:
        print(f"Performing Grid Search for {optimizer_name}")
        for lr in TUNABLE_HYPERPARAMETERS['learning_rate'][optimizer_name]:
            for init_type in TUNABLE_HYPERPARAMETERS['weight_initialization']:
                model = FeedForwardNN()
                model.apply(lambda m: initialize_weights(m, init_type))  # Apply weight initialization

                if optimizer_name == 'RMSprop':
                    optimizer = optim.RMSprop(model.parameters(), lr=lr, momentum=0.9)
                elif optimizer_name == 'Adam':
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                elif optimizer_name == 'AdamW':
                    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

                criterion = nn.CrossEntropyLoss()
                loss = train_model(model, train_loader, optimizer, criterion)

                # Update best parameters if the current configuration performs better
                if loss < best_loss[optimizer_name]:
                    best_loss[optimizer_name] = loss
                    best_params[optimizer_name] = {'learning_rate': lr, 'weight_initialization': init_type}
                    print("best parameter ",best_params)

                    print("&&&&&&&&&&&")
    return best_params

# Random Search implementation
def perform_random_search():
    best_params = {
        'RMSprop': None,
        'Adam': None,
        'AdamW': None
    }
    best_loss = {
        'RMSprop': float('inf'),
        'Adam': float('inf'),
        'AdamW': float('inf')
    }

    param_dist = {
        'learning_rate': [0.001, 0.0001],
        'weight_initialization': ['xavier_uniform', 'he_uniform'],
        'momentum': [0.9, 0.95, 0.99]
    }

    # Iterate over optimizers
    for optimizer_name in ['RMSprop']:
        print(f"Performing Random Search for {optimizer_name}")
        for i in range(3):  # Randomly sample 3 configurations
            lr = np.random.choice(param_dist['learning_rate'])
            init_type = np.random.choice(param_dist['weight_initialization'])
            model = FeedForwardNN()
            model.apply(lambda m: initialize_weights(m, init_type))  # Apply weight initialization

            if optimizer_name == 'RMSprop':
                optimizer = optim.RMSprop(model.parameters(), lr=lr, momentum=0.9)
            elif optimizer_name == 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=lr)
            elif optimizer_name == 'AdamW':
                optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

            criterion = nn.CrossEntropyLoss()
            loss = train_model(model, train_loader, optimizer, criterion)

            # Update best parameters if the current configuration performs better
            if loss < best_loss[optimizer_name]:
                best_loss[optimizer_name] = loss
                best_params[optimizer_name] = {'learning_rate': lr, 'weight_initialization': init_type}

                print(best_params)

                print("**********")

    return best_params

if __name__ == "__main__":
    # Perform Grid Search
    grid_search_best_params = perform_grid_search()
    print(f"Best parameters (Grid Search): {grid_search_best_params}")
    
    # Perform Random Search
    random_search_best_params = perform_random_search()
    print(f"Best parameters (Random Search): {random_search_best_params}")
