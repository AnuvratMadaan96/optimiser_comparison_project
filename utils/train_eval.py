import torch
from torch.utils.data import DataLoader, Subset
from optimisers.optimiser_comparison import get_optimizer
import torch.nn.functional as F
import time

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    start_time = time.time()

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    avg_loss = total_loss / len(train_loader.dataset)
    accuracy = 100. * correct / len(train_loader.dataset)
    training_time = time.time() - start_time

    return avg_loss, accuracy, training_time

def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    avg_loss = total_loss / len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    return avg_loss, accuracy

# K-fold Cross Validation logic
def k_fold_cross_validation(model_class, optimizer_name, dataset, k=5, epochs=10, lr=0.001):
    kfold_results = []
    dataset = dataset.dataset
    length = len(dataset)
    indices = list(range(length))
    fold_size = length // k
    criterion = torch.nn.CrossEntropyLoss()

    device = 'cuda:0' if torch.cuda.is_available() else 'mps' if torch.mps else 'cpu'
    for fold in range(k):
        model = model_class().to(device)
        optimizer = get_optimizer(optimizer_name, model, lr)
        
        # Splitting dataset
        
        val_indices = indices[fold * fold_size : (fold + 1) * fold_size]
        train_indices = indices[:fold * fold_size] + indices[(fold + 1) * fold_size:]

        # Create subset for training and validation sets
        train_set = Subset(dataset, train_indices)
        val_set = Subset(dataset, val_indices)
        train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

        for epoch in range(epochs):
            train_loss, train_acc, train_time = train(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            kfold_results.append({'fold': fold, 'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc, 'training_time': train_time})

    return kfold_results