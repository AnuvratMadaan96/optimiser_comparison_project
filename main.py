import matplotlib.pyplot as plt
import pandas as pd
from data.kminst import get_kmnist_data
from models.feedforward_nn import FeedForwardNN
from optimisers.optimiser_comparison import get_optimizer
from utils.train_eval import k_fold_cross_validation
import os

def plot_performance(results):
    # Convert list of results to DataFrame for easier plotting
    df = pd.DataFrame(results)

    # Plot training accuracy
    plt.figure(figsize=(10, 6))
    for optimizer in df['optimizer'].unique():
        subset = df[df['optimizer'] == optimizer]
        subset['epoch_count'] = str(subset['fold'])+str(subset['epoch'])
        plt.plot(subset['epoch_count'], subset['train_acc'], label=f'{optimizer} Train Accuracy')

    plt.title('Training Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot validation accuracy
    plt.figure(figsize=(10, 6))
    for optimizer in df['optimizer'].unique():
        subset = df[df['optimizer'] == optimizer]
        plt.plot(subset['epoch'], subset['val_acc'], label=f'{optimizer} Validation Accuracy')

    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot training loss
    plt.figure(figsize=(10, 6))
    for optimizer in df['optimizer'].unique():
        subset = df[df['optimizer'] == optimizer]
        plt.plot(subset['epoch'], subset['train_loss'], label=f'{optimizer} Train Loss')

    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot training time
    plt.figure(figsize=(10, 6))
    for optimizer in df['optimizer'].unique():
        subset = df[df['optimizer'] == optimizer]
        plt.plot(subset['epoch'], subset['training_time'], label=f'{optimizer} Train Time')

    plt.title('Training Time Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Time')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Load KMNIST dataset
    train_loader, test_loader = get_kmnist_data()

    # Compare optimizers
    optimizers = ['Adam', 'RMSprop', 'AdamW']
    results = []

    for optimizer_name in optimizers:
        print(f"Running K-Fold CV for optimizer: {optimizer_name}")
        kfold_results = k_fold_cross_validation(FeedForwardNN, optimizer_name, train_loader, k=5, epochs=10, lr=0.001)

        # Add optimizer name to results for plotting
        for result in kfold_results:
            result['optimizer'] = optimizer_name
        
        results.extend(kfold_results)

    # Save results to CSV
    df = pd.DataFrame(results)
    results_dir = './optimiser_comparison_project/results'

    # Check if directory exists, if not, create it
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    save_path = os.path.join(results_dir, 'results.csv')
    df.to_csv(save_path, index=False)
    # df.to_csv('results/results.csv', index=False)

    # Plot the performance
    plot_performance(results)

    print("Optimization comparison completed and performance graphs plotted!")