import matplotlib.pyplot as plt
import pandas as pd
from data.kminst import get_kmnist_data
from models.feedforward_nn import FeedForwardNN
from optimisers.optimiser_comparison import get_optimizer
from utils.train_eval import k_fold_cross_validation
from utils.hyperparameter_search import perform_random_search, perform_grid_search
import os

def plot_performance(results):
    # Convert list of results to DataFrame for easier plotting
    if type(results) != type(pd.DataFrame()):
        df = pd.DataFrame(results)
        print("Converting to Dataframe")
    else:
        df = results

    grouped_df = df.groupby(['optimizer', 'epoch']).mean().reset_index()
    
    results_dir = './optimiser_comparison_project/results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    save_path = os.path.join(results_dir, 'results_mean.csv')
    grouped_df.to_csv(save_path, index=False)

    # Create subplots for train_loss, val_loss, train_acc, val_acc, and training_time
    fig, axs = plt.subplots(5, 1, figsize=(15, 15))

    # Flatten the array of axes for easy indexing
    axs = axs.flatten()

    # Metrics to plot
    metrics = ['train_loss', 'val_loss', 'train_acc', 'val_acc', 'training_time']
    titles = ['Train Loss', 'Validation Loss', 'Train Accuracy', 'Validation Accuracy', 'Training Time']

    # Iterate over each metric and plot the optimizer performances
    for i, metric in enumerate(metrics):
        for optimizer in grouped_df['optimizer'].unique():
            optimizer_data = grouped_df[grouped_df['optimizer'] == optimizer]
            axs[i].plot(optimizer_data['epoch'], optimizer_data[metric], label=optimizer)

        axs[i].set_title(titles[i])
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel(metric.replace('_', ' ').capitalize())
        axs[i].legend()

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load KMNIST dataset
    train_loader, test_loader = get_kmnist_data()
    for images, labels in train_loader:
        print(images.shape, labels.shape)  # Prints the shape of the batch of images and labels
        break 
    optimizers = ['RMSprop', 'Adam', 'AdamW']

    # Perform Grid Search
    grid_search_best_params = perform_grid_search(train_loader, optimizers)
    print(f"Best parameters (Grid Search): {grid_search_best_params}")

    best_parameters = {
        'GridSearch': grid_search_best_params,
    }

    # Compare optimizers
    results = []

    for optimizer_name in optimizers:
        print(f"Running K-Fold CV for optimizer: {optimizer_name}")
        for search_name, parameters in best_parameters.items():
            kfold_results = k_fold_cross_validation(FeedForwardNN, optimizer_name, train_loader, k=5, epochs=15, lr=parameters[optimizer_name]['learning_rate'])

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

    result_df = pd.read_csv(save_path)
    # Plot the performance
    plot_performance(result_df)

    print("Optimization comparison completed and performance graphs plotted!")
    # results = pd.read_csv("results/results.csv")
    # # Plot the performance
    # plot_performance(results)
 