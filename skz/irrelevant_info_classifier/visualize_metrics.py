# plot_metrics.py

import json
import os
import matplotlib.pyplot as plt
import argparse

def plot_metrics(metrics, output_dir):
    """Plot training and validation metrics for all layers."""
    os.makedirs(output_dir, exist_ok=True)
    layer_indices = list(metrics.keys())

    for layer_idx in layer_indices:
        layer_metrics = metrics[layer_idx]

        plt.figure(figsize=(10, 4))

        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(layer_metrics['train_loss'], label='Train Loss')
        plt.plot(layer_metrics['val_loss'], label='Val Loss')
        plt.title(f'Layer {layer_idx} Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(layer_metrics['train_acc'], label='Train Acc')
        plt.plot(layer_metrics['val_acc'], label='Val Acc')
        plt.title(f'Layer {layer_idx} Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()

        plt.tight_layout()

        # Save the plot
        plot_path = os.path.join(output_dir, f"layer_{layer_idx}_metrics.png")
        plt.savefig(plot_path)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot metrics from metrics.json')
    parser.add_argument('--metrics_path', type=str, required=True, help='Path to metrics.json')
    args = parser.parse_args()

    metrics_path = args.metrics_path
    output_dir = os.path.dirname(metrics_path)

    # Load the metrics from the JSON file
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # Plot the metrics
    plot_metrics(metrics, output_dir)

if __name__ == "__main__":
    main()
