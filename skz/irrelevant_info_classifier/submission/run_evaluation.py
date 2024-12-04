import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
    auc,
)
import seaborn as sns
import torch
import torch.nn as nn
from training_from_cached_activations import load_probe
from plot_roc import load_activations_and_labels, plot_roc_curve_2


def load_metrics(metrics_path):
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    return metrics


def plot_confusion_matrix(cm, labels, layer_idx, output_dir):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title(f"Confusion Matrix - Layer {layer_idx}")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"confusion_matrix_layer_{layer_idx}.png")
    plt.savefig(plot_path)
    plt.close()


# def plot_roc_curve(fpr, tpr, roc_auc, layer_idx, output_dir):
#     plt.figure()
#     plt.plot(fpr, tpr, color='darkorange',
#              lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title(f'Receiver Operating Characteristic - Layer {layer_idx}')
#     plt.legend(loc="lower right")
#     plt.tight_layout()
#     plot_path = os.path.join(output_dir, f'roc_curve_layer_{layer_idx}.png')
#     plt.savefig(plot_path)
#     plt.close()


def plot_auc_vs_layer(layer_auc_dict, output_dir):
    layers = sorted(layer_auc_dict.keys())
    auc_scores = [layer_auc_dict[layer] for layer in layers]
    plt.figure()
    plt.plot(layers, auc_scores, marker="o", linestyle="-")
    plt.xlabel("Layer Index")
    plt.ylabel("ROC AUC")
    plt.title("ROC AUC vs Layer")
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "auc_vs_layer.png")
    plt.savefig(plot_path)
    plt.close()


def plot_val_acc_vs_layer(layer_acc_dict, output_dir):
    layers = sorted(layer_acc_dict.keys())
    val_acc = [layer_acc_dict[layer] for layer in layers]
    plt.figure()
    plt.plot(layers, val_acc, marker="o", linestyle="-", color="green")
    plt.xlabel("Layer Index")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Validation Accuracy vs Layer")
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "val_acc_vs_layer.png")
    plt.savefig(plot_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Probe Metrics and Generate Plots"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to the probe checkpoint dir",
    )
    parser.add_argument(
        "--activation_cache_dir",
        type=str,
        required=True,
        help="Path to the activation cache directory containing activations and labels.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the evaluation plots. Defaults to the checkpoint directory.",
    )
    parser.add_argument(
        "--threshold_logit",
        type=float,
        default=0.0,
        help="Threshold for logit values to determine positive",
    )
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir
    activation_cache_dir = args.activation_cache_dir
    output_dir = (
        args.output_dir if args.output_dir else checkpoint_dir
    )
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize dictionaries to hold summary metrics
    layer_auc = {}
    layer_val_acc = {}

    layer_indices = list(range(1, 32)) 

    label_to_index = {"negative" : 0, "positive": 1}

    name = os.path.basename(os.path.normpath(checkpoint_dir))
    pooling_type = name.split("_")[2]

    for layer_idx in layer_indices:
        # Load the probe model and its hyperparameters
        checkpoint_path = os.path.join(checkpoint_dir, f"llama_linear_{pooling_type}_probe_layer_{layer_idx}_lr0.0001_epochs200_final.pt")
        print(f"Loading probe from {checkpoint_path}")
        probe, hyperparams = load_probe(checkpoint_path, device)

        probe.to(device)
        probe.eval()

        # Load and evaluate for validation set
        val_activations, val_labels = load_activations_and_labels(
            activation_cache_dir, pooling_type, "val", layer_idx, device
        )

        # Load and evaluate for training set
        train_activations, train_labels = load_activations_and_labels(
            activation_cache_dir, pooling_type, "train", layer_idx, device
        )

        # combine val and train activations and labels
        activations = torch.cat([train_activations, val_activations])
        labels = torch.cat([train_labels, val_labels])

        # Compute predictions
        with torch.no_grad():
            val_probs = probe(activations).cpu().numpy()
            # val_pred = np.argmax(val_probs, axis=1)
            val_pred = np.where(val_probs[:, 1] > args.threshold_logit, 1, 0)
            val_true = labels.cpu().numpy()

        # Compute ROC metrics
        fpr, tpr, _ = roc_curve(val_true, val_probs[:, 1])
        roc_auc_score = auc(fpr, tpr)

        # Compute confusion matrix
        cm = confusion_matrix(val_true, val_pred)
        # Compute confusion matrix components
        tn, fp, fn, tp = cm.ravel()
        # Compute fpr and tpr at the threshold
        fpr_value = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr_value = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Compute precision and recall at max F1 score
        precision, recall, f1, _ = precision_recall_fscore_support(
            val_true, val_pred, average="binary"
        )
        # Assuming binary classification; adjust if multiclass

        # Update summary metrics
        layer_auc[layer_idx] = roc_auc_score
        # layer_val_acc[layer_idx] = (recall + precision) / 2  # Example metric

        # Plot ROC Curve
        # plot_roc_curve(fpr, tpr, roc_auc_score, layer_idx, output_dir)
        plot_roc_curve_2(
            val_true,
            val_probs,
            output_dir,
            layer_idx,
        )

        # Plot Confusion Matrix
        labels = list(label_to_index.keys())
        plot_confusion_matrix(cm, labels, layer_idx, output_dir)

        # Add code to output metrics for layer 31
        if layer_idx == 31:
            metrics = {
                'precision': precision,
                'recall': recall,
                'fpr': fpr_value,
                'tpr': tpr_value,
                'fp': fp,
                'tp': tp,
                'tn': tn,
                'fn': fn,
                'f1': f1,
            }
            # Write metrics to a text file
            with open(os.path.join(output_dir, 'layer_31_metrics.txt'), 'w') as f:
                for key, value in metrics.items():
                    f.write(f"{key}: {value}\n")

    # Plot AUC vs Layer
    if layer_auc:
        plot_auc_vs_layer(layer_auc, output_dir)

    # Plot Validation Accuracy vs Layer
    # if layer_val_acc:
    #     plot_val_acc_vs_layer(layer_val_acc, output_dir)

    print(f"Evaluation plots have been saved to {output_dir}")


if __name__ == "__main__":
    main()
