import torch
import argparse
import os
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import re

# Import necessary classes and functions from your existing script
from training_from_cached_activations import (
    LinearProbe,
    ComplexProbe,
    load_probe,
    ActivationDataset,
    SimpleDataset,
)


def main():
    parser = argparse.ArgumentParser(
        description="Load and use a probe with cached activations and plot ROC curve"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the probe checkpoint (.pt file)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save the ROC curve plot. Defaults to the checkpoint directory",
    )
    args = parser.parse_args()


    if "llama" in args.checkpoint:
        cache_dir = "/home/paperspace/src/MathLM/skz/irrelevant_info_classifier/activation_cache/llama"
    else:
        cache_dir = "/home/paperspace/src/MathLM/skz/irrelevant_info_classifier/activation_cache/openmath"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the probe and hyperparameters
    probe, hyperparams = load_probe(args.checkpoint, device)
    
    match = re.search(r'layer_(\d+)', args.checkpoint)
    if match:
        layer_idx = int(match.group(1))
    else:
        raise ValueError("Layer index not found in checkpoint filename.")
    # layer_idx = hyperparams["layer_idx"]
    label_to_index = {"negative" : 0, "positive": 1}
    # label_to_index = hyperparams["label_to_index"]
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    num_classes = hyperparams["num_classes"]


    # Load and evaluate for validation set
    val_activations, val_labels = load_activations_and_labels(cache_dir, "val", layer_idx, device)
    val_probs, val_true_labels = evaluate_probe(probe, val_activations, val_labels, layer_idx, args.batch_size, device)

    # Load and evaluate for training set
    train_activations, train_labels = load_activations_and_labels(cache_dir, "train", layer_idx, device)
    train_probs, train_true_labels = evaluate_probe(probe, train_activations, train_labels, layer_idx, args.batch_size, device)

    # Determine output directory
    output_dir = args.output_dir if args.output_dir else os.path.dirname(args.checkpoint)

    # Compute and plot ROC curve
    plot_roc_curve(
        val_true_labels,
        val_probs,
        train_true_labels,
        train_probs,
        output_dir,
        layer_idx,
    )


def load_activations_and_labels(cache_dir, split, layer_idx, device):
    activations_path = os.path.join(cache_dir, f"{split}_activations.pt")
    labels_path = os.path.join(cache_dir, f"{split}_labels.pt")

    if not os.path.exists(activations_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"Cached activations or labels not found in {cache_dir}"
        )

    activations = torch.load(activations_path, map_location=device)
    labels = torch.load(labels_path, map_location=device)

    if layer_idx not in activations:
        raise ValueError(
            f"Layer {layer_idx} activations not found in cached activations."
        )
    return activations[layer_idx], labels


def evaluate_probe(probe, activations, labels, layer_idx, batch_size, device):
    probe.to(device)
    probe.eval()
    activation_dataset = ActivationDataset(
        {layer_idx: activations}, labels
    )
    data_loader = DataLoader(
        activation_dataset, batch_size=batch_size, shuffle=False
    )

    all_probs = []
    all_true_labels = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            activations = batch[f"layer_{layer_idx}"].to(device)
            # print(activations.shape)
            labels = batch["label"].to(device)
            logits = probe(activations)
            probs = torch.softmax(logits, dim=1)
            all_true_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return all_probs, all_true_labels


def plot_roc_curve(
    val_true_labels,
    val_probs,
    train_true_labels,
    train_probs,
    output_dir,
    layer_idx,
):
    import numpy as np
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix

    # Convert probabilities to NumPy arrays
    val_probs = np.array(val_probs)
    train_probs = np.array(train_probs)

    plt.figure()
    lw = 2

    # Binary classification
    # Compute Precision-Recall curve for validation set
    precision_val, recall_val, thresholds_val = precision_recall_curve(
        val_true_labels, val_probs[:, 1]
    )
    f1_scores_val = 2 * recall_val * precision_val / (recall_val + precision_val + 1e-8)
    max_f1_idx_val = np.argmax(f1_scores_val)
    max_f1_val = f1_scores_val[max_f1_idx_val]
    optimal_threshold_val = thresholds_val[max_f1_idx_val]

    # Compute FPR and TPR at the optimal threshold for validation set
    pred_labels_val = (val_probs[:, 1] >= optimal_threshold_val).astype(int)
    tn_val, fp_val, fn_val, tp_val = confusion_matrix(val_true_labels, pred_labels_val).ravel()
    fpr_val_max_f1 = fp_val / (fp_val + tn_val) if (fp_val + tn_val) > 0 else 0.0
    tpr_val_max_f1 = tp_val / (tp_val + fn_val) if (tp_val + fn_val) > 0 else 0.0

    # Compute Precision-Recall curve for training set
    precision_train, recall_train, thresholds_train = precision_recall_curve(
        train_true_labels, train_probs[:, 1]
    )
    f1_scores_train = 2 * recall_train * precision_train / (recall_train + precision_train + 1e-8)
    max_f1_idx_train = np.argmax(f1_scores_train)
    max_f1_train = f1_scores_train[max_f1_idx_train]
    optimal_threshold_train = thresholds_train[max_f1_idx_train]

    # Compute FPR and TPR at the optimal threshold for training set
    pred_labels_train = (train_probs[:, 1] >= optimal_threshold_train).astype(int)
    tn_train, fp_train, fn_train, tp_train = confusion_matrix(train_true_labels, pred_labels_train).ravel()
    fpr_train_max_f1 = fp_train / (fp_train + tn_train) if (fp_train + tn_train) > 0 else 0.0
    tpr_train_max_f1 = tp_train / (tp_train + fn_train) if (tp_train + fn_train) > 0 else 0.0

    # Compute ROC curve and AUC
    fpr_val, tpr_val, _ = roc_curve(val_true_labels, val_probs[:, 1])
    roc_auc_val = auc(fpr_val, tpr_val)

    fpr_train, tpr_train, _ = roc_curve(train_true_labels, train_probs[:, 1])
    roc_auc_train = auc(fpr_train, tpr_train)

    # Plot ROC curves
    plt.plot(
        fpr_val,
        tpr_val,
        lw=lw,
        label="Val ROC curve (area = {0:0.2f})".format(roc_auc_val),
        color='orange'
    )
    plt.plot(
        fpr_train,
        tpr_train,
        lw=lw,
        label="Train ROC curve (area = {0:0.2f})".format(roc_auc_train),
        color='blue'
    )

    # Plot the point with maximum F1 score for validation set
    plt.scatter(
        fpr_val_max_f1,
        tpr_val_max_f1,
        color='red',
        label=f'Val Max F1={max_f1_val:.2f} at threshold={optimal_threshold_val:.2f}'
    )

    # Plot the point with maximum F1 score for training set
    plt.scatter(
        fpr_train_max_f1,
        tpr_train_max_f1,
        color='green',
        label=f'Train Max F1={max_f1_train:.2f} at threshold={optimal_threshold_train:.2f}'
    )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver Operating Characteristic - Layer {layer_idx}")
    plt.legend(loc="lower right")

    os.makedirs(output_dir, exist_ok=True)
    plot_filename = f"roc_curve_layer_{layer_idx}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"ROC curve saved to {plot_path}")


def compute_roc(true_labels, probs):
    fpr, tpr, _ = roc_curve(true_labels, probs[:, 1])
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


if __name__ == "__main__":
    main()
