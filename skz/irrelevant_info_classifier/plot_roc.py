import torch
import argparse
import os
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

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
        "--cache_dir",
        type=str,
        required=True,
        help="Path to the activation cache directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val"],
        default="val",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save the ROC curve plot",
    )
    args = parser.parse_args()

    # Load the probe and hyperparameters
    probe, hyperparams = load_probe(args.checkpoint)
    layer_idx = hyperparams["layer_idx"]
    label_to_index = hyperparams["label_to_index"]
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    num_classes = hyperparams["num_classes"]

    # Load the cached activations and labels
    activations_path = os.path.join(args.cache_dir, f"{args.split}_activations.pt")
    labels_path = os.path.join(args.cache_dir, f"{args.split}_labels.pt")

    if not os.path.exists(activations_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"Cached activations or labels not found in {args.cache_dir}"
        )

    activations = torch.load(activations_path)
    labels = torch.load(labels_path)

    # Get activations for the specified layer
    if layer_idx not in activations:
        raise ValueError(
            f"Layer {layer_idx} activations not found in cached activations."
        )
    layer_activations = activations[layer_idx]
    dataset_labels = labels

    # Create ActivationDataset
    activation_dataset = ActivationDataset(
        {layer_idx: layer_activations}, dataset_labels
    )

    # Create DataLoader
    data_loader = DataLoader(
        activation_dataset, batch_size=args.batch_size, shuffle=False
    )

    # Move probe to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe.to(device)

    # Evaluate the probe on the activations
    probe.eval()
    all_predictions = []
    all_true_labels = []
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            activations = batch[f"layer_{layer_idx}"].to(device)
            labels = batch["label"].to(device)
            logits = probe(activations)
            probs = torch.softmax(logits, dim=1)
            predicted_indices = torch.argmax(logits, dim=1)
            all_predictions.extend(predicted_indices.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Map indices to labels
    predicted_labels = [index_to_label[idx] for idx in all_predictions]
    true_labels = [index_to_label[idx] for idx in all_true_labels]

    # Print results
    for i, (pred_label, true_label) in enumerate(zip(predicted_labels, true_labels)):
        print(f"Sample {i}: True Label: {true_label}, Predicted Label: {pred_label}")

    # Compute and plot ROC curve
    plot_roc_curve(
        all_true_labels,
        all_probs,
        num_classes,
        index_to_label,
        args.output_dir,
        args.split,
        layer_idx,
    )


def plot_roc_curve(
    all_true_labels,
    all_probs,
    num_classes,
    index_to_label,
    output_dir,
    split_name,
    layer_idx,
):
    """
    Compute ROC curve and ROC area for each class and plot the multiclass ROC curve.
    """
    import numpy as np
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    # Binarize the labels
    all_true_labels = label_binarize(all_true_labels, classes=range(num_classes))
    all_probs = np.array(all_probs)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(all_true_labels[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(
        all_true_labels.ravel(), all_probs.ravel()
    )
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curve
    plt.figure()
    lw = 2

    # Plot ROC curve for each class
    for i in range(num_classes):
        plt.plot(
            fpr[i],
            tpr[i],
            lw=lw,
            label="Class {0} ({1}) ROC curve (area = {2:0.2f})"
            "".format(i, index_to_label[i], roc_auc[i]),
        )

    # Plot micro-average ROC curve
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="Micro-average ROC curve (area = {0:0.2f})" "".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(
        f"Receiver Operating Characteristic - Layer {layer_idx} - {split_name} Set"
    )
    plt.legend(loc="lower right")

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = f"roc_curve_layer_{layer_idx}_{split_name}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"ROC curve saved to {plot_path}")


if __name__ == "__main__":
    main()
