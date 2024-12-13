# llama_probing.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import json
import os
from typing import List, Dict
from transformers import LlamaModel
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from datasets import load_dataset
from tqdm import tqdm
from accelerate import Accelerator
from datetime import datetime
import itertools
import argparse  # For command-line argument parsing

class LlamaModelWrapper:
    def __init__(self, model_path: str):
        self.model_path = model_path

        # Use the fast tokenizer by setting use_fast=True
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

        # Handle the pad_token if it's not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = LlamaModel.from_pretrained(model_path)

        print(f"Loaded model from {model_path}")
        print(f"Model configuration: {self.model.config}")

    def get_layer_output(self, layer_idx: int, input_ids: torch.Tensor) -> torch.Tensor:
        """Get the output of a specific layer."""
        # Get hidden states from the model
        outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        # Hidden states are a tuple with (embeddings + layer outputs)
        # Layer indexing: hidden_states[0] is embeddings, hidden_states[1] is first layer, etc.
        return outputs.hidden_states[layer_idx + 1]  # +1 to account for embeddings

class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class ComplexProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super(ComplexProbe, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class SimpleDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: List[str],
        tokenizer: AutoTokenizer,
        max_length: int = 128,
        label_to_index: Dict[str, int] = None,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        if label_to_index is None:
            # Create a mapping from label strings to indices
            unique_labels = sorted(set(labels))
            self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.label_to_index = label_to_index

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]

        # Prepend the prompt to the text
        prompt = "The following is a math word problem: "
        full_text = prompt + text

        # Tokenize text
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Map the label string to an index
        label_idx = self.label_to_index[label]

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'label': torch.tensor(label_idx, dtype=torch.long)
        }

def train_probes(model: LlamaModelWrapper,
                 probes: List[nn.Module],
                 layer_indices: List[int],
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 num_epochs: int = 5,
                 learning_rate: float = 1e-3,
                 accelerator: Accelerator = None,
                 output_dir: str = 'output',
                 model_name: str = 'model',
                 probe_type: str = 'complex'):
    """Train probes for specified layers."""
    criterion = nn.CrossEntropyLoss()
    # Initialize separate optimizers for each probe
    optimizers = [AdamW(probe.parameters(), lr=learning_rate) for probe in probes]

    model.model, train_loader, val_loader = accelerator.prepare(
            model.model, train_loader, val_loader
        )
    probes = [accelerator.prepare(probe) for probe in probes]
    optimizers = [accelerator.prepare(optimizer) for optimizer in optimizers]

    # Metrics tracking
    metrics = {
        layer_idx: {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        } for layer_idx in layer_indices
    }

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Training
        for probe in probes:
            probe.train()

        train_losses = [0.0 for _ in layer_indices]
        train_correct = [0 for _ in layer_indices]
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Training", unit="batch")
        for batch in progress_bar:
            input_ids = batch['input_ids']  # Already on device
            labels = batch['label']          # Already on device

            # Get layer outputs
            with torch.no_grad():
                layer_outputs = [model.get_layer_output(idx, input_ids) for idx in layer_indices]

            # Extract the representation of the last token in each sequence
            layer_outputs = [output[:, -1, :] for output in layer_outputs]

            for i, (probe, optimizer) in enumerate(zip(probes, optimizers)):
                optimizer.zero_grad()
                outputs = probe(layer_outputs[i])
                loss = criterion(outputs, labels)
                # Use Accelerator's backward method
                accelerator.backward(loss)
                optimizer.step()

                train_losses[i] += loss.item()
                _, predicted = outputs.max(1)
                train_correct[i] += predicted.eq(labels).sum().item()

            total += labels.size(0)

        for i, layer_idx in enumerate(layer_indices):
            metrics[layer_idx]['train_loss'].append(train_losses[i] / len(train_loader))
            metrics[layer_idx]['train_acc'].append(100. * train_correct[i] / total)

        # Validation
        for probe in probes:
            probe.eval()

        val_losses = [0.0 for _ in layer_indices]
        val_correct = [0 for _ in layer_indices]
        val_total = 0

        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Validation", unit="batch")
            for batch in progress_bar:
                input_ids = batch['input_ids']  # Already on device
                labels = batch['label']          # Already on device

                layer_outputs = [model.get_layer_output(idx, input_ids) for idx in layer_indices]
                layer_outputs = [output[:, -1, :] for output in layer_outputs]

                for i, probe in enumerate(probes):
                    outputs = probe(layer_outputs[i])
                    loss = criterion(outputs, labels)
                    val_losses[i] += loss.item()
                    _, predicted = outputs.max(1)
                    val_correct[i] += predicted.eq(labels).sum().item()

                val_total += labels.size(0)

        for i, layer_idx in enumerate(layer_indices):
            metrics[layer_idx]['val_loss'].append(val_losses[i] / len(val_loader))
            metrics[layer_idx]['val_acc'].append(100. * val_correct[i] / val_total)

        # Print metrics for each layer
        for i, layer_idx in enumerate(layer_indices):
            print(f"\nLayer {layer_idx}:")
            print(f"Train Loss: {metrics[layer_idx]['train_loss'][-1]:.4f}")
            print(f"Train Acc: {metrics[layer_idx]['train_acc'][-1]:.2f}%")
            print(f"Val Loss: {metrics[layer_idx]['val_loss'][-1]:.4f}")
            print(f"Val Acc: {metrics[layer_idx]['val_acc'][-1]:.2f}%")
            print("\n" + "-"*50 + "\n")

        # Synchronize before saving
        accelerator.wait_for_everyone()

        # Save checkpoints and metrics at the end of each epoch
        if accelerator.is_main_process:
            # Save probes
            for i, probe in enumerate(probes):
                probe_filename = f"{model_name}_{probe_type}_probe_layer_{layer_indices[i]}_epoch_{epoch+1}.pt"
                probe_path = os.path.join(output_dir, probe_filename)
                accelerator.save(probe.state_dict(), probe_path)

            # Save metrics
            metrics_filename = f"{model_name}_{probe_type}_metrics.json"
            metrics_path = os.path.join(output_dir, metrics_filename)
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f)

    return metrics

def plot_metrics(metrics, layer_indices, output_dir, model_name, probe_type):
    """Plot training and validation metrics for each layer."""
    os.makedirs(output_dir, exist_ok=True)
    for layer_idx in layer_indices:
        plt.figure(figsize=(10, 4))

        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(metrics[layer_idx]['train_loss'], label='Train Loss')
        plt.plot(metrics[layer_idx]['val_loss'], label='Val Loss')
        plt.title(f'{model_name} {probe_type} Layer {layer_idx} Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(metrics[layer_idx]['train_acc'], label='Train Acc')
        plt.plot(metrics[layer_idx]['val_acc'], label='Val Acc')
        plt.title(f'{model_name} {probe_type} Layer {layer_idx} Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()

        plt.tight_layout()

        # Save the plot
        plot_filename = f"{model_name}_{probe_type}_layer_{layer_idx}_metrics.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path)
        plt.close()

def run_experiment(model, train_dataset, val_dataset, layer_indices, num_classes, hyperparams, model_name, probe_type):
    """Run a single experiment with specified hyperparameters."""
    learning_rate = hyperparams['learning_rate']
    num_epochs = hyperparams['num_epochs']
    batch_size = hyperparams['batch_size']

    # For complex probes, get hidden_dim from hyperparams
    if probe_type == 'complex':
        hidden_dim = hyperparams['hidden_dim']
    else:
        hidden_dim = None  # Hidden dimension is not used for linear probes

    # Initialize the Accelerator
    accelerator = Accelerator()

    # Create data loaders with current batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # Create probes with the current hidden_dim
    if probe_type == 'linear':
        probes = [
            LinearProbe(model.model.config.hidden_size, num_classes=num_classes)
            for _ in layer_indices
        ]
    elif probe_type == 'complex':
        probes = [
            ComplexProbe(model.model.config.hidden_size, hidden_dim=hidden_dim, num_classes=num_classes)
            for _ in layer_indices
        ]
    else:
        raise ValueError(f"Unknown probe type: {probe_type}")

    # Create a timestamped output directory for this configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if probe_type == 'complex':
        output_dir = os.path.join(
            "training_runs",
            f"{model_name}_{probe_type}_run_{timestamp}_lr{learning_rate}_epochs{num_epochs}_hd{hidden_dim}_bs{batch_size}"
        )
    else:
        # Do not include hidden_dim in the directory name for linear probes
        output_dir = os.path.join(
            "training_runs",
            f"{model_name}_{probe_type}_run_{timestamp}_lr{learning_rate}_epochs{num_epochs}_bs{batch_size}"
        )
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)

    # Train probes
    metrics = train_probes(
        model=model,
        probes=probes,
        layer_indices=layer_indices,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        accelerator=accelerator,
        output_dir=output_dir,
        model_name=model_name,
        probe_type=probe_type  # Pass probe_type here
    )

    # Get validation accuracy at the last epoch for the last layer probed
    last_layer_idx = layer_indices[-1]
    val_acc = metrics[last_layer_idx]['val_acc'][-1]
    print(f"Validation Accuracy for layer {last_layer_idx}: {val_acc}%")

    # Store the results
    result = {
        'model_name': model_name,
        'probe_type': probe_type,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'val_acc': val_acc,
        'metrics': metrics,
        'output_dir': output_dir
    }
    if probe_type == 'complex':
        result['hidden_dim'] = hidden_dim  # Include hidden_dim only for complex probes

    # Plot metrics
    if accelerator.is_main_process:
        plot_metrics(metrics, layer_indices, output_dir, model_name, probe_type)

    # Save the final probes
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        for i, probe in enumerate(probes):
            probe_filename = f"{model_name}_{probe_type}_probe_layer_{layer_indices[i]}_final.pt"
            probe_path = os.path.join(output_dir, probe_filename)
            accelerator.save(probe.state_dict(), probe_path)

    return result

def main():
    # Added argument parsing for model and probe selection
    parser = argparse.ArgumentParser(description='LLAMA Probing')
    parser.add_argument('--model', choices=['llama', 'openmath'], default='llama',
                        help='Choose which model to use: "llama" or "openmath"')
    parser.add_argument('--probe_type', choices=['linear', 'complex'], default='complex',
                        help='Choose which type of probe to use: "linear" or "complex"')
    args = parser.parse_args()

    if args.model == 'llama':
        model_path = os.path.expanduser("~/.llama/checkpoints/Llama3.1-8B-Instruct-hf")
        model_name = 'llama'
    elif args.model == 'openmath':
        model_path = os.path.expanduser("~/.llama/checkpoints/OpenMath2-Llama3.1-8B")
        model_name = 'openmath'
    else:
        raise ValueError(f"Unknown model choice: {args.model}")

    probe_type = args.probe_type

    print(f"Loading model from {model_path}")
    model = LlamaModelWrapper(model_path)

    # Load the dataset from a JSONL file
    dataset_path = "datasets/distraction_clauses_dataset.jsonl"
    with open(dataset_path, 'r') as f:
        dataset = [json.loads(line) for line in f]

    # Extract texts and labels
    texts = [item['problem'] for item in dataset]
    labels = [item['label'] for item in dataset]  # Labels are strings like "positive" or "negative"

    # Create a mapping from label strings to indices
    label_set = set(labels)
    label_to_index = {label: idx for idx, label in enumerate(sorted(label_set))}
    num_classes = len(label_to_index)
    print(f"Label to index mapping: {label_to_index}")

    # Create dataset
    dataset = SimpleDataset(texts, labels, model.tokenizer, max_length=128, label_to_index=label_to_index)

    # Use a subset of the dataset for training (optional)
    subset_indices = torch.randperm(len(dataset))[:5*32]  # Adjust as needed
    dataset = torch.utils.data.Subset(dataset, subset_indices)

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Specify layers to probe
    layer_indices = [16, 26, 31]  # Adjust based on model architecture

    # Define hyperparameter ranges
    learning_rates = [1e-5, 2e-5, 5e-5]
    num_epochs_list = [10]
    batch_sizes = [16, 32]

    # For complex probes, define hidden_dims
    if probe_type == 'complex':
        hidden_dims = [32, 64]
        # Create list of hyperparameter configurations for complex probes
        hyperparameter_configs = list(itertools.product(learning_rates, num_epochs_list, hidden_dims, batch_sizes))
    else:
        # For linear probes, no hidden_dims needed
        hyperparameter_configs = list(itertools.product(learning_rates, num_epochs_list, batch_sizes))

    # For storing results
    results = []

    # For each hyperparameter configuration
    for config in hyperparameter_configs:
        if probe_type == 'complex':
            lr, num_epochs, hidden_dim, batch_size = config
        else:
            lr, num_epochs, batch_size = config
            hidden_dim = None  # Hidden dimension is not used for linear probes

        print(f"Running with learning_rate={lr}, num_epochs={num_epochs}, batch_size={batch_size}")
        if hidden_dim:
            print(f"Hidden Dimension: {hidden_dim}")

        hyperparams = {
            'learning_rate': lr,
            'num_epochs': num_epochs,
            'batch_size': batch_size
        }
        if hidden_dim:
            hyperparams['hidden_dim'] = hidden_dim  # Only include if not None

        result = run_experiment(model, train_dataset, val_dataset, layer_indices, num_classes, hyperparams, model_name, probe_type)

        results.append(result)

    # After all configurations, find the best one
    # Assuming higher validation accuracy is better
    best_result = max(results, key=lambda x: x['val_acc'])
    print("\nBest hyperparameter configuration:")
    print(f"Model: {best_result['model_name']}")
    print(f"Probe Type: {best_result['probe_type']}")
    print(f"Learning Rate: {best_result['learning_rate']}")
    print(f"Number of Epochs: {best_result['num_epochs']}")
    print(f"Batch Size: {best_result['batch_size']}")
    if 'hidden_dim' in best_result:
        print(f"Hidden Dimension: {best_result['hidden_dim']}")
    print(f"Validation Accuracy: {best_result['val_acc']}%")
    print(f"Results saved in: {best_result['output_dir']}")

    # Save results to a JSON file
    results_filename = f"{model_name}_{probe_type}_hyperparameter_search_results.json"
    results_path = os.path.join("training_runs", results_filename)
    with open(results_path, 'w') as f:
        json.dump(results, f)

    return model, results

if __name__ == "__main__":
    model, results = main()
