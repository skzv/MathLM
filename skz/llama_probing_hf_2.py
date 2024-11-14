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

class LlamaModelWrapper:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = device
        self.model_path = model_path

        # Use the fast tokenizer by setting use_fast=True
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

        # Handle the pad_token if it's not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = LlamaModel.from_pretrained(model_path).to(device)

        print(f"Loaded model from {model_path}")
        print(f"Model configuration: {self.model.config}")
        
    def get_layer_output(self, layer_idx: int, input_ids: torch.Tensor) -> torch.Tensor:
        """Get the output of a specific layer."""
        # Get hidden states from the model
        outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        # Hidden states are a tuple with (layer + embedding outputs)
        # Layer indexing: hidden_states[0] is embeddings, hidden_states[1] is first layer, etc.
        return outputs.hidden_states[layer_idx + 1]  # +1 to account for embeddings

class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class SimpleDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: AutoTokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

def train_probes(model: LlamaModelWrapper, 
                 probes: List[LinearProbe], 
                 layer_indices: List[int],
                 train_loader: DataLoader, 
                 val_loader: DataLoader,
                 num_epochs: int = 5,
                 learning_rate: float = 1e-3):
    """Train linear probes for specified layers."""
    device = model.device
    criterion = nn.CrossEntropyLoss()
    optimizers = [AdamW(probe.parameters(), lr=learning_rate) for probe in probes]
    
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
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training
        for probe in probes:
            probe.train()
        
        train_losses = [0.0 for _ in layer_indices]
        train_correct = [0 for _ in layer_indices]
        total = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            # Get layer outputs
            with torch.no_grad():
                layer_outputs = [model.get_layer_output(idx, input_ids) for idx in layer_indices]
            
            layer_outputs = [output[:, -1, :] for output in layer_outputs]
            
            for i, (probe, optimizer) in enumerate(zip(probes, optimizers)):
                optimizer.zero_grad()
                outputs = probe(layer_outputs[i])
                loss = criterion(outputs, labels)
                loss.backward()
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
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['label'].to(device)
                
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
    
    return metrics

def plot_metrics(metrics, layer_indices):
    """Plot training and validation metrics for each layer."""
    for layer_idx in layer_indices:
        plt.figure(figsize=(10, 4))
        
        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(metrics[layer_idx]['train_loss'], label='Train Loss')
        plt.plot(metrics[layer_idx]['val_loss'], label='Val Loss')
        plt.title(f'Layer {layer_idx} Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(metrics[layer_idx]['train_acc'], label='Train Acc')
        plt.plot(metrics[layer_idx]['val_acc'], label='Val Acc')
        plt.title(f'Layer {layer_idx} Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

        # save
        plt.savefig(f"training/layer_{layer_idx}.png")

def main():
    # Example sentiment classification task
    texts = [
        "I love this movie, it's amazing!",
        "This was a terrible waste of time.",
        "The food was delicious and the service excellent.",
        "I regret watching this, very disappointing.",
        "An absolutely fantastic performance!",
        "Not my cup of tea, I didn't enjoy it.",
        "The plot was predictable and boring.",
        "A delightful experience from start to finish.",
        # Add more examples...
    ]
    
    # Labels: 0 for negative, 1 for positive
    labels = [1, 0, 1, 0, 1, 0, 0, 1]
    
    # Initialize model
    model_path = os.path.expanduser("~/.llama/checkpoints/Llama3.1-8B-Instruct-hf")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LlamaModelWrapper(model_path, device=device)
    
    # Create dataset
    dataset = SimpleDataset(texts, labels, model.tokenizer)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2)
    
    # Specify layers to probe
    layer_indices = [8, 16, 24, 31]  # Example layer indices
    
    # Create probes (2 classes for binary classification)
    probes = [
        LinearProbe(model.model.config.hidden_size, num_classes=2).to(model.device)
        for _ in layer_indices
    ]
    
    # Train probes
    metrics = train_probes(model, probes, layer_indices, train_loader, val_loader)
    
    # Plot metrics
    plot_metrics(metrics, layer_indices)

    return model, probes

if __name__ == "__main__":
    model, probes = main()

    # Save the probes
    for i, probe in enumerate(probes):
        torch.save(probe.state_dict(), f"probe_{i}.pt")

    