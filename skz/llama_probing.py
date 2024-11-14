import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import json
import os
from typing import List, Dict
import sentencepiece as spm

class LlamaModel:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = device
        self.model_path = model_path
        
        # Load model parameters
        with open(os.path.join(model_path, "params.json")) as f:
            self.params = json.load(f)
            
        print("Model parameters:", self.params)
        
        # Load state dict
        self.state_dict = torch.load(
            os.path.join(model_path, "consolidated.00.pth"),
            map_location=torch.device(device)
        )
        
        # Initialize tokenizer
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(os.path.join(model_path, "tokenizer.model"))
        
        # Extract layer modules from state dict
        self.layer_modules = self._extract_layer_modules()
        
        print(f"Loaded {len(self.layer_modules)} layer modules")
        
    def _extract_layer_modules(self) -> List[Dict[str, torch.Tensor]]:
        """Extract individual layer parameters from state dict"""
        layers = []
        n_layers = self.params.get("n_layers", 32)
        
        for i in range(n_layers):
            layer_dict = {}
            prefix = f"layers.{i}."
            for key, value in self.state_dict.items():
                if key.startswith(prefix):
                    layer_dict[key] = value
            if layer_dict:
                layers.append(layer_dict)
        
        return layers
    
    def get_layer_output(self, layer_idx: int, input_ids: torch.Tensor) -> torch.Tensor:
        """Get the output of a specific layer"""
        # This is a simplified version - you'd need to implement the actual
        # forward pass through the specific layer using the architecture
        # defined in params.json
        layer_dict = self.layer_modules[layer_idx]
        # Implement forward pass here
        # For now, returning dummy tensor
        return torch.randn(input_ids.size(0), self.params.get("dim", 4096)).to(self.device)

class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class SimpleDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: spm.SentencePieceProcessor, max_length: int = 128):
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
        tokens = self.tokenizer.encode(text)
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [self.tokenizer.pad_id()] * (self.max_length - len(tokens))
        
        return {
            'input_ids': torch.tensor(tokens),
            'label': torch.tensor(label)
        }

def train_probes(model: LlamaModel, 
                 probes: List[LinearProbe], 
                 layer_indices: List[int],
                 train_loader: DataLoader, 
                 val_loader: DataLoader,
                 num_epochs: int = 5,
                 learning_rate: float = 1e-3):
    """Train linear probes for specified layers"""
    device = model.device
    criterion = nn.CrossEntropyLoss()
    optimizers = [AdamW(probe.parameters(), lr=learning_rate) for probe in probes]
    
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
            layer_outputs = [model.get_layer_output(idx, input_ids) for idx in layer_indices]
            
            # Train each probe
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
                
                for i, probe in enumerate(probes):
                    outputs = probe(layer_outputs[i])
                    loss = criterion(outputs, labels)
                    val_losses[i] += loss.item()
                    _, predicted = outputs.max(1)
                    val_correct[i] += predicted.eq(labels).sum().item()
                
                val_total += labels.size(0)
        
        # Print metrics for each layer
        for i, layer_idx in enumerate(layer_indices):
            print(f"\nLayer {layer_idx}:")
            print(f"Train Loss: {train_losses[i]/len(train_loader):.4f}")
            print(f"Train Acc: {100.*train_correct[i]/total:.2f}%")
            print(f"Val Loss: {val_losses[i]/len(val_loader):.4f}")
            print(f"Val Acc: {100.*val_correct[i]/val_total:.2f}%")

def main():
    # Example sentiment classification task
    texts = [
        "I love this movie, it's amazing!",
        "This was a terrible waste of time.",
        "The food was delicious and the service excellent.",
        "I regret watching this, very disappointing.",
        # Add more examples...
    ]
    
    # Labels: 0 for negative, 1 for positive
    labels = [1, 0, 1, 0]
    
    # Initialize model
    model_path = os.path.expanduser("~/.llama/checkpoints/Llama3.1-8B-Instruct")
    model = LlamaModel(model_path)
    
    # Create dataset
    dataset = SimpleDataset(texts, labels, model.tokenizer)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2)
    
    # Specify layers to probe (adjust based on model architecture)
    layer_indices = [8, 16, 24, 31]  # Example layer indices
    
    # Create probes (2 classes for binary classification)
    probes = [
        LinearProbe(model.params["dim"], num_classes=2).to(model.device)
        for _ in layer_indices
    ]
    
    # Train probes
    train_probes(model, probes, layer_indices, train_loader, val_loader)
    
    return model, probes

if __name__ == "__main__":
    model, probes = main()