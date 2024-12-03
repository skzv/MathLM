import torch
import torch.nn.functional as F

def last_pooling(embeddings):
    return embeddings[:,-1,:]

def variance_pooling(embeddings):
    """
    Perform variance pooling on embeddings.

    Args:
        embeddings (torch.Tensor): Tensor of shape (batch_size, seq_length, hidden_dim)

    Returns:
        torch.Tensor: Variance pooled embeddings of shape (batch_size, hidden_dim)
    """
    # Compute the variance across the sequence length dimension
    variance_pooled = torch.var(embeddings, dim=1, unbiased=False)
    return variance_pooled


def energy_based_pooling(embeddings):
    """
    Perform energy-based pooling on embeddings.

    Args:
        embeddings (torch.Tensor): Tensor of shape (batch_size, seq_length, hidden_dim)

    Returns:
        torch.Tensor: Energy pooled embeddings of shape (batch_size, hidden_dim)
    """
    # Compute the energy (L2 norm squared) for each token embedding
    # Shape: (batch_size, seq_length)
    energy = torch.norm(embeddings, p=2, dim=2) ** 2

    # Compute attention weights by normalizing the energy values
    # Use softmax for better numerical stability
    attention_weights = F.softmax(energy, dim=1)  # Shape: (batch_size, seq_length)

    # Expand attention weights to match the embedding dimensions
    attention_weights = attention_weights.unsqueeze(2)  # Shape: (batch_size, seq_length, 1)

    # Compute the weighted sum of embeddings
    energy_pooled = torch.sum(embeddings * attention_weights, dim=1)  # Shape: (batch_size, hidden_dim)

    return energy_pooled

# Example usage:
batch_size = 32
seq_length = 4096
hidden_dim = 512

# Random embeddings tensor
embeddings = torch.randn(batch_size, seq_length, hidden_dim)

# Apply variance pooling
variance_pooled_embeddings = variance_pooling(embeddings)

print(f"Variance Pooled Embeddings Shape: {variance_pooled_embeddings.shape}")

# Apply energy-based pooling
energy_pooled_embeddings = energy_based_pooling(embeddings)

print(f"Energy Pooled Embeddings Shape: {energy_pooled_embeddings.shape}")