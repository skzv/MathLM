from datasets import load_dataset

# Load the SST-2 dataset
dataset = load_dataset("glue", "sst2")

# Use a subset of the dataset for training
train_data = dataset['train'].shuffle(seed=42).select(range(100))  # Select 100 examples for training

# Extract texts and labels
texts = train_data['sentence']
labels = train_data['label']

# Example sentiment classification task
# texts and labels are now loaded from the SST-2 dataset

# Print some examples
for i in range(20):
    print(f"Text: {texts[i]}")
    print(f"Label: {labels[i]}")
    print()