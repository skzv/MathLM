from datasets import load_dataset

# Load the IMDb dataset
dataset = load_dataset("imdb")

# Use a subset of the dataset for training
train_data = dataset['train'].shuffle(seed=42).select(range(100))  # Select 100 examples for training

# Extract texts and labels
texts = train_data['text']
labels = train_data['label']

# Example sentiment classification task
# texts and labels are now loaded from the IMDb dataset

# print some examples
for i in range(5):
    print(f"Text: {texts[i]}")
    print(f"Label: {labels[i]}")
    print()