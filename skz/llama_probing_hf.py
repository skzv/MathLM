import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch.nn as nn
import torch.optim as optim
import os

# Load the Llama3.1-8B-Instruct model
model_path = os.path.expanduser("~/.llama/checkpoints/Llama3.1-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Freeze the base model
for param in model.parameters():
    param.requires_grad = False

# Add a classification head
class SentimentClassifier(nn.Module):
    def __init__(self, base_model, hidden_size, num_labels):
        super(SentimentClassifier, self).__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])  # Use [CLS] token representation
        return logits

hidden_size = model.config.hidden_size
num_labels = 2  # Positive, Negative
classifier_model = SentimentClassifier(model, hidden_size, num_labels)

# Load sentiment dataset
dataset = load_dataset("imdb")
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# Tokenize dataset
def preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier_model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier_model.classifier.parameters(), lr=1e-5)

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    classifier_model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = classifier_model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

# Evaluate
classifier_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = classifier_model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.2f}")
