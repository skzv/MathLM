from datasets import load_dataset
import os

# Specify the directory to save the datasets
save_directory = os.path.expanduser("~/src/local_datasets")
os.makedirs(save_directory, exist_ok=True)

# Function to download all splits of a dataset
def download_all_splits(dataset_name, config_name=None):
    print(f"Downloading all splits for dataset: {dataset_name}...")
    dataset_splits = load_dataset(dataset_name, config_name)
    for split_name, dataset_split in dataset_splits.items():
        file_name = f"{dataset_name.replace('/', '_')}_{split_name}.json"
        file_path = os.path.join(save_directory, file_name)
        dataset_split.to_json(file_path)
        print(f"Saved {split_name} split of {dataset_name} to {file_path}")

# Download all splits of the GSM8k dataset
download_all_splits("openai/gsm8k", "main")

# Download all splits of the GSM-Plus dataset
download_all_splits("qintongli/GSM-Plus")

print(f"All dataset splits saved to {save_directory}")
