from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import random
import json
import os
from accelerate import Accelerator

# Set the path to your local Llama model
model_path = os.path.expanduser("~/.llama/checkpoints/Llama3.1-8B-Instruct-hf")

# Initialize Accelerator
accelerator = Accelerator()

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Handle the pad_token if it's not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Move model to optimized device (e.g., GPU if available)
model = accelerator.prepare(model)

# Function to generate math-related distraction sentences in batches
def generate_distraction_sentences(batch_size):
    prompt = "Write a distracting but plausible sentence that could fit into a simple math word problem:"
    prompts = [prompt] * batch_size  # Repeat the same prompt for the batch
    # inputs = tokenizer.encode(prompts, return_tensors="pt").to(accelerator.device)  # Move inputs to the correct device
    # outputs = model.generate(inputs, max_length=100, do_sample=True, temperature=0.7)
    
    # Use tokenizer's `__call__` method for batch processing
    inputs = tokenizer(
        prompts,
        return_tensors="pt",  # Return PyTorch tensors
        padding=True,         # Pad sequences to the same length
        truncation=True,      # Truncate sequences to model's max length
    ).to(accelerator.device)  # Move inputs to the correct device

    # Generate outputs for the batch
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=100,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id,  # Ensure padding tokens are handled
    )

    sentences = [
        tokenizer.decode(output, skip_special_tokens=True).replace('"', "").split('.')[0].replace(prompt, "").strip()
        for output in outputs
    ]
    return sentences

# Function to inject a distraction sentence into the problem
def inject_distraction(problem, distraction_sentence):
    sentences = problem.split(". ")  # Split the problem into sentences
    if len(sentences) > 1:
        insert_position = random.randint(0, len(sentences) - 2)  # Pick a random position to insert
        sentences.insert(insert_position + 1, distraction_sentence)  # Inject the distraction after a sentence
    else:
        sentences.append(distraction_sentence)  # If only one sentence, append it
    return ". ".join(sentences)

# Function to process and combine datasets into train/test
def process_and_combine_datasets(datasets, split_name, num_examples=-1, batch_size=8):
    combined_data = []
    for dataset in datasets:
        if num_examples < 0:
            dataset_iterable = dataset  # Use the entire dataset
        else:
            dataset_iterable = dataset.select(range(min(num_examples, len(dataset))))  # Limit the number of examples
        
        # Progress bar
        total_examples = len(dataset_iterable)
        for batch_start in tqdm(range(0, total_examples, batch_size), desc=f"Processing {split_name}"):
            # Create a batch using `.select()`
            batch_indices = list(range(batch_start, min(batch_start + batch_size, total_examples)))
            batch = dataset_iterable.select(batch_indices)

            # Generate distraction sentences for the batch
            distraction_sentences = generate_distraction_sentences(len(batch))

            for item, distraction_sentence in zip(batch, distraction_sentences):
                problem = item.get("question", item.get("problem", ""))
                solution = item.get("answer", item.get("solution", ""))
                label = random.choice([0, 1])  # 0 for original problem, 1 for distraction problem
                if label == 1:
                    original_problem = problem
                    problem = inject_distraction(problem, distraction_sentence)
                    # print(f"Original Problem: {original_problem}")
                    # print(f"Distraction Problem: {problem}")
                    # print()
                combined_data.append({
                    "problem": problem,
                    "solution": solution,
                    "label": label
                })
    # Save the combined dataset
    output_file = f"datasets/combined_{split_name}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(combined_data, f, indent=4)
    print(f"Combined {split_name} dataset saved to {output_file}")

# Load GSM8k dataset splits
gsm8k = load_dataset("openai/gsm8k", 'main')
gsm8k_train = gsm8k["train"]
gsm8k_test = gsm8k["test"]

# Load GSM-Plus dataset split
gsm_plus = load_dataset("qintongli/GSM-Plus")
gsm_plus_train = gsm_plus["test"]
gsm_plus_test = gsm_plus["testmini"]

# Combine and process training datasets (process all examples or a few examples)
process_and_combine_datasets([gsm8k_train, gsm_plus_train], "train", num_examples=-1, batch_size=64)

# Combine and process test datasets (process all examples or a few examples)
process_and_combine_datasets([gsm8k_test, gsm_plus_test], "test", num_examples=-1, batch_size=64)
