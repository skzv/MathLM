from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
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

# Move model to optimized device (e.g., GPU if available)
model = accelerator.prepare(model)

# Function to generate a math-related distraction sentence
def generate_distraction_sentence():
    prompt = "Write a distracting but plausible sentence that could fit into a simple math word problem:"
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(accelerator.device)  # Move inputs to the correct device
    outputs = model.generate(inputs, max_length=100, do_sample=True, temperature=0.7)
    sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return sentence.replace('"',"").split('.')[0].replace(prompt, "").strip()

# Function to inject a distraction sentence into the problem
def inject_distraction(problem):
    distraction_sentence = generate_distraction_sentence()
    sentences = problem.split(". ")  # Split the problem into sentences
    if len(sentences) > 1:
        insert_position = random.randint(0, len(sentences) - 2)  # Pick a random position to insert
        sentences.insert(insert_position + 1, distraction_sentence)  # Inject the distraction after a sentence
    else:
        sentences.append(distraction_sentence)  # If only one sentence, append it
    return ". ".join(sentences)

# Function to process and combine datasets into train/test
def process_and_combine_datasets(datasets, split_name, num_examples=5):
    combined_data = []
    for dataset in datasets:
        for item in dataset.select(range(num_examples)):  # Process only `num_examples` examples
            problem = item.get("question", item.get("problem", ""))
            solution = item.get("answer", item.get("solution", ""))
            label = random.choice([0, 1]) # 0 for original problem, 1 for distraction problem
            if label == 1:
                original_problem = problem
                problem = inject_distraction(problem)
                print(f"Original Problem: {original_problem}")
                print(f"Distraction Problem: {problem}")
                print()
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

# Combine and process training datasets (process only a few examples for testing)
process_and_combine_datasets([gsm8k_train, gsm_plus_train], "train", num_examples=5)

# Combine and process test datasets (process only a few examples for testing)
process_and_combine_datasets([gsm8k_test, gsm_plus_test], "test", num_examples=5)
