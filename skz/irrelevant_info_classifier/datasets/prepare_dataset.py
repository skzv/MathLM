import json

# Define the input and output file paths
input_file_path = 'original_test.jsonl'
output_file_path = 'distraction_clauses_dataset.jsonl'

# Read the input file and process it line by line
with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
    for line in infile:
        entry = json.loads(line.strip())
        
        # Create positive and negative examples
        positive_example = {
            "problem": entry["question"],
            "label": "positive"
        }
        negative_example = {
            "problem": entry["seed_question"],
            "label": "negative"
        }
        
        # Write the examples to the output file in JSONL format
        outfile.write(json.dumps(positive_example) + '\n')
        outfile.write(json.dumps(negative_example) + '\n')

print(f"Dataset saved to {output_file_path}")
