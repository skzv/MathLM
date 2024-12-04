import json

# Define the input and output file paths
input_file_path = 'gsm_train_with_rewritten.jsonl'
output_file_path = 'gsm_train_with_rewritten_formatted.jsonl'

# Read the input file and process it line by line
with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
    for line in infile:
        entry = json.loads(line.strip())

        
        # Create positive and negative examples
        positive_example = {
            "problem": entry["question"],
            "label": "negative"
        }
        negative_example = {
            "problem": entry["rewritten_question"],
            "label": "positive"
        }
        
        # Write the examples to the output file in JSONL format
        outfile.write(json.dumps(positive_example) + '\n')
        outfile.write(json.dumps(negative_example) + '\n')

print(f"Dataset saved to {output_file_path}")
