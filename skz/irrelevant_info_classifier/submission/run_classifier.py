import torch
import argparse
import os
import re
from training_from_cached_activations import (
    LlamaModelWrapper,
    load_probe,
    ActivationDataset,
)


def load_model(model_choice: str, device):
    if model_choice == "llama":
        model_path = os.path.expanduser("~/llama/checkpoints/Llama3.1-8B-Instruct-hf")
        model_name = "llama"
    elif model_choice == "openmath":
        model_path = os.path.expanduser("~/llama/checkpoints/OpenMath2-Llama3.1-8B")
        model_name = "openmath"
    else:
        raise ValueError(f"Unknown model choice: {model_choice}")

    model = LlamaModelWrapper(model_path)
    model.load_model()
    model.model.to(device)
    return model, model_name


def classify_example(model, device, probe, tokenizer, example_text: str, layer_idx: int):
    encoding = tokenizer(
        example_text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    
    input_ids = encoding["input_ids"].to(device)
    with torch.no_grad():
        # print(input_ids.shape) # (1, 512)
        layer_output = model.get_layer_output(layer_idx, input_ids)

        # Extract last token representation
        layer_output = layer_output[:, -1, :]

        # print(layer_output.shape) # (512, 4029)
        # print(layer_output)
        logits = probe(layer_output)
        # print(logits.shape)
        if logits.shape[-1] != 2:
            raise ValueError(f"Expected 2 logits, got {logits.shape[-1]}")
        prediction = torch.argmax(logits, dim=-1).item()
        return prediction


def main():
    parser = argparse.ArgumentParser(
        description="Run classification on an example input."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the probe checkpoint (.pt file)",
    )
    # parser.add_argument(
    #     "--example_text", type=str, required=True, help="Input text for classification"
    # )
    args = parser.parse_args()

    if "llama" in args.checkpoint:
        model_flavor = "llama"
    else:
        model_flavor = "openmath"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, model_name = load_model(model_flavor, device)
    tokenizer = model.tokenizer

    match = re.search(r"layer_(\d+)", args.checkpoint)
    if match:
        layer_idx = int(match.group(1))
    else:
        raise ValueError("Layer index not found in checkpoint filename.")

    probe, hyperparams = load_probe(args.checkpoint, device)
    probe.to(device)

    example_text = "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    example_text = "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

    prediction = classify_example(model, device, probe, tokenizer, example_text, layer_idx)
    label_to_index = {"negative": 0, "positive": 1}
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    predicted_label = index_to_label.get(prediction, "Unknown")

    print(f"Input Text: {example_text}")
    print(f"Predicted Class: {predicted_label}")


if __name__ == "__main__":
    main()
