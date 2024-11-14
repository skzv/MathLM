import os
from sentiment_classifier import load_model_and_probes, classify_sentiment

if __name__ == "__main__":
    # Paths
    model_path = os.path.expanduser("~/.llama/checkpoints/Llama3.1-8B-Instruct-hf")
    probe_paths = [f"training/probe_{i}.pt" for i in range(4)]  # Adjust based on number of probes
    layer_indices = [8, 16, 24, 31]  # Same as training

    # Load model and probes
    model_wrapper, probes = load_model_and_probes(model_path, probe_paths)

    # Classify sentiment for a given input
    input_text = "This is one of the best experiences I've ever had!"
    sentiment = classify_sentiment(model_wrapper, probes, layer_indices, input_text)
    sentiment_label = "Positive" if sentiment == 1 else "Negative"

    print("\n\n\n")
    print(f"Input: {input_text}")
    print(f"Predicted Sentiment: {sentiment_label}")
