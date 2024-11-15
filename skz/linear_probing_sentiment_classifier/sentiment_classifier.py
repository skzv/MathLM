import torch
from typing import List
from llama_probing_training import LlamaModelWrapper, LinearProbe, ComplexProbe

def load_model_and_probes(model_path: str, probe_paths: List[str], device: str = 'cuda'):
    """Load the saved model and probes."""
    # Load the Llama model
    model_wrapper = LlamaModelWrapper(model_path, device=device)

    # Load the probes
    probes = []
    for probe_path in probe_paths:
        probe = ComplexProbe(model_wrapper.model.config.hidden_size, hidden_dim = 64, num_classes=2).to(device)
        probe.load_state_dict(torch.load(probe_path))
        probe.eval()  # Set to evaluation mode
        probes.append(probe)

    return model_wrapper, probes

def classify_sentiment(model_wrapper: LlamaModelWrapper, probes: List[ComplexProbe], layer_indices: List[int], text: str):
    """Classify sentiment of a given text using the loaded model and probes."""
    device = model_wrapper.device

    # Tokenize the input text
    encoding = model_wrapper.tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)

    # Get layer outputs
    with torch.no_grad():
        layer_outputs = [model_wrapper.get_layer_output(idx, input_ids) for idx in layer_indices]
        layer_outputs = [output[:, -1, :] for output in layer_outputs]

    # Predict sentiment using the probes
    predictions = []
    for probe, layer_output in zip(probes, layer_outputs):
        outputs = probe(layer_output)
        predicted_label = torch.argmax(outputs, dim=1).item()
        predictions.append(predicted_label)

    # Majority voting for final prediction
    print(predictions)
    final_prediction = max(set(predictions), key=predictions.count)
    return final_prediction
