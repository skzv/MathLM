# Linear Probing Sentiment Classifier

An exploration of using linear probing of LLama3.1-8B-Instruct layers to do sentiment analysis.

## Prereqs
Set up conda env with 

```
conda env create -f llama.yml
conda activate llama.yml
```

## Training
Train model by running `llama_probing_training.py`. Probes and training loss charts are saved to `training` directory.

## Running Model
Run `run_sentiment_classifier.py` which loads the LLama3.1-8B-Instruct model and probes it. 

## Useful Notes

### Maximizing GPU Utilization
Training was slow, and I noticed that most of the GPU memory was unused during utilization. I was watching the nvidia-smi output continously with `watch -n 1 nvidia-smi` and noticed that most of the GPU memory was unused. So I increased the batch size to maximize the memory utilization. 