## Install

```bash
conda env create -f environment.yml
conda activate your_env_name
```

## Run

```bash
accelerate launch training.py
```

## Copy models

Put HF format models in `~llama/checkpoints/LlamaXXXXXX`.

## Prepare Dataset

```bash
python prepare_dataset.py
```

## Plot Metrics

```bash
python visualize_metrics.py --metrics_path training_runs/run_TIMESTAMP/metrics.json
```