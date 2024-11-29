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