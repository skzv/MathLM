## Install

```bash
conda env create -f llama.yml
conda activate llama
```

## Run

```bash
accelerate launch training.py --model [llama, openmath] --probe_type [linear, complex]
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

## Connect to tmux Training Job

```bash
ssh ubuntu@192.222.53.27 -t "tmux attach-session -t training-session || tmux new-session -s training-session"
```

## Github Setup

```bash
sudo apt install gh
git clone https://github.com/skzv/MathLM.git
```

## Install Conda

```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```