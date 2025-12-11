# LLM Evaluation Framework

Evaluate LLMs on QMSum and TruthfulQA benchmarks.

## Run Experiments

1. Edit `run_experiments.sh` to configure models and temperatures:

```bash
MODELS=(
    "google/gemma-3-1b-it"
    "Qwen/Qwen3-0.6B"
    # Add more models...
)

TEMPERATURES=(0 0.5 0.7 1.0)
```

2. Run:

```bash
./run_experiments.sh
```

Results will be saved to `./results/`.

## More Documentation

- **Plotting results**: See [draw/README.md](draw/README.md)
- **Model training**: See [training/README.md](training/README.md)
