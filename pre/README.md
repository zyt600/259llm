# Model Size vs Score Plotting Tool

This tool generates scatter plots showing the relationship between model size (in billions of parameters) and composite evaluation scores.

## Overview

The script `plot_model_size_vs_score.py` visualizes model performance by plotting:
- **X-axis**: Model size (billion parameters, e.g., 0.6B, 1.7B, 4B)
- **Y-axis**: Composite score (sum of three metrics)

### Composite Score Calculation

The composite score is calculated as:

```
Composite Score = QMSum ROUGE-L + TruthfulQA Accuracy + Normalized Max Score
```

Where:
- **QMSum ROUGE-L**: Summarization quality score (0-1)
- **TruthfulQA Accuracy**: Answer accuracy (0-1)
- **Normalized Max Score**: TruthfulQA max score normalized from [-1, 0] to [0, 1]

## Usage

### Basic Usage

```bash
# Generate plot with default temperature (0.0)
python plot_model_size_vs_score.py

# Specify a different temperature
python plot_model_size_vs_score.py --temp 0.5

# Show all models (disable filtering)
python plot_model_size_vs_score.py --no-filter
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--temp` | float | 0.0 | Temperature value to filter results. Only data points at this temperature will be used. |
| `--no-filter` | flag | False | Disable blacklist/whitelist filtering. Shows all available models. |
| `--max_temp` | float | None | *Deprecated*. Use `--temp` instead. |

### Examples

```bash
# Plot results at temperature 0.0 (default, most deterministic)
python plot_model_size_vs_score.py

# Plot results at temperature 0.5
python plot_model_size_vs_score.py --temp 0.5

# Plot all models without filtering
python plot_model_size_vs_score.py --no-filter

# Combine options
python plot_model_size_vs_score.py --temp 0.7 --no-filter
```

## Input Data

### Results Directory

The script reads evaluation results from `../results/` directory. Expected file naming format:

```
{org}_{model}_{dataset}_temp{temperature}_results.json
```

Examples:
- `Qwen_Qwen3-1.7B_qmsum_temp0.0_results.json`
- `unsloth_Phi-4-mini-reasoning_truthfulqa_temp0.5_results.json`

### Result File Format

**QMSum results** (`*_qmsum_*_results.json`):
```json
{
  "rougeL": 0.25
}
```

**TruthfulQA results** (`*_truthfulqa_*_results.json`):
```json
{
  "accuracy": 0.45,
  "avg_max_score": -0.35
}
```

## Filtering

### Whitelist (`white_list.txt`)

If a whitelist file exists and contains entries, **only** models in the whitelist will be plotted. Whitelist takes priority over blacklist.

Format (one model per line):
```
Qwen3-0.6B
Qwen3-1.7B
unsloth/Phi-4-mini-reasoning
```

### Blacklist (`black_list.txt`)

If no whitelist is provided, models in the blacklist will be excluded from the plot.

Format (one model per line, comments start with `#`):
```
# GGUF models
unsloth/Qwen3-1.7B-GGUF
unsloth/Qwen3-4B-GGUF

# Large models
unsloth/Qwen3-14B
```

## Output

### Generated Files

The script generates a PNG image in the same directory:

```
model_size_vs_score_temp{temperature}.png
```

Example: `model_size_vs_score_temp0.0.png`

### Plot Features

- Scatter plot with model names as labels
- Linear trend line with slope value
- Grid lines for easy reading
- Data summary printed to console

### Console Output

The script prints a summary table:

```
============================================================
Data Summary
============================================================
Model                                    Size (B)     Composite Score ROUGE-L    Accuracy   Max Score 
----------------------------------------------------------------------------------------------------
Qwen3-0.6B                              0.60         1.2345          0.2100     0.4500     -0.3500   
Qwen3-1.7B                              1.70         1.4567          0.2500     0.5000     -0.2800   
```

## Requirements

- Python 3.7+
- matplotlib
- numpy

Install dependencies:
```bash
pip install matplotlib numpy
```

## File Structure

```
pre/
├── plot_model_size_vs_score.py    # Main plotting script
├── black_list.txt                  # Blacklist file (optional)
├── white_list.txt                  # Whitelist file (optional)
├── model_size_vs_score_temp0.0.png # Output image
└── README.md                       # This file
```

## Notes

1. **Model Size Extraction**: The script automatically extracts model size from model names (e.g., "Qwen3-1.7B" → 1.7B). Models without recognizable size patterns will be skipped.

2. **Complete Data Required**: Only models with both QMSum and TruthfulQA results at the specified temperature will be plotted.

3. **Display Names**: Model names are simplified for display (e.g., `Qwen_Qwen3-1.7B` → `Qwen3-1.7B`).
