# Results Visualization Tool

A Python tool for plotting experiment results as temperature-score curves. It visualizes model performance across different temperature settings for QMSum and TruthfulQA benchmarks.

## Overview

This tool reads experiment result JSON files from the `../results/` directory and generates comparison plots showing how different models perform at various temperature settings.

### Generated Plots

The script produces three side-by-side plots:

1. **QMSum (ROUGE-L)** - Summarization quality score
2. **TruthfulQA (BLEURT Accuracy)** - Truthfulness accuracy metric
3. **TruthfulQA (BLEURT Max Score)** - Maximum BLEURT score

## Usage

### Basic Usage

```bash
python plot_results.py
```

This will:
- Load all result files from `../results/`
- Generate `results_comparison.png` in the current directory
- Print a model ranking table sorted by combined score

### Command Line Options

```bash
python plot_results.py [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--max_temp` | float | 1.0 | Maximum temperature to include in plots (only shows data points with temp ≤ max_temp) |
| `--rank_only` | flag | - | Only print the ranking table, skip plot generation |

### Examples

```bash
# Plot results with temperature up to 0.7
python plot_results.py --max_temp 0.7

# Only show ranking, no plot
python plot_results.py --rank_only

# Combine options
python plot_results.py --max_temp 0.5 --rank_only
```

## Input File Format

The script expects result JSON files in `../results/` with the following naming convention:

```
{model_name}_{dataset}_temp{temperature}_results.json
```

### Examples:
- `Qwen3-4B_qmsum_temp0.5_results.json`
- `unsloth_Llama-3.2-3B_truthfulqa_temp0.7_results.json`

### JSON Structure

**QMSum results:**
```json
{
  "rougeL": 0.1234
}
```

**TruthfulQA results:**
```json
{
  "accuracy": 0.5678,
  "avg_max_score": 0.4321
}
```

## Filtering Models

### Whitelist (`white_list.txt`)

If the whitelist is not empty, **only** models listed in the whitelist will be included. The blacklist is ignored when whitelist is active.

```
# Example white_list.txt
Qwen3-4B-Base
unsloth/Qwen3-1.7B
```

### Blacklist (`black_list.txt`)

When whitelist is empty, models listed in the blacklist will be excluded from plots.

```
# Example black_list.txt
# Too large models
unsloth/Qwen3-14B
unsloth/gpt-oss-20b

# Low performance
Llama-3.2-1B
```

### Filtering Rules

- Lines starting with `#` are comments
- Empty lines are ignored
- Matching is **case-insensitive**
- Use the **display name** (as shown in the legend) for matching:
  - For `unsloth/` models: use `unsloth/ModelName`
  - For other models: use just the model name (e.g., `Qwen3-4B-Base`)

## Output

### Plot Output

- **File:** `results_comparison.png`
- **Resolution:** 150 DPI
- **Size:** 24×7 inches (3 subplots)

Each model is plotted with a unique combination of:
- Color (20 colors from tab20 colormap)
- Marker style (14 different markers)
- Line style (solid, dashed, dash-dot, dotted)

### Console Output

The script prints:
1. Loading status and filter information
2. Number of models per dataset
3. Data summary (temperature range, score range, data points)
4. **Model ranking table** sorted by combined score

### Ranking Table

The ranking combines all three metrics:
```
Combined Score = ROUGE-L + Accuracy + Max BLEURT
```

Only models with data in **all three metrics** at the **same temperature** are ranked.

Example output:
```
================================================================================
Model Ranking (High to Low) - Best Temperature for Each Model
Combined Score = QMSum ROUGE-L + TruthfulQA Accuracy + TruthfulQA Max BLEURT
================================================================================
Rank  Model Name                     Best Temp  Combined    ROUGE-L    Accuracy   Max BLEURT
--------------------------------------------------------------------------------
1     Qwen3-4B-Base                  0.50       0.8234      0.2345     0.3456     0.2433
2     unsloth/Qwen3-1.7B             0.70       0.7891      0.2100     0.3200     0.2591
...
================================================================================
```

## Requirements

- Python 3.x
- matplotlib

## File Structure

```
draw/
├── plot_results.py         # Main script
├── black_list.txt          # Models to exclude
├── white_list.txt          # Models to include (priority)
├── results_comparison.png  # Generated plot
└── README.md               # This file
```
