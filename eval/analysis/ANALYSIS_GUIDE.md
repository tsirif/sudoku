# Sudoku Evaluation Analysis Guide

This document describes all analysis scripts in `/data/szhang967/dlm_agents/sudoku/eval/analysis` and their outputs.

## Overview

The analysis scripts generate visualizations and comparison plots for Sudoku evaluation results. They process evaluation CSV/JSON files and create publication-quality figures.

---

## Analysis Scripts

### 1. `analyze_all.py` - Main Analysis Orchestrator

**Purpose**: Runs all individual analysis plotting scripts automatically.

**Usage**:
```bash
cd /data/szhang967/dlm_agents/sudoku/eval/analysis

# Use default paths
python analyze_all.py

# Specify custom paths
python analyze_all.py \
    --results-root /path/to/eval_results_all \
    --output-root /path/to/analysis_results_all
```

**What it does**:
- Automatically runs three plotting scripts:
  1. `plot_absorbing_board_accuracy.py`
  2. `plot_uniform_noise_metrics.py`
  3. `plot_uniform_noise_diffusion_metrics.py`

**Output**: See individual script outputs below.

---

### 2. `plot_absorbing_board_accuracy.py` - Absorbing Board Accuracy Plots

**Purpose**: Plot board accuracy vs sampling steps for absorbing evaluation mode.

**Usage**:
```bash
python plot_absorbing_board_accuracy.py \
    --results-root /path/to/eval_results_all \
    --output-root /path/to/output
```

**Generated Files**:
- **Location**: `{output_root}/{checkpoint_path}/`
- **Files**:
  - `mask_{mask_ratio}.png` - Line plot showing board accuracy vs step for each mask_ratio
  - `sources.json` - Metadata about source CSV files

**Example Output Path**:
```
{output_root}/
  checkpoints_absorbing/
    step_2000/
      mask_0.1.png
      mask_0.2.png
      mask_0.3.png
      sources.json
```

**Plot Content**:
- X-axis: Sampling step
- Y-axis: Board accuracy (0.0 to 1.0)
- Multiple lines: One for each algorithm
- Title: Mask ratio value

---

### 3. `plot_uniform_noise_metrics.py` - Uniform Noise Confidence Metrics

**Purpose**: Plot confidence comparison (clean vs noise tokens) for uniform_noise_only evaluation.

**Usage**:
```bash
python plot_uniform_noise_metrics.py \
    --results-root /path/to/eval_results_all \
    --output-root /path/to/output
```

**Generated Files**:
- **Location**: `{output_root}/{checkpoint_path}/`
- **Files**:
  - `uniform_noise_confidence_ratio_{noise_ratio}.png` - Bar chart comparing clean vs noise token confidence
  - `uniform_noise_sources.json` - Metadata about source CSV files

**Example Output Path**:
```
{output_root}/
  checkpoints_absorbing/
    step_2000/
      uniform_noise_confidence_ratio_0.1.png
      uniform_noise_confidence_ratio_0.2.png
      uniform_noise_confidence_ratio_0.3.png
      uniform_noise_sources.json
```

**Plot Content**:
- Two bars: "Noise" (red) and "Clean" (blue)
- Y-axis: Confidence value `p(token = current)`
- Annotation: Shows clean/noise ratio
- Title: Confidence (noise_ratio={value})

---

### 4. `plot_uniform_noise_diffusion_metrics.py` - Uniform Noise Diffusion Metrics

**Purpose**: Visualize board accuracy for uniform_noise_diffusion evaluation across different steps and parameter combinations.

**Usage**:
```bash
python plot_uniform_noise_diffusion_metrics.py \
    --results-root /path/to/eval_results_all \
    --output-root /path/to/output
```

**Generated Files**:
- **Location**: `{output_root}/{checkpoint_path}/`
- **Files**:
  - `step_{step}/noise_{noise_ratio}_editable_{editable_ratio}.png` - Bar chart comparing algorithms at specific step
  - `lines/noise_{noise_ratio}_editable_{editable_ratio}.png` - Line plot showing board accuracy vs steps

**Example Output Path**:
```
{output_root}/
  checkpoints_absorbing/
    step_2000/
      step_2/
        noise_0.1_editable_0.2.png
        noise_0.1_editable_0.3.png
        ...
      step_3/
        noise_0.1_editable_0.2.png
        ...
      lines/
        noise_0.1_editable_0.2.png
        noise_0.1_editable_0.3.png
        ...
      uniform_noise_diffusion_sources.json
```

**Plot Content**:
- **Bar charts** (in `step_{step}/`):
  - X-axis: Algorithm names
  - Y-axis: Board accuracy
  - Colors: Blue for self_conf algorithms, Red for others
  - Title: noise_ratio={value}, editable_ratio={value}
  
- **Line charts** (in `lines/`):
  - X-axis: Sampling steps
  - Y-axis: Board accuracy (0.0 to 1.0)
  - Multiple lines: One for each algorithm
  - Title: noise_ratio={value}, editable_ratio={value}

---

### 5. `compare_all.py` - Cross-Checkpoint Comparison (Uniform Noise)

**Purpose**: Compare uniform_noise_only results across multiple checkpoints/models. Generates grouped bar charts.

**Usage**:
```bash
python compare_all.py \
    --results-dir /path/to/results \
    --output-dir /path/to/output
```

**Generated Files**:
- **Location**: `{output_dir}/`
- **Files**:
  - `sudoku_uniform_noise_comparison.pdf` - Publication-quality PDF with grouped bar charts

**Plot Content**:
- **Layout**: Multiple subplots (one per noise_ratio)
- **Bars**: Grouped bars for each model showing:
  - Clean token confidence (lighter color)
  - Noise token confidence (darker color with hatch pattern)
- **Annotations**: 
  - Values on each bar
  - Clean/noise ratio above each pair
- **Models**: 
  - Absorbing Objective (red)
  - Mixed Objective (blue, different shades for different prob values)
- **Font**: Times New Roman (publication quality)

---

### 6. `compare_all_line_plots.py` - Cross-Checkpoint Line Plots (Uniform Noise Diffusion)

**Purpose**: Compare uniform_noise_diffusion results across multiple checkpoints/models. Generates line plots.

**Usage**:
```bash
python compare_all_line_plots.py \
    --results-dir /path/to/results \
    --output-dir /path/to/output
```

**Generated Files**:
- **Location**: `{output_dir}/uniform_noise_diffusion_compare/`
- **Files**:
  - `sudoku_uniform_noise_diffusion_{algorithm}_{params}_comparison.pdf` - Line plots for each algorithm/parameter combination

**Plot Content**:
- **Layout**: Grid of subplots (one per noise_ratio × editable_ratio combination)
- **Lines**: 
  - X-axis: Sampling Steps (displayed as 1, 2, 3, but data is 2, 3, 4)
  - Y-axis: Board Accuracy (0.0 to 0.9)
  - Multiple lines: One for each model/checkpoint
- **Legend**: Shows model names (Absorbing Objective, Mixed Objective)
- **Colors**: Model-specific colors
- **Line styles**: Different styles for different Mixed Objective variants
- **Font**: Times New Roman (publication quality)

---

### 7. `compare_completion_line_plots.py` - Cross-Checkpoint Line Plots (Absorbing)

**Purpose**: Compare absorbing results across multiple checkpoints/models. Generates line plots.

**Usage**:
```bash
python compare_completion_line_plots.py \
    --results-dir /path/to/results \
    --output-dir /path/to/output
```

**Generated Files**:
- **Location**: `{output_dir}/absorbing_completion_compare/`
- **Files**:
  - `sudoku_absorbing_{algorithm}_{params}_comparison.pdf` - Line plots for each algorithm/parameter combination

**Plot Content**:
- **Layout**: Grid of subplots (one per mask_ratio)
- **Lines**:
  - X-axis: Sampling Steps
  - Y-axis: Board Accuracy
  - Multiple lines: One for each model/checkpoint
- **Legend**: Shows model names
- **Font**: Times New Roman (publication quality)

---

## Shell Scripts

### `compare_all.sh` - Batch Comparison Script

**Purpose**: Convenience script to run `compare_all.py` with predefined settings.

**Usage**:
```bash
bash compare_all.sh
```

**Configuration**: Edit the script to modify:
- `RESULTS_DIR`: Input results directory
- `OUTPUT_DIR`: Output directory

---

### `compare_all_line_plots.sh` - Batch Line Plots Script

**Purpose**: Convenience script to run `compare_all_line_plots.py` with predefined settings.

**Usage**:
```bash
bash compare_all_line_plots.sh
```

---

## Default Paths

### Input (Results)
- **Default**: `{project_root}/eval_results_all/`
- **Fallback**: `{project_root}/eval_results/`

### Output (Analysis Results)
- **Default**: `{project_root}/analyiss_results_all/` (Note: typo in original code)
- **Individual scripts**: `{script_dir}/` (analysis directory itself)

---

## Data Requirements

### Input Files

All scripts expect CSV or JSON files in the results directory with specific columns:

1. **Absorbing evaluation**:
   - Required columns: `eval_mode`, `mask_ratio`, `step`, `board_accuracy`, `algorithm`
   - Files: `absorbing_results.csv` or `absorbing_results.json`

2. **Uniform noise only**:
   - Required columns: `eval_mode`, `noise_ratio`, `noise_token_confidence`, `clean_token_confidence`
   - Files: `uniform_noise_only_results.csv` or `uniform_noise_only_results.json`

3. **Uniform noise diffusion**:
   - Required columns: `eval_mode`, `noise_ratio`, `editable_ratio`, `steps`, `algorithm`, `board_accuracy`
   - Files: `uniform_noise_diffusion_results.csv` or `uniform_noise_diffusion_results.json`

---

## Complete Workflow

### Step 1: Run Individual Analysis (Per Checkpoint)

```bash
cd /data/szhang967/dlm_agents/sudoku/eval/analysis

# Run all individual analyses
python analyze_all.py \
    --results-root /path/to/eval_results_all \
    --output-root /path/to/analysis_results_all
```

**Outputs**:
- Per-checkpoint plots in `{output_root}/{checkpoint_path}/`
- Bar charts, line plots, confidence comparisons

### Step 2: Run Cross-Checkpoint Comparisons

```bash
# Compare uniform noise results
python compare_all.py \
    --results-dir /path/to/results \
    --output-dir /path/to/output

# Compare uniform noise diffusion results
python compare_all_line_plots.py \
    --results-dir /path/to/results \
    --output-dir /path/to/output

# Compare absorbing results
python compare_completion_line_plots.py \
    --results-dir /path/to/results \
    --output-dir /path/to/output
```

**Outputs**:
- Cross-checkpoint comparison PDFs
- Publication-quality figures

---

## Output File Structure Summary

```
analysis_results_all/  (or custom output-root)
  {checkpoint_path}/
    ├── mask_{ratio}.png                    # Absorbing board accuracy
    ├── sources.json
    ├── uniform_noise_confidence_ratio_{ratio}.png
    ├── uniform_noise_sources.json
    ├── step_{step}/
    │   └── noise_{ratio}_editable_{ratio}.png
    ├── lines/
    │   └── noise_{ratio}_editable_{ratio}.png
    └── uniform_noise_diffusion_sources.json

{output_dir}/
  ├── sudoku_uniform_noise_comparison.pdf
  ├── uniform_noise_diffusion_compare/
  │   └── sudoku_uniform_noise_diffusion_{algorithm}_{params}_comparison.pdf
  └── absorbing_completion_compare/
      └── sudoku_absorbing_{algorithm}_{params}_comparison.pdf
```

---

## Model Name Mapping

The scripts automatically map model names for display:

- `absorbing` → `Absorbing Objective` (in some scripts) or `MDLM` (in newer versions)
- `prob_0.1`, `prob_0.2`, `prob_0.02` → `Mixed Objective` or `CDLM` (in newer versions)

---

## Quick Reference Commands

```bash
# 1. Run all individual analyses
python analyze_all.py

# 2. Compare uniform noise across checkpoints
python compare_all.py \
    --results-dir /path/to/results \
    --output-dir /path/to/output

# 3. Compare uniform noise diffusion across checkpoints
python compare_all_line_plots.py \
    --results-dir /path/to/results \
    --output-dir /path/to/output

# 4. Compare absorbing across checkpoints
python compare_completion_line_plots.py \
    --results-dir /path/to/results \
    --output-dir /path/to/output
```

---

## Notes

- All PDF outputs use Times New Roman font for publication quality
- PNG outputs use default matplotlib fonts
- Scripts automatically discover all checkpoints in the results directory
- Missing data is handled gracefully (skips with warnings)
- Metadata files (`sources.json`) track which CSV files were used

