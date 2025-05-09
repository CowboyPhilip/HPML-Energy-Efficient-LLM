# Energy‐Efficient LLM Benchmark

This repository provides **experiments.ipynb** as the single entry point to evaluate energy, latency and accuracy across several tasks (generation, MATH, MBPP, MMLU, GLUE) and quantization/kernel modes.

## Prerequisites

-   Python 3.8+ on a CUDA-capable GPU
-   Internet access to download models & datasets

## Quick Start

1. **Open** `experiments.ipynb` in Colab or Jupyter.
2. ```sh
    # run this cell if you are in colab with a single notebook opened, otherwise ignore this cell
    !git clone https://github.com/CowboyPhilip/HPML-Energy-Efficient-LLM
    %cd HPML-Energy-Efficient-LLM 
    ```
3. **Run** the first cell to install all dependencies.

    ```
    # 1. Install dependencies
    !pip install --upgrade pip setuptools
    !pip install \
        transformers \
        bitsandbytes \
        zeus-ml \
        torch \
        datasets \
        evaluate \
        scikit-learn \
        geocoder \
        requests \
        flash-attn==2.0.5 \
        triton==2.0.0 \
        vllm \
        numpy
    ```
4. **Edit** the **Config** cell at the top:

    - `cfg["task"]`: one of `"generation"`, `"math"`, `"mbpp"`, `"mmlu"`, `"glue"`
    - `cfg["model"]`: e.g. `"deepseek-ai/deepseek-coder-1.3b-instruct"`
    - `cfg["modes"]`: list of quant+kernel modes, e.g.
        
        ```python
        ["fp16_vanilla","int8_vanilla","int4_vanilla","adaptive"]
        ```
    - Other fields: dataset_name, dataset_config, split, num_examples, prompt, tokens, etc.
5. **Run all cells**. The notebook will:
    - Measure per-token energy & latency with `EnergyTracker`
    - Switch precision on-the-fly for `"adaptive"` mode
    - Print a summary table of **avg energy (J)**, **latency (s)**, **accuracy (%)**, and **CO₂ (g)**
    - Generate two plots:
        - `plot_energy_comparison(results)` for overall energy comparison
        - `plot_component_energy(results, task_type, quant_mode)` for component-level breakdown
    - Save raw metrics to `cfg["output_file"]` (default `results.json`)

## Results Interpretation

-   **Summary table** shows per-mode metrics for the selected task.
-   **Overall plot** compares total energy across modes/tasks.
-   **Component plot** shows energy consumed by embeddings, attention, FFN, etc., for a given mode.

## Wanbd URL：
https://wandb.ai/HPML-Energy-Efficient-LLM?shareProfileType=copy
