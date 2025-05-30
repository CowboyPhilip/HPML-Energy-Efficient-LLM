#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universal runner for energy, accuracy and latency benchmarks.
Supports tasks: generation, math, mbpp, mmlu, glue.
"""

import json
import argparse
from pathlib import Path
import wandb


# bench functions
from utils.test_generation import (
    compare_generation_energy,
    quick_test_generation,
    test_generation_MATH,
    test_generation_MBPP
)
from utils.test_mmlu import (
    quick_test_mmlu,
    test_quantized_models_on_mmlu
)
from utils.test_glue import test_quantized_models_on_glue
from utils.energy_utils import get_carbon_intensity, joules_to_co2
from utils.adaptive_quant import AdaptiveQuantGenerator


def run_generation(args):
    """Run text generation benchmarks, including adaptive mode."""
    modes = list(args.modes)
    results = {}

    # adaptive mode
    if "adaptive" in modes:
        print("\n=== Testing ADAPTIVE generation ===")
        agent = AdaptiveQuantGenerator(
            args.model,
            high_mode=args.high_mode,
            low_mode=args.low_mode,
            device_map=args.device_map    # pass through
        )
        # generate will log energy & latency internally
        _ = agent.generate(args.prompt, max_new_tokens=args.tokens)
        results["adaptive"] = {"note": "see adaptive_quant logs"}
        modes.remove("adaptive")

    # non-adaptive modes
    if modes:
        if len(modes) == 1:
            mode = modes[0]
            print(f"\n=== Testing {mode.upper()} generation ===")
            stats = quick_test_generation(
                model_name=args.model,
                quant_mode=mode,
                prompt=args.prompt,
                max_new_tokens=args.tokens,
                device_map=args.device_map    # pass through
            )
            results[mode] = stats
        else:
            print(f"\n=== Comparing modes: {modes} ===")
            stats = compare_generation_energy(
                model_name=args.model,
                prompt=args.prompt,
                quantization_modes=modes,
                max_new_tokens=args.tokens,
                verbose=args.verbose,
                device_map=args.device_map    # pass through
            )
            results.update(stats)

    return {"generation": results}


def run_math(args):
    """Run MATH dataset benchmark."""
    print(f"\n=== Testing MATH on {args.model} ===")
    
    stats = test_generation_MATH(
        model_name=args.model,
        quantization_modes=args.modes,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        num_examples=args.num_examples,
        verbose=args.verbose,
        device_map=args.device_map,
        temperature=0,
        top_p=0.9,
        high_mode=args.high_mode,
        low_mode=args.low_mode,
        ctx_threshold=args.ctx_threshold,
        latency_threshold=args.latency_threshold
    )
    return {"math": stats}


def run_mbpp(args):
    """Run MBPP dataset benchmark."""
    print(f"\n=== Testing MBPP on {args.model} ===")
    stats = test_generation_MBPP(
        model_name=args.model,
        quantization_modes=args.modes,
        num_examples=args.num_examples,
        verbose=args.verbose,
        device_map=args.device_map,
        temperature=0,
        top_p=0.9,
        high_mode=args.high_mode,
        low_mode=args.low_mode,
        ctx_threshold=args.ctx_threshold,
        latency_threshold=args.latency_threshold
    )
    return {"mbpp": stats}


def run_mmlu(args):
    """Run MMLU dataset benchmark."""
    print(f"\n=== Testing MMLU on {args.model} ===")
    if args.quick:
        stats = quick_test_mmlu(
            model_name=args.model,
            quant_mode=args.modes[0],
            subjects=args.subjects,
            max_samples=args.max_samples,
            device_map=args.device_map    # pass through
        )
    else:
        stats = test_quantized_models_on_mmlu(
            model_name=args.model,
            quantization_modes=args.modes,
            subjects=args.subjects,
            device_map=args.device_map    # pass through
        )
    return {"mmlu": stats}


def run_glue(args):
    """Run GLUE tasks benchmark."""
    print(f"\n=== Testing GLUE on {args.model} ===")
    stats = test_quantized_models_on_glue(
        model_name=args.model,
        tasks=args.glue_tasks,
        quantization_modes=args.modes,
        batch_size=args.batch_size,
        device_map=args.device_map    # pass through
    )
    return {"glue": stats}


def summarize_results(results):
    """Print summary of energy, latency, accuracy and CO₂."""
    ci = get_carbon_intensity()
    print("\n=== SUMMARY ===")
    for task, modes in results.items():
        print(f"\n--- {task.upper()} ---")
        for mode, data in modes.items():
            summary = data.get("summary", data)
            e = summary.get("avg_energy", summary.get("total_energy", 0.0))
            t = summary.get("avg_time", summary.get("total_time", 0.0))
            ept = summary.get("energy_per_token", None)
            acc = summary.get("accuracy", None)
            co2 = summary.get(
                "carbon_emissions",
                joules_to_co2(summary.get("total_energy", e), ci)
            )
            line = f"{mode:>12}: E={e:.2f} J, Lat={t:.3f} s"
            if ept is not None:
                line += f", E/token={ept:.6f} J"
            if acc is not None:
                line += f", Acc={acc:.2f}%"
            line += f", CO₂={co2:.4f} g"
            print(line)


def main():
    parser = argparse.ArgumentParser(
        description="Run energy/accuracy/latency benchmarks"
    )
    parser.add_argument(
        "--task", required=True,
        choices=["generation", "math", "mbpp", "mmlu", "glue"],
        help="Which benchmark to run"
    )
    parser.add_argument(
        "--model", required=True,
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--modes", nargs="+", default=["int8_vanilla"],
        help="List of quant+kernel modes, e.g. fp16_flash int8_flash int4_paged adaptive"
    )
    parser.add_argument(
        "--device_map", choices=["auto","cuda"],
        default="auto",
        help="Deployment mode: 'auto' for offload, 'cuda' for full GPU"
    )
    # adaptive endpoints
    parser.add_argument(
        "--high_mode", default="fp16_vanilla",
        help="High-precision mode for adaptive"
    )
    parser.add_argument(
        "--low_mode", default="int8_vanilla",
        help="Low-precision mode for adaptive"
    )
    parser.add_argument(
        "--ctx_threshold", type=int, default=1024,
        help="context-length threshold to switch modes"
    )
    parser.add_argument(
        "--latency_threshold", type=float, default=0.08,
        help="per-token latency threshold to switch modes"
    )
    # generation
    parser.add_argument(
        "--prompt", default="Hello world",
        help="Prompt for generation"
    )
    parser.add_argument(
        "--tokens", type=int, default=128,
        help="Number of new tokens to generate"
    )
    # math
    parser.add_argument(
        "--dataset_name", default="deepmind/math_dataset",
        help="HuggingFace dataset for MATH"
    )
    parser.add_argument(
        "--dataset_config", default="algebra__linear_1d",
        help="Dataset config for MATH"
    )
    parser.add_argument(
        "--split", default="test",
        help="Data split: test or validation"
    )
    parser.add_argument(
        "--num_examples", type=int, default=50,
        help="Number of examples for MATH/MBPP"
    )
    # mmlu
    parser.add_argument(
        "--subjects", nargs="+", default=None,
        help="List of MMLU subjects (None for all)"
    )
    parser.add_argument(
        "--max_samples", type=int, default=50,
        help="Max samples per subject for quick MMLU"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run quick MMLU sampling"
    )
    # glue
    parser.add_argument(
        "--glue_tasks", nargs="+", default=["sst2"],
        help="List of GLUE tasks"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size for GLUE"
    )
    # misc
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--out", default="results.json",
        help="Path to save JSON results"
    )

    args = parser.parse_args()

    # # adaptive only for generation
    # modes = args.modes
    # if args.task != "generation" and "adaptive" in modes:
    #     print("Warning: 'adaptive' only supported for generation; skipping.")
    #     modes.remove("adaptive")
    # args.modes = modes

    wandb.init(
    project="HPML-Energy-Efficient-LLM",
    entity = "HPML-Energy-Efficient-LLM",
    name=f"{args.model}-{args.task}-{args.modes}",
    tags=[args.model.split('/')[-1], args.task] + args.modes,
    group=args.model.split('/')[-1],
    job_type=args.task,
    config=vars(args)
)

    # dispatch
    if args.task == "generation":
        results = run_generation(args)
    elif args.task == "math":
        results = run_math(args)
    elif args.task == "mbpp":
        results = run_mbpp(args)
    elif args.task == "mmlu":
        results = run_mmlu(args)
    else:
        results = run_glue(args)

    # summarize & save
    summarize_results(results)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nSaved results to {args.out}")

    wandb.log(results)
    wandb.finish()
    print(f"\nSaved results to wandb")

if __name__ == "__main__":
    main()



## cmd for testing mbpp

# coder 50 samples
# python run_experiment.py --task mbpp --model deepseek-ai/deepseek-coder-1.3b-instruct --modes fp16 int8_vanilla int4_vanilla adaptive --num_examples 50  --device_map cuda --out results/results_mbpp_coder_50examples.json

# qwen distill 50 samples
# python run_experiment.py --task mbpp --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --modes fp16 int8_vanilla int4_vanilla adaptive --num_examples 50  --device_map cuda --out results/results_mbpp_qwen_50examples.json


## cmd for math

# coder 50 samples
# python run_experiment.py --task math --model deepseek-ai/deepseek-coder-1.3b-instruct --modes fp16 int8_vanilla int4_vanilla adaptive --num_examples 1  --device_map cuda --out results/results_math_coder_50examples.json

# qwen 50 samples
# python run_experiment.py --task math --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --modes fp16 int8_vanilla int4_vanilla adaptive --num_examples 20  --device_map cuda --out results/results_math_qwen_20examples.json

## cmd for mmlu

# coder 50 samples
# python run_experiment.py --task mmlu --model deepseek-ai/deepseek-coder-1.3b-instruct --modes fp16 int8_vanilla int4_vanilla adaptive --num_examples 1  --dataset_name  --out results/results_mmlu_coder_50examples.json

# qwen distill 50 samples