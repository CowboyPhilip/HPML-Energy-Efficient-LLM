#!/usr/bin/env python
"""
Script to run MMLU benchmark with energy monitoring on DeepSeek models.
"""

import os
import sys
import argparse
import json
from datetime import datetime
import torch
from utils.memory_utils import clean_memory, print_gpu_memory
from patched_energy_utils import PatchedEnergyTracker as EnergyTracker, get_carbon_intensity, joules_to_co2
from basework.basetest.test_mmlu import test_quantized_models_on_mmlu, quick_test_mmlu
import ctypes
import pynvml


def main():
    if not hasattr(pynvml, 'nvmlFieldValue_t'):
        class nvmlFieldValue_t(ctypes.Structure):
            _fields_ = [
                ('fieldId', ctypes.c_uint),
                ('scopeId', ctypes.c_uint),
                ('timestamp', ctypes.c_ulonglong),
                ('latencyUsec', ctypes.c_uint),
                ('valueType', ctypes.c_uint),
                ('nvmlReturn', ctypes.c_uint),
                ('value', ctypes.c_ulonglong),  # Union; simplified
            ]
        pynvml.nvmlFieldValue_t = nvmlFieldValue_t
        print("Patched missing nvmlFieldValue_t struct.")
        

    # ⚡ NEW PATCH: Patch NVML_FI_DEV_POWER_INSTANT
    if not hasattr(pynvml, 'NVML_FI_DEV_POWER_INSTANT'):
        if hasattr(pynvml, 'NVML_FI_DEV_TOTAL_ENERGY_CONSUMPTION'):
            pynvml.NVML_FI_DEV_POWER_INSTANT = pynvml.NVML_FI_DEV_TOTAL_ENERGY_CONSUMPTION
            print("Patched missing NVML_FI_DEV_POWER_INSTANT with TOTAL_ENERGY_CONSUMPTION.")
        else:
            pynvml.NVML_FI_DEV_POWER_INSTANT = 0
            print("Patched missing NVML_FI_DEV_POWER_INSTANT with dummy value 0.")
    
    parser = argparse.ArgumentParser(description="Run MMLU benchmark with energy monitoring")
    
    # Model selection
    parser.add_argument("--model", type=str, default="deepseek-ai/deepseek-coder-1.3b-base",
                       help="DeepSeek model to test")
    
    # Quantization options
    parser.add_argument("--quantization", type=str, nargs="+", default=["int4"],
                       choices=["fp16", "int8", "int4"],
                       help="Quantization modes to test")
    
    # MMLU options
    parser.add_argument("--subjects", type=str, nargs="+", 
                       default=["high_school_mathematics", "high_school_physics"],
                       help="MMLU subjects to test")
    parser.add_argument("--max_samples", type=int, default=50,
                       help="Maximum number of samples to test per subject")
    
    # Test mode
    parser.add_argument("--quick", action="store_true",
                       help="Run quick test with minimal samples")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Directory to save results")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print setup information
    print("\n===== MMLU Benchmark Setup =====")
    print(f"Model: {args.model}")
    print(f"Quantization modes: {', '.join(args.quantization)}")
    print(f"Subjects: {', '.join(args.subjects)}")
    print(f"Max samples: {args.max_samples}")
    print(f"Output directory: {args.output_dir}")
    
    # Print GPU information
    print("\n===== GPU Information =====")
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print_gpu_memory()
    else:
        print("CUDA is not available. This test requires a GPU.")
        return
    
    # Run benchmark
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(args.output_dir, f"mmlu_results_{timestamp}.json")
    
    if args.quick:
        # Run quick test with first quantization mode
        print("\n===== Running Quick MMLU Test =====")
        mode = args.quantization[0]
        results = quick_test_mmlu(
            model_name=args.model,
            quant_mode=mode,
            subjects=args.subjects,
            max_samples=min(10, args.max_samples)  # Limit to 10 samples for quick test
        )
        
        # Save quick test results
        with open(results_file, 'w') as f:
            json.dump({f"{mode}_quick_test": results}, f, indent=2)
            
        print(f"Quick test results saved to {results_file}")
        
    else:
        # Run full benchmark
        print("\n===== Running MMLU Benchmark =====")
        results = test_quantized_models_on_mmlu(
            model_name=args.model,
            subjects=args.subjects,
            quantization_modes=args.quantization,
            batch_size=1  # Using batch_size=1 to minimize memory usage
        )
        
        # Save full benchmark results
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Benchmark results saved to {results_file}")
    
    print("\n===== Benchmark Complete =====")

if __name__ == "__main__":
    main()