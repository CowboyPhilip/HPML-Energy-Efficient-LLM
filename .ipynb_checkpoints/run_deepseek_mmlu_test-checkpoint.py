import os
import sys
import torch
import argparse
import json
from datetime import datetime

# Import our testing modules
from utils.test_mmlu import test_quantized_models_on_mmlu, quick_test_mmlu
from utils.plot_utils import plot_energy_comparison, plot_component_energy
from utils.memory_utils import clean_memory, print_gpu_memory

def main():
    parser = argparse.ArgumentParser(description='Test DeepSeek model energy consumption on MMLU benchmark')
    
    # Model arguments
    parser.add_argument('--model', type=str, default="deepseek-ai/deepseek-coder-1.3b-base",
                        help='DeepSeek model name from HuggingFace')
    
    # Test configuration
    parser.add_argument('--quantization', type=str, nargs='+', default=['int4'],
                        choices=['fp16', 'int8', 'int4'], 
                        help='Quantization modes to test')
    parser.add_argument('--quick_test', action='store_true',
                        help='Run a quick test with limited samples')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for evaluation')
    parser.add_argument('--max_samples', type=int, default=50,
                        help='Maximum number of samples to evaluate per subject')
    
    # MMLU-specific arguments
    parser.add_argument('--subjects', type=str, nargs='+', default=None,
                        help='List of MMLU subjects to evaluate (None for all)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--plot', action='store_true',
                        help='Generate energy consumption plots')
    
    args = parser.parse_args()
    
    # Print GPU info
    print("\n===== GPU Information =====")
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print_gpu_memory()
    else:
        print("CUDA is not available. This test requires a GPU.")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate timestamp for unique results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set up result file paths
    result_file = os.path.join(args.output_dir, f"mmlu_energy_{timestamp}.json")
    
    # Run tests
    results = {}
    
    if args.quick_test:
        print("\n===== Running Quick MMLU Test =====")
        mode = args.quantization[0]  # Use first mode for quick test
        quick_results = quick_test_mmlu(
            model_name=args.model,
            quant_mode=mode,
            subjects=args.subjects,
            max_samples=min(10, args.max_samples)  # Limit to 10 samples for quick test
        )
        
        results[f'quick_test_{mode}'] = quick_results
        
        # Save quick test results
        quick_result_file = os.path.join(args.output_dir, f"mmlu_quick_test_{timestamp}.json")
        with open(quick_result_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Quick test results saved to {quick_result_file}")
    else:
        print("\n===== Running Full MMLU Benchmark =====")
        test_results = test_quantized_models_on_mmlu(
            model_name=args.model,
            subjects=args.subjects,
            quantization_modes=args.quantization,
            batch_size=args.batch_size,
            max_samples=args.max_samples
        )
        
        results['mmlu'] = test_results
        
        # Save full test results
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Full benchmark results saved to {result_file}")
        
        # Generate plots if requested
        if args.plot:
            try:
                print("\n===== Generating Energy Consumption Plots =====")
                
                # Format results for plotting
                plot_data = {'generation': {}}  # The plot_utils expects 'generation' key
                for mode in args.quantization:
                    if mode in test_results and 'total_energy' in test_results[mode]:
                        # Copy results to the expected structure
                        plot_data['generation'][mode] = test_results[mode]
                
                # Generate plots
                if len(args.quantization) > 1:
                    plot_energy_comparison(plot_data)
                
                # Plot component energy for each mode
                for mode in args.quantization:
                    if mode in test_results and 'component_energy' in test_results[mode]:
                        plot_component_energy(plot_data, quant_mode=mode)
                
                print("Plots generated successfully.")
            except Exception as e:
                print(f"Error generating plots: {e}")
    
    print("\n===== Testing Complete =====")

if __name__ == "__main__":
    main()