from utils.energy_utils import EnergyTracker, get_carbon_intensity, joules_to_co2
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding
)
import torch
from utils.load_llm import load_llm, load_classifier
from utils.memory_utils import clean_memory, print_gpu_memory

def compare_generation_energy(model_name, prompt, quantization_modes=['int4'], verbose=True):
    """
    Compare energy consumption of different quantization methods for text generation

    Args:
        model_name: HuggingFace model name
        prompt: Text prompt for inference
        quantization_modes: List of quantization modes to test
        verbose: Whether to print detailed results

    Returns:
        Dictionary containing comparison results
    """
    results = {}

    # Get carbon intensity once for all tests
    carbon_intensity = get_carbon_intensity()
    print(f"Carbon intensity: {carbon_intensity} gCO2eq/kWh")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Test each quantization mode one by one
    for mode in quantization_modes:
        try:
            print(f"\n===== Testing {mode.upper()} Mode =====")

            # Free GPU memory
            clean_memory()

            # Load model with specific quantization
            model = load_llm(model_name, mode=mode)

            # Create energy tracker
            precision = 'float16' if mode == 'fp16' else None
            tracker = EnergyTracker(model, precision_mode=precision)

            # Measure energy with safety measures
            print(f"Running inference...")
            try:
                # Start with smaller prompt length for safety
                if mode == 'fp16':
                    # For FP16, use very small input to save memory
                    print("Using truncated prompt for FP16 mode to save memory")
                    tokens = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=32)
                    _, stats = tracker.measure_text(tokens.input_ids, tokenizer)
                else:
                    # For INT8 and INT4, we can use the full prompt
                    _, stats = tracker.measure_text(prompt, tokenizer)

            except torch.cuda.OutOfMemoryError:
                print(f"Out of memory error with {mode} mode. Trying with smaller input...")
                # Tokenize and truncate prompt
                tokens = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=32)
                _, stats = tracker.measure_text(tokens.input_ids, tokenizer)

            # Calculate carbon footprint
            carbon_emissions = joules_to_co2(stats['total_energy'], carbon_intensity)
            stats['carbon_emissions'] = carbon_emissions

            # Save results
            results[mode] = stats

            # Print results
            if verbose:
                print(f"Total Energy: {stats['total_energy']:.4f} J")
                print(f"Energy per token: {stats['energy_per_token']:.6f} J/token")
                print(f"Inference time: {stats['time']:.3f} s")
                print(f"Carbon emissions: {carbon_emissions:.6f} gCO2eq")

                # Print component-wise energy
                if 'components' in stats:
                    print("\nComponent Energy Breakdown:")
                    total_comp = sum(stats['components'].values())
                    for comp, energy in stats['components'].items():
                        if energy > 0:  # Only show components with energy usage
                            percentage = 100 * energy / total_comp if total_comp > 0 else 0
                            print(f"  {comp}: {energy:.4f} J ({percentage:.1f}%)")

            # Clean up
            del model, tracker
            clean_memory()

        except Exception as e:
            print(f"Error testing {mode} mode: {e}")
            results[mode] = {"error": str(e)}

    # Compare efficiency if we have results for FP16
    if 'fp16' in results and 'total_energy' in results['fp16']:
        baseline = results['fp16']['total_energy']
        print("\n===== Efficiency Comparison =====")
        for mode in ['int8', 'int4']:
            if mode in results and 'total_energy' in results[mode]:
                savings = 100 * (baseline - results[mode]['total_energy']) / baseline
                results[mode]['energy_savings'] = savings
                print(f"{mode.upper()} saves {savings:.2f}% energy compared to FP16")
    elif len(quantization_modes) > 1:
        # If no FP16, compare to highest energy mode
        highest_energy_mode = max(
            [m for m in quantization_modes if m in results and 'total_energy' in results[m]],
            key=lambda m: results[m]['total_energy']
        )
        baseline = results[highest_energy_mode]['total_energy']
        print(f"\n===== Efficiency Comparison (against {highest_energy_mode.upper()}) =====")
        for mode in quantization_modes:
            if mode != highest_energy_mode and mode in results and 'total_energy' in results[mode]:
                savings = 100 * (baseline - results[mode]['total_energy']) / baseline
                results[mode]['energy_savings'] = savings
                print(f"{mode.upper()} saves {savings:.2f}% energy compared to {highest_energy_mode.upper()}")

    # Display summary table
    print("\n===== Summary Table =====")
    headers = ["Mode", "Energy (J)", "Time (s)", "Energy/Token (J)", "CO2 (gCO2eq)"]
    print(" | ".join(headers))
    print("-" * 80)

    for mode in quantization_modes:
        if mode in results and 'total_energy' in results[mode]:
            stats = results[mode]
            values = [
                mode.upper(),
                f"{stats['total_energy']:.4f}",
                f"{stats['time']:.3f}",
                f"{stats['energy_per_token']:.6f}",
                f"{stats.get('carbon_emissions', 0):.6f}"
            ]
            print(" | ".join(values))

    return results


def quick_test_generation(model_name, quant_mode='int4'):
    """Run a quick test for a single quantization mode on generation task"""
    print(f"Quick test for {model_name} with {quant_mode} quantization")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model
    model = load_llm(model_name, mode=quant_mode)

    # Create energy tracker
    precision = 'float16' if quant_mode == 'fp16' else None
    tracker = EnergyTracker(model, precision_mode=precision)

    # Run inference
    prompt = "DeepSeek AI is an advanced open-source language model designed to power AI applications."
    print(f"Running inference with prompt: '{prompt}'")

    _, stats = tracker.measure_text(prompt, tokenizer)

    # Calculate carbon footprint
    carbon_intensity = get_carbon_intensity()
    carbon_emissions = joules_to_co2(stats['total_energy'], carbon_intensity)

    # Print results
    print("\nResults:")
    print(f"Total Energy: {stats['total_energy']:.4f} J")
    print(f"Energy per token: {stats['energy_per_token']:.6f} J/token")
    print(f"Inference time: {stats['time']:.3f} s")
    print(f"Carbon emissions: {carbon_emissions:.6f} gCO2eq")

    # Component breakdown
    print("\nComponent Energy Breakdown:")
    total_comp = sum(stats['components'].values())
    for comp, energy in stats['components'].items():
        if energy > 0:  # Only show components with energy usage
            percentage = 100 * energy / total_comp if total_comp > 0 else 0
            print(f"  {comp}: {energy:.4f} J ({percentage:.1f}%)")

    # Clean up
    del model, tracker
    clean_memory()

    return stats