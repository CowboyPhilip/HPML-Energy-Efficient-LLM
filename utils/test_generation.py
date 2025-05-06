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
from datasets import load_dataset
from tqdm import tqdm

def compare_generation_energy(model_name, prompt, quantization_modes=['fp16'], verbose=True):
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


def quick_test_generation(model_name, quant_mode='fp16'):
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


from datasets import load_dataset
from tqdm import tqdm
import contextlib
import io

def evaluate_generated_code(generated_code, test_cases):
    """
    Evaluate generated code by executing it and running provided test cases.

    Args:
        generated_code (str): Generated Python function code
        test_cases (list of str): List of test expressions to evaluate

    Returns:
        bool: True if all test cases pass, False otherwise
    """
    try:
        # Prepare an isolated execution environment
        exec_globals = {}
        exec_locals = {}

        # Execute the generated code
        with contextlib.redirect_stdout(io.StringIO()):
            exec(generated_code, exec_globals, exec_locals)

        # Now run all test cases
        for test in test_cases:
            # Each test is a string like "assert my_func(5) == 120"
            with contextlib.redirect_stdout(io.StringIO()):
                exec(test, exec_globals, exec_locals)

        return True  # All tests passed

    except Exception as e:
        # If any error happens (syntax error, wrong output, etc.), the function is incorrect
        return False
    
def test_generation_MBPP(model_name, quantization_modes=['fp16'], num_examples = 500, verbose=True):
    """
    Test MBPP dataset with energy tracking and pass@1 accuracy, recording full stats.
    """
    # Prepare final results container
    results = {mode: {"examples": [], "summary": {}} for mode in quantization_modes}

    # Load MBPP dataset
    dataset = load_dataset("mbpp", split="test")

    # Get carbon intensity once
    carbon_intensity = get_carbon_intensity()
    print(f"Carbon intensity: {carbon_intensity} gCO2eq/kWh")

    # Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for mode in quantization_modes:
        print(f"\n===== Testing {mode.upper()} Mode on MBPP =====")
        
        try:
            clean_memory()
            model = load_llm(model_name, mode=mode)
            precision = 'float16' if mode == 'fp16' else None
            tracker = EnergyTracker(model, precision_mode=precision)

            # for example in tqdm(dataset, desc=f"Testing {mode.upper()}"):
            for i, example in enumerate(tqdm(dataset, desc=f"Testing {mode.upper()}")):
                if i >= num_examples:
                    break
                header_for_clean_output = "output only the code, no explanation: "
                if "Qwen" in model_name:
                    prompt = header_for_clean_output + example['text']
                elif "coder" in model_name:
                    # prompt = example['text']
                    prompt = header_for_clean_output + example['text']
                    # print(f"===== the prompt is {prompt} =====")
                else:
                    print("unknown deepseek version, use original MBPP text, may lead to low code generation acc")
                    prompt = example['text']

                # prompt = "who are ShakeSpeare?"
                ground_truth_code = example['code']
                test_cases = example['test_list']
                
                try:
                    # Inference
                    if mode == 'fp16':
                        tokens = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
                        logits, stats = tracker.measure_text(tokens.input_ids, tokenizer)
                    else:
                        logits, stats = tracker.measure_text(prompt, tokenizer)

                    # Decode logits -> generated text
                    generated_tokens = torch.argmax(logits, dim=-1)
                    generated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

                except torch.cuda.OutOfMemoryError:
                    print(f"OOM in {mode}, truncating input...")
                    tokens = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=256)
                    logits, stats = tracker.measure_text(tokens.input_ids, tokenizer)
                    generated_tokens = torch.argmax(logits, dim=-1)
                    generated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

                # Evaluate correctness
                is_correct = evaluate_generated_code(generated_text, test_cases)

                # Record full info
                results[mode]["examples"].append({
                    "prompt": prompt,
                    "ground_truth_code": ground_truth_code,
                    "generated_code": generated_text,
                    "test_cases": test_cases,
                    "is_correct": is_correct,
                    "stats": stats
                })

            # Clean up
            del model, tracker
            clean_memory()

        except Exception as e:
            print(f"Error testing {mode} mode: {e}")
            results[mode]['error'] = str(e)

    # Summarize
    print("\n===== Summary =====")
    headers = ["Mode", "Avg Energy per Infer(J)", "Avg Time per Infer (s)", "Energy/Token (J)", "Accuracy (%)", "CO2 (gCO2eq)"]
    print(" | ".join(headers))
    print("-" * 100)

    for mode in quantization_modes:
        if "examples" in results[mode] and len(results[mode]["examples"]) > 0:
            examples = results[mode]["examples"]
            total_energy = sum(ex["stats"]["total_energy"] for ex in examples)
            total_time = sum(ex["stats"]["time"] for ex in examples)
            total_tokens = sum(ex["stats"]["num_tokens"] for ex in examples)
            correct = sum(ex["is_correct"] for ex in examples)
            count = len(examples)

            avg_energy = total_energy / count
            avg_time = total_time / count
            energy_per_token = total_energy / total_tokens if total_tokens > 0 else 0
            accuracy = 100.0 * correct / count
            carbon_emissions = joules_to_co2(total_energy, carbon_intensity)

            # Print
            print(f"{mode.upper()} | {avg_energy:.4f} | {avg_time:.3f} | {energy_per_token:.6f} | {accuracy:.2f} | {carbon_emissions:.6f}")

            # Save summary
            results[mode]["summary"] = {
                "avg_energy": avg_energy,
                "avg_time": avg_time,
                "energy_per_token": energy_per_token,
                "accuracy": accuracy,
                "carbon_emissions": carbon_emissions,
                "total_examples": count
            }


            # print("\nComponent Energy Breakdown:")
            # total_comp = sum(stats['components'].values())
            # for comp, energy in stats['components'].items():
            #     if energy > 0:  # Only show components with energy usage
            #         percentage = 100 * energy / total_comp if total_comp > 0 else 0
            #         print(f"  {comp}: {energy:.4f} J ({percentage:.1f}%)")

            component_totals = {}
            for ex in examples:
                for comp, energy in ex["stats"]["components"].items():
                    if comp not in component_totals:
                        component_totals[comp] = 0.0
                    component_totals[comp] += energy

            grand_total = sum(component_totals.values())

            print("\nComponent Energy Breakdown for", mode.upper())
            for comp, energy in sorted(component_totals.items(), key=lambda x: -x[1]):  # sort descending
                if energy > 0:
                    perc = 100 * energy / grand_total if grand_total > 0 else 0
                    print(f"  {comp}: {energy:.4f} J ({perc:.1f}%)")
    return results

def test_generation_MATH(
    model_name,
    quantization_modes=['fp16'],
    dataset_name='math',
    dataset_config='all',
    split='test',
    num_examples=50,
    verbose=True
):
    """
    Benchmark energy use and accuracy on MATH dataset.
    """
    # Prepare result container
    results = {mode: {"examples": [], "summary": {}} for mode in quantization_modes}

    # Load and sample dataset
    ds = load_dataset(dataset_name, dataset_config, split=split)
    ds = ds.select(range(num_examples))

    # Get carbon intensity
    carbon_intensity = get_carbon_intensity()
    if verbose:
        print(f"Carbon intensity: {carbon_intensity} gCO2eq/kWh")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Test each quantization mode
    for mode in quantization_modes:
        if verbose:
            print(f"\n=== Testing {mode.upper()} on MATH ===")
        try:
            clean_memory()
            # Load model with given quantization
            model = load_llm(model_name, mode=mode)
            precision = 'float16' if mode == 'fp16' else None
            tracker = EnergyTracker(model, precision_mode=precision)

            correct = 0
            total_tokens = 0

            # Iterate over examples
            for item in tqdm(ds, desc=f"MATH {mode.upper()}"):
                question = item['question']
                answer = item['answer']

                # Measure energy and get logits
                try:
                    logits, stats = tracker.measure_text(question, tokenizer)
                except torch.cuda.OutOfMemoryError:
                    # Retry with shorter input on OOM
                    tokens = tokenizer(question, return_tensors='pt', truncation=True, max_length=256)
                    logits, stats = tracker.measure_text(tokens.input_ids, tokenizer)

                # Decode prediction
                pred_tokens = torch.argmax(logits, dim=-1)
                pred_text = tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)[0].strip()

                # Exact match accuracy
                is_correct = (pred_text == answer.strip())
                correct += int(is_correct)
                total_tokens += stats.get('num_tokens', 1)

                # Record example result
                results[mode]["examples"].append({
                    "question": question,
                    "ground_truth": answer,
                    "prediction": pred_text,
                    "is_correct": is_correct,
                    "stats": stats
                })

            # Compute summary metrics
            count = len(results[mode]["examples"])
            total_energy = sum(e["stats"]["total_energy"] for e in results[mode]["examples"])
            total_time = sum(e["stats"]["time"] for e in results[mode]["examples"])
            energy_per_token = total_energy / total_tokens if total_tokens else 0
            accuracy = 100 * correct / count
            carbon_emissions = joules_to_co2(total_energy, carbon_intensity)

            results[mode]["summary"] = {
                "examples": count,
                "avg_energy": total_energy / count,
                "avg_time": total_time / count,
                "energy_per_token": energy_per_token,
                "accuracy": accuracy,
                "carbon_emissions": carbon_emissions
            }

            if verbose:
                print(f"\n{mode.upper()} SUMMARY:")
                print(f"  Samples       : {count}")
                print(f"  Accuracy      : {accuracy:.2f}%")
                print(f"  Energy/Infer  : {results[mode]['summary']['avg_energy']:.4f} J")
                print(f"  Time/Infer    : {results[mode]['summary']['avg_time']:.3f} s")
                print(f"  Energy/Token  : {energy_per_token:.6f} J/token")
                print(f"  CO2 Emissions : {carbon_emissions:.6f} gCO2eq")

            # Cleanup
            del model, tracker
            clean_memory()

        except Exception as e:
            print(f"Error in {mode}: {e}")
            results[mode]["summary"]["error"] = str(e)

    return results


def test_generation_MMLU(
    model_name,
    quantization_modes=['fp16'],
    dataset_name='mmlu',
    split='validation',
    num_examples=50,
    verbose=True
):
    """
    Benchmark energy use and accuracy on MMLU dataset.
    """
    # Load and sample dataset
    ds = load_dataset(dataset_name, split=split)
    ds = ds.select(range(num_examples))

    # Get carbon intensity once
    carbon_intensity = get_carbon_intensity()
    if verbose:
        print(f"Carbon intensity: {carbon_intensity} gCO2eq/kWh")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare results container
    results = {mode: {"examples": [], "summary": {}} for mode in quantization_modes}

    for mode in quantization_modes:
        if verbose:
            print(f"\n=== Testing {mode.upper()} on MMLU ===")

        # Free GPU memory and load model + tracker
        clean_memory()
        model = load_llm(model_name, mode=mode)
        precision = 'float16' if mode == 'fp16' else None
        tracker = EnergyTracker(model, precision_mode=precision)

        correct = 0
        total_tokens = 0

        # Iterate through sampled examples
        for item in tqdm(ds, desc=f"MMLU {mode.upper()}"):
            question = item['question']
            choices  = item.get('choices', [])
            answer   = item.get('answer')

            # Build prompt with multiple-choice options
            prompt = question + "\nChoices: " + ", ".join(choices) + "\nAnswer:"

            # Measure energy & get logits
            try:
                logits, stats = tracker.measure_text(prompt, tokenizer)
            except torch.cuda.OutOfMemoryError:
                # Retry with truncated input on OOM
                tokens = tokenizer(prompt, return_tensors='pt',
                                   truncation=True, max_length=256)
                logits, stats = tracker.measure_text(tokens.input_ids, tokenizer)

            # Decode prediction and compute correctness
            pred_tokens = torch.argmax(logits, dim=-1)
            pred_text   = tokenizer.batch_decode(pred_tokens,
                             skip_special_tokens=True)[0].strip()
            is_correct  = (pred_text == answer.strip())
            correct    += int(is_correct)
            total_tokens += stats.get('num_tokens', 1)

            # Record example-level data
            results[mode]["examples"].append({
                "question": question,
                "choices": choices,
                "ground_truth": answer,
                "prediction": pred_text,
                "is_correct": is_correct,
                "stats": stats
            })

        # Summarize metrics
        count         = len(results[mode]["examples"])
        total_energy  = sum(ex["stats"]["total_energy"]
                            for ex in results[mode]["examples"])
        total_time    = sum(ex["stats"]["time"]
                            for ex in results[mode]["examples"])
        energy_per_tok= total_energy / total_tokens if total_tokens else 0
        accuracy      = 100 * correct / count
        carbon_emis   = joules_to_co2(total_energy, carbon_intensity)

        results[mode]["summary"] = {
            "examples": count,
            "avg_energy": total_energy / count,
            "avg_time": total_time / count,
            "energy_per_token": energy_per_tok,
            "accuracy": accuracy,
            "carbon_emissions": carbon_emis
        }

        if verbose:
            print(f"{mode.upper()} SUMMARY: Samples={count}, "
                  f"Acc={accuracy:.2f}%, E/token={energy_per_tok:.6f} J")

        # Clean up
        del model, tracker
        clean_memory()

    return results
