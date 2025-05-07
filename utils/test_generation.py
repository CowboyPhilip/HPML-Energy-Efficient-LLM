from utils.energy_utils import EnergyTracker, get_carbon_intensity, joules_to_co2
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding
)
import torch
from utils.load_llm import load_llm, _parse_mode
from utils.memory_utils import clean_memory, print_gpu_memory
from datasets import load_dataset
from tqdm import tqdm
import wandb

def compare_generation_energy(model_name, prompt, quantization_modes=['fp32'], verbose=True, device_map: str = "auto"):
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
            model = load_llm(model_name, mode=mode, device_map=device_map)

             # Parse mode to extract precision ('fp32','fp16','int8','int4')
            q_mode, _ = _parse_mode(mode)
            tracker = EnergyTracker(model, precision_mode=q_mode)

            # Tokenize once for all modes
            tokens = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
            print("Running inference...")
            _, stats = tracker.measure_text(tokens.input_ids, tokenizer)

            # Calculate carbon footprint
            carbon_emissions = joules_to_co2(stats['total_energy'], carbon_intensity)
            stats['carbon_emissions'] = carbon_emissions

            # Save results
            results[mode] = stats

            # W&B: log per-mode metrics
            wandb.log({
                'mode': mode,
                'total_energy_J': stats['total_energy'],
                'energy_per_token_J': stats.get('energy_per_token', 0),
                'inference_time_s': stats.get('time', 0),
                'carbon_g': stats['carbon_emissions'],
                'energy_efficiency': stats.get('accuracy', 0) / stats['total_energy'],
                'energy_savings_percent': stats.get('energy_savings', 0)
            })

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

    # Compare efficiency if we have results for FP32
    if 'fp32' in results and 'total_energy' in results['fp32']:
        baseline_mode = 'fp32'
    elif 'fp16' in results and 'total_energy' in results['fp16']:
        baseline_mode = 'fp16'
    else:
        # pick the mode with highest total_energy
        valid = [m for m in quantization_modes if m in results and 'total_energy' in results[m]]
        baseline_mode = max(valid, key=lambda m: results[m]['total_energy']) if valid else None

    if baseline_mode:
        baseline_energy = results[baseline_mode]['total_energy']
        print(f"\n===== Efficiency Comparison (baseline: {baseline_mode.upper()}) =====")
        for mode in quantization_modes:
            if mode != baseline_mode and mode in results and 'total_energy' in results[mode]:
                savings = 100 * (baseline_energy - results[mode]['total_energy']) / baseline_energy
                results[mode]['energy_savings'] = savings
                print(f"{mode.upper()} saves {savings:.2f}% energy compared to {baseline_mode.upper()}")

    # Summary table
    print("\n===== Summary Table =====")
    headers = ["Mode", "Energy (J)", "Time (s)", "Energy/Token (J)", "CO2 (gCO2eq)", "Savings (%)"]
    print(" | ".join(headers))
    print("-" * 90)
    for mode in quantization_modes:
        if mode in results and 'total_energy' in results[mode]:
            stats = results[mode]
            save_pct = f"{stats.get('energy_savings', 0):.2f}"
            row = [
                mode.upper(),
                f"{stats['total_energy']:.4f}",
                f"{stats['time']:.3f}",
                f"{stats.get('energy_per_token',0):.6f}",
                f"{stats.get('carbon_emissions',0):.6f}",
                save_pct
            ]
            print(" | ".join(row))

    return results


def quick_test_generation(model_name, quant_mode='fp16', device_map: str = "auto"):
    """Run a quick test for a single quantization mode on generation task"""
    print(f"Quick test for {model_name} with {quant_mode} quantization")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    clean_memory()
    # Load model
    model = load_llm(model_name, mode=quant_mode, device_map=device_map)

    # Parse quantization mode
    q_mode, _ = _parse_mode(quant_mode)
    tracker = EnergyTracker(model, precision_mode=q_mode)

    # Prepare and run inference on a fixed prompt
    prompt = "DeepSeek AI is an advanced open-source language model designed to power AI applications."
    tokens = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=128)
    print(f"Running inference with prompt: '{prompt}'")
    _, stats = tracker.measure_text(tokens.input_ids, tokenizer)

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
    
def test_generation_MBPP(model_name, quantization_modes=['fp16'], num_examples = 500, verbose=True, device_map: str = "auto", temperature = 0.5, top_p = 0.9):
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
           # Load model and set precision mode for energy tracker
            model = load_llm(model_name, mode=mode, device_map=device_map)
            q_mode, _ = _parse_mode(mode)                          # Parse quantization
            tracker = EnergyTracker(model, precision_mode=q_mode)  # Set correct precision

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

                # prompt = "what is computer?"
                formatted_prompt = f"""
                <｜begin▁of▁sentence｜><｜User｜>{prompt}<｜Assistant｜><think>
                """
                ground_truth_code = example['code']
                test_cases = example['test_list']

                 # Tokenize uniformly
                tokens = tokenizer(formatted_prompt, return_tensors='pt', truncation=True, max_length=512)
                
                try:
                    # Inference
                    # if mode == 'fp16':
                    #     tokens = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
                    #     logits, stats = tracker.measure_text(tokens.input_ids, tokenizer)
                    # else:
                    #     logits, stats = tracker.measure_text(prompt, tokenizer)

                    # # Decode logits -> generated text
                    # generated_tokens = torch.argmax(logits, dim=-1)
                    # generated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

                    

                    logits, stats = tracker.measure_text(tokens.input_ids, tokenizer, temperature = temperature, top_p=top_p)
                except torch.cuda.OutOfMemoryError:
                    # Retry with shorter input
                    print(f"OOM in {mode}, truncating input...")
                    tokens = tokenizer(formatted_prompt, return_tensors='pt', truncation=True, max_length=256)
                    logits, stats = tracker.measure_text(tokens.input_ids, tokenizer, temperature, top_p)
                    
                # Decode logits -> generated text
                generated_tokens = torch.argmax(logits, dim=-1)
                generated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

                # Decode and evaluate correctness
                generated = tokenizer.batch_decode(torch.argmax(logits, dim=-1), skip_special_tokens=True)[0]
                is_correct = evaluate_generated_code(generated, test_cases)

                # Record full info
                results[mode]["examples"].append({
                    "prompt": formatted_prompt,
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
    verbose=True,
    device_map: str = 'auto',
    temperature=0.5,
    top_p=0.9
):
    """
    Benchmark energy use and accuracy on MATH dataset, with optional sampling and prompt formatting.
    Only the final answer is returned, without intermediate steps.
    """
    results = {mode: {"examples": [], "summary": {}} for mode in quantization_modes}

    # Load and sample dataset
    ds = load_dataset(dataset_name, dataset_config, split=split)
    ds = ds.select(range(num_examples))

    # Carbon intensity
    carbon_intensity = get_carbon_intensity()
    if verbose:
        print(f"Carbon intensity: {carbon_intensity} gCO2eq/kWh")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Register special tokens if needed
    special_tokens = ['<｜User｜>', '<｜Assistant｜>', '<think>']
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

    for mode in quantization_modes:
        if verbose:
            print(f"\n=== Testing {mode.upper()} on MATH ===")
        try:
            clean_memory()
            model = load_llm(model_name, mode=mode, device_map=device_map)
            q_mode, _ = _parse_mode(mode)
            tracker = EnergyTracker(model, precision_mode=q_mode)

            correct = 0
            total_tokens = 0

            for item in tqdm(ds, desc=f"MATH {mode.upper()}"):
                if len(results[mode]['examples']) >= num_examples:
                    break
                question = item['question']
                answer   = item['answer'].strip()

                # Format prompt to only output final answer
                formatted_prompt = (
                    f"<｜User｜>{question}"  \
                    f"<｜Assistant｜><think>"  
                )

                # Tokenize
                tokens = tokenizer(
                    formatted_prompt,
                    return_tensors='pt',
                    truncation=True,
                    max_length=512
                )

                # Inference with sampling parameters
                try:
                    logits, stats = tracker.measure_text(
                        tokens.input_ids,
                        tokenizer,
                        temperature=temperature,
                        top_p=top_p
                    )
                except torch.cuda.OutOfMemoryError:
                    tokens = tokenizer(
                        formatted_prompt,
                        return_tensors='pt',
                        truncation=True,
                        max_length=256
                    )
                    logits, stats = tracker.measure_text(
                        tokens.input_ids,
                        tokenizer,
                        temperature=temperature,
                        top_p=top_p
                    )

                # Decode only final answer: take last generated token sequence
                pred_tokens = torch.argmax(logits, dim=-1)
                pred_text = tokenizer.batch_decode(
                    pred_tokens,
                    skip_special_tokens=True
                )[-1].strip()

                # Accuracy
                is_correct = (pred_text == answer)
                correct += int(is_correct)
                total_tokens += stats.get('num_tokens', 1)

                # Record example
                results[mode]['examples'].append({
                    'prompt': formatted_prompt,
                    'ground_truth': answer,
                    'prediction': pred_text,
                    'is_correct': is_correct,
                    'stats': stats
                })

            # Summarize
            count = len(results[mode]['examples'])
            total_energy = sum(e['stats']['total_energy'] for e in results[mode]['examples'])
            total_time   = sum(e['stats']['time']        for e in results[mode]['examples'])
            energy_per_token = total_energy / total_tokens if total_tokens else 0
            accuracy = 100 * correct / count if count else 0
            carbon_emissions = joules_to_co2(total_energy, carbon_intensity)

            results[mode]['summary'] = {
                'examples': count,
                'avg_energy': total_energy / count if count else 0,
                'avg_time': total_time / count if count else 0,
                'energy_per_token': energy_per_token,
                'accuracy': accuracy,
                'carbon_emissions': carbon_emissions
            }

            if verbose:
                print(f"{mode.upper()} SUMMARY: Samples={count}, Acc={accuracy:.2f}%,")

            del model, tracker
            clean_memory()

        except Exception as e:
            print(f"Error in {mode}: {e}")
            results[mode]['summary']['error'] = str(e)

    return results



def test_generation_MMLU(
    model_name,
    quantization_modes=['fp16'],
    dataset_name='mmlu',
    split='validation',
    num_examples=50,
    verbose=True, device_map: str = "auto"
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
        model = load_llm(model_name, mode=mode, device_map=device_map)
        # parse quantization to get correct precision_mode
        q_mode, _ = _parse_mode(mode)
        tracker = EnergyTracker(model, precision_mode=q_mode)

        correct = 0
        total_tokens = 0

        # Iterate through sampled examples
        for item in tqdm(ds, desc=f"MMLU {mode.upper()}"):
            question = item['question']
            choices  = item.get('choices', [])
            answer   = item.get('answer')

            # Build prompt with multiple-choice options
            prompt = question + "\nChoices: " + ", ".join(choices) + "\nAnswer:"

             # Tokenize prompt consistently
            tokens = tokenizer(prompt, return_tensors='pt',
                               truncation=True, max_length=512)
            
            # Measure energy & get logits
            try:
                logits, stats = tracker.measure_text(tokens.input_ids, tokenizer)
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
