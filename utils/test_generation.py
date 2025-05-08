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
from utils.adaptive_quant import AdaptiveQuantGenerator
import re
import ast

import contextlib
import io

def extract_clean_function_code_from_output(text: str) -> str:
    """
    从 LLM 输出中提取第一个有效函数定义。优先解析 [BEGIN]... 区块，
    如果不存在，则尝试提取 markdown ```python``` 块或 def 函数块。
    返回 AST 校验通过的第一个函数代码。
    """

    def extract_all_def_blocks(raw: str) -> list[str]:
        blocks = []
        lines = raw.strip().splitlines()
        current_block = []
        inside_func = False

        for line in lines:
            if re.match(r"^def\s+\w+\(.*\):", line):
                if current_block:
                    blocks.append("\n".join(current_block))
                    current_block = []
                inside_func = True
                current_block.append(line)
            elif inside_func and (line.startswith("    ") or line.startswith("\t")):
                current_block.append(line)
            elif inside_func:
                blocks.append("\n".join(current_block))
                current_block = []
                inside_func = False

        if current_block:
            blocks.append("\n".join(current_block))

        return blocks

    # 1. 尝试 [BEGIN] block
    begin_pattern = r"\[BEGIN\]<｜Assistant｜><think>\n(.*?)\n\[END\]<｜Assistant｜><think>"
    match = re.search(begin_pattern, text, re.DOTALL)
    if match:
        block = match.group(1).strip()
        md_inner = re.search(r"```python\n(.*?)```", block, re.DOTALL | re.IGNORECASE)
        code_block = md_inner.group(1) if md_inner else block
    else:
        md_match = re.search(r"```python\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if md_match:
            code_block = md_match.group(1)
        else:
            code_block = text

    # 2. 提取并返回第一个合法 def 函数
    for code in extract_all_def_blocks(code_block):
        try:
            tree = ast.parse(code)
            func_defs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
            if func_defs:
                return ast.unparse(func_defs[0])
        except Exception:
            continue

    return "ast grammar error!"

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
    
def test_generation_MBPP(
    model_name,
    quantization_modes=['fp16'],
    num_examples=500,
    verbose=True,
    device_map: str = "auto",
    temperature=0.5,
    top_p=0.9,
    high_mode='fp16_vanilla',
    low_mode='int8_vanilla',
    ctx_threshold=1024,
    latency_threshold=0.08
):
    """
    Test MBPP dataset with energy tracking and pass@1 accuracy, supporting adaptive quant.
    Returns per-mode examples and summaries.
    """
    # prepare results container
    results = {mode: {"examples": [], "summary": {}} for mode in quantization_modes}

    # load dataset and sample
    ds = load_dataset("mbpp", split="test").select(range(num_examples))

    # carbon intensity
    carbon_intensity = get_carbon_intensity()
    if verbose:
        print(f"Carbon intensity: {carbon_intensity} gCO2eq/kWh")

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for mode in quantization_modes:
        if verbose:
            print(f"\n=== Testing {mode.upper()} on MBPP ===")
        try:
            clean_memory()

            # adaptive branch
            if mode == 'adaptive':
                agent = AdaptiveQuantGenerator(
                    model_name,
                    high_mode=high_mode,
                    low_mode=low_mode,
                    ctx_threshold=ctx_threshold,
                    latency_threshold=latency_threshold,
                    device_map=device_map
                )
                examples = []
                correct = 0
                total_tokens = 0

                for ex in tqdm(ds, desc="MBPP ADAPTIVE"):
                    # prepare prompt
                    task = ex['text']
                    test = "\n".join(ex['test_list'])
                    prompt_body = f"You are an expert Python programmer, and here is your task: {task} Your code should pass these tests:\n\n{test}\n[BEGIN]"
                    prompt = f"<｜begin▁of▁sentence｜><｜User｜>{prompt_body}<｜Assistant｜><think>"
                    # record input length
                    input_ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).input_ids.to(agent.high_model.device)
                    input_len = input_ids.size(1)

                    # generate
                    gen_ids, logits, stats = agent.evaluate(
                        prompt,
                        tokenizer,
                        max_new_tokens=128,
                        temperature=temperature,
                        top_p=top_p
                    )
                    # extract and decode new tokens as code
                    gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                    gen_code = extract_clean_function_code_from_output(gen_text)

                    # gen_tokens = gen_ids[0, input_len:]
                    # pred_code = tokenizer.decode(gen_tokens, skip_special_tokens=True)

                    # evaluate pass@1
                    is_corr = evaluate_generated_code(gen_code, ex['test_list'])
                    correct += int(is_corr)
                    total_tokens += stats.get('num_tokens', 1)

                    examples.append({
                        'prompt': prompt,
                        'ground_truth_code': ex['code'],
                        'generated_text':gen_text,
                        'generated_code': gen_code,
                        'test_list': ex['test_list'],
                        'is_correct': is_corr,
                        'stats': stats
                    })
                # summarize adaptive
                count = len(examples)
                total_energy = sum(e['stats']['total_energy'] for e in examples)
                total_time = sum(e['stats']['time'] for e in examples)
                energy_per_token = total_energy / total_tokens if total_tokens else 0
                accuracy = 100.0 * correct / count if count else 0
                carbon = joules_to_co2(total_energy, carbon_intensity)

                results[mode]['examples'] = examples
                results[mode]['summary'] = {
                    'examples': count,
                    'avg_energy': total_energy / count if count else 0,
                    'avg_time': total_time / count if count else 0,
                    'energy_per_token': energy_per_token,
                    'accuracy': accuracy,
                    'carbon_emissions': carbon
                }

                clean_memory()
                continue

            # static branch
            model = load_llm(model_name, mode=mode, device_map=device_map)
            q_mode, _ = _parse_mode(mode)
            tracker = EnergyTracker(model, precision_mode=q_mode)

            examples = []
            correct = 0
            total_tokens = 0

            for ex in tqdm(ds, desc=f"MBPP {mode.upper()}"):
                # prompt
                # hdr = "output only the code, no explanation: "
                task = ex['text']
                test = "\n".join(ex['test_list'])
                prompt_body = f"You are an expert Python programmer, and here is your task: {task} You should only output code without explanation and your code should pass these tests:\n\n{test}\n[BEGIN]"
                prompt = f"<｜begin▁of▁sentence｜><｜User｜>{prompt_body}<｜Assistant｜><think>"

                # inference
                try:
                    # print("===1===")
                    gen_ids, stats = tracker.measure_generation(prompt, tokenizer, temperature, top_p)
                except torch.cuda.OutOfMemoryError:
                    # 如果mem不足 需要截断prompt 以节省memory
                    # print("===2===")
                    tokenized_prompt = tokenizer(prompt, return_tensors='pt',
                            padding=True, truncation=True, max_length=128)
                    gen_ids, stats = tracker.measure_generation(tokenized_prompt.input_ids.to(model.device), tokenizer, temperature, top_p)

                # decode
                # print("===3===")
                gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                gen_code = extract_clean_function_code_from_output(gen_text)
                # eval
                is_corr = evaluate_generated_code(gen_code, ex['test_list'])
                correct += int(is_corr)
                total_tokens += stats.get('num_tokens', 1)

                examples.append({
                    'prompt': prompt,
                    'ground_truth_code': ex['code'],
                    'generated_text': gen_text,
                    'generated_code': gen_code,
                    'test_list': ex['test_list'],
                    'is_correct': is_corr,
                    'stats': stats
                })
            # cleanup static
            del model, tracker
            clean_memory()

            # summarize static
            count = len(examples)
            total_energy = sum(e['stats']['total_energy'] for e in examples)
            total_time = sum(e['stats']['time'] for e in examples)
            energy_per_token = total_energy / total_tokens if total_tokens else 0
            accuracy = 100.0 * correct / count if count else 0
            carbon = joules_to_co2(total_energy, carbon_intensity)

            results[mode]['examples'] = examples
            results[mode]['summary'] = {
                'examples': count,
                'avg_energy': total_energy / count if count else 0,
                'avg_time': total_time / count if count else 0,
                'energy_per_token': energy_per_token,
                'accuracy': accuracy,
                'carbon_emissions': carbon
            }
            if verbose:
                print(f"{mode.upper()} SUMMARY: Samples={count}, Acc={accuracy:.2f}%,")

        except Exception as e:
            print(f"Error testing {mode} mode: {e}")
            results[mode]['summary']['error'] = str(e)

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
    top_p=0.9,
    high_mode='fp16_vanilla',
    low_mode='int8_vanilla',
    ctx_threshold=1024,
    latency_threshold=0.08
):
    """
    Benchmark energy use and accuracy on MATH dataset with static or adaptive quant.
    """
    results = {mode: {"examples": [], "summary": {}} for mode in quantization_modes}

    # load and sample dataset
    ds = load_dataset(dataset_name, dataset_config, split=split).select(range(num_examples))

    carbon = get_carbon_intensity()
    if verbose:
        print(f"Carbon intensity: {carbon} gCO2eq/kWh")

    # prepare tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    special_tokens = ['<｜User｜>', '<｜Assistant｜>', '<think>']
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

    for mode in quantization_modes:
        if verbose:
            print(f"\n=== Testing {mode.upper()} on MATH ===")
        try:
            clean_memory()

            # adaptive branch
            if mode == 'adaptive':
                agent = AdaptiveQuantGenerator(
                    model_name,
                    high_mode=high_mode,
                    low_mode=low_mode,
                    ctx_threshold=ctx_threshold,
                    latency_threshold=latency_threshold,
                    device_map=device_map
                )
                correct = 0
                total_tokens = 0

                for item in tqdm(ds, desc="MATH ADAPTIVE"):
                    q = item['question']
                    ans = item['answer'].strip()

                    # build instruction-style prompt
                    prompt = (
                        "# Instruction: Solve the following problem.\n"
                        f"# Problem: {q}\n"
                        "# Answer:"
                    )

                    # measure with adaptive generator
                    gen_ids, _, stats = agent.evaluate(
                        prompt, tokenizer,
                        max_new_tokens=128,
                        temperature=temperature,
                        top_p=top_p
                    )

                    # decode only new tokens
                    input_len = tokenizer(prompt, return_tensors='pt').input_ids.size(1)
                    gen_tokens = gen_ids[0, input_len:]
                    pred = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

                    is_corr = (pred == ans)
                    correct += int(is_corr)
                    total_tokens += stats.get('num_tokens', 1)

                    results[mode]['examples'].append({
                        'prompt': prompt,
                        'ground_truth': ans,
                        'prediction': pred,
                        'is_correct': is_corr,
                        'stats': stats
                    })

                # summarize adaptive
                ex = results[mode]['examples']
                n = len(ex)
                e_total = sum(x['stats']['total_energy'] for x in ex)
                t_total = sum(x['stats']['time'] for x in ex)
                results[mode]['summary'] = {
                    'examples': n,
                    'avg_energy': e_total / n if n else 0,
                    'avg_time': t_total / n if n else 0,
                    'energy_per_token': e_total / total_tokens if total_tokens else 0,
                    'accuracy': 100 * correct / n if n else 0,
                    'carbon_emissions': joules_to_co2(e_total, carbon)
                }

                del agent
                clean_memory()
                continue

            # static branch
            model = load_llm(model_name, mode=mode, device_map=device_map)
            q_mode, _ = _parse_mode(mode)
            tracker = EnergyTracker(model, precision_mode=q_mode)

            correct = 0
            total_tokens = 0

            for item in tqdm(ds, desc=f"MATH {mode.upper()}"):
                q = item['question']
                ans = item['answer'].strip()

                # build instruction-style prompt
                prompt = (
                    "# Instruction: Solve the following problem.\n"
                    f"# Problem: {q}\n"
                    "# Answer:"
                )

                # energy-tracked generation
                try:
                    gen_ids, stats = tracker.measure_generation(
                        prompt, tokenizer,
                        temperature=temperature,
                        top_p=top_p
                    )
                except torch.cuda.OutOfMemoryError:
                    # retry with shorter prompt
                    short = tokenizer(
                        prompt, return_tensors='pt',
                        truncation=True, max_length=256
                    )
                    gen_ids, stats = tracker.measure_generation(
                        short.input_ids.to(model.device), tokenizer,
                        temperature=temperature,
                        top_p=top_p
                    )

                # decode output
                pred = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()

                is_corr = (pred == ans)
                correct += int(is_corr)
                total_tokens += stats.get('num_tokens', 1)

                results[mode]['examples'].append({
                    'prompt': prompt,
                    'ground_truth': ans,
                    'prediction': pred,
                    'is_correct': is_corr,
                    'stats': stats
                })

            # summarize static
            ex = results[mode]['examples']
            n = len(ex)
            e_total = sum(x['stats']['total_energy'] for x in ex)
            t_total = sum(x['stats']['time'] for x in ex)
            results[mode]['summary'] = {
                'examples': n,
                'avg_energy': e_total / n if n else 0,
                'avg_time': t_total / n if n else 0,
                'energy_per_token': e_total / total_tokens if total_tokens else 0,
                'accuracy': 100 * correct / n if n else 0,
                'carbon_emissions': joules_to_co2(e_total, carbon)
            }

            if verbose:
                print(f"{mode.upper()} SUMMARY: Samples={n}, Acc={results[mode]['summary']['accuracy']:.2f}%")

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
