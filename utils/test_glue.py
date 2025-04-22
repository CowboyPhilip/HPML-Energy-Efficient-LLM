from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding
)
from memory_utils import clean_memory, print_gpu_memory
from datasets import load_dataset, concatenate_datasets
from utils.energy_utils import EnergyTracker, get_carbon_intensity, joules_to_co2
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import numpy as np
from utils.load_llm import load_llm, load_classifier

def run_glue_energy_monitoring(model, tokenizer, task="sst2", batch_size=1, precision_mode=None):
    """
    Run GLUE benchmark with energy monitoring, optimized for memory constraints

    Args:
        model: The model to test
        tokenizer: The tokenizer
        task: GLUE task name
        batch_size: Batch size for evaluation (default=1 to minimize memory usage)
        precision_mode: Precision mode for inference

    Returns:
        key_metrics: Batch-level metrics
        avg_metrics: Task-level aggregated metrics
    """
    print(f"Running GLUE task: {task}")

    # Load dataset
    from datasets import load_dataset
    try:
        dataset = load_dataset("glue", task)
        print(f"Successfully loaded {task} dataset")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Downloading dataset manually...")
        # Alternative loading method
        dataset = load_dataset("glue", task, cache_dir="./cache")

    # Use shorter sequence length to save memory
    def preprocess_function(examples):
        max_len = 48  # Short sequence length to save memory
        if task == 'mrpc':
            return tokenizer(examples['sentence1'], examples['sentence2'], padding="max_length", truncation=True, max_length=max_len)
        elif task == 'mnli':
            return tokenizer(examples['premise'], examples['hypothesis'], padding="max_length", truncation=True, max_length=max_len)
        elif task in ['sst2', 'cola']:
            return tokenizer(examples['sentence'], padding="max_length", truncation=True, max_length=max_len)

    # Apply preprocessing
    encoded_dataset = dataset.map(preprocess_function, batched=True)

    # Get validation dataset
    if task == 'mnli':
        validation_dataset = concatenate_datasets([encoded_dataset['validation_matched'], encoded_dataset['validation_mismatched']])
    else:
        validation_dataset = encoded_dataset['validation']

    # Use only a small subset to save memory
    max_samples = 50  # Limit to 50 samples to save memory
    if len(validation_dataset) > max_samples:
        print(f"Limiting validation dataset to {max_samples} samples (from {len(validation_dataset)})")
        validation_dataset = validation_dataset.select(range(max_samples))

    # Show dataset info
    print(f"Dataset columns before processing: {validation_dataset.column_names}")

    # Keep only the necessary columns
    columns_to_keep = ['input_ids', 'attention_mask', 'label']
    validation_dataset = validation_dataset.remove_columns(
        [col for col in validation_dataset.column_names if col not in columns_to_keep]
    )
    validation_dataset.set_format('torch')

    print(f"Final dataset format: {validation_dataset.format}")
    print(f"Dataset columns after processing: {validation_dataset.column_names}")

    # Create DataLoader with minimal batch size
    dataloader = DataLoader(validation_dataset, batch_size=batch_size)
    print(f"Created DataLoader with {len(dataloader)} batches")

    # Initialize tracker
    tracker = EnergyTracker(model, precision_mode=precision_mode)

    # Metrics
    key_metrics = []
    all_predictions = []
    all_true_labels = []

    # Evaluate batches
    print(f"Evaluating {len(dataloader)} batches...")

    # Clear cache before starting
    clean_memory()

    # Process only a few batches for memory safety
    max_batch_eval = min(10, len(dataloader))

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batch_eval:
            print(f"Reached batch limit ({max_batch_eval}). Stopping early to save time.")
            break

        try:
            # Show batch info for first batch
            if batch_idx == 0:
                print(f"Batch keys: {batch.keys()}")
                for key in batch.keys():
                    print(f"{key} shape: {batch[key].shape}")

            # Clean GPU memory before each batch
            clean_memory()

            # Process one sample at a time
            try:
                # Move data to GPU
                input_ids = batch['input_ids'].to('cuda')
                attention_mask = batch['attention_mask'].to('cuda')

                # Get label
                if 'label' in batch:
                    labels = batch['label'].to('cuda')
                elif 'labels' in batch:
                    labels = batch['labels'].to('cuda')
                else:
                    print(f"Warning: No label field. Using dummy label.")
                    labels = torch.zeros(input_ids.size(0), dtype=torch.long, device='cuda')

                # Measure energy
                print(f"Processing batch {batch_idx+1}/{max_batch_eval}...")
                logits, energy_metrics = tracker.measure_batch(input_ids, attention_mask)

                # Get predictions
                predictions = torch.argmax(logits, dim=-1).cpu().numpy()
                true_labels = labels.cpu().numpy()

            except torch.cuda.OutOfMemoryError as e:
                print(f"OOM error. Skipping batch {batch_idx}.")
                clean_memory()
                continue

            all_predictions.extend(predictions)
            all_true_labels.extend(true_labels)

            # Store batch metrics
            batch_metrics = {
                'batch_idx': batch_idx,
                'batch_size': input_ids.size(0),
                'energy': energy_metrics['total_energy'],
                'time': energy_metrics['time'],
                'tokens': energy_metrics['num_tokens'],
                'energy_per_token': energy_metrics['energy_per_token'],
                'components': energy_metrics['components']
            }
            key_metrics.append(batch_metrics)

            # Clean GPU memory after each batch
            clean_memory()

        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            continue

    # Calculate GLUE score
    if len(all_predictions) == 0 or len(all_true_labels) == 0:
        print("No valid predictions were made. Cannot calculate metrics.")
        return [], {"error": "No valid predictions"}

    try:
        if task in ['sst2', 'mnli']:
            score = accuracy_score(all_true_labels, all_predictions)
        elif task == 'cola':
            score = matthews_corrcoef(all_true_labels, all_predictions)
        elif task == 'mrpc':
            f1 = f1_score(all_true_labels, all_predictions)
            acc = accuracy_score(all_true_labels, all_predictions)
            score = (f1 + acc) / 2
        else:
            print(f"Unknown task: {task}, using accuracy as default metric")
            score = accuracy_score(all_true_labels, all_predictions)
        print(f"GLUE score: {score:.4f}")
    except Exception as e:
        print(f"Error calculating score: {e}")
        score = 0

    # Aggregate metrics
    if len(key_metrics) == 0:
        print("No batches were successfully processed")
        return [], {"error": "No batches processed"}

    total_energy = sum(m['energy'] for m in key_metrics)
    total_tokens = sum(m['tokens'] for m in key_metrics)
    total_time = sum(m['time'] for m in key_metrics)

    # Component energy
    component_energy = {}
    for comp in key_metrics[0]['components']:
        component_energy[comp] = sum(m['components'][comp] for m in key_metrics if comp in m['components'])

    # Summary metrics
    avg_metrics = {
        'task': task,
        'glue_score': score,
        'total_energy': total_energy,
        'energy_per_token': total_energy / total_tokens if total_tokens > 0 else 0,
        'throughput': total_tokens / total_time if total_time > 0 else 0,
        'total_time': total_time,
        'total_tokens': total_tokens,
        'component_energy': component_energy
    }

    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        else:
            return obj

    clean_avg_metrics = convert_numpy(avg_metrics)

    return key_metrics, clean_avg_metrics


def test_quantized_models_on_glue(model_name, tasks=['sst2'], quantization_modes=['int4'], batch_size=1):
    """
    Test different quantization modes on GLUE tasks

    Args:
        model_name: HuggingFace model name
        tasks: List of GLUE tasks to evaluate
        quantization_modes: List of quantization modes to test
        batch_size: Batch size for evaluation

    Returns:
        Dictionary with results for each task and quantization mode
    """
    # Results dictionary
    results = {task: {} for task in tasks}

    # Get carbon intensity once
    carbon_intensity = get_carbon_intensity()
    print(f"Carbon intensity: {carbon_intensity} gCO2eq/kWh")

    # Test each task
    for task in tasks:
        print(f"\n===== Testing GLUE Task: {task} =====")

        # Get tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Test each quantization mode
        for mode in quantization_modes:
            try:
                print(f"\n----- Testing {mode.upper()} Mode -----")

                # Clear GPU memory
                clean_memory()

                # Detect number of classes
                try:
                    from datasets import load_dataset
                    dataset = load_dataset("glue", task)
                    num_labels = dataset['train'].features['label'].num_classes
                    print(f"Task {task} has {num_labels} classes")
                except Exception as e:
                    print(f"Error detecting label count: {e}")
                    num_labels = 2  # Default to binary classification

                # Load classifier model
                model = load_classifier(
                    model_name=model_name,
                    mode=mode,
                    num_labels=num_labels
                )

                # Create energy tracker
                precision = 'float16' if mode == 'fp16' else None
                tracker = EnergyTracker(model, precision_mode=precision)

                # Measure energy consumption
                print(f"Running GLUE task {task} with {mode} quantization...")
                _, metrics = run_glue_energy_monitoring(
                    model=model,
                    tokenizer=tokenizer,
                    task=task,
                    batch_size=batch_size,
                    precision_mode=precision
                )

                # Check for errors
                if isinstance(metrics, dict) and metrics.get("error"):
                    print(f"Error in metrics: {metrics['error']}")
                    results[task][mode] = metrics
                    continue

                # Calculate carbon footprint
                carbon_emissions = joules_to_co2(metrics['total_energy'], carbon_intensity)
                metrics['carbon_emissions'] = carbon_emissions

                # Save results
                results[task][mode] = metrics

                # Print results
                print(f"Total Energy: {metrics['total_energy']:.4f} J")
                print(f"Energy per token: {metrics['energy_per_token']:.6f} J/token")
                print(f"GLUE Score: {metrics['glue_score']:.4f}")
                print(f"Carbon emissions: {carbon_emissions:.6f} gCO2eq")

                # Clean up
                del model, tracker
                clean_memory()

            except Exception as e:
                print(f"Error testing {mode} mode on {task}: {e}")
                results[task][mode] = {"error": str(e)}

        # Compare efficiency if we have multiple modes
        modes_with_data = [m for m in quantization_modes if m in results[task] and
                          isinstance(results[task][m], dict) and
                          'total_energy' in results[task][m]]

        if len(modes_with_data) >= 2:
            # Find highest energy mode as baseline
            baseline_mode = max(modes_with_data, key=lambda m: results[task][m]['total_energy'])
            baseline = results[task][baseline_mode]['total_energy']

            print(f"\n----- Efficiency Comparison for {task} -----")
            print(f"Using {baseline_mode.upper()} as baseline ({baseline:.4f} J)")

            for mode in modes_with_data:
                if mode != baseline_mode:
                    savings = 100 * (baseline - results[task][mode]['total_energy']) / baseline
                    results[task][mode]['energy_savings'] = savings
                    print(f"{mode.upper()} saves {savings:.2f}% energy compared to {baseline_mode.upper()}")

    return results