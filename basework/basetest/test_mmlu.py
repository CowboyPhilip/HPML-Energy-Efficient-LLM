from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding
)
from utils.memory_utils import clean_memory, print_gpu_memory
from datasets import load_dataset, concatenate_datasets
from utils.energy_utils import EnergyTracker, get_carbon_intensity, joules_to_co2
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import accuracy_score
import numpy as np
from utils.load_llm import load_llm, load_classifier

def run_mmlu_energy_monitoring(model, tokenizer, subjects=None, batch_size=1, precision_mode=None):
    """
    Run MMLU benchmark with energy monitoring, optimized for memory constraints

    Args:
        model: The model to test
        tokenizer: The tokenizer
        subjects: List of MMLU subjects to evaluate (None for all)
        batch_size: Batch size for evaluation (default=1 to minimize memory usage)
        precision_mode: Precision mode for inference

    Returns:
        key_metrics: Batch-level metrics
        avg_metrics: Task-level aggregated metrics
    """
    print(f"Running MMLU benchmark")

    # Load dataset
    try:
        dataset = load_dataset("cais/mmlu", "all")
        print(f"Successfully loaded MMLU dataset")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Downloading dataset manually...")
        # Alternative loading method
        dataset = load_dataset("cais/mmlu", "all", cache_dir="./cache")

    # Get validation dataset
    validation_dataset = dataset['validation']
    
    # Filter subjects if specified
    if subjects:
        print(f"Filtering for subjects: {subjects}")
        validation_dataset = validation_dataset.filter(lambda x: x['subject'] in subjects)
    
    # Print subject distribution
    subject_counts = {}
    for item in validation_dataset:
        subject = item['subject']
        if subject in subject_counts:
            subject_counts[subject] += 1
        else:
            subject_counts[subject] = 1
    
    print(f"Subject distribution in validation set:")
    for subject, count in subject_counts.items():
        print(f"  {subject}: {count} examples")

    # Use only a small subset to save memory
    max_samples = 50  # Limit to 50 samples to save memory
    if len(validation_dataset) > max_samples:
        print(f"Limiting validation dataset to {max_samples} samples (from {len(validation_dataset)})")
        validation_dataset = validation_dataset.select(range(max_samples))

    # Define preprocessing function
    def preprocess_function(examples):
        max_len = 128  # Longer sequence length for MMLU
        
        # MMLU format: question + options (A, B, C, D)
        prompts = []
        for question, options in zip(examples['question'], examples['choices']):
            prompt = f"{question}\n"
            for i, option in enumerate(options):
                prompt += f"{chr(65+i)}. {option}\n"  # A, B, C, D
            prompts.append(prompt)
        
        # Tokenize
        tokenized = tokenizer(prompts, padding="max_length", truncation=True, max_length=max_len)
        
        # Convert labels from strings (A, B, C, D) to indices (0, 1, 2, 3)
        label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        tokenized['labels'] = [label_map.get(label, 0) for label in examples['answer']]
        
        return tokenized

    # Apply preprocessing
    encoded_dataset = validation_dataset.map(preprocess_function, batched=True)

    # Show dataset info
    print(f"Dataset columns after processing: {encoded_dataset.column_names}")

    # Keep only the necessary columns
    columns_to_keep = ['input_ids', 'attention_mask', 'labels', 'subject']
    encoded_dataset = encoded_dataset.remove_columns(
        [col for col in encoded_dataset.column_names if col not in columns_to_keep]
    )
    encoded_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    print(f"Final dataset format: {encoded_dataset.format}")
    print(f"Dataset ready with {len(encoded_dataset)} examples")

    # Create DataLoader with minimal batch size
    dataloader = DataLoader(encoded_dataset, batch_size=batch_size)
    print(f"Created DataLoader with {len(dataloader)} batches")

    # Initialize tracker
    tracker = EnergyTracker(model, precision_mode=precision_mode)

    # Metrics
    key_metrics = []
    all_predictions = []
    all_true_labels = []
    all_subjects = []

    # Evaluate batches
    print(f"Evaluating {len(dataloader)} batches...")

    # Clear cache before starting
    clean_memory()

    # Process only a few batches for memory safety
    max_batch_eval = min(20, len(dataloader))

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batch_eval:
            print(f"Reached batch limit ({max_batch_eval}). Stopping early to save time.")
            break

        try:
            # Show batch info for first batch
            if batch_idx == 0:
                print(f"Batch keys: {batch.keys()}")
                for key in batch.keys():
                    if key != 'subject':  # subject is not a tensor
                        print(f"{key} shape: {batch[key].shape}")

            # Clean GPU memory before each batch
            clean_memory()

            # Process one sample at a time
            try:
                # Move data to GPU
                input_ids = batch['input_ids'].to('cuda')
                attention_mask = batch['attention_mask'].to('cuda')
                labels = batch['labels'].to('cuda')
                
                # Store subjects for analysis
                if 'subject' in batch:
                    subjects_in_batch = batch['subject']
                    all_subjects.extend(subjects_in_batch)

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
                'components': energy_metrics['components'],
                'subjects': subjects_in_batch if 'subject' in batch else []
            }
            key_metrics.append(batch_metrics)

            # Clean GPU memory after each batch
            clean_memory()

        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            continue

    # Calculate MMLU score (accuracy)
    if len(all_predictions) == 0 or len(all_true_labels) == 0:
        print("No valid predictions were made. Cannot calculate metrics.")
        return [], {"error": "No valid predictions"}

    try:
        # Overall accuracy
        score = accuracy_score(all_true_labels, all_predictions)
        print(f"MMLU overall accuracy: {score:.4f}")
        
        # Subject-wise accuracy if subjects are available
        subject_metrics = {}
        if all_subjects and len(all_subjects) == len(all_predictions):
            unique_subjects = set(all_subjects)
            for subject in unique_subjects:
                indices = [i for i, s in enumerate(all_subjects) if s == subject]
                subject_preds = [all_predictions[i] for i in indices]
                subject_labels = [all_true_labels[i] for i in indices]
                subject_acc = accuracy_score(subject_labels, subject_preds)
                subject_metrics[subject] = {
                    'accuracy': subject_acc,
                    'count': len(indices)
                }
                print(f"Subject: {subject}, Accuracy: {subject_acc:.4f}, Count: {len(indices)}")
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
        'benchmark': 'mmlu',
        'score': score,
        'total_energy': total_energy,
        'energy_per_token': total_energy / total_tokens if total_tokens > 0 else 0,
        'throughput': total_tokens / total_time if total_time > 0 else 0,
        'total_time': total_time,
        'total_tokens': total_tokens,
        'component_energy': component_energy,
        'subject_metrics': subject_metrics if 'subject_metrics' in locals() else {}
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


def test_quantized_models_on_mmlu(model_name, subjects=None, quantization_modes=['int4'], batch_size=1):
    """
    Test different quantization modes on MMLU benchmark

    Args:
        model_name: HuggingFace model name
        subjects: List of MMLU subjects to evaluate (None for all)
        quantization_modes: List of quantization modes to test
        batch_size: Batch size for evaluation

    Returns:
        Dictionary with results for each quantization mode
    """
    # Results dictionary
    results = {}

    # Get carbon intensity once
    carbon_intensity = get_carbon_intensity()
    print(f"Carbon intensity: {carbon_intensity} gCO2eq/kWh")

    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Test each quantization mode
    for mode in quantization_modes:
        try:
            print(f"\n----- Testing {mode.upper()} Mode -----")

            # Clear GPU memory
            clean_memory()

            # Load classifier model with 4 classes (A, B, C, D choices in MMLU)
            model = load_classifier(
                model_name=model_name,
                mode=mode,
                num_labels=4  # MMLU has 4 choices
            )

            # Create energy tracker
            precision = 'float16' if mode == 'fp16' else None
            tracker = EnergyTracker(model, precision_mode=precision)

            # Measure energy consumption
            print(f"Running MMLU benchmark with {mode} quantization...")
            _, metrics = run_mmlu_energy_monitoring(
                model=model,
                tokenizer=tokenizer,
                subjects=subjects,
                batch_size=batch_size,
                precision_mode=precision
            )

            # Check for errors
            if isinstance(metrics, dict) and metrics.get("error"):
                print(f"Error in metrics: {metrics['error']}")
                results[mode] = metrics
                continue

            # Calculate carbon footprint
            carbon_emissions = joules_to_co2(metrics['total_energy'], carbon_intensity)
            metrics['carbon_emissions'] = carbon_emissions

            # Save results
            results[mode] = metrics

            # Print results
            print(f"Total Energy: {metrics['total_energy']:.4f} J")
            print(f"Energy per token: {metrics['energy_per_token']:.6f} J/token")
            print(f"MMLU Score: {metrics['score']:.4f}")
            print(f"Carbon emissions: {carbon_emissions:.6f} gCO2eq")

            # Print component breakdown
            print("\nComponent Energy Breakdown:")
            total_comp = sum(metrics['component_energy'].values())
            for comp, energy in metrics['component_energy'].items():
                if energy > 0:  # Only show components with energy usage
                    percentage = 100 * energy / total_comp if total_comp > 0 else 0
                    print(f"  {comp}: {energy:.4f} J ({percentage:.1f}%)")

            # Clean up
            del model, tracker
            clean_memory()

        except Exception as e:
            print(f"Error testing {mode} mode: {e}")
            results[mode] = {"error": str(e)}

    # Compare efficiency if we have multiple modes
    modes_with_data = [m for m in quantization_modes if m in results and
                      isinstance(results[m], dict) and
                      'total_energy' in results[m]]

    if len(modes_with_data) >= 2:
        # Find highest energy mode as baseline
        baseline_mode = max(modes_with_data, key=lambda m: results[m]['total_energy'])
        baseline = results[baseline_mode]['total_energy']

        print(f"\n----- Efficiency Comparison -----")
        print(f"Using {baseline_mode.upper()} as baseline ({baseline:.4f} J)")

        for mode in modes_with_data:
            if mode != baseline_mode:
                savings = 100 * (baseline - results[mode]['total_energy']) / baseline
                results[mode]['energy_savings'] = savings
                print(f"{mode.upper()} saves {savings:.2f}% energy compared to {baseline_mode.upper()}")

    # Display summary table
    print("\n===== Summary Table =====")
    headers = ["Mode", "Energy (J)", "Time (s)", "Energy/Token (J)", "MMLU Score", "CO2 (gCO2eq)"]
    print(" | ".join(headers))
    print("-" * 80)

    for mode in quantization_modes:
        if mode in results and 'total_energy' in results[mode]:
            stats = results[mode]
            values = [
                mode.upper(),
                f"{stats['total_energy']:.4f}",
                f"{stats['total_time']:.3f}",
                f"{stats['energy_per_token']:.6f}",
                f"{stats['score']:.4f}",
                f"{stats.get('carbon_emissions', 0):.6f}"
            ]
            print(" | ".join(values))

    return results


def quick_test_mmlu(model_name, quant_mode='int4', subjects=None, max_samples=10):
    """Run a quick test for a single quantization mode on MMLU benchmark"""
    print(f"Quick test for {model_name} with {quant_mode} quantization")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model with 4 classes (A, B, C, D choices in MMLU)
    model = load_classifier(model_name, mode=quant_mode, num_labels=4)

    # Create energy tracker
    precision = 'float16' if quant_mode == 'fp16' else None
    tracker = EnergyTracker(model, precision_mode=precision)

    # Run MMLU benchmark with limited samples
    print(f"Running MMLU benchmark with {max_samples} samples...")
    _, metrics = run_mmlu_energy_monitoring(
        model=model,
        tokenizer=tokenizer,
        subjects=subjects,
        batch_size=1,
        precision_mode=precision
    )

    # Calculate carbon footprint
    carbon_intensity = get_carbon_intensity()
    carbon_emissions = joules_to_co2(metrics['total_energy'], carbon_intensity)
    metrics['carbon_emissions'] = carbon_emissions

    # Print results
    print("\nResults:")
    print(f"Total Energy: {metrics['total_energy']:.4f} J")
    print(f"Energy per token: {metrics['energy_per_token']:.6f} J/token")
    print(f"MMLU Score: {metrics['score']:.4f}")
    print(f"Inference time: {metrics['total_time']:.3f} s")
    print(f"Carbon emissions: {carbon_emissions:.6f} gCO2eq")

    # Component breakdown
    print("\nComponent Energy Breakdown:")
    total_comp = sum(metrics['component_energy'].values())
    for comp, energy in metrics['component_energy'].items():
        if energy > 0:  # Only show components with energy usage
            percentage = 100 * energy / total_comp if total_comp > 0 else 0
            print(f"  {comp}: {energy:.4f} J ({percentage:.1f}%)")

    # Clean up
    del model, tracker
    clean_memory()

    return metrics


if __name__ == "__main__":
    # Example usage
    model_name = "deepseek-ai/deepseek-coder-1.3b-base"
    subjects = ["high_school_mathematics", "high_school_physics"]
    
    # Run quick test
    quick_test_mmlu(
        model_name=model_name,
        quant_mode='int4',
        subjects=subjects,
        max_samples=10
    )
    
    # Run full test
    test_quantized_models_on_mmlu(
        model_name=model_name,
        subjects=subjects,
        quantization_modes=['int4', 'int8'],
        batch_size=1
    )