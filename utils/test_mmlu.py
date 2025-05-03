# utils/test_mmlu.py

from datasets import load_dataset
from tqdm import tqdm
import torch

from utils.energy_utils import EnergyTracker, get_carbon_intensity, joules_to_co2
from utils.load_llm import load_llm
from utils.memory_utils import clean_memory

def quick_test_mmlu(
    model_name,
    quant_mode,
    subjects=None,
    max_samples=50
):
    """
    Quickly sample up to max_samples per subject,
    measure energy & accuracy for one quant_mode.
    """
    carbon_intensity = get_carbon_intensity()
    tokenizer = load_llm.tokenizer_from_pretrained(model_name)
    results = {}
    subjects = subjects or load_dataset("mmlu").config_names

    for subj in subjects:
        ds = load_dataset("mmlu", subj, split="validation")
        ds = ds.select(range(min(len(ds), max_samples)))

        clean_memory()
        model = load_llm(model_name, mode=quant_mode)
        tracker = EnergyTracker(model)

        correct = total_tokens = total_energy = total_time = 0

        for item in tqdm(ds, desc=f"Quick MMLU [{subj}]"):
            prompt = item["question"] + "\nChoices: " + \
                     ", ".join(item["choices"]) + "\nAnswer:"
            logits, stats = tracker.measure_text(prompt, tokenizer)
            pred = torch.argmax(logits, dim=-1)
            text = tokenizer.batch_decode(pred, skip_special_tokens=True)[0].strip()

            is_corr = (text == item["answer"].strip())
            correct += int(is_corr)
            total_tokens += stats.get("num_tokens", 1)
            total_energy += stats["total_energy"]
            total_time += stats["time"]

        count = len(ds)
        acc = 100 * correct / count
        co2 = joules_to_co2(total_energy, carbon_intensity)

        results[subj] = {
            "samples": count,
            "accuracy": acc,
            "total_energy": total_energy,
            "total_time": total_time,
            "energy_per_token": total_energy / total_tokens if total_tokens else 0,
            "carbon_emissions": co2
        }

        del model, tracker
        clean_memory()

    return results


def test_quantized_models_on_mmlu(
    model_name,
    quantization_modes,
    subjects=None
):
    """
    For each quant mode, run full validation
    across all selected subjects.
    """
    summary = {}
    for mode in quantization_modes:
        stats = quick_test_mmlu(
            model_name=model_name,
            quant_mode=mode,
            subjects=subjects,
            max_samples=None  # None => use full split
        )
        # summarize across subjects
        all_energy = sum(v["total_energy"] for v in stats.values())
        all_time   = sum(v["total_time"] for v in stats.values())
        all_tokens = sum(v["energy_per_token"] * v["samples"] for v in stats.values())
        all_correct= sum(v["accuracy"] * v["samples"]/100 for v in stats.values())
        all_samples= sum(v["samples"] for v in stats.values())

        summary[mode] = {
            "subjects": stats,
            "total_energy": all_energy,
            "total_time": all_time,
            "energy_per_token": all_energy / all_tokens if all_tokens else 0,
            "accuracy": 100 * all_correct / all_samples,
        }

    return summary
