# utils/test_mmlu.py

from datasets import load_dataset
from tqdm import tqdm
import torch
import accelerate
from utils.energy_utils import EnergyTracker, get_carbon_intensity, joules_to_co2
from utils.load_llm import load_llm
from utils.memory_utils import clean_memory
from transformers import AutoTokenizer


def quick_test_mmlu(
    model_name,
    quant_mode,
    subjects=None,
    max_samples=50
):
    """
    Quickly sample up to max_samples per subject,
    measure energy, CO2, and accuracy for one quant_mode.
    """
    carbon_intensity = get_carbon_intensity()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    results = {}
    subjects = subjects or load_dataset("mmlu").config_names

    for subj in subjects:
        ds = load_dataset("cais/mmlu", subj, split="validation")
        ds = ds.select(range(min(len(ds), max_samples)))

        clean_memory()
        model = load_llm(model_name, mode=quant_mode, device_map="cuda")
        tracker = EnergyTracker(model)

        correct = total_tokens = total_energy = total_time = 0

        for item in tqdm(ds, desc=f"Quick MMLU [{subj}]"):
            prompt_body = (
                f"You are an expert Python programmer, and here is your task: "
                f"{item['question']} Choices: {', '.join(item['choices'])}. "
                f"Select the correct answer by writing ONLY the choice text (no explanation)."
            )
            prompt = f"<｜begin▁of▁sentence｜><｜User｜>{prompt_body}<｜Assistant｜><think>"

            # Generate with energy tracking
            generated_ids, stats = tracker.measure_generation(prompt, tokenizer, temperature=0, top_p=0.9)
            text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

            # Filter output to match choices
            matched_choice = None
            for choice in item['choices']:
                if choice.lower() in text.lower():
                    matched_choice = choice
                    break
            if matched_choice is not None:
                text = matched_choice
            else:
                print(f"[WARN] Prediction did not match any choice: '{text}' → keeping raw output")

            # Get correct answer
            answer = item["answer"]
            if isinstance(answer, int):
                answer = item["choices"][answer]
            answer = str(answer).strip()

            is_corr = (text == answer)

#             # 🌟 Debug output
#             print(f"\n[DEBUG] Subject: {subj}")
#             print(f"Question: {item['question']}")
#             print(f"Choices: {item['choices']}")
#             print(f"Predicted: {text}")
#             print(f"Actual: {answer}")
#             print(f"Correct? {'✅' if is_corr else '❌'}")
                
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

        print("summary", summary[mode])
    return summary
