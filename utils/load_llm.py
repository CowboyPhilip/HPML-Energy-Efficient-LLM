# utils/load_llm.py
from utils.memory_utils import clean_memory, print_gpu_memory
from utils.kernel_utils import apply_flash_attention, build_vllm_wrapper
from transformers import (
    AutoModelForCausalLM, AutoModelForSequenceClassification,
    AutoTokenizer, BitsAndBytesConfig
)
import torch


def _parse_mode(mode: str):
    """
    Parse mode string into quantisation mode and kernel type.
    Returns a tuple: (q_mode, kernel).
    """
    parts = mode.lower().split("_", 1)
    q_mode = parts[0]
    kernel = parts[1] if len(parts) > 1 else "vanilla"
    if q_mode not in ("fp32","fp16", "int8", "int4"):
        raise ValueError(f"Unsupported quantisation mode: {q_mode}")
    if kernel not in ("vanilla", "flash-v2", "flash-v3", "paged"):
        raise ValueError(f"Unknown kernel option: {kernel}")
    return q_mode, kernel


def load_llm(model_name: str, mode: str = "fp16", device_map: str = "cuda"):
    """
    Load LLM with quantisation + optional attention-kernel replacement.
    device_map: "auto" for CPU/GPU offload, "cuda" for full GPU.
    """
    q_mode, kernel = _parse_mode(mode)

    # ===== 1) quantization =====
    clean_memory()
    print(f"Loading {q_mode.upper()} model …")
    print_gpu_memory()

    # Common args
    common = dict(
        device_map=device_map,
        offload_folder="offload",
        low_cpu_mem_usage=True
    )
    offload = False if device_map == "cuda" else True
    max_mem = {0: "30GB"}

    # Configure kwargs
    if q_mode == "fp32":
        kwargs = dict(
            torch_dtype=torch.float32,
            offload_state_dict=offload,
            max_memory=max_mem,
            **common
        )
    elif q_mode == "fp16":
        kwargs = dict(
            torch_dtype=torch.float16,
            offload_state_dict=offload,
            max_memory=max_mem,
            **common
        )
    elif q_mode == "int8":
        bnb = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True
        )
        kwargs = dict(
            quantization_config=bnb,
            torch_dtype=torch.float16,
            offload_state_dict=offload,
            max_memory=max_mem,
            **common
        )
    elif q_mode == "int4":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True
        )
        kwargs = dict(
            quantization_config=bnb,
            torch_dtype=torch.float16,
            offload_state_dict=offload,
            max_memory=max_mem,
            **common
        )
    else:
        raise ValueError(f"Unsupported quantisation mode: {q_mode}")

    # Add attention kernel if needed
    if kernel == "flash-v2":
        kwargs['attn_implementation'] = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    # ===== 2) Attention-kernel post-processing =====
    if kernel == "flash-v3":
        model = apply_flash_attention(model)  # Flash-Attn v3
    elif kernel == "paged":
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = build_vllm_wrapper(model_name, tokenizer, q_mode)
    # vanilla and flash-v2 handled above

    print(f"Model ready → quantisation: {q_mode.upper()}, kernel: {kernel}")
    print_gpu_memory()
    return model


def load_classifier(
    model_name: str,
    mode: str = "fp16",
    num_labels: int = 2,
    device_map: str = "auto"
):
    """
    Load a SequenceClassification model with quantisation + memory optimizations.
    device_map: "auto" or "cuda".
    """
    q_mode, _ = _parse_mode(mode)
    clean_memory()
    print(f"Loading {q_mode.upper()} classifier …")
    print_gpu_memory()

    common = dict(
        device_map=device_map,
        offload_folder="offload",
        low_cpu_mem_usage=True
    )
    offload = False if device_map == "cuda" else True
    max_mem = {0: "30GB"}

    if q_mode == "fp32":
        classifier = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            torch_dtype=torch.float32,
            offload_state_dict=offload,
            max_memory=max_mem,
            **common
        )
    elif q_mode == "fp16":
        classifier = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            torch_dtype=torch.float16,
            offload_state_dict=offload,
            max_memory=max_mem,
            **common
        )
    elif q_mode == "int8":
        bnb = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True
        )
        classifier = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            quantization_config=bnb,
            torch_dtype=torch.float16,
            offload_state_dict=offload,
            max_memory=max_mem,
            **common
        )
    elif q_mode == "int4":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True
        )
        classifier = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            quantization_config=bnb,
            torch_dtype=torch.float16,
            offload_state_dict=offload,
            max_memory=max_mem,
            **common
        )
    else:
        raise ValueError(f"Unsupported quant mode: {q_mode}")

    print_gpu_memory()
    return classifier
