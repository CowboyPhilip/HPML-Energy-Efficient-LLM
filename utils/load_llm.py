# utils/load_llm.py
from utils.memory_utils import clean_memory, print_gpu_memory
from utils.kernel_utils import apply_flash_attention, build_vllm_wrapper  # NEW
from transformers import (
    AutoModelForCausalLM, AutoModelForSequenceClassification,
    AutoTokenizer, BitsAndBytesConfig
)
import torch, os, re


def _parse_mode(mode: str):
    """
    Split a mode string like 'int8_flash' → ('int8', 'flash')
    Defaults to ('fp16', 'vanilla')
    """
    parts = mode.lower().split("_", 1)
    base = parts[0] if parts else "fp16"
    kernel = parts[1] if len(parts) == 2 else "vanilla"
    return base, kernel


def load_llm(model_name: str, mode: str = "fp16"):
    """
    Load LLM with quantisation + optional attention-kernel replacement.
    `mode` pattern:
        fp16            → FP16 + vanilla attention   (原逻辑)
        int8_flash      → INT8 + Flash-Attention 3
        int4_paged      → INT4 + vLLM paged-attention engine
        fp16_flash      → FP16 + Flash-Attention 3
        fp16_paged      → FP16 + vLLM engine
    """
    q_mode, kernel = _parse_mode(mode)

    # ===== 1) 原有量化分支 =====
    clean_memory();  print(f"Loading {q_mode.upper()} model …");  print_gpu_memory()
    common = dict(device_map="auto", offload_folder="offload", low_cpu_mem_usage=True)

    if q_mode == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16,
            offload_state_dict=True, max_memory={0: "30GB"}, **common
        )
    elif q_mode == "int8":
        bnb = BitsAndBytesConfig(load_in_8bit=True,
                                 llm_int8_enable_fp32_cpu_offload=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb, torch_dtype=torch.float16, **common
        )
    elif q_mode == "int4":
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16,
                                 bnb_4bit_use_double_quant=True,
                                 llm_int8_enable_fp32_cpu_offload=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb, **common
        )
    else:
        raise ValueError(f"Unsupported quantisation mode: {q_mode}")

    # ===== 2) Attention-kernel 替换 =====
    if kernel == "flash":
        model = apply_flash_attention(model)   # ← Flash-Attn 3 替换
    elif kernel == "paged":
        # Return a lightweight wrapper around vLLM engine (uses paged attention)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = build_vllm_wrapper(model_name, tokenizer, q_mode)
    elif kernel != "vanilla":
        raise ValueError(f"Unknown kernel option: {kernel}")

    print(f"Model ready → quantisation: {q_mode.upper()}, kernel: {kernel}")
    print_gpu_memory()
    return model

def load_classifier(model_name: str, mode: str = "fp16", num_labels: int = 2):
    """
    Load a SequenceClassification model with quantisation + memory optimizations.
    """
    q_mode, kernel = _parse_mode(mode)
    clean_memory(); print(f"Loading classifier {q_mode.upper()} …"); print_gpu_memory()
    common = dict(device_map="auto", offload_folder="offload", low_cpu_mem_usage=True)
    from transformers import AutoModelForSequenceClassification
    if q_mode == "fp16":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, torch_dtype=torch.float16,
            offload_state_dict=True, max_memory={0:"30GB"}, **common
        )
    elif q_mode == "int8":
        bnb = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, quantization_config=bnb,
            torch_dtype=torch.float16, **common
        )
    elif q_mode == "int4":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, quantization_config=bnb, **common
        )
    else:
        raise ValueError(f"Unsupported quant mode: {q_mode}")
    print_gpu_memory()
    return model

