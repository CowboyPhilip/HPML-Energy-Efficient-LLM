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


def load_llm(model_name: str, mode: str = "fp32", device_map: str = "cuda"):
    """
    Load LLM with quantisation + optional attention-kernel replacement.
    `mode` pattern:
        fp16            → FP16 + vanilla attention
        fp16_flash-v2   → FP16 + Flash-Attention 2
        fp16_flash-v3   → FP16 + Flash-Attention 3
        fp16_paged      → FP16 + vLLM engine
        int8_flash-v2   → INT8 + Flash-Attention 2
        int8_flash-v3   → INT8 + Flash-Attention 3
        int8_paged      → INT8 + vLLM paged-attention engine
        int4_paged      → INT4 + vLLM paged-attention engine
        int4_flash-v2      → INT4 + vLLM paged-attention engine
        int4_flash-v2      → INT4 + vLLM paged-attention engine
    """
    q_mode, kernel = _parse_mode(mode)

    # ===== 1) quantization =====
    clean_memory();  
    print(f"Loading {q_mode.upper()} model …");  
    print_gpu_memory()
    common = dict(
        device_map=device_map,            # "auto" or "cuda"
        offload_folder="offload",
        low_cpu_mem_usage=True
    )
    offload = False if device_map=="cuda" else True

    if q_mode == "fp32":
        # Full precision (no quantization)
        if kernel == "flash-v2":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                attn_implementation="flash_attention_2",
                offload_state_dict=offload,
                max_memory={0: "30GB"},
                **common
            )
        elif kernel == "vanilla":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                offload_state_dict=offload,
                max_memory={0: "30GB"},
                **common
            )
        else:
            raise ValueError(f"Unsupported kernel option: {kernel} for FP32 mode")

    elif q_mode == "fp16":
        if kernel=="flash-v2":
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16,
                attn_implementation="flash_attention_2",
                offload_state_dict=offload, max_memory={0: "30GB"}, **common
            )
        elif kernel == "vanilla":
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16,
                offload_state_dict=offload, max_memory={0: "30GB"}, **common
            )
        else:
            raise ValueError(f"Unknown kernel option: {kernel}")

    elif q_mode == "int8":
        bnb = BitsAndBytesConfig(load_in_8bit=True,
                         llm_int8_enable_fp32_cpu_offload=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb, torch_dtype=torch.float16,
            offload_state_dict=offload, max_memory={0: "30GB"}, **common
        )
        if kernel=="flash-v2":
            model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=bnb, 
                attn_implementation="flash_attention_2",
                torch_dtype=torch.float16, **common
            )
        elif kernel == "vanilla":
            model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=bnb, torch_dtype=torch.float16, **common
            )
        else:
            raise ValueError(f"Unknown kernel option: {kernel}")

    elif q_mode == "int4":
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16,
                                 bnb_4bit_use_double_quant=True,
                                 llm_int8_enable_fp32_cpu_offload=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb,
            offload_state_dict=offload, max_memory={0: "30GB"}, **common
        )
        if kernel=="flash-v2":
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                attn_implementation="flash_attention_2",
                quantization_config=bnb, **common
            )
        elif kernel == "vanilla":
            model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=bnb, **common
            )
        else:
            raise ValueError(f"Unknown kernel option: {kernel}")

    else:
        raise ValueError(f"Unsupported quantisation mode: {q_mode}")

    # ===== 2) Attention-kernel  =====
    if kernel == "flash-v2":
        # model = apply_flash_attention(model)   
        pass
    elif kernel == "flash-v3":
        model = apply_flash_attention(model)  # ← Flash-Attn 3
    elif kernel == "paged":
        # Return a lightweight wrapper around vLLM engine (uses paged attention)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = build_vllm_wrapper(model_name, tokenizer, q_mode)
    elif kernel != "vanilla":
        raise ValueError(f"Unknown kernel option: {kernel}")

    print(f"Model ready → quantisation: {q_mode.upper()}, kernel: {kernel}")
    print_gpu_memory()
    return model


def load_classifier(model_name: str, mode: str = "fp16", num_labels: int = 2, device_map: str = "cuda"):
    """
    Load a SequenceClassification model with quantisation + memory optimizations.
    """
    q_mode, kernel = _parse_mode(mode)

    clean_memory();  print(f"Loading {q_mode.upper()} classifier …");  print_gpu_memory()
    common = dict(device_map="auto", offload_folder="offload", low_cpu_mem_usage=True)

    if q_mode == "fp32":
        classifier = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, torch_dtype=torch.float32, **common
        )
    elif q_mode == "fp16":
        classifier = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, torch_dtype=torch.float16,
            offload_state_dict=True, max_memory={0:"30GB"}, **common
        )
    elif q_mode == "int8":
        bnb = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
        classifier = AutoModelForSequenceClassification.from_pretrained(
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
        classifier = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, quantization_config=bnb,
            torch_dtype=torch.float16, **common
        )
    else:
        raise ValueError(f"Unsupported quant mode: {q_mode}")

    print_gpu_memory()
    return classifier

