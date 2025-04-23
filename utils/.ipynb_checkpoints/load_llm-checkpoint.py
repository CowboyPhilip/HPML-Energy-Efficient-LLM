from utils.memory_utils import clean_memory, print_gpu_memory
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding
)
import torch
def load_llm(model_name, mode):
    """
    Load LLM with specified quantization mode and memory optimization

    Args:
        model_name: HuggingFace model name
        mode: 'fp16', 'int8', or 'int4'

    Returns:
        Loaded model
    """
    # Free memory before loading
    clean_memory()

    print(f"Starting to load model in {mode.upper()} mode...")
    print_gpu_memory()

    # Common parameters for all loading methods
    common = {
        "device_map": "auto",
        "offload_folder": "offload",
        "low_cpu_mem_usage": True
    }

    if mode == 'fp16':
        # For FP16, we need aggressive memory offloading
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            offload_state_dict=True,  # Enable CPU offloading
            max_memory={0: "30GB"},  # Limit GPU memory usage
            **common
        )
    elif mode == 'int8':
        # 8-bit quantization
        bnb = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_skip_modules=None,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb,
            torch_dtype=torch.float16,
            **common
        )
    elif mode == 'int4':
        # 4-bit quantization
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True  # Important for INT4 too
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb,
            **common
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    print(f"Model loaded successfully in {mode.upper()} mode")
    print_gpu_memory()
    return model

def load_classifier(model_name, mode, num_labels=2):
    """
    Load a classifier model with specified quantization mode

    Args:
        model_name: HuggingFace model name
        mode: 'fp16', 'int8', or 'int4'
        num_labels: Number of output classes

    Returns:
        Loaded model
    """
    # Free memory before loading
    clean_memory()

    print(f"Starting to load classifier in {mode.upper()} mode...")
    print_gpu_memory()

    # Common parameters for all loading methods
    common = {
        "num_labels": num_labels,
        "device_map": "auto",
        "offload_folder": "offload",
        "low_cpu_mem_usage": True
    }

    if mode == 'fp16':
        # For FP16, we need aggressive memory offloading
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            offload_state_dict=True,  # Enable CPU offloading
            max_memory={0: "30GB"},  # Limit GPU memory usage
            **common
        )
    elif mode == 'int8':
        # 8-bit quantization
        bnb = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            quantization_config=bnb,
            torch_dtype=torch.float16,
            **common
        )
    elif mode == 'int4':
        # 4-bit quantization
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True  # Important for INT4 too
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            quantization_config=bnb,
            **common
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    print(f"Classifier loaded successfully in {mode.upper()} mode")
    print_gpu_memory()
    return model