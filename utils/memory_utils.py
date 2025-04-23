import torch
import os
import numpy as np
import gc

def clean_memory():
    """Free unused memory and run garbage collection"""
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

def print_gpu_memory():
    """Print current GPU memory usage"""
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    max_allocated = torch.cuda.max_memory_allocated() / 1e9

    print(f"GPU Memory: Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB | Max: {max_allocated:.2f} GB")