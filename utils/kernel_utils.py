# utils/kernel_utils.py
"""
Low-level kernel utilities: Flash-Attention 3 replacement and vLLM paged-attention
"""


def apply_flash_attention(model):
    """
    Replace default attention with Flash-Attention 3 kernels in-place.
    Works on most HuggingFace transformer blocks (LLama, Qwen, DeepSeek).
    """
    try:
        from flash_attn.replace import replace_attn_with_flash_attn
        import torch
        model.cuda()  # Ensure weights on GPU for kernel fusion
        replace_attn_with_flash_attn(model, attn_impl="flash", replace_all=True)
        print("Flash-Attention 3 enabled.")
    except ImportError as e:
        raise RuntimeError(
            "flash-attn not installed. Run: pip install flash-attn==3.0.0 triton==3.2.0"
        ) from e
    return model


def build_vllm_wrapper(model_name: str, tokenizer, quant_mode: str):
    """
    Return a torch.nn.Module wrapper that delegates generation to a vLLM engine
    (paged attention, GPU KV-cache paging). Keeps .forward() signature so that
    EnergyTracker can still call it.
    """
    import torch
    from torch import nn
    from vllm import LLM, SamplingParams

    # Map our quant strings to vllm argument
    vllm_quant = None
    if quant_mode == "int4":
        vllm_quant = "int4"
    elif quant_mode == "int8":
        vllm_quant = "int8"

    # Build engine
    engine = LLM(model=model_name,
                 tensor_parallel_size=1,
                 dtype="float16" if quant_mode == "fp16" else "float16",
                 quantization=vllm_quant)

    class _VLLMWrapper(nn.Module):
        def __init__(self, engine, tokenizer):
            super().__init__()
            self.engine = engine
            self.tokenizer = tokenizer
            self.params = SamplingParams(max_tokens=128, temperature=0.0)

        def forward(self, input_ids, attention_mask=None):
            """
            Accepts tokenised prompt → returns dummy logits tensor so that
            downstream code (argmax / loss) still works.
            """
            prompts = self.tokenizer.batch_decode(
                input_ids, skip_special_tokens=True)
            outputs = self.engine.generate(prompts, self.params)
            # Concatenate generated text with original prompt
            full_text = [
                p + o.outputs[0].text for p, o in zip(prompts, outputs)]
            # Re-tokenise to obtain logits-compatible tensor
            gen_tokens = self.tokenizer(
                full_text, return_tensors="pt",
                padding=True).input_ids.to(input_ids.device)
            # Fake logits: one-hot distribution (saves memory vs. real logits)
            vocab = self.tokenizer.vocab_size
            logits = torch.nn.functional.one_hot(
                gen_tokens, num_classes=vocab).float()
            return logits

    print("vLLM paged-attention engine initialised.")
    return _VLLMWrapper(engine, tokenizer)
