#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AdaptiveQuantGenerator: single-shot adaptive quantization based on prompt length.
- Pre-loads two quantized models (high and low precision)
- Uses EnergyTracker.measure_generation() for energy/tokenization/inference measurement
- Dynamically switches precision per example
"""
import torch
from transformers import AutoTokenizer

from utils.load_llm import load_llm, _parse_mode
from utils.energy_utils import EnergyTracker
from utils.memory_utils import print_gpu_memory

class AdaptiveQuantGenerator:
    """
    Adaptive quant: choose precision per prompt length and measure energy exactly like EnergyTracker.measure_generation.
    """
    def __init__(
        self,
        model_name: str,
        high_mode: str = "fp16_vanilla",
        low_mode:  str = "int8_vanilla",
        ctx_threshold: int = 256,
        device_map: str = "auto"
    ):
        # thresholds and tokenizer
        self.ctx_threshold = ctx_threshold
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        # load models
        self.high_model = load_llm(model_name, high_mode, device_map=device_map)
        self.low_model  = load_llm(model_name, low_mode,  device_map=device_map)
        # create trackers
        qh, _ = _parse_mode(high_mode)
        ql, _ = _parse_mode(low_mode)
        self.high_tracker = EnergyTracker(self.high_model, precision_mode=qh)
        self.low_tracker  = EnergyTracker(self.low_model,  precision_mode=ql)

    @torch.inference_mode()
    def evaluate(
        self,
        prompt: str,
        tokenizer: AutoTokenizer,
        max_new_tokens: int = 128,
        temperature: float = 0.5,
        top_p: float = 0.9
    ):
        """
        Run generation and measure energy exactly as measure_generation.
        Returns (generated_ids, None, stats).
        """
        # decide mode
        length = len(self.tokenizer(prompt, return_tensors="pt").input_ids[0])
        if length > self.ctx_threshold:
            tracker = self.low_tracker
            model = self.low_model
        else:
            tracker = self.high_tracker
            model = self.high_model
        # measure via EnergyTracker.measure_generation
        gen_ids, stats = tracker.measure_generation(
            prompt,
            tokenizer,
            temperature=temperature,
            top_p=top_p
        )
        # optional memory stats
        print_gpu_memory()
        return gen_ids, None, stats

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.5,
        top_p: float = 0.9
    ) -> str:
        """
        Single-shot generation with built-in measurement.
        Returns decoded string.
        """
        # call evaluate and decode
        gen_ids, _, stats = self.evaluate(
            prompt,
            self.tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
        input_len = len(self.tokenizer(prompt, return_tensors="pt").input_ids[0])
        tokens = gen_ids[0, input_len:]
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

# end of adaptive_quant.py
