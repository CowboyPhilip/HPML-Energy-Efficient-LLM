#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AdaptiveQuantGenerator: single-shot adaptive quantization based on prompt length.
- Pre-loads two quantized models (high and low precision)
- Instantiates one EnergyTracker per precision once
- Uses one-shot .generate() to minimize monitor overhead
- Selects tracker/model per example, reusing monitors
"""
import time
import numpy as np
import torch
from transformers import AutoTokenizer

from utils.load_llm import load_llm, _parse_mode
from utils.energy_utils import EnergyTracker
from utils.memory_utils import print_gpu_memory

# Monkey-patch EnergyTracker to context manager and save_results

def _et_enter(self):
    if getattr(self, 'zeus', None):
        try:
            self.zeus.begin_window('inference')
            self.active_windows.add('inference')
        except:
            pass
    self._enter_ts = time.time()
    return self

def _et_exit(self, exc_type, exc_val, exc_tb):
    end_ts = time.time()
    inf_e = 0
    if getattr(self, 'zeus', None) and 'inference' in self.active_windows:
        try:
            m = self.zeus.end_window('inference')
            inf_e = m.total_energy
            self.active_windows.remove('inference')
        except:
            pass
    elapsed = end_ts - getattr(self, '_enter_ts', end_ts)
    comp = {k: np.sum(v) for k, v in self.comp_energy.items()}
    self.stats = {
        'total_energy': inf_e,
        'time': elapsed,
        'components': comp,
        'num_tokens': None
    }
    return False

def _save_results(self, extra_metrics):
    if not hasattr(self, 'stats'):
        self.stats = {}
    self.stats.update(extra_metrics)

EnergyTracker.__enter__    = _et_enter
EnergyTracker.__exit__     = _et_exit
EnergyTracker.save_results = _save_results


class AdaptiveQuantGenerator:
    """
    Adaptive quant: choose static precision per example based on prompt length.
    """
    def __init__(
        self,
        model_name: str,
        high_mode: str = "fp16_vanilla",
        low_mode:  str = "int8_vanilla",
        ctx_threshold: int = 512,
        latency_threshold: float = 0.08,
        device_map: str = "auto"
    ):
        # thresholds
        self.ctx_threshold     = ctx_threshold
        self.latency_threshold = latency_threshold
        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        # load two quantized models
        self.high_mode = high_mode
        self.low_mode  = low_mode
        self.high_model = load_llm(model_name, high_mode, device_map=device_map)
        self.low_model  = load_llm(model_name, low_mode,  device_map=device_map)
        # parse precision modes
        self.high_q_mode, _ = _parse_mode(high_mode)
        self.low_q_mode,  _ = _parse_mode(low_mode)
        # initialize trackers once
        self.high_tracker = EnergyTracker(self.high_model, precision_mode=self.high_q_mode)
        self.low_tracker  = EnergyTracker(self.low_model,  precision_mode=self.low_q_mode)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = None,
        top_p: float = None
    ) -> str:
        """
        One-shot generation: choose precision by prompt length, reuse trackers.
        """
        # tokenize with mask
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        prompt_len = inputs.input_ids.size(1)

        # pick model and tracker
        if prompt_len > self.ctx_threshold:
            model, tracker = self.low_model, self.low_tracker
        else:
            model, tracker = self.high_model, self.high_tracker
        # move inputs to device
        for k, v in inputs.items():
            inputs[k] = v.to(model.device)

        # run generation under one monitor
        with tracker:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature is not None),
                temperature=temperature or 1.0,
                top_p=top_p or 1.0,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        # compute and save metrics
        new_toks = outputs.sequences.size(1) - prompt_len
        tracker.save_results({
            'tokens_generated':    new_toks,
            'ctx_threshold':       self.ctx_threshold,
            'latency_threshold':   self.latency_threshold
        })
        # optional memory print
        print_gpu_memory()
        # decode and return
        return self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

    @torch.inference_mode()
    def evaluate(
        self,
        prompt: str,
        tokenizer: AutoTokenizer,
        max_new_tokens: int = 256,
        temperature: float = 0.5,
        top_p: float = 0.9
    ):
        """
        One-shot evaluation: returns (ids, logits, stats).
        """
        # tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        prompt_len = inputs.input_ids.size(1)
        # pick model and tracker
        if prompt_len > self.ctx_threshold:
            model, tracker = self.low_model, self.low_tracker
        else:
            model, tracker = self.high_model, self.high_tracker
        # move inputs to device
        for k, v in inputs.items():
            inputs[k] = v.to(model.device)
        # run under one monitor
        with tracker:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature>0),
                temperature=temperature,
                top_p=top_p,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.eos_token_id
            )
        seqs = outputs.sequences
        logits = outputs.scores
        new_toks = seqs.size(1) - prompt_len
        tracker.save_results({
            'num_tokens':        new_toks,
            'input_length':      prompt_len,
            'ctx_threshold':     self.ctx_threshold,
            'latency_threshold': self.latency_threshold
        })
        return seqs, logits, tracker.stats

# end of adaptive_quant.py
