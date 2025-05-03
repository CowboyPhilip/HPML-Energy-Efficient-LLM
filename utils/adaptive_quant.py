#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive precision generator:
  • Pre-loads two models (high-precision & low-precision)
  • Switches model on-the-fly based on context length and observed latency
  • Measures energy, latency, VRAM via existing EnergyTracker
"""

import time
import numpy as np
import torch
from transformers import AutoTokenizer

from utils.load_llm import load_llm        # enhanced loader (fp16_flash, int8_flash …)
from utils.energy_utils import EnergyTracker
from utils.memory_utils import print_gpu_memory

# -------------------------------------------------------------------
# Monkey-patch EnergyTracker:
# 1) support `with tracker:` context manager
# 2) add save_results(extra_metrics) method
# -------------------------------------------------------------------
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
    # populate stats
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
    # merge extra metrics into stats
    self.stats.update(extra_metrics)

# attach to EnergyTracker
EnergyTracker.__enter__      = _et_enter
EnergyTracker.__exit__       = _et_exit
EnergyTracker.save_results   = _save_results
# -------------------------------------------------------------------

class AdaptiveQuantGenerator:
    """
    Switch rule (default):
        - if current context length > ctx_threshold → low-precision
        - else if avg latency / token > latency_threshold → low-precision
        otherwise keep high-precision
    """

    def __init__(
        self,
        model_name: str,
        high_mode: str = "fp16_flash",
        low_mode: str = "int8_flash",
        ctx_threshold: int = 1024,
        latency_threshold: float = 0.08,   # seconds per token
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.high_model = load_llm(model_name, high_mode)
        self.low_model  = load_llm(model_name, low_mode)
        self.active_model = self.high_model

        self.ctx_threshold     = ctx_threshold
        self.latency_threshold = latency_threshold

        self.tok_count = 0
        self.lat_sum   = 0.0

    # ---------- internal helpers ---------- #
    def _maybe_switch(self, cur_ctx_len: int):
        """Decide whether to switch active model."""
        avg_lat = (self.lat_sum / self.tok_count) if self.tok_count else 0.0
        need_low = (cur_ctx_len > self.ctx_threshold) or (avg_lat > self.latency_threshold)
        self.active_model = self.low_model if need_low else self.high_model

    def _step_latency(self, start_t: float):
        """Update latency stats after each generation step."""
        self.lat_sum  += time.time() - start_t
        self.tok_count += 1

    # ---------- public API ---------- #
    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """
        Token-by-token generation so we can monitor latency and switch precision.
        Returns full generated text.
        """
        # ---------------- priming ---------------- #
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        self._maybe_switch(input_ids.size(1))

        tracker = EnergyTracker("adaptive_quant")
        with tracker:
            for _ in range(max_new_tokens):
                t0 = time.time()
                # generate one token; greedy decoding for determinism
                out_ids = self.active_model.generate(
                    input_ids,
                    max_new_tokens=1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                self._step_latency(t0)

                # append new token & maybe switch model
                input_ids = out_ids
                self._maybe_switch(input_ids.size(1))

        # attach extra metrics
        tracker.save_results({
            "avg_latency_s_per_tok": self.lat_sum / max(1, self.tok_count),
            "tokens_generated":      self.tok_count,
            "ctx_threshold":         self.ctx_threshold,
            "latency_threshold":     self.latency_threshold,
        })
        print_gpu_memory()
        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)


# ---------------- quick CLI ---------------- #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-ai/deepseek-coder-1.3b-instruct")
    parser.add_argument("--prompt", default="Explain the quick-sort algorithm.")
    parser.add_argument("--tokens", type=int, default=128)
    args = parser.parse_args()

    gen  = AdaptiveQuantGenerator(model_name=args.model)
    text = gen.generate(args.prompt, max_new_tokens=args.tokens)
    print("\n=== Generated text ===\n", text)
