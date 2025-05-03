from zeus.monitor import ZeusMonitor
import torch
import torch.nn as nn
from functools import partial
import numpy as np
import time
import geocoder

class EnergyTracker:
    def __init__(self, model, precision_mode=None):
        """
        Initialize energy tracker for a model

        Args:
            model: GPU nn.Module
            precision_mode: None|'float16'
        """
        self.model = model
        self.precision_mode = precision_mode
        self.active_windows = set()  # Track active measurement windows

        # Initialize ZeusMonitor to measure GPU energy
        try:
            self.zeus = ZeusMonitor(
                approx_instant_energy=True,
                gpu_indices=[torch.cuda.current_device()]
            )
            print("Successfully initialized ZeusMonitor")
        except Exception as e:
            print(f"Error initializing ZeusMonitor: {e}")
            print("Falling back to simple time measurement without energy tracking")
            self.zeus = None

        # Energy consumption buckets
        self.comp_energy = {
            'embeddings': [],
            'attention': [],
            'ffn': [],
            'layernorm': [],
            'output_layer': []
        }

        # Register hooks if ZeusMonitor is available
        if self.zeus is not None:
            self._register_hooks()
        else:
            print("Skipping hook registration since ZeusMonitor is not available")

    def _register_hooks(self):
        """Register hooks based on model type; fallback to Linear/Conv1D if unknown."""
        modules_to_hook = []

        # GPT-style causal LM
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            base = self.model.model
            # embeddings
            if hasattr(base, 'embed_tokens'):
                modules_to_hook.append((base.embed_tokens, 'embeddings'))
            # each layer
            for layer in base.layers:
                if hasattr(layer, 'self_attn'):
                    modules_to_hook.append((layer.self_attn, 'attention'))
                if hasattr(layer, 'mlp'):
                    modules_to_hook.append((layer.mlp, 'ffn'))
            # final norm
            if hasattr(base, 'norm'):
                modules_to_hook.append((base.norm, 'layernorm'))
            # lm_head
            if hasattr(self.model, 'lm_head'):
                modules_to_hook.append((self.model.lm_head, 'output_layer'))

        # BERT-like classification
        elif hasattr(self.model, 'bert') or hasattr(self.model, 'distilbert'):
            if hasattr(self.model, 'bert'):
                base = self.model.bert
                classifier = self.model.classifier
            else:
                base = self.model.distilbert
                classifier = self.model.classifier

            # embeddings
            if hasattr(base, 'embeddings'):
                modules_to_hook.append((base.embeddings, 'embeddings'))
            # encoder layers
            if hasattr(base, 'encoder'):
                for layer in base.encoder.layer:
                    if hasattr(layer, 'attention'):
                        modules_to_hook.append((layer.attention, 'attention'))
                    if hasattr(layer, 'intermediate'):
                        modules_to_hook.append((layer.intermediate, 'ffn'))
            # classifier head
            modules_to_hook.append((classifier, 'output_layer'))

        # Fallback: hook all nn.Linear and HuggingFace Conv1D layers
        if not modules_to_hook:
            for m in self.model.modules():
                if isinstance(m, nn.Linear) or m.__class__.__name__ == 'Conv1D':
                    modules_to_hook.append((m, 'ffn'))

        # Register hooks or warn if none found
        if modules_to_hook:
            for module, name in modules_to_hook:
                module.register_forward_pre_hook(partial(self._hook_begin, name))
                module.register_forward_hook(partial(self._hook_end, name))
        else:
            print("Warning: Unsupported model architecture. Energy tracking may be incomplete.")

    def _hook_begin(self, name, module, inp):
        """Pre-forward hook to start energy measurement"""
        if self.zeus is None:
            return
        torch.cuda.synchronize()
        if name in self.active_windows:
            return
        try:
            self.zeus.begin_window(name)
            self.active_windows.add(name)
        except Exception as e:
            print(f"Error in begin_window for {name}: {e}")

    def _hook_end(self, name, module, inp, out):
        """Post-forward hook to end energy measurement and record results"""
        if self.zeus is None:
            return
        torch.cuda.synchronize()
        if name not in self.active_windows:
            return
        try:
            e = self.zeus.end_window(name).total_energy
            self.comp_energy[name].append(e)
            self.active_windows.remove(name)
        except Exception as e:
            print(f"Error in end_window for {name}: {e}")
            self.active_windows.discard(name)

    def measure_text(self, text, tokenizer):
        """Measure energy for a generation prompt."""
        # clear prior data
        for v in self.comp_energy.values():
            v.clear()
        if self.zeus:
            for w in list(self.active_windows):
                try:
                    self.zeus.end_window(w)
                except:
                    pass
                self.active_windows.discard(w)

        start_time = time.time()

        # tokenization measurement
        tok_energy = 0
        if self.zeus:
            try:
                self.zeus.begin_window('tokenization')
                self.active_windows.add('tokenization')
            except Exception as e:
                print(f"Error starting tokenization measurement: {e}")

        if isinstance(text, str):
            tokens = tokenizer(text, return_tensors='pt',
                               padding=True, truncation=True, max_length=64)
        else:
            tokens = {"input_ids": text}

        if self.zeus and 'tokenization' in self.active_windows:
            try:
                tok_meas = self.zeus.end_window('tokenization')
                tok_energy = tok_meas.total_energy
                self.active_windows.remove('tokenization')
            except Exception as e:
                print(f"Error ending tokenization measurement: {e}")
                self.active_windows.discard('tokenization')

        # prepare input
        input_ids = tokens["input_ids"].to('cuda')
        attention_mask = tokens.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to('cuda')

        # set dtype
        dtype = torch.float16 if self.precision_mode=='float16' else torch.float32

        # inference measurement
        inf_energy = 0
        if self.zeus:
            try:
                self.zeus.begin_window('inference')
                self.active_windows.add('inference')
            except Exception as e:
                print(f"Error starting inference measurement: {e}")

        try:
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=dtype):
                outputs = self.model(input_ids, attention_mask=attention_mask)
        except Exception as e:
            if self.zeus and 'inference' in self.active_windows:
                try: self.zeus.end_window('inference')
                except: pass
                self.active_windows.discard('inference')
            raise e

        if self.zeus and 'inference' in self.active_windows:
            try:
                meas = self.zeus.end_window('inference')
                inf_energy = meas.total_energy
                self.active_windows.remove('inference')
            except Exception as e:
                print(f"Error ending inference measurement: {e}")
                self.active_windows.discard('inference')

        elapsed = time.time() - start_time
        total_energy = tok_energy + inf_energy
        tokens_count = input_ids.numel()

        components = {k: np.sum(v) for k, v in self.comp_energy.items()}
        return outputs.logits, {
            'total_energy': total_energy,
            'tokenization_energy': tok_energy,
            'inference_energy': inf_energy,
            'energy_per_token': total_energy / tokens_count if tokens_count else 0,
            'time': elapsed,
            'components': components,
            'num_tokens': tokens_count
        }

    def measure_batch(self, input_ids, attention_mask=None):
        """Measure energy for classification batch."""
        for v in self.comp_energy.values():
            v.clear()
        if self.zeus:
            for w in list(self.active_windows):
                try: self.zeus.end_window(w)
                except: pass
                self.active_windows.discard(w)

        start_time = time.time()
        dtype = torch.float16 if self.precision_mode=='float16' else torch.float32

        total_e = 0
        if self.zeus:
            try:
                self.zeus.begin_window('inference')
                self.active_windows.add('inference')
            except Exception as e:
                print(f"Error starting inference measurement: {e}")

        try:
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=dtype):
                out = self.model(input_ids, attention_mask=attention_mask)
        except Exception as e:
            if self.zeus and 'inference' in self.active_windows:
                try: self.zeus.end_window('inference')
                except: pass
                self.active_windows.discard('inference')
            raise e

        if self.zeus and 'inference' in self.active_windows:
            try:
                meas = self.zeus.end_window('inference')
                total_e = meas.total_energy
                self.active_windows.remove('inference')
            except Exception as e:
                print(f"Error ending inference measurement: {e}")
                self.active_windows.discard('inference')

        elapsed = time.time() - start_time
        tokens = input_ids.numel()
        components = {k: np.sum(v) for k, v in self.comp_energy.items()}
        return out.logits, {
            'total_energy': total_e,
            'energy_per_token': total_e / tokens if tokens else 0,
            'time': elapsed,
            'components': components,
            'num_tokens': tokens
        }

def get_carbon_intensity():
    """
    Get carbon intensity at current location (gCO2eq/kWh)
    """
    try:
        g = geocoder.ip('me')
        if g.latlng is None:
            print("Could not determine your location. Using global average carbon intensity.")
            return 475
        lat, lon = g.latlng
        print(f"Location detected: {g.city}, {g.country} (lat: {lat}, lon: {lon})")
        print("Using estimated carbon intensity.")
        country_code = g.country
        estimates = {
            "US": 417, "CN": 620, "IN": 708, "GB": 231,
            "DE": 350, "FR": 70,  "CA": 150, "AU": 520,
            "JP": 462, "BR": 110
        }
        if country_code in estimates:
            val = estimates[country_code]
            print(f"Using estimate for {country_code}: {val} gCO2eq/kWh")
            return val
        print(f"No specific estimate for {country_code}. Using global average: 475 gCO2eq/kWh")
        return 475
    except Exception as e:
        print(f"Error getting carbon intensity: {e}")
        print("Using global average: 475 gCO2eq/kWh")
        return 475

def joules_to_co2(joules, intensity=None):
    """
    Convert joules to grams CO2 equivalent.
    """
    if intensity is None:
        intensity = get_carbon_intensity()
    return joules / 3600 * intensity
