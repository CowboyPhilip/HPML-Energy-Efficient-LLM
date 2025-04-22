from zeus.monitor import ZeusMonitor
import torch
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

        # Energy consumption
        self.comp_energy = {
            'embeddings': [],
            'attention': [],
            'ffn': [],
            'layernorm': [],
            'output_layer': []
        }

        # Register hooks only if zeus monitor is available
        if self.zeus is not None:
            self._register_hooks()
        else:
            print("Skipping hook registration since ZeusMonitor is not available")

    def _register_hooks(self):
        """Register hooks based on model type"""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # For CausalLM models
            base = self.model.model

            # Register hooks for embeddings
            if hasattr(base, 'embed_tokens'):
                base.embed_tokens.register_forward_pre_hook(
                    partial(self._hook_begin, 'embeddings')
                )
                base.embed_tokens.register_forward_hook(
                    partial(self._hook_end, 'embeddings')
                )

            # Register hooks for attention & FFN in each layer
            for layer in base.layers:
                if hasattr(layer, 'self_attn'):
                    layer.self_attn.register_forward_pre_hook(
                        partial(self._hook_begin, 'attention')
                    )
                    layer.self_attn.register_forward_hook(
                        partial(self._hook_end, 'attention')
                    )

                if hasattr(layer, 'mlp'):
                    layer.mlp.register_forward_pre_hook(
                        partial(self._hook_begin, 'ffn')
                    )
                    layer.mlp.register_forward_hook(
                        partial(self._hook_end, 'ffn')
                    )

            # Register hook for final LayerNorm
            if hasattr(base, 'norm'):
                base.norm.register_forward_pre_hook(
                    partial(self._hook_begin, 'layernorm')
                )
                base.norm.register_forward_hook(
                    partial(self._hook_end, 'layernorm')
                )

            # Register hook for output layer
            if hasattr(self.model, 'lm_head'):
                self.model.lm_head.register_forward_pre_hook(
                    partial(self._hook_begin, 'output_layer')
                )
                self.model.lm_head.register_forward_hook(
                    partial(self._hook_end, 'output_layer')
                )
        elif hasattr(self.model, 'bert') or hasattr(self.model, 'distilbert'):
            # For SequenceClassification models (BERT-like)
            if hasattr(self.model, 'bert'):
                base = self.model.bert
                classifier = self.model.classifier
            elif hasattr(self.model, 'distilbert'):
                base = self.model.distilbert
                classifier = self.model.classifier
            else:
                print("Warning: Unknown model architecture. Energy tracking may be incomplete.")
                return

            # Register embeddings hooks
            if hasattr(base, 'embeddings'):
                base.embeddings.register_forward_pre_hook(
                    partial(self._hook_begin, 'embeddings')
                )
                base.embeddings.register_forward_hook(
                    partial(self._hook_end, 'embeddings')
                )

            # Register encoder layer hooks
            if hasattr(base, 'encoder'):
                for layer in base.encoder.layer:
                    if hasattr(layer, 'attention'):
                        layer.attention.register_forward_pre_hook(
                            partial(self._hook_begin, 'attention')
                        )
                        layer.attention.register_forward_hook(
                            partial(self._hook_end, 'attention')
                        )

                    if hasattr(layer, 'intermediate'):
                        layer.intermediate.register_forward_pre_hook(
                            partial(self._hook_begin, 'ffn')
                        )
                        layer.intermediate.register_forward_hook(
                            partial(self._hook_end, 'ffn')
                        )

            # Register classifier hooks
            classifier.register_forward_pre_hook(
                partial(self._hook_begin, 'output_layer')
            )
            classifier.register_forward_hook(
                partial(self._hook_end, 'output_layer')
            )
        else:
            print("Warning: Unsupported model architecture. Energy tracking may be incomplete.")

    def _hook_begin(self, name, module, inp):
        """Pre-forward hook to start energy measurement"""
        if self.zeus is not None:
            torch.cuda.synchronize()
            try:
                # Check if window is already active
                if name in self.active_windows:
                    print(f"Warning: Measurement window '{name}' already exists. Skipping.")
                    return

                self.zeus.begin_window(name)
                self.active_windows.add(name)
            except Exception as e:
                print(f"Error in begin_window for {name}: {e}")

    def _hook_end(self, name, module, inp, out):
        """Post-forward hook to end energy measurement and record results"""
        if self.zeus is not None:
            torch.cuda.synchronize()
            try:
                # Check if window is active
                if name not in self.active_windows:
                    print(f"Warning: Measurement window '{name}' not found. Skipping.")
                    return

                e = self.zeus.end_window(name).total_energy
                self.comp_energy[name].append(e)
                self.active_windows.remove(name)
            except Exception as e:
                print(f"Error in end_window for {name}: {e}")
                # Force remove from active windows to prevent future errors
                if name in self.active_windows:
                    self.active_windows.remove(name)

    def measure_text(self, text, tokenizer):
        """
        Measure energy consumption for text generation

        Args:
            text: Text prompt or tokenized input
            tokenizer: HuggingFace tokenizer

        Returns:
            (logits, metrics_dict)
        """
        # Clear component energy tracking
        for v in self.comp_energy.values():
            v.clear()

        # Ensure no active windows before starting
        if self.zeus is not None:
            for window in list(self.active_windows):
                print(f"Warning: Closing stale measurement window '{window}'")
                try:
                    self.zeus.end_window(window)
                except:
                    pass
                self.active_windows.remove(window)

        # Start measurement
        start_time = time.time()

        # Tokenization
        tok_energy = 0
        if self.zeus is not None:
            try:
                self.zeus.begin_window('tokenization')
                self.active_windows.add('tokenization')
            except Exception as e:
                print(f"Error starting tokenization measurement: {e}")

        if isinstance(text, str):
            tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=64)
        else:
            # Assume text is already tokenized
            tokens = {"input_ids": text}

        if self.zeus is not None and 'tokenization' in self.active_windows:
            try:
                tok_meas = self.zeus.end_window('tokenization')
                self.active_windows.remove('tokenization')
                tok_energy = tok_meas.total_energy
            except Exception as e:
                print(f"Error ending tokenization measurement: {e}")
                self.active_windows.remove('tokenization')

        # Handle different input formats
        if isinstance(text, str):
            input_ids = tokens.input_ids.to('cuda')
            attention_mask = tokens.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to('cuda')
        else:
            # If input is already tensors
            input_ids = text.to('cuda') if not text.is_cuda else text
            attention_mask = None

        # Set precision mode
        dtype = torch.float16 if self.precision_mode == 'float16' else torch.float32

        # Inference energy consumption
        inf_energy = 0
        if self.zeus is not None:
            try:
                self.zeus.begin_window('inference')
                self.active_windows.add('inference')
            except Exception as e:
                print(f"Error starting inference measurement: {e}")

        try:
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=dtype):
                outputs = self.model(input_ids, attention_mask=attention_mask)
        except Exception as e:
            print(f"Error during model inference: {e}")
            if self.zeus is not None and 'inference' in self.active_windows:
                try:
                    self.zeus.end_window('inference')
                except:
                    pass
                self.active_windows.remove('inference')
            raise e

        if self.zeus is not None and 'inference' in self.active_windows:
            try:
                inf_meas = self.zeus.end_window('inference')
                self.active_windows.remove('inference')
                inf_energy = inf_meas.total_energy
            except Exception as e:
                print(f"Error ending inference measurement: {e}")
                if 'inference' in self.active_windows:
                    self.active_windows.remove('inference')

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Calculate total energy and token count
        total_e = inf_energy + tok_energy
        tokens_count = input_ids.shape[0] * input_ids.shape[1]  # batch_size * seq_len

        # Component energy breakdown
        if self.zeus is not None:
            component_energy = {k: np.sum(v) for k, v in self.comp_energy.items()}
        else:
            component_energy = {k: 0 for k in self.comp_energy.keys()}

        return outputs.logits, {
            'total_energy': total_e,
            'tokenization_energy': tok_energy,
            'inference_energy': inf_energy,
            'energy_per_token': total_e / tokens_count if total_e > 0 and tokens_count > 0 else 0,
            'time': elapsed_time,
            'components': component_energy,
            'num_tokens': tokens_count
        }

    def measure_batch(self, input_ids, attention_mask=None):
        """Batch Measurement for Classification Tasks"""
        # Clear component energy tracking
        for v in self.comp_energy.values():
            v.clear()

        # Ensure no active windows before starting
        if self.zeus is not None:
            for window in list(self.active_windows):
                print(f"Warning: Closing stale measurement window '{window}'")
                try:
                    self.zeus.end_window(window)
                except:
                    pass
                self.active_windows.remove(window)

        # Start measurement
        start_time = time.time()

        # Set precision mode
        dtype = torch.float16 if self.precision_mode == 'float16' else torch.float32

        # Measure inference
        total_e = 0
        if self.zeus is not None:
            try:
                self.zeus.begin_window('inference')
                self.active_windows.add('inference')
            except Exception as e:
                print(f"Error starting inference measurement: {e}")

        try:
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=dtype):
                out = self.model(input_ids, attention_mask=attention_mask)
        except Exception as e:
            print(f"Error during model inference: {e}")
            # Make sure to end measurement window even if inference fails
            if self.zeus is not None and 'inference' in self.active_windows:
                try:
                    self.zeus.end_window('inference')
                except:
                    pass
                self.active_windows.remove('inference')
            raise e

        if self.zeus is not None and 'inference' in self.active_windows:
            try:
                meas = self.zeus.end_window('inference')
                self.active_windows.remove('inference')
                total_e = meas.total_energy
            except Exception as e:
                print(f"Error ending inference measurement: {e}")
                if 'inference' in self.active_windows:
                    self.active_windows.remove('inference')

        end_time = time.time()
        elapsed_time = end_time - start_time

        tokens = input_ids.numel()  # batch_size * seq_len

        # Component energy breakdown
        if self.zeus is not None:
            component_energy = {k: np.sum(v) for k, v in self.comp_energy.items()}
        else:
            component_energy = {k: 0 for k in self.comp_energy.keys()}

        return out.logits, {
            'total_energy': total_e,
            'energy_per_token': total_e / tokens if total_e > 0 and tokens > 0 else 0,
            'time': elapsed_time,
            'components': component_energy,
            'num_tokens': tokens
        }
    
def get_carbon_intensity():
    """
    Get carbon intensity at current location (gCO2eq/kWh)
    """
    try:
        # Try to get location data
        g = geocoder.ip('me')
        if g.latlng is None:
            print("Could not determine your location. Using global average carbon intensity.")
            return 475  # Global average estimate

        lat, lon = g.latlng
        print(f"Location detected: {g.city}, {g.country} (lat: {lat}, lon: {lon})")

        # For demo purposes, return an estimated value
        print("Using estimated carbon intensity.")

        # Use country-specific values if available
        country_code = g.country
        country_estimates = {
            "US": 417,  # United States average
            "CN": 620,  # China average
            "IN": 708,  # India average
            "GB": 231,  # United Kingdom average
            "DE": 350,  # Germany average
            "FR": 70,   # France (mostly nuclear)
            "CA": 150,  # Canada (hydro heavy)
            "AU": 520,  # Australia (coal heavy)
            "JP": 462,  # Japan average
            "BR": 110,  # Brazil (hydro heavy)
        }

        if country_code in country_estimates:
            intensity = country_estimates[country_code]
            print(f"Using estimated carbon intensity for {country_code}: {intensity} gCO2eq/kWh")
            return intensity
        else:
            print(f"No specific estimate for {country_code}. Using global average: 475 gCO2eq/kWh")
            return 475  # Global average estimate

    except Exception as e:
        print(f"Error getting carbon intensity: {str(e)}")
        print("Using global average carbon intensity: 475 gCO2eq/kWh")
        return 475  # Global average estimate as fallback

def joules_to_co2(joules, intensity=None):
    """
    Convert energy in joules to CO2 equivalent emissions in grams

    Args:
        joules: Energy in joules
        intensity: Carbon intensity in gCO2eq/kWh (if None, will be fetched)

    Returns:
        CO2 emissions in grams
    """
    if intensity is None:
        intensity = get_carbon_intensity()

    # J -> kWh: joules/3600, * intensity (gCO2eq/kWh)
    emissions = joules/3600 * intensity
    return emissions