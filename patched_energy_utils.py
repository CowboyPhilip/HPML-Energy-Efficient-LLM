# patched_energy_utils.py
from zeus.monitor import ZeusMonitor
import torch
from functools import partial
import numpy as np
import time
import geocoder
import warnings

class PatchedEnergyTracker:
    def __init__(self, model, precision_mode=None):
        """
        Initialize energy tracker for a model with enhanced error handling

        Args:
            model: GPU nn.Module
            precision_mode: None|'float16'
        """
        self.model = model
        self.precision_mode = precision_mode
        self.active_windows = set()  # Track active measurement windows
        self.energy_measurement_available = False  # Flag to indicate if energy measurement is available
        self.silent_mode = True  # Suppress NVML errors
        
        # Initialize ZeusMonitor to measure GPU energy
        try:
            self.zeus = ZeusMonitor(
                approx_instant_energy=True,
                gpu_indices=[torch.cuda.current_device()]
            )
            
            # Test if energy measurement actually works
            try:
                self.zeus.begin_window("test_window")
                result = self.zeus.end_window("test_window")
                if hasattr(result, 'total_energy') and result.total_energy > 0:
                    self.energy_measurement_available = True
                    print("Successfully initialized ZeusMonitor with energy measurement capability")
                else:
                    print("Energy measurement is not providing valid data, falling back to time measurements only")
                    self.energy_measurement_available = False
            except Exception as e:
                print(f"Energy measurement is not available: {str(e).split(':')[0]}")
                print("Continuing with time measurement only")
                self.energy_measurement_available = False
        except Exception as e:
            print(f"Error initializing ZeusMonitor: {str(e).split(':')[0]}")
            print("Falling back to time measurement only")
            self.zeus = None

        # Energy consumption
        self.comp_energy = {
            'embeddings': [],
            'attention': [],
            'ffn': [],
            'layernorm': [],
            'output_layer': []
        }
        
        # Time measurements (as a fallback)
        self.comp_time = {
            'embeddings': [],
            'attention': [],
            'ffn': [],
            'layernorm': [],
            'output_layer': []
        }

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register hooks based on model type"""
        # [Same as original implementation]
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
        # Always record start time for timing measurements
        setattr(module, '_time_start', time.time())
        
        # Try energy measurement if available
        if self.zeus is not None and self.energy_measurement_available:
            torch.cuda.synchronize()
            try:
                # Check if window is already active
                if name in self.active_windows:
                    if not self.silent_mode:
                        print(f"Warning: Measurement window '{name}' already exists. Skipping.")
                    return

                self.zeus.begin_window(name)
                self.active_windows.add(name)
            except Exception as e:
                if not self.silent_mode:
                    # Only print if it's not the NVML error we're expecting
                    if "NVML_FI_DEV_POWER_INSTANT" not in str(e):
                        print(f"Error in begin_window for {name}: {e}")
                self.energy_measurement_available = False

    def _hook_end(self, name, module, inp, out):
        """Post-forward hook to end energy measurement and record results"""
        # Always record timing information
        if hasattr(module, '_time_start'):
            elapsed = time.time() - module._time_start
            if name not in self.comp_time:
                self.comp_time[name] = []
            self.comp_time[name].append(elapsed)
        
        # Try energy measurement if available
        if self.zeus is not None and self.energy_measurement_available:
            torch.cuda.synchronize()
            try:
                # Check if window is active
                if name not in self.active_windows:
                    if not self.silent_mode:
                        print(f"Warning: Measurement window '{name}' not found. Skipping.")
                    return

                e = self.zeus.end_window(name).total_energy
                if name not in self.comp_energy:
                    self.comp_energy[name] = []
                self.comp_energy[name].append(e)
                self.active_windows.remove(name)
            except Exception as e:
                if not self.silent_mode:
                    # Only print if it's not the NVML error we're expecting
                    if "NVML_FI_DEV_POWER_INSTANT" not in str(e):
                        print(f"Error in end_window for {name}: {e}")
                self.energy_measurement_available = False
                # Force remove from active windows to prevent future errors
                if name in self.active_windows:
                    self.active_windows.remove(name)

    def measure_batch(self, input_ids, attention_mask=None):
        """Batch Measurement for Classification Tasks"""
        # Clear tracking data
        for v in self.comp_energy.values():
            v.clear()
        
        for v in self.comp_time.values():
            v.clear()

        # Ensure no active windows before starting
        if self.zeus is not None:
            for window in list(self.active_windows):
                if not self.silent_mode:
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
        if self.zeus is not None and self.energy_measurement_available:
            try:
                self.zeus.begin_window('inference')
                self.active_windows.add('inference')
            except Exception as e:
                if not self.silent_mode and "NVML_FI_DEV_POWER_INSTANT" not in str(e):
                    print(f"Error starting inference measurement: {e}")
                self.energy_measurement_available = False

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

        if self.zeus is not None and self.energy_measurement_available and 'inference' in self.active_windows:
            try:
                meas = self.zeus.end_window('inference')
                self.active_windows.remove('inference')
                total_e = meas.total_energy
            except Exception as e:
                if not self.silent_mode and "NVML_FI_DEV_POWER_INSTANT" not in str(e):
                    print(f"Error ending inference measurement: {e}")
                self.energy_measurement_available = False
                if 'inference' in self.active_windows:
                    self.active_windows.remove('inference')

        end_time = time.time()
        elapsed_time = end_time - start_time

        tokens = input_ids.numel()  # batch_size * seq_len

        # Component energy (if available)
        component_energy = {}
        if self.energy_measurement_available:
            for k, v in self.comp_energy.items():
                component_energy[k] = np.sum(v) if v else 0
        else:
            for k in self.comp_energy.keys():
                component_energy[k] = 0
        
        # Component timing (always available)
        component_time = {}
        for k, v in self.comp_time.items():
            component_time[k] = np.sum(v) if v else 0

        return out.logits, {
            'total_energy': total_e,
            'energy_per_token': total_e / tokens if total_e > 0 and tokens > 0 else 0,
            'time': elapsed_time,
            'components': component_energy,
            'component_times': component_time,
            'num_tokens': tokens,
            'energy_measurement_available': self.energy_measurement_available
        }

# Keep the original carbon intensity and joules_to_co2 functions
def get_carbon_intensity():
    """Get carbon intensity at current location (gCO2eq/kWh)"""
    # [Same as original implementation]
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
    """Convert energy in joules to CO2 equivalent emissions in grams"""
    # [Same as original implementation]
    if intensity is None:
        intensity = get_carbon_intensity()

    # J -> kWh: joules/3600, * intensity (gCO2eq/kWh)
    emissions = joules/3600 * intensity
    return emissions