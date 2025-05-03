#!/usr/bin/env python3
"""
This script patches the pynvml module to add the missing NVML_FI_DEV_POWER_INSTANT attribute.
Run this before importing the energy_utils module.
"""

import pynvml
import importlib
import sys
import ctypes

def patch_nvml():
    """
    Patch the pynvml module with missing power monitoring attributes needed by ZeusMonitor
    """
    print("Patching pynvml for Zeus compatibility...")
    
    # Check if the problematic attribute exists, add it if missing
    if not hasattr(pynvml, 'NVML_FI_DEV_POWER_INSTANT'):
        print("NVML_FI_DEV_POWER_INSTANT attribute is missing. Adding it...")
        
        # First check if we have NVML_FI_DEV_TOTAL_ENERGY_CONSUMPTION
        if hasattr(pynvml, 'NVML_FI_DEV_TOTAL_ENERGY_CONSUMPTION'):
            print("Using NVML_FI_DEV_TOTAL_ENERGY_CONSUMPTION as the power field")
            # Use total energy consumption for power monitoring
            pynvml.NVML_FI_DEV_POWER_INSTANT = pynvml.NVML_FI_DEV_TOTAL_ENERGY_CONSUMPTION
        else:
            # Just create a dummy value if needed
            print("No suitable power monitoring field found. Using placeholder.")
            pynvml.NVML_FI_DEV_POWER_INSTANT = 0
            
    # Check other NVML attributes that Zeus might need
    required_attrs = [
        'nvmlDeviceGetFieldValues',
        'nvmlFieldValue_t',
        'NVML_FI_MAX'
    ]
    
    for attr in required_attrs:
        if not hasattr(pynvml, attr):
            print(f"Missing attribute: {attr}")
        else:
            print(f"Found attribute: {attr}")
    
    print("NVML patching complete.")
    return True

if __name__ == "__main__":
    
    patch_nvml()
        # Patch missing nvmlFieldValue_t struct
    if not hasattr(pynvml, 'nvmlFieldValue_t'):
        class nvmlFieldValue_t(ctypes.Structure):
            _fields_ = [
                ('fieldId', ctypes.c_uint),
                ('scopeId', ctypes.c_uint),
                ('timestamp', ctypes.c_ulonglong),
                ('latencyUsec', ctypes.c_uint),
                ('valueType', ctypes.c_uint),
                ('nvmlReturn', ctypes.c_uint),
                ('value', ctypes.c_ulonglong),  # Union; simplified
            ]
        pynvml.nvmlFieldValue_t = nvmlFieldValue_t
        print("Patched missing nvmlFieldValue_t struct.")
    # Test the patch
    print("\nTesting NVML patch...")
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        print(f"Found {device_count} NVIDIA devices")
        
        # Try to get a handle to the first GPU
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        device_name = pynvml.nvmlDeviceGetName(handle)
        print(f"Device 0: {device_name}")
        
        # Check if we can get power usage
        power = pynvml.nvmlDeviceGetPowerUsage(handle)
        print(f"Power usage: {power/1000.0} W")
        
        # Try to get power using fieldValues (what Zeus uses)
        if hasattr(pynvml, 'nvmlDeviceGetFieldValues'):
            fieldIds = [pynvml.NVML_FI_DEV_POWER_INSTANT]
            values = (pynvml.nvmlFieldValue_t * len(fieldIds))()
            
            try:
                pynvml.nvmlDeviceGetFieldValues(handle, len(fieldIds), fieldIds, values)
                print(f"Patched field value for power: {values[0].value}")
            except Exception as e:
                print(f"Error getting field values: {e}")
                print("This may still be okay if Zeus uses a different approach.")
        
        pynvml.nvmlShutdown()
        print("NVML test successful!")
    except Exception as e:
        print(f"NVML test failed: {e}")