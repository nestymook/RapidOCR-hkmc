# Error Messages and Logging Validation Report

## Overview

This document validates that all error messages and logging in the rapidocr_hkmc package are clear, actionable, and follow best practices.

## Validation Date

November 2, 2025

## Error Message Categories

### 1. Configuration Errors

#### Invalid Device Name
**Location**: `rapidocr_hkmc/inference_engine/openvino.py:_get_target_device()`

**Error Message**:
```
Invalid device_name configuration: '{device_name}'. 
Allowed values are: CPU, NPU, GPU. 
Please update your configuration file with a valid device name.
```

**Validation**: ✅ PASS
- Clearly states what is invalid
- Lists all valid options
- Provides actionable guidance

#### Invalid Performance Hint
**Location**: `rapidocr_hkmc/inference_engine/openvino.py:_init_config()`

**Error Message**:
```
Invalid performance_hint configuration: '{performance_hint}'. 
Allowed values are: LATENCY, THROUGHPUT, CUMULATIVE_THROUGHPUT. 
Please update your configuration file with a valid performance hint.
```

**Validation**: ✅ PASS
- Clearly states what is invalid
- Lists all valid options
- Provides actionable guidance

#### Invalid Thread Count
**Location**: `rapidocr_hkmc/inference_engine/openvino.py:_init_config()`

**Error Message**:
```
Invalid inference_num_threads configuration: {infer_num_threads}. 
Must be between 1 and {os.cpu_count()} (available CPU cores), 
or -1 for automatic selection.
```

**Validation**: ✅ PASS
- Clearly states what is invalid
- Provides valid range
- Explains automatic option

### 2. Hardware Unavailability Warnings

#### NPU Not Available
**Location**: `rapidocr_hkmc/inference_engine/openvino.py:__init__()`

**Warning Message**:
```
Requested device 'NPU' is not available. Falling back to CPU.
```

**Validation**: ✅ PASS
- Clearly states what was requested
- Explains fallback behavior
- Appropriate warning level

#### CUDA/GPU Not Available
**Location**: `rapidocr_hkmc/inference_engine/onnxruntime/provider_config.py:is_cuda_available()`

**Warning Message**:
```
CUDAExecutionProvider is not in available providers (['CPUExecutionProvider']). 
Use CPUExecutionProvider inference by default.
```

**Followed by Installation Instructions**:
```
If you want to use CUDAExecutionProvider acceleration, you must do:
First, uninstall all onnxruntime packages in current environment.
Second, install onnxruntime-gpu by `pip install onnxruntime-gpu`.
Note the onnxruntime-gpu version must match your cuda and cudnn version.
You can refer this link: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
Third, ensure CUDAExecutionProvider is in available providers list.
```

**Validation**: ✅ PASS
- Clearly states what is unavailable
- Explains fallback behavior
- Provides detailed installation instructions
- Includes reference link

### 3. Runtime Errors

#### Model Loading Failure
**Location**: `rapidocr_hkmc/inference_engine/openvino.py:__init__()`

**Error Message**:
```
Failed to read model from {model_path}: {str(e)}. 
Please verify the model file exists and is a valid OpenVINO model.
```

**Validation**: ✅ PASS
- Clearly states what failed
- Includes specific error details
- Provides actionable guidance

#### Model Compilation Failure
**Location**: `rapidocr_hkmc/inference_engine/openvino.py:__init__()`

**Error Message**:
```
Failed to compile model on {actual_device} device. 
Error: {str(e)}. 
Diagnostic information: 
Device={actual_device}, 
Available devices={core.available_devices}, 
Model path={model_path}, 
Config={config}
```

**Validation**: ✅ PASS
- Clearly states what failed
- Includes comprehensive diagnostic information
- Helps with troubleshooting

#### Compilation Failure with Fallback
**Location**: `rapidocr_hkmc/inference_engine/openvino.py:__init__()`

**Warning Message**:
```
Attempting to fall back to CPU device after {actual_device} compilation failure
```

**Success Message**:
```
Successfully compiled model on CPU device as fallback in {fallback_time:.3f} seconds
```

**Validation**: ✅ PASS
- Clearly explains fallback attempt
- Reports success with timing
- Appropriate warning level

## Logging Validation

### 1. Informational Logging

#### Device Selection
**Location**: `rapidocr_hkmc/inference_engine/openvino.py:__init__()`

**Log Messages**:
```
INFO: Available OpenVINO devices: ['CPU', 'NPU']
INFO: Requested device: NPU
INFO: Using device: NPU
```

**Validation**: ✅ PASS
- Logs available devices for transparency
- Logs requested device for verification
- Logs actual device being used
- Appropriate INFO level

#### Model Loading Timing
**Location**: `rapidocr_hkmc/inference_engine/openvino.py:__init__()`

**Log Messages**:
```
INFO: Model compiled successfully on NPU device in 0.523 seconds
INFO: OpenVINO inference session initialized successfully. Execution provider: NPU
INFO: Total model loading time: 0.845 seconds
```

**Validation**: ✅ PASS
- Provides performance metrics
- Confirms successful initialization
- Reports actual execution provider
- Appropriate INFO level

### 2. Debug Logging

#### Hardware Configuration
**Location**: `rapidocr_hkmc/inference_engine/openvino.py:__init__()`

**Log Message**:
```
DEBUG: Hardware configuration - Device: NPU, Config: {...}, Available devices: ['CPU', 'NPU']
```

**Validation**: ✅ PASS
- Provides detailed configuration information
- Useful for troubleshooting
- Appropriate DEBUG level

#### Model Read Timing
**Location**: `rapidocr_hkmc/inference_engine/openvino.py:__init__()`

**Log Message**:
```
DEBUG: Model read completed in 0.234 seconds
```

**Validation**: ✅ PASS
- Provides detailed timing information
- Useful for performance analysis
- Appropriate DEBUG level

#### Device-Specific Settings
**Location**: `rapidocr_hkmc/inference_engine/openvino.py:_init_config()`

**Log Message**:
```
INFO: Using OpenVINO config for NPU: {'PERFORMANCE_HINT': 'LATENCY'}
DEBUG: Device-specific settings applied for NPU
```

**Validation**: ✅ PASS
- Shows actual configuration being used
- Confirms settings application
- Appropriate log levels

### 3. Warning Logging

#### Provider Mismatch
**Location**: `rapidocr_hkmc/inference_engine/onnxruntime/provider_config.py:verify_providers()`

**Log Message**:
```
WARNING: CUDAExecutionProvider is available, but the inference part is automatically 
shifted to be executed under CPUExecutionProvider.
WARNING: The available lists are ['CPUExecutionProvider']
```

**Validation**: ✅ PASS
- Clearly explains unexpected behavior
- Provides diagnostic information
- Appropriate WARNING level

## Log Level Distribution

### Appropriate Use of Log Levels

| Level | Usage | Examples |
|-------|-------|----------|
| ERROR | Configuration errors, critical failures | Invalid device name, model loading failure |
| WARNING | Hardware unavailability, fallback scenarios | NPU not available, CUDA not detected |
| INFO | Important events, initialization, device selection | Model loaded, device selected, timing info |
| DEBUG | Detailed diagnostic information | Configuration details, internal state |

**Validation**: ✅ PASS
- Log levels are appropriate for message severity
- ERROR for critical issues that prevent operation
- WARNING for issues with graceful degradation
- INFO for important operational events
- DEBUG for detailed diagnostic information

## Error Handling Patterns

### 1. Graceful Degradation

**Pattern**: Request NPU → NPU unavailable → Fallback to CPU

**Implementation**:
```python
if not self._check_device_availability(core, device_name):
    logger.warning(f"Requested device '{device_name}' is not available. Falling back to CPU.")
    actual_device = "CPU"
```

**Validation**: ✅ PASS
- Checks availability before use
- Logs warning with clear message
- Falls back to safe default
- Continues operation

### 2. Fail-Fast with Clear Errors

**Pattern**: Invalid configuration → Immediate error with guidance

**Implementation**:
```python
if device_name not in valid_devices:
    error_msg = (
        f"Invalid device_name configuration: '{device_name}'. "
        f"Allowed values are: {', '.join(valid_devices)}. "
        f"Please update your configuration file with a valid device name."
    )
    logger.error(error_msg)
    raise ConfigurationError(error_msg)
```

**Validation**: ✅ PASS
- Validates configuration early
- Provides clear error message
- Lists valid options
- Raises appropriate exception

### 3. Diagnostic Information on Failure

**Pattern**: Compilation failure → Detailed diagnostic information

**Implementation**:
```python
error_msg = (
    f"Failed to compile model on {actual_device} device. "
    f"Error: {str(e)}. "
    f"Diagnostic information: "
    f"Device={actual_device}, "
    f"Available devices={core.available_devices}, "
    f"Model path={model_path}, "
    f"Config={config}"
)
```

**Validation**: ✅ PASS
- Includes original error
- Provides context (device, path, config)
- Lists available alternatives
- Helps with troubleshooting

## Actionable Guidance

### Installation Instructions

**CUDA/GPU Setup**:
- ✅ Explains what needs to be installed
- ✅ Provides exact pip command
- ✅ Mentions version compatibility
- ✅ Includes reference link
- ✅ Explains verification step

**DirectML Setup**:
- ✅ Explains platform requirements
- ✅ Provides installation command
- ✅ Explains verification step

**CANN Setup**:
- ✅ Mentions prerequisite software
- ✅ Provides reference link
- ✅ Explains verification step

### Configuration Guidance

**Device Selection**:
- ✅ Lists all valid device names
- ✅ Explains what each device is for
- ✅ Provides example configurations

**Performance Tuning**:
- ✅ Lists all valid performance hints
- ✅ Explains when to use each option
- ✅ Provides recommended settings

## Summary

### Overall Validation Result: ✅ PASS

All error messages and logging in the rapidocr_hkmc package meet the following criteria:

1. **Clarity**: Messages clearly state what went wrong or what is happening
2. **Actionability**: Messages provide guidance on how to fix issues
3. **Appropriate Severity**: Log levels match the severity of events
4. **Diagnostic Information**: Errors include sufficient context for troubleshooting
5. **User-Friendly**: Messages are written for end users, not just developers
6. **Consistent Format**: Similar types of messages follow consistent patterns

### Key Strengths

1. **Comprehensive Installation Instructions**: GPU/NPU unavailability warnings include detailed setup instructions
2. **Graceful Degradation**: System falls back to CPU with clear warnings when hardware is unavailable
3. **Diagnostic Information**: Errors include device lists, configurations, and paths for troubleshooting
4. **Performance Metrics**: Timing information helps users understand performance characteristics
5. **Configuration Validation**: Invalid configurations are caught early with clear error messages

### Recommendations

All error messages and logging are production-ready. No changes required.

## Test Evidence

The following test output demonstrates the error messages and logging in action:

```
[WARNING] CUDAExecutionProvider is not in available providers. Use CPUExecutionProvider inference by default.
[INFO] If you want to use CUDAExecutionProvider acceleration, you must do:
[INFO] First, uninstall all onnxruntime packages in current environment.
[INFO] Second, install onnxruntime-gpu by `pip install onnxruntime-gpu`.
[INFO] Note the onnxruntime-gpu version must match your cuda and cudnn version.
[INFO] You can refer this link: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
[INFO] Third, ensure CUDAExecutionProvider is in available providers list.
```

This demonstrates:
- Clear warning about unavailability
- Detailed installation instructions
- Reference links for more information
- Step-by-step guidance

## Conclusion

The error messages and logging in rapidocr_hkmc are well-designed, clear, and actionable. They provide users with the information needed to:

1. Understand what went wrong
2. Diagnose hardware and configuration issues
3. Take corrective action
4. Verify successful operation

The implementation follows best practices for production software and requires no modifications.
