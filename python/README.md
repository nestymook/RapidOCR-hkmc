# RapidOCR HKMC

RapidOCR HKMC is an enhanced version of RapidOCR with hardware acceleration support for NPU (Neural Processing Unit) and GPU devices.

## 📚 Documentation

- **[QUICK_START.md](QUICK_START.md)** - Get started in 5 minutes
- **[OFFLINE_MODE_QUICK_START.md](OFFLINE_MODE_QUICK_START.md)** - Offline mode setup (no internet required)
- **[CONFIGURATION_SUMMARY.md](CONFIGURATION_SUMMARY.md)** - Quick reference for configuration usage
- **[CONFIGURATION_USAGE_GUIDE.md](CONFIGURATION_USAGE_GUIDE.md)** - Detailed configuration guide
- **[CONFIGURATION_WORKFLOW.md](CONFIGURATION_WORKFLOW.md)** - Visual workflow diagrams
- **[OFFLINE_MODE_GUIDE.md](OFFLINE_MODE_GUIDE.md)** - Complete offline mode documentation
- **[example_usage.py](example_usage.py)** - Runnable code examples
- **[ERROR_MESSAGES_VALIDATION.md](ERROR_MESSAGES_VALIDATION.md)** - Troubleshooting guide

## Features

- **NPU Acceleration**: Use Intel NPU for text classification models via OpenVINO
- **GPU Acceleration**: Use NVIDIA CUDA for text detection and recognition models via ONNXRuntime
- **Flexible Configuration**: Mix and match hardware acceleration for different model types
- **Automatic Fallback**: Gracefully falls back to CPU when hardware is unavailable
- **Offline Mode**: Full support for air-gapped environments with local model files
- **Backward Compatible**: Works with existing RapidOCR configurations

## Installation

### Basic Installation

```bash
pip install rapidocr_hkmc
```

### Installation with GPU Support

```bash
pip install rapidocr_hkmc onnxruntime-gpu
```

### Installation from Wheel

```bash
pip install rapidocr_hkmc-*.whl
```

## Building from Source

### Building the Wheel

To build the wheel distribution package from source:

```bash
# Install build dependencies
pip install setuptools wheel

# Build the wheel
python setup.py bdist_wheel

# The wheel file will be created in the dist/ directory
# Example: dist/rapidocr_hkmc-1.0.0-py3-none-any.whl
```

### Installing the Built Wheel

After building, install the wheel:

```bash
# Install the wheel (replace version number with actual version)
pip install dist/rapidocr_hkmc-*.whl

# Or install with GPU support
pip install dist/rapidocr_hkmc-*.whl onnxruntime-gpu
```

### Building with Custom Version

To build with a specific version number:

```bash
python setup.py bdist_wheel 1.2.3
```

### Development Installation

For development, install in editable mode:

```bash
# Install in editable mode with dependencies
pip install -e .

# Or with development dependencies
pip install -e .[dev]
```

## Hardware Requirements

### For NPU Support
- Intel CPU with integrated NPU (e.g., Intel Core Ultra processors)
- OpenVINO 2023.0 or later
- Intel NPU drivers installed

### For GPU Support
- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.x or 12.x
- cuDNN library
- onnxruntime-gpu package

## Quick Start

### Basic Usage (CPU)

```python
from rapidocr_hkmc import RapidOCR

# Initialize with default CPU configuration
ocr = RapidOCR()

# Process an image
result, elapse = ocr('path/to/image.jpg')
print(result)
```

### Using NPU for Classification

```python
from rapidocr_hkmc import RapidOCR

# Initialize with NPU for classification model
ocr = RapidOCR(config_path='config_npu.yaml')

result, elapse = ocr('path/to/image.jpg')
```

### Using GPU for Detection and Recognition

```python
from rapidocr_hkmc import RapidOCR

# Initialize with GPU for detection and recognition
ocr = RapidOCR(config_path='config_gpu.yaml')

result, elapse = ocr('path/to/image.jpg')
```

## Configuration

### NPU Configuration for Classification Models

To enable NPU acceleration for the text classification model, configure the OpenVINO engine in your `config.yaml`:

```yaml
EngineConfig:
  openvino:
    device_name: "NPU"                    # Use NPU device
    performance_hint: "LATENCY"           # Optimize for low latency
    inference_num_threads: -1             # Auto-detect threads

Cls:
  engine_type: "openvino"                 # Use OpenVINO engine
  # ... other cls parameters
```

**Available device_name options:**
- `"NPU"` - Use Neural Processing Unit (Intel NPU)
- `"CPU"` - Use CPU (default)
- `"GPU"` - Use integrated/discrete GPU via OpenVINO

**Performance hints:**
- `"LATENCY"` - Optimize for low latency (single image processing)
- `"THROUGHPUT"` - Optimize for high throughput (batch processing)
- `"CUMULATIVE_THROUGHPUT"` - Balance between latency and throughput

### GPU Configuration for Detection and Recognition Models

To enable GPU acceleration for detection and recognition models, configure ONNXRuntime with CUDA:

```yaml
EngineConfig:
  onnxruntime:
    use_cuda: true                        # Enable CUDA
    cuda_ep_cfg:
      device_id: 0                        # GPU device ID (for multi-GPU)
      arena_extend_strategy: "kNextPowerOfTwo"
      cudnn_conv_algo_search: "EXHAUSTIVE"
      do_copy_in_default_stream: true

Det:
  engine_type: "onnxruntime"              # Use ONNXRuntime engine
  # ... other det parameters

Rec:
  engine_type: "onnxruntime"              # Use ONNXRuntime engine
  # ... other rec parameters
```

**CUDA configuration options:**
- `device_id` - GPU device ID (0 for first GPU, 1 for second, etc.)
- `arena_extend_strategy` - Memory allocation strategy
- `cudnn_conv_algo_search` - Convolution algorithm search mode
- `do_copy_in_default_stream` - Stream synchronization behavior

### Mixed Configuration (NPU + GPU)

You can use NPU for classification and GPU for detection/recognition simultaneously:

```yaml
EngineConfig:
  openvino:
    device_name: "NPU"
    performance_hint: "LATENCY"
  
  onnxruntime:
    use_cuda: true
    cuda_ep_cfg:
      device_id: 0

Cls:
  engine_type: "openvino"                 # Cls uses NPU

Det:
  engine_type: "onnxruntime"              # Det uses GPU

Rec:
  engine_type: "onnxruntime"              # Rec uses GPU
```

## Configuration Examples

### Example 1: NPU-Only Setup

Best for Intel systems with NPU, optimizing power efficiency:

```yaml
EngineConfig:
  openvino:
    device_name: "NPU"
    performance_hint: "LATENCY"

Cls:
  engine_type: "openvino"

Det:
  engine_type: "openvino"

Rec:
  engine_type: "openvino"
```

### Example 2: GPU-Only Setup

Best for systems with NVIDIA GPU, maximizing throughput:

```yaml
EngineConfig:
  onnxruntime:
    use_cuda: true
    cuda_ep_cfg:
      device_id: 0
      arena_extend_strategy: "kNextPowerOfTwo"

Cls:
  engine_type: "onnxruntime"

Det:
  engine_type: "onnxruntime"

Rec:
  engine_type: "onnxruntime"
```

### Example 3: Hybrid Setup (Recommended)

Best for systems with both NPU and GPU:

```yaml
EngineConfig:
  openvino:
    device_name: "NPU"              # NPU for lightweight cls model
    performance_hint: "LATENCY"
  
  onnxruntime:
    use_cuda: true                  # GPU for compute-intensive det/rec
    cuda_ep_cfg:
      device_id: 0

Cls:
  engine_type: "openvino"           # Cls on NPU (low power, fast)

Det:
  engine_type: "onnxruntime"        # Det on GPU (high throughput)

Rec:
  engine_type: "onnxruntime"        # Rec on GPU (high throughput)
```

## Automatic Fallback

The system automatically falls back to CPU if the requested hardware is unavailable:

```python
# If NPU is not available, automatically uses CPU
ocr = RapidOCR(config_path='config_npu.yaml')

# Check logs for fallback warnings:
# WARNING: NPU not available, falling back to CPU
```

## Logging

Enable detailed logging to monitor hardware usage:

```python
import logging

logging.basicConfig(level=logging.INFO)

ocr = RapidOCR()
# Logs will show:
# INFO: Requested device: NPU
# INFO: Available devices: ['CPU', 'NPU']
# INFO: Using device: NPU
```

## Troubleshooting

### NPU Not Detected

**Problem**: NPU is not available even though hardware supports it.

**Solutions**:
1. Install Intel NPU drivers from Intel's website
2. Update OpenVINO to version 2023.0 or later
3. Verify NPU is enabled in BIOS
4. Check device availability:
   ```python
   from openvino.runtime import Core
   core = Core()
   print(core.available_devices())  # Should include 'NPU'
   ```

### GPU/CUDA Not Working

**Problem**: GPU acceleration is not working.

**Solutions**:
1. Install `onnxruntime-gpu` instead of `onnxruntime`:
   ```bash
   pip uninstall onnxruntime
   pip install onnxruntime-gpu
   ```
2. Verify CUDA installation:
   ```bash
   nvidia-smi
   ```
3. Check CUDA version compatibility with onnxruntime-gpu
4. Ensure cuDNN is installed

### Performance Not Improved

**Problem**: Hardware acceleration doesn't improve performance.

**Solutions**:
1. Check that the correct device is being used (check logs)
2. For NPU: Try different `performance_hint` values
3. For GPU: Ensure model files are not too small (overhead may dominate)
4. Verify hardware is not throttling due to temperature
5. Use batch processing for better GPU utilization

### Import Errors

**Problem**: Cannot import rapidocr_hkmc.

**Solutions**:
1. Verify installation:
   ```bash
   pip list | grep rapidocr
   ```
2. Reinstall the package:
   ```bash
   pip install --force-reinstall rapidocr_hkmc
   ```
3. Check Python version compatibility (requires Python 3.6+)

## API Compatibility

RapidOCR HKMC maintains full API compatibility with the original RapidOCR package:

```python
# All original RapidOCR code works with rapidocr_hkmc
from rapidocr_hkmc import RapidOCR

ocr = RapidOCR()
result, elapse = ocr('image.jpg')

# Same parameters and return values
# result: List of detected text with coordinates
# elapse: Processing time breakdown
```

## Command Line Interface

```bash
# Basic usage
rapidocr_hkmc -img path/to/image.jpg

# With custom config
rapidocr_hkmc -img path/to/image.jpg -config config_npu.yaml

# Output to file
rapidocr_hkmc -img path/to/image.jpg -o output.txt
```

## Performance Benchmarks

Typical performance improvements (compared to CPU baseline):

| Model | Hardware | Speedup | Power Efficiency |
|-------|----------|---------|------------------|
| Cls   | NPU      | 2-3x    | 5-10x better     |
| Det   | GPU      | 5-10x   | Similar          |
| Rec   | GPU      | 5-10x   | Similar          |

*Note: Actual performance varies based on hardware, image size, and configuration.*

## License

Same license as the original RapidOCR project.

## Acknowledgments

Based on [RapidOCR](https://github.com/RapidAI/RapidOCR) by RapidAI.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review configuration examples
3. Enable debug logging to diagnose issues

---

### Original Documentation

For additional information, see [RapidOCR Documentation](https://rapidai.github.io/RapidOCRDocs/main/install_usage/rapidocr/install/)
