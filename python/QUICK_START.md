# Quick Start Guide - RapidOCR HKMC

## Installation

```bash
# Basic installation
pip install rapidocr_hkmc

# With GPU support
pip install rapidocr_hkmc onnxruntime-gpu

# From wheel (after building)
pip install dist\rapidocr_hkmc-*.whl
```

## Usage

### 1. CPU Only (Default - Works Everywhere)

```python
from rapidocr_hkmc import RapidOCR

ocr = RapidOCR()  # Uses CPU by default
result, elapse = ocr('image.jpg')
print(result)
```

### 2. NPU + GPU (Recommended for Intel + NVIDIA)

```python
from rapidocr_hkmc import RapidOCR

# Copy config_npu_gpu_hybrid.yaml to your project first
ocr = RapidOCR(config_path='config_npu_gpu_hybrid.yaml')
result, elapse = ocr('image.jpg')
```

### 3. NPU Only (Power Efficient)

```python
from rapidocr_hkmc import RapidOCR

# Copy config_npu_only.yaml to your project first
ocr = RapidOCR(config_path='config_npu_only.yaml')
result, elapse = ocr('image.jpg')
```

### 4. GPU Only (Maximum Throughput)

```python
from rapidocr_hkmc import RapidOCR

# Copy config_gpu_only.yaml to your project first
ocr = RapidOCR(config_path='config_gpu_only.yaml')
result, elapse = ocr('image.jpg')
```

## Configuration Files

| File | Use Case | Hardware Required |
|------|----------|-------------------|
| `config_cpu_only.yaml` | Maximum compatibility | Any CPU |
| `config_npu_only.yaml` | Power efficient | Intel NPU |
| `config_gpu_only.yaml` | Maximum throughput | NVIDIA GPU |
| `config_npu_gpu_hybrid.yaml` | **Recommended** | Intel NPU + NVIDIA GPU |

## Before Packaging (Development)

### Option 1: Modify Default Config

Edit `rapidocr_hkmc/config.yaml`:

```yaml
EngineConfig:
  openvino:
    device_name: "NPU"  # Change from "CPU" to "NPU"
  onnxruntime:
    use_cuda: true      # Enable GPU

Cls:
  engine_type: "openvino"      # NPU for classification
Det:
  engine_type: "onnxruntime"   # GPU for detection
Rec:
  engine_type: "onnxruntime"   # GPU for recognition
```

Then use:
```python
from rapidocr_hkmc import RapidOCR
ocr = RapidOCR()  # Uses modified default config
```

### Option 2: Use Example Configs

```python
from rapidocr_hkmc import RapidOCR

# Use example config directly
ocr = RapidOCR(config_path='config_npu_gpu_hybrid.yaml')
result, elapse = ocr('image.jpg')
```

## After Packaging (Production)

### Step 1: Copy Config to Your Project

```bash
# Windows
copy config_npu_gpu_hybrid.yaml C:\my_project\ocr_config.yaml

# Linux/Mac
cp config_npu_gpu_hybrid.yaml /my_project/ocr_config.yaml
```

### Step 2: Use in Your Application

```python
from rapidocr_hkmc import RapidOCR

# Use your custom config
ocr = RapidOCR(config_path='ocr_config.yaml')

# Or use absolute path
ocr = RapidOCR(config_path='C:/my_project/ocr_config.yaml')

result, elapse = ocr('image.jpg')
```

## Verify Hardware Acceleration

```python
import logging
from rapidocr_hkmc import RapidOCR

# Enable logging
logging.basicConfig(level=logging.INFO)

# Initialize with your config
ocr = RapidOCR(config_path='config_npu_gpu_hybrid.yaml')

# Check logs for:
# INFO: Requested device: NPU
# INFO: Using device: NPU
# INFO: CUDAExecutionProvider available

result, elapse = ocr('image.jpg')
```

## Check Available Hardware

### Check NPU

```python
from openvino.runtime import Core
print(Core().available_devices)
# Should include 'NPU' if available
```

### Check GPU/CUDA

```python
import onnxruntime
print(onnxruntime.get_available_providers())
# Should include 'CUDAExecutionProvider' if available
```

```bash
# Check NVIDIA GPU
nvidia-smi
```

## Common Issues

### NPU Not Detected
- Install Intel NPU drivers
- Update OpenVINO: `pip install --upgrade openvino`
- Check BIOS settings (enable Intel AI Boost)

### GPU Not Detected
- Install onnxruntime-gpu: `pip install onnxruntime-gpu`
- Verify CUDA installation: `nvidia-smi`
- Check CUDA version compatibility

### Config File Not Found
```python
import os
# Use absolute path
config_path = os.path.join(os.path.dirname(__file__), 'ocr_config.yaml')
ocr = RapidOCR(config_path=config_path)
```

## Performance Tips

1. **Use hybrid config** (NPU + GPU) for best balance
2. **Enable logging** to verify hardware usage
3. **Use batch processing** for multiple images
4. **Monitor resource usage** to ensure hardware is being utilized

## Example: Complete Application

```python
from rapidocr_hkmc import RapidOCR
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Initialize with best config for your hardware
ocr = RapidOCR(config_path='config_npu_gpu_hybrid.yaml')

# Process images
image_dir = Path('documents')
for image_path in image_dir.glob('*.jpg'):
    result, elapse = ocr(str(image_path))
    
    # Print results
    print(f"\n{image_path.name}:")
    for line in result:
        text = line[1]
        confidence = line[2]
        print(f"  {text} (confidence: {confidence:.2f})")
    
    # Print timing
    total_time = sum(elapse.values())
    print(f"  Total time: {total_time:.2f}s")
```

## Next Steps

1. Read [CONFIGURATION_USAGE_GUIDE.md](CONFIGURATION_USAGE_GUIDE.md) for detailed usage
2. Read [README.md](README.md) for complete documentation
3. Check [ERROR_MESSAGES_VALIDATION.md](ERROR_MESSAGES_VALIDATION.md) for troubleshooting

## Support

- Check logs with `logging.basicConfig(level=logging.DEBUG)`
- Verify hardware with the check commands above
- Review configuration examples in the config files
- See troubleshooting section in README.md
