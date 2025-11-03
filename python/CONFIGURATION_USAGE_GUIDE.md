# Configuration Usage Guide for RapidOCR HKMC

## Overview

This guide explains how to use the different configuration files for NPU/GPU acceleration in rapidocr_hkmc, both during development and after installation.

## Configuration Files Available

1. **`config_cpu_only.yaml`** - CPU-only (maximum compatibility)
2. **`config_npu_only.yaml`** - NPU-only (power efficient)
3. **`config_gpu_only.yaml`** - GPU-only (maximum throughput)
4. **`config_npu_gpu_hybrid.yaml`** - NPU + GPU (recommended)
5. **`rapidocr_hkmc/config.yaml`** - Default configuration (included in package)

---

## Usage Scenarios

### Scenario 1: Development/Testing (Before Packaging)

When you're developing or testing locally without installing the wheel package:

#### Option A: Modify the Default Configuration

Edit `rapidocr_hkmc/config.yaml` directly:

```bash
# Open the default config
notepad rapidocr_hkmc\config.yaml

# Or use your preferred editor
code rapidocr_hkmc\config.yaml
```

**For NPU + GPU (Recommended)**:
```yaml
EngineConfig:
  openvino:
    device_name: "NPU"              # Change from "CPU" to "NPU"
    performance_hint: "LATENCY"
  
  onnxruntime:
    use_cuda: true                  # Enable GPU

Cls:
  engine_type: "openvino"           # Use NPU for classification

Det:
  engine_type: "onnxruntime"        # Use GPU for detection

Rec:
  engine_type: "onnxruntime"        # Use GPU for recognition
```

Then run your code:

```python
from rapidocr_hkmc import RapidOCR

# Uses the modified default config
ocr = RapidOCR()
result, elapse = ocr('test_image.jpg')
print(result)
```

#### Option B: Use a Specific Configuration File

Copy one of the example configs and use it:

```python
from rapidocr_hkmc import RapidOCR

# Use NPU + GPU hybrid configuration
ocr = RapidOCR(config_path='config_npu_gpu_hybrid.yaml')

# Or use NPU-only
ocr = RapidOCR(config_path='config_npu_only.yaml')

# Or use GPU-only
ocr = RapidOCR(config_path='config_gpu_only.yaml')

# Process image
result, elapse = ocr('test_image.jpg')
print(result)
```

#### Option C: Programmatic Configuration

Override specific settings in code:

```python
from rapidocr_hkmc import RapidOCR
from omegaconf import OmegaConf

# Load base config
config = OmegaConf.load('rapidocr_hkmc/config.yaml')

# Modify for NPU
config.EngineConfig.openvino.device_name = "NPU"
config.Cls.engine_type = "openvino"

# Modify for GPU
config.EngineConfig.onnxruntime.use_cuda = True
config.Det.engine_type = "onnxruntime"
config.Rec.engine_type = "onnxruntime"

# Initialize with custom config
ocr = RapidOCR(config=config)
result, elapse = ocr('test_image.jpg')
```

---

### Scenario 2: After Building the Wheel Package

After building the wheel with `python setup.py bdist_wheel`:

#### Step 1: Install the Package

```bash
# Install the wheel
pip install dist\rapidocr_hkmc-*.whl

# Or install with GPU support
pip install dist\rapidocr_hkmc-*.whl onnxruntime-gpu
```

#### Step 2: Create Your Configuration File

Copy one of the example configs to your project directory:

```bash
# Copy the hybrid config (recommended)
copy config_npu_gpu_hybrid.yaml my_project\ocr_config.yaml

# Or copy NPU-only config
copy config_npu_only.yaml my_project\ocr_config.yaml

# Or copy GPU-only config
copy config_gpu_only.yaml my_project\ocr_config.yaml
```

#### Step 3: Use the Configuration

**Method 1: Specify config path**

```python
from rapidocr_hkmc import RapidOCR

# Use your custom config file
ocr = RapidOCR(config_path='ocr_config.yaml')

# Or use absolute path
ocr = RapidOCR(config_path='C:/my_project/ocr_config.yaml')

result, elapse = ocr('image.jpg')
print(result)
```

**Method 2: Use default config (CPU-only)**

```python
from rapidocr_hkmc import RapidOCR

# Uses the default config from the installed package
# This will use CPU by default
ocr = RapidOCR()

result, elapse = ocr('image.jpg')
print(result)
```

**Method 3: Programmatic configuration**

```python
from rapidocr_hkmc import RapidOCR
from omegaconf import OmegaConf

# Create config from scratch
config = OmegaConf.create({
    'EngineConfig': {
        'openvino': {
            'device_name': 'NPU',
            'performance_hint': 'LATENCY'
        },
        'onnxruntime': {
            'use_cuda': True,
            'cuda_ep_cfg': {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'cudnn_conv_algo_search': 'EXHAUSTIVE'
            }
        }
    },
    'Cls': {'engine_type': 'openvino'},
    'Det': {'engine_type': 'onnxruntime'},
    'Rec': {'engine_type': 'onnxruntime'}
})

ocr = RapidOCR(config=config)
result, elapse = ocr('image.jpg')
```

---

## Recommended Workflow

### For Development

1. **Start with CPU-only** to verify basic functionality:
   ```python
   from rapidocr_hkmc import RapidOCR
   ocr = RapidOCR(config_path='config_cpu_only.yaml')
   ```

2. **Test NPU** if available:
   ```python
   ocr = RapidOCR(config_path='config_npu_only.yaml')
   ```

3. **Test GPU** if available:
   ```python
   ocr = RapidOCR(config_path='config_gpu_only.yaml')
   ```

4. **Use hybrid** for production:
   ```python
   ocr = RapidOCR(config_path='config_npu_gpu_hybrid.yaml')
   ```

### For Production Deployment

1. **Choose the appropriate configuration** based on your hardware:
   - Intel CPU with NPU + NVIDIA GPU → `config_npu_gpu_hybrid.yaml`
   - Intel CPU with NPU only → `config_npu_only.yaml`
   - NVIDIA GPU only → `config_gpu_only.yaml`
   - No special hardware → `config_cpu_only.yaml`

2. **Copy the config to your deployment directory**:
   ```bash
   copy config_npu_gpu_hybrid.yaml C:\production\ocr_config.yaml
   ```

3. **Use the config in your application**:
   ```python
   from rapidocr_hkmc import RapidOCR
   import os
   
   # Use environment variable for flexibility
   config_path = os.getenv('OCR_CONFIG_PATH', 'ocr_config.yaml')
   ocr = RapidOCR(config_path=config_path)
   ```

---

## Configuration File Locations

### Before Packaging (Development)

```
project_root/
├── rapidocr_hkmc/
│   └── config.yaml              # Default config (modify this for dev)
├── config_npu_only.yaml         # Example configs
├── config_gpu_only.yaml
├── config_npu_gpu_hybrid.yaml
└── config_cpu_only.yaml
```

### After Installation (Production)

```
your_project/
├── ocr_config.yaml              # Your custom config (copied from examples)
└── main.py                      # Your application
```

The installed package includes a default config, but you should provide your own for production.

---

## Quick Start Examples

### Example 1: Simple CPU Usage

```python
from rapidocr_hkmc import RapidOCR

# No config needed - uses CPU by default
ocr = RapidOCR()
result, elapse = ocr('document.jpg')

for line in result:
    print(f"Text: {line[1]}, Confidence: {line[2]}")
```

### Example 2: NPU + GPU (Recommended)

```python
from rapidocr_hkmc import RapidOCR
import logging

# Enable logging to see device selection
logging.basicConfig(level=logging.INFO)

# Use hybrid configuration
ocr = RapidOCR(config_path='config_npu_gpu_hybrid.yaml')

# Process image
result, elapse = ocr('document.jpg')

# Check logs for:
# INFO: Requested device: NPU
# INFO: Using device: NPU
# INFO: CUDAExecutionProvider available
```

### Example 3: Batch Processing with GPU

```python
from rapidocr_hkmc import RapidOCR
from pathlib import Path

# Use GPU-only config for maximum throughput
ocr = RapidOCR(config_path='config_gpu_only.yaml')

# Process multiple images
image_dir = Path('documents')
for image_path in image_dir.glob('*.jpg'):
    result, elapse = ocr(str(image_path))
    print(f"Processed {image_path.name} in {sum(elapse.values()):.2f}s")
    print(f"  Detection: {elapse.get('det', 0):.2f}s")
    print(f"  Recognition: {elapse.get('rec', 0):.2f}s")
```

### Example 4: Dynamic Configuration Selection

```python
from rapidocr_hkmc import RapidOCR
from openvino.runtime import Core
import onnxruntime

def select_best_config():
    """Automatically select the best configuration based on available hardware."""
    
    # Check for NPU
    has_npu = 'NPU' in Core().available_devices
    
    # Check for CUDA
    has_cuda = 'CUDAExecutionProvider' in onnxruntime.get_available_providers()
    
    if has_npu and has_cuda:
        return 'config_npu_gpu_hybrid.yaml'
    elif has_npu:
        return 'config_npu_only.yaml'
    elif has_cuda:
        return 'config_gpu_only.yaml'
    else:
        return 'config_cpu_only.yaml'

# Use the best available configuration
config_path = select_best_config()
print(f"Using configuration: {config_path}")

ocr = RapidOCR(config_path=config_path)
result, elapse = ocr('document.jpg')
```

---

## Troubleshooting Configuration Issues

### Issue 1: Config file not found

**Error**: `FileNotFoundError: config_npu_gpu_hybrid.yaml`

**Solution**: Use absolute path or ensure file is in current directory
```python
import os
config_path = os.path.join(os.path.dirname(__file__), 'config_npu_gpu_hybrid.yaml')
ocr = RapidOCR(config_path=config_path)
```

### Issue 2: NPU not detected

**Symptom**: Warning message "NPU not available, falling back to CPU"

**Solution**: 
1. Check if NPU is available:
   ```python
   from openvino.runtime import Core
   print(Core().available_devices)  # Should include 'NPU'
   ```
2. If not available, install Intel NPU drivers
3. Or use CPU/GPU-only config instead

### Issue 3: GPU not detected

**Symptom**: Warning message "CUDA not available, falling back to CPU"

**Solution**:
1. Install onnxruntime-gpu:
   ```bash
   pip uninstall onnxruntime
   pip install onnxruntime-gpu
   ```
2. Verify CUDA installation:
   ```bash
   nvidia-smi
   ```
3. Check available providers:
   ```python
   import onnxruntime
   print(onnxruntime.get_available_providers())
   ```

### Issue 4: Configuration not taking effect

**Symptom**: Using config file but still running on CPU

**Solution**: Enable logging to verify configuration is loaded
```python
import logging
logging.basicConfig(level=logging.INFO)

ocr = RapidOCR(config_path='config_npu_gpu_hybrid.yaml')
# Check logs for device selection messages
```

---

## Best Practices

1. **Always test with CPU first** to verify basic functionality
2. **Use logging** to verify hardware acceleration is working
3. **Keep configs in version control** but not in the package
4. **Use environment variables** for config paths in production
5. **Provide fallback configs** for different deployment environments
6. **Document hardware requirements** for your chosen configuration
7. **Test on target hardware** before production deployment

---

## Summary

**Before Packaging (Development)**:
- Modify `rapidocr_hkmc/config.yaml` directly, OR
- Use example configs with `config_path` parameter

**After Packaging (Production)**:
- Copy example config to your project directory
- Use `config_path` parameter to specify your config
- Or use programmatic configuration for dynamic setups

**Recommended Approach**:
```python
# Development
ocr = RapidOCR(config_path='config_npu_gpu_hybrid.yaml')

# Production
ocr = RapidOCR(config_path='/path/to/production/ocr_config.yaml')
```

This approach keeps your configuration separate from the package and allows easy updates without rebuilding the wheel.
