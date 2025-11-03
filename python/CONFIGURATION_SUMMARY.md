# Configuration Summary - RapidOCR HKMC

## Quick Answer to Your Question

**"How to use these configurations before packaging wheels or importing rapidocr_hkmc?"**

### Before Packaging (Development)

**Option 1: Use example configs directly**
```python
from rapidocr_hkmc import RapidOCR

# Use any of the example config files
ocr = RapidOCR(config_path='config_npu_gpu_hybrid.yaml')
result, elapse = ocr('image.jpg')
```

**Option 2: Modify the default config**
```bash
# Edit rapidocr_hkmc/config.yaml
# Change device_name: "CPU" → "NPU"
# Change use_cuda: false → true
```
```python
from rapidocr_hkmc import RapidOCR

# Uses your modified default config
ocr = RapidOCR()
result, elapse = ocr('image.jpg')
```

### After Packaging (Production)

**Step 1: Build and install**
```bash
python setup.py bdist_wheel
pip install dist\rapidocr_hkmc-*.whl
```

**Step 2: Copy config to your project**
```bash
copy config_npu_gpu_hybrid.yaml C:\my_project\ocr_config.yaml
```

**Step 3: Use in your application**
```python
from rapidocr_hkmc import RapidOCR

ocr = RapidOCR(config_path='ocr_config.yaml')
result, elapse = ocr('image.jpg')
```

---

## Available Configuration Files

| File | Purpose | Hardware | Use When |
|------|---------|----------|----------|
| `config_cpu_only.yaml` | CPU-only | Any CPU | Testing, compatibility |
| `config_npu_only.yaml` | NPU-only | Intel NPU | Power efficiency |
| `config_gpu_only.yaml` | GPU-only | NVIDIA GPU | Maximum throughput |
| `config_npu_gpu_hybrid.yaml` | **NPU + GPU** | Intel NPU + NVIDIA GPU | **Production (recommended)** |
| `rapidocr_hkmc/config.yaml` | Default | Any | Fallback |

---

## Documentation Files Created

### 1. **QUICK_START.md** - Start here!
- Quick installation and usage examples
- Copy-paste ready code snippets
- Common issues and solutions

### 2. **CONFIGURATION_USAGE_GUIDE.md** - Detailed guide
- Complete usage scenarios
- Development vs production workflows
- Troubleshooting and best practices

### 3. **CONFIGURATION_WORKFLOW.md** - Visual guide
- Decision trees and diagrams
- File location references
- Configuration patterns

### 4. **example_usage.py** - Runnable examples
- 6 different configuration examples
- Hardware detection
- Image processing demo

### 5. **ERROR_MESSAGES_VALIDATION.md** - Troubleshooting
- Validation of all error messages
- Logging verification
- Diagnostic information

---

## Recommended Workflow

### For Development

```bash
# 1. Clone/download the project
cd rapidocr_hkmc_project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Test with example config
python
```
```python
from rapidocr_hkmc import RapidOCR

# Use hybrid config for testing
ocr = RapidOCR(config_path='config_npu_gpu_hybrid.yaml')
result, elapse = ocr('test_image.jpg')
print(result)
```

### For Production

```bash
# 1. Build wheel
python setup.py bdist_wheel

# 2. Install in production environment
pip install dist\rapidocr_hkmc-*.whl

# 3. Copy config to production
copy config_npu_gpu_hybrid.yaml C:\production\ocr_config.yaml

# 4. Use in your application
```
```python
from rapidocr_hkmc import RapidOCR

ocr = RapidOCR(config_path='C:/production/ocr_config.yaml')
result, elapse = ocr('document.jpg')
```

---

## Key Points

### ✅ DO

1. **Use example configs with `config_path` parameter** during development
2. **Copy and customize configs** for production deployment
3. **Enable logging** to verify hardware acceleration is working
4. **Test on target hardware** before production deployment
5. **Keep configs in version control** (but not in the package)

### ❌ DON'T

1. **Don't hardcode configs** in your application code
2. **Don't modify the package's default config** in production
3. **Don't assume hardware is available** - always check logs
4. **Don't skip testing** with CPU-only config first
5. **Don't package config files** in the wheel (keep them separate)

---

## Testing Your Configuration

### Quick Test Script

```python
import logging
from rapidocr_hkmc import RapidOCR

# Enable logging to see what's happening
logging.basicConfig(level=logging.INFO)

# Test your configuration
ocr = RapidOCR(config_path='config_npu_gpu_hybrid.yaml')

# Check logs for:
# - "Requested device: NPU"
# - "Using device: NPU"
# - "CUDAExecutionProvider" in providers

print("Configuration loaded successfully!")
```

### Verify Hardware

```python
# Check NPU
from openvino.runtime import Core
print("Available devices:", Core().available_devices)

# Check GPU
import onnxruntime
print("Available providers:", onnxruntime.get_available_providers())
```

---

## Example Usage Patterns

### Pattern 1: Simple (Recommended for most users)

```python
from rapidocr_hkmc import RapidOCR

# Copy config_npu_gpu_hybrid.yaml to your project first
ocr = RapidOCR(config_path='config_npu_gpu_hybrid.yaml')
result, elapse = ocr('image.jpg')
```

### Pattern 2: Environment-Based

```python
import os
from rapidocr_hkmc import RapidOCR

env = os.getenv('ENVIRONMENT', 'development')
config_map = {
    'development': 'config_cpu_only.yaml',
    'production': 'config_npu_gpu_hybrid.yaml'
}

ocr = RapidOCR(config_path=config_map[env])
```

### Pattern 3: Auto-Detection

```python
from rapidocr_hkmc import RapidOCR
from openvino.runtime import Core
import onnxruntime

# Detect available hardware
has_npu = 'NPU' in Core().available_devices
has_cuda = 'CUDAExecutionProvider' in onnxruntime.get_available_providers()

# Select best config
if has_npu and has_cuda:
    config = 'config_npu_gpu_hybrid.yaml'
elif has_npu:
    config = 'config_npu_only.yaml'
elif has_cuda:
    config = 'config_gpu_only.yaml'
else:
    config = None  # Use default CPU

ocr = RapidOCR(config_path=config) if config else RapidOCR()
```

---

## Common Questions

### Q: Do I need to modify the package's config.yaml?
**A:** No, use the example configs with `config_path` parameter instead.

### Q: Where should I put my config file in production?
**A:** In your application directory, not in the installed package.

### Q: Can I use multiple configs in the same application?
**A:** Yes, create multiple RapidOCR instances with different configs.

### Q: What if my hardware isn't available?
**A:** The system automatically falls back to CPU with a warning.

### Q: How do I know which hardware is being used?
**A:** Enable logging with `logging.basicConfig(level=logging.INFO)`.

---

## Next Steps

1. **Read QUICK_START.md** for immediate usage
2. **Run example_usage.py** to see all configurations in action
3. **Test with your images** using different configs
4. **Choose the best config** for your hardware
5. **Deploy to production** with your custom config

---

## Support and Troubleshooting

If you encounter issues:

1. **Check logs** - Enable `logging.basicConfig(level=logging.DEBUG)`
2. **Verify hardware** - Use the verification scripts above
3. **Test with CPU** - Start with `config_cpu_only.yaml`
4. **Read error messages** - They include installation instructions
5. **Check documentation** - See ERROR_MESSAGES_VALIDATION.md

---

## Summary

**The simplest approach:**

1. **Development**: Use `RapidOCR(config_path='config_npu_gpu_hybrid.yaml')`
2. **Production**: Copy config to your project, use `RapidOCR(config_path='ocr_config.yaml')`

This keeps your configuration separate from the package and makes it easy to adjust settings without rebuilding the wheel.

**That's it!** You're ready to use RapidOCR HKMC with hardware acceleration.
