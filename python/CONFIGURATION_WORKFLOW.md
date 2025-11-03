# Configuration Workflow Diagram

## Overview: How to Use Configurations

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONFIGURATION WORKFLOW                        │
└─────────────────────────────────────────────────────────────────┘

                    ┌──────────────────┐
                    │  Choose Your     │
                    │  Use Case        │
                    └────────┬─────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
        ┌───────▼────────┐       ┌───────▼────────┐
        │  DEVELOPMENT   │       │   PRODUCTION   │
        │  (Before Pkg)  │       │  (After Pkg)   │
        └───────┬────────┘       └───────┬────────┘
                │                         │
                │                         │
    ┌───────────┴──────────┐             │
    │                      │             │
┌───▼────┐          ┌──────▼───┐        │
│ Option │          │ Option   │        │
│   A    │          │    B     │        │
└───┬────┘          └──────┬───┘        │
    │                      │             │
    │                      │             │
    ▼                      ▼             ▼
```

---

## Development Workflow (Before Packaging)

### Option A: Modify Default Config

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Edit rapidocr_hkmc/config.yaml                           │
│    - Change device_name: "CPU" → "NPU"                      │
│    - Change use_cuda: false → true                          │
│    - Set engine_type for each model                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Use in code without config_path parameter                │
│                                                              │
│    from rapidocr_hkmc import RapidOCR                       │
│    ocr = RapidOCR()  # Uses modified default                │
│    result, elapse = ocr('image.jpg')                        │
└─────────────────────────────────────────────────────────────┘
```

**Pros**: Simple, no extra files needed  
**Cons**: Changes affect all development, harder to switch configs

---

### Option B: Use Example Configs

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Choose appropriate example config:                       │
│    • config_npu_gpu_hybrid.yaml (recommended)               │
│    • config_npu_only.yaml                                   │
│    • config_gpu_only.yaml                                   │
│    • config_cpu_only.yaml                                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Use in code with config_path parameter                   │
│                                                              │
│    from rapidocr_hkmc import RapidOCR                       │
│    ocr = RapidOCR(config_path='config_npu_gpu_hybrid.yaml')│
│    result, elapse = ocr('image.jpg')                        │
└─────────────────────────────────────────────────────────────┘
```

**Pros**: Easy to switch configs, clean separation  
**Cons**: Need to manage config files

---

## Production Workflow (After Packaging)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Build and install wheel package                          │
│                                                              │
│    python setup.py bdist_wheel                              │
│    pip install dist\rapidocr_hkmc-*.whl                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Copy example config to your project                      │
│                                                              │
│    copy config_npu_gpu_hybrid.yaml C:\my_app\ocr_config.yaml│
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Customize config for your environment                    │
│    - Adjust device_id for multi-GPU                         │
│    - Tune performance_hint                                  │
│    - Modify batch sizes                                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Use in your application                                  │
│                                                              │
│    from rapidocr_hkmc import RapidOCR                       │
│    ocr = RapidOCR(config_path='ocr_config.yaml')           │
│    result, elapse = ocr('image.jpg')                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Configuration Selection Decision Tree

```
                    Start
                      │
                      ▼
        ┌─────────────────────────┐
        │ Do you have Intel NPU?  │
        └──────┬──────────┬───────┘
               │          │
             Yes         No
               │          │
               ▼          ▼
    ┌──────────────┐  ┌──────────────┐
    │ Do you have  │  │ Do you have  │
    │ NVIDIA GPU?  │  │ NVIDIA GPU?  │
    └──┬───────┬───┘  └──┬───────┬───┘
       │       │         │       │
      Yes     No        Yes     No
       │       │         │       │
       ▼       ▼         ▼       ▼
    ┌─────┐ ┌─────┐  ┌─────┐ ┌─────┐
    │ NPU │ │ NPU │  │ GPU │ │ CPU │
    │  +  │ │Only │  │Only │ │Only │
    │ GPU │ │     │  │     │ │     │
    └─────┘ └─────┘  └─────┘ └─────┘
       │       │         │       │
       ▼       ▼         ▼       ▼
    config_ config_  config_  config_
    npu_gpu npu_    gpu_     cpu_
    hybrid  only    only     only
    .yaml   .yaml   .yaml    .yaml
```

---

## File Locations by Stage

### Development Stage

```
project_root/
├── rapidocr_hkmc/
│   ├── config.yaml                    ← Default (can modify)
│   ├── inference_engine/
│   └── ...
├── config_npu_only.yaml               ← Example configs
├── config_gpu_only.yaml               ← (use with config_path)
├── config_npu_gpu_hybrid.yaml         ←
├── config_cpu_only.yaml               ←
├── setup.py
└── your_test_script.py
```

**Usage in development**:
```python
# Option A: Modified default
ocr = RapidOCR()

# Option B: Example config
ocr = RapidOCR(config_path='config_npu_gpu_hybrid.yaml')
```

---

### Production Stage

```
your_application/
├── ocr_config.yaml                    ← Your custom config
│                                        (copied from example)
├── main.py                            ← Your application
├── requirements.txt
└── data/
    └── images/
```

**Usage in production**:
```python
# Use your custom config
ocr = RapidOCR(config_path='ocr_config.yaml')

# Or use absolute path
ocr = RapidOCR(config_path='/app/config/ocr_config.yaml')

# Or use environment variable
import os
config_path = os.getenv('OCR_CONFIG', 'ocr_config.yaml')
ocr = RapidOCR(config_path=config_path)
```

---

## Configuration Priority

When RapidOCR initializes, it looks for configuration in this order:

```
1. config parameter (programmatic)
   ↓ (if not provided)
2. config_path parameter (file path)
   ↓ (if not provided)
3. Default config from installed package
```

**Example**:
```python
from omegaconf import OmegaConf

# Priority 1: Programmatic config (highest)
config = OmegaConf.create({'EngineConfig': {...}})
ocr = RapidOCR(config=config)

# Priority 2: Config file path
ocr = RapidOCR(config_path='my_config.yaml')

# Priority 3: Default config (lowest)
ocr = RapidOCR()
```

---

## Common Patterns

### Pattern 1: Environment-Based Configuration

```python
import os
from rapidocr_hkmc import RapidOCR

# Select config based on environment
env = os.getenv('ENVIRONMENT', 'development')

config_map = {
    'development': 'config_cpu_only.yaml',
    'staging': 'config_gpu_only.yaml',
    'production': 'config_npu_gpu_hybrid.yaml'
}

config_path = config_map.get(env, 'config_cpu_only.yaml')
ocr = RapidOCR(config_path=config_path)
```

### Pattern 2: Auto-Detection

```python
from rapidocr_hkmc import RapidOCR
from openvino.runtime import Core
import onnxruntime

def get_best_config():
    has_npu = 'NPU' in Core().available_devices
    has_cuda = 'CUDAExecutionProvider' in onnxruntime.get_available_providers()
    
    if has_npu and has_cuda:
        return 'config_npu_gpu_hybrid.yaml'
    elif has_npu:
        return 'config_npu_only.yaml'
    elif has_cuda:
        return 'config_gpu_only.yaml'
    return 'config_cpu_only.yaml'

ocr = RapidOCR(config_path=get_best_config())
```

### Pattern 3: Dynamic Override

```python
from rapidocr_hkmc import RapidOCR
from omegaconf import OmegaConf

# Load base config
config = OmegaConf.load('config_npu_gpu_hybrid.yaml')

# Override specific settings
config.EngineConfig.onnxruntime.cuda_ep_cfg.device_id = 1  # Use GPU 1
config.Cls.cls_batch_num = 12  # Increase batch size

ocr = RapidOCR(config=config)
```

---

## Summary

| Stage | Config Location | Usage Method |
|-------|----------------|--------------|
| **Development** | `rapidocr_hkmc/config.yaml` | Modify default, use `RapidOCR()` |
| **Development** | `config_*.yaml` (examples) | Use `RapidOCR(config_path='...')` |
| **Production** | Your project directory | Copy example, use `config_path` |
| **Production** | Any location | Use absolute path or env var |

**Recommended Approach**:
- **Development**: Use example configs with `config_path` parameter
- **Production**: Copy and customize config in your project directory

This keeps configuration separate from code and makes it easy to adjust settings without modifying the package.
