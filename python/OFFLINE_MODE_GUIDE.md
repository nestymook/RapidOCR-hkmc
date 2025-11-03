# Offline Mode Guide - RapidOCR HKMC

## Overview

RapidOCR HKMC **fully supports offline mode** by allowing you to specify local model paths in the configuration file. This is essential for:

- Air-gapped environments (no internet access)
- Production deployments with pre-downloaded models
- Faster initialization (no download time)
- Version control of specific model versions
- Compliance with security policies

## How It Works

### Default Behavior (Online Mode)

When `model_path` is `null` (default), the system:
1. Automatically downloads models from the internet
2. Caches them locally in the default model directory
3. Reuses cached models on subsequent runs

### Offline Mode

When you specify `model_path` in the config:
1. The system uses your local model file directly
2. No internet connection required
3. No automatic downloads
4. Full control over model versions

---

## Configuration Parameters

Each model (Det, Cls, Rec) supports these parameters:

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `model_path` | string or null | Path to a single model file | `"C:/models/det.onnx"` |
| `model_dir` | string or null | Path to a directory containing model files | `"C:/models/det_model/"` |

**Priority**: `model_path` takes precedence over `model_dir`

---

## Quick Start - Offline Mode

### Step 1: Download Models

First, download the models you need. You can do this once while online:

```python
from rapidocr_hkmc import RapidOCR

# Run once with internet to download models
ocr = RapidOCR()

# Models are cached in the default location
# Check logs for the cache location
```

Or download manually from the RapidOCR model repository.

### Step 2: Create Offline Configuration

Create a config file with local model paths:

```yaml
# config_offline.yaml

Det:
  engine_type: "onnxruntime"
  model_path: "C:/models/ch_PP-OCRv4_det_infer.onnx"  # Absolute path
  # ... other Det parameters

Cls:
  engine_type: "openvino"
  model_path: "C:/models/ch_ppocr_mobile_v2.0_cls_infer.xml"  # OpenVINO model
  # ... other Cls parameters

Rec:
  engine_type: "onnxruntime"
  model_path: "C:/models/ch_PP-OCRv4_rec_infer.onnx"
  rec_keys_path: "C:/models/ppocr_keys_v1.txt"  # Character dictionary
  # ... other Rec parameters
```

### Step 3: Use Offline Configuration

```python
from rapidocr_hkmc import RapidOCR

# Works completely offline
ocr = RapidOCR(config_path='config_offline.yaml')
result, elapse = ocr('image.jpg')
```

---

## Complete Offline Configuration Examples

### Example 1: Offline with Absolute Paths

```yaml
# config_offline_absolute.yaml

Global:
  text_score: 0.5
  use_det: true
  use_cls: true
  use_rec: true

EngineConfig:
  openvino:
    device_name: "NPU"
    performance_hint: "LATENCY"
  
  onnxruntime:
    use_cuda: true
    cuda_ep_cfg:
      device_id: 0

Det:
  engine_type: "onnxruntime"
  lang_type: "ch"
  model_type: "mobile"
  ocr_version: "PP-OCRv4"
  task_type: "det"
  
  # Offline: Specify local model path
  model_path: "C:/ocr_models/ch_PP-OCRv4_det_infer.onnx"
  
  limit_side_len: 736
  limit_type: min
  thresh: 0.3
  box_thresh: 0.5

Cls:
  engine_type: "openvino"
  lang_type: "ch"
  model_type: "mobile"
  ocr_version: "PP-OCRv4"
  task_type: "cls"
  
  # Offline: Specify local model path (OpenVINO uses .xml)
  model_path: "C:/ocr_models/ch_ppocr_mobile_v2.0_cls_infer.xml"
  
  cls_image_shape: [3, 48, 192]
  cls_batch_num: 6
  cls_thresh: 0.9

Rec:
  engine_type: "onnxruntime"
  lang_type: "ch"
  model_type: "mobile"
  ocr_version: "PP-OCRv4"
  task_type: "rec"
  
  # Offline: Specify local model path
  model_path: "C:/ocr_models/ch_PP-OCRv4_rec_infer.onnx"
  
  # Offline: Specify local character dictionary
  rec_keys_path: "C:/ocr_models/ppocr_keys_v1.txt"
  
  rec_img_shape: [3, 48, 320]
  rec_batch_num: 6
```

### Example 2: Offline with Relative Paths

```yaml
# config_offline_relative.yaml

Det:
  engine_type: "onnxruntime"
  model_path: "./models/det.onnx"  # Relative to current directory

Cls:
  engine_type: "openvino"
  model_path: "./models/cls.xml"

Rec:
  engine_type: "onnxruntime"
  model_path: "./models/rec.onnx"
  rec_keys_path: "./models/keys.txt"
```

**Project Structure**:
```
my_project/
├── config_offline_relative.yaml
├── models/
│   ├── det.onnx
│   ├── cls.xml
│   ├── cls.bin          # OpenVINO also needs .bin file
│   ├── rec.onnx
│   └── keys.txt
└── main.py
```

### Example 3: Offline with model_dir

```yaml
# config_offline_dir.yaml

Det:
  engine_type: "onnxruntime"
  model_dir: "C:/ocr_models/det_model/"  # Directory containing model files

Cls:
  engine_type: "openvino"
  model_dir: "C:/ocr_models/cls_model/"

Rec:
  engine_type: "onnxruntime"
  model_dir: "C:/ocr_models/rec_model/"
  rec_keys_path: "C:/ocr_models/ppocr_keys_v1.txt"
```

---

## Model File Requirements

### For ONNXRuntime (Det, Rec)

**Required Files**:
- `.onnx` file (the model)

**Example**:
```
models/
└── ch_PP-OCRv4_det_infer.onnx
```

**Configuration**:
```yaml
Det:
  engine_type: "onnxruntime"
  model_path: "models/ch_PP-OCRv4_det_infer.onnx"
```

### For OpenVINO (Cls)

**Required Files**:
- `.xml` file (model structure)
- `.bin` file (model weights) - must be in same directory

**Example**:
```
models/
├── ch_ppocr_mobile_v2.0_cls_infer.xml
└── ch_ppocr_mobile_v2.0_cls_infer.bin
```

**Configuration**:
```yaml
Cls:
  engine_type: "openvino"
  model_path: "models/ch_ppocr_mobile_v2.0_cls_infer.xml"
  # .bin file is automatically found in same directory
```

### Character Dictionary (Rec)

**Required for Recognition**:
- `.txt` file containing character list

**Example**:
```
models/
└── ppocr_keys_v1.txt
```

**Configuration**:
```yaml
Rec:
  engine_type: "onnxruntime"
  model_path: "models/ch_PP-OCRv4_rec_infer.onnx"
  rec_keys_path: "models/ppocr_keys_v1.txt"
```

---

## Finding Model Files

### Option 1: Use Default Cache

After running online once, models are cached:

**Windows**:
```
C:\Users\<username>\.rapidocr\models\
```

**Linux/Mac**:
```
~/.rapidocr/models/
```

You can copy these files to your offline environment.

### Option 2: Download from Repository

Models are available from the RapidOCR repository:
- https://github.com/RapidAI/RapidOCR

### Option 3: Extract from Logs

Run once with logging enabled to see download URLs:

```python
import logging
logging.basicConfig(level=logging.INFO)

from rapidocr_hkmc import RapidOCR
ocr = RapidOCR()

# Check logs for download URLs and cache locations
```

---

## Complete Offline Setup Guide

### Step-by-Step: Prepare for Offline Deployment

#### 1. Download Models (Online Environment)

```python
import logging
from rapidocr_hkmc import RapidOCR
from pathlib import Path

# Enable logging to see cache location
logging.basicConfig(level=logging.INFO)

# Initialize to download models
ocr = RapidOCR()

# Models are now cached locally
# Check logs for cache location
```

#### 2. Locate Cached Models

Check the logs or default location:

```python
from pathlib import Path
import os

# Default cache location
if os.name == 'nt':  # Windows
    cache_dir = Path.home() / '.rapidocr' / 'models'
else:  # Linux/Mac
    cache_dir = Path.home() / '.rapidocr' / 'models'

print(f"Model cache: {cache_dir}")

# List cached models
for model_file in cache_dir.glob('*'):
    print(f"  {model_file.name}")
```

#### 3. Copy Models to Deployment Location

```bash
# Create models directory in your project
mkdir C:\my_project\models

# Copy models from cache
copy %USERPROFILE%\.rapidocr\models\* C:\my_project\models\
```

#### 4. Create Offline Configuration

```yaml
# C:\my_project\config_offline.yaml

Det:
  engine_type: "onnxruntime"
  model_path: "C:/my_project/models/ch_PP-OCRv4_det_infer.onnx"
  # ... other parameters

Cls:
  engine_type: "openvino"
  model_path: "C:/my_project/models/ch_ppocr_mobile_v2.0_cls_infer.xml"
  # ... other parameters

Rec:
  engine_type: "onnxruntime"
  model_path: "C:/my_project/models/ch_PP-OCRv4_rec_infer.onnx"
  rec_keys_path: "C:/my_project/models/ppocr_keys_v1.txt"
  # ... other parameters
```

#### 5. Test Offline (Disconnect Internet)

```python
from rapidocr_hkmc import RapidOCR

# This should work without internet
ocr = RapidOCR(config_path='C:/my_project/config_offline.yaml')
result, elapse = ocr('test_image.jpg')

print("Offline mode working!" if result else "Check configuration")
```

#### 6. Package for Deployment

```
deployment_package/
├── config_offline.yaml
├── models/
│   ├── ch_PP-OCRv4_det_infer.onnx
│   ├── ch_ppocr_mobile_v2.0_cls_infer.xml
│   ├── ch_ppocr_mobile_v2.0_cls_infer.bin
│   ├── ch_PP-OCRv4_rec_infer.onnx
│   └── ppocr_keys_v1.txt
├── requirements.txt
└── main.py
```

---

## Usage Examples

### Example 1: Simple Offline Usage

```python
from rapidocr_hkmc import RapidOCR

# Use offline configuration
ocr = RapidOCR(config_path='config_offline.yaml')

# Process images
result, elapse = ocr('document.jpg')

for line in result:
    print(f"Text: {line[1]}")
```

### Example 2: Verify Offline Mode

```python
import logging
from rapidocr_hkmc import RapidOCR

# Enable logging to verify no downloads
logging.basicConfig(level=logging.INFO)

ocr = RapidOCR(config_path='config_offline.yaml')

# Check logs - should NOT see any download messages
# Should see: "Using C:/my_project/models/..."
```

### Example 3: Programmatic Offline Configuration

```python
from rapidocr_hkmc import RapidOCR
from omegaconf import OmegaConf

# Create offline config programmatically
config = OmegaConf.create({
    'Det': {
        'engine_type': 'onnxruntime',
        'model_path': 'C:/models/det.onnx',
        # ... other parameters
    },
    'Cls': {
        'engine_type': 'openvino',
        'model_path': 'C:/models/cls.xml',
        # ... other parameters
    },
    'Rec': {
        'engine_type': 'onnxruntime',
        'model_path': 'C:/models/rec.onnx',
        'rec_keys_path': 'C:/models/keys.txt',
        # ... other parameters
    }
})

ocr = RapidOCR(config=config)
```

### Example 4: Environment-Based Paths

```python
import os
from rapidocr_hkmc import RapidOCR
from omegaconf import OmegaConf

# Load base config
config = OmegaConf.load('config_offline.yaml')

# Override with environment variables
model_dir = os.getenv('OCR_MODEL_DIR', 'C:/default/models')

config.Det.model_path = f"{model_dir}/det.onnx"
config.Cls.model_path = f"{model_dir}/cls.xml"
config.Rec.model_path = f"{model_dir}/rec.onnx"
config.Rec.rec_keys_path = f"{model_dir}/keys.txt"

ocr = RapidOCR(config=config)
```

---

## Troubleshooting Offline Mode

### Issue 1: Model File Not Found

**Error**: `FileNotFoundError: model_path not found`

**Solutions**:
1. Verify file path is correct (use absolute paths)
2. Check file exists: `Path('model.onnx').exists()`
3. Use forward slashes or raw strings: `r"C:\models\det.onnx"`

```python
from pathlib import Path

model_path = "C:/models/det.onnx"
if not Path(model_path).exists():
    print(f"Model not found: {model_path}")
else:
    print(f"Model found: {model_path}")
```

### Issue 2: OpenVINO .bin File Missing

**Error**: OpenVINO fails to load model

**Solution**: Ensure both .xml and .bin files are present

```python
from pathlib import Path

xml_path = Path("models/cls.xml")
bin_path = xml_path.with_suffix('.bin')

print(f"XML exists: {xml_path.exists()}")
print(f"BIN exists: {bin_path.exists()}")  # Must be True
```

### Issue 3: Character Dictionary Not Found

**Error**: Recognition fails or produces garbage

**Solution**: Specify `rec_keys_path` in config

```yaml
Rec:
  model_path: "models/rec.onnx"
  rec_keys_path: "models/ppocr_keys_v1.txt"  # Required!
```

### Issue 4: Still Trying to Download

**Symptom**: System attempts download despite model_path

**Solution**: Ensure model_path is not null

```yaml
# ❌ Wrong - will download
Det:
  model_path: null

# ✅ Correct - uses local file
Det:
  model_path: "models/det.onnx"
```

---

## Best Practices

### 1. Use Absolute Paths in Production

```yaml
# ✅ Recommended for production
Det:
  model_path: "C:/production/ocr/models/det.onnx"

# ⚠️ Relative paths depend on working directory
Det:
  model_path: "./models/det.onnx"
```

### 2. Verify Models Before Deployment

```python
from pathlib import Path

def verify_offline_models(config_path):
    """Verify all model files exist before deployment."""
    from omegaconf import OmegaConf
    
    config = OmegaConf.load(config_path)
    
    models_to_check = [
        ('Det', config.Det.model_path),
        ('Cls', config.Cls.model_path),
        ('Rec', config.Rec.model_path),
        ('Rec Keys', config.Rec.rec_keys_path),
    ]
    
    all_exist = True
    for name, path in models_to_check:
        if path and not Path(path).exists():
            print(f"❌ {name} not found: {path}")
            all_exist = False
        else:
            print(f"✅ {name} found: {path}")
    
    return all_exist

# Verify before deployment
if verify_offline_models('config_offline.yaml'):
    print("\n✅ All models ready for offline deployment")
else:
    print("\n❌ Some models missing - check paths")
```

### 3. Include Models in Deployment Package

```
deployment/
├── app/
│   └── main.py
├── config/
│   └── config_offline.yaml
├── models/                    # Include all models
│   ├── det.onnx
│   ├── cls.xml
│   ├── cls.bin
│   ├── rec.onnx
│   └── keys.txt
└── requirements.txt
```

### 4. Document Model Versions

```yaml
# config_offline.yaml

# Model Information:
# - Det: PP-OCRv4 Detection Model (2024-01-15)
# - Cls: PP-OCRv2 Classification Model (2023-06-20)
# - Rec: PP-OCRv4 Recognition Model (2024-01-15)
# - Keys: Chinese character set v1

Det:
  model_path: "models/ch_PP-OCRv4_det_infer.onnx"
  # ... rest of config
```

---

## Summary

### Offline Mode Checklist

- ✅ Download models once while online
- ✅ Copy models to deployment location
- ✅ Create config with `model_path` parameters
- ✅ Use absolute paths for production
- ✅ Include both .xml and .bin for OpenVINO
- ✅ Specify `rec_keys_path` for recognition
- ✅ Verify all files exist before deployment
- ✅ Test completely offline (disconnect internet)

### Key Configuration Parameters

```yaml
Det:
  model_path: "path/to/det.onnx"      # ONNX model file

Cls:
  model_path: "path/to/cls.xml"       # OpenVINO XML (+ .bin)

Rec:
  model_path: "path/to/rec.onnx"      # ONNX model file
  rec_keys_path: "path/to/keys.txt"   # Character dictionary
```

### Quick Test

```python
from rapidocr_hkmc import RapidOCR

# Should work without internet
ocr = RapidOCR(config_path='config_offline.yaml')
result, elapse = ocr('test.jpg')
print("✅ Offline mode working!" if result else "❌ Check config")
```

---

**Offline mode is fully supported and production-ready!** 🎉
