# Offline Mode Quick Start

## Yes, Offline Mode is Fully Supported! ✅

RapidOCR HKMC supports **complete offline operation** by allowing you to specify local model paths in your configuration file.

## Quick Answer

**Q: Can I use local model files instead of downloading?**  
**A: Yes! Just specify `model_path` in your config file.**

```yaml
Det:
  model_path: "C:/models/det.onnx"  # Your local file

Cls:
  model_path: "C:/models/cls.xml"   # Your local file

Rec:
  model_path: "C:/models/rec.onnx"  # Your local file
  rec_keys_path: "C:/models/keys.txt"
```

---

## 3-Step Setup

### Step 1: Get the Models

**Option A: Download once while online**
```python
from rapidocr_hkmc import RapidOCR

# Run once with internet - models are cached
ocr = RapidOCR()
```

Models are saved to:
- Windows: `C:\Users\<username>\.rapidocr\models\`
- Linux/Mac: `~/.rapidocr/models/`

**Option B: Copy from another system**

Copy the model files from a system that has them.

### Step 2: Create Offline Config

Copy and edit `config_offline_example.yaml`:

```yaml
Det:
  engine_type: "onnxruntime"
  model_path: "C:/my_models/ch_PP-OCRv4_det_infer.onnx"

Cls:
  engine_type: "openvino"
  model_path: "C:/my_models/ch_ppocr_mobile_v2.0_cls_infer.xml"

Rec:
  engine_type: "onnxruntime"
  model_path: "C:/my_models/ch_PP-OCRv4_rec_infer.onnx"
  rec_keys_path: "C:/my_models/ppocr_keys_v1.txt"
```

### Step 3: Use Offline

```python
from rapidocr_hkmc import RapidOCR

# Works without internet!
ocr = RapidOCR(config_path='config_offline.yaml')
result, elapse = ocr('image.jpg')
```

---

## Verify Your Setup

Run the verification script:

```bash
python verify_offline_setup.py config_offline.yaml
```

This checks that all model files exist and are accessible.

---

## Required Files

### For Detection (Det)
- `ch_PP-OCRv4_det_infer.onnx` (ONNX model)

### For Classification (Cls)
- `ch_ppocr_mobile_v2.0_cls_infer.xml` (OpenVINO structure)
- `ch_ppocr_mobile_v2.0_cls_infer.bin` (OpenVINO weights) ⚠️ **Must be in same directory**

### For Recognition (Rec)
- `ch_PP-OCRv4_rec_infer.onnx` (ONNX model)
- `ppocr_keys_v1.txt` (Character dictionary) ⚠️ **Required**

---

## Example Directory Structure

```
my_project/
├── config_offline.yaml
├── models/
│   ├── ch_PP-OCRv4_det_infer.onnx
│   ├── ch_ppocr_mobile_v2.0_cls_infer.xml
│   ├── ch_ppocr_mobile_v2.0_cls_infer.bin
│   ├── ch_PP-OCRv4_rec_infer.onnx
│   └── ppocr_keys_v1.txt
└── main.py
```

---

## Common Issues

### Issue: "Model file not found"
**Solution**: Use absolute paths or verify file exists
```python
from pathlib import Path
print(Path("models/det.onnx").exists())
```

### Issue: OpenVINO fails to load
**Solution**: Ensure both .xml and .bin files exist
```python
from pathlib import Path
xml_path = Path("models/cls.xml")
bin_path = xml_path.with_suffix('.bin')
print(f"XML: {xml_path.exists()}, BIN: {bin_path.exists()}")
```

### Issue: Still downloading models
**Solution**: Ensure `model_path` is NOT null
```yaml
# ❌ Wrong - will download
Det:
  model_path: null

# ✅ Correct - uses local file
Det:
  model_path: "models/det.onnx"
```

---

## Complete Example

```python
from rapidocr_hkmc import RapidOCR
import logging

# Enable logging to verify offline mode
logging.basicConfig(level=logging.INFO)

# Use offline configuration
ocr = RapidOCR(config_path='config_offline.yaml')

# Check logs - should see:
# "Using C:/my_models/ch_PP-OCRv4_det_infer.onnx"
# No download messages!

# Process image
result, elapse = ocr('document.jpg')

for line in result:
    print(f"Text: {line[1]}, Confidence: {line[2]}")
```

---

## Documentation

For complete offline mode documentation, see:
- **[OFFLINE_MODE_GUIDE.md](OFFLINE_MODE_GUIDE.md)** - Complete guide
- **[config_offline_example.yaml](config_offline_example.yaml)** - Template config
- **[verify_offline_setup.py](verify_offline_setup.py)** - Verification tool

---

## Summary

✅ **Offline mode is fully supported**  
✅ **Specify `model_path` in config file**  
✅ **Works in air-gapped environments**  
✅ **No internet required after setup**  
✅ **Full control over model versions**

**That's it!** Your RapidOCR HKMC can run completely offline. 🎉
