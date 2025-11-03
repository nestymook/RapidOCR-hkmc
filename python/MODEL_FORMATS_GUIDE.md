# Model Formats Guide - RapidOCR HKMC

## Quick Answer

**Q: Do I need to convert ONNX models to OpenVINO format (.xml/.bin)?**

**A: It depends on which engine you're using:**

| Engine | Supported Formats | Conversion Needed? |
|--------|-------------------|-------------------|
| **ONNXRuntime** | `.onnx` | ❌ No - Use ONNX directly |
| **OpenVINO** | `.xml` + `.bin` OR `.onnx` | ⚠️ Recommended to pre-convert |

---

## Detailed Explanation

### OpenVINO Model Format Support

OpenVINO's `core.read_model()` can read **multiple formats**:

1. **OpenVINO IR format** (`.xml` + `.bin`) - **Recommended**
   - Pre-optimized for OpenVINO
   - Fastest loading and inference
   - Best performance on NPU/GPU

2. **ONNX format** (`.onnx`) - **Supported but not optimal**
   - Can be loaded directly by OpenVINO
   - Converted on-the-fly during loading
   - Slower first load (conversion overhead)
   - May not be fully optimized for target hardware

3. **PaddlePaddle format** (`.pdmodel`) - **Supported**
   - Can be loaded directly
   - Converted on-the-fly

### Recommendation

**For Production**: Pre-convert ONNX to OpenVINO IR format (`.xml` + `.bin`)

**Why?**
- ✅ Faster model loading
- ✅ Better optimization for NPU/GPU
- ✅ Smaller model size
- ✅ Hardware-specific optimizations applied
- ✅ No conversion overhead at runtime

---

## Model Format by Engine

### 1. ONNXRuntime Engine (Det, Rec)

**Supported Format**: `.onnx` only

**Configuration**:
```yaml
Det:
  engine_type: "onnxruntime"
  model_path: "models/ch_PP-OCRv4_det_infer.onnx"  # ONNX format

Rec:
  engine_type: "onnxruntime"
  model_path: "models/ch_PP-OCRv4_rec_infer.onnx"  # ONNX format
```

**No conversion needed** - ONNXRuntime only works with ONNX format.

---

### 2. OpenVINO Engine (Cls)

**Supported Formats**:
- `.xml` + `.bin` (OpenVINO IR) - **Recommended**
- `.onnx` (ONNX) - Supported but slower

#### Option A: Use Pre-Converted OpenVINO IR (Recommended)

**Configuration**:
```yaml
Cls:
  engine_type: "openvino"
  model_path: "models/ch_ppocr_mobile_v2.0_cls_infer.xml"  # OpenVINO IR
  # Corresponding .bin file must be in same directory
```

**Required Files**:
```
models/
├── ch_ppocr_mobile_v2.0_cls_infer.xml  ← Specify this
└── ch_ppocr_mobile_v2.0_cls_infer.bin  ← Must exist
```

**Advantages**:
- ✅ Fastest loading
- ✅ Best performance
- ✅ Optimized for NPU/GPU
- ✅ No runtime conversion

#### Option B: Use ONNX Directly (Not Recommended)

**Configuration**:
```yaml
Cls:
  engine_type: "openvino"
  model_path: "models/ch_ppocr_mobile_v2.0_cls_infer.onnx"  # ONNX format
```

**What Happens**:
1. OpenVINO loads the ONNX file
2. Converts it to IR format in memory (on-the-fly)
3. Compiles for target device
4. Runs inference

**Disadvantages**:
- ⚠️ Slower first load (conversion overhead)
- ⚠️ May not be fully optimized
- ⚠️ Larger memory footprint during conversion
- ⚠️ Conversion happens every time you initialize

---

## Converting ONNX to OpenVINO IR

If you have ONNX models and want to use OpenVINO engine, you should convert them first.

### Method 1: Using OpenVINO Model Optimizer (Recommended)

**Install OpenVINO**:
```bash
pip install openvino-dev
```

**Convert ONNX to OpenVINO IR**:
```bash
# Convert classification model
mo --input_model ch_ppocr_mobile_v2.0_cls_infer.onnx \
   --output_dir models/ \
   --model_name ch_ppocr_mobile_v2.0_cls_infer

# This creates:
# - models/ch_ppocr_mobile_v2.0_cls_infer.xml
# - models/ch_ppocr_mobile_v2.0_cls_infer.bin
```

**Windows PowerShell**:
```powershell
mo --input_model ch_ppocr_mobile_v2.0_cls_infer.onnx `
   --output_dir models\ `
   --model_name ch_ppocr_mobile_v2.0_cls_infer
```

### Method 2: Using Python API

```python
from openvino.tools import mo
from openvino.runtime import serialize

# Convert ONNX to OpenVINO IR
ov_model = mo.convert_model("ch_ppocr_mobile_v2.0_cls_infer.onnx")

# Save to disk
serialize(ov_model, "models/ch_ppocr_mobile_v2.0_cls_infer.xml")

print("✅ Conversion complete!")
print("Created:")
print("  - models/ch_ppocr_mobile_v2.0_cls_infer.xml")
print("  - models/ch_ppocr_mobile_v2.0_cls_infer.bin")
```

### Method 3: Using ovc Tool (OpenVINO 2023.0+)

```bash
# New conversion tool in OpenVINO 2023.0+
ovc ch_ppocr_mobile_v2.0_cls_infer.onnx \
    --output_model models/ch_ppocr_mobile_v2.0_cls_infer
```

---

## Complete Conversion Script

```python
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Convert ONNX models to OpenVINO IR format for optimal performance.
"""

import sys
from pathlib import Path

def convert_onnx_to_openvino(onnx_path: str, output_dir: str = "models"):
    """Convert ONNX model to OpenVINO IR format."""
    try:
        from openvino.tools import mo
        from openvino.runtime import serialize
    except ImportError:
        print("❌ OpenVINO not installed")
        print("Install with: pip install openvino-dev")
        return False
    
    onnx_path = Path(onnx_path)
    output_dir = Path(output_dir)
    
    if not onnx_path.exists():
        print(f"❌ ONNX model not found: {onnx_path}")
        return False
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output file name (without extension)
    output_name = onnx_path.stem
    output_xml = output_dir / f"{output_name}.xml"
    output_bin = output_dir / f"{output_name}.bin"
    
    print(f"Converting: {onnx_path}")
    print(f"Output: {output_xml}")
    
    try:
        # Convert ONNX to OpenVINO IR
        print("Converting...")
        ov_model = mo.convert_model(str(onnx_path))
        
        # Save to disk
        print("Saving...")
        serialize(ov_model, str(output_xml))
        
        print("✅ Conversion successful!")
        print(f"Created:")
        print(f"  - {output_xml}")
        print(f"  - {output_bin}")
        
        # Verify files exist
        if output_xml.exists() and output_bin.exists():
            xml_size = output_xml.stat().st_size / (1024 * 1024)
            bin_size = output_bin.stat().st_size / (1024 * 1024)
            print(f"\nFile sizes:")
            print(f"  - XML: {xml_size:.2f} MB")
            print(f"  - BIN: {bin_size:.2f} MB")
            return True
        else:
            print("❌ Output files not created")
            return False
            
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python convert_to_openvino.py <onnx_model> [output_dir]")
        print("\nExample:")
        print("  python convert_to_openvino.py cls_model.onnx models/")
        sys.exit(1)
    
    onnx_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "models"
    
    success = convert_onnx_to_openvino(onnx_path, output_dir)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
```

**Usage**:
```bash
# Convert single model
python convert_to_openvino.py ch_ppocr_mobile_v2.0_cls_infer.onnx

# Convert with custom output directory
python convert_to_openvino.py cls_model.onnx converted_models/
```

---

## Recommended Configuration

### For Best Performance (Production)

```yaml
# Use OpenVINO IR format for Cls (pre-converted)
Cls:
  engine_type: "openvino"
  model_path: "models/ch_ppocr_mobile_v2.0_cls_infer.xml"  # OpenVINO IR

# Use ONNX format for Det and Rec (ONNXRuntime)
Det:
  engine_type: "onnxruntime"
  model_path: "models/ch_PP-OCRv4_det_infer.onnx"  # ONNX

Rec:
  engine_type: "onnxruntime"
  model_path: "models/ch_PP-OCRv4_rec_infer.onnx"  # ONNX
```

### For Quick Testing (Development)

```yaml
# Can use ONNX for all (OpenVINO will convert on-the-fly)
Cls:
  engine_type: "openvino"
  model_path: "models/cls_model.onnx"  # ONNX (converted at runtime)

Det:
  engine_type: "onnxruntime"
  model_path: "models/det_model.onnx"  # ONNX

Rec:
  engine_type: "onnxruntime"
  model_path: "models/rec_model.onnx"  # ONNX
```

---

## Performance Comparison

### OpenVINO IR (.xml + .bin) vs ONNX

| Metric | OpenVINO IR | ONNX (on-the-fly) |
|--------|-------------|-------------------|
| **First Load Time** | Fast (~1-2s) | Slow (~5-10s) |
| **Inference Speed** | Optimal | Good |
| **Memory Usage** | Lower | Higher (during conversion) |
| **Optimization** | Hardware-specific | Generic |
| **File Size** | Smaller | Larger |

**Example Timing**:
```
OpenVINO IR:
  Model loading: 1.2s
  First inference: 15ms
  
ONNX (on-the-fly):
  Model loading + conversion: 8.5s
  First inference: 18ms
```

---

## Default Models from RapidOCR

When you use RapidOCR without specifying `model_path`, it downloads pre-converted models:

**Downloaded Models**:
- Det: `ch_PP-OCRv4_det_infer.onnx` (ONNX for ONNXRuntime)
- Cls: `ch_ppocr_mobile_v2.0_cls_infer.xml` + `.bin` (OpenVINO IR)
- Rec: `ch_PP-OCRv4_rec_infer.onnx` (ONNX for ONNXRuntime)

**These are already in the optimal format for each engine!**

---

## Summary

### Key Points

1. **ONNXRuntime** (Det, Rec):
   - ✅ Use `.onnx` format
   - ❌ Cannot use OpenVINO IR format

2. **OpenVINO** (Cls):
   - ✅ **Recommended**: Use `.xml` + `.bin` (OpenVINO IR)
   - ⚠️ **Supported**: Use `.onnx` (converted on-the-fly)

3. **Conversion**:
   - ❌ **Not required** - OpenVINO can load ONNX directly
   - ✅ **Recommended** - Pre-convert for better performance

4. **Default Models**:
   - Already in optimal format
   - No conversion needed

### Decision Tree

```
Do you have ONNX models?
│
├─ Using ONNXRuntime engine?
│  └─ ✅ Use ONNX directly (no conversion)
│
└─ Using OpenVINO engine?
   │
   ├─ Development/Testing?
   │  └─ ⚠️ Can use ONNX (slower first load)
   │
   └─ Production?
      └─ ✅ Convert to OpenVINO IR (best performance)
```

### Quick Commands

```bash
# Check if you need to convert
# If using OpenVINO engine with .onnx file → Convert recommended

# Convert ONNX to OpenVINO IR
pip install openvino-dev
ovc your_model.onnx --output_model models/your_model

# Or use the conversion script
python convert_to_openvino.py your_model.onnx
```

---

## Conclusion

**You don't NEED to convert**, but you SHOULD convert ONNX to OpenVINO IR format for production use with OpenVINO engine for:
- ✅ Faster loading
- ✅ Better performance
- ✅ Optimal NPU/GPU utilization

For development and testing, using ONNX directly with OpenVINO is fine (it will convert on-the-fly).
