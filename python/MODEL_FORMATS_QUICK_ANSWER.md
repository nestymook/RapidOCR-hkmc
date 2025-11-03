# Model Formats - Quick Answer

## Your Question

**"Is OpenVINO required to generate models to .xml and .bin, or will the application convert ONNX models on the fly?"**

## Quick Answer

**OpenVINO can load ONNX directly (on-the-fly conversion), but pre-converting is recommended for production.**

### Summary Table

| Scenario | ONNX Conversion Required? | Recommendation |
|----------|---------------------------|----------------|
| **ONNXRuntime engine** (Det, Rec) | ❌ No - Use ONNX as-is | Use `.onnx` files |
| **OpenVINO engine** (Cls) - Development | ❌ No - Converts on-the-fly | Can use `.onnx` (slower) |
| **OpenVINO engine** (Cls) - Production | ✅ Yes - Pre-convert recommended | Use `.xml` + `.bin` |

---

## Detailed Answer

### What Happens with Each Format

#### 1. Using ONNX with OpenVINO (On-the-Fly)

```yaml
Cls:
  engine_type: "openvino"
  model_path: "models/cls_model.onnx"  # ONNX file
```

**What happens**:
1. OpenVINO loads the ONNX file
2. **Converts it to IR format in memory** (on-the-fly)
3. Compiles for target device (NPU/GPU/CPU)
4. Runs inference

**Pros**:
- ✅ No pre-conversion needed
- ✅ Works immediately
- ✅ Good for development/testing

**Cons**:
- ⚠️ Slower first load (5-10 seconds for conversion)
- ⚠️ Conversion happens every time you initialize
- ⚠️ Higher memory usage during conversion
- ⚠️ May not be fully optimized

#### 2. Using OpenVINO IR (Pre-Converted)

```yaml
Cls:
  engine_type: "openvino"
  model_path: "models/cls_model.xml"  # OpenVINO IR
  # Corresponding .bin file must exist
```

**What happens**:
1. OpenVINO loads the pre-converted IR files
2. Compiles for target device (NPU/GPU/CPU)
3. Runs inference

**Pros**:
- ✅ Fast loading (1-2 seconds)
- ✅ Optimized for target hardware
- ✅ Lower memory usage
- ✅ Best performance

**Cons**:
- ⚠️ Requires pre-conversion step

---

## When to Convert

### ❌ Don't Convert If:

1. **Using ONNXRuntime engine** (Det, Rec models)
   - ONNXRuntime only works with ONNX
   - No conversion possible

2. **Quick testing/development**
   - On-the-fly conversion is fine
   - Saves time during development

### ✅ Do Convert If:

1. **Using OpenVINO engine in production**
   - Faster initialization
   - Better performance
   - Lower resource usage

2. **Deploying to NPU/GPU**
   - Hardware-specific optimizations
   - Maximum performance

3. **Air-gapped/offline environments**
   - Pre-convert before deployment
   - No runtime conversion overhead

---

## How to Convert

### Quick Conversion

```bash
# Install OpenVINO tools
pip install openvino-dev

# Convert ONNX to OpenVINO IR
python convert_to_openvino.py your_model.onnx

# Creates:
# - your_model.xml
# - your_model.bin
```

### Using the Conversion Script

```bash
# Single model
python convert_to_openvino.py cls_model.onnx

# Multiple models
python convert_to_openvino.py model1.onnx model2.onnx model3.onnx

# Custom output directory
python convert_to_openvino.py cls_model.onnx converted_models/
```

### Manual Conversion

```bash
# Using OpenVINO Model Optimizer
mo --input_model model.onnx --output_dir models/

# Or using ovc tool (OpenVINO 2023.0+)
ovc model.onnx --output_model models/model
```

---

## Performance Comparison

### Loading Time

| Format | First Load | Subsequent Loads |
|--------|-----------|------------------|
| **ONNX (on-the-fly)** | 8-10 seconds | 8-10 seconds |
| **OpenVINO IR** | 1-2 seconds | 1-2 seconds |

### Inference Speed

| Format | Speed | Optimization |
|--------|-------|--------------|
| **ONNX (on-the-fly)** | Good | Generic |
| **OpenVINO IR** | Optimal | Hardware-specific |

---

## Recommended Configuration

### Development (Quick Testing)

```yaml
# Can use ONNX for all models
Cls:
  engine_type: "openvino"
  model_path: "models/cls.onnx"  # On-the-fly conversion

Det:
  engine_type: "onnxruntime"
  model_path: "models/det.onnx"

Rec:
  engine_type: "onnxruntime"
  model_path: "models/rec.onnx"
```

### Production (Best Performance)

```yaml
# Use OpenVINO IR for Cls (pre-converted)
Cls:
  engine_type: "openvino"
  model_path: "models/cls.xml"  # Pre-converted

Det:
  engine_type: "onnxruntime"
  model_path: "models/det.onnx"  # ONNX for ONNXRuntime

Rec:
  engine_type: "onnxruntime"
  model_path: "models/rec.onnx"  # ONNX for ONNXRuntime
```

---

## Default Models

When you use RapidOCR without specifying `model_path`, it downloads **pre-optimized models**:

- **Det**: `ch_PP-OCRv4_det_infer.onnx` (ONNX)
- **Cls**: `ch_ppocr_mobile_v2.0_cls_infer.xml` + `.bin` (OpenVINO IR)
- **Rec**: `ch_PP-OCRv4_rec_infer.onnx` (ONNX)

**These are already in the optimal format!** No conversion needed.

---

## Summary

### Key Takeaways

1. **Conversion is NOT required** - OpenVINO can load ONNX directly
2. **Conversion is RECOMMENDED** - For production use with OpenVINO
3. **ONNXRuntime uses ONNX** - No conversion possible or needed
4. **Default models are optimized** - Already in best format

### Decision Flow

```
Using OpenVINO engine?
│
├─ Development/Testing?
│  └─ Use ONNX (on-the-fly conversion is fine)
│
└─ Production?
   └─ Convert to OpenVINO IR (best performance)
```

### Quick Commands

```bash
# Check if conversion is recommended
# If using OpenVINO with .onnx → Convert for production

# Convert ONNX to OpenVINO IR
python convert_to_openvino.py your_model.onnx

# Use converted model
# Update config: model_path: "models/your_model.xml"
```

---

## More Information

- **[MODEL_FORMATS_GUIDE.md](MODEL_FORMATS_GUIDE.md)** - Complete guide
- **[convert_to_openvino.py](convert_to_openvino.py)** - Conversion script
- **[OFFLINE_MODE_GUIDE.md](OFFLINE_MODE_GUIDE.md)** - Offline deployment

---

**Bottom Line**: OpenVINO converts ONNX on-the-fly, but pre-converting gives you better performance. For production, always pre-convert! 🚀
