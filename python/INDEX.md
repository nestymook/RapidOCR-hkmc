# RapidOCR HKMC Documentation Index

## 🚀 Getting Started

Start here if you're new to RapidOCR HKMC:

1. **[QUICK_START.md](QUICK_START.md)** ⭐ **START HERE**
   - Installation instructions
   - Basic usage examples
   - Quick configuration guide
   - Common issues and solutions

2. **[CONFIGURATION_SUMMARY.md](CONFIGURATION_SUMMARY.md)** 📋 **QUICK REFERENCE**
   - Quick answer to "How do I use configurations?"
   - Before/after packaging workflows
   - Recommended patterns
   - Common questions

## 📖 Detailed Documentation

### Configuration Guides

3. **[CONFIGURATION_USAGE_GUIDE.md](CONFIGURATION_USAGE_GUIDE.md)** 📚 **COMPREHENSIVE GUIDE**
   - Complete usage scenarios
   - Development vs production workflows
   - Configuration file locations
   - Troubleshooting and best practices
   - Multiple usage patterns

4. **[CONFIGURATION_WORKFLOW.md](CONFIGURATION_WORKFLOW.md)** 🔄 **VISUAL GUIDE**
   - Decision trees and flowcharts
   - Configuration selection guide
   - File location diagrams
   - Common patterns illustrated

5. **[OFFLINE_MODE_GUIDE.md](OFFLINE_MODE_GUIDE.md)** 🔌 **OFFLINE MODE**
   - Complete offline setup guide
   - Model path configuration
   - Air-gapped deployment
   - Model file requirements
   - Troubleshooting offline issues

5. **[MODEL_FORMATS_GUIDE.md](MODEL_FORMATS_GUIDE.md)** 🔄 **MODEL FORMATS**
   - ONNX vs OpenVINO IR formats
   - When to convert models
   - Conversion instructions
   - Performance comparison
   - Format requirements by engine

### Code Examples

6. **[example_usage.py](example_usage.py)** 💻 **RUNNABLE EXAMPLES**
   - 6 different configuration examples
   - Hardware detection code
   - Image processing demo
   - Auto-configuration selection
   - Run with: `python example_usage.py`

7. **[verify_offline_setup.py](verify_offline_setup.py)** 🔍 **OFFLINE VERIFICATION**
   - Verify model files exist
   - Check configuration validity
   - Validate offline setup
   - Run with: `python verify_offline_setup.py config_offline_example.yaml`

8. **[convert_to_openvino.py](convert_to_openvino.py)** 🔄 **MODEL CONVERTER**
   - Convert ONNX to OpenVINO IR
   - Optimize models for NPU/GPU
   - Batch conversion support
   - Run with: `python convert_to_openvino.py model.onnx`

### Main Documentation

9. **[README.md](README.md)** 📄 **MAIN DOCUMENTATION**
   - Feature overview
   - Installation instructions
   - Configuration examples
   - Hardware requirements
   - API documentation
   - Performance benchmarks
   - Troubleshooting

### Validation and Testing

10. **[ERROR_MESSAGES_VALIDATION.md](ERROR_MESSAGES_VALIDATION.md)** 🔍 **TROUBLESHOOTING**
   - Error message validation
   - Logging verification
   - Diagnostic information
   - Common error scenarios
   - Resolution steps

## 📁 Configuration Files

### Example Configurations

All configuration files are ready to use:

- **[config_cpu_only.yaml](config_cpu_only.yaml)** - CPU-only (maximum compatibility)
- **[config_npu_only.yaml](config_npu_only.yaml)** - NPU-only (power efficient)
- **[config_gpu_only.yaml](config_gpu_only.yaml)** - GPU-only (maximum throughput)
- **[config_npu_gpu_hybrid.yaml](config_npu_gpu_hybrid.yaml)** - ⭐ **NPU + GPU (recommended)**
- **[config_offline_example.yaml](config_offline_example.yaml)** - 🔌 **Offline mode template**

### Default Configuration

- **[rapidocr_hkmc/config.yaml](rapidocr_hkmc/config.yaml)** - Default configuration (included in package)

## 🎯 Quick Navigation

### By Use Case

#### "I want to get started quickly"
→ [QUICK_START.md](QUICK_START.md)

#### "I need to understand how to use configurations"
→ [CONFIGURATION_SUMMARY.md](CONFIGURATION_SUMMARY.md)

#### "I want detailed configuration information"
→ [CONFIGURATION_USAGE_GUIDE.md](CONFIGURATION_USAGE_GUIDE.md)

#### "I prefer visual guides and diagrams"
→ [CONFIGURATION_WORKFLOW.md](CONFIGURATION_WORKFLOW.md)

#### "I want to see working code examples"
→ [example_usage.py](example_usage.py)

#### "I'm having issues and need troubleshooting"
→ [ERROR_MESSAGES_VALIDATION.md](ERROR_MESSAGES_VALIDATION.md)

#### "I need complete API documentation"
→ [README.md](README.md)

#### "I need offline mode (no internet)"
→ [OFFLINE_MODE_GUIDE.md](OFFLINE_MODE_GUIDE.md)

#### "I need to understand model formats (ONNX vs OpenVINO)"
→ [MODEL_FORMATS_GUIDE.md](MODEL_FORMATS_GUIDE.md)

### By Stage

#### Development (Before Packaging)
1. Read [QUICK_START.md](QUICK_START.md) - "Before Packaging" section
2. Choose a config from the example files
3. Use: `RapidOCR(config_path='config_npu_gpu_hybrid.yaml')`
4. Test with [example_usage.py](example_usage.py)

#### Production (After Packaging)
1. Read [CONFIGURATION_SUMMARY.md](CONFIGURATION_SUMMARY.md) - "After Packaging" section
2. Copy example config to your project
3. Customize for your environment
4. Use: `RapidOCR(config_path='ocr_config.yaml')`

### By Hardware

#### Intel NPU + NVIDIA GPU (Recommended)
- Config: [config_npu_gpu_hybrid.yaml](config_npu_gpu_hybrid.yaml)
- Guide: [README.md](README.md) - "Mixed Configuration" section

#### Intel NPU Only
- Config: [config_npu_only.yaml](config_npu_only.yaml)
- Guide: [README.md](README.md) - "NPU Configuration" section

#### NVIDIA GPU Only
- Config: [config_gpu_only.yaml](config_gpu_only.yaml)
- Guide: [README.md](README.md) - "GPU Configuration" section

#### CPU Only
- Config: [config_cpu_only.yaml](config_cpu_only.yaml)
- Guide: [QUICK_START.md](QUICK_START.md) - "CPU Only" section

## 🔧 Common Tasks

### Install and Test
```bash
# Install
pip install rapidocr_hkmc

# Test
python example_usage.py 6
```
See: [QUICK_START.md](QUICK_START.md)

### Use NPU + GPU
```python
from rapidocr_hkmc import RapidOCR
ocr = RapidOCR(config_path='config_npu_gpu_hybrid.yaml')
result, elapse = ocr('image.jpg')
```
See: [CONFIGURATION_SUMMARY.md](CONFIGURATION_SUMMARY.md)

### Check Hardware
```python
from openvino.runtime import Core
print(Core().available_devices)

import onnxruntime
print(onnxruntime.get_available_providers())
```
See: [QUICK_START.md](QUICK_START.md) - "Check Available Hardware"

### Troubleshoot Issues
1. Enable logging: `logging.basicConfig(level=logging.INFO)`
2. Check hardware availability
3. Review error messages
4. See: [ERROR_MESSAGES_VALIDATION.md](ERROR_MESSAGES_VALIDATION.md)

## 📊 Document Overview

| Document | Length | Purpose | Audience |
|----------|--------|---------|----------|
| QUICK_START.md | Short | Quick reference | Everyone |
| CONFIGURATION_SUMMARY.md | Short | Quick answers | Everyone |
| CONFIGURATION_USAGE_GUIDE.md | Long | Detailed guide | Developers |
| CONFIGURATION_WORKFLOW.md | Medium | Visual guide | Visual learners |
| example_usage.py | Code | Working examples | Developers |
| README.md | Long | Complete docs | Everyone |
| ERROR_MESSAGES_VALIDATION.md | Medium | Troubleshooting | Support/Debug |

## 🎓 Learning Path

### Beginner
1. [QUICK_START.md](QUICK_START.md) - Learn basics
2. [example_usage.py](example_usage.py) - Run examples
3. [CONFIGURATION_SUMMARY.md](CONFIGURATION_SUMMARY.md) - Understand configs

### Intermediate
1. [CONFIGURATION_USAGE_GUIDE.md](CONFIGURATION_USAGE_GUIDE.md) - Deep dive
2. [CONFIGURATION_WORKFLOW.md](CONFIGURATION_WORKFLOW.md) - Understand workflows
3. [README.md](README.md) - Complete reference

### Advanced
1. [ERROR_MESSAGES_VALIDATION.md](ERROR_MESSAGES_VALIDATION.md) - Troubleshooting
2. Custom configuration development
3. Performance optimization

## 🔗 External Resources

- [RapidOCR Original Documentation](https://rapidai.github.io/RapidOCRDocs/main/install_usage/rapidocr/install/)
- [OpenVINO Documentation](https://docs.openvino.ai/)
- [ONNXRuntime Documentation](https://onnxruntime.ai/)
- [CUDA Documentation](https://docs.nvidia.com/cuda/)

## 📝 Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│                    QUICK REFERENCE                           │
├─────────────────────────────────────────────────────────────┤
│ DEVELOPMENT:                                                 │
│   from rapidocr_hkmc import RapidOCR                        │
│   ocr = RapidOCR(config_path='config_npu_gpu_hybrid.yaml') │
│                                                              │
│ PRODUCTION:                                                  │
│   1. Copy config to your project                            │
│   2. ocr = RapidOCR(config_path='ocr_config.yaml')         │
│                                                              │
│ CHECK HARDWARE:                                              │
│   from openvino.runtime import Core                         │
│   print(Core().available_devices)                           │
│                                                              │
│ TROUBLESHOOT:                                                │
│   import logging                                             │
│   logging.basicConfig(level=logging.INFO)                   │
└─────────────────────────────────────────────────────────────┘
```

## 💡 Tips

- **Always start with CPU-only** to verify basic functionality
- **Enable logging** to see what hardware is being used
- **Test on target hardware** before production deployment
- **Keep configs separate** from the package
- **Use version control** for your custom configs

## 🆘 Getting Help

1. Check [QUICK_START.md](QUICK_START.md) for common issues
2. Review [ERROR_MESSAGES_VALIDATION.md](ERROR_MESSAGES_VALIDATION.md) for error messages
3. Enable debug logging: `logging.basicConfig(level=logging.DEBUG)`
4. Check hardware availability with the verification scripts
5. Review the example code in [example_usage.py](example_usage.py)

---

**Last Updated**: November 2, 2025

**Version**: 1.0.0

**Status**: Production Ready ✅
