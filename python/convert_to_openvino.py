#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Convert ONNX models to OpenVINO IR format for optimal performance.

This script converts ONNX models to OpenVINO's Intermediate Representation (IR)
format (.xml + .bin), which provides better performance when using OpenVINO engine.

Usage:
    python convert_to_openvino.py <onnx_model> [output_dir]

Examples:
    python convert_to_openvino.py cls_model.onnx
    python convert_to_openvino.py cls_model.onnx converted_models/
"""

import sys
from pathlib import Path


def check_openvino_installation():
    """Check if OpenVINO is installed."""
    try:
        import openvino
        print(f"✅ OpenVINO version: {openvino.__version__}")
        return True
    except ImportError:
        print("❌ OpenVINO not installed")
        print("\nTo install OpenVINO:")
        print("  pip install openvino-dev")
        print("\nOr for runtime only:")
        print("  pip install openvino")
        return False


def convert_onnx_to_openvino(onnx_path: str, output_dir: str = "models"):
    """
    Convert ONNX model to OpenVINO IR format.
    
    Args:
        onnx_path: Path to input ONNX model file
        output_dir: Directory to save converted model
        
    Returns:
        True if conversion successful, False otherwise
    """
    try:
        from openvino.tools import mo
        from openvino.runtime import serialize
    except ImportError:
        print("❌ OpenVINO conversion tools not available")
        print("\nInstall with: pip install openvino-dev")
        return False
    
    onnx_path = Path(onnx_path)
    output_dir = Path(output_dir)
    
    # Validate input
    if not onnx_path.exists():
        print(f"❌ ONNX model not found: {onnx_path}")
        return False
    
    if not onnx_path.suffix == '.onnx':
        print(f"❌ Input file is not an ONNX model: {onnx_path}")
        print("   Expected .onnx extension")
        return False
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output file paths
    output_name = onnx_path.stem
    output_xml = output_dir / f"{output_name}.xml"
    output_bin = output_dir / f"{output_name}.bin"
    
    print("="*70)
    print("ONNX to OpenVINO IR Conversion")
    print("="*70)
    print(f"\nInput:  {onnx_path}")
    print(f"Output: {output_xml}")
    print()
    
    try:
        # Get input file size
        input_size_mb = onnx_path.stat().st_size / (1024 * 1024)
        print(f"Input size: {input_size_mb:.2f} MB")
        print()
        
        # Convert ONNX to OpenVINO IR
        print("Converting ONNX to OpenVINO IR...")
        print("This may take a few moments...")
        
        ov_model = mo.convert_model(str(onnx_path))
        
        print("✅ Conversion successful")
        print()
        
        # Save to disk
        print("Saving model files...")
        serialize(ov_model, str(output_xml))
        
        print("✅ Model saved")
        print()
        
        # Verify files exist and show sizes
        if output_xml.exists() and output_bin.exists():
            xml_size_mb = output_xml.stat().st_size / (1024 * 1024)
            bin_size_mb = output_bin.stat().st_size / (1024 * 1024)
            total_size_mb = xml_size_mb + bin_size_mb
            
            print("="*70)
            print("Conversion Complete!")
            print("="*70)
            print("\nCreated files:")
            print(f"  ✅ {output_xml}")
            print(f"     Size: {xml_size_mb:.2f} MB")
            print(f"  ✅ {output_bin}")
            print(f"     Size: {bin_size_mb:.2f} MB")
            print(f"\nTotal size: {total_size_mb:.2f} MB")
            print(f"Size reduction: {((input_size_mb - total_size_mb) / input_size_mb * 100):.1f}%")
            print()
            
            # Usage instructions
            print("="*70)
            print("Usage in Configuration")
            print("="*70)
            print("\nAdd to your config.yaml:")
            print(f"""
Cls:
  engine_type: "openvino"
  model_path: "{output_xml}"
  # The .bin file will be loaded automatically
""")
            
            return True
        else:
            print("❌ Output files not created")
            return False
            
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        print("\nCommon issues:")
        print("  1. ONNX model is corrupted or invalid")
        print("  2. Unsupported ONNX operations")
        print("  3. Insufficient memory")
        print("\nFor detailed error information:")
        import traceback
        traceback.print_exc()
        return False


def convert_batch(onnx_files: list, output_dir: str = "models"):
    """
    Convert multiple ONNX models to OpenVINO IR format.
    
    Args:
        onnx_files: List of ONNX model file paths
        output_dir: Directory to save converted models
    """
    print("="*70)
    print(f"Batch Conversion: {len(onnx_files)} models")
    print("="*70)
    print()
    
    results = []
    for i, onnx_file in enumerate(onnx_files, 1):
        print(f"\n[{i}/{len(onnx_files)}] Converting: {onnx_file}")
        print("-"*70)
        success = convert_onnx_to_openvino(onnx_file, output_dir)
        results.append((onnx_file, success))
        print()
    
    # Summary
    print("="*70)
    print("Batch Conversion Summary")
    print("="*70)
    print()
    
    successful = sum(1 for _, success in results if success)
    failed = len(results) - successful
    
    print(f"Total: {len(results)}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print()
    
    if failed > 0:
        print("Failed conversions:")
        for onnx_file, success in results:
            if not success:
                print(f"  ❌ {onnx_file}")


def print_usage():
    """Print usage information."""
    print("="*70)
    print("ONNX to OpenVINO IR Converter")
    print("="*70)
    print()
    print("Usage:")
    print("  python convert_to_openvino.py <onnx_model> [output_dir]")
    print()
    print("Arguments:")
    print("  onnx_model  : Path to ONNX model file (.onnx)")
    print("  output_dir  : Output directory (default: 'models')")
    print()
    print("Examples:")
    print("  # Convert single model")
    print("  python convert_to_openvino.py cls_model.onnx")
    print()
    print("  # Convert with custom output directory")
    print("  python convert_to_openvino.py cls_model.onnx converted/")
    print()
    print("  # Convert multiple models")
    print("  python convert_to_openvino.py model1.onnx model2.onnx model3.onnx")
    print()
    print("Why convert?")
    print("  ✅ Faster model loading")
    print("  ✅ Better performance on NPU/GPU")
    print("  ✅ Smaller model size")
    print("  ✅ Hardware-specific optimizations")
    print()
    print("Note:")
    print("  - Conversion is recommended but not required")
    print("  - OpenVINO can load ONNX directly (slower)")
    print("  - For production, always use converted models")
    print()


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    # Check OpenVINO installation
    if not check_openvino_installation():
        sys.exit(1)
    
    print()
    
    # Parse arguments
    onnx_files = []
    output_dir = "models"
    
    for arg in sys.argv[1:]:
        if arg.endswith('.onnx'):
            onnx_files.append(arg)
        else:
            output_dir = arg
    
    if not onnx_files:
        print("❌ No ONNX model files specified")
        print()
        print_usage()
        sys.exit(1)
    
    # Convert models
    if len(onnx_files) == 1:
        # Single model conversion
        success = convert_onnx_to_openvino(onnx_files[0], output_dir)
        sys.exit(0 if success else 1)
    else:
        # Batch conversion
        convert_batch(onnx_files, output_dir)
        sys.exit(0)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Conversion interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
