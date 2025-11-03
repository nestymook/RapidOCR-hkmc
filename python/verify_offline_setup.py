#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Verification script for offline mode setup.

This script helps you verify that all model files are in place
for offline operation of RapidOCR HKMC.
"""

import sys
from pathlib import Path
from typing import List, Tuple


def check_file_exists(file_path: str, description: str) -> bool:
    """Check if a file exists and print status."""
    path = Path(file_path)
    exists = path.exists()
    
    status = "✅" if exists else "❌"
    print(f"{status} {description}")
    
    if exists:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"   Path: {file_path}")
        print(f"   Size: {size_mb:.2f} MB")
    else:
        print(f"   Path: {file_path}")
        print(f"   Status: NOT FOUND")
    
    print()
    return exists


def verify_offline_config(config_path: str) -> Tuple[bool, List[str]]:
    """Verify all model files specified in config exist."""
    try:
        from omegaconf import OmegaConf
    except ImportError:
        print("❌ omegaconf not installed. Install with: pip install omegaconf")
        return False, []
    
    print(f"Loading configuration: {config_path}\n")
    
    if not Path(config_path).exists():
        print(f"❌ Configuration file not found: {config_path}\n")
        return False, []
    
    try:
        config = OmegaConf.load(config_path)
    except Exception as e:
        print(f"❌ Error loading configuration: {e}\n")
        return False, []
    
    print("="*70)
    print("OFFLINE MODE VERIFICATION")
    print("="*70)
    print()
    
    all_exist = True
    missing_files = []
    
    # Check Detection model
    print("1. Detection Model (Det)")
    print("-" * 70)
    det_path = config.get('Det', {}).get('model_path')
    if det_path and det_path != 'null':
        if not check_file_exists(det_path, "Detection model (.onnx)"):
            all_exist = False
            missing_files.append(det_path)
    else:
        print("⚠️  No model_path specified (will download online)")
        print()
    
    # Check Classification model
    print("2. Classification Model (Cls)")
    print("-" * 70)
    cls_path = config.get('Cls', {}).get('model_path')
    if cls_path and cls_path != 'null':
        # Check .xml file
        if not check_file_exists(cls_path, "Classification model structure (.xml)"):
            all_exist = False
            missing_files.append(cls_path)
        
        # Check corresponding .bin file
        bin_path = Path(cls_path).with_suffix('.bin')
        if not check_file_exists(str(bin_path), "Classification model weights (.bin)"):
            all_exist = False
            missing_files.append(str(bin_path))
    else:
        print("⚠️  No model_path specified (will download online)")
        print()
    
    # Check Recognition model
    print("3. Recognition Model (Rec)")
    print("-" * 70)
    rec_path = config.get('Rec', {}).get('model_path')
    if rec_path and rec_path != 'null':
        if not check_file_exists(rec_path, "Recognition model (.onnx)"):
            all_exist = False
            missing_files.append(rec_path)
    else:
        print("⚠️  No model_path specified (will download online)")
        print()
    
    # Check character dictionary
    print("4. Character Dictionary (Rec)")
    print("-" * 70)
    keys_path = config.get('Rec', {}).get('rec_keys_path')
    if keys_path and keys_path != 'null':
        if not check_file_exists(keys_path, "Character dictionary (.txt)"):
            all_exist = False
            missing_files.append(keys_path)
    else:
        print("⚠️  No rec_keys_path specified (may cause issues)")
        print()
    
    return all_exist, missing_files


def print_summary(all_exist: bool, missing_files: List[str]):
    """Print verification summary."""
    print("="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    print()
    
    if all_exist:
        print("✅ SUCCESS: All model files are present!")
        print()
        print("Your offline configuration is ready to use.")
        print()
        print("Next steps:")
        print("  1. Test offline mode (disconnect internet)")
        print("  2. Run: python -c \"from rapidocr_hkmc import RapidOCR; ocr = RapidOCR(config_path='your_config.yaml')\"")
        print()
    else:
        print("❌ FAILED: Some model files are missing")
        print()
        print("Missing files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print()
        print("To fix:")
        print("  1. Download models (see OFFLINE_MODE_GUIDE.md)")
        print("  2. Update config file with correct paths")
        print("  3. Run this script again to verify")
        print()


def check_default_cache():
    """Check if models exist in default cache location."""
    import os
    
    print("="*70)
    print("CHECKING DEFAULT MODEL CACHE")
    print("="*70)
    print()
    
    if os.name == 'nt':  # Windows
        cache_dir = Path.home() / '.rapidocr' / 'models'
    else:  # Linux/Mac
        cache_dir = Path.home() / '.rapidocr' / 'models'
    
    print(f"Cache location: {cache_dir}")
    print()
    
    if not cache_dir.exists():
        print("❌ Cache directory does not exist")
        print()
        print("To create cache:")
        print("  1. Run RapidOCR once with internet connection")
        print("  2. Models will be downloaded to cache automatically")
        print()
        return
    
    model_files = list(cache_dir.glob('*'))
    
    if not model_files:
        print("❌ No models found in cache")
        print()
        return
    
    print(f"✅ Found {len(model_files)} files in cache:")
    print()
    
    for model_file in sorted(model_files):
        size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"  • {model_file.name} ({size_mb:.2f} MB)")
    
    print()
    print("You can copy these files to your offline deployment location.")
    print()


def print_usage():
    """Print usage information."""
    print("="*70)
    print("OFFLINE MODE VERIFICATION TOOL")
    print("="*70)
    print()
    print("Usage:")
    print(f"  python {sys.argv[0]} <config_file>")
    print()
    print("Examples:")
    print(f"  python {sys.argv[0]} config_offline_example.yaml")
    print(f"  python {sys.argv[0]} config_npu_gpu_hybrid.yaml")
    print()
    print("Options:")
    print(f"  python {sys.argv[0]} --check-cache    # Check default cache location")
    print()


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    if sys.argv[1] == '--check-cache':
        check_default_cache()
        sys.exit(0)
    
    config_path = sys.argv[1]
    
    all_exist, missing_files = verify_offline_config(config_path)
    print_summary(all_exist, missing_files)
    
    # Also check default cache
    check_default_cache()
    
    sys.exit(0 if all_exist else 1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
