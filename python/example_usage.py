#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Example usage of RapidOCR HKMC with different configurations.

This script demonstrates how to use the various configuration files
for NPU/GPU acceleration in different scenarios.
"""

import logging
import os
import sys
from pathlib import Path

# Setup logging to see device selection and performance info
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def check_hardware_availability():
    """Check what hardware acceleration is available on this system."""
    logger.info("Checking hardware availability...")
    
    # Check NPU
    try:
        from openvino.runtime import Core
        available_devices = Core().available_devices
        has_npu = 'NPU' in available_devices
        logger.info(f"OpenVINO devices: {available_devices}")
        logger.info(f"NPU available: {has_npu}")
    except Exception as e:
        logger.warning(f"Could not check NPU availability: {e}")
        has_npu = False
    
    # Check CUDA/GPU
    try:
        import onnxruntime
        available_providers = onnxruntime.get_available_providers()
        has_cuda = 'CUDAExecutionProvider' in available_providers
        logger.info(f"ONNXRuntime providers: {available_providers}")
        logger.info(f"CUDA available: {has_cuda}")
    except Exception as e:
        logger.warning(f"Could not check CUDA availability: {e}")
        has_cuda = False
    
    return has_npu, has_cuda


def example_1_default_cpu():
    """Example 1: Use default CPU configuration (no config file needed)."""
    logger.info("\n" + "="*70)
    logger.info("Example 1: Default CPU Configuration")
    logger.info("="*70)
    
    from rapidocr_hkmc import RapidOCR
    
    # No config needed - uses CPU by default
    ocr = RapidOCR()
    
    logger.info("✓ RapidOCR initialized with default CPU configuration")
    logger.info("This works on any system without special hardware")
    
    return ocr


def example_2_npu_gpu_hybrid():
    """Example 2: Use NPU + GPU hybrid configuration (recommended)."""
    logger.info("\n" + "="*70)
    logger.info("Example 2: NPU + GPU Hybrid Configuration")
    logger.info("="*70)
    
    from rapidocr_hkmc import RapidOCR
    
    config_path = 'config_npu_gpu_hybrid.yaml'
    
    if not Path(config_path).exists():
        logger.error(f"Config file not found: {config_path}")
        logger.info("Please ensure config_npu_gpu_hybrid.yaml is in the current directory")
        return None
    
    # Use hybrid configuration
    ocr = RapidOCR(config_path=config_path)
    
    logger.info("✓ RapidOCR initialized with NPU + GPU hybrid configuration")
    logger.info("Cls model uses NPU, Det/Rec models use GPU")
    logger.info("Check logs above for device selection details")
    
    return ocr


def example_3_npu_only():
    """Example 3: Use NPU-only configuration (power efficient)."""
    logger.info("\n" + "="*70)
    logger.info("Example 3: NPU-Only Configuration")
    logger.info("="*70)
    
    from rapidocr_hkmc import RapidOCR
    
    config_path = 'config_npu_only.yaml'
    
    if not Path(config_path).exists():
        logger.error(f"Config file not found: {config_path}")
        return None
    
    # Use NPU-only configuration
    ocr = RapidOCR(config_path=config_path)
    
    logger.info("✓ RapidOCR initialized with NPU-only configuration")
    logger.info("All models (Cls, Det, Rec) use NPU")
    logger.info("Best for power-efficient inference")
    
    return ocr


def example_4_gpu_only():
    """Example 4: Use GPU-only configuration (maximum throughput)."""
    logger.info("\n" + "="*70)
    logger.info("Example 4: GPU-Only Configuration")
    logger.info("="*70)
    
    from rapidocr_hkmc import RapidOCR
    
    config_path = 'config_gpu_only.yaml'
    
    if not Path(config_path).exists():
        logger.error(f"Config file not found: {config_path}")
        return None
    
    # Use GPU-only configuration
    ocr = RapidOCR(config_path=config_path)
    
    logger.info("✓ RapidOCR initialized with GPU-only configuration")
    logger.info("All models (Cls, Det, Rec) use GPU")
    logger.info("Best for maximum throughput")
    
    return ocr


def example_5_programmatic_config():
    """Example 5: Create configuration programmatically."""
    logger.info("\n" + "="*70)
    logger.info("Example 5: Programmatic Configuration")
    logger.info("="*70)
    
    from rapidocr_hkmc import RapidOCR
    from omegaconf import OmegaConf
    
    # Create config from scratch
    config = OmegaConf.create({
        'EngineConfig': {
            'openvino': {
                'device_name': 'NPU',
                'performance_hint': 'LATENCY',
                'inference_num_threads': -1
            },
            'onnxruntime': {
                'use_cuda': True,
                'cuda_ep_cfg': {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True
                }
            }
        },
        'Cls': {'engine_type': 'openvino'},
        'Det': {'engine_type': 'onnxruntime'},
        'Rec': {'engine_type': 'onnxruntime'}
    })
    
    # Initialize with programmatic config
    ocr = RapidOCR(config=config)
    
    logger.info("✓ RapidOCR initialized with programmatic configuration")
    logger.info("Config created in code without external file")
    
    return ocr


def example_6_auto_select_config():
    """Example 6: Automatically select best configuration based on hardware."""
    logger.info("\n" + "="*70)
    logger.info("Example 6: Auto-Select Best Configuration")
    logger.info("="*70)
    
    from rapidocr_hkmc import RapidOCR
    
    # Check available hardware
    has_npu, has_cuda = check_hardware_availability()
    
    # Select best config
    if has_npu and has_cuda:
        config_path = 'config_npu_gpu_hybrid.yaml'
        logger.info("Selected: NPU + GPU hybrid (best performance)")
    elif has_npu:
        config_path = 'config_npu_only.yaml'
        logger.info("Selected: NPU-only (power efficient)")
    elif has_cuda:
        config_path = 'config_gpu_only.yaml'
        logger.info("Selected: GPU-only (high throughput)")
    else:
        config_path = None
        logger.info("Selected: CPU-only (maximum compatibility)")
    
    # Initialize with selected config
    if config_path and Path(config_path).exists():
        ocr = RapidOCR(config_path=config_path)
    else:
        ocr = RapidOCR()  # Fallback to default
    
    logger.info("✓ RapidOCR initialized with auto-selected configuration")
    
    return ocr


def process_image_example(ocr, image_path='test_image.jpg'):
    """Example of processing an image with OCR."""
    logger.info("\n" + "="*70)
    logger.info("Processing Image Example")
    logger.info("="*70)
    
    if not Path(image_path).exists():
        logger.warning(f"Test image not found: {image_path}")
        logger.info("Skipping image processing example")
        logger.info("To test with your own image, provide the path:")
        logger.info(f"  python {sys.argv[0]} --image your_image.jpg")
        return
    
    logger.info(f"Processing image: {image_path}")
    
    try:
        # Process the image
        result, elapse = ocr(image_path)
        
        # Print results
        logger.info(f"\nOCR Results:")
        logger.info(f"Found {len(result)} text regions")
        
        for i, line in enumerate(result, 1):
            bbox = line[0]  # Bounding box coordinates
            text = line[1]  # Recognized text
            confidence = line[2]  # Confidence score
            
            logger.info(f"\nRegion {i}:")
            logger.info(f"  Text: {text}")
            logger.info(f"  Confidence: {confidence:.2f}")
            logger.info(f"  BBox: {bbox}")
        
        # Print timing information
        logger.info(f"\nTiming Information:")
        logger.info(f"  Detection: {elapse.get('det', 0):.3f}s")
        logger.info(f"  Classification: {elapse.get('cls', 0):.3f}s")
        logger.info(f"  Recognition: {elapse.get('rec', 0):.3f}s")
        logger.info(f"  Total: {sum(elapse.values()):.3f}s")
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to run all examples."""
    logger.info("="*70)
    logger.info("RapidOCR HKMC Configuration Examples")
    logger.info("="*70)
    
    # Check hardware availability first
    has_npu, has_cuda = check_hardware_availability()
    
    # Run examples based on command line arguments
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        image_path = sys.argv[2] if len(sys.argv) > 2 else 'test_image.jpg'
        
        examples = {
            '1': example_1_default_cpu,
            '2': example_2_npu_gpu_hybrid,
            '3': example_3_npu_only,
            '4': example_4_gpu_only,
            '5': example_5_programmatic_config,
            '6': example_6_auto_select_config,
        }
        
        if example_num in examples:
            ocr = examples[example_num]()
            if ocr:
                process_image_example(ocr, image_path)
        else:
            logger.error(f"Unknown example: {example_num}")
            print_usage()
    else:
        # Run all examples (without image processing)
        logger.info("\nRunning all configuration examples...\n")
        
        example_1_default_cpu()
        example_2_npu_gpu_hybrid()
        example_3_npu_only()
        example_4_gpu_only()
        example_5_programmatic_config()
        example_6_auto_select_config()
        
        logger.info("\n" + "="*70)
        logger.info("All examples completed!")
        logger.info("="*70)
        logger.info("\nTo test with an image, run:")
        logger.info(f"  python {sys.argv[0]} <example_number> <image_path>")
        logger.info("\nExample:")
        logger.info(f"  python {sys.argv[0]} 6 my_document.jpg")


def print_usage():
    """Print usage information."""
    print("\nUsage:")
    print(f"  python {sys.argv[0]}                    # Run all examples")
    print(f"  python {sys.argv[0]} <num> [image]     # Run specific example")
    print("\nExamples:")
    print("  1 - Default CPU configuration")
    print("  2 - NPU + GPU hybrid (recommended)")
    print("  3 - NPU-only (power efficient)")
    print("  4 - GPU-only (maximum throughput)")
    print("  5 - Programmatic configuration")
    print("  6 - Auto-select best configuration")
    print("\nExample commands:")
    print(f"  python {sys.argv[0]} 1")
    print(f"  python {sys.argv[0]} 6 document.jpg")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
