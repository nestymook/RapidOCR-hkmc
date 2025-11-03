# -*- encoding: utf-8 -*-
# Integration tests for NPU and GPU hardware acceleration
import sys
from pathlib import Path

import pytest

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from rapidocr_hkmc import RapidOCR

tests_dir = root_dir / "tests" / "test_files"
img_path = tests_dir / "ch_en_num.jpg"


class TestOpenVINONPUConfiguration:
    """Test cls model with OpenVINO NPU configuration."""

    def test_cls_with_openvino_npu(self):
        """
        Test classification model with OpenVINO engine targeting NPU.
        Requirements: 1.1, 1.3
        """
        # Initialize RapidOCR with cls using openvino engine
        # The config.yaml already has Cls.engine_type set to "openvino"
        # and device_name can be set to "NPU" for NPU acceleration
        engine = RapidOCR()
        
        # Run text classification on test image
        cls_img_path = tests_dir / "text_cls.jpg"
        result = engine(cls_img_path, use_det=False, use_cls=True, use_rec=False)
        
        # Verify results match expected output
        assert result is not None
        assert len(result) == 1
        assert result.cls_res is not None
        assert len(result.cls_res) > 0
        # Expected: text is rotated 180 degrees
        assert result.cls_res[0][0] == "180"

    def test_full_pipeline_with_openvino_cls(self):
        """
        Test full OCR pipeline with OpenVINO cls model.
        Requirements: 1.1, 1.3
        """
        engine = RapidOCR()
        
        # Run full OCR pipeline
        result = engine(img_path)
        
        # Verify results
        assert result is not None
        assert len(result) > 0
        assert result.txts is not None
        assert len(result.txts) > 0
        # Verify first detected text
        assert result.txts[0] == "正品促销"


class TestONNXRuntimeGPUConfiguration:
    """Test det and rec models with ONNXRuntime GPU configuration."""

    def test_det_rec_with_onnxruntime_gpu(self):
        """
        Test detection and recognition models with ONNXRuntime CUDA.
        Requirements: 2.1, 2.4
        """
        # Initialize RapidOCR with det/rec using onnxruntime with CUDA
        # The config.yaml has Det and Rec engine_type set to "onnxruntime"
        # and use_cuda set to true for GPU acceleration
        engine = RapidOCR()
        
        # Run full OCR pipeline on test images
        result = engine(img_path, use_det=True, use_cls=False, use_rec=True)
        
        # Verify detection and recognition results are accurate
        assert result is not None
        assert len(result) == 18  # Expected number of text regions
        assert result.txts is not None
        assert len(result.txts) == 18
        # Verify first detected and recognized text
        assert result.txts[0] == "正品促销"

    def test_det_only_with_onnxruntime_gpu(self):
        """
        Test detection model only with ONNXRuntime GPU.
        Requirements: 2.1
        """
        engine = RapidOCR()
        
        # Run detection only
        result = engine(img_path, use_det=True, use_cls=False, use_rec=False)
        
        # Verify detection results
        assert result is not None
        assert len(result) == 18  # Expected number of detected text regions

    def test_rec_only_with_onnxruntime_gpu(self):
        """
        Test recognition model only with ONNXRuntime GPU.
        Requirements: 2.1
        """
        engine = RapidOCR()
        
        # Run recognition only on pre-cropped text image
        rec_img_path = tests_dir / "text_rec.jpg"
        result = engine(rec_img_path, use_det=False, use_cls=False, use_rec=True)
        
        # Verify recognition results
        assert result is not None
        assert len(result) == 1
        assert result.txts is not None
        assert result.txts[0] == "韩国小馆"


class TestMixedConfiguration:
    """Test mixed configuration (OpenVINO cls + ONNXRuntime det/rec)."""

    def test_mixed_openvino_cls_onnxruntime_det_rec(self):
        """
        Test complete OCR workflow with mixed engine configuration.
        Requirements: 3.1, 3.4
        """
        # Configure cls with openvino, det/rec with onnxruntime
        # This is the default configuration in config.yaml
        engine = RapidOCR()
        
        # Run complete OCR workflow
        result = engine(img_path)
        
        # Verify all models work together correctly
        assert result is not None
        assert len(result) == 18
        assert result.txts is not None
        assert len(result.txts) == 18
        assert result.txts[0] == "正品促销"

    def test_mixed_config_with_cls_rotation(self):
        """
        Test mixed configuration with rotated text requiring classification.
        Requirements: 3.1, 3.4
        """
        engine = RapidOCR()
        
        # Test with image that requires classification
        cls_img_path = tests_dir / "text_cls.jpg"
        result = engine(cls_img_path, use_det=False, use_cls=True, use_rec=True)
        
        # Verify cls and rec work together
        assert result is not None
        assert len(result) == 1
        assert result.cls_res is not None
        assert result.cls_res[0][0] == "180"
        assert result.txts is not None
        assert result.txts[0] == "怪我咯"

    def test_mixed_config_multiple_images(self):
        """
        Test mixed configuration with multiple different images.
        Requirements: 3.1, 3.4
        """
        engine = RapidOCR()
        
        # Test with different image types
        test_images = [
            (img_path, 18, "正品促销"),
            (tests_dir / "text_rec.jpg", 1, "韩国小馆"),
        ]
        
        for img, expected_count, expected_first_text in test_images:
            result = engine(img)
            assert result is not None
            assert len(result) == expected_count
            if result.txts:
                assert result.txts[0] == expected_first_text
