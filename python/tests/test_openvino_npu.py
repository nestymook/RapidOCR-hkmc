# -*- encoding: utf-8 -*-
# Test NPU support for OpenVINO inference engine

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from omegaconf import DictConfig

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from rapidocr_hkmc.inference_engine.openvino import (
    ConfigurationError,
    OpenVINOInferSession,
    OpenVIONError,
)


class TestOpenVINONPUDeviceSelection:
    """Test OpenVINO NPU device selection functionality."""

    @patch("rapidocr_hkmc.inference_engine.openvino.OpenVINOInferSession._verify_model")
    @patch("rapidocr_hkmc.inference_engine.openvino.Core")
    @patch("rapidocr_hkmc.inference_engine.openvino.DownloadFile")
    def test_npu_device_selection_when_available(
        self, mock_download, mock_core_class, mock_verify
    ):
        """Test that NPU device is selected when available."""
        # Setup mock Core instance
        mock_core = Mock()
        mock_core_class.return_value = mock_core
        mock_core.available_devices = ["CPU", "NPU", "GPU"]
        
        # Setup mock model and compilation
        mock_model = Mock()
        mock_core.read_model.return_value = mock_model
        
        mock_compiled_model = Mock()
        mock_infer_request = Mock()
        mock_compiled_model.create_infer_request.return_value = mock_infer_request
        mock_core.compile_model.return_value = mock_compiled_model

        # Create configuration with NPU device
        cfg = DictConfig({
            "model_path": str(root_dir / "tests" / "test_files" / "ch_ppocr_mobile_v2.0_cls_infer.onnx"),
            "engine_type": "openvino",
            "ocr_version": "v4",
            "task_type": "cls",
            "lang_type": "ch",
            "model_type": "mobile",
            "engine_cfg": {
                "device_name": "NPU",
                "inference_num_threads": -1,
            },
        })

        # Initialize session
        session = OpenVINOInferSession(cfg)

        # Verify NPU was requested and used
        assert mock_core.compile_model.called
        compile_call_args = mock_core.compile_model.call_args
        assert compile_call_args[1]["device_name"] == "NPU"
        
        # Verify session was created
        assert session.session == mock_infer_request

    @patch("rapidocr_hkmc.inference_engine.openvino.OpenVINOInferSession._verify_model")
    @patch("rapidocr_hkmc.inference_engine.openvino.Core")
    @patch("rapidocr_hkmc.inference_engine.openvino.DownloadFile")
    def test_cpu_device_selection_default(self, mock_download, mock_core_class, mock_verify):
        """Test that CPU is used as default when device_name not specified."""
        # Setup mock Core instance
        mock_core = Mock()
        mock_core_class.return_value = mock_core
        mock_core.available_devices = ["CPU"]
        
        # Setup mock model and compilation
        mock_model = Mock()
        mock_core.read_model.return_value = mock_model
        
        mock_compiled_model = Mock()
        mock_infer_request = Mock()
        mock_compiled_model.create_infer_request.return_value = mock_infer_request
        mock_core.compile_model.return_value = mock_compiled_model

        # Create configuration without device_name (should default to CPU)
        cfg = DictConfig({
            "model_path": str(root_dir / "tests" / "test_files" / "ch_ppocr_mobile_v2.0_cls_infer.onnx"),
            "engine_type": "openvino",
            "ocr_version": "v4",
            "task_type": "cls",
            "lang_type": "ch",
            "model_type": "mobile",
            "engine_cfg": {
                "inference_num_threads": -1,
            },
        })

        # Initialize session
        session = OpenVINOInferSession(cfg)

        # Verify CPU was used
        assert mock_core.compile_model.called
        compile_call_args = mock_core.compile_model.call_args
        assert compile_call_args[1]["device_name"] == "CPU"

    @patch("rapidocr_hkmc.inference_engine.openvino.OpenVINOInferSession._verify_model")
    @patch("rapidocr_hkmc.inference_engine.openvino.Core")
    @patch("rapidocr_hkmc.inference_engine.openvino.DownloadFile")
    def test_gpu_device_selection_when_available(
        self, mock_download, mock_core_class, mock_verify
    ):
        """Test that GPU device is selected when available."""
        # Setup mock Core instance
        mock_core = Mock()
        mock_core_class.return_value = mock_core
        mock_core.available_devices = ["CPU", "GPU"]
        
        # Setup mock model and compilation
        mock_model = Mock()
        mock_core.read_model.return_value = mock_model
        
        mock_compiled_model = Mock()
        mock_infer_request = Mock()
        mock_compiled_model.create_infer_request.return_value = mock_infer_request
        mock_core.compile_model.return_value = mock_compiled_model

        # Create configuration with GPU device
        cfg = DictConfig({
            "model_path": str(root_dir / "tests" / "test_files" / "ch_ppocr_mobile_v2.0_cls_infer.onnx"),
            "engine_type": "openvino",
            "ocr_version": "v4",
            "task_type": "cls",
            "lang_type": "ch",
            "model_type": "mobile",
            "engine_cfg": {
                "device_name": "GPU",
                "inference_num_threads": -1,
            },
        })

        # Initialize session
        session = OpenVINOInferSession(cfg)

        # Verify GPU was requested and used
        assert mock_core.compile_model.called
        compile_call_args = mock_core.compile_model.call_args
        assert compile_call_args[1]["device_name"] == "GPU"


class TestOpenVINONPUFallback:
    """Test NPU fallback to CPU when NPU is unavailable."""

    @patch("rapidocr_hkmc.inference_engine.openvino.OpenVINOInferSession._verify_model")
    @patch("rapidocr_hkmc.inference_engine.openvino.Core")
    @patch("rapidocr_hkmc.inference_engine.openvino.DownloadFile")
    @patch("rapidocr_hkmc.inference_engine.openvino.logger")
    def test_npu_unavailable_fallback_to_cpu(
        self, mock_logger, mock_download, mock_core_class, mock_verify
    ):
        """Test fallback to CPU when NPU is requested but unavailable."""
        # Setup mock Core instance - NPU not in available devices
        mock_core = Mock()
        mock_core_class.return_value = mock_core
        mock_core.available_devices = ["CPU"]  # Only CPU available
        
        # Setup mock model and compilation
        mock_model = Mock()
        mock_core.read_model.return_value = mock_model
        
        mock_compiled_model = Mock()
        mock_infer_request = Mock()
        mock_compiled_model.create_infer_request.return_value = mock_infer_request
        mock_core.compile_model.return_value = mock_compiled_model

        # Create configuration requesting NPU
        cfg = DictConfig({
            "model_path": str(root_dir / "tests" / "test_files" / "ch_ppocr_mobile_v2.0_cls_infer.onnx"),
            "engine_type": "openvino",
            "ocr_version": "v4",
            "task_type": "cls",
            "lang_type": "ch",
            "model_type": "mobile",
            "engine_cfg": {
                "device_name": "NPU",  # Request NPU
                "inference_num_threads": -1,
            },
        })

        # Initialize session
        session = OpenVINOInferSession(cfg)

        # Verify warning was logged about NPU unavailability
        assert mock_logger.warning.called
        warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
        assert any("NPU" in call and "not available" in call for call in warning_calls)
        
        # Verify CPU was used as fallback
        assert mock_core.compile_model.called
        compile_call_args = mock_core.compile_model.call_args
        assert compile_call_args[1]["device_name"] == "CPU"
        
        # Verify session was created successfully
        assert session.session == mock_infer_request

    @patch("rapidocr_hkmc.inference_engine.openvino.OpenVINOInferSession._verify_model")
    @patch("rapidocr_hkmc.inference_engine.openvino.Core")
    @patch("rapidocr_hkmc.inference_engine.openvino.DownloadFile")
    @patch("rapidocr_hkmc.inference_engine.openvino.logger")
    def test_gpu_unavailable_fallback_to_cpu(
        self, mock_logger, mock_download, mock_core_class, mock_verify
    ):
        """Test fallback to CPU when GPU is requested but unavailable."""
        # Setup mock Core instance - GPU not in available devices
        mock_core = Mock()
        mock_core_class.return_value = mock_core
        mock_core.available_devices = ["CPU"]  # Only CPU available
        
        # Setup mock model and compilation
        mock_model = Mock()
        mock_core.read_model.return_value = mock_model
        
        mock_compiled_model = Mock()
        mock_infer_request = Mock()
        mock_compiled_model.create_infer_request.return_value = mock_infer_request
        mock_core.compile_model.return_value = mock_compiled_model

        # Create configuration requesting GPU
        cfg = DictConfig({
            "model_path": str(root_dir / "tests" / "test_files" / "ch_ppocr_mobile_v2.0_cls_infer.onnx"),
            "engine_type": "openvino",
            "ocr_version": "v4",
            "task_type": "cls",
            "lang_type": "ch",
            "model_type": "mobile",
            "engine_cfg": {
                "device_name": "GPU",  # Request GPU
                "inference_num_threads": -1,
            },
        })

        # Initialize session
        session = OpenVINOInferSession(cfg)

        # Verify warning was logged about GPU unavailability
        assert mock_logger.warning.called
        warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
        assert any("GPU" in call and "not available" in call for call in warning_calls)
        
        # Verify CPU was used as fallback
        assert mock_core.compile_model.called
        compile_call_args = mock_core.compile_model.call_args
        assert compile_call_args[1]["device_name"] == "CPU"

    @patch("rapidocr_hkmc.inference_engine.openvino.OpenVINOInferSession._verify_model")
    @patch("rapidocr_hkmc.inference_engine.openvino.Core")
    @patch("rapidocr_hkmc.inference_engine.openvino.DownloadFile")
    @patch("rapidocr_hkmc.inference_engine.openvino.logger")
    def test_cpu_execution_after_npu_fallback(
        self, mock_logger, mock_download, mock_core_class, mock_verify
    ):
        """Test that CPU execution works correctly after NPU fallback."""
        # Setup mock Core instance
        mock_core = Mock()
        mock_core_class.return_value = mock_core
        mock_core.available_devices = ["CPU"]
        
        # Setup mock model and compilation
        mock_model = Mock()
        mock_core.read_model.return_value = mock_model
        
        mock_compiled_model = Mock()
        mock_infer_request = Mock()
        
        # Setup mock inference output
        mock_output_tensor = Mock()
        mock_output_tensor.data = np.array([[0.1, 0.9]])
        mock_infer_request.get_output_tensor.return_value = mock_output_tensor
        
        mock_compiled_model.create_infer_request.return_value = mock_infer_request
        mock_core.compile_model.return_value = mock_compiled_model

        # Create configuration requesting NPU
        cfg = DictConfig({
            "model_path": str(root_dir / "tests" / "test_files" / "ch_ppocr_mobile_v2.0_cls_infer.onnx"),
            "engine_type": "openvino",
            "ocr_version": "v4",
            "task_type": "cls",
            "lang_type": "ch",
            "model_type": "mobile",
            "engine_cfg": {
                "device_name": "NPU",
                "inference_num_threads": -1,
            },
        })

        # Initialize session
        session = OpenVINOInferSession(cfg)

        # Test inference execution
        test_input = np.random.rand(1, 3, 48, 192).astype(np.float32)
        result = session(test_input)

        # Verify inference was called and returned correct result
        assert mock_infer_request.infer.called
        assert result is not None
        assert isinstance(result, np.ndarray)


class TestOpenVINOConfigurationValidation:
    """Test configuration validation for OpenVINO engine."""

    @patch("rapidocr_hkmc.inference_engine.openvino.OpenVINOInferSession._verify_model")
    @patch("rapidocr_hkmc.inference_engine.openvino.Core")
    @patch("rapidocr_hkmc.inference_engine.openvino.DownloadFile")
    def test_invalid_device_name_raises_error(self, mock_download, mock_core_class, mock_verify):
        """Test that invalid device names raise ConfigurationError."""
        # Setup mock Core instance
        mock_core = Mock()
        mock_core_class.return_value = mock_core
        mock_core.available_devices = ["CPU"]

        # Create configuration with invalid device name
        cfg = DictConfig({
            "model_path": str(root_dir / "tests" / "test_files" / "ch_ppocr_mobile_v2.0_cls_infer.onnx"),
            "engine_type": "openvino",
            "ocr_version": "v4",
            "task_type": "cls",
            "lang_type": "ch",
            "model_type": "mobile",
            "engine_cfg": {
                "device_name": "INVALID_DEVICE",  # Invalid device
                "inference_num_threads": -1,
            },
        })

        # Verify ConfigurationError is raised
        with pytest.raises(ConfigurationError) as exc_info:
            OpenVINOInferSession(cfg)
        
        # Verify error message is descriptive
        error_message = str(exc_info.value)
        assert "Invalid device_name" in error_message
        assert "INVALID_DEVICE" in error_message
        assert "CPU" in error_message
        assert "NPU" in error_message
        assert "GPU" in error_message

    @patch("rapidocr_hkmc.inference_engine.openvino.OpenVINOInferSession._verify_model")
    @patch("rapidocr_hkmc.inference_engine.openvino.Core")
    @patch("rapidocr_hkmc.inference_engine.openvino.DownloadFile")
    def test_invalid_performance_hint_raises_error(
        self, mock_download, mock_core_class, mock_verify
    ):
        """Test that invalid performance hints raise ConfigurationError."""
        # Setup mock Core instance
        mock_core = Mock()
        mock_core_class.return_value = mock_core
        mock_core.available_devices = ["CPU", "NPU"]

        # Create configuration with invalid performance hint
        cfg = DictConfig({
            "model_path": str(root_dir / "tests" / "test_files" / "ch_ppocr_mobile_v2.0_cls_infer.onnx"),
            "engine_type": "openvino",
            "ocr_version": "v4",
            "task_type": "cls",
            "lang_type": "ch",
            "model_type": "mobile",
            "engine_cfg": {
                "device_name": "NPU",
                "performance_hint": "INVALID_HINT",  # Invalid hint
                "inference_num_threads": -1,
            },
        })

        # Verify ConfigurationError is raised
        with pytest.raises(ConfigurationError) as exc_info:
            OpenVINOInferSession(cfg)
        
        # Verify error message is descriptive
        error_message = str(exc_info.value)
        assert "Invalid performance_hint" in error_message
        assert "INVALID_HINT" in error_message
        assert "LATENCY" in error_message
        assert "THROUGHPUT" in error_message

    @patch("rapidocr_hkmc.inference_engine.openvino.OpenVINOInferSession._verify_model")
    @patch("rapidocr_hkmc.inference_engine.openvino.Core")
    @patch("rapidocr_hkmc.inference_engine.openvino.DownloadFile")
    def test_valid_performance_hints_accepted(self, mock_download, mock_core_class, mock_verify):
        """Test that valid performance hints are accepted."""
        # Setup mock Core instance
        mock_core = Mock()
        mock_core_class.return_value = mock_core
        mock_core.available_devices = ["CPU", "NPU"]
        
        # Setup mock model and compilation
        mock_model = Mock()
        mock_core.read_model.return_value = mock_model
        
        mock_compiled_model = Mock()
        mock_infer_request = Mock()
        mock_compiled_model.create_infer_request.return_value = mock_infer_request
        mock_core.compile_model.return_value = mock_compiled_model

        valid_hints = ["LATENCY", "THROUGHPUT", "CUMULATIVE_THROUGHPUT"]
        
        for hint in valid_hints:
            # Create configuration with valid performance hint
            cfg = DictConfig({
                "model_path": str(root_dir / "tests" / "test_files" / "ch_ppocr_mobile_v2.0_cls_infer.onnx"),
                "engine_type": "openvino",
                "ocr_version": "v4",
                "task_type": "cls",
                "lang_type": "ch",
                "model_type": "mobile",
                "engine_cfg": {
                    "device_name": "NPU",
                    "performance_hint": hint,
                    "inference_num_threads": -1,
                },
            })

            # Should not raise any error
            session = OpenVINOInferSession(cfg)
            assert session is not None

    @patch("rapidocr_hkmc.inference_engine.openvino.OpenVINOInferSession._verify_model")
    @patch("rapidocr_hkmc.inference_engine.openvino.Core")
    @patch("rapidocr_hkmc.inference_engine.openvino.DownloadFile")
    def test_invalid_thread_count_raises_error(self, mock_download, mock_core_class, mock_verify):
        """Test that invalid thread counts raise ConfigurationError."""
        # Setup mock Core instance
        mock_core = Mock()
        mock_core_class.return_value = mock_core
        mock_core.available_devices = ["CPU"]

        # Create configuration with invalid thread count
        cfg = DictConfig({
            "model_path": str(root_dir / "tests" / "test_files" / "ch_ppocr_mobile_v2.0_cls_infer.onnx"),
            "engine_type": "openvino",
            "ocr_version": "v4",
            "task_type": "cls",
            "lang_type": "ch",
            "model_type": "mobile",
            "engine_cfg": {
                "device_name": "CPU",
                "inference_num_threads": 9999,  # Invalid - exceeds CPU count
            },
        })

        # Verify ConfigurationError is raised
        with pytest.raises(ConfigurationError) as exc_info:
            OpenVINOInferSession(cfg)
        
        # Verify error message is descriptive
        error_message = str(exc_info.value)
        assert "Invalid inference_num_threads" in error_message
        assert "9999" in error_message

    @patch("rapidocr_hkmc.inference_engine.openvino.OpenVINOInferSession._verify_model")
    @patch("rapidocr_hkmc.inference_engine.openvino.Core")
    @patch("rapidocr_hkmc.inference_engine.openvino.DownloadFile")
    def test_configuration_error_messages_are_descriptive(
        self, mock_download, mock_core_class, mock_verify
    ):
        """Test that all configuration errors provide clear, actionable messages."""
        # Setup mock Core instance
        mock_core = Mock()
        mock_core_class.return_value = mock_core
        mock_core.available_devices = ["CPU"]

        # Test various invalid configurations
        invalid_configs = [
            {
                "device_name": "FPGA",
                "expected_in_message": ["Invalid device_name", "FPGA", "CPU", "NPU", "GPU"],
            },
            {
                "device_name": "CPU",
                "performance_hint": "FAST",
                "expected_in_message": ["Invalid performance_hint", "FAST", "LATENCY"],
            },
            {
                "device_name": "CPU",
                "inference_num_threads": 0,
                "expected_in_message": ["Invalid inference_num_threads", "0"],
            },
        ]

        for invalid_cfg in invalid_configs:
            cfg = DictConfig({
                "model_path": str(root_dir / "tests" / "test_files" / "ch_ppocr_mobile_v2.0_cls_infer.onnx"),
                "engine_type": "openvino",
                "ocr_version": "v4",
                "task_type": "cls",
                "lang_type": "ch",
                "model_type": "mobile",
                "engine_cfg": invalid_cfg.copy(),
            })
            
            # Remove expected_in_message from engine_cfg
            expected_messages = cfg["engine_cfg"].pop("expected_in_message")

            with pytest.raises(ConfigurationError) as exc_info:
                OpenVINOInferSession(cfg)
            
            error_message = str(exc_info.value)
            for expected_text in expected_messages:
                assert expected_text in error_message, (
                    f"Expected '{expected_text}' in error message, "
                    f"but got: {error_message}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
