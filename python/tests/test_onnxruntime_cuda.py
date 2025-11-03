# -*- encoding: utf-8 -*-
# Test ONNXRuntime CUDA execution provider configuration and GPU support

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from omegaconf import DictConfig

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from rapidocr_hkmc.inference_engine.onnxruntime.main import OrtInferSession
from rapidocr_hkmc.inference_engine.onnxruntime.provider_config import (
    EP,
    ProviderConfig,
)


class TestCUDAExecutionProvider:
    """Test CUDA execution provider configuration for OrtInferSession."""

    @patch("rapidocr_hkmc.inference_engine.onnxruntime.provider_config.get_device")
    @patch(
        "rapidocr_hkmc.inference_engine.onnxruntime.provider_config.get_available_providers"
    )
    def test_cuda_provider_in_session_initialization(
        self, mock_get_providers, mock_get_device
    ):
        """Test that CUDA EP is configured when initializing OrtInferSession with use_cuda=True."""
        # Simulate CUDA available
        mock_get_device.return_value = "GPU"
        mock_get_providers.return_value = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

        # Configuration with CUDA enabled
        engine_cfg = DictConfig({
            "use_cuda": True,
            "use_dml": False,
            "use_cann": False,
            "enable_cpu_mem_arena": False,
            "cpu_ep_cfg": {"arena_extend_strategy": "kSameAsRequested"},
            "cuda_ep_cfg": {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
            },
        })

        # Create ProviderConfig and get provider list
        provider_cfg = ProviderConfig(engine_cfg=engine_cfg)
        ep_list = provider_cfg.get_ep_list()

        # Verify CUDA is in the provider list
        assert len(ep_list) >= 1

        # CUDA should be first provider (highest priority)
        cuda_provider = ep_list[0]
        assert cuda_provider[0] == EP.CUDA_EP.value

        # Verify CUDA configuration is included
        cuda_config = cuda_provider[1]
        assert cuda_config["device_id"] == 0
        assert cuda_config["arena_extend_strategy"] == "kNextPowerOfTwo"
        assert cuda_config["cudnn_conv_algo_search"] == "EXHAUSTIVE"
        assert cuda_config["do_copy_in_default_stream"] is True

    @patch("rapidocr_hkmc.inference_engine.onnxruntime.provider_config.get_device")
    @patch(
        "rapidocr_hkmc.inference_engine.onnxruntime.provider_config.get_available_providers"
    )
    def test_cuda_config_parameters_applied_correctly(
        self, mock_get_providers, mock_get_device
    ):
        """Test that cuda_ep_cfg parameters are correctly applied."""
        mock_get_device.return_value = "GPU"
        mock_get_providers.return_value = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

        # Custom CUDA configuration
        cuda_config = {
            "device_id": 2,  # Third GPU
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": True,
        }

        engine_cfg = DictConfig({
            "use_cuda": True,
            "use_dml": False,
            "use_cann": False,
            "cpu_ep_cfg": {"arena_extend_strategy": "kSameAsRequested"},
            "cuda_ep_cfg": cuda_config,
        })

        provider_cfg = ProviderConfig(engine_cfg=engine_cfg)
        ep_list = provider_cfg.get_ep_list()

        # Verify CUDA is first provider
        assert ep_list[0][0] == EP.CUDA_EP.value

        # Verify all CUDA config parameters are present
        cuda_ep_config = ep_list[0][1]
        assert cuda_ep_config["device_id"] == 2
        assert cuda_ep_config["arena_extend_strategy"] == "kNextPowerOfTwo"
        assert cuda_ep_config["cudnn_conv_algo_search"] == "EXHAUSTIVE"
        assert cuda_ep_config["do_copy_in_default_stream"] is True

    @patch("rapidocr_hkmc.inference_engine.onnxruntime.provider_config.get_device")
    @patch(
        "rapidocr_hkmc.inference_engine.onnxruntime.provider_config.get_available_providers"
    )
    def test_multi_gpu_device_selection(self, mock_get_providers, mock_get_device):
        """Test device_id parameter for multi-GPU systems."""
        mock_get_device.return_value = "GPU"
        mock_get_providers.return_value = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

        # Test different device IDs
        for device_id in [0, 1, 2, 3]:
            engine_cfg = DictConfig({
                "use_cuda": True,
                "use_dml": False,
                "use_cann": False,
                "cpu_ep_cfg": {"arena_extend_strategy": "kSameAsRequested"},
                "cuda_ep_cfg": {
                    "device_id": device_id,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                },
            })

            provider_cfg = ProviderConfig(engine_cfg=engine_cfg)
            ep_list = provider_cfg.get_ep_list()

            # Verify correct device_id is set
            cuda_config = ep_list[0][1]
            assert cuda_config["device_id"] == device_id

    @patch("rapidocr_hkmc.inference_engine.onnxruntime.provider_config.get_device")
    @patch(
        "rapidocr_hkmc.inference_engine.onnxruntime.provider_config.get_available_providers"
    )
    def test_cuda_memory_allocation_strategies(
        self, mock_get_providers, mock_get_device
    ):
        """Test different CUDA memory allocation strategies."""
        mock_get_device.return_value = "GPU"
        mock_get_providers.return_value = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

        strategies = ["kNextPowerOfTwo", "kSameAsRequested"]

        for strategy in strategies:
            engine_cfg = DictConfig({
                "use_cuda": True,
                "use_dml": False,
                "use_cann": False,
                "cpu_ep_cfg": {"arena_extend_strategy": "kSameAsRequested"},
                "cuda_ep_cfg": {
                    "device_id": 0,
                    "arena_extend_strategy": strategy,
                },
            })

            provider_cfg = ProviderConfig(engine_cfg=engine_cfg)
            ep_list = provider_cfg.get_ep_list()

            cuda_config = ep_list[0][1]
            assert cuda_config["arena_extend_strategy"] == strategy

    @patch("rapidocr_hkmc.inference_engine.onnxruntime.provider_config.get_device")
    @patch(
        "rapidocr_hkmc.inference_engine.onnxruntime.provider_config.get_available_providers"
    )
    def test_cudnn_conv_algo_search_options(
        self, mock_get_providers, mock_get_device
    ):
        """Test different cuDNN convolution algorithm search options."""
        mock_get_device.return_value = "GPU"
        mock_get_providers.return_value = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

        search_options = ["EXHAUSTIVE", "HEURISTIC", "DEFAULT"]

        for option in search_options:
            engine_cfg = DictConfig({
                "use_cuda": True,
                "use_dml": False,
                "use_cann": False,
                "cpu_ep_cfg": {"arena_extend_strategy": "kSameAsRequested"},
                "cuda_ep_cfg": {
                    "device_id": 0,
                    "cudnn_conv_algo_search": option,
                },
            })

            provider_cfg = ProviderConfig(engine_cfg=engine_cfg)
            ep_list = provider_cfg.get_ep_list()

            cuda_config = ep_list[0][1]
            assert cuda_config["cudnn_conv_algo_search"] == option


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestGPUFallbackToCPU:
    """Test GPU fallback to CPU when CUDA is unavailable."""

    @patch("rapidocr_hkmc.inference_engine.onnxruntime.provider_config.get_device")
    @patch(
        "rapidocr_hkmc.inference_engine.onnxruntime.provider_config.get_available_providers"
    )
    @patch("rapidocr_hkmc.inference_engine.onnxruntime.provider_config.logger")
    def test_cuda_unavailable_fallback_to_cpu(
        self, mock_logger, mock_get_providers, mock_get_device
    ):
        """Test that system falls back to CPU when CUDA is requested but unavailable."""
        # Simulate CUDA not available
        mock_get_device.return_value = "CPU"
        mock_get_providers.return_value = ["CPUExecutionProvider"]

        engine_cfg = DictConfig({
            "use_cuda": True,  # User requests CUDA
            "use_dml": False,
            "use_cann": False,
            "cpu_ep_cfg": {"arena_extend_strategy": "kSameAsRequested"},
            "cuda_ep_cfg": {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
            },
        })

        provider_cfg = ProviderConfig(engine_cfg=engine_cfg)

        # is_cuda_available should return False
        assert provider_cfg.is_cuda_available() is False

        # Should only have CPU provider in the list
        ep_list = provider_cfg.get_ep_list()
        assert len(ep_list) == 1
        assert ep_list[0][0] == EP.CPU_EP.value

        # Verify warning was logged
        assert mock_logger.warning.called

    @patch("rapidocr_hkmc.inference_engine.onnxruntime.provider_config.get_device")
    @patch(
        "rapidocr_hkmc.inference_engine.onnxruntime.provider_config.get_available_providers"
    )
    @patch("rapidocr_hkmc.inference_engine.onnxruntime.provider_config.logger")
    def test_warning_logged_on_cuda_unavailable(
        self, mock_logger, mock_get_providers, mock_get_device
    ):
        """Test that warning is logged when CUDA is requested but unavailable."""
        # Simulate CUDA not available
        mock_get_device.return_value = "CPU"
        mock_get_providers.return_value = ["CPUExecutionProvider"]

        engine_cfg = DictConfig({
            "use_cuda": True,
            "use_dml": False,
            "use_cann": False,
            "cpu_ep_cfg": {"arena_extend_strategy": "kSameAsRequested"},
            "cuda_ep_cfg": {"device_id": 0},
        })

        provider_cfg = ProviderConfig(engine_cfg=engine_cfg)
        provider_cfg.is_cuda_available()

        # Verify warning was logged about CUDA not being available
        assert mock_logger.warning.called
        warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
        assert any("CUDAExecutionProvider" in call for call in warning_calls)

    @patch("rapidocr_hkmc.inference_engine.onnxruntime.provider_config.get_device")
    @patch(
        "rapidocr_hkmc.inference_engine.onnxruntime.provider_config.get_available_providers"
    )
    def test_cpu_execution_after_gpu_fallback(
        self, mock_get_providers, mock_get_device
    ):
        """Test that CPU execution works correctly after GPU fallback."""
        # Simulate CUDA not available
        mock_get_device.return_value = "CPU"
        mock_get_providers.return_value = ["CPUExecutionProvider"]

        engine_cfg = DictConfig({
            "use_cuda": True,  # Request CUDA
            "use_dml": False,
            "use_cann": False,
            "cpu_ep_cfg": {"arena_extend_strategy": "kSameAsRequested"},
            "cuda_ep_cfg": {"device_id": 0},
        })

        provider_cfg = ProviderConfig(engine_cfg=engine_cfg)
        ep_list = provider_cfg.get_ep_list()

        # Verify CPU provider is configured correctly
        assert len(ep_list) == 1
        assert ep_list[0][0] == EP.CPU_EP.value

        # Verify CPU configuration is applied
        cpu_config = ep_list[0][1]
        assert "arena_extend_strategy" in cpu_config
        assert cpu_config["arena_extend_strategy"] == "kSameAsRequested"

    @patch("rapidocr_hkmc.inference_engine.onnxruntime.provider_config.get_device")
    @patch(
        "rapidocr_hkmc.inference_engine.onnxruntime.provider_config.get_available_providers"
    )
    def test_cuda_disabled_uses_cpu(self, mock_get_providers, mock_get_device):
        """Test that CPU is used when use_cuda is False."""
        # Even if CUDA is available, it shouldn't be used
        mock_get_device.return_value = "GPU"
        mock_get_providers.return_value = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

        engine_cfg = DictConfig({
            "use_cuda": False,  # CUDA disabled
            "use_dml": False,
            "use_cann": False,
            "cpu_ep_cfg": {"arena_extend_strategy": "kSameAsRequested"},
            "cuda_ep_cfg": {"device_id": 0},
        })

        provider_cfg = ProviderConfig(engine_cfg=engine_cfg)
        ep_list = provider_cfg.get_ep_list()

        # Should only have CPU provider
        assert len(ep_list) == 1
        assert ep_list[0][0] == EP.CPU_EP.value

    @patch("rapidocr_hkmc.inference_engine.onnxruntime.provider_config.get_device")
    @patch(
        "rapidocr_hkmc.inference_engine.onnxruntime.provider_config.get_available_providers"
    )
    @patch("rapidocr_hkmc.inference_engine.onnxruntime.provider_config.logger")
    def test_verify_providers_warns_on_fallback(
        self, mock_logger, mock_get_providers, mock_get_device
    ):
        """Test that verify_providers logs warning when CUDA requested but CPU used."""
        # Simulate CUDA available in system
        mock_get_device.return_value = "GPU"
        mock_get_providers.return_value = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

        engine_cfg = DictConfig({
            "use_cuda": True,
            "use_dml": False,
            "use_cann": False,
            "cpu_ep_cfg": {"arena_extend_strategy": "kSameAsRequested"},
            "cuda_ep_cfg": {"device_id": 0},
        })

        provider_cfg = ProviderConfig(engine_cfg=engine_cfg)

        # Simulate session using CPU despite CUDA being available
        # This can happen if CUDA initialization fails
        provider_cfg.verify_providers(["CPUExecutionProvider"])

        # Should log warning about CUDA being available but not used
        assert mock_logger.warning.called
        warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
        assert any("CUDAExecutionProvider" in call for call in warning_calls)

    @patch("rapidocr_hkmc.inference_engine.onnxruntime.provider_config.get_device")
    @patch(
        "rapidocr_hkmc.inference_engine.onnxruntime.provider_config.get_available_providers"
    )
    def test_fallback_maintains_cpu_config(
        self, mock_get_providers, mock_get_device
    ):
        """Test that CPU configuration is maintained after GPU fallback."""
        # Simulate CUDA not available
        mock_get_device.return_value = "CPU"
        mock_get_providers.return_value = ["CPUExecutionProvider"]

        custom_cpu_config = {
            "arena_extend_strategy": "kNextPowerOfTwo",
            "enable_cpu_mem_arena": True,
        }

        engine_cfg = DictConfig({
            "use_cuda": True,  # Request CUDA
            "use_dml": False,
            "use_cann": False,
            "cpu_ep_cfg": custom_cpu_config,
            "cuda_ep_cfg": {"device_id": 0},
        })

        provider_cfg = ProviderConfig(engine_cfg=engine_cfg)
        ep_list = provider_cfg.get_ep_list()

        # Verify CPU configuration is preserved
        assert len(ep_list) == 1
        cpu_config = ep_list[0][1]
        assert cpu_config["arena_extend_strategy"] == "kNextPowerOfTwo"
        assert cpu_config["enable_cpu_mem_arena"] is True
