# -*- encoding: utf-8 -*-
# Test GPU fallback behavior for ONNXRuntime CUDA execution provider

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import DictConfig

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from rapidocr_hkmc.inference_engine.onnxruntime.provider_config import (
    EP,
    ProviderConfig,
)


class TestGPUFallback:
    """Test GPU fallback behavior when CUDA is not available."""

    def test_cuda_disabled_in_config(self):
        """Test that CUDA is not used when use_cuda is False."""
        engine_cfg = DictConfig({
            "use_cuda": False,
            "use_dml": False,
            "use_cann": False,
            "cpu_ep_cfg": {"arena_extend_strategy": "kSameAsRequested"},
            "cuda_ep_cfg": {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
            },
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
    def test_cuda_unavailable_fallback_to_cpu(
        self, mock_get_providers, mock_get_device
    ):
        """Test fallback to CPU when CUDA is requested but unavailable."""
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

    @patch("rapidocr_hkmc.inference_engine.onnxruntime.provider_config.get_device")
    @patch(
        "rapidocr_hkmc.inference_engine.onnxruntime.provider_config.get_available_providers"
    )
    def test_cuda_available_and_enabled(self, mock_get_providers, mock_get_device):
        """Test that CUDA is used when available and enabled."""
        # Simulate CUDA available
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
            "cuda_ep_cfg": {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
            },
        })

        provider_cfg = ProviderConfig(engine_cfg=engine_cfg)

        # is_cuda_available should return True
        assert provider_cfg.is_cuda_available() is True

        # Should have CUDA as first provider, CPU as fallback
        ep_list = provider_cfg.get_ep_list()
        assert len(ep_list) == 2
        assert ep_list[0][0] == EP.CUDA_EP.value
        assert ep_list[1][0] == EP.CPU_EP.value

    @patch("rapidocr_hkmc.inference_engine.onnxruntime.provider_config.get_device")
    @patch(
        "rapidocr_hkmc.inference_engine.onnxruntime.provider_config.get_available_providers"
    )
    def test_cuda_config_parameters_applied(
        self, mock_get_providers, mock_get_device
    ):
        """Test that CUDA configuration parameters are correctly applied."""
        mock_get_device.return_value = "GPU"
        mock_get_providers.return_value = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

        cuda_config = {
            "device_id": 1,  # Second GPU
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

        # Verify CUDA config is returned correctly
        cuda_ep_config = ep_list[0][1]
        assert cuda_ep_config["device_id"] == 1
        assert cuda_ep_config["arena_extend_strategy"] == "kNextPowerOfTwo"
        assert cuda_ep_config["cudnn_conv_algo_search"] == "EXHAUSTIVE"
        assert cuda_ep_config["do_copy_in_default_stream"] is True

    def test_verify_providers_with_cpu_fallback(self):
        """Test verify_providers logs warning when CUDA requested but CPU used."""
        engine_cfg = DictConfig({
            "use_cuda": True,
            "use_dml": False,
            "use_cann": False,
            "cpu_ep_cfg": {"arena_extend_strategy": "kSameAsRequested"},
            "cuda_ep_cfg": {"device_id": 0},
        })

        with patch(
            "rapidocr_hkmc.inference_engine.onnxruntime.provider_config.get_available_providers"
        ) as mock_providers:
            mock_providers.return_value = [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]

            provider_cfg = ProviderConfig(engine_cfg=engine_cfg)

            # Simulate session using CPU despite CUDA being available
            with patch(
                "rapidocr_hkmc.inference_engine.onnxruntime.provider_config.logger"
            ) as mock_logger:
                # Mock is_cuda_available to return True
                with patch.object(provider_cfg, "is_cuda_available", return_value=True):
                    provider_cfg.verify_providers(["CPUExecutionProvider"])

                    # Should log warning about CUDA being available but not used
                    assert mock_logger.warning.called
                    # Check that warning was called (may be multiple warnings)
                    assert any(
                        "CUDAExecutionProvider" in str(call)
                        for call in mock_logger.warning.call_args_list
                    )

    def test_cpu_execution_after_fallback(self):
        """Test that CPU execution works correctly after GPU fallback."""
        engine_cfg = DictConfig({
            "use_cuda": False,
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

        cpu_config = ep_list[0][1]
        assert "arena_extend_strategy" in cpu_config
        assert cpu_config["arena_extend_strategy"] == "kSameAsRequested"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
