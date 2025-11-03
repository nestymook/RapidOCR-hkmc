# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict

import numpy as np
from omegaconf import DictConfig
from openvino.runtime import Core

from ..utils.download_file import DownloadFile, DownloadFileInput
from ..utils.log import logger
from .base import FileInfo, InferSession


class OpenVINOInferSession(InferSession):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        core = Core()

        model_path = cfg.get("model_path", None)
        if model_path is None:
            model_info = self.get_model_url(
                FileInfo(
                    engine_type=cfg.engine_type,
                    ocr_version=cfg.ocr_version,
                    task_type=cfg.task_type,
                    lang_type=cfg.lang_type,
                    model_type=cfg.model_type,
                )
            )
            model_path = self.DEFAULT_MODEL_PATH / Path(model_info["model_dir"]).name
            download_params = DownloadFileInput(
                file_url=model_info["model_dir"],
                sha256=model_info["SHA256"],
                save_path=model_path,
                logger=logger,
            )
            DownloadFile.run(download_params)

        logger.info(f"Using {model_path}")
        model_path = Path(model_path)
        self._verify_model(model_path)

        # Get target device from configuration
        device_name = self._get_target_device(cfg)
        
        # Check device availability and fallback if needed
        available_devices = core.available_devices
        logger.info(f"Available OpenVINO devices: {available_devices}")
        
        actual_device = device_name
        if not self._check_device_availability(core, device_name):
            logger.warning(
                f"Requested device '{device_name}' is not available. "
                f"Falling back to CPU."
            )
            actual_device = "CPU"
        
        logger.info(f"Requested device: {device_name}")
        logger.info(f"Using device: {actual_device}")

        # Initialize device-specific configuration
        config = self._init_config(cfg, actual_device)
        
        # Set properties for the target device
        if config:
            core.set_property(actual_device, config)

        # Read and compile model with timing
        model_load_start = time.time()
        
        try:
            model_onnx = core.read_model(model_path)
            model_read_time = time.time() - model_load_start
            logger.debug(f"Model read completed in {model_read_time:.3f} seconds")
        except Exception as e:
            error_msg = (
                f"Failed to read model from {model_path}: {str(e)}. "
                f"Please verify the model file exists and is a valid OpenVINO model."
            )
            logger.error(error_msg)
            raise OpenVIONError(error_msg) from e
        
        compile_start = time.time()
        try:
            compile_model = core.compile_model(
                model=model_onnx,
                device_name=actual_device,
                config=config if config else {}
            )
            compile_time = time.time() - compile_start
            logger.info(
                f"Model compiled successfully on {actual_device} device "
                f"in {compile_time:.3f} seconds"
            )
        except Exception as e:
            # Provide detailed diagnostic information
            error_msg = (
                f"Failed to compile model on {actual_device} device. "
                f"Error: {str(e)}. "
                f"Diagnostic information: "
                f"Device={actual_device}, "
                f"Available devices={core.available_devices}, "
                f"Model path={model_path}, "
                f"Config={config}"
            )
            logger.error(error_msg)
            
            # Attempt graceful degradation to CPU if not already on CPU
            if actual_device != "CPU":
                logger.warning(
                    f"Attempting to fall back to CPU device after "
                    f"{actual_device} compilation failure"
                )
                try:
                    fallback_start = time.time()
                    compile_model = core.compile_model(
                        model=model_onnx,
                        device_name="CPU",
                        config={}
                    )
                    fallback_time = time.time() - fallback_start
                    logger.info(
                        f"Successfully compiled model on CPU device as fallback "
                        f"in {fallback_time:.3f} seconds"
                    )
                    actual_device = "CPU"
                except Exception as cpu_error:
                    fallback_error_msg = (
                        f"Failed to compile model on both {actual_device} and CPU. "
                        f"Original error: {str(e)}. "
                        f"CPU fallback error: {str(cpu_error)}. "
                        f"Please check your OpenVINO installation and model compatibility."
                    )
                    logger.error(fallback_error_msg)
                    raise OpenVIONError(fallback_error_msg) from e
            else:
                raise OpenVIONError(error_msg) from e
        
        self.session = compile_model.create_infer_request()
        
        # Log execution provider verification
        total_load_time = time.time() - model_load_start
        logger.info(
            f"OpenVINO inference session initialized successfully. "
            f"Execution provider: {actual_device}"
        )
        logger.info(f"Total model loading time: {total_load_time:.3f} seconds")
        logger.debug(
            f"Hardware configuration - Device: {actual_device}, "
            f"Config: {config}, "
            f"Available devices: {core.available_devices}"
        )

    def _get_target_device(self, cfg: DictConfig) -> str:
        """Extract and validate device name from configuration.
        
        Args:
            cfg: Configuration dictionary containing engine settings.
            
        Returns:
            Validated device name string (CPU, NPU, or GPU).
            
        Raises:
            ConfigurationError: If device name is invalid.
        """
        engine_cfg = cfg.get("engine_cfg", {})
        device_name = engine_cfg.get("device_name", "CPU")
        
        # Validate device name
        valid_devices = ["CPU", "NPU", "GPU"]
        if device_name not in valid_devices:
            error_msg = (
                f"Invalid device_name configuration: '{device_name}'. "
                f"Allowed values are: {', '.join(valid_devices)}. "
                f"Please update your configuration file with a valid device name."
            )
            logger.error(error_msg)
            raise ConfigurationError(error_msg)
        
        return device_name
    
    def _check_device_availability(self, core: Core, device: str) -> bool:
        """Check if requested device is available in OpenVINO.
        
        Args:
            core: OpenVINO Core instance.
            device: Device name to check (CPU, NPU, GPU).
            
        Returns:
            True if device is available, False otherwise.
        """
        try:
            available_devices = core.available_devices
            return device in available_devices
        except Exception as e:
            logger.warning(
                f"Error checking device availability: {str(e)}"
            )
            return False
    
    def _init_config(
        self,
        cfg: DictConfig,
        device: str = "CPU"
    ) -> Dict[Any, Any]:
        """Initialize device-specific configuration.
        
        Args:
            cfg: Configuration dictionary containing engine settings.
            device: Target device name (CPU, NPU, GPU).
            
        Returns:
            Dictionary of OpenVINO configuration properties.
            
        Raises:
            ConfigurationError: If configuration validation fails.
        """
        config = {}
        engine_cfg = cfg.get("engine_cfg", {})

        # Thread configuration (primarily for CPU)
        infer_num_threads = engine_cfg.get("inference_num_threads", -1)
        if infer_num_threads != -1:
            if not (1 <= infer_num_threads <= os.cpu_count()):
                error_msg = (
                    f"Invalid inference_num_threads configuration: {infer_num_threads}. "
                    f"Must be between 1 and {os.cpu_count()} (available CPU cores), "
                    f"or -1 for automatic selection."
                )
                logger.error(error_msg)
                raise ConfigurationError(error_msg)
            config["INFERENCE_NUM_THREADS"] = str(infer_num_threads)

        # Performance hint configuration
        performance_hint = engine_cfg.get("performance_hint", None)
        if performance_hint is not None:
            valid_hints = [
                "LATENCY",
                "THROUGHPUT",
                "CUMULATIVE_THROUGHPUT"
            ]
            if performance_hint not in valid_hints:
                error_msg = (
                    f"Invalid performance_hint configuration: '{performance_hint}'. "
                    f"Allowed values are: {', '.join(valid_hints)}. "
                    f"Please update your configuration file with a valid performance hint."
                )
                logger.error(error_msg)
                raise ConfigurationError(error_msg)
            config["PERFORMANCE_HINT"] = str(performance_hint)

        performance_num_requests = engine_cfg.get("performance_num_requests", -1)
        if performance_num_requests != -1:
            config["PERFORMANCE_HINT_NUM_REQUESTS"] = str(performance_num_requests)

        # CPU-specific configurations
        if device == "CPU":
            enable_cpu_pinning = engine_cfg.get("enable_cpu_pinning", None)
            if enable_cpu_pinning is not None:
                config["ENABLE_CPU_PINNING"] = str(enable_cpu_pinning)

            num_streams = engine_cfg.get("num_streams", -1)
            if num_streams != -1:
                config["NUM_STREAMS"] = str(num_streams)

            enable_hyper_threading = engine_cfg.get("enable_hyper_threading", None)
            if enable_hyper_threading is not None:
                config["ENABLE_HYPER_THREADING"] = str(enable_hyper_threading)

            scheduling_core_type = engine_cfg.get("scheduling_core_type", None)
            if scheduling_core_type is not None:
                config["SCHEDULING_CORE_TYPE"] = str(scheduling_core_type)

        logger.info(f"Using OpenVINO config for {device}: {config}")
        logger.debug(f"Device-specific settings applied for {device}")
        return config

    def __call__(self, input_content: np.ndarray) -> np.ndarray:
        try:
            self.session.infer(inputs=[input_content])
            return self.session.get_output_tensor().data
        except Exception as e:
            error_info = traceback.format_exc()
            raise OpenVIONError(error_info) from e

    def have_key(self, key: str = "character") -> bool:
        return False


class OpenVIONError(Exception):
    """Base exception for OpenVINO inference errors."""
    pass


class ConfigurationError(OpenVIONError):
    """Exception raised for invalid configuration settings."""
    pass
