# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import platform
from enum import Enum
from typing import Any, Dict, List, Sequence, Tuple

from onnxruntime import get_available_providers, get_device

from ...utils.log import logger


class EP(Enum):
    CPU_EP = "CPUExecutionProvider"
    CUDA_EP = "CUDAExecutionProvider"
    DIRECTML_EP = "DmlExecutionProvider"
    CANN_EP = "CANNExecutionProvider"


class ProviderConfig:
    """Configuration manager for ONNXRuntime execution providers.
    
    This class handles the selection and configuration of execution providers
    for ONNXRuntime inference sessions. It supports multiple hardware acceleration
    options including CUDA (GPU), DirectML (Windows GPU), and CANN (Huawei NPU).
    
    The class automatically detects available providers and configures them based
    on user settings, with graceful fallback to CPU when requested hardware is
    unavailable.
    """
    
    def __init__(self, engine_cfg: Dict[str, Any]):
        """Initialize provider configuration.
        
        Args:
            engine_cfg: Engine configuration dictionary containing provider settings.
                Expected keys:
                - use_cuda (bool): Enable CUDA execution provider for GPU
                - use_dml (bool): Enable DirectML execution provider (Windows)
                - use_cann (bool): Enable CANN execution provider (Huawei NPU)
                - cuda_ep_cfg (dict): CUDA-specific configuration parameters
                - cpu_ep_cfg (dict): CPU-specific configuration parameters
        """
        # Query available execution providers from ONNXRuntime
        self.had_providers: List[str] = get_available_providers()
        self.default_provider = self.had_providers[0]

        # Extract user-requested provider flags from configuration
        self.cfg_use_cuda = engine_cfg.get("use_cuda", False)
        self.cfg_use_dml = engine_cfg.get("use_dml", False)
        self.cfg_use_cann = engine_cfg.get("use_cann", False)

        # Store full configuration for provider-specific settings
        self.cfg = engine_cfg

    def get_ep_list(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Build prioritized list of execution providers with configurations.
        
        Constructs a list of (provider_name, provider_config) tuples in priority order.
        ONNXRuntime will attempt to use providers in the order they appear in the list,
        falling back to the next provider if the current one is unavailable.
        
        Priority order (highest to lowest):
        1. CANN (if available and enabled)
        2. DirectML (if available and enabled)
        3. CUDA (if available and enabled)
        4. CPU (always available, fallback)
        
        Returns:
            List of (provider_name, config_dict) tuples for InferenceSession.
        """
        # CPU is always available as fallback
        results = [(EP.CPU_EP.value, self.cpu_ep_cfg())]

        # Add CUDA provider if GPU is available and enabled
        if self.is_cuda_available():
            results.insert(0, (EP.CUDA_EP.value, self.cuda_ep_cfg()))

        # Add DirectML provider if on Windows 10+ and enabled
        if self.is_dml_available():
            logger.info(
                "Windows 10 or above detected, try to use DirectML as primary provider"
            )
            results.insert(0, (EP.DIRECTML_EP.value, self.dml_ep_cfg()))

        # Add CANN provider if Huawei NPU is available and enabled
        if self.is_cann_available():
            logger.info("Try to use CANNExecutionProvider to infer")
            results.insert(0, (EP.CANN_EP.value, self.cann_ep_cfg()))

        return results

    def cpu_ep_cfg(self) -> Dict[str, Any]:
        return dict(self.cfg.cpu_ep_cfg)

    def cuda_ep_cfg(self) -> Dict[str, Any]:
        """Get CUDA execution provider configuration.
        
        Returns CUDA-specific configuration parameters for GPU acceleration.
        Configuration is read from the cuda_ep_cfg section of the engine config.
        
        Supported parameters:
        - device_id (int): GPU device ID for multi-GPU systems (default: 0)
        - arena_extend_strategy (str): Memory allocation strategy
            * "kNextPowerOfTwo": Allocate memory in powers of 2 (recommended)
            * "kSameAsRequested": Allocate exact requested size
        - cudnn_conv_algo_search (str): cuDNN convolution algorithm selection
            * "EXHAUSTIVE": Search all algorithms, best performance (slower startup)
            * "HEURISTIC": Use heuristics, faster startup
            * "DEFAULT": Use default algorithm
        - do_copy_in_default_stream (bool): Use default CUDA stream for copies
        
        Returns:
            Dictionary of CUDA execution provider configuration parameters.
        """
        return dict(self.cfg.cuda_ep_cfg)

    def dml_ep_cfg(self) -> Dict[str, Any]:
        if self.cfg.dm_ep_cfg is not None:
            return self.cfg.dm_ep_cfg

        if self.is_cuda_available():
            return self.cuda_ep_cfg()
        return self.cpu_ep_cfg()

    def cann_ep_cfg(self) -> Dict[str, Any]:
        return dict(self.cfg.cann_ep_cfg)

    def verify_providers(self, session_providers: Sequence[str]):
        """Verify that the session is using the expected execution provider.
        
        After InferenceSession creation, this method checks if the actual provider
        being used matches the requested configuration. Logs warnings if there's
        a mismatch between requested and actual providers.
        
        This is important for GPU fallback scenarios where CUDA may be requested
        but CPU is actually used due to hardware unavailability.
        
        Args:
            session_providers: List of providers actually used by the session,
                obtained from InferenceSession.get_providers().
        
        Raises:
            ValueError: If session_providers list is empty.
        """
        if not session_providers:
            raise ValueError("Session Providers is empty")

        first_provider = session_providers[0]

        providers_to_check = {
            EP.CUDA_EP: self.is_cuda_available,
            EP.DIRECTML_EP: self.is_dml_available,
            EP.CANN_EP: self.is_cann_available,
        }

        for ep, check_func in providers_to_check.items():
            if check_func() and first_provider != ep.value:
                logger.warning(
                    f"{ep.value} is available, but the inference part is automatically shifted to be executed under {first_provider}. "
                )
                logger.warning(f"The available lists are {session_providers}")

    def is_cuda_available(self) -> bool:
        """Check if CUDA execution provider is available and enabled.
        
        Verifies three conditions for CUDA availability:
        1. User has enabled use_cuda in configuration
        2. GPU device is detected by ONNXRuntime
        3. CUDAExecutionProvider is in available providers list
        
        If CUDA is requested but unavailable, logs warning with installation
        instructions and returns False, allowing fallback to CPU.
        
        Multi-GPU Support:
        - Device selection is controlled via device_id in cuda_ep_cfg
        - Default device_id is 0 (first GPU)
        - For multi-GPU systems, set device_id to target specific GPU
        
        Returns:
            True if CUDA is available and enabled, False otherwise.
        """
        if not self.cfg_use_cuda:
            return False

        CUDA_EP = EP.CUDA_EP.value
        if get_device() == "GPU" and CUDA_EP in self.had_providers:
            return True

        logger.warning(
            f"{CUDA_EP} is not in available providers ({self.had_providers}). Use {self.default_provider} inference by default."
        )
        install_instructions = [
            f"If you want to use {CUDA_EP} acceleration, you must do:"
            "(For reference only) If you want to use GPU acceleration, you must do:",
            "First, uninstall all onnxruntime packages in current environment.",
            "Second, install onnxruntime-gpu by `pip install onnxruntime-gpu`.",
            "Note the onnxruntime-gpu version must match your cuda and cudnn version.",
            "You can refer this link: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html",
            f"Third, ensure {CUDA_EP} is in available providers list. e.g. ['CUDAExecutionProvider', 'CPUExecutionProvider']",
        ]
        self.print_log(install_instructions)
        return False

    def is_dml_available(self) -> bool:
        if not self.cfg_use_dml:
            return False

        cur_os = platform.system()
        if cur_os != "Windows":
            logger.warning(
                f"DirectML is only supported in Windows OS. The current OS is {cur_os}. Use {self.default_provider} inference by default.",
            )
            return False

        window_build_number_str = platform.version().split(".")[-1]
        window_build_number = (
            int(window_build_number_str) if window_build_number_str.isdigit() else 0
        )
        if window_build_number < 18362:
            logger.warning(
                f"DirectML is only supported in Windows 10 Build 18362 and above OS. The current Windows Build is {window_build_number}. Use {self.default_provider} inference by default.",
            )
            return False

        DML_EP = EP.DIRECTML_EP.value
        if DML_EP in self.had_providers:
            return True

        logger.warning(
            f"{DML_EP} is not in available providers ({self.had_providers}). Use {self.default_provider} inference by default."
        )
        install_instructions = [
            "If you want to use DirectML acceleration, you must do:",
            "First, uninstall all onnxruntime packages in current environment.",
            "Second, install onnxruntime-directml by `pip install onnxruntime-directml`",
            f"Third, ensure {DML_EP} is in available providers list. e.g. ['DmlExecutionProvider', 'CPUExecutionProvider']",
        ]
        self.print_log(install_instructions)
        return False

    def is_cann_available(self) -> bool:
        if not self.cfg_use_cann:
            return False

        CANN_EP = EP.CANN_EP.value
        if CANN_EP in self.had_providers:
            return True

        logger.warning(
            f"{CANN_EP} is not in available providers ({self.had_providers}). Use {self.default_provider} inference by default."
        )
        install_instructions = [
            "If you want to use CANN acceleration, you must do:",
            "First, ensure you have installed Huawei Ascend software stack.",
            "Second, install onnxruntime with CANN support by following the instructions at:",
            "\thttps://onnxruntime.ai/docs/execution-providers/community-maintained/CANN-ExecutionProvider.html",
            f"Third, ensure {CANN_EP} is in available providers list. e.g. ['CANNExecutionProvider', 'CPUExecutionProvider']",
        ]
        self.print_log(install_instructions)
        return False

    def print_log(self, log_list: List[str]):
        for log_info in log_list:
            logger.info(log_info)
