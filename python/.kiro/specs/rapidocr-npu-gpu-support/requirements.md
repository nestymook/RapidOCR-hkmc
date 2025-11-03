# Requirements Document

## Introduction

This document specifies the requirements for updating the RapidOCR project to support hardware acceleration through NPU (Neural Processing Unit) for classification models and GPU for detection and recognition models. The update will enable the system to leverage OpenVINO for NPU acceleration on cls models and ONNXRuntime for GPU acceleration on det and rec models, while maintaining backward compatibility with existing configurations. Additionally, the project will be packaged as a custom wheel named "rapidocr_hkmc".

## Glossary

- **RapidOCR_System**: The optical character recognition system being modified
- **NPU**: Neural Processing Unit, specialized hardware for AI inference
- **GPU**: Graphics Processing Unit, used for parallel computing acceleration
- **OpenVINO**: Intel's toolkit for optimizing and deploying AI inference
- **ONNXRuntime**: Cross-platform inference engine for ONNX models
- **Cls_Model**: Text classification model that determines text orientation
- **Det_Model**: Text detection model that locates text regions in images
- **Rec_Model**: Text recognition model that converts text images to strings
- **Engine_Configuration**: Settings that control which inference engine and hardware to use
- **Wheel_Package**: Python distribution package format (.whl file)

## Requirements

### Requirement 1: NPU Support for Classification Models

**User Story:** As a developer deploying RapidOCR on Intel hardware with NPU capabilities, I want the classification model to automatically use NPU acceleration through OpenVINO, so that I can achieve faster text orientation detection with lower power consumption.

#### Acceptance Criteria

1. WHEN the Cls_Model is initialized with OpenVINO engine type, THE RapidOCR_System SHALL configure the inference session to target NPU device
2. WHEN NPU device is not available, THE RapidOCR_System SHALL log a warning message and fall back to CPU execution
3. WHEN the Cls_Model processes text images on NPU, THE RapidOCR_System SHALL maintain accuracy equivalent to CPU execution
4. WHERE OpenVINO engine is selected for Cls_Model, THE RapidOCR_System SHALL support NPU-specific configuration parameters including device selection and performance hints

### Requirement 2: GPU Support for Detection and Recognition Models

**User Story:** As a developer processing large volumes of documents, I want the detection and recognition models to use GPU acceleration through ONNXRuntime, so that I can achieve higher throughput for text detection and recognition tasks.

#### Acceptance Criteria

1. WHEN the Det_Model is initialized with ONNXRuntime engine type and GPU enabled, THE RapidOCR_System SHALL configure the CUDA execution provider
2. WHEN the Rec_Model is initialized with ONNXRuntime engine type and GPU enabled, THE RapidOCR_System SHALL configure the CUDA execution provider
3. WHEN GPU device is not available, THE RapidOCR_System SHALL log a warning message and fall back to CPU execution
4. WHEN Det_Model or Rec_Model processes images on GPU, THE RapidOCR_System SHALL maintain accuracy equivalent to CPU execution
5. WHERE ONNXRuntime engine is selected with GPU, THE RapidOCR_System SHALL support CUDA-specific configuration parameters including device ID and memory management settings

### Requirement 3: Unified Configuration System

**User Story:** As a system administrator, I want a single configuration file that supports both OpenVINO and ONNXRuntime settings for different models, so that I can easily manage hardware acceleration settings across all OCR components.

#### Acceptance Criteria

1. THE RapidOCR_System SHALL support a configuration file that includes both OpenVINO and ONNXRuntime engine configurations
2. WHEN a model is initialized, THE RapidOCR_System SHALL select the appropriate engine configuration based on the model's engine_type setting
3. THE RapidOCR_System SHALL validate configuration parameters at initialization time and report invalid settings with clear error messages
4. WHERE different models use different engines, THE RapidOCR_System SHALL allow independent configuration of each model's engine type and hardware target
5. THE RapidOCR_System SHALL maintain backward compatibility with existing configuration files that do not specify NPU or GPU settings

### Requirement 4: OpenVINO NPU Configuration

**User Story:** As a performance engineer, I want to configure OpenVINO-specific NPU settings such as device selection and performance hints, so that I can optimize the classification model for my specific hardware and use case.

#### Acceptance Criteria

1. THE RapidOCR_System SHALL support configuration of NPU device selection through OpenVINO device_name parameter
2. THE RapidOCR_System SHALL support OpenVINO performance hints including LATENCY, THROUGHPUT, and CUMULATIVE_THROUGHPUT
3. THE RapidOCR_System SHALL support configuration of inference precision modes for NPU execution
4. WHERE NPU-specific settings are provided, THE RapidOCR_System SHALL apply these settings only to models using OpenVINO engine
5. WHEN invalid NPU configuration is detected, THE RapidOCR_System SHALL raise a configuration error with descriptive message

### Requirement 5: ONNXRuntime GPU Configuration

**User Story:** As a performance engineer, I want to configure ONNXRuntime-specific GPU settings such as device ID and memory allocation strategy, so that I can optimize detection and recognition models for my GPU hardware.

#### Acceptance Criteria

1. THE RapidOCR_System SHALL support configuration of GPU device ID for multi-GPU systems
2. THE RapidOCR_System SHALL support CUDA execution provider configuration including arena_extend_strategy and cudnn_conv_algo_search
3. THE RapidOCR_System SHALL support configuration of GPU memory limits and allocation strategies
4. WHERE GPU-specific settings are provided, THE RapidOCR_System SHALL apply these settings only to models using ONNXRuntime engine with CUDA enabled
5. WHEN invalid GPU configuration is detected, THE RapidOCR_System SHALL raise a configuration error with descriptive message

### Requirement 6: Custom Wheel Package

**User Story:** As a deployment engineer, I want to build and distribute a custom wheel package named "rapidocr_hkmc", so that I can easily install the modified RapidOCR system with NPU and GPU support across multiple environments.

#### Acceptance Criteria

1. THE RapidOCR_System SHALL provide a setup.py configuration that builds a wheel package named "rapidocr_hkmc"
2. WHEN the wheel package is built, THE RapidOCR_System SHALL include all necessary model files, configuration files, and dependencies
3. WHEN the wheel package is installed, THE RapidOCR_System SHALL be importable as "rapidocr_hkmc" module
4. THE RapidOCR_System SHALL maintain the same API interface as the original rapidocr package for backward compatibility
5. THE RapidOCR_System SHALL include metadata in the wheel package specifying version, author, and hardware acceleration capabilities

### Requirement 7: Error Handling and Fallback

**User Story:** As a developer integrating RapidOCR, I want the system to gracefully handle hardware unavailability and provide clear error messages, so that my application can run reliably across different deployment environments.

#### Acceptance Criteria

1. WHEN NPU is configured but not available, THE RapidOCR_System SHALL log a warning and fall back to CPU execution for Cls_Model
2. WHEN GPU is configured but not available, THE RapidOCR_System SHALL log a warning and fall back to CPU execution for Det_Model and Rec_Model
3. WHEN an inference engine fails to initialize, THE RapidOCR_System SHALL raise an exception with diagnostic information including engine type, model type, and error details
4. THE RapidOCR_System SHALL log the actual execution provider being used for each model at initialization time
5. WHERE hardware acceleration is unavailable, THE RapidOCR_System SHALL continue to function with CPU execution without requiring configuration changes


