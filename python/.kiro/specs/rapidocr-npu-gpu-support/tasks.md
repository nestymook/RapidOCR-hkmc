# Implementation Plan

- [x] 1. Create rapidocr_hkmc package structure





  - Copy the entire rapidocr directory to rapidocr_hkmc to create the new package structure
  - Update setup.py to use MODULE_NAME = "rapidocr_hkmc" and configure package metadata
  - Verify package structure includes all necessary files (models, configs, code)
  - _Requirements: 6.1, 6.3, 6.5_

- [x] 2. Enhance OpenVINO engine for NPU support





  - [x] 2.1 Add device selection capability to OpenVINOInferSession

    - Implement _get_target_device() method to extract device_name from configuration
    - Add device name validation (CPU, NPU, GPU)
    - Update __init__ to accept and use device_name parameter
    - _Requirements: 1.1, 4.1_
  
  - [x] 2.2 Implement device availability checking

    - Create _check_device_availability() method using OpenVINO Core.available_devices
    - Add logic to query available devices before model compilation
    - Implement fallback to CPU when requested device is unavailable
    - _Requirements: 1.2, 7.1_
  
  - [x] 2.3 Update configuration initialization for device-specific settings

    - Modify _init_config() to accept device parameter
    - Add device-specific configuration options (performance hints for NPU)
    - Ensure backward compatibility when device_name is not specified
    - _Requirements: 1.4, 4.2, 4.3_
  
  - [x] 2.4 Add logging for device selection and fallback

    - Log requested device, available devices, and actual device being used
    - Add warning logs when falling back from NPU to CPU
    - Include device information in debug logs
    - _Requirements: 7.1, 7.4_
  
  - [x] 2.5 Update model compilation to use selected device

    - Modify core.compile_model() call to use device_name parameter
    - Apply device-specific configuration properties
    - Handle compilation errors with descriptive messages
    - _Requirements: 1.1, 4.5_

- [x] 3. Verify and document ONNXRuntime GPU configuration





  - [x] 3.1 Review existing CUDA execution provider implementation


    - Verify ProviderConfig correctly handles use_cuda flag
    - Confirm cuda_ep_cfg parameters are properly applied
    - Check device_id selection works for multi-GPU systems
    - _Requirements: 2.1, 2.2, 5.1_
  
  - [x] 3.2 Validate GPU fallback behavior


    - Test fallback to CPU when CUDA is not available
    - Verify warning messages are logged appropriately
    - Ensure CPU execution works correctly after fallback
    - _Requirements: 2.2, 7.2_
  
  - [x] 3.3 Document GPU configuration parameters


    - Add comments explaining cuda_ep_cfg options
    - Document device_id, arena_extend_strategy, and cudnn_conv_algo_search
    - Update config.yaml with clear GPU configuration examples
    - _Requirements: 2.5, 5.2, 5.3_


- [x] 4. Update configuration file with NPU and GPU settings




  - Add device_name parameter to openvino section in config.yaml
  - Add performance_hint options for NPU optimization
  - Set Cls engine_type to "openvino" for NPU usage
  - Set Det and Rec engine_type to "onnxruntime" with use_cuda: true
  - Add configuration comments explaining hardware acceleration options
  - _Requirements: 3.1, 3.2, 4.1, 4.2, 5.1_

- [x] 5. Update setup.py for rapidocr_hkmc package




  - Change MODULE_NAME from "rapidocr" to "rapidocr_hkmc"
  - Update package_dir to point to rapidocr_hkmc directory
  - Update entry_points to use rapidocr_hkmc module name
  - Update description to mention NPU/GPU support
  - Verify install_requires includes openvino and onnxruntime dependencies
  - _Requirements: 6.1, 6.2, 6.5_

- [x] 6. Implement error handling and validation





  - [x] 6.1 Add configuration validation in OpenVINOInferSession

    - Validate device_name is one of allowed values
    - Check performance_hint is valid if specified
    - Raise ConfigurationError for invalid settings with clear messages
    - _Requirements: 3.3, 4.5, 7.3_
  

  - [x] 6.2 Add error handling for device initialization failures

    - Catch and handle OpenVINO device compilation errors
    - Provide diagnostic information in error messages
    - Ensure graceful degradation to CPU on failures
    - _Requirements: 7.1, 7.3, 7.5_
  
  - [x] 6.3 Implement logging for execution provider verification


    - Log actual execution provider being used after initialization
    - Add timing information for model loading
    - Include hardware configuration in debug logs

   - _Requirements: 7.4_

- [x] 7. Test NPU support with OpenVINO




  - [x] 7.1 Create test for OpenVINO NPU device selection


    - Write test that initializes OpenVINOInferSession with device_name="NPU"
    - Verify device selection logic works correctly
    - Test with mock Core object to simulate NPU availability
    - _Requirements: 1.1, 4.1_
  
  - [x] 7.2 Create test for NPU fallback to CPU

    - Write test that requests NPU when unavailable
    - Verify warning is logged
    - Confirm CPU is used as fallback
    - _Requirements: 1.2, 7.1_
  
  - [x] 7.3 Create test for configuration validation

    - Test invalid device names raise appropriate errors
    - Test invalid performance hints are caught
    - Verify error messages are descriptive
    - _Requirements: 3.3, 4.5_


- [x] 8. Test GPU support with ONNXRuntime




  - [x] 8.1 Create test for CUDA execution provider configuration


    - Write test that initializes OrtInferSession with use_cuda=True
    - Verify CUDA EP is in provider list when available
    - Test cuda_ep_cfg parameters are applied correctly
    - _Requirements: 2.1, 5.2_
  
  - [x] 8.2 Create test for GPU fallback to CPU


    - Write test that requests CUDA when unavailable
    - Verify warning is logged
    - Confirm CPU EP is used as fallback
    - _Requirements: 2.2, 7.2_

- [-] 9. Integration testing with full OCR pipeline



  - [x] 9.1 Test cls model with OpenVINO NPU configuration


    - Initialize RapidOCR with cls using openvino engine
    - Run text classification on test images
    - Verify results match expected output
    - _Requirements: 1.1, 1.3_
  
  - [ ] 9.2 Test det and rec models with ONNXRuntime GPU configuration
    - Initialize RapidOCR with det/rec using onnxruntime with CUDA
    - Run full OCR pipeline on test images
    - Verify detection and recognition results are accurate
    - _Requirements: 2.1, 2.4_
  
  - [ ] 9.3 Test mixed configuration (OpenVINO cls + ONNXRuntime det/rec)
    - Configure cls with openvino, det/rec with onnxruntime
    - Run complete OCR workflow
    - Verify all models work together correctly
    - _Requirements: 3.1, 3.4_

- [-] 10. Build and test rapidocr_hkmc wheel package



  - [x] 10.1 Build wheel package


    - Run python setup.py bdist_wheel to create .whl file
    - Verify wheel file is named rapidocr_hkmc-*.whl
    - Check wheel contents include all necessary files
    - _Requirements: 6.1, 6.2_
  

  - [x] 10.2 Test wheel installation and import

    - Install wheel in clean virtual environment
    - Test import rapidocr_hkmc works correctly
    - Verify RapidOCR class is accessible
    - Test command-line entry point works
    - _Requirements: 6.3, 6.4_
  
  - [x] 10.3 Test API compatibility




    - Run existing rapidocr code examples with rapidocr_hkmc
    - Verify all public APIs work identically
    - Test backward compatibility with existing configurations
    - _Requirements: 6.4, 3.5_

- [x] 11. Documentation and validation





  - [x] 11.1 Update README with NPU/GPU configuration instructions

    - Document how to enable NPU for cls models
    - Document how to enable GPU for det/rec models
    - Provide example configurations
    - List hardware and driver requirements
    - _Requirements: 1.4, 2.5, 3.1_
  
  - [x] 11.2 Create configuration examples


    - Create example config for NPU-only setup
    - Create example config for GPU-only setup
    - Create example config for mixed NPU+GPU setup
    - Add comments explaining each setting
    - _Requirements: 3.1, 3.4_
  

  - [x] 11.3 Validate error messages and logging

    - Review all error messages for clarity
    - Ensure warning messages provide actionable guidance
    - Verify log levels are appropriate
    - Test error handling with various failure scenarios
    - _Requirements: 7.3, 7.4, 7.5_
