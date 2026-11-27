cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # For multi-architecture releases, pass -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90;120".
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(fusionrt VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(FUSIONRT_BUILD_TESTS "Build C++ test suite" ON)
option(FUSIONRT_ENABLE_CUDA "Enable CUDA backend" ON)
option(FUSIONRT_ENABLE_VULKAN "Enable Vulkan backend" ON)

if(FUSIONRT_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 20)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        find_package(CUDAToolkit REQUIRED)
    else()
        message(STATUS "CUDA compiler not found; building without CUDA backend")
        set(FUSIONRT_ENABLE_CUDA OFF)
    endif()
endif()

if(FUSIONRT_ENABLE_VULKAN)
    find_package(Vulkan QUIET)
    find_program(GLSLC_EXECUTABLE NAMES glslc HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
    if(NOT Vulkan_FOUND OR NOT GLSLC_EXECUTABLE)
        message(STATUS "Vulkan SDK or glslc not found; building without Vulkan backend")
        set(FUSIONRT_ENABLE_VULKAN OFF)
    endif()
endif()

if(MSVC)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
        $<$<COMPILE_LANGUAGE:CXX>:/permissive->
        $<$<COMPILE_LANGUAGE:CXX>:/EHsc>
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CXX>:/O2>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W3,/EHsc,/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    )
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
else()
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    )
endif()

set(FUSIONRT_SOURCES
    src/server/agent_request.cpp
    src/server/agent_template.cpp
    src/server/agent_response.cpp
    src/server/http_request.cpp
    src/core/tensor.cpp
    src/core/cpu_moe.cpp
    src/core/json.cpp
    src/core/sampler.cpp
    src/core/device_profile.cpp
    src/core/model_trace.cpp
    src/core/graph_ir.cpp
    src/io/gguf.cpp
    src/io/image.cpp
    src/models/vision.cpp
    src/io/safetensors.cpp
    src/io/tokenizer.cpp
    src/models/model_loader.cpp
    src/models/model_graph.cpp
    src/models/cuda_model.cpp
    src/models/vulkan_model.cpp
    src/bench/bench_runner.cpp
)

if((MSVC AND CMAKE_CXX_COMPILER_ARCHITECTURE_ID STREQUAL "x64") OR
   (NOT MSVC AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x86_64)$"))
    list(APPEND FUSIONRT_SOURCES src/core/cpu_moe_avx2.cpp src/core/cpu_moe_avx512.cpp)
    set_source_files_properties(src/core/cpu_moe.cpp PROPERTIES COMPILE_DEFINITIONS "FUSIONRT_CPU_MOE_AVX2;FUSIONRT_CPU_MOE_AVX512")
    if(MSVC)
        set_source_files_properties(src/core/cpu_moe_avx2.cpp PROPERTIES COMPILE_OPTIONS /arch:AVX2)
        set_source_files_properties(src/core/cpu_moe_avx512.cpp PROPERTIES COMPILE_OPTIONS /arch:AVX512)
    else()
        set_source_files_properties(src/core/cpu_moe_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
        set_source_files_properties(src/core/cpu_moe_avx512.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512vnni")
    endif()
endif()

if(FUSIONRT_ENABLE_CUDA)
    list(APPEND FUSIONRT_SOURCES
        src/models/cuda_dispatch.cpp
        src/models/cuda_loader.cpp
        src/models/cuda_session.cpp
        src/models/cuda_ir.cpp
        src/models/cuda_executor.cpp
        src/models/cuda_generate.cpp
        src/cuda/cuda_backend.cpp
        src/cuda/cuda_common.cuh
        src/cuda/matmul/weight_tiles.cuh
        src/cuda/matmul/gemv_decode.cu
        src/cuda/matmul/mmq_prefill.cu
        src/cuda/sequence/flash_attention.cu
        src/cuda/sequence/mla.cu
        src/cuda/sequence/linear_recurrence.cu
        src/cuda/epilogue/elementwise_ops.cu
        src/cuda/epilogue/hadamard_ops.cu
        src/cuda/epilogue/codec_moe_ops.cu
        src/server/http_server.cpp
    )
endif()

if(FUSIONRT_ENABLE_VULKAN)
    set(SHADER_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
    set(SHADER_BIN_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
    file(MAKE_DIRECTORY "${SHADER_BIN_DIR}")

    file(GLOB SHADER_SOURCES CONFIGURE_DEPENDS "${SHADER_SRC_DIR}/*.comp")
    set(SHADER_SPV_OUTPUTS "")

    foreach(SHADER_SRC ${SHADER_SOURCES})
        get_filename_component(SHADER_NAME "${SHADER_SRC}" NAME)
        set(SHADER_SPV "${SHADER_BIN_DIR}/${SHADER_NAME}.spv")
        add_custom_command(
            OUTPUT "${SHADER_SPV}"
            COMMAND "${GLSLC_EXECUTABLE}" --target-env=vulkan1.3 -O "${SHADER_SRC}" -o "${SHADER_SPV}"
            DEPENDS "${SHADER_SRC}"
            COMMENT "Compiling Vulkan shader ${SHADER_NAME}"
            VERBATIM
        )
        list(APPEND SHADER_SPV_OUTPUTS "${SHADER_SPV}")
    endforeach()

    add_custom_target(fusionrt_shaders ALL DEPENDS ${SHADER_SPV_OUTPUTS})
    list(APPEND FUSIONRT_SOURCES
        src/vulkan/vulkan_backend.cpp
        src/vulkan/vulkan_ops.cpp
    )
endif()

add_library(fusionrt_core STATIC ${FUSIONRT_SOURCES})

target_include_directories(fusionrt_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_compile_definitions(fusionrt_core PUBLIC
    FUSIONRT_VERSION="0.2.0"
)

if(FUSIONRT_ENABLE_CUDA)
    target_compile_definitions(fusionrt_core PUBLIC FUSIONRT_ENABLE_CUDA=1)
    target_link_libraries(fusionrt_core PUBLIC
        CUDA::cudart
        CUDA::cuda_driver
        CUDA::cublas
    )
    if(WIN32)
        target_link_libraries(fusionrt_core PRIVATE ws2_32 dxgi)
    endif()
    if(TARGET CUDA::nvml)
        target_link_libraries(fusionrt_core PUBLIC CUDA::nvml)
        target_compile_definitions(fusionrt_core PUBLIC FUSIONRT_HAS_NVML=1)
    endif()
endif()

if(FUSIONRT_ENABLE_VULKAN)
    add_dependencies(fusionrt_core fusionrt_shaders)
    target_compile_definitions(fusionrt_core PUBLIC FUSIONRT_ENABLE_VULKAN=1)
    target_link_libraries(fusionrt_core PUBLIC Vulkan::Vulkan)
endif()

add_executable(fusionrt src/cli/main.cpp)
target_link_libraries(fusionrt PRIVATE fusionrt_core)

if(FUSIONRT_ENABLE_VULKAN)
    add_custom_command(TARGET fusionrt POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${SHADER_BIN_DIR}" "$<TARGET_FILE_DIR:fusionrt>/shaders"
    )
endif()

if(FUSIONRT_BUILD_TESTS)
    enable_testing()
    add_executable(fusionrt_tests
        tests/test_main.cpp
        tests/test_quant_codecs.cpp
        tests/test_kernels_cuda.cpp
        tests/test_models_moe.cpp
        tests/test_model_state.cpp
        tests/test_spec_and_protocols.cpp
    )
    target_include_directories(fusionrt_tests PRIVATE src/server)
    target_link_libraries(fusionrt_tests PRIVATE fusionrt_core)
    if(FUSIONRT_ENABLE_VULKAN)
        add_custom_command(TARGET fusionrt_tests POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory "${SHADER_BIN_DIR}" "$<TARGET_FILE_DIR:fusionrt_tests>/shaders"
        )
    endif()
    add_test(NAME fusionrt_cpp_suite COMMAND fusionrt_tests)
endif()
