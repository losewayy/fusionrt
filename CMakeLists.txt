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
option(FUSIONRT_BUILD_CLI "Build command-line executable" ON)
option(FUSIONRT_BUILD_OPS_ONLY "Build only the standalone operator library" OFF)
option(FUSIONRT_ENABLE_CUDA "Enable CUDA backend" ON)
option(FUSIONRT_ENABLE_VULKAN "Enable Vulkan backend" ON)

if(FUSIONRT_ENABLE_CUDA)
    foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
        if(arch MATCHES "^([0-9]+)(-real|-virtual)?$")
            if(CMAKE_MATCH_1 LESS 80)
                message(FATAL_ERROR "FusionRT CUDA MMQ kernels require sm_80+ (requested architecture: ${arch})")
            endif()
        endif()
    endforeach()
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 20)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        find_package(CUDAToolkit REQUIRED)
        set(FUSIONRT_HAS_NVFP4 0)
        if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8" AND TARGET CUDA::cublasLt)
            include(CheckSourceCompiles)
            include(CMakePushCheckState)
            cmake_push_check_state(RESET)
            set(CMAKE_REQUIRED_LIBRARIES CUDA::cublasLt)
            set(CMAKE_REQUIRED_INCLUDES ${CUDAToolkit_INCLUDE_DIRS})
            check_source_compiles(CUDA [[
                #include <cuda_fp4.h>
                #include <cublasLt.h>
                int main() {
                    static_cast<void>(CUDA_R_4F_E2M1);
                    static_cast<void>(CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3);
                    return 0;
                }
            ]] FUSIONRT_CUDA_HAS_FP4_HEADERS)
            cmake_pop_check_state()
            if(FUSIONRT_CUDA_HAS_FP4_HEADERS)
                set(FUSIONRT_HAS_NVFP4 1)
            endif()
        endif()
    else()
        message(STATUS "CUDA compiler not found; building without CUDA backend")
        set(FUSIONRT_ENABLE_CUDA OFF)
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

add_library(fusionrt_ops STATIC
    src/core/tensor.cpp
    src/core/repack.cpp
    src/core/cpu_moe.cpp
    src/core/json.cpp
    src/core/device_profile.cpp
)
set_target_properties(fusionrt_ops PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(fusionrt_ops PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(fusionrt_ops PUBLIC cxx_std_20)
find_package(Threads REQUIRED)
target_link_libraries(fusionrt_ops PRIVATE Threads::Threads)

if((MSVC AND CMAKE_CXX_COMPILER_ARCHITECTURE_ID STREQUAL "x64") OR
   (NOT MSVC AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x86_64)$"))
    target_sources(fusionrt_ops PRIVATE src/core/cpu_moe_avx2.cpp src/core/cpu_moe_avx512.cpp)
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
    target_sources(fusionrt_ops PRIVATE
        src/cuda/cuda_backend.cpp
        src/cuda/cuda_common.cuh
        src/cuda/matmul/weight_tiles.cuh
        src/cuda/matmul/gemv_decode.cu
        src/cuda/matmul/mmq_prefill.cu
        src/cuda/matmul/dit_linear_ops.cu
        src/cuda/matmul/nvfp4_gemm.cu
        src/cuda/sequence/flash_attention.cu
        src/cuda/sequence/mla.cu
        src/cuda/sequence/linear_recurrence.cu
        src/cuda/epilogue/elementwise_ops.cu
        src/cuda/epilogue/hadamard_ops.cu
        src/cuda/epilogue/codec_moe_ops.cu
    )
    target_compile_definitions(fusionrt_ops PUBLIC
        FUSIONRT_ENABLE_CUDA=1 FUSIONRT_HAS_NVFP4=${FUSIONRT_HAS_NVFP4})
    target_link_libraries(fusionrt_ops PUBLIC CUDA::cudart CUDA::cuda_driver CUDA::cublas)
    if(FUSIONRT_HAS_NVFP4)
        target_link_libraries(fusionrt_ops PUBLIC CUDA::cublasLt)
    endif()
    if(TARGET CUDA::nvml)
        target_link_libraries(fusionrt_ops PUBLIC CUDA::nvml)
        target_compile_definitions(fusionrt_ops PUBLIC FUSIONRT_HAS_NVML=1)
    endif()
endif()

if(FUSIONRT_BUILD_OPS_ONLY)
    return()
endif()

set(FUSIONRT_SOURCES
    src/server/agent_request.cpp
    src/server/agent_template.cpp
    src/server/agent_response.cpp
    src/server/http_request.cpp
    src/core/sampler.cpp
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

if(FUSIONRT_ENABLE_CUDA)
    list(APPEND FUSIONRT_SOURCES
        src/models/cuda_dispatch.cpp
        src/models/cuda_loader.cpp
        src/models/cuda_session.cpp
        src/models/cuda_ir.cpp
        src/models/cuda_executor.cpp
        src/models/cuda_generate.cpp
    )
endif()

if(FUSIONRT_ENABLE_VULKAN)
    find_package(Vulkan QUIET)
    find_program(GLSLC_EXECUTABLE NAMES glslc HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
    if(NOT Vulkan_FOUND OR NOT GLSLC_EXECUTABLE)
        message(STATUS "Vulkan SDK or glslc not found; building without Vulkan backend")
        set(FUSIONRT_ENABLE_VULKAN OFF)
    endif()
endif()

if(FUSIONRT_ENABLE_VULKAN)
    set(SHADER_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
    set(SHADER_BIN_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
    file(MAKE_DIRECTORY "${SHADER_BIN_DIR}")

    # Share the CPU/CUDA codebooks without requiring shaderInt64 on Vulkan.
    set(SHADER_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/shader_includes")
    set(WEIGHT_CODEBOOKS "${CMAKE_CURRENT_SOURCE_DIR}/include/fusionrt/detail/weight_codebooks.inc")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${WEIGHT_CODEBOOKS}")
    file(READ "${WEIGHT_CODEBOOKS}" WEIGHT_CODEBOOK_GLSL)
    string(REPEAT "[0-9a-fA-F]" 8 CODEBOOK_WORD)
    string(REGEX REPLACE "0x(${CODEBOOK_WORD})(${CODEBOOK_WORD})," "uvec2(0x\\2u, 0x\\1u),"
        WEIGHT_CODEBOOK_GLSL "${WEIGHT_CODEBOOK_GLSL}")
    string(REPLACE "uint64_t" "uvec2" WEIGHT_CODEBOOK_GLSL "${WEIGHT_CODEBOOK_GLSL}")
    string(REPLACE "uint32_t" "uint" WEIGHT_CODEBOOK_GLSL "${WEIGHT_CODEBOOK_GLSL}")
    string(REPLACE "uint8_t" "uint" WEIGHT_CODEBOOK_GLSL "${WEIGHT_CODEBOOK_GLSL}")
    string(REPLACE "int8_t" "int" WEIGHT_CODEBOOK_GLSL "${WEIGHT_CODEBOOK_GLSL}")
    file(CONFIGURE OUTPUT "${SHADER_INCLUDE_DIR}/weight_codebooks.glsl" CONTENT
        "#define FUSIONRT_CODEBOOK(type, name, count) const type name[count] = {\n${WEIGHT_CODEBOOK_GLSL}\n#undef FUSIONRT_CODEBOOK\n" @ONLY)

    file(GLOB SHADER_SOURCES CONFIGURE_DEPENDS
        "${SHADER_SRC_DIR}/*.comp"
        "${SHADER_SRC_DIR}/matmul/*.comp"
        "${SHADER_SRC_DIR}/sequence/*.comp"
        "${SHADER_SRC_DIR}/epilogue/*.comp"
    )
    set(SHADER_SPV_OUTPUTS "")

    foreach(SHADER_SRC ${SHADER_SOURCES})
        file(RELATIVE_PATH SHADER_NAME "${SHADER_SRC_DIR}" "${SHADER_SRC}")
        if(SHADER_NAME MATCHES "/")
            # The orthogonal kernels are loaded as matmul/name.spv or sequence/name.spv.
            string(REGEX REPLACE "[.]comp$" "" SHADER_STEM "${SHADER_NAME}")
        else()
            set(SHADER_STEM "${SHADER_NAME}")
        endif()
        set(SHADER_SPV "${SHADER_BIN_DIR}/${SHADER_STEM}.spv")
        set(SHADER_DEPFILE "${SHADER_INCLUDE_DIR}/${SHADER_STEM}.d")
        get_filename_component(SHADER_OUTPUT_DIR "${SHADER_SPV}" DIRECTORY)
        get_filename_component(SHADER_DEPFILE_DIR "${SHADER_DEPFILE}" DIRECTORY)
        add_custom_command(
            OUTPUT "${SHADER_SPV}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${SHADER_OUTPUT_DIR}" "${SHADER_DEPFILE_DIR}"
            COMMAND "${GLSLC_EXECUTABLE}" --target-env=vulkan1.3 -O
                -I "${SHADER_INCLUDE_DIR}" -MD -MF "${SHADER_DEPFILE}" "${SHADER_SRC}" -o "${SHADER_SPV}"
            DEPENDS "${SHADER_SRC}"
            DEPFILE "${SHADER_DEPFILE}"
            COMMENT "Compiling Vulkan shader ${SHADER_NAME}"
            VERBATIM
        )
        list(APPEND SHADER_SPV_OUTPUTS "${SHADER_SPV}")
    endforeach()

    add_custom_target(fusionrt_shaders ALL DEPENDS ${SHADER_SPV_OUTPUTS})
    list(APPEND FUSIONRT_SOURCES
        src/models/vulkan_weights.cpp
        src/models/vulkan_session.cpp
        src/models/vulkan_executor.cpp
        src/models/vulkan_dispatch.cpp
        src/models/vulkan_vision.cpp
        src/vulkan/vulkan_backend.cpp
        src/vulkan/vulkan_ops.cpp
    )
endif()

if(FUSIONRT_ENABLE_CUDA OR FUSIONRT_ENABLE_VULKAN)
    list(APPEND FUSIONRT_SOURCES src/server/http_server.cpp)
endif()

add_library(fusionrt_core STATIC ${FUSIONRT_SOURCES})
target_link_libraries(fusionrt_core PUBLIC fusionrt_ops)

target_compile_definitions(fusionrt_core PUBLIC
    FUSIONRT_VERSION="0.2.0"
)

if(FUSIONRT_ENABLE_CUDA AND WIN32)
    target_link_libraries(fusionrt_core PRIVATE dxgi)
endif()

if(WIN32 AND (FUSIONRT_ENABLE_CUDA OR FUSIONRT_ENABLE_VULKAN))
    target_link_libraries(fusionrt_core PRIVATE ws2_32)
endif()

if(FUSIONRT_ENABLE_VULKAN)
    add_dependencies(fusionrt_core fusionrt_shaders)
    target_compile_definitions(fusionrt_core PUBLIC FUSIONRT_ENABLE_VULKAN=1)
    target_link_libraries(fusionrt_core PUBLIC Vulkan::Vulkan)
endif()

if(FUSIONRT_BUILD_CLI)
    add_executable(fusionrt src/cli/main.cpp)
    target_link_libraries(fusionrt PRIVATE fusionrt_core)
    if(FUSIONRT_ENABLE_VULKAN)
        add_custom_command(TARGET fusionrt POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory "${SHADER_BIN_DIR}" "$<TARGET_FILE_DIR:fusionrt>/shaders"
        )
    endif()
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
        tests/test_wave1.cpp
        tests/test_dit_ops.cpp
        tests/test_nvfp4_ops.cpp
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
