cmake_minimum_required(VERSION 3.12)
project("bert.cpp" C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

#
# Option list
#

# debug
option(BERT_ALL_WARNINGS           "bert: enable all compiler warnings"                   OFF)
option(BERT_ALL_WARNINGS_3RD_PARTY "bert: enable all compiler warnings in 3rd party libs" OFF)
option(BERT_GPROF                  "bert: enable gprof"                                   OFF)
option(BERT_PERF                   "bert: enable performance logging"                     OFF)

# instruction set specific
option(BERT_AVX                    "bert: enable AVX"                                     ON)
option(BERT_AVX2                   "bert: enable AVX2"                                    ON)
option(BERT_FMA                    "bert: enable FMA"                                     ON)
option(BERT_CUBLAS                 "bert: enable cuBLAS"                                  OFF)

#
# Compile flags
#

set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_C_STANDARD_REQUIRED true)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

if (BERT_ALL_WARNINGS)
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Wcast-qual
        -Wno-unused-function
    )
endif()

add_compile_options(-Wno-format)

if(BERT_PERF)
    add_compile_definitions(GGML_PERF)
endif()

set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

if (BERT_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT result OUTPUT output)
    if (result)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(WARNING "IPO is not supported: ${output}")
    endif()
endif()

# architecture specific
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "^(x86_64|i686|AMD64)$")
    message(STATUS "x86 detected")
    add_compile_options(-mf16c)
    if (BERT_FMA)
        add_compile_options(-mfma)
    endif()
    if (BERT_AVX)
        add_compile_options(-mavx)
    endif()
    if (BERT_AVX2)
        add_compile_options(-mavx2)
    endif()
    if (GGML_CUBLAS)
        add_compile_definitions(GGML_USE_CUBLAS)
    endif()
else()
    message(STATUS "Unknown architecture")
endif()

#
# Build libraries
#

add_subdirectory(ggml)
add_subdirectory(examples)
add_subdirectory(models)

add_library(bert bert.cpp bert.h)

target_include_directories(bert PUBLIC .)
target_compile_features(bert PUBLIC cxx_std_20)
target_link_libraries(bert PRIVATE ggml ${BERT_EXTRA_LIBS})

# for shared libraries
set_target_properties(ggml PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(bert PROPERTIES POSITION_INDEPENDENT_CODE ON)
