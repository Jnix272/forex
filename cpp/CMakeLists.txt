cmake_minimum_required(VERSION 3.20)
project(ForexScalingRuntime CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler warning and optimization flags
if(MSVC)
    add_compile_options(/W4 /permissive- /utf-8 /MP)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Oi /Ot /Gy /DNDEBUG")
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
endif()

# Execution provider options
option(ENABLE_DIRECTML "Enable DirectML Execution Provider support on Windows" OFF)

# Find dependencies (vcpkg, system, or package config)
find_package(onnxruntime CONFIG QUIET)
if(NOT onnxruntime_FOUND)
    find_path(ONNXRUNTIME_INCLUDE_DIR NAMES onnxruntime_cxx_api.h 
        PATHS "${CMAKE_CURRENT_SOURCE_DIR}/deps/onnxruntime/include"
        PATH_SUFFIXES onnxruntime
    )
    find_library(ONNXRUNTIME_LIB NAMES onnxruntime 
        PATHS "${CMAKE_CURRENT_SOURCE_DIR}/deps/onnxruntime/lib"
    )
    if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIB)
        add_library(onnxruntime UNKNOWN IMPORTED)
        set_target_properties(onnxruntime PROPERTIES
            IMPORTED_LOCATION "${ONNXRUNTIME_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
        )
    else()
        message(STATUS "onnxruntime CMake config not found; attempting standard search via vcpkg/system.")
        find_package(onnxruntime REQUIRED)
    endif()
endif()

find_package(cppzmq CONFIG QUIET)
find_package(CURL QUIET)

find_package(nlohmann_json CONFIG QUIET)
if(NOT nlohmann_json_FOUND)
    include(FetchContent)
    message(STATUS "nlohmann_json not found via CONFIG; fetching via FetchContent fallback...")
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# Platform-specific runtime linkage (Windows networking & sockets)
set(PLATFORM_LIBS "")
if(WIN32)
    list(APPEND PLATFORM_LIBS ws2_32 iphlpapi)
endif()

# Main inference executable (requires ZeroMQ)
if(cppzmq_FOUND)
    add_executable(forex_runner
        src/main.cpp
        src/onnx_runner.cpp
        src/zmq_receiver.cpp
        src/ensemble_runner.cpp
    )

    target_include_directories(forex_runner PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    target_link_libraries(forex_runner PRIVATE 
        onnxruntime 
        cppzmq 
        nlohmann_json::nlohmann_json
        ${PLATFORM_LIBS}
    )

    if(ENABLE_DIRECTML)
        target_compile_definitions(forex_runner PRIVATE USE_DIRECTML=1)
    endif()

    if(MSVC)
        target_compile_definitions(forex_runner PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS)
    endif()
endif()

# OANDA streaming tick receiver — publishes bid/ask over ZMQ PUB
if(cppzmq_FOUND AND CURL_FOUND)
    add_executable(oanda_stream
        src/oanda_stream_main.cpp
        src/oanda_stream.cpp
    )
    target_include_directories(oanda_stream PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(oanda_stream PRIVATE
        cppzmq
        CURL::libcurl
        nlohmann_json::nlohmann_json
        ${PLATFORM_LIBS}
    )
    if(MSVC)
        target_compile_definitions(oanda_stream PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS)
    endif()
    message(STATUS "oanda_stream target: enabled (cppzmq + curl found)")
else()
    message(STATUS "oanda_stream target: SKIPPED (needs cppzmq and libcurl)")
endif()

# Ensemble benchmark executable (standalone low-latency performance benchmark)
add_executable(ensemble_benchmark
    src/ensemble_benchmark.cpp
    src/ensemble_runner.cpp
    src/onnx_runner.cpp
)

target_include_directories(ensemble_benchmark PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(ensemble_benchmark PRIVATE 
    onnxruntime
    ${PLATFORM_LIBS}
)

if(ENABLE_DIRECTML)
    target_compile_definitions(ensemble_benchmark PRIVATE USE_DIRECTML=1)
endif()

if(MSVC)
    target_compile_definitions(ensemble_benchmark PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS)
endif()

# Verification test executable
enable_testing()
add_executable(onnx_verifier
    src/onnx_verifier.cpp
    src/onnx_runner.cpp
)

target_include_directories(onnx_verifier PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(onnx_verifier PRIVATE 
    onnxruntime
    ${PLATFORM_LIBS}
)

if(ENABLE_DIRECTML)
    target_compile_definitions(onnx_verifier PRIVATE USE_DIRECTML=1)
endif()

if(MSVC)
    target_compile_definitions(onnx_verifier PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS)
endif()

