cmake_minimum_required(VERSION 3.20)
project(ForexScalingRuntime CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies (assumed installed via vcpkg or system)
find_package(onnxruntime CONFIG REQUIRED)
find_package(cppzmq CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

# Include directories
include_directories(include)

# Executable
add_executable(forex_runner
    src/main.cpp
    src/onnx_runner.cpp
    src/zmq_receiver.cpp
    src/ensemble_runner.cpp
)

# Link libraries
target_link_libraries(forex_runner PRIVATE onnxruntime cppzmq nlohmann_json::nlohmann_json)

# Test Executable
enable_testing()
add_executable(onnx_verifier
    src/onnx_verifier.cpp
    src/onnx_runner.cpp
)
target_link_libraries(onnx_verifier PRIVATE onnxruntime)
