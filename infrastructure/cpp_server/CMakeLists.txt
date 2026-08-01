cmake_minimum_required(VERSION 3.15)
project(ForexExecutionServer CXX)

# Use modern C++20 for optimal performance and features
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find ONNX Runtime (assume vcpkg or system install)
find_package(onnxruntime REQUIRED)

# Find cURL for REST API calls
find_package(CURL REQUIRED)

# Find nlohmann_json
find_package(nlohmann_json CONFIG REQUIRED)

# Include directories
include_directories(include)

# Add main executable (exclude ParityChecker.cpp)
file(GLOB SOURCES "src/*.cpp")
list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/ParityChecker.cpp")
add_executable(ForexExecutionServer ${SOURCES})

# Add Parity Checker executable
add_executable(ParityCheck "src/ParityChecker.cpp" "src/RingBuffer.cpp")

# Link libraries
target_link_libraries(ForexExecutionServer PRIVATE 
    onnxruntime 
    CURL::libcurl
    nlohmann_json::nlohmann_json
)

# Optimization flags for HFT
if(MSVC)
    target_compile_options(ForexExecutionServer PRIVATE /O2 /Oi /Ot /fp:fast)
else()
    target_compile_options(ForexExecutionServer PRIVATE -O3 -ffast-math -march=native)
endif()
