#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <onnxruntime_cxx_api.h>

class ONNXRunner {
public:
    enum class ExecutionProvider {
        CPU,
        DirectML
    };

    explicit ONNXRunner(
        const std::string& model_path,
        ExecutionProvider ep = ExecutionProvider::CPU,
        int intra_op_threads = 1
    );
    ~ONNXRunner();

    // Disable copying due to Ort::Session handle semantics; enable move operations
    ONNXRunner(const ONNXRunner&) = delete;
    ONNXRunner& operator=(const ONNXRunner&) = delete;
    ONNXRunner(ONNXRunner&& other) noexcept;
    ONNXRunner& operator=(ONNXRunner&& other) noexcept;

    // Run prediction for input data [batch_size, seq_len, n_features]
    std::vector<float> predict(
        const std::vector<float>& input_data, 
        int batch_size, 
        int seq_len, 
        int n_features
    );

    // Overload for raw pointer input (supports zero-copy slicing of sliding window buffers)
    std::vector<float> predict(
        const float* input_data, 
        size_t num_elements, 
        int batch_size, 
        int seq_len, 
        int n_features
    );

    // Introspection accessors
    const std::vector<std::string>& get_input_names() const noexcept { return input_node_names_allocated_; }
    const std::vector<std::string>& get_output_names() const noexcept { return output_node_names_allocated_; }
    const std::string& get_model_path() const noexcept { return model_path_; }
    ExecutionProvider get_execution_provider() const noexcept { return ep_; }

private:
    void init_node_metadata();
    void refresh_node_name_ptrs();

    std::string model_path_;
    ExecutionProvider ep_{ExecutionProvider::CPU};
    int intra_op_threads_{1};

    Ort::Env env_;
    Ort::Session session_{nullptr};
    Ort::MemoryInfo memory_info_{nullptr};

    // Stored string allocations guaranteeing pointer validity during inference
    std::vector<std::string> input_node_names_allocated_;
    std::vector<std::string> output_node_names_allocated_;
    std::vector<const char*> input_node_names_;
    std::vector<const char*> output_node_names_;
};

