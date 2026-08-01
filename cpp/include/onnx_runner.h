#pragma once

#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>

class ONNXRunner {
public:
    ONNXRunner(const std::string& model_path);
    ~ONNXRunner();

    std::vector<float> predict(const std::vector<float>& input_data, int batch_size, int seq_len, int n_features);

private:
    Ort::Env env;
    Ort::Session session{nullptr};
    Ort::MemoryInfo memory_info{nullptr};

    std::vector<const char*> input_node_names;
    std::vector<const char*> output_node_names;
};
