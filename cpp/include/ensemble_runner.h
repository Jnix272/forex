#pragma once
#include "onnx_runner.h"
#include <vector>
#include <string>
#include <memory>

class EnsembleRunner {
public:
    EnsembleRunner(const std::vector<std::string>& model_paths, 
                   int batch_size, int seq_len, int n_features, 
                   float max_variance_threshold = 0.5f);

    // Runs all models, computes mean and variance. 
    // Returns 0.0 if variance > max_variance_threshold, else returns the mean.
    float infer(const std::vector<float>& features);

private:
    std::vector<std::unique_ptr<ONNXRunner>> runners_;
    float max_variance_;
    int batch_size_;
    int seq_len_;
    int n_features_;
};
