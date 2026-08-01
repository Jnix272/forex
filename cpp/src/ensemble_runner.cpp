#include "ensemble_runner.h"
#include <iostream>
#include <cmath>
#include <numeric>

EnsembleRunner::EnsembleRunner(const std::vector<std::string>& model_paths, 
                               int batch_size, int seq_len, int n_features, 
                               float max_variance_threshold)
    : max_variance_(max_variance_threshold),
      batch_size_(batch_size),
      seq_len_(seq_len),
      n_features_(n_features) {
    
    for (const auto& path : model_paths) {
        runners_.push_back(std::make_unique<ONNXRunner>(path));
    }
}

float EnsembleRunner::infer(const std::vector<float>& features) {
    if (runners_.empty()) {
        return 0.0f;
    }

    std::vector<float> predictions;
    predictions.reserve(runners_.size());

    // Run inference on all models
    for (auto& runner : runners_) {
        std::vector<float> out = runner->predict(features, batch_size_, seq_len_, n_features_);
        if (!out.empty()) {
            predictions.push_back(out[0]);
        }
    }

    if (predictions.empty()) {
        return 0.0f;
    }

    // Compute Mean
    float sum = std::accumulate(predictions.begin(), predictions.end(), 0.0f);
    float mean = sum / predictions.size();

    // Compute Variance
    float variance = 0.0f;
    for (float p : predictions) {
        variance += (p - mean) * (p - mean);
    }
    variance /= predictions.size();

    std::cout << "[Ensemble] Mean: " << mean << " | Variance: " << variance << std::endl;

    if (variance > max_variance_) {
        std::cout << "[Ensemble] Variance exceeds threshold! Filtering signal to 0.0" << std::endl;
        return 0.0f;
    }

    return mean;
}
