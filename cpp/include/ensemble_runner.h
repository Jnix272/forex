#pragma once
#include "onnx_runner.h"
#include <vector>
#include <string>
#include <memory>
#include <mutex>

struct SubModelConfig {
    std::string name;
    std::string path;
    int seq_len{60};       // Per-model required sequence length
    int n_features{584};   // Per-model required feature dimension
    float weight{1.0f};    // Aggregation weight
    ONNXRunner::ExecutionProvider ep{ONNXRunner::ExecutionProvider::CPU};
};

struct EnsembleResult {
    float final_signal{0.0f};    // Directional signal in [-1.0, +1.0]
    float weighted_mean{0.0f};   // Weighted mean of individual model signals
    float weighted_variance{0.0f};// Weighted variance (uncertainty)
    float agreement_ratio{0.0f}; // Fraction of models in agreement with mean direction
    bool circuit_broken{false};  // True if variance exceeded threshold and output was zeroed
    std::vector<float> individual_signals;
    std::vector<std::string> model_names;
};

class EnsembleRunner {
public:
    // Legacy constructor for homogeneous model configurations
    EnsembleRunner(const std::vector<std::string>& model_paths, 
                   int batch_size, int seq_len, int n_features, 
                   float max_variance_threshold = 0.5f);

    // Advanced constructor for heterogeneous architectures (TFT, HAELT, Mamba, GNN)
    EnsembleRunner(const std::vector<SubModelConfig>& model_configs,
                   int batch_size,
                   float max_variance_threshold = 0.5f,
                   bool enable_concurrent_execution = true);

    ~EnsembleRunner();

    EnsembleRunner(const EnsembleRunner&) = delete;
    EnsembleRunner& operator=(const EnsembleRunner&) = delete;
    EnsembleRunner(EnsembleRunner&&) noexcept;
    EnsembleRunner& operator=(EnsembleRunner&&) noexcept;

    // Backward-compatible interface returning scalar signal
    float infer(const std::vector<float>& features);

    // Detailed inference returning telemetry, model breakdown, and uncertainty metrics
    EnsembleResult infer_detailed(const std::vector<float>& features);

    // Converts raw model outputs (3-class logits, 1-class regression, 5-class RL) to signal [-1.0, +1.0]
    static float logits_to_signal(const std::vector<float>& logits);

    size_t model_count() const noexcept { return models_.size(); }
    int max_required_seq_len() const noexcept { return max_seq_len_; }

private:
    struct ModelEntry {
        SubModelConfig config;
        std::unique_ptr<ONNXRunner> runner;
    };

    std::vector<ModelEntry> models_;
    float max_variance_{0.5f};
    int batch_size_{1};
    int max_seq_len_{60};
    int common_n_features_{584};
    bool concurrent_{true};
    mutable std::mutex inference_mutex_;
};

