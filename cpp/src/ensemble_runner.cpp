#include "ensemble_runner.h"
#include <iostream>
#include <cmath>
#include <numeric>
#include <future>
#include <algorithm>
#include <stdexcept>

float EnsembleRunner::logits_to_signal(const std::vector<float>& logits) {
    if (logits.empty()) {
        return 0.0f;
    }

    // Check for NaN or Inf in inputs
    for (float v : logits) {
        if (std::isnan(v) || std::isinf(v)) {
            std::cerr << "[Ensemble] Warning: NaN/Inf detected in model logits. Returning neutral signal.\n";
            return 0.0f;
        }
    }

    if (logits.size() == 1) {
        // Scalar regression signal clamped to [-1.0, 1.0]
        return std::clamp(logits[0], -1.0f, 1.0f);
    }

    if (logits.size() == 2) {
        // Binary classification [Short, Long]
        float max_l = std::max(logits[0], logits[1]);
        float e0 = std::exp(logits[0] - max_l);
        float e1 = std::exp(logits[1] - max_l);
        float sum = e0 + e1;
        return (sum > 0.0f && !std::isnan(sum)) ? ((e1 - e0) / sum) : 0.0f;
    }

    if (logits.size() == 3) {
        // Standard Forex 3-class classification [Class 0: Sell, Class 1: Hold, Class 2: Buy]
        float max_l = std::max({logits[0], logits[1], logits[2]});
        float e0 = std::exp(logits[0] - max_l);
        float e1 = std::exp(logits[1] - max_l);
        float e2 = std::exp(logits[2] - max_l);
        float sum = e0 + e1 + e2;
        if (sum <= 0.0f || std::isnan(sum)) return 0.0f;
        
        float p_sell = e0 / sum;
        float p_buy  = e2 / sum;
        // Directional conviction in [-1.0, +1.0]
        return std::clamp(p_buy - p_sell, -1.0f, 1.0f);
    }

    if (logits.size() == 5) {
        // 5-class discrete RL actions [-1.0, -0.5, 0.0, +0.5, +1.0]
        float max_l = *std::max_element(logits.begin(), logits.end());
        float sum = 0.0f;
        std::vector<float> exp_l(5);
        for (size_t i = 0; i < 5; ++i) {
            exp_l[i] = std::exp(logits[i] - max_l);
            sum += exp_l[i];
        }
        if (sum <= 0.0f || std::isnan(sum)) return 0.0f;

        const float actions[5] = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
        float expected_val = 0.0f;
        for (size_t i = 0; i < 5; ++i) {
            expected_val += actions[i] * (exp_l[i] / sum);
        }
        return std::clamp(expected_val, -1.0f, 1.0f);
    }

    if (logits.size() == 10) {
        // 10-class ScalingAction: 0=HOLD, 1=OPEN_LONG, 2=OPEN_SHORT, 3-5=SCALE_IN, 6-8=SCALE_OUT, 9=CLOSE_ALL
        // Full softmax over all 10 classes; sum hold-group probabilities.
        // Previous code used max(hold_logits) as one class — underestimates hold mass
        // when uncertainty is spread across multiple hold actions.
        float max_l = *std::max_element(logits.begin(), logits.end());
        float exps[10];
        float total = 0.0f;
        for (int i = 0; i < 10; ++i) { exps[i] = std::exp(logits[i] - max_l); total += exps[i]; }
        if (total <= 0.0f || std::isnan(total)) return 0.0f;
        const float p_buy  = exps[1] / total;
        const float p_sell = exps[2] / total;
        return std::clamp(p_buy - p_sell, -1.0f, 1.0f);
    }

    // Default fallback for general multidimensional logits: neutral signal
    std::cerr << "[Ensemble] Warning: Unrecognized logits dimension (" << logits.size() 
              << "). Returning neutral signal 0.0\n";
    return 0.0f;
}

EnsembleRunner::EnsembleRunner(
    const std::vector<std::string>& model_paths, 
    int batch_size, 
    int seq_len, 
    int n_features, 
    float max_variance_threshold
) : max_variance_(max_variance_threshold),
    batch_size_(batch_size > 0 ? batch_size : 1),
    max_seq_len_(seq_len > 0 ? seq_len : 60),
    common_n_features_(n_features > 0 ? n_features : 584),
    concurrent_(true) {
    
    models_.reserve(model_paths.size());
    for (size_t i = 0; i < model_paths.size(); ++i) {
        SubModelConfig cfg;
        cfg.name = "model_" + std::to_string(i);
        cfg.path = model_paths[i];
        cfg.seq_len = seq_len;
        cfg.n_features = n_features;
        cfg.weight = 1.0f;
        cfg.ep = ONNXRunner::ExecutionProvider::CPU;

        auto runner = std::make_unique<ONNXRunner>(cfg.path, cfg.ep);
        models_.push_back(ModelEntry{std::move(cfg), std::move(runner)});
    }
}

EnsembleRunner::EnsembleRunner(
    const std::vector<SubModelConfig>& model_configs,
    int batch_size,
    float max_variance_threshold,
    bool enable_concurrent_execution
) : max_variance_(max_variance_threshold),
    batch_size_(batch_size > 0 ? batch_size : 1),
    max_seq_len_(0),
    common_n_features_(0),
    concurrent_(enable_concurrent_execution) {
    
    models_.reserve(model_configs.size());
    for (const auto& cfg : model_configs) {
        if (cfg.seq_len > max_seq_len_) {
            max_seq_len_ = cfg.seq_len;
        }
        if (cfg.n_features > common_n_features_) {
            common_n_features_ = cfg.n_features;
        }
        auto runner = std::make_unique<ONNXRunner>(cfg.path, cfg.ep);
        models_.push_back(ModelEntry{cfg, std::move(runner)});
    }
    if (common_n_features_ <= 0) {
        common_n_features_ = 584;
    }
    if (max_seq_len_ <= 0) {
        max_seq_len_ = 60;
    }
}

EnsembleRunner::~EnsembleRunner() = default;

EnsembleRunner::EnsembleRunner(EnsembleRunner&& other) noexcept {
    // Must lock other.inference_mutex_ to prevent a concurrent infer_detailed() on
    // other from reading other.models_ while we move it out.
    std::lock_guard<std::mutex> lock(other.inference_mutex_);
    models_            = std::move(other.models_);
    max_variance_      = other.max_variance_;
    batch_size_        = other.batch_size_;
    max_seq_len_       = other.max_seq_len_;
    common_n_features_ = other.common_n_features_;
    concurrent_        = other.concurrent_;
}

EnsembleRunner& EnsembleRunner::operator=(EnsembleRunner&& other) noexcept {
    if (this != &other) {
        std::scoped_lock lock(inference_mutex_, other.inference_mutex_);
        models_ = std::move(other.models_);
        max_variance_ = other.max_variance_;
        batch_size_ = other.batch_size_;
        max_seq_len_ = other.max_seq_len_;
        common_n_features_ = other.common_n_features_;
        concurrent_ = other.concurrent_;
    }
    return *this;
}

float EnsembleRunner::infer(const std::vector<float>& features) {
    return infer_detailed(features).final_signal;
}

EnsembleResult EnsembleRunner::infer_detailed(const std::vector<float>& features) {
    std::lock_guard<std::mutex> lock(inference_mutex_);
    EnsembleResult result;

    if (models_.empty()) {
        std::cerr << "[Ensemble] Warning: No models loaded in ensemble.\n";
        return result;
    }

    if (features.empty()) {
        std::cerr << "[Ensemble] Error: Empty input features received.\n";
        return result;
    }

    // Determine incoming sequence length
    const size_t total_elements = features.size();
    const size_t elements_per_step = static_cast<size_t>(batch_size_ * common_n_features_);
    if (elements_per_step == 0 || total_elements % elements_per_step != 0) {
        std::cerr << "[Ensemble] Error: Features size " << total_elements 
                  << " is not divisible by (batch_size * n_features) = " << elements_per_step << "\n";
        return result;
    }
    const size_t incoming_seq_len = total_elements / elements_per_step;

    // Helper lambda to run one model with appropriate sliding-window slice
    auto run_single_model = [this, &features, incoming_seq_len](ModelEntry& entry) -> float {
        const int required_seq = entry.config.seq_len;
        const int required_feat = entry.config.n_features;

        if (incoming_seq_len < static_cast<size_t>(required_seq)) {
            std::cerr << "[Ensemble] Error: Model [" << entry.config.name 
                      << "] requires seq_len=" << required_seq 
                      << " but incoming buffer only has " << incoming_seq_len << " steps.\n";
            return 0.0f;
        }

        // Slice the most recent required_seq steps from the incoming feature stream
        const size_t step_offset = incoming_seq_len - static_cast<size_t>(required_seq);
        const size_t float_offset = step_offset * static_cast<size_t>(common_n_features_);
        const size_t slice_count = static_cast<size_t>(batch_size_ * required_seq * required_feat);

        try {
            std::vector<float> packed_buffer;
            const float* slice_ptr = nullptr;

            if (required_feat == common_n_features_ && (batch_size_ == 1 || step_offset == 0)) {
                // Zero-copy contiguous slice directly from the incoming buffer (valid when batch_size == 1 or no step slicing)
                slice_ptr = features.data() + float_offset;
            } else if (required_feat <= common_n_features_) {
                // Multi-batch (batch_size > 1) or heterogeneous feature dimension: pack row-by-row to guarantee memory correctness
                packed_buffer.resize(slice_count);
                for (size_t b = 0; b < static_cast<size_t>(batch_size_); ++b) {
                    for (size_t s = 0; s < static_cast<size_t>(required_seq); ++s) {
                        const size_t src_step = step_offset + s;
                        // Wire layout is time-major: [step0_b0, step0_b1, ..., step1_b0, step1_b1, ...]
                        // (elements_per_step = batch_size * n_features, consistent with line above).
                        // Previous code used batch-major index (b*seq+step), which is wrong for batch>1.
                        const size_t src_idx = (src_step * static_cast<size_t>(batch_size_) + b) * static_cast<size_t>(common_n_features_);
                        const size_t dst_idx = (b * static_cast<size_t>(required_seq) + s) * static_cast<size_t>(required_feat);
                        std::memcpy(packed_buffer.data() + dst_idx, features.data() + src_idx, static_cast<size_t>(required_feat) * sizeof(float));
                    }
                }
                slice_ptr = packed_buffer.data();
            } else {
                std::cerr << "[Ensemble] Error: Model [" << entry.config.name 
                          << "] requires " << required_feat << " features, but input stream only provides " 
                          << common_n_features_ << "\n";
                return 0.0f;
            }

            std::vector<float> raw_out = entry.runner->predict(
                slice_ptr, slice_count, batch_size_, required_seq, required_feat
            );
            return logits_to_signal(raw_out);
        } catch (const std::exception& e) {
            std::cerr << "[Ensemble] Error evaluating model [" << entry.config.name << "]: " 
                      << e.what() << "\n";
            return 0.0f;
        } catch (...) {
            std::cerr << "[Ensemble] Unknown exception evaluating model [" << entry.config.name << "]\n";
            return 0.0f;
        }
    };

    std::vector<float> signals(models_.size(), 0.0f);
    result.model_names.reserve(models_.size());
    result.individual_signals.reserve(models_.size());

    if (concurrent_ && models_.size() > 1) {
        // Execute models in parallel across CPU threads
        std::vector<std::future<float>> futures;
        futures.reserve(models_.size());
        for (auto& entry : models_) {
            futures.push_back(std::async(std::launch::async, [&entry, &run_single_model]() {
                return run_single_model(entry);
            }));
        }
        for (size_t i = 0; i < futures.size(); ++i) {
            try {
                signals[i] = futures[i].get();
            } catch (const std::exception& e) {
                std::cerr << "[Ensemble] Future error for model [" << models_[i].config.name << "]: " << e.what() << "\n";
                signals[i] = 0.0f;
            } catch (...) {
                std::cerr << "[Ensemble] Unknown future error for model [" << models_[i].config.name << "]\n";
                signals[i] = 0.0f;
            }
        }
    } else {
        // Sequential execution
        for (size_t i = 0; i < models_.size(); ++i) {
            signals[i] = run_single_model(models_[i]);
        }
    }

    float weight_sum = 0.0f;
    float weighted_signal_sum = 0.0f;

    for (size_t i = 0; i < models_.size(); ++i) {
        float w = std::max(models_[i].config.weight, 0.0f);
        weight_sum += w;
        weighted_signal_sum += w * signals[i];
        result.individual_signals.push_back(signals[i]);
        result.model_names.push_back(models_[i].config.name);
    }

    if (weight_sum <= 0.0f) {
        result.circuit_broken = true;
        result.final_signal = 0.0f;
        return result;
    }

    const float weighted_mean = weighted_signal_sum / weight_sum;
    result.weighted_mean = weighted_mean;

    // Compute weighted variance
    float var_sum = 0.0f;
    size_t agreement_count = 0;

    for (size_t i = 0; i < models_.size(); ++i) {
        float diff = signals[i] - weighted_mean;
        float w = std::max(models_[i].config.weight, 0.0f);
        var_sum += w * (diff * diff);

        // Directional agreement check
        bool sign_match = (signals[i] * weighted_mean > 0.0f) || 
                          (std::abs(signals[i]) < 0.05f && std::abs(weighted_mean) < 0.05f);
        if (sign_match) {
            agreement_count++;
        }
    }

    const float weighted_var = var_sum / weight_sum;
    result.weighted_variance = weighted_var;
    result.agreement_ratio = static_cast<float>(agreement_count) / static_cast<float>(models_.size());

    // Circuit breaker / uncertainty threshold check (robust against NaN/Inf)
    if (std::isnan(weighted_var) || std::isnan(weighted_mean) || 
        std::isinf(weighted_var) || std::isinf(weighted_mean) || 
        weighted_var > max_variance_) {
        result.circuit_broken = true;
        result.final_signal = 0.0f;
    } else {
        result.circuit_broken = false;
        result.final_signal = std::clamp(weighted_mean, -1.0f, 1.0f);
    }

    return result;
}

