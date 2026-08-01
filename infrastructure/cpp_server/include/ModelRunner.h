#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstddef>
// Forward declarations for ONNX Runtime to avoid including heavy headers in our header
namespace Ort {
    class Env;
    class Session;
    class MemoryInfo;
}

class ModelRunner {
public:
    explicit ModelRunner(const std::string& model_path);
    ~ModelRunner();

    // Prevent copy
    ModelRunner(const ModelRunner&) = delete;
    ModelRunner& operator=(const ModelRunner&) = delete;

    // Run inference. Input shape is [1, seq_len, num_features]
    // Returns: 1 (Long), 0 (Hold), -1 (Short)
    int predict(const std::vector<float>& input_features, int seq_len, int num_features);

    // Run native RL execution inference. Input shape is
    // features=[1, seq_len, num_features], agent_state=[1, agent_state_size].
    // Returns the raw ScalingAction index.
    int predictExecution(
        const std::vector<float>& input_features,
        int seq_len,
        int num_features,
        const std::vector<float>& agent_state
    );

    int seqLen() const { return seq_len_; }
    int numFeatures() const { return num_features_; }
    int agentStateSize() const { return agent_state_size_; }
    int outputCount() const { return output_count_; }
    bool hasAgentStateInput() const { return has_agent_state_input_; }
    const std::string& inputName() const { return input_node_name_storage_; }
    const std::string& agentStateInputName() const { return agent_state_node_name_storage_; }
    const std::string& outputName() const { return output_node_name_storage_; }

private:
    void loadModelMetadata();

    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::MemoryInfo> memory_info_;

    int seq_len_;
    int num_features_;
    int agent_state_size_;
    int output_count_;
    bool has_agent_state_input_;
    std::string input_node_name_storage_;
    std::string agent_state_node_name_storage_;
    std::string output_node_name_storage_;
    
    std::vector<const char*> input_node_names_;
    std::vector<const char*> output_node_names_;
};
