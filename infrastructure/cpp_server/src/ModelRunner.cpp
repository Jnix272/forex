#include "ModelRunner.h"
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <sstream>
#include <array>

ModelRunner::ModelRunner(const std::string& model_path)
    : seq_len_(0),
      num_features_(0),
      agent_state_size_(0),
      output_count_(0),
      has_agent_state_input_(false) {
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ForexModelRunner");
    
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef _WIN32
    std::wstring widestr(model_path.begin(), model_path.end());
    session_ = std::make_unique<Ort::Session>(*env_, widestr.c_str(), session_options);
#else
    session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);
#endif

    memory_info_ = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

    loadModelMetadata();
}

ModelRunner::~ModelRunner() = default;

void ModelRunner::loadModelMetadata() {
    size_t input_count = session_->GetInputCount();
    if ((input_count != 1 && input_count != 2) || session_->GetOutputCount() != 1) {
        throw std::runtime_error("Expected one or two ONNX inputs and exactly one ONNX output");
    }

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session_->GetInputNameAllocated(0, allocator);
    auto output_name = session_->GetOutputNameAllocated(0, allocator);
    input_node_name_storage_ = input_name.get();
    output_node_name_storage_ = output_name.get();
    if (input_count == 2) {
        auto state_name = session_->GetInputNameAllocated(1, allocator);
        agent_state_node_name_storage_ = state_name.get();
        has_agent_state_input_ = true;
    }

    input_node_names_ = {input_node_name_storage_.c_str()};
    if (has_agent_state_input_) {
        input_node_names_.push_back(agent_state_node_name_storage_.c_str());
    }
    output_node_names_ = {output_node_name_storage_.c_str()};

    auto input_info = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
    auto input_shape = input_info.GetShape();
    if (input_shape.size() != 3) {
        std::ostringstream oss;
        oss << "Expected ONNX input rank 3 [batch, seq_len, features], got rank "
            << input_shape.size();
        throw std::runtime_error(oss.str());
    }

    if (input_shape[1] <= 0 || input_shape[2] <= 0) {
        std::ostringstream oss;
        oss << "ONNX model must have fixed seq_len and feature dimensions. Got ["
            << input_shape[0] << ", " << input_shape[1] << ", " << input_shape[2] << "]";
        throw std::runtime_error(oss.str());
    }

    seq_len_ = static_cast<int>(input_shape[1]);
    num_features_ = static_cast<int>(input_shape[2]);

    if (has_agent_state_input_) {
        auto state_info = session_->GetInputTypeInfo(1).GetTensorTypeAndShapeInfo();
        auto state_shape = state_info.GetShape();
        if (state_shape.size() != 2 || state_shape[1] <= 0) {
            std::ostringstream oss;
            oss << "Expected agent_state input rank 2 [batch, state], got rank "
                << state_shape.size();
            throw std::runtime_error(oss.str());
        }
        agent_state_size_ = static_cast<int>(state_shape[1]);
    }

    auto output_info = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
    auto output_shape = output_info.GetShape();
    if (!output_shape.empty() && output_shape.back() > 0) {
        output_count_ = static_cast<int>(output_shape.back());
    }

    std::cout << "[MODEL] input=" << input_node_name_storage_
              << " output=" << output_node_name_storage_
              << " shape=[1," << seq_len_ << "," << num_features_ << "]"
              << " state=" << agent_state_size_
              << " outputs=" << output_count_
              << std::endl;
}

int ModelRunner::predict(const std::vector<float>& input_features, int seq_len, int num_features) {
    if (input_features.size() != static_cast<size_t>(seq_len * num_features)) {
        throw std::invalid_argument("Input feature size mismatch");
    }
    if (seq_len != seq_len_ || num_features != num_features_) {
        std::ostringstream oss;
        oss << "Input shape mismatch. Model expects [1," << seq_len_ << "," << num_features_
            << "], got [1," << seq_len << "," << num_features << "]";
        throw std::invalid_argument(oss.str());
    }

    std::vector<int64_t> input_node_dims = {1, seq_len, num_features};
    
    // Use const_cast safely because ONNX API requires non-const pointer, but it doesn't modify the data for inputs
    auto input_tensor = Ort::Value::CreateTensor<float>(
        *memory_info_, 
        const_cast<float*>(input_features.data()), 
        input_features.size(), 
        input_node_dims.data(), 
        input_node_dims.size()
    );

    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr}, 
        input_node_names_.data(), 
        &input_tensor, 1, 
        output_node_names_.data(), 1
    );

    float* floatarr = output_tensors.front().GetTensorMutableData<float>();
    size_t output_count = output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount();
    if (output_count != 3) {
        throw std::runtime_error("Expected ONNX output to contain exactly 3 logits [Short, Hold, Long]");
    }
    
    // Assume output is logits for [Short, Hold, Long] (3 classes)
    // Find argmax
    int max_idx = 0;
    float max_val = floatarr[0];
    for (int i = 1; i < 3; ++i) {
        if (floatarr[i] > max_val) {
            max_val = floatarr[i];
            max_idx = i;
        }
    }

    // Map: 0 -> Short (-1), 1 -> Hold (0), 2 -> Long (1)
    if (max_idx == 0) return -1;
    if (max_idx == 2) return 1;
    return 0;
}

int ModelRunner::predictExecution(
    const std::vector<float>& input_features,
    int seq_len,
    int num_features,
    const std::vector<float>& agent_state
) {
    if (!has_agent_state_input_) {
        throw std::runtime_error("Execution policy ONNX must have features and agent_state inputs");
    }
    if (input_features.size() != static_cast<size_t>(seq_len * num_features)) {
        throw std::invalid_argument("Input feature size mismatch");
    }
    if (seq_len != seq_len_ || num_features != num_features_) {
        std::ostringstream oss;
        oss << "Input shape mismatch. Model expects [1," << seq_len_ << "," << num_features_
            << "], got [1," << seq_len << "," << num_features << "]";
        throw std::invalid_argument(oss.str());
    }
    if (agent_state.size() != static_cast<size_t>(agent_state_size_)) {
        std::ostringstream oss;
        oss << "Agent state size mismatch. Model expects " << agent_state_size_
            << ", got " << agent_state.size();
        throw std::invalid_argument(oss.str());
    }

    std::vector<int64_t> feature_dims = {1, seq_len, num_features};
    std::vector<int64_t> state_dims = {1, agent_state_size_};

    auto feature_tensor = Ort::Value::CreateTensor<float>(
        *memory_info_,
        const_cast<float*>(input_features.data()),
        input_features.size(),
        feature_dims.data(),
        feature_dims.size()
    );
    auto state_tensor = Ort::Value::CreateTensor<float>(
        *memory_info_,
        const_cast<float*>(agent_state.data()),
        agent_state.size(),
        state_dims.data(),
        state_dims.size()
    );

    std::array<Ort::Value, 2> input_tensors = {
        std::move(feature_tensor),
        std::move(state_tensor)
    };

    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_node_names_.data(),
        input_tensors.data(),
        input_tensors.size(),
        output_node_names_.data(),
        1
    );

    float* floatarr = output_tensors.front().GetTensorMutableData<float>();
    size_t output_count = output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount();
    if (output_count == 0) {
        throw std::runtime_error("Execution policy returned no action logits");
    }

    int max_idx = 0;
    float max_val = floatarr[0];
    for (size_t i = 1; i < output_count; ++i) {
        if (floatarr[i] > max_val) {
            max_val = floatarr[i];
            max_idx = static_cast<int>(i);
        }
    }
    return max_idx;
}
