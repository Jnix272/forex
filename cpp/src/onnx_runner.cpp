#include "onnx_runner.h"
#include <iostream>
#include <stdexcept>
#include <string>
#include <filesystem>
#include <limits>

#if defined(USE_DIRECTML)
#include <dml_provider_factory.h>
#endif

ONNXRunner::ONNXRunner(const std::string& model_path, ExecutionProvider ep, int intra_op_threads)
    : model_path_(model_path),
      ep_(ep),
      intra_op_threads_(intra_op_threads > 0 ? intra_op_threads : 1),
      env_(ORT_LOGGING_LEVEL_WARNING, "ONNXRunner"),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(intra_op_threads_);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#if defined(USE_DIRECTML)
    if (ep_ == ExecutionProvider::DirectML) {
        try {
            // Append DirectML execution provider (device_id 0)
            OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_DML(
                static_cast<OrtSessionOptions*>(session_options), 0
            );
            if (status != nullptr) {
                std::string err_msg = Ort::GetApi().GetErrorMessage(status);
                Ort::GetApi().ReleaseStatus(status);
                throw std::runtime_error(err_msg);
            }
            std::cout << "[ONNXRunner] DirectML execution provider enabled for " << model_path_ << "\n";
        } catch (const std::exception& e) {
            std::cerr << "[ONNXRunner] DirectML setup failed: " << e.what() 
                      << ". Falling back to CPU.\n";
            ep_ = ExecutionProvider::CPU;
        }
    }
#else
    if (ep_ == ExecutionProvider::DirectML) {
        std::cerr << "[ONNXRunner] DirectML requested but compiled without USE_DIRECTML. Using CPU.\n";
        ep_ = ExecutionProvider::CPU;
    }
#endif

    std::filesystem::path fs_path(model_path);
    if (!std::filesystem::exists(fs_path)) {
        throw std::invalid_argument("Model file does not exist: " + model_path);
    }

    try {
        session_ = Ort::Session(env_, fs_path.c_str(), session_options);
    } catch (const Ort::Exception& e) {
        std::cerr << "[ONNXRunner] Failed to load ONNX model [" << model_path << "]: " << e.what() << std::endl;
        throw;
    }

    init_node_metadata();
}

ONNXRunner::~ONNXRunner() = default;

ONNXRunner::ONNXRunner(ONNXRunner&& other) noexcept
    : model_path_(std::move(other.model_path_)),
      ep_(other.ep_),
      intra_op_threads_(other.intra_op_threads_),
      env_(std::move(other.env_)),
      session_(std::move(other.session_)),
      memory_info_(std::move(other.memory_info_)),
      input_node_names_allocated_(std::move(other.input_node_names_allocated_)),
      output_node_names_allocated_(std::move(other.output_node_names_allocated_)) {
    other.input_node_names_.clear();
    other.output_node_names_.clear();
    refresh_node_name_ptrs();
}

ONNXRunner& ONNXRunner::operator=(ONNXRunner&& other) noexcept {
    if (this != &other) {
        model_path_ = std::move(other.model_path_);
        ep_ = other.ep_;
        intra_op_threads_ = other.intra_op_threads_;
        // Critical: Release session before environment to preserve handle lifetime invariants
        session_ = std::move(other.session_);
        env_ = std::move(other.env_);
        memory_info_ = std::move(other.memory_info_);
        input_node_names_allocated_ = std::move(other.input_node_names_allocated_);
        output_node_names_allocated_ = std::move(other.output_node_names_allocated_);
        other.input_node_names_.clear();
        other.output_node_names_.clear();
        refresh_node_name_ptrs();
    }
    return *this;
}

void ONNXRunner::refresh_node_name_ptrs() {
    input_node_names_.clear();
    input_node_names_.reserve(input_node_names_allocated_.size());
    for (const auto& name : input_node_names_allocated_) {
        input_node_names_.push_back(name.c_str());
    }

    output_node_names_.clear();
    output_node_names_.reserve(output_node_names_allocated_.size());
    for (const auto& name : output_node_names_allocated_) {
        output_node_names_.push_back(name.c_str());
    }
}

void ONNXRunner::init_node_metadata() {
    Ort::AllocatorWithDefaultOptions allocator;

    // Discover input node names dynamically
    size_t num_inputs = session_.GetInputCount();
    input_node_names_allocated_.clear();
    input_node_names_allocated_.reserve(num_inputs);
    for (size_t i = 0; i < num_inputs; ++i) {
        auto name_ptr = session_.GetInputNameAllocated(i, allocator);
        if (name_ptr && name_ptr.get()) {
            input_node_names_allocated_.emplace_back(name_ptr.get());
        }
    }
    if (input_node_names_allocated_.empty()) {
        input_node_names_allocated_.emplace_back("features");
    }
    if (num_inputs > 1) {
        std::cerr << "[ONNXRunner] Notice: Model has " << num_inputs 
                  << " inputs; single-tensor predict will supply primary input [" 
                  << input_node_names_allocated_[0] << "].\n";
    }

    // Discover output node names dynamically
    size_t num_outputs = session_.GetOutputCount();
    output_node_names_allocated_.clear();
    output_node_names_allocated_.reserve(num_outputs);
    for (size_t i = 0; i < num_outputs; ++i) {
        auto name_ptr = session_.GetOutputNameAllocated(i, allocator);
        if (name_ptr && name_ptr.get()) {
            output_node_names_allocated_.emplace_back(name_ptr.get());
        }
    }
    if (output_node_names_allocated_.empty()) {
        output_node_names_allocated_.emplace_back("logits");
    }

    refresh_node_name_ptrs();
}

std::vector<float> ONNXRunner::predict(
    const std::vector<float>& input_data, 
    int batch_size, 
    int seq_len, 
    int n_features
) {
    return predict(input_data.data(), input_data.size(), batch_size, seq_len, n_features);
}

std::vector<float> ONNXRunner::predict(
    const float* input_data, 
    size_t num_elements, 
    int batch_size, 
    int seq_len, 
    int n_features
) {
    if (batch_size <= 0 || seq_len <= 0 || n_features <= 0) {
        throw std::invalid_argument("[ONNXRunner] batch_size, seq_len, and n_features must all be > 0");
    }
    if (input_data == nullptr) {
        throw std::invalid_argument("[ONNXRunner] input_data pointer is null");
    }
    if (input_node_names_.empty()) {
        throw std::runtime_error("[ONNXRunner] No input nodes configured for [" + model_path_ + "]");
    }

    const uint64_t expected_total = static_cast<uint64_t>(batch_size) * 
                                    static_cast<uint64_t>(seq_len) * 
                                    static_cast<uint64_t>(n_features);
    if (expected_total > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::overflow_error("[ONNXRunner] Input dimensions trigger integer overflow");
    }
    const size_t expected_elements = static_cast<size_t>(expected_total);
    if (num_elements != expected_elements) {
        throw std::invalid_argument(
            "[ONNXRunner] Input data size (" + std::to_string(num_elements) + 
            ") does not match expected dimensions (" + std::to_string(expected_elements) + ")"
        );
    }

    auto in_type_info = session_.GetInputTypeInfo(0);
    auto in_tensor_info = in_type_info.GetTensorTypeAndShapeInfo();
    size_t input_rank = in_tensor_info.GetDimensionsCount();

    std::vector<int64_t> input_node_dims;
    if (input_rank == 2) {
        input_node_dims = {
            static_cast<int64_t>(batch_size),
            static_cast<int64_t>(seq_len * n_features)
        };
    } else {
        input_node_dims = {
            static_cast<int64_t>(batch_size),
            static_cast<int64_t>(seq_len),
            static_cast<int64_t>(n_features)
        };
    }
    
    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_,
        const_cast<float*>(input_data),
        expected_elements,
        input_node_dims.data(),
        input_node_dims.size()
    );

    const char* const single_input_name[] = { input_node_names_[0] };
    std::vector<Ort::Value> output_tensors;
    try {
        output_tensors = session_.Run(
            Ort::RunOptions{nullptr},
            single_input_name,
            &input_tensor,
            1,
            output_node_names_.data(),
            output_node_names_.size()
        );
    } catch (const Ort::Exception& e) {
        throw std::runtime_error(
            "[ONNXRunner] session.Run failed for [" + model_path_ + "]: " + e.what()
        );
    }

    if (output_tensors.empty()) {
        throw std::runtime_error("[ONNXRunner] No output generated by model [" + model_path_ + "]");
    }

    const auto& out_tensor = output_tensors.front();
    auto type_info = out_tensor.GetTensorTypeAndShapeInfo();
    if (type_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error("[ONNXRunner] Model output element type is not float32 in [" + model_path_ + "]");
    }

    size_t count = type_info.GetElementCount();
    const float* floatarr = out_tensor.GetTensorData<float>();

    if (floatarr == nullptr || count == 0) {
        throw std::runtime_error("[ONNXRunner] Invalid or empty output buffer from model [" + model_path_ + "]");
    }

    return std::vector<float>(floatarr, floatarr + count);
}

