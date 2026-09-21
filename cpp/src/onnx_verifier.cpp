#include "onnx_runner.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <filesystem>
#include <iomanip>

static std::vector<float> load_binary(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open binary file: " + path);
    }
    
    std::streamsize size = file.tellg();
    if (size <= 0) {
        throw std::runtime_error("Binary file is empty or unreadable: " + path);
    }

    if (size % static_cast<std::streamsize>(sizeof(float)) != 0) {
        throw std::runtime_error("Binary file size (" + std::to_string(size) + 
                                 " bytes) is not a multiple of sizeof(float): " + path);
    }

    file.seekg(0, std::ios::beg);
    
    size_t float_count = static_cast<size_t>(size) / sizeof(float);
    std::vector<float> buffer(float_count);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw std::runtime_error("Failed to read " + std::to_string(size) + " bytes from " + path);
    }
    return buffer;
}

static void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " <model.onnx> <test_input.bin> <test_output.bin> <batch_size> <seq_len> <n_features> [--tolerance <val>]\n"
              << "\nVerifies parity between PyTorch test binary outputs and ONNX Runtime predictions.\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string first_arg = argv[1];
    if (first_arg == "--help" || first_arg == "-h") {
        print_usage(argv[0]);
        return 0;
    }

    if (argc < 7) {
        std::cerr << "Error: Missing required arguments.\n";
        print_usage(argv[0]);
        return 1;
    }

    float tolerance = 1e-4f;
    for (int i = 7; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--tolerance" && (i + 1 < argc)) {
            try {
                tolerance = std::stof(argv[++i]);
            } catch (...) {
                std::cerr << "Warning: Invalid tolerance value. Defaulting to 1e-4\n";
            }
        }
    }

    try {
        std::string model_path = argv[1];
        std::string input_path = argv[2];
        std::string output_path = argv[3];

        int batch_size = std::stoi(argv[4]);
        int seq_len = std::stoi(argv[5]);
        int n_features = std::stoi(argv[6]);

        if (batch_size <= 0 || seq_len <= 0 || n_features <= 0) {
            std::cerr << "Error: batch_size, seq_len, and n_features must all be > 0\n";
            return 1;
        }

        std::cout << "[Verifier] Loading input binary: " << input_path << "\n";
        std::vector<float> input = load_binary(input_path);

        std::cout << "[Verifier] Loading expected output binary: " << output_path << "\n";
        std::vector<float> expected_output = load_binary(output_path);

        const size_t expected_input_elements = static_cast<size_t>(batch_size) * 
                                              static_cast<size_t>(seq_len) * 
                                              static_cast<size_t>(n_features);
        if (input.size() != expected_input_elements) {
            std::cerr << "Error: Input binary element count (" << input.size() 
                      << ") does not match expected dimension product (" 
                      << expected_input_elements << ")\n";
            return 1;
        }

        std::cout << "[Verifier] Initializing ONNX Runner for: " << model_path << "\n";
        ONNXRunner runner(model_path);
        
        std::cout << "[Verifier] Executing ONNX inference...\n";
        std::vector<float> actual_output = runner.predict(input, batch_size, seq_len, n_features);

        if (actual_output.size() != expected_output.size()) {
            std::cerr << "FAILED: Output dimension mismatch! Expected " 
                      << expected_output.size() << " floats, got " << actual_output.size() << "\n";
            return 1;
        }

        float max_diff = 0.0f;
        double sum_abs_diff = 0.0;
        double sum_sq_diff = 0.0;

        bool nan_detected = false;
        for (size_t i = 0; i < actual_output.size(); ++i) {
            float act = actual_output[i];
            float exp = expected_output[i];
            if (std::isnan(act) || std::isnan(exp) || std::isinf(act) || std::isinf(exp)) {
                std::cerr << "FAILED: NaN or Inf detected at index " << i 
                          << " (actual=" << act << ", expected=" << exp << ")!\n";
                nan_detected = true;
                break;
            }
            float diff = std::abs(act - exp);
            if (diff > max_diff) {
                max_diff = diff;
            }
            double ddiff = static_cast<double>(diff);
            sum_abs_diff += ddiff;
            sum_sq_diff += ddiff * ddiff;
        }

        if (nan_detected) {
            std::cerr << "\n[FAILED] Output contains invalid numeric values (NaN or Inf).\n";
            return 1;
        }

        double mae = sum_abs_diff / static_cast<double>(actual_output.size());
        double rmse = std::sqrt(sum_sq_diff / static_cast<double>(actual_output.size()));

        std::cout << std::scientific << std::setprecision(6);
        std::cout << "\n================ Parity Statistics ================\n"
                  << "  Evaluated elements : " << actual_output.size() << "\n"
                  << "  Max Absolute Diff  : " << max_diff << "\n"
                  << "  Mean Absolute Diff : " << mae << "\n"
                  << "  RMSE               : " << rmse << "\n"
                  << "  Tolerance Limit    : " << tolerance << "\n"
                  << "===================================================\n";

        if (max_diff > tolerance) {
            std::cerr << "\n[FAILED] Output divergence (" << max_diff << ") exceeds tolerance (" 
                      << tolerance << ")\n";
            return 1;
        }
        
        std::cout << "\n[PASSED] PyTorch vs ONNX Runtime parity strictly verified.\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[Exception] " << e.what() << "\n";
        return 1;
    }
}

