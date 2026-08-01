#include "onnx_runner.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

std::vector<float> load_binary(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Could not open " + path);
    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<float> buffer(size / sizeof(float));
    if (file.read(reinterpret_cast<char*>(buffer.data()), size))
        return buffer;
    throw std::runtime_error("Could not read " + path);
}

int main(int argc, char** argv) {
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0] << " <model.onnx> <test_input.bin> <test_output.bin> <batch_size> <seq_len> <n_features>\n";
        return 1;
    }

    try {
        std::string model_path = argv[1];
        std::vector<float> input = load_binary(argv[2]);
        std::vector<float> expected_output = load_binary(argv[3]);
        int batch_size = std::stoi(argv[4]);
        int seq_len = std::stoi(argv[5]);
        int n_features = std::stoi(argv[6]);

        ONNXRunner runner(model_path);
        std::vector<float> actual_output = runner.predict(input, batch_size, seq_len, n_features);

        if (actual_output.size() != expected_output.size()) {
            std::cerr << "Size mismatch! Expected: " << expected_output.size() << " Got: " << actual_output.size() << "\n";
            return 1;
        }

        float max_diff = 0.0f;
        for (size_t i = 0; i < actual_output.size(); ++i) {
            float diff = std::abs(actual_output[i] - expected_output[i]);
            if (diff > max_diff) max_diff = diff;
        }

        std::cout << "Max diff: " << max_diff << "\n";
        if (max_diff > 1e-4) { // Slightly looser for fp32 across frameworks
            std::cerr << "FAILED: Differences exceed 1e-4\n";
            return 1;
        }
        
        std::cout << "PASSED: PyTorch vs ONNX parity confirmed.\n";
        return 0;

    } catch(const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
}
