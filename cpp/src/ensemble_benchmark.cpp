#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <random>
#include <filesystem>
#include "ensemble_runner.h"

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cout << "Usage: " << argv[0] << " <model1.onnx> <batch_size> <seq_len> <n_features> [--iterations <N>] [--threshold <float>]\n";
        return 1;
    }

    std::string model_path = argv[1];
    int batch_size = std::stoi(argv[2]);
    int seq_len = std::stoi(argv[3]);
    int n_features = std::stoi(argv[4]);
    int iterations = 1000;
    float threshold = 0.5f;

    for (int i = 5; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--iterations" && i + 1 < argc) {
            iterations = std::stoi(argv[++i]);
        } else if (arg == "--threshold" && i + 1 < argc) {
            threshold = std::stof(argv[++i]);
        }
    }

    if (!std::filesystem::exists(model_path)) {
        std::cerr << "Error: Model file does not exist: " << model_path << "\n";
        return 1;
    }

    std::cout << "======================================================================\n";
    std::cout << "FOREX HIGH-PERFORMANCE C++20 INFERENCE ENGINE BENCHMARK\n";
    std::cout << "======================================================================\n";
    std::cout << "Model Path   : " << model_path << "\n";
    std::cout << "Dimensions   : Batch=" << batch_size << ", SeqLen=" << seq_len << ", Features=" << n_features << "\n";
    const int64_t payload_elems = static_cast<int64_t>(batch_size) * seq_len * n_features;
    std::cout << "Payload Size : " << payload_elems << " float32 elements ("
              << (payload_elems * static_cast<int64_t>(sizeof(float)) / 1024.0) << " KB)\n";
    std::cout << "Iterations   : " << iterations << " benchmark ticks\n";
    std::cout << "----------------------------------------------------------------------\n";

    try {
        SubModelConfig cfg;
        cfg.name = std::filesystem::path(model_path).stem().string();
        cfg.path = model_path;
        cfg.seq_len = seq_len;
        cfg.n_features = n_features;
        cfg.weight = 1.0f;
        cfg.ep = ONNXRunner::ExecutionProvider::CPU;

        std::vector<SubModelConfig> specs = { cfg };
        EnsembleRunner runner(specs, batch_size, threshold, true);

        // Generate synthetic market features
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        size_t total_floats = static_cast<size_t>(batch_size) * seq_len * n_features;
        std::vector<float> features(total_floats);
        for (size_t i = 0; i < total_floats; ++i) {
            features[i] = dist(rng);
        }

        // Warmup (10 iterations)
        std::cout << "[Benchmark] Warming up runtime (10 passes)...\n";
        for (int i = 0; i < 10; ++i) {
            runner.infer_detailed(features);
        }

        // Timed benchmark loop
        std::cout << "[Benchmark] Executing " << iterations << " live inference ticks...\n";
        std::vector<double> latencies_us;
        latencies_us.reserve(iterations);

        EnsembleResult sample_res;
        for (int i = 0; i < iterations; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            sample_res = runner.infer_detailed(features);
            auto t1 = std::chrono::high_resolution_clock::now();

            double elapsed_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            latencies_us.push_back(elapsed_us);
        }

        // Compute statistics
        std::sort(latencies_us.begin(), latencies_us.end());
        double sum = std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0);
        double mean_us = sum / iterations;
        double min_us = latencies_us.front();
        double max_us = latencies_us.back();
        double p50_us = latencies_us[static_cast<size_t>(iterations * 0.50)];
        double p90_us = latencies_us[static_cast<size_t>(iterations * 0.90)];
        double p99_us = latencies_us[static_cast<size_t>(iterations * 0.99)];

        std::cout << "\n======================================================================\n";
        std::cout << "BENCHMARK RESULTS & TELEMETRY\n";
        std::cout << "======================================================================\n";
        std::cout << "Sample Output Signal   : " << std::fixed << std::setprecision(4) << sample_res.final_signal << "\n";
        std::cout << "Sample Model Mean      : " << sample_res.weighted_mean << "\n";
        std::cout << "Sample Model Variance  : " << sample_res.weighted_variance << "\n";
        std::cout << "Sample Agreement Ratio : " << (sample_res.agreement_ratio * 100.0f) << "%\n";
        std::cout << "Circuit Breaker Tripped: " << (sample_res.circuit_broken ? "YES" : "NO (Clean Pass)") << "\n";
        std::cout << "----------------------------------------------------------------------\n";
        std::cout << "Min Latency  : " << std::setw(8) << std::setprecision(2) << min_us << " us (" << (min_us / 1000.0) << " ms)\n";
        std::cout << "Mean Latency : " << std::setw(8) << std::setprecision(2) << mean_us << " us (" << (mean_us / 1000.0) << " ms)\n";
        std::cout << "P50 Latency  : " << std::setw(8) << std::setprecision(2) << p50_us << " us (" << (p50_us / 1000.0) << " ms)\n";
        std::cout << "P90 Latency  : " << std::setw(8) << std::setprecision(2) << p90_us << " us (" << (p90_us / 1000.0) << " ms)\n";
        std::cout << "P99 Latency  : " << std::setw(8) << std::setprecision(2) << p99_us << " us (" << (p99_us / 1000.0) << " ms)\n";
        std::cout << "Max Latency  : " << std::setw(8) << std::setprecision(2) << max_us << " us (" << (max_us / 1000.0) << " ms)\n";
        std::cout << "Throughput   : " << std::setprecision(1) << (1000000.0 / mean_us) << " ticks / sec\n";
        std::cout << "----------------------------------------------------------------------\n";

        if (mean_us < 1000.0) {
            std::cout << "[CERTIFICATION] SUB-MILLISECOND LATENCY TARGET ACHIEVED! (" << std::setprecision(3) << (mean_us / 1000.0) << " ms < 1.000 ms)\n";
        } else {
            std::cout << "[STATUS] Latency: " << (mean_us / 1000.0) << " ms per tick\n";
        }
        std::cout << "======================================================================\n";

    } catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
