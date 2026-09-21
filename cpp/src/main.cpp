#include <iostream>
#include <vector>
#include <string>
#include <atomic>
#include <filesystem>
#include <iomanip>
#include "ensemble_runner.h"
#include "zmq_receiver.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
static std::atomic<bool> g_running{true};

static BOOL WINAPI console_ctrl_handler(DWORD ctrlType) {
    switch (ctrlType) {
        case CTRL_C_EVENT:
        case CTRL_BREAK_EVENT:
        case CTRL_CLOSE_EVENT:
            std::cout << "\n[Signal] Interrupted by user/system. Shutting down...\n";
            g_running.store(false, std::memory_order_relaxed);
            return TRUE;
        default:
            return FALSE;
    }
}

static void install_signal_handlers() {
    SetConsoleCtrlHandler(console_ctrl_handler, TRUE);
}
#else
#include <csignal>
static std::atomic<bool> g_running{true};

static void posix_signal_handler(int sig) {
    std::cout << "\n[Signal] Caught signal (" << sig << "). Shutting down...\n";
    g_running.store(false, std::memory_order_relaxed);
}

static void install_signal_handlers() {
    std::signal(SIGINT, posix_signal_handler);
    std::signal(SIGTERM, posix_signal_handler);
}
#endif

static void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " <zmq_endpoint> <batch_size> <seq_len> <n_features> [options] <model1.onnx> [model2.onnx ...]\n"
              << "\nArguments:\n"
              << "  zmq_endpoint         ZMQ address (e.g. tcp://127.0.0.1:5555 or ipc:///tmp/fx_stream)\n"
              << "  batch_size           Batch dimension (typically 1 for live trading)\n"
              << "  seq_len              Expected sequence length (e.g. 120, 80, 60)\n"
              << "  n_features           Feature dimension per timestep (e.g. 584)\n"
              << "  model.onnx           One or more paths to ONNX model weights\n"
              << "\nOptions:\n"
              << "  --sub                Use ZMQ_SUB socket (default is ZMQ_PULL)\n"
              << "  --connect            Connect socket instead of binding\n"
              << "  --threshold <float>  Variance cutoff threshold for ensemble (default: 0.5)\n"
              << "  --directml           Attempt DirectML GPU execution (if compiled in)\n"
              << "  --help, -h           Display this help documentation\n";
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

    if (argc < 6) {
        std::cerr << "[Main] Error: Insufficient arguments.\n";
        print_usage(argv[0]);
        return 1;
    }

    install_signal_handlers();

    std::string zmq_endpoint;
    int batch_size = 1;
    int seq_len = 60;
    int n_features = 584;
    float max_variance_threshold = 0.5f;
    ZmqSocketType sock_type = ZmqSocketType::Pull;
    ZmqEndpointRole sock_role = ZmqEndpointRole::Bind;
    ONNXRunner::ExecutionProvider ep = ONNXRunner::ExecutionProvider::CPU;

    try {
        zmq_endpoint = argv[1];
        batch_size = std::stoi(argv[2]);
        seq_len = std::stoi(argv[3]);
        n_features = std::stoi(argv[4]);

        if (batch_size <= 0 || seq_len <= 0 || n_features <= 0) {
            std::cerr << "[Main] Error: Dimensions must all be positive integers.\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "[Main] Error parsing dimension arguments: " << e.what() << "\n";
        return 1;
    }

    std::vector<std::string> model_paths;
    for (int i = 5; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--sub") {
            sock_type = ZmqSocketType::Sub;
        } else if (arg == "--connect") {
            sock_role = ZmqEndpointRole::Connect;
        } else if (arg == "--directml") {
            ep = ONNXRunner::ExecutionProvider::DirectML;
        } else if (arg == "--threshold" && (i + 1 < argc)) {
            try {
                max_variance_threshold = std::stof(argv[++i]);
            } catch (const std::exception& e) {
                std::cerr << "[Main] Warning: Invalid threshold value (" << e.what() << "). Using default 0.5\n";
            }
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            // Model path argument
            if (!std::filesystem::exists(arg)) {
                std::cerr << "[Main] Error: Model file not found: " << arg << "\n";
                return 1;
            }
            model_paths.push_back(arg);
        }
    }

    if (model_paths.empty()) {
        std::cerr << "[Main] Error: No valid ONNX model paths provided.\n";
        return 1;
    }

    try {
        std::cout << "[Main] Initializing Ensemble Engine with " << model_paths.size() 
                  << " model(s)... [Threshold: " << max_variance_threshold << "]\n";
        
        // Build model specifications
        std::vector<SubModelConfig> specs;
        specs.reserve(model_paths.size());
        for (size_t i = 0; i < model_paths.size(); ++i) {
            SubModelConfig cfg;
            cfg.name = std::filesystem::path(model_paths[i]).stem().string();
            cfg.path = model_paths[i];
            cfg.seq_len = seq_len;
            cfg.n_features = n_features;
            cfg.weight = 1.0f;
            cfg.ep = ep;
            specs.push_back(cfg);
        }

        EnsembleRunner ensemble(specs, batch_size, max_variance_threshold, true);

        std::cout << "[Main] Initializing ZeroMQ Ingestion on " << zmq_endpoint << "...\n";
        ZmqReceiver receiver(zmq_endpoint, sock_type, sock_role, 500, 10);

        const size_t expected_total_elements = static_cast<size_t>(batch_size) * 
                                              static_cast<size_t>(seq_len) * 
                                              static_cast<size_t>(n_features);

        std::cout << "[Main] Entering real-time inference loop (Expected payload: " 
                  << expected_total_elements << " floats / tick)...\n";
        
        uint64_t tick_count = 0;
        while (g_running.load(std::memory_order_relaxed)) {
            std::vector<float> features;
            if (receiver.receive(features)) {
                if (!g_running.load(std::memory_order_relaxed)) break;

                tick_count++;

                // Validate payload bounds
                if (features.size() != expected_total_elements) {
                    std::cerr << "[Main] Warning: Tick " << tick_count 
                              << " feature size (" << features.size() 
                              << ") != expected (" << expected_total_elements << "). Skipping.\n";
                    continue;
                }

                // Execute ensemble inference
                EnsembleResult res = ensemble.infer_detailed(features);
                
                std::cout << "[Main] Tick #" << tick_count 
                          << " | Signal: " << std::fixed << std::setprecision(4) << res.final_signal 
                          << " | Mean: " << res.weighted_mean 
                          << " | Var: " << res.weighted_variance 
                          << " | Agreement: " << std::setprecision(1) << (res.agreement_ratio * 100.0f) << "%"
                          << (res.circuit_broken ? " [CIRCUIT BREAKER ACTIVATED]" : "")
                          << "\n";
            }
        }

        std::cout << "[Main] Loop exited. Cleaning up resources...\n";
        receiver.stop();

    } catch (const std::exception& e) {
        std::cerr << "[Main] Fatal Error: " << e.what() << "\n";
        return 1;
    }

    std::cout << "[Main] Shutdown complete. Goodbye.\n";
    return 0;
}

