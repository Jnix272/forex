#include <iostream>
#include <vector>
#include <string>
#include "ensemble_runner.h"
#include "zmq_receiver.h"

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <zmq_endpoint> <batch_size> <seq_len> <n_features> <model1.onnx> [model2.onnx ...]\n";
        return 1;
    }

    std::string zmq_endpoint = argv[1];
    int batch_size = std::stoi(argv[2]);
    int seq_len = std::stoi(argv[3]);
    int n_features = std::stoi(argv[4]);

    std::vector<std::string> model_paths;
    for (int i = 5; i < argc; ++i) {
        model_paths.push_back(argv[i]);
    }

    try {
        std::cout << "[Main] Initializing EnsembleRunner with " << model_paths.size() << " models...\n";
        EnsembleRunner ensemble(model_paths, batch_size, seq_len, n_features);

        std::cout << "[Main] Initializing ZMQ Receiver on " << zmq_endpoint << "...\n";
        ZmqReceiver receiver(zmq_endpoint);

        std::cout << "[Main] Entering event loop...\n";
        while (true) {
            std::vector<float> features;
            if (receiver.receive(features)) {
                // Validate size
                if (features.size() != static_cast<size_t>(batch_size * seq_len * n_features)) {
                    std::cerr << "[Main] Warning: Received feature array of size " << features.size() 
                              << " but expected " << (batch_size * seq_len * n_features) << "\n";
                    continue;
                }

                // Run inference
                float prediction = ensemble.infer(features);
                std::cout << "[Main] Final Signal: " << prediction << "\n";
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
