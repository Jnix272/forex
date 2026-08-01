#include "zmq_receiver.h"
#include <zmq.hpp>
#include <nlohmann/json.hpp>
#include <iostream>
#include <stdexcept>

using json = nlohmann::json;

ZmqReceiver::ZmqReceiver(const std::string& endpoint) {
    context_ = std::make_unique<zmq::context_t>(1);
    socket_ = std::make_unique<zmq::socket_t>(*context_, zmq::socket_type::pull);
    try {
        socket_->bind(endpoint);
        std::cout << "[ZmqReceiver] Bound to " << endpoint << std::endl;
    } catch (const zmq::error_t& e) {
        std::cerr << "[ZmqReceiver] Failed to bind to " << endpoint << ": " << e.what() << std::endl;
        throw;
    }
}

ZmqReceiver::~ZmqReceiver() {
    socket_->close();
    context_->close();
}

bool ZmqReceiver::receive(std::vector<float>& features) {
    zmq::message_t request;
    
    // Receive message (blocking)
    auto recv_res = socket_->recv(request, zmq::recv_flags::none);
    if (!recv_res) {
        return false;
    }

    try {
        std::string payload(static_cast<char*>(request.data()), request.size());
        json j = json::parse(payload);
        
        if (j.contains("features") && j["features"].is_array()) {
            features = j["features"].get<std::vector<float>>();
            return true;
        } else {
            std::cerr << "[ZmqReceiver] Invalid payload: missing 'features' array" << std::endl;
            return false;
        }
    } catch (const std::exception& e) {
        std::cerr << "[ZmqReceiver] JSON parse error: " << e.what() << std::endl;
        return false;
    }
}
