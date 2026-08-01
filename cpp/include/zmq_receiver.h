#pragma once
#include <string>
#include <vector>
#include <memory>

// Forward declarations to avoid exposing zmq in header
namespace zmq {
    class context_t;
    class socket_t;
}

class ZmqReceiver {
public:
    ZmqReceiver(const std::string& endpoint);
    ~ZmqReceiver();

    // Blocks until a message is received. Returns false on failure.
    // The message is expected to be a JSON string like {"features": [1.0, 2.0, ...]}
    bool receive(std::vector<float>& features);

private:
    std::unique_ptr<zmq::context_t> context_;
    std::unique_ptr<zmq::socket_t> socket_;
};
