#pragma once
#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include <mutex>

namespace zmq {
    class context_t;
    class socket_t;
}

enum class ZmqSocketType {
    Pull,
    Sub
};

enum class ZmqEndpointRole {
    Bind,
    Connect
};

class ZmqReceiver {
public:
    explicit ZmqReceiver(
        const std::string& endpoint,
        ZmqSocketType socket_type = ZmqSocketType::Pull,
        ZmqEndpointRole role = ZmqEndpointRole::Bind,
        int rcv_timeout_ms = 500,
        int rcv_hwm = 10
    );
    ~ZmqReceiver();

    ZmqReceiver(const ZmqReceiver&) = delete;
    ZmqReceiver& operator=(const ZmqReceiver&) = delete;
    ZmqReceiver(ZmqReceiver&& other) noexcept;
    ZmqReceiver& operator=(ZmqReceiver&& other) noexcept;

    // Receives features via either high-throughput binary framing or JSON fallback.
    // Returns true when a valid feature array is received.
    // Returns false on timeout or shutdown, allowing caller to check is_running().
    bool receive(std::vector<float>& features);

    // Signals receiver to terminate blocking loops
    void stop() noexcept;

    bool is_running() const noexcept { return running_.load(std::memory_order_relaxed); }
    const std::string& get_endpoint() const noexcept { return endpoint_; }

private:
    std::string endpoint_;
    ZmqSocketType socket_type_{ZmqSocketType::Pull};
    ZmqEndpointRole role_{ZmqEndpointRole::Bind};
    int rcv_timeout_ms_{500};
    int rcv_hwm_{10};
    std::atomic<bool> running_{true};

    std::unique_ptr<zmq::context_t> context_;
    std::unique_ptr<zmq::socket_t> socket_;
    mutable std::mutex receive_mutex_;
};

