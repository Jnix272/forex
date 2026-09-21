#include "zmq_receiver.h"
#include <zmq.hpp>
#include <nlohmann/json.hpp>
#include <iostream>
#include <cstring>
#include <stdexcept>

using json = nlohmann::json;

// Binary Framing Magic constant ("FXST" = 0x54535846 in little-endian)
static constexpr uint32_t FX_BINARY_MAGIC = 0x54535846;

ZmqReceiver::ZmqReceiver(
    const std::string& endpoint,
    ZmqSocketType socket_type,
    ZmqEndpointRole role,
    int rcv_timeout_ms,
    int rcv_hwm
) : endpoint_(endpoint),
    socket_type_(socket_type),
    role_(role),
    rcv_timeout_ms_(rcv_timeout_ms > 0 ? rcv_timeout_ms : 500),
    rcv_hwm_(rcv_hwm > 0 ? rcv_hwm : 10) {

    context_ = std::make_unique<zmq::context_t>(1);

    auto z_type = (socket_type_ == ZmqSocketType::Sub) 
        ? zmq::socket_type::sub 
        : zmq::socket_type::pull;

    socket_ = std::make_unique<zmq::socket_t>(*context_, z_type);

    // Apply low-latency socket configuration
    try {
        socket_->set(zmq::sockopt::rcvhwm, rcv_hwm_);
        socket_->set(zmq::sockopt::linger, 0);
        socket_->set(zmq::sockopt::rcvtimeo, rcv_timeout_ms_);

        if (socket_type_ == ZmqSocketType::Sub) {
            // Subscribe to all topics by default
            socket_->set(zmq::sockopt::subscribe, "");
        }

        if (role_ == ZmqEndpointRole::Bind) {
            socket_->bind(endpoint_);
            std::cout << "[ZmqReceiver] Bound to " << endpoint_ << "\n";
        } else {
            socket_->connect(endpoint_);
            std::cout << "[ZmqReceiver] Connected to " << endpoint_ << "\n";
        }
    } catch (const zmq::error_t& e) {
        std::cerr << "[ZmqReceiver] Socket setup failed on " << endpoint_ << ": " << e.what() << "\n";
        throw;
    }
}

ZmqReceiver::~ZmqReceiver() {
    stop();
    if (socket_) {
        try {
            socket_->set(zmq::sockopt::linger, 0);
            socket_->close();
        } catch (...) {}
    }
    if (context_) {
        try {
            context_->close();
        } catch (...) {}
    }
}

ZmqReceiver::ZmqReceiver(ZmqReceiver&& other) noexcept {
    // Lock other.receive_mutex_ so a concurrent receive() on other cannot read
    // other.socket_ while we move it out (data race / use-after-move).
    other.running_.store(false, std::memory_order_release);
    std::lock_guard<std::mutex> lock(other.receive_mutex_);
    endpoint_       = std::move(other.endpoint_);
    socket_type_    = other.socket_type_;
    role_           = other.role_;
    rcv_timeout_ms_ = other.rcv_timeout_ms_;
    rcv_hwm_        = other.rcv_hwm_;
    running_.store(false, std::memory_order_relaxed);  // source was stopped above
    context_        = std::move(other.context_);
    socket_         = std::move(other.socket_);
}

ZmqReceiver& ZmqReceiver::operator=(ZmqReceiver&& other) noexcept {
    if (this != &other) {
        stop();
        endpoint_ = std::move(other.endpoint_);
        socket_type_ = other.socket_type_;
        role_ = other.role_;
        rcv_timeout_ms_ = other.rcv_timeout_ms_;
        rcv_hwm_ = other.rcv_hwm_;
        running_.store(other.running_.load());
        context_ = std::move(other.context_);
        socket_ = std::move(other.socket_);
    }
    return *this;
}

void ZmqReceiver::stop() noexcept {
    running_.store(false, std::memory_order_release);
}

bool ZmqReceiver::receive(std::vector<float>& features) {
    if (!running_.load(std::memory_order_relaxed)) {
        return false;
    }

    std::lock_guard<std::mutex> lock(receive_mutex_);
    if (!running_.load(std::memory_order_relaxed) || !socket_) {
        return false;
    }

    zmq::message_t request;
    try {
        // Receive message with configured timeout
        auto recv_res = socket_->recv(request, zmq::recv_flags::none);
        if (!recv_res || request.size() == 0) {
            // Timeout (EAGAIN) or zero-length message
            return false;
        }
    } catch (const zmq::error_t& e) {
        if (e.num() == EAGAIN || e.num() == EINTR || e.num() == ETERM) {
            return false;
        }
        std::cerr << "[ZmqReceiver] recv error: " << e.what() << "\n";
        return false;
    } catch (...) {
        return false;
    }

    const uint8_t* raw_bytes = static_cast<const uint8_t*>(request.data());
    const size_t msg_size = request.size();

    // Mode 1: Binary framed protocol with magic header [uint32_t magic, uint32_t count, float data...]
    if (msg_size >= 8) {
        uint32_t magic = 0;
        std::memcpy(&magic, raw_bytes, sizeof(uint32_t));
        if (magic == FX_BINARY_MAGIC) {
            uint32_t count = 0;
            std::memcpy(&count, raw_bytes + sizeof(uint32_t), sizeof(uint32_t));
            // Guard against network-controlled count causing OOM or size_t overflow.
            // 16 M floats = 64 MB is a safe upper bound for any feature vector.
            constexpr uint32_t MAX_FLOAT_COUNT = 1u << 24;
            if (count == 0 || count > MAX_FLOAT_COUNT) {
                std::cerr << "[ZmqReceiver] Warning: Rejecting frame with out-of-range count=" << count << "\n";
                return false;
            }
            const size_t expected_payload_bytes = static_cast<size_t>(count) * sizeof(float);
            if (msg_size == 8 + expected_payload_bytes) {
                features.resize(count);
                std::memcpy(features.data(), raw_bytes + 8, expected_payload_bytes);
                return true;
            }
            // Magic matched but payload size is wrong — this is a corrupt framed packet,
            // not a raw binary payload. Do not fall through to Mode 3.
            std::cerr << "[ZmqReceiver] Warning: FX_BINARY_MAGIC matched but payload size mismatch "
                      << "(msg=" << msg_size << ", expected=" << (8 + expected_payload_bytes)
                      << " for count=" << count << "). Dropping packet.\n";
            return false;
        }
    }

    // Mode 2: JSON payload fallback (supports {"features": [...]} and top-level array [...])
    // Identify text/JSON by checking first non-whitespace character
    size_t first_non_ws = 0;
    while (first_non_ws < msg_size && (raw_bytes[first_non_ws] == ' ' || raw_bytes[first_non_ws] == '\t' ||
                                        raw_bytes[first_non_ws] == '\r' || raw_bytes[first_non_ws] == '\n')) {
        ++first_non_ws;
    }

    if (first_non_ws < msg_size && (raw_bytes[first_non_ws] == '{' || raw_bytes[first_non_ws] == '[')) {
        try {
            const char* text_data = reinterpret_cast<const char*>(raw_bytes);
            json j = json::parse(text_data, text_data + msg_size);
            if (j.contains("features") && j["features"].is_array()) {
                features = j["features"].get<std::vector<float>>();
                return true;
            } else if (j.is_array()) {
                features = j.get<std::vector<float>>();
                return true;
            } else {
                std::cerr << "[ZmqReceiver] Warning: JSON payload missing 'features' array\n";
            }
        } catch (...) {
            // Not valid JSON; fall through to raw binary check below
        }
    }

    // Mode 3: Raw IEEE-754 binary float32 payload (fast zero-copy ingestion)
    if (msg_size >= sizeof(float) && (msg_size % sizeof(float) == 0)) {
        size_t count = msg_size / sizeof(float);
        features.resize(count);
        std::memcpy(features.data(), raw_bytes, msg_size);
        return true;
    }

    std::cerr << "[ZmqReceiver] Warning: Received payload of size " << msg_size 
              << " bytes does not match any supported framing format.\n";
    return false;
}

