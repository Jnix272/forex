#pragma once
#include <string>
#include <atomic>
#include <mutex>
#include <memory>
#include <functional>

namespace zmq {
    class context_t;
    class socket_t;
}

// Tick published on the ZMQ PUB socket.
// Wire format (binary, little-endian):
//   uint32_t magic  = 0x4F414E44  ("OAND")
//   uint64_t ts_us  = microseconds since Unix epoch
//   double   bid
//   double   ask
//   char     instrument[16]  (null-padded, e.g. "EUR_USD\0\0...")
// Total: 4 + 8 + 8 + 8 + 16 = 44 bytes
struct OandaTick {
    static constexpr uint32_t MAGIC = 0x4F414E44u;  // "OAND"
    uint32_t magic{MAGIC};
    uint64_t ts_us{0};
    double   bid{0.0};
    double   ask{0.0};
    char     instrument[16]{};
};
static_assert(sizeof(OandaTick) == 44, "OandaTick layout changed");

// Callback invoked on each parsed tick (used internally and for testing).
using TickCallback = std::function<void(const OandaTick&)>;

class OandaStreamReceiver {
public:
    OandaStreamReceiver(
        std::string token,
        std::string account_id,
        std::string instruments,          // comma-separated: "EUR_USD,GBP_USD"
        std::string zmq_pub_endpoint,     // e.g. "tcp://127.0.0.1:5557"
        bool        practice = true,
        TickCallback on_tick = nullptr    // optional; fires after ZMQ publish
    );
    ~OandaStreamReceiver();

    OandaStreamReceiver(const OandaStreamReceiver&) = delete;
    OandaStreamReceiver& operator=(const OandaStreamReceiver&) = delete;

    // Blocking: opens the SSE stream and publishes ticks until stop() is called.
    // Reconnects automatically on disconnect with exponential back-off.
    void run();

    void stop() noexcept;
    bool is_running() const noexcept { return running_.load(std::memory_order_relaxed); }

    // Stats (thread-safe reads)
    uint64_t ticks_received()    const noexcept { return ticks_received_.load(); }
    uint64_t publish_errors()    const noexcept { return publish_errors_.load(); }
    uint64_t reconnect_count()   const noexcept { return reconnect_count_.load(); }

    // Called from the libcurl write callback (C linkage — must be public)
    void publish_internal(const OandaTick& tick);

private:
    bool stream_once();                          // one HTTP session, returns false to stop
    bool publish(const OandaTick& tick);
    void init_zmq();

    std::string token_;
    std::string account_id_;
    std::string instruments_;
    std::string zmq_endpoint_;
    bool        practice_;
    TickCallback on_tick_;

    std::atomic<bool>     running_{true};
    std::atomic<uint64_t> ticks_received_{0};
    std::atomic<uint64_t> publish_errors_{0};
    std::atomic<uint64_t> reconnect_count_{0};

    std::unique_ptr<zmq::context_t> zmq_ctx_;
    std::unique_ptr<zmq::socket_t>  zmq_pub_;
    mutable std::mutex              zmq_mutex_;
};
