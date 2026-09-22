#include "oanda_stream.h"

#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <zmq.hpp>

#include <iostream>
#include <chrono>
#include <thread>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <stdexcept>

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// libcurl write callback — accumulates response bytes into a std::string
// ---------------------------------------------------------------------------
struct CurlBuffer {
    std::string data;
    // Unprocessed partial line from previous callback invocation
    std::string partial;
    OandaStreamReceiver* receiver{nullptr};
};

static size_t curl_write_cb(char* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* buf = static_cast<CurlBuffer*>(userdata);
    if (!buf->receiver->is_running()) return 0;  // abort transfer

    const size_t bytes = size * nmemb;
    buf->partial.append(ptr, bytes);

    // Process complete newline-delimited SSE lines
    std::string& partial = buf->partial;
    size_t pos = 0;
    while (true) {
        size_t nl = partial.find('\n', pos);
        if (nl == std::string::npos) break;
        std::string line = partial.substr(pos, nl - pos);
        // Strip CR if present (SSE can use \r\n)
        if (!line.empty() && line.back() == '\r') line.pop_back();
        pos = nl + 1;

        if (line.empty()) continue;

        // OANDA v20 streaming sends one JSON object per line
        try {
            json j = json::parse(line);
            const std::string type = j.value("type", "");
            if (type == "PRICE") {
                OandaTick tick;
                // Timestamp: OANDA RFC3339, e.g. "2024-01-15T10:00:00.123456789Z"
                const std::string ts_str = j.value("time", "");
                if (!ts_str.empty()) {
                    // Parse microseconds from RFC3339 string
                    // Use system_clock for portability; strptime on Linux, sscanf fallback
                    struct tm tm_utc{};
                    int micros = 0;
                    int parsed = sscanf(ts_str.c_str(),
                        "%4d-%2d-%2dT%2d:%2d:%2d",
                        &tm_utc.tm_year, &tm_utc.tm_mon, &tm_utc.tm_mday,
                        &tm_utc.tm_hour, &tm_utc.tm_min, &tm_utc.tm_sec);
                    if (parsed == 6) {
                        tm_utc.tm_year -= 1900;
                        tm_utc.tm_mon  -= 1;
                        tm_utc.tm_isdst = 0;
#ifdef _WIN32
                        time_t t = _mkgmtime(&tm_utc);
#else
                        time_t t = timegm(&tm_utc);
#endif
                        if (t != static_cast<time_t>(-1)) {
                            // Extract sub-second part
                            size_t dot = ts_str.find('.');
                            if (dot != std::string::npos) {
                                std::string frac = ts_str.substr(dot + 1);
                                // Remove trailing 'Z' or other suffix
                                while (!frac.empty() && (frac.back() < '0' || frac.back() > '9'))
                                    frac.pop_back();
                                if (!frac.empty()) {
                                    frac.resize(6, '0');
                                    try {
                                        micros = std::stoi(frac.substr(0, 6));
                                    } catch (...) {
                                        micros = 0;
                                    }
                                }
                            }
                            tick.ts_us = static_cast<uint64_t>(t) * 1'000'000ULL
                                       + static_cast<uint64_t>(micros);
                        }
                    }
                }
                if (tick.ts_us == 0) {
                    // Fallback: current system time
                    tick.ts_us = static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::system_clock::now().time_since_epoch()
                        ).count()
                    );
                }

                // Best bid / ask — prefer tradeable bids[]/asks[]; use
                // closeoutBid/Ask ONLY as fallback (they are wider margin-closeout
                // prices, not the real tradeable spread).
                double bid = 0.0, ask = 0.0;
                if (j.contains("bids") && j["bids"].is_array() && !j["bids"].empty())
                    bid = std::stod(j["bids"][0].value("price", "0"));
                if (j.contains("asks") && j["asks"].is_array() && !j["asks"].empty())
                    ask = std::stod(j["asks"][0].value("price", "0"));
                // Fallback only when arrays absent
                if (bid == 0.0 && j.contains("closeoutBid"))
                    bid = std::stod(j.value("closeoutBid", "0"));
                if (ask == 0.0 && j.contains("closeoutAsk"))
                    ask = std::stod(j.value("closeoutAsk", "0"));
                // non-tradeable ticks still published for bar construction

                tick.bid = bid;
                tick.ask = ask;

                const std::string inst = j.value("instrument", "");
                std::strncpy(tick.instrument, inst.c_str(), sizeof(tick.instrument) - 1);
                tick.instrument[sizeof(tick.instrument) - 1] = '\0';

                if (bid > 0.0 && ask > 0.0) {
                    buf->receiver->publish_internal(tick);  // declared friend below
                }
            }
            // heartbeat / other types — ignore
        } catch (const json::exception&) {
            // Malformed line — skip
        }
    }
    partial = partial.substr(pos);
    return bytes;
}

// ---------------------------------------------------------------------------
// OandaStreamReceiver
// ---------------------------------------------------------------------------

OandaStreamReceiver::OandaStreamReceiver(
    std::string token,
    std::string account_id,
    std::string instruments,
    std::string zmq_pub_endpoint,
    bool        practice,
    TickCallback on_tick
) : token_(std::move(token)),
    account_id_(std::move(account_id)),
    instruments_(std::move(instruments)),
    zmq_endpoint_(std::move(zmq_pub_endpoint)),
    practice_(practice),
    on_tick_(std::move(on_tick))
{
    curl_global_init(CURL_GLOBAL_DEFAULT);
    init_zmq();
}

OandaStreamReceiver::~OandaStreamReceiver() {
    stop();
    {
        std::lock_guard<std::mutex> lk(zmq_mutex_);
        if (zmq_pub_) {
            try { zmq_pub_->close(); } catch (...) {}
        }
        if (zmq_ctx_) {
            try { zmq_ctx_->close(); } catch (...) {}
        }
    }
    curl_global_cleanup();
}

void OandaStreamReceiver::init_zmq() {
    std::lock_guard<std::mutex> lk(zmq_mutex_);
    zmq_ctx_ = std::make_unique<zmq::context_t>(1);
    zmq_pub_ = std::make_unique<zmq::socket_t>(*zmq_ctx_, zmq::socket_type::pub);
    zmq_pub_->set(zmq::sockopt::sndhwm, 1000);
    zmq_pub_->set(zmq::sockopt::linger, 0);
    zmq_pub_->bind(zmq_endpoint_);
    std::cout << "[OandaStream] ZMQ PUB bound to " << zmq_endpoint_ << "\n";
}

void OandaStreamReceiver::stop() noexcept {
    running_.store(false, std::memory_order_release);
}

bool OandaStreamReceiver::publish(const OandaTick& tick) {
    std::lock_guard<std::mutex> lk(zmq_mutex_);
    if (!zmq_pub_) return false;
    try {
        // Topic prefix = instrument name so subscribers can filter
        const std::string topic(tick.instrument);
        zmq::message_t topic_msg(topic.data(), topic.size());
        zmq::message_t data_msg(&tick, sizeof(OandaTick));
        zmq_pub_->send(topic_msg, zmq::send_flags::sndmore);
        zmq_pub_->send(data_msg, zmq::send_flags::none);
        return true;
    } catch (...) {
        publish_errors_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
}

// Called from curl_write_cb (friend relationship via publish_internal alias)
void OandaStreamReceiver::publish_internal(const OandaTick& tick) {
    ticks_received_.fetch_add(1, std::memory_order_relaxed);
    publish(tick);
    if (on_tick_) on_tick_(tick);
}

bool OandaStreamReceiver::stream_once() {
    const std::string host = practice_
        ? "https://stream-fxpractice.oanda.com"
        : "https://stream-fxtrade.oanda.com";

    // URL-encode instruments using libcurl so special chars are handled correctly.
    // OANDA accepts both raw commas and %2C — use curl_easy_escape for correctness.
    CURL* curl_enc = curl_easy_init();
    std::string inst_param;
    if (curl_enc) {
        // Encode each instrument separately and join with %2C
        std::string remaining = instruments_;
        bool first = true;
        while (!remaining.empty()) {
            size_t comma = remaining.find(',');
            std::string token = (comma == std::string::npos) ? remaining : remaining.substr(0, comma);
            char* esc = curl_easy_escape(curl_enc, token.c_str(), static_cast<int>(token.size()));
            if (!first) inst_param += "%2C";
            if (esc) { inst_param += esc; curl_free(esc); }
            else      { inst_param += token; }  // fallback: raw
            first = false;
            if (comma == std::string::npos) break;
            remaining = remaining.substr(comma + 1);
        }
        curl_easy_cleanup(curl_enc);
    } else {
        inst_param = instruments_;  // fallback: pass raw (OANDA accepts commas)
    }

    const std::string url = host + "/v3/accounts/" + account_id_
                          + "/pricing/stream?instruments=" + inst_param;

    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "[OandaStream] curl_easy_init() failed\n";
        return false;
    }

    struct curl_slist* headers = nullptr;
    const std::string auth = "Authorization: Bearer " + token_;
    const std::string accept_dt = "Accept-Datetime-Format: RFC3339";
    headers = curl_slist_append(headers, auth.c_str());
    headers = curl_slist_append(headers, accept_dt.c_str());

    CurlBuffer buf;
    buf.receiver = this;

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buf);
    curl_easy_setopt(curl, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_1_1);
    curl_easy_setopt(curl, CURLOPT_TCP_KEEPALIVE, 1L);
    curl_easy_setopt(curl, CURLOPT_TCP_KEEPIDLE, 30L);
    curl_easy_setopt(curl, CURLOPT_TCP_KEEPINTVL, 15L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 15L);
    // No CURLOPT_TIMEOUT — the stream is indefinite
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);

    std::cout << "[OandaStream] Connecting to " << url << "\n";
    CURLcode res = curl_easy_perform(curl);

    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (!running_.load(std::memory_order_relaxed)) {
        return false;  // clean shutdown
    }

    if (res == CURLE_OK || res == CURLE_WRITE_ERROR) {
        // CURLE_WRITE_ERROR is returned when write callback returns 0 (our stop signal)
        if (http_code == 401) {
            std::cerr << "[OandaStream] HTTP 401: invalid token. Stopping.\n";
            stop();
            return false;
        }
        if (http_code == 403) {
            std::cerr << "[OandaStream] HTTP 403: forbidden. Stopping.\n";
            stop();
            return false;
        }
        if (http_code >= 400) {
            std::cerr << "[OandaStream] HTTP " << http_code << " — will retry\n";
            return true;  // retriable
        }
    } else {
        std::cerr << "[OandaStream] curl error: " << curl_easy_strerror(res) << " — will retry\n";
    }
    return true;
}

void OandaStreamReceiver::run() {
    uint32_t backoff_ms = 500;
    while (running_.load(std::memory_order_relaxed)) {
        bool should_retry = stream_once();
        if (!running_.load(std::memory_order_relaxed)) break;
        if (!should_retry) break;

        reconnect_count_.fetch_add(1, std::memory_order_relaxed);
        std::cerr << "[OandaStream] Reconnecting in " << backoff_ms << " ms "
                  << "(attempt #" << reconnect_count_.load() << ")\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
        backoff_ms = std::min(backoff_ms * 2, 30'000u);  // cap at 30 s
    }
    std::cout << "[OandaStream] Stopped. Ticks received: " << ticks_received_.load() << "\n";
}
