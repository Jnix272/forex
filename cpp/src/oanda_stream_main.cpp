#include "oanda_stream.h"
#include <iostream>
#include <string>
#include <cstdlib>
#include <csignal>
#include <atomic>

static std::atomic<OandaStreamReceiver*> g_receiver{nullptr};

static void on_signal(int) {
    OandaStreamReceiver* r = g_receiver.load();
    if (r) r->stop();
}

int main(int argc, char** argv) {
    // Usage: oanda_stream [--practice|--live] [--instruments EUR_USD,GBP_USD]
    //                     [--zmq tcp://127.0.0.1:5557]
    // All settings also readable from environment variables.

    bool        practice    = true;
    std::string instruments = "EUR_USD";
    std::string zmq_ep      = "tcp://127.0.0.1:5557";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--live")    { practice = false; }
        else if (arg == "--practice") { practice = true; }
        else if ((arg == "--instruments" || arg == "-i") && i + 1 < argc) {
            instruments = argv[++i];
        } else if ((arg == "--zmq" || arg == "-z") && i + 1 < argc) {
            zmq_ep = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout <<
                "Usage: oanda_stream [--live|--practice] "
                "[--instruments EUR_USD,GBP_USD] "
                "[--zmq tcp://127.0.0.1:5557]\n"
                "Env vars: OANDA_API_KEY (or OANDA_BEARER_TOKEN), OANDA_ACCOUNT_ID, "
                "OANDA_ENV (live|practice), OANDA_INSTRUMENTS, OANDA_ZMQ_ENDPOINT\n";
            return 0;
        }
    }

    // Environment variable overrides
    auto env = [](const char* k, const char* def) -> std::string {
        const char* v = std::getenv(k);
        return v ? v : def;
    };

    const std::string token = env("OANDA_BEARER_TOKEN",
                               env("OANDA_API_TOKEN",
                               env("OANDA_API_KEY", "")));
    const std::string account_id = env("OANDA_ACCOUNT_ID", "");
    const std::string oanda_env  = env("OANDA_ENV", practice ? "practice" : "live");
    if (oanda_env == "live" || oanda_env == "prod" || oanda_env == "production")
        practice = false;

    instruments = env("OANDA_INSTRUMENTS", instruments.c_str());
    zmq_ep      = env("OANDA_ZMQ_ENDPOINT", zmq_ep.c_str());

    if (token.empty()) {
        std::cerr << "[oanda_stream] ERROR: no API token found. "
                     "Set OANDA_API_KEY or OANDA_BEARER_TOKEN.\n";
        return 1;
    }
    if (account_id.empty()) {
        std::cerr << "[oanda_stream] ERROR: no account ID found. "
                     "Set OANDA_ACCOUNT_ID.\n";
        return 1;
    }

    std::cout << "[oanda_stream] Starting — env=" << (practice ? "practice" : "LIVE")
              << " instruments=" << instruments
              << " zmq=" << zmq_ep << "\n";

    // Log every 10,000th tick to stdout so the operator can see it's alive
    uint64_t log_every = 10'000;
    auto on_tick = [&log_every](const OandaTick& t) {
        static uint64_t count = 0;
        if (++count % log_every == 0) {
            std::cout << "[oanda_stream] tick #" << count
                      << " " << t.instrument
                      << " bid=" << t.bid << " ask=" << t.ask << "\n";
        }
    };

    OandaStreamReceiver receiver(token, account_id, instruments, zmq_ep, practice, on_tick);
    g_receiver.store(&receiver);

    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    receiver.run();
    return 0;
}
