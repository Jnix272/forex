#pragma once

#include <string>

struct TickData {
    double price;
    long long timestamp_ms; // Unix epoch in milliseconds
};

class OandaClient {
public:
    OandaClient(const std::string& account_id, const std::string& token, bool is_paper = true);
    ~OandaClient();

    // Prevent copy
    OandaClient(const OandaClient&) = delete;
    OandaClient& operator=(const OandaClient&) = delete;

    // Fetch the latest tick for a given instrument (e.g., "EUR_USD")
    TickData fetchLatestTick(const std::string& instrument);

    // Send a Market Order
    bool sendMarketOrder(const std::string& instrument, int units, double stop_loss, double take_profit);

private:
    std::string account_id_;
    std::string token_;
    std::string base_url_;
    
    // Internal helper for cURL
    std::string performGetRequest(const std::string& endpoint);
    std::string performPostRequest(const std::string& endpoint, const std::string& json_payload);
};
