#include "OandaClient.h"
#include <curl/curl.h>
#include <iostream>
#include <stdexcept>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Libcurl callback
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

OandaClient::OandaClient(const std::string& account_id, const std::string& token, bool is_paper)
    : account_id_(account_id), token_(token) {
    if (is_paper) {
        base_url_ = "https://api-fxpractice.oanda.com/v3";
    } else {
        base_url_ = "https://api-fxtrade.oanda.com/v3";
    }
    curl_global_init(CURL_GLOBAL_DEFAULT);
}

OandaClient::~OandaClient() {
    curl_global_cleanup();
}

std::string OandaClient::performGetRequest(const std::string& endpoint) {
    CURL* curl = curl_easy_init();
    std::string readBuffer;
    if (curl) {
        std::string url = base_url_ + endpoint;
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        
        struct curl_slist* headers = NULL;
        headers = curl_slist_append(headers, ("Authorization: Bearer " + token_).c_str());
        headers = curl_slist_append(headers, "Accept-Datetime-Format: UNIX");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        
        CURLcode res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        }
        
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
    }
    return readBuffer;
}

std::string OandaClient::performPostRequest(const std::string& endpoint, const std::string& json_payload) {
    CURL* curl = curl_easy_init();
    std::string readBuffer;
    if (curl) {
        std::string url = base_url_ + endpoint;
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        
        struct curl_slist* headers = NULL;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        headers = curl_slist_append(headers, ("Authorization: Bearer " + token_).c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_payload.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        
        CURLcode res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        }
        
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
    }
    return readBuffer;
}

TickData OandaClient::fetchLatestTick(const std::string& instrument) {
    std::string endpoint = "/accounts/" + account_id_ + "/pricing?instruments=" + instrument;
    std::string response = performGetRequest(endpoint);
    
    try {
        auto j = json::parse(response);
        // Assuming we want the mid price between bid and ask
        double bid = std::stod(j["prices"][0]["bids"][0]["price"].get<std::string>());
        double ask = std::stod(j["prices"][0]["asks"][0]["price"].get<std::string>());
        double price = (bid + ask) / 2.0;

        // Parse UNIX timestamp string (seconds since epoch)
        std::string time_str = j["prices"][0]["time"].get<std::string>();
        double time_sec = std::stod(time_str);
        long long timestamp_ms = static_cast<long long>(time_sec * 1000.0);

        return {price, timestamp_ms};
    } catch (const std::exception& e) {
        std::cerr << "Error parsing JSON in fetchLatestTick: " << e.what() << std::endl;
        return {0.0, 0};
    }
}

bool OandaClient::sendMarketOrder(const std::string& instrument, int units, double stop_loss, double take_profit) {
    std::string endpoint = "/accounts/" + account_id_ + "/orders";
    
    json payload = {
        {"order", {
            {"units", std::to_string(units)},
            {"instrument", instrument},
            {"timeInForce", "FOK"},
            {"type", "MARKET"},
            {"positionFill", "DEFAULT"},
            {"stopLossOnFill", {
                {"price", std::to_string(stop_loss)}
            }},
            {"takeProfitOnFill", {
                {"price", std::to_string(take_profit)}
            }}
        }}
    };
    
    std::string response = performPostRequest(endpoint, payload.dump());
    
    try {
        auto j = json::parse(response);
        if (j.contains("orderFillTransaction")) {
            return true;
        }
        std::cerr << "Order failed: " << response << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Error parsing JSON in sendMarketOrder: " << e.what() << std::endl;
        return false;
    }
}
