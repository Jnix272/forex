#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <memory>
#include <cmath>
#include <nlohmann/json.hpp>
#include "OandaClient.h"
#include "RingBuffer.h"
#include "ModelRunner.h"
#include "RiskManager.h"

using json = nlohmann::json;

namespace {
std::string getEnvOrDefault(const char* name, const std::string& fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || std::string(value).empty()) {
        return fallback;
    }
    return value;
}

std::string getRequiredEnv(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || std::string(value).empty()) {
        throw std::runtime_error(std::string("Missing required environment variable: ") + name);
    }
    return value;
}

bool getBoolEnv(const char* name, bool fallback) {
    std::string value = getEnvOrDefault(name, fallback ? "1" : "0");
    for (char& ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return value == "1" || value == "true" || value == "yes" || value == "on";
}

int getIntEnv(const char* name, int fallback) {
    std::string value = getEnvOrDefault(name, "");
    return value.empty() ? fallback : std::stoi(value);
}

double getDoubleEnv(const char* name, double fallback) {
    std::string value = getEnvOrDefault(name, "");
    return value.empty() ? fallback : std::stod(value);
}

std::string resolveModelPath(const std::string& configured_path) {
    std::filesystem::path model_path(configured_path);
    if (std::filesystem::exists(model_path)) {
        return model_path.string();
    }
    if (model_path.is_relative()) {
        std::filesystem::path source_root = std::filesystem::path(__FILE__).parent_path().parent_path();
        std::filesystem::path source_relative = source_root / model_path;
        if (std::filesystem::exists(source_relative)) {
            return source_relative.string();
        }
    }
    return configured_path;
}

std::string signalLabel(int signal) {
    if (signal > 0) return "LONG";
    if (signal < 0) return "SHORT";
    return "HOLD";
}

std::string executionActionLabel(int action) {
    switch (action) {
        case 0: return "HOLD";
        case 1: return "OPEN_LONG";
        case 2: return "OPEN_SHORT";
        case 3: return "SCALE_IN_25";
        case 4: return "SCALE_IN_50";
        case 5: return "SCALE_IN_100";
        case 6: return "SCALE_OUT_25";
        case 7: return "SCALE_OUT_50";
        case 8: return "SCALE_OUT_100";
        case 9: return "CLOSE_ALL";
        default: return "UNKNOWN";
    }
}

double actionFraction(int action) {
    switch (action) {
        case 3:
        case 6:
            return 0.25;
        case 4:
        case 7:
            return 0.50;
        case 5:
        case 8:
        case 9:
            return 1.00;
        default:
            return 1.00;
    }
}

int executionActionToUnits(int action, int direction_signal, int current_units, int base_units) {
    if (action == 1) {
        if (current_units > 0) return 0;
        if (current_units < 0) return std::abs(current_units) + base_units;
        return base_units;
    }
    if (action == 2) {
        if (current_units < 0) return 0;
        if (current_units > 0) return -current_units - base_units;
        return -base_units;
    }
    if (action >= 3 && action <= 5) {
        int side = current_units > 0 ? 1 : (current_units < 0 ? -1 : direction_signal);
        if (side == 0) return 0;
        return static_cast<int>(std::round(side * base_units * actionFraction(action)));
    }
    if (action >= 6 && action <= 9) {
        if (current_units == 0) return 0;
        return static_cast<int>(std::round(-current_units * actionFraction(action)));
    }
    return 0;
}

std::vector<float> buildAgentState(
    int position_units,
    double entry_price,
    double equity,
    double initial_equity,
    int holding_bars,
    double current_price,
    int max_position_units
) {
    double position_norm = max_position_units > 0
        ? static_cast<double>(position_units) / static_cast<double>(max_position_units)
        : 0.0;
    position_norm = std::max(-1.0, std::min(1.0, position_norm));

    double upnl = 0.0;
    if (position_units != 0 && entry_price > 0.0 && current_price > 0.0) {
        upnl = (current_price - entry_price) * static_cast<double>(position_units);
    }
    double upnl_norm = initial_equity > 0.0 ? upnl / initial_equity : 0.0;
    upnl_norm = std::max(-0.5, std::min(0.5, upnl_norm));

    double equity_norm = initial_equity > 0.0 ? (equity - initial_equity) / initial_equity : 0.0;
    equity_norm = std::max(-0.5, std::min(0.5, equity_norm));

    return {
        static_cast<float>(position_norm),
        static_cast<float>(upnl_norm),
        static_cast<float>(std::min(static_cast<double>(holding_bars) / 100.0, 1.0)),
        static_cast<float>(equity_norm),
        position_units != 0 ? 1.0f : 0.0f,
    };
}

uint64_t hashFeatureVector(const std::vector<float>& features) {
    const auto* bytes = reinterpret_cast<const unsigned char*>(features.data());
    size_t byte_count = features.size() * sizeof(float);
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < byte_count; ++i) {
        hash ^= static_cast<uint64_t>(bytes[i]);
        hash *= 1099511628211ull;
    }
    return hash;
}

std::vector<float> buildFeatureVector(const RingBuffer& buffer, size_t seq_len, int num_features) {
    std::vector<float> features(seq_len * static_cast<size_t>(num_features), 0.0f);

    for (size_t i = 0; i < seq_len; ++i) {
        size_t offset = i * static_cast<size_t>(num_features);
        if (num_features > 0) {
            features[offset] = static_cast<float>(buffer.getPriceAt(seq_len - 1 - i));
        }
        if (num_features > 1) {
            features[offset + 1] = static_cast<float>(buffer.getSMA(14));
        }
        if (num_features > 2) {
            features[offset + 2] = static_cast<float>(buffer.getEMA(14, buffer.getPriceAt(0)));
        }
    }

    return features;
}

void writeShadowJournal(
    std::ofstream& journal,
    const std::string& instrument,
    const TickData& tick,
    int signal,
    int units,
    double stop_loss,
    double take_profit,
    int execution_action,
    long long inference_latency_us,
    long long total_latency_us,
    uint64_t feature_hash
) {
    if (!journal.is_open()) {
        return;
    }

    json event = {
        {"instrument", instrument},
        {"timestamp_ms", tick.timestamp_ms},
        {"price", tick.price},
        {"signal", signalLabel(signal)},
        {"units", units},
        {"stop_loss", stop_loss},
        {"take_profit", take_profit},
        {"execution_action", execution_action >= 0 ? executionActionLabel(execution_action) : "none"},
        {"inference_latency_us", inference_latency_us},
        {"tick_to_trade_latency_us", total_latency_us},
        {"feature_hash", feature_hash}
    };
    journal << event.dump() << '\n';
    journal.flush();
}
}

int main() {
    try {
        std::cout << "Starting Forex C++ HFT Server..." << std::endl;

        std::string account_id = getRequiredEnv("OANDA_ACCOUNT_ID");
        std::string token = getRequiredEnv("OANDA_BEARER_TOKEN");
        std::string instrument = getEnvOrDefault("OANDA_INSTRUMENT", "EUR_USD");
        std::string model_path = resolveModelPath(getEnvOrDefault("MODEL_PATH", "models/haelt.onnx"));
        std::string execution_model_cfg = getEnvOrDefault("EXECUTION_MODEL_PATH", "");
        std::string execution_model_path = execution_model_cfg.empty() ? "" : resolveModelPath(execution_model_cfg);
        bool paper_trading = getBoolEnv("OANDA_PAPER", true);
        bool shadow_mode = getBoolEnv("SHADOW_MODE", true);
        bool allow_partial_features = getBoolEnv("ALLOW_PARTIAL_FEATURES", false);
        int base_trade_units = getIntEnv("TRADE_UNITS", 10000);

        if (!std::filesystem::exists(model_path)) {
            throw std::runtime_error("ONNX model file not found: " + model_path);
        }
        if (!execution_model_path.empty() && !std::filesystem::exists(execution_model_path)) {
            throw std::runtime_error("Execution ONNX model file not found: " + execution_model_path);
        }

        OandaClient oanda(account_id, token, paper_trading);
        ModelRunner runner(model_path);
        std::unique_ptr<ModelRunner> execution_runner;
        if (!execution_model_path.empty()) {
            execution_runner = std::make_unique<ModelRunner>(execution_model_path);
            if (!execution_runner->hasAgentStateInput()) {
                throw std::runtime_error("EXECUTION_MODEL_PATH must point to a two-input RL execution ONNX model");
            }
            if (execution_runner->seqLen() != runner.seqLen() ||
                execution_runner->numFeatures() != runner.numFeatures()) {
                throw std::runtime_error("Direction and execution ONNX models must use the same seq_len and feature count");
            }
        }

        size_t seq_len = static_cast<size_t>(runner.seqLen());
        int num_features = runner.numFeatures();
        if (num_features > 3 && !allow_partial_features) {
            throw std::runtime_error(
                "The loaded model expects more than the 3 prototype C++ features. "
                "Implement the full feature pipeline or set ALLOW_PARTIAL_FEATURES=1 for shadow-only experiments."
            );
        }

        RingBuffer buffer(seq_len);
        RiskManager risk_manager(
            getIntEnv("MAX_POSITION_UNITS", 100000),
            getIntEnv("MAX_STALE_DATA_MS", 5000),
            getDoubleEnv("MAX_DRAWDOWN", 500.0)
        );
        int max_position_units = getIntEnv("MAX_POSITION_UNITS", 100000);
        double initial_equity = getDoubleEnv("INITIAL_EQUITY", 10000.0);
        double tracked_equity = initial_equity;
        int tracked_position_units = 0;
        int tracked_holding_bars = 0;
        double tracked_entry_price = 0.0;

        std::ofstream shadow_journal;
        if (shadow_mode) {
            std::filesystem::path journal_path = getEnvOrDefault("SHADOW_JOURNAL", "logs/cpp_shadow_journal.jsonl");
            if (journal_path.has_parent_path()) {
                std::filesystem::create_directories(journal_path.parent_path());
            }
            shadow_journal.open(journal_path, std::ios::app);
            if (!shadow_journal.is_open()) {
                throw std::runtime_error("Failed to open shadow journal: " + journal_path.string());
            }
        }

        std::cout << "[CONFIG] instrument=" << instrument
                  << " paper=" << (paper_trading ? "true" : "false")
                  << " shadow=" << (shadow_mode ? "true" : "false")
                  << " model=" << model_path
                  << " execution_model=" << (execution_model_path.empty() ? "none" : execution_model_path)
                  << std::endl;
        std::cout << "Waiting to fill RingBuffer..." << std::endl;

        while (true) {
            auto tick_start_time = std::chrono::high_resolution_clock::now();
            TickData tick = oanda.fetchLatestTick(instrument);
            if (tick.price > 0.0) {
                buffer.addTick(tick.price);
            }

            if (buffer.isFull()) {
                risk_manager.checkStaleData(tick.timestamp_ms);
                risk_manager.checkDrawdown();

                std::vector<float> features = buildFeatureVector(buffer, seq_len, num_features);
                uint64_t feature_hash = hashFeatureVector(features);

                auto inference_start = std::chrono::high_resolution_clock::now();
                int direction_signal = runner.predict(features, static_cast<int>(seq_len), num_features);
                int signal = direction_signal;
                int execution_action = -1;
                if (execution_runner) {
                    std::vector<float> agent_state = buildAgentState(
                        tracked_position_units,
                        tracked_entry_price,
                        tracked_equity,
                        initial_equity,
                        tracked_holding_bars,
                        tick.price,
                        max_position_units
                    );
                    execution_action = execution_runner->predictExecution(
                        features,
                        static_cast<int>(seq_len),
                        num_features,
                        agent_state
                    );
                    if (execution_action == 1) signal = 1;
                    else if (execution_action == 2) signal = -1;
                    else if (execution_action >= 3 && execution_action <= 5) {
                        signal = tracked_position_units > 0 ? 1 : (tracked_position_units < 0 ? -1 : direction_signal);
                    } else if (execution_action >= 6 && execution_action <= 9) {
                        signal = 0;
                    } else {
                        signal = 0;
                    }
                }
                auto inference_end = std::chrono::high_resolution_clock::now();

                int units = 0;
                double tp = 0.0;
                double sl = 0.0;
                double realized_pnl = 0.0;
                if (execution_runner && execution_action >= 0) {
                    units = executionActionToUnits(
                        execution_action,
                        direction_signal,
                        tracked_position_units,
                        base_trade_units
                    );
                } else if (signal != 0) {
                    units = signal > 0 ? base_trade_units : -base_trade_units;
                }

                if (units != 0) {
                    if (units > 0 && tracked_position_units + units > max_position_units) {
                        units = max_position_units - tracked_position_units;
                    } else if (units < 0 && tracked_position_units + units < -max_position_units) {
                        units = -max_position_units - tracked_position_units;
                    }
                }

                if (units != 0) {
                    risk_manager.checkFatFinger(tracked_position_units + units);
                    int order_side = units > 0 ? 1 : -1;
                    tp = order_side > 0 ? tick.price + 0.0015 : tick.price - 0.0015;
                    sl = order_side > 0 ? tick.price - 0.0008 : tick.price + 0.0008;
                }

                auto trade_end = std::chrono::high_resolution_clock::now();
                auto total_latency = std::chrono::duration_cast<std::chrono::microseconds>(trade_end - tick_start_time).count();
                auto inf_latency = std::chrono::duration_cast<std::chrono::microseconds>(inference_end - inference_start).count();

                if (units != 0) {
                    std::cout << signalLabel(units > 0 ? 1 : -1) << " SIGNAL FIRED!";
                    if (execution_action >= 0) {
                        std::cout << " execution=" << executionActionLabel(execution_action);
                    }
                    std::cout << std::endl;
                    if (shadow_mode) {
                        std::cout << "[SHADOW MODE] Logged intended " << signalLabel(units > 0 ? 1 : -1)
                                  << " order of " << units << " units at " << tick.price << std::endl;
                    } else {
                        std::cout << "Submitting MARKET order..." << std::endl;
                        oanda.sendMarketOrder(instrument, units, sl, tp);
                    }
                    if (tracked_position_units != 0 && ((tracked_position_units > 0 && units < 0) || (tracked_position_units < 0 && units > 0))) {
                        int closed_units = std::min(std::abs(tracked_position_units), std::abs(units));
                        double pnl_per_unit = (tick.price - tracked_entry_price) * (tracked_position_units > 0 ? 1.0 : -1.0);
                        realized_pnl = closed_units * pnl_per_unit;
                        tracked_equity += realized_pnl;
                        risk_manager.updateDrawdown(realized_pnl);
                    }

                    int previous_position = tracked_position_units;
                    int new_position = tracked_position_units + units;
                    bool is_reversal = (previous_position > 0 && new_position < 0) || 
                                       (previous_position < 0 && new_position > 0);
                    bool is_same_side_add = previous_position != 0 &&
                                            ((previous_position > 0 && units > 0) ||
                                             (previous_position < 0 && units < 0));
                    
                    if (previous_position == 0 || is_reversal) {
                        tracked_entry_price = tick.price;
                        tracked_holding_bars = 0;
                    } else if (is_same_side_add && std::abs(new_position) > std::abs(previous_position)) {
                        double prev_abs = static_cast<double>(std::abs(previous_position));
                        double add_abs = static_cast<double>(std::abs(units));
                        tracked_entry_price = (
                            tracked_entry_price * prev_abs + tick.price * add_abs
                        ) / std::max(prev_abs + add_abs, 1.0);
                    }
                    tracked_position_units = new_position;
                    if (tracked_position_units == 0) {
                        tracked_entry_price = 0.0;
                        tracked_holding_bars = 0;
                    }
                    std::cout << "[LATENCY] Inference: " << inf_latency << " us | Tick-to-Trade: " << total_latency << " us" << std::endl;
                } else if (tracked_position_units != 0) {
                    tracked_holding_bars += 1;
                }

                if (shadow_mode) {
                    int journal_signal = units > 0 ? 1 : (units < 0 ? -1 : signal);
                    writeShadowJournal(
                        shadow_journal,
                        instrument,
                        tick,
                        journal_signal,
                        units,
                        sl,
                        tp,
                        execution_action,
                        inf_latency,
                        total_latency,
                        feature_hash
                    );
                }
            }

            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    } catch (const std::exception& e) {
        std::cerr << "CRITICAL ERROR: " << e.what() << std::endl;
        std::cerr << "Shutting down trading engine immediately!" << std::endl;
        return 1;
    }

    return 0;
}
