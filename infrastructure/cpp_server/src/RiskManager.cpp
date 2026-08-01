#include "RiskManager.h"
#include <chrono>
#include <cmath>

RiskManager::RiskManager(int max_position_size, long max_stale_data_ms, double max_drawdown)
    : max_position_size_(max_position_size),
      max_stale_data_ms_(max_stale_data_ms),
      max_drawdown_(max_drawdown),
      current_drawdown_(0.0) {}

void RiskManager::checkFatFinger(int units) const {
    if (std::abs(units) > max_position_size_) {
        throw std::runtime_error("FAT FINGER CIRCUIT BREAKER: Order units (" + 
                                 std::to_string(units) + ") exceeded hard limit (" + 
                                 std::to_string(max_position_size_) + ")");
    }
}

void RiskManager::checkStaleData(long long last_tick_timestamp_ms) const {
    auto now = std::chrono::system_clock::now();
    long long current_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    now.time_since_epoch()).count();

    if ((current_time_ms - last_tick_timestamp_ms) > max_stale_data_ms_) {
        throw std::runtime_error("STALE DATA CIRCUIT BREAKER: Last tick is older than " + 
                                 std::to_string(max_stale_data_ms_) + " ms");
    }
}

void RiskManager::updateDrawdown(double current_pnl) {
    // If PnL is negative, it adds to drawdown
    if (current_pnl < 0) {
        current_drawdown_ += std::abs(current_pnl);
    } else {
        // Simple mock: winning trades reduce drawdown 
        current_drawdown_ -= current_pnl;
        if (current_drawdown_ < 0) current_drawdown_ = 0;
    }
}

void RiskManager::checkDrawdown() const {
    if (current_drawdown_ > max_drawdown_) {
        throw std::runtime_error("DRAWDOWN CIRCUIT BREAKER: Account exceeded maximum drawdown of $" + 
                                 std::to_string(max_drawdown_));
    }
}
