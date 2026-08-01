#pragma once

#include <stdexcept>
#include <string>

class RiskManager {
public:
    RiskManager(int max_position_size, long max_stale_data_ms, double max_drawdown);

    // Fat Finger Check
    void checkFatFinger(int units) const;

    // Stale Data Check
    void checkStaleData(long long last_tick_timestamp_ms) const;

    // Drawdown Check
    void updateDrawdown(double current_pnl);
    void checkDrawdown() const;

private:
    int max_position_size_;
    long long max_stale_data_ms_;
    double max_drawdown_;
    double current_drawdown_;
};
