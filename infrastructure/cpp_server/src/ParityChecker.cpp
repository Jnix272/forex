#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include "RingBuffer.h"

int main() {
    std::cout << "Starting Tier-1 Feature Scale Parity Check..." << std::endl;

    std::ifstream tick_file("../../../parity_ticks.csv");
    std::ifstream feature_file("../../../parity_features.csv");

    if (!tick_file.is_open() || !feature_file.is_open()) {
        std::cerr << "Error: Run python scripts/export_parity_data.py first to generate CSVs." << std::endl;
        return 1;
    }

    RingBuffer buffer(100);
    std::string tick_line, feature_line;
    int line_num = 1;
    bool passed = true;

    while (std::getline(tick_file, tick_line) && std::getline(feature_file, feature_line)) {
        double price = std::stod(tick_line);
        buffer.addTick(price);

        // Parse Python features
        std::stringstream ss(feature_line);
        std::string py_price_str, py_sma_str, py_ema_str;
        std::getline(ss, py_price_str, ',');
        std::getline(ss, py_sma_str, ',');
        std::getline(ss, py_ema_str, ',');

        double py_sma = std::stod(py_sma_str);
        double py_ema = std::stod(py_ema_str);

        // Calculate C++ features
        double cpp_sma = buffer.getSMA(14);
        double cpp_ema = buffer.getEMA(14, buffer.getPriceAt(0)); // simple mock

        // If we haven't warmed up 14 periods, Python backfills. C++ will differ initially.
        // We only check parity after period 14.
        if (line_num >= 14) {
            double diff_sma = std::abs(py_sma - cpp_sma);
            double diff_ema = std::abs(py_ema - cpp_ema);

            if (diff_sma > 0.00001 || diff_ema > 0.00001) {
                std::cerr << "PARITY FAILURE at line " << line_num << std::endl;
                std::cerr << "Python SMA: " << py_sma << " | C++ SMA: " << cpp_sma << " | Diff: " << diff_sma << std::endl;
                std::cerr << "Python EMA: " << py_ema << " | C++ EMA: " << cpp_ema << " | Diff: " << diff_ema << std::endl;
                passed = false;
                break;
            }
        }
        line_num++;
    }

    if (passed) {
        std::cout << "SUCCESS: C++ Engine features perfectly match Python Pandas features (Scale Mismatch Prevention Verified)." << std::endl;
        return 0;
    } else {
        std::cerr << "FATAL: Feature Scale Mismatch detected. Blocking compilation." << std::endl;
        return 1;
    }
}
