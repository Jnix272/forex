#pragma once

#include <vector>
#include <cstddef>

class RingBuffer {
public:
    explicit RingBuffer(size_t capacity);

    void addTick(double price);
    
    // Feature extraction
    double getPriceAt(size_t lag) const;
    double getSMA(size_t period) const;
    double getEMA(size_t period, double previous_ema) const;
    double getATR(size_t period, const std::vector<double>& highs, const std::vector<double>& lows) const;

    size_t size() const { return count_; }
    size_t capacity() const { return capacity_; }
    bool isFull() const { return count_ >= capacity_; }

private:
    std::vector<double> buffer_;
    size_t capacity_;
    size_t head_;
    size_t count_;
};
