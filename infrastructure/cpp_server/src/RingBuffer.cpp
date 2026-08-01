#include "RingBuffer.h"
#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <cmath>

RingBuffer::RingBuffer(size_t capacity) : capacity_(capacity), head_(0), count_(0) {
    buffer_.resize(capacity_, 0.0);
}

void RingBuffer::addTick(double price) {
    buffer_[head_] = price;
    head_ = (head_ + 1) % capacity_;
    if (count_ < capacity_) {
        count_++;
    }
}

double RingBuffer::getPriceAt(size_t lag) const {
    if (lag >= count_) {
        throw std::out_of_range("Lag exceeds current buffer count");
    }
    // Head points to next insertion. Current latest is (head - 1 + capacity) % capacity
    size_t index = (head_ + capacity_ - 1 - lag) % capacity_;
    return buffer_[index];
}

double RingBuffer::getSMA(size_t period) const {
    if (period > count_ || period == 0) return 0.0;
    
    double sum = 0.0;
    for (size_t i = 0; i < period; ++i) {
        sum += getPriceAt(i);
    }
    return sum / period;
}

double RingBuffer::getEMA(size_t period, double previous_ema) const {
    if (period == 0) return 0.0;
    double alpha = 2.0 / (period + 1);
    double current_price = getPriceAt(0);
    return (current_price - previous_ema) * alpha + previous_ema;
}

double RingBuffer::getATR(size_t period, const std::vector<double>& highs, const std::vector<double>& lows) const {
    // True range logic would require High/Low/Close buffers.
    // This is a stub placeholder to demonstrate where it fits.
    return 0.0010; 
}
