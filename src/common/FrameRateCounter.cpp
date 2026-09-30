#include "FrameRateCounter.hpp"

#include <chrono>

namespace tcbs {

FrameRateCounter::FrameRateCounter(std::size_t window_size) : window_size_(window_size) {}

void FrameRateCounter::tick() {
    using namespace std::chrono;
    const auto now = high_resolution_clock::now();

    // 首次调用只记录时间基准（否则会把"从 epoch 到现在的耗时"当成第一帧）。
    if (last_frame_time_.time_since_epoch().count() == 0) {
        last_frame_time_ = now;
        return;
    }

    const double frame_time = duration_cast<duration<double>>(now - last_frame_time_).count();
    last_frame_time_ = now;

    frame_times_.push_back(frame_time);
    time_sum_ += frame_time;
    while (frame_times_.size() > window_size_) {
        time_sum_ -= frame_times_.front();
        frame_times_.pop_front();
    }
}

double FrameRateCounter::fps() const {
    if (frame_times_.empty() || time_sum_ <= 0.0) return 0.0;
    return static_cast<double>(frame_times_.size()) / time_sum_;
}

double FrameRateCounter::avg_frame_time() const {
    if (frame_times_.empty()) return 0.0;
    return time_sum_ / static_cast<double>(frame_times_.size());
}

void FrameRateCounter::reset() {
    frame_times_.clear();
    time_sum_ = 0.0;
    last_frame_time_ = {};
}

void FrameRateCounter::set_window_size(std::size_t new_size) {
    window_size_ = new_size;
    while (frame_times_.size() > window_size_) {
        time_sum_ -= frame_times_.front();
        frame_times_.pop_front();
    }
}

}  // namespace tcbs
