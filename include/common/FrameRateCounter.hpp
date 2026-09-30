#pragma once

// FrameRateCounter —— 滑动窗口帧率统计（后台 loop 的真实帧率）。
//
// 用法：每帧结束调用一次 tick()，随时用 fps() / avg_frame_time() 读取窗口内的
// 平均值。窗口按"帧耗时"的滑动平均计算，因此统计的是**真实帧率**（含求解与
// sleep 的全部时间），而不是名义周期。
//
// 与参考工程 /home/huhu233/rm2027/TorqueController 的 tcs::FrameRateCounter
// 形式一致（滑动窗口 60 帧），命名空间改为 tcbs。

#include <chrono>
#include <cstddef>
#include <deque>

namespace tcbs {

class FrameRateCounter {
public:
    /// @param window_size 滑动平均窗口大小（帧数），默认 60
    explicit FrameRateCounter(std::size_t window_size = 60);

    /// 每帧结束时调用一次（首次调用只记录时间基准，不产生样本）。
    void tick();

    /// 窗口内平均帧率（帧/秒）；无样本时为 0。
    double fps() const;

    /// 窗口内平均每帧耗时（秒）；无样本时为 0。
    double avg_frame_time() const;

    /// 清空全部历史数据。
    void reset();

    std::size_t window_size() const { return window_size_; }

    /// 修改窗口大小（会丢弃超出新窗口的旧样本）。
    void set_window_size(std::size_t new_size);

private:
    std::chrono::high_resolution_clock::time_point last_frame_time_{};
    std::deque<double> frame_times_;  ///< 帧耗时滑动窗口
    double time_sum_ = 0.0;           ///< 窗口内帧耗时总和
    std::size_t window_size_;
};

}  // namespace tcbs
