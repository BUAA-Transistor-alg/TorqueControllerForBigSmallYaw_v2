#pragma once

// tcbs::mpc::McuMpcController —— 实车 MCU 控制封装
//（双级 yaw MPC + 发送参数维护 + 后台发送线程）
//
// 与参考工程 /home/huhu233/rm2027/TorqueController 的 tcs::McuMpcController
// 形式一致，区别只在于被控对象是双连杆（大 yaw + 小 yaw）：
//
//   - set(auto_aim_enable, yaw_torque_only_mode, target_psi_b, target_psi_s,
//         pitch_target_angle, fire, integral_enable):
//       一次性设置全部发送参数与 MPC 目标。两个 yaw 目标都是**世界系方位角**
//       （psi_b / psi_s，rad）。设置时各自自动转换到与**当前**世界方位角角度差
//       最小的等效角（|差| ≤ π，且同角度），避免参考序列引入整圈偏差。
//       除两个目标传给 DualYawMpcController 外，其余参数由本类自己维护。
//
//   - 后台线程以 loop_period（秒，默认 0.01 = 100Hz）为周期运行：
//       取最新目标调用 DualYawMpcController::step，以 mpc 结果 + 最新发送参数
//       构造 com::mcu::SendPacket 发给 MCU。
//       下发的 yaw 目标角/角速度是**关节系**量（协议语义）：
//           yaw_big_target_angle   = 一步后 θ_b = pred_psi_b − θ_c
//           yaw_small_target_angle = 一步后 θ_s = pred_psi_s − pred_psi_b
//       力矩为 MPC（含积分补偿、已限幅）输出的第一步力矩。
//
//   - 循环：开始处取 steady_clock::now()，结束处 sleep_until(start + loop_period)，
//     不跟随绝对时间点，避免误差累计。
//   - 循环真实帧率由 FrameRateCounter 记录（滑动窗口 60 帧），随 State::loop_fps 返回。
//   - 设置参数与最新结果（last_state_）使用独立的锁保护。

#include <atomic>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#include "Communications.hpp"
#include "FrameRateCounter.hpp"
#include "dual_yaw_mpc_controller.hpp"
#include "params.hpp"

namespace tcbs {
namespace mpc {

class McuMpcController {
public:
    // 最新 MPC 求解结果（供显示/日志）
    struct State {
        // ── 下发给 MCU 的量（关节系，一步后预测）──
        double yaw_big_target_angle = 0.0;
        double yaw_big_target_velocity = 0.0;
        double yaw_big_torque = 0.0;
        double yaw_small_target_angle = 0.0;
        double yaw_small_target_velocity = 0.0;
        double yaw_small_torque = 0.0;

        // ── 世界系预测（显示用）──
        double pred_psi_b = 0.0;
        double pred_psi_s = 0.0;

        // ── 当前（延迟 dt*N 步的）目标，供显示 ──
        double delayed_target_b = 0.0;
        double delayed_target_s = 0.0;

        // ── 最近一次 set() 写入（并已 wrap 到当前世界方位角最近等效角）的目标 ──
        // 与 delayed_target_* 的区别：这是"操作员刚下发的意图"，不受 N 步延迟影响，
        // 且即使在通信未就绪（求解跳过）时也可用。
        double set_target_b = 0.0;
        double set_target_s = 0.0;

        // ── 积分补偿累积量（两轴各自独立）──
        double integral_b = 0.0;
        double integral_s = 0.0;

        double loop_fps = 0.0;         // 后台 loop 线程真实帧率（滑动平均，帧/秒）
        uint64_t ticks_since_set = 0;  // 距上一次 set() 已过去的后台 loop 次数

        std::vector<double> ref_psi_b_seq;   // 本次参考（目标）序列（N 个）
        std::vector<double> ref_psi_s_seq;
        std::vector<double> pred_psi_b_seq;  // 本次预测方位角序列（N 个）
        std::vector<double> pred_psi_s_seq;
    };

    /// @param comm            机器人通信（可为 nullptr，此时不退化为发送、step 返回 invalid）
    /// @param params          动力学参数
    /// @param mpc_options     MPC 求解器配置（dt / N / 限幅 / 权重 / refinement，
    ///                        以及 use_gravity：默认 false = 模型里没有重力项，
    ///                        实测 gx/gy 仍照常读出但不参与求解）
    /// @param wrapper_options 积分补偿配置（两轴各自增益；默认不积分）
    /// @param loop_period     后台 loop 周期 [s]（默认 0.01 = 100Hz）
    McuMpcController(com::RobotCommunication* comm,
                     const dm::Params& params,
                     const MPCController::Options& mpc_options,
                     const DualYawMpcController::Options& wrapper_options,
                     double loop_period = 0.01);
    ~McuMpcController();

    McuMpcController(const McuMpcController&) = delete;
    McuMpcController& operator=(const McuMpcController&) = delete;

    void start();  ///< 启动后台发送线程（可重复调用，幂等）
    void stop();   ///< 停止并 join

    // ------------------------------------------------------------------
    // 单目标 set：设置发送参数 + 两个 yaw 的 MPC 目标（线程安全）
    //
    // 目标为世界系方位角；内部自动转换到与当前世界方位角差最小的等效角。
    // integral_enable：本步是否启用两轴积分补偿（透传 DualYawMpcController::step）。
    // ------------------------------------------------------------------
    void set(bool auto_aim_enable,
             bool yaw_torque_only_mode,
             double target_psi_b,
             double target_psi_s,
             double pitch_target_angle,
             bool fire,
             bool integral_enable = false);

    // ------------------------------------------------------------------
    // 序列版 set：传入 target_psi_b / target_psi_s / pitch / fire 四个序列
    //（不截断，各序列独立存储；两个 yaw 目标序列：第一个值 remainder 到当前
    //  世界方位角 ±π 内，后续值 remainder 到前一个值 ±π 内）。
    // 后台线程按序逐通道消费：某序列非空取首值，空则对应成员保持当前值（回原模式）。
    // 调用单目标 set 会清空这些序列。
    // ------------------------------------------------------------------
    void set(bool auto_aim_enable,
             bool yaw_torque_only_mode,
             const std::vector<double>& target_psi_b_seq,
             const std::vector<double>& target_psi_s_seq,
             const std::vector<double>& pitch_seq,
             const std::vector<bool>& fire_seq,
             bool integral_enable = false);

    /// 最新 MPC 结果（线程安全，显示用）
    State state() const;

private:
    void loop();

    com::RobotCommunication* comm_;
    DualYawMpcController     mpc_;

    // 设置参数锁（后台线程读取）
    mutable std::mutex set_mtx_;
    bool   auto_aim_enable_ = true;
    bool   yaw_torque_only_mode_ = false;
    bool   integral_enable_ = false;  // 积分补偿开关（后台线程读取）
    double target_psi_b_ = 0.0;
    double target_psi_s_ = 0.0;
    double pitch_target_angle_ = 0.0;
    bool   fire_ = false;

    // 序列模式成员（非空时 loop 优先消费）
    std::deque<double> target_psi_b_seq_;
    std::deque<double> target_psi_s_seq_;
    std::deque<double> pitch_seq_;
    std::deque<bool>   fire_seq_;

    // 最新结果锁（显示线程读取）
    mutable std::mutex state_mtx_;
    State last_state_;

    double loop_period_ = 0.01;      // 后台 loop 周期（秒，构造传入）
    FrameRateCounter fps_counter_;   // loop 真实帧率统计（滑动窗口 60 帧）

    // 距上一次 set() 已过去的后台 loop 次数（set 时清零，loop 每循环一次 +1）
    std::atomic<uint64_t> ticks_since_set_{0};

    std::thread thread_;
    std::atomic<bool> running_{false};
};

}  // namespace mpc
}  // namespace tcbs
