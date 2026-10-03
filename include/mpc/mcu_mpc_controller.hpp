#pragma once

// tcbs::mpc::McuMpcController —— 实车 MCU 控制封装
//（双级 yaw MPC + 发送参数维护 + 后台发送线程）
//
// 与参考工程 /home/huhu233/rm2027/TorqueController 的 tcs::McuMpcController
// 形式一致，区别只在于被控对象是双连杆（大 yaw + 小 yaw）：
//
//   - set(auto_aim_enable, yaw_torque_only_mode_b, yaw_torque_only_mode_s,
//         target_psi_b, target_psi_s, pitch_target_angle, fire,
//         integral_enable_b, integral_enable_s):
//       一次性设置全部发送参数与 MPC 目标。两个 yaw 目标都是**世界系方位角**
//       （psi_b / psi_s，rad）。设置时各自自动转换到与**当前**世界方位角角度差
//       最小的等效角（|差| ≤ π，且同角度），避免参考序列引入整圈偏差。
//       除两个目标传给 DualYawMpcController 外，其余参数由本类自己维护；
//       两个 yaw 通道的模式与积分开关各自独立。
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
//
//   - **首个 set 之前的安全返回包**：后台线程从启动起就在发帧，若电控上电即在等
//     上位机 yaw 力矩，构造到首次 set 之间会有一段"零目标 MPC 力矩"窗口（且冷启动时
//     状态全 0，见 FullStrictPoseBuilder 的样本缓存）。因此本类持有一个构造期就建好的
//     安全包 safe_packet_（auto_aim / fire 关闭、pitch 目标 0、两轴模式均为仅力矩、
//     yaw 目标角/角速度/力矩全 0），并在首个 set 的数据"被解算出来但还没应用"之前
//     替代真实控制量下发。替换只发生在发送处：
//       * 解算、延迟缓冲、积分补偿、热启动、last_state_ 全部照常执行（该次解算照常
//         计入 MPC 内部状态，控制连续性不受影响）；
//       * 首个 set 的参数**第一次参与解算的那一帧**发安全包，它的 MPC 结果不发送；
//         从下一帧起恢复正常发送。构造后、首个 set 之前的每一帧也都是安全包。
//     ★ 只挡**首个** set：三态状态机 safe_send_state_ 只在 set_mtx_ 内读写，且是单向
//       闩锁（WAITING_FIRST_SET → FIRST_SET_PENDING → APPLIED，APPLIED 之后不再回退），
//       所以首个 set 之后的每次 set() 都是普通的目标刷新，直接按新参数出控制量。
//       因此无论首个 set 落在 loop 的哪个时刻（含正好落在 loop 持锁前），它的控制量
//       都最早只能从解算它的下一帧生效。

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
    ///                        以及 use_gravity：默认 true = 使用重力，
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
    // yaw_torque_only_mode_b / _s：两轴各自的 YawMode 选择（true = 仅力矩）。
    // integral_enable_b / _s：本步两轴是否启用积分补偿（各自透传给
    // DualYawMpcController::step）。
    //
    // ★ 仅**首次**调用本函数（或序列版）会触发安全包交接：状态从 WAITING_FIRST_SET
    //   推到 FIRST_SET_PENDING，后台线程把这次参数解算出来的那一帧仍发 safe_packet_，
    //   从下一帧起才发真实控制量；之后的调用只是普通目标刷新，不再影响安全包。
    // ------------------------------------------------------------------
    void set(bool auto_aim_enable,
             bool yaw_torque_only_mode_b,
             bool yaw_torque_only_mode_s,
             double target_psi_b,
             double target_psi_s,
             double pitch_target_angle,
             bool fire,
             bool integral_enable_b = false,
             bool integral_enable_s = false);

    // ------------------------------------------------------------------
    // 序列版 set：传入 target_psi_b / target_psi_s / pitch / fire 四个序列
    //（不截断，各序列独立存储；两个 yaw 目标序列：第一个值 remainder 到当前
    //  世界方位角 ±π 内，后续值 remainder 到前一个值 ±π 内）。
    // 后台线程按序逐通道消费：某序列非空取首值，空则对应成员保持当前值（回原模式）。
    // 调用单目标 set 会清空这些序列。
    // ------------------------------------------------------------------
    void set(bool auto_aim_enable,
             bool yaw_torque_only_mode_b,
             bool yaw_torque_only_mode_s,
             const std::vector<double>& target_psi_b_seq,
             const std::vector<double>& target_psi_s_seq,
             const std::vector<double>& pitch_seq,
             const std::vector<bool>& fire_seq,
             bool integral_enable_b = false,
             bool integral_enable_s = false);

    /// 最新 MPC 结果（线程安全，显示用）
    State state() const;

private:
    void loop();

    com::RobotCommunication* comm_;
    DualYawMpcController     mpc_;

    // 设置参数锁（后台线程读取）
    mutable std::mutex set_mtx_;
    bool   auto_aim_enable_ = true;
    bool   yaw_torque_only_mode_b_ = true;  // 大 yaw 通道模式（true = 仅力矩）
    bool   yaw_torque_only_mode_s_ = true;  // 小 yaw 通道模式
    bool   integral_enable_b_ = false;       // 大 yaw 积分补偿开关
    bool   integral_enable_s_ = false;       // 小 yaw 积分补偿开关
    double target_psi_b_ = 0.0;
    double target_psi_s_ = 0.0;
    double pitch_target_angle_ = 0.0;
    bool   fire_ = false;

    // ★ 安全包状态机（见类头"首个 set 之前的安全返回包"）。用三态而不是一个 bool，
    //   因为"构造后到首个 set 之间一直发安全包"与"首个 set 被解算的那一帧发安全包"
    //   是两个独立条件，单个 bool 无法同时表达：
    //     WAITING_FIRST_SET : 外部还从未 set ⇒ 本帧发安全包（覆盖构造后**每一帧**）
    //     FIRST_SET_PENDING : 首个外部 set 已到达、但还没有一帧解算过它
    //                         ⇒ 本帧解算它、仍发安全包，然后转 APPLIED
    //     APPLIED           : 本帧发真实控制包（**吸收态**）
    //   ★ 只有在 WAITING_FIRST_SET 状态下收到 set 才会跳到 FIRST_SET_PENDING，即
    //     "只挡**首个** set。之后的 set 是正常运行时的周期性目标刷新，直接按新参数出
    //     控制量，绝不重置状态机" —— 否则正常运行时每个 set() 都会把状态推回起点，
    //     导致每一帧都发安全包、机器人永远收不到力矩。
    //   三个状态只在 set_mtx_ 内读写，故"首个 set 的控制量最早从解算它的下一帧生效"
    //   与 set() 落在 loop 哪个时刻无关。
    enum class SafeSendState { WAITING_FIRST_SET, FIRST_SET_PENDING, APPLIED };
    // 初值 WAITING_FIRST_SET：构造后既无操作员意图、冷启动时状态样本也可能还是全 0
    //（FullStrictPoseBuilder 从未收到样本时全 0），此时解算出来的"零目标 MPC 力矩"
    // 正是要避免的东西 ⇒ 首个 set 到达前每一帧都发 safe_packet_。
    SafeSendState safe_send_state_ = SafeSendState::WAITING_FIRST_SET;

    // 序列模式成员（非空时 loop 优先消费）
    std::deque<double> target_psi_b_seq_;
    std::deque<double> target_psi_s_seq_;
    std::deque<double> pitch_seq_;
    std::deque<bool>   fire_seq_;

    // 最新结果锁（显示线程读取）
    mutable std::mutex state_mtx_;
    State last_state_;

    // ★ 安全返回包：**构造期一次性建好**，内容在对象生命周期内不再变化（故为 const）。
    //   仅在"首个 set 的数据尚未被解算"期间替代真实控制量下发：
    //     auto_aim_enable = 0、fire = 0、pitch_target_angle = 0（映射前的关节角语义，
    //     仍照常过 McuDataPreprocessor::processSend）、两轴 yaw_big/small_mode 均为
    //     YAW_MODE_TORQUE_ONLY（仅力矩，不叠加电控位置/速度内环）、
    //     yaw 目标角/目标角速度/力矩两轴全 0。
    //   （成员初始化列表里用 makeSafeSendPacket() 逐字段赋值；SendPacket 多数字段没有
    //     默认成员初始化器，必须先值初始化再赋，否则会按未定义值发出。）
    const com::mcu::SendPacket safe_packet_;

    double loop_period_ = 0.01;      // 后台 loop 周期（秒，构造传入）
    FrameRateCounter fps_counter_;   // loop 真实帧率统计（滑动窗口 60 帧）

    // 距上一次 set() 已过去的后台 loop 次数（set 时清零，loop 每循环一次 +1）
    std::atomic<uint64_t> ticks_since_set_{0};

    std::thread thread_;
    std::atomic<bool> running_{false};
};

}  // namespace mpc
}  // namespace tcbs
