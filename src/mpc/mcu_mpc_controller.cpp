#include "mcu_mpc_controller.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace tcbs {
namespace mpc {
namespace {

/// 2π（不依赖 M_PI，保证 -std=c++17 严格模式下也可用）。
constexpr double kTwoPi = 2.0 * 3.14159265358979323846;

/// 把一个绝对目标角 remainder 到与 ref 角度差最小的等效角：
///   adj = ref + remainder(target − ref, 2π)
/// 满足 |adj − ref| ≤ π 且 adj ≡ target (mod 2π)。
inline double wrapToNearest(double target, double ref) {
    return ref + std::remainder(target - ref, kTwoPi);
}

/// 构造"安全返回包"（构造 McuMpcController 时调用一次，之后内容不再变化）。
///
/// 用途：后台发送线程从构造起就在发帧，而首个外部 set 之前既没有操作员意图、冷启动时
/// 状态样本也可能还是全 0；此包让电控在这一窗口内收到一个**明确的、不产生动作**的指令，
/// 而不是"零目标 MPC 解算出来的力矩"。
///
/// 内容（与类头说明一致）：
///   - auto_aim_enable = 0、fire = 0      —— 总开关与火控关闭；
///   - pitch_target_angle = 0             —— 映射前的关节角语义（仍照常过
///                                           McuDataPreprocessor::processSend）；
///   - 两轴 yaw 模式均为 YAW_MODE_TORQUE_ONLY（仅力矩，不叠加电控位置/速度内环）；
///   - yaw 目标角 / 目标角速度 / 力矩两轴全 0。
///
/// 注意：`SendPacket()` 是**值初始化**——协议结构体多数字段没有默认成员初始化器，
/// 不显式初始化就会按未定义值发送。
inline com::mcu::SendPacket makeSafeSendPacket() {
    com::mcu::SendPacket pkt = com::mcu::SendPacket();
    pkt.auto_aim_enable = 0;
    pkt.fire = 0;
    pkt.pitch_target_angle = 0.0f;
    pkt.yaw_big_mode = com::mcu::YAW_MODE_TORQUE_ONLY;
    pkt.yaw_small_mode = com::mcu::YAW_MODE_TORQUE_ONLY;
    pkt.yaw_big_target_angle = 0.0;
    pkt.yaw_big_target_velocity = 0.0f;
    pkt.yaw_big_torque = 0.0f;
    pkt.yaw_small_target_angle = 0.0f;
    pkt.yaw_small_target_velocity = 0.0f;
    pkt.yaw_small_torque = 0.0f;
    return pkt;
}

}  // namespace

McuMpcController::McuMpcController(com::RobotCommunication* comm,
                                   const dm::Params& params,
                                   const MPCController::Options& mpc_options,
                                   const DualYawMpcController::Options& wrapper_options,
                                   double loop_period)
    : comm_(comm),
      mpc_(comm, params, mpc_options, wrapper_options),
      // 安全返回包在**构造期**一次性建好，之后只读（private 成员 safe_packet_）。
      // 初始化顺序必须与头文件里的成员声明顺序一致（safe_packet_ 在 loop_period_ 之前，
      // 否则触发 -Wreorder）。
      safe_packet_(makeSafeSendPacket()),
      loop_period_(loop_period) {}

McuMpcController::~McuMpcController() {
    stop();
}

void McuMpcController::start() {
    if (!running_.exchange(true)) {
        thread_ = std::thread(&McuMpcController::loop, this);
    }
}

void McuMpcController::stop() {
    if (running_.exchange(false)) {
        if (thread_.joinable()) thread_.join();
    }
}

void McuMpcController::set(bool auto_aim_enable,
                           bool yaw_torque_only_mode_b,
                           bool yaw_torque_only_mode_s,
                           double target_psi_b,
                           double target_psi_s,
                           double pitch_target_angle,
                           bool fire,
                           bool integral_enable_b,
                           bool integral_enable_s) {
    // 目标自动转换到与当前世界方位角差最小的等效角（避免参考序列引入整圈偏差）。
    // 读状态不持锁（内部走通信快照），与参考工程一致。
    const DualYawMpcController::Measurement meas = mpc_.measure();
    const double ref_b = meas.valid ? meas.psi_b : 0.0;
    const double ref_s = meas.valid ? meas.psi_s : 0.0;
    const double adj_b = wrapToNearest(target_psi_b, ref_b);
    const double adj_s = wrapToNearest(target_psi_s, ref_s);

    std::lock_guard<std::mutex> lock(set_mtx_);
    auto_aim_enable_ = auto_aim_enable;
    yaw_torque_only_mode_b_ = yaw_torque_only_mode_b;
    yaw_torque_only_mode_s_ = yaw_torque_only_mode_s;
    integral_enable_b_ = integral_enable_b;
    integral_enable_s_ = integral_enable_s;
    target_psi_b_ = adj_b;
    target_psi_s_ = adj_s;
    pitch_target_angle_ = pitch_target_angle;
    fire_ = fire;
    // ★ **只认首个 set**：仅在"从未 set 过"时把状态从 WAITING_FIRST_SET 推到
    //   FIRST_SET_PENDING（那条路径由 loop 解算该 set 后转 APPLIED）。此后本函数不再
    //   碰状态机 —— 后续 set 是周期性目标刷新，必须直接用新参数出控制量。若无条件置回
    //   PENDING，正常运行时每个 set() 都会重置状态机，导致每帧都发安全包（零力矩）。
    if (safe_send_state_ == SafeSendState::WAITING_FIRST_SET) {
        safe_send_state_ = SafeSendState::FIRST_SET_PENDING;
    }
    // 单目标 set：清空序列模式数据
    target_psi_b_seq_.clear();
    target_psi_s_seq_.clear();
    pitch_seq_.clear();
    fire_seq_.clear();
    // 重置"距上一次 set 已过去的 loop 次数"
    ticks_since_set_ = 0;
}

void McuMpcController::set(bool auto_aim_enable,
                           bool yaw_torque_only_mode_b,
                           bool yaw_torque_only_mode_s,
                           const std::vector<double>& target_psi_b_seq,
                           const std::vector<double>& target_psi_s_seq,
                           const std::vector<double>& pitch_seq,
                           const std::vector<bool>& fire_seq,
                           bool integral_enable_b,
                           bool integral_enable_s) {
    // 两个 yaw 序列各自的 wrap 基准：当前世界方位角。
    const DualYawMpcController::Measurement meas = mpc_.measure();
    const double ref_b0 = meas.valid ? meas.psi_b : 0.0;
    const double ref_s0 = meas.valid ? meas.psi_s : 0.0;

    std::lock_guard<std::mutex> lock(set_mtx_);
    auto_aim_enable_ = auto_aim_enable;
    yaw_torque_only_mode_b_ = yaw_torque_only_mode_b;
    yaw_torque_only_mode_s_ = yaw_torque_only_mode_s;
    integral_enable_b_ = integral_enable_b;
    integral_enable_s_ = integral_enable_s;
    // ★ 同单目标 set：只认首个 set（见该处注释）。
    if (safe_send_state_ == SafeSendState::WAITING_FIRST_SET) {
        safe_send_state_ = SafeSendState::FIRST_SET_PENDING;
    }

    // 第一个值 remainder 到当前世界方位角 ±π 内，后续值 remainder 到前一个值 ±π 内。
    target_psi_b_seq_.clear();
    double prev_b = ref_b0;
    for (double v : target_psi_b_seq) {
        prev_b = wrapToNearest(v, prev_b);
        target_psi_b_seq_.push_back(prev_b);
    }
    target_psi_s_seq_.clear();
    double prev_s = ref_s0;
    for (double v : target_psi_s_seq) {
        prev_s = wrapToNearest(v, prev_s);
        target_psi_s_seq_.push_back(prev_s);
    }
    pitch_seq_.assign(pitch_seq.begin(), pitch_seq.end());
    fire_seq_.assign(fire_seq.begin(), fire_seq.end());
    ticks_since_set_ = 0;
}

McuMpcController::State McuMpcController::state() const {
    std::lock_guard<std::mutex> lock(state_mtx_);
    return last_state_;
}

void McuMpcController::loop() {
    while (running_) {
        // 循环开始处获取本次循环时间基准
        auto start = std::chrono::steady_clock::now();

        // 取最新设置的发送参数与 MPC 目标；若序列模式非空则优先消费序列
        bool aa, mode_b, mode_s, fire, integral_enable_b, integral_enable_s, use_seq;
        double target_b, target_s, pitch;
        std::vector<double> seq_b, seq_s;
        // 本帧是否用安全返回包替代真实控制量下发（见类头"首个 set 之前的安全返回包"）。
        // 状态机（私有成员 safe_send_state_，只在 set_mtx_ 内读写）：
        //   WAITING_FIRST_SET —— 外部从未 set ⇒ 发安全包（构造后每帧）；
        //   FIRST_SET_PENDING —— 首个 set 已到、本帧解算它 ⇒ 仍发安全包，转 APPLIED；
        //   APPLIED           —— 发真实控制包（吸收态，后续 set 不再回到安全包）。
        bool send_safe_packet;
        {
            std::lock_guard<std::mutex> lock(set_mtx_);
            aa = auto_aim_enable_;
            mode_b = yaw_torque_only_mode_b_;
            mode_s = yaw_torque_only_mode_s_;
            integral_enable_b = integral_enable_b_;
            integral_enable_s = integral_enable_s_;
            target_b = target_psi_b_;
            target_s = target_psi_s_;
            pitch = pitch_target_angle_;
            fire = fire_;

            // ★ 与上面参数读取在**同一个临界区**内推进状态机：读到 WAITING_FIRST_SET 说明
            //   还没有任何 set 参数（构造后窗口）；读到 FIRST_SET_PENDING 说明本帧读到的
            //   就是首个 set 的参数，本帧解算它但发安全包；APPLIED 之后一直是真实控制量。
            switch (safe_send_state_) {
                case SafeSendState::WAITING_FIRST_SET:
                    send_safe_packet = true;
                    break;
                case SafeSendState::FIRST_SET_PENDING:
                    send_safe_packet = true;
                    safe_send_state_ = SafeSendState::APPLIED;
                    break;
                case SafeSendState::APPLIED:
                default:
                    send_safe_packet = false;
                    break;
            }

            // 各通道独立消费：序列非空时取首值覆盖成员并移除；某序列用完（空）时
            // 对应成员保持当前值（回原模式），其余通道继续用序列。
            use_seq = !target_psi_b_seq_.empty() || !target_psi_s_seq_.empty();
            if (!target_psi_b_seq_.empty()) {
                seq_b.assign(target_psi_b_seq_.begin(), target_psi_b_seq_.end());
                target_psi_b_ = target_psi_b_seq_.front();
                target_psi_b_seq_.pop_front();
                target_b = target_psi_b_;
            }
            if (!target_psi_s_seq_.empty()) {
                seq_s.assign(target_psi_s_seq_.begin(), target_psi_s_seq_.end());
                target_psi_s_ = target_psi_s_seq_.front();
                target_psi_s_seq_.pop_front();
                target_s = target_psi_s_;
            }
            if (!pitch_seq_.empty()) {
                pitch_target_angle_ = pitch_seq_.front();
                pitch_seq_.pop_front();
                pitch = pitch_target_angle_;
            }
            if (!fire_seq_.empty()) {
                fire_ = fire_seq_.front();
                fire_seq_.pop_front();
                fire = fire_;
            }
        }

        // MPC 求解（序列模式传整个剩余目标缓冲；内部读状态、含积分补偿）
        DualYawMpcController::Result res;
        if (use_seq) {
            // 某一路序列已用完时，用该路"保持值"组一个单元素序列（内部会补齐到 N）。
            if (seq_b.empty()) seq_b.assign(1, target_b);
            if (seq_s.empty()) seq_s.assign(1, target_s);
            res = mpc_.step(seq_b, seq_s, integral_enable_b, integral_enable_s);
        } else {
            res = mpc_.step(target_b, target_s, integral_enable_b, integral_enable_s);
        }

        // 替换点：解算与状态记录都照常完成后，**只在这里**决定实际发出的是真实控制量
        // 还是构造期就建好的安全返回包（send_safe_packet 为 true 的那些帧）。
        if (send_safe_packet) {
            // 真实控制包连构造都不做：这些帧 MCU 只应收到安全返回包
            if (comm_) comm_->sendToMcu(safe_packet_);
        } else {
            // 配合最新设置构造发送包（yaw 目标角/角速度为**关节系**量）
            com::mcu::SendPacket pkt{};
            pkt.auto_aim_enable = aa ? 1 : 0;
            pkt.fire = fire ? 1 : 0;
            pkt.pitch_target_angle = static_cast<float>(pitch);
            // 两个 yaw 通道的模式各自独立（YawMode：1 = 仅力矩，2 = 力矩 + 内环）
            pkt.yaw_big_mode = mode_b ? com::mcu::YAW_MODE_TORQUE_ONLY
                                      : com::mcu::YAW_MODE_TORQUE_PLUS_PID;
            pkt.yaw_small_mode = mode_s ? com::mcu::YAW_MODE_TORQUE_ONLY
                                        : com::mcu::YAW_MODE_TORQUE_PLUS_PID;

            pkt.yaw_big_target_angle = res.pred_theta_b;
            pkt.yaw_big_target_velocity = static_cast<float>(res.pred_dtheta_b);
            pkt.yaw_big_torque = static_cast<float>(res.torque_b);

            pkt.yaw_small_target_angle = static_cast<float>(res.pred_theta_s);
            pkt.yaw_small_target_velocity = static_cast<float>(res.pred_dtheta_s);
            pkt.yaw_small_torque = static_cast<float>(res.torque_s);

            if (comm_) comm_->sendToMcu(pkt);
        }

        // 缓存最新结果（供显示）
        {
            std::lock_guard<std::mutex> lock(state_mtx_);
            last_state_.yaw_big_target_angle = res.pred_theta_b;
            last_state_.yaw_big_target_velocity = res.pred_dtheta_b;
            last_state_.yaw_big_torque = res.torque_b;
            last_state_.yaw_small_target_angle = res.pred_theta_s;
            last_state_.yaw_small_target_velocity = res.pred_dtheta_s;
            last_state_.yaw_small_torque = res.torque_s;
            last_state_.pred_psi_b = res.pred_psi_b;
            last_state_.pred_psi_s = res.pred_psi_s;
            last_state_.delayed_target_b = res.target_psi_b;
            last_state_.delayed_target_s = res.target_psi_s;
            last_state_.set_target_b = target_b;
            last_state_.set_target_s = target_s;
            last_state_.integral_b = res.integral_b;
            last_state_.integral_s = res.integral_s;
            last_state_.ref_psi_b_seq = res.ref_psi_b;
            last_state_.ref_psi_s_seq = res.ref_psi_s;
            last_state_.pred_psi_b_seq = res.pred_psi_b_seq;
            last_state_.pred_psi_s_seq = res.pred_psi_s_seq;
            last_state_.loop_fps = fps_counter_.fps();
            last_state_.ticks_since_set = ticks_since_set_.load();
        }

        // 循环结束处：等待到 start + loop_period（默认 10ms = 100Hz），
        // 不严格跟随绝对时间点
        std::this_thread::sleep_until(start + std::chrono::duration<double>(loop_period_));

        // 记录本帧真实耗时（含求解 + sleep），用于统计 loop 真实帧率；
        // 距上一次 set 已过去的 loop 次数 +1（每次循环完成计一个时间点）
        fps_counter_.tick();
        ++ticks_since_set_;
    }
}

}  // namespace mpc
}  // namespace tcbs
