#pragma once

// tcbs::mpc::DualYawMpcController —— 双级 yaw 力矩 MPC 的实车封装（参考序列 + 求解）。
//
// 与参考工程 /home/huhu233/rm2027/TorqueController 的 tcs::YawMpcController 对应，
// 职责相同（读状态 → 维护延迟目标序列 → 求解 → 积分补偿 → 返回待发送值，不发送），
// 差别在于：
//   * 被控对象是双连杆（大 yaw θ_b + 小 yaw θ_s），一次同时给出两轴力矩；
//   * 目标序列**直接就是世界系方位角** psi_b* / psi_s*（rad），不再按底盘做变换：
//     trajectory 内部已按 theta_c / dtheta_c / ddtheta_c 外推基座，参考序列里
//     不要再叠加 (i+1)·dt·ω_c 之类的补偿项；
//   * 积分补偿**两轴各自独立**（各自的增益、累积量与限幅）。
//
// 状态读取 com::RobotCommunication::getStrictPose()：
//   θ_c        = chassis_azimuth
//   θ_b        = yaw_big_angle           θ_s       = yaw_small_angle
//   dθ_c       = chassis_omega           dθ_b      = big_motor_omega
//   dθ_s       = small_motor_omega
//   当前世界方位角按模型定义计算：psi_b = θ_c + θ_b，psi_s = psi_b + θ_s
// （FullStrictPoseBuilder 的反解算法尚未实现，chassis_azimuth / *_azimuth 目前为 0；
//  实现后本封装无需改动。基座角加速度暂用 0，可用 setBaseAngularAcceleration 覆盖。）
//
// 两种 step 模式（与参考工程一致）：
//   * step(target_psi_b, target_psi_s, enable)：把单点目标压入延迟缓冲
//     （缓冲最多 N 个，最旧在前），因此参考序列相对目标延迟 N·dt；
//   * step(target_psi_b_buf, target_psi_s_buf, enable)：直接用整条序列替换缓冲
//     （取前 N 个，不足用最后一个值补齐）。
//
// 积分补偿（integral_enable = true 时）：
//   积分值 += gain · (上一步预测的一步后方位角 − 这一步实测方位角)；
//   返回的力矩 = clamp(MPC 力矩 + 积分值, ±max_torque)；第一次 step 无上一步预测，
//   不计算积分增量。integral_enable = false 时积分值清空为 0。

#include <vector>

#include "Communications.hpp"
#include "mpc_controller.hpp"
#include "params.hpp"

namespace tcbs {
namespace mpc {

class DualYawMpcController {
public:
    struct Options {
        double integral_gain_b = 0.0;  ///< 大 yaw 积分比例系数
        double integral_gain_s = 0.0;  ///< 小 yaw 积分比例系数
    };

    struct Result {
        bool valid = false;  ///< 状态读取与求解是否正常

        double torque_b = 0.0;  ///< 待发送的大 yaw 力矩 [N·m]
        double torque_s = 0.0;  ///< 待发送的小 yaw 力矩 [N·m]

        double pred_psi_b = 0.0;      ///< 一步后预测世界方位角（= 预测序列第 0 项）
        double pred_psi_s = 0.0;
        double pred_dpsi_b = 0.0;     ///< 一步后预测世界角速度
        double pred_dpsi_s = 0.0;

        double state_psi_b = 0.0;     ///< 求解时读到的当前世界方位角
        double state_psi_s = 0.0;
        double target_psi_b = 0.0;    ///< 当前（延迟）目标，供显示
        double target_psi_s = 0.0;

        double integral_b = 0.0;      ///< 当前积分累积量（大 yaw，未限幅）
        double integral_s = 0.0;      ///< 当前积分累积量（小 yaw，未限幅）

        std::vector<double> ref_psi_b;         ///< 本次参考序列（N 个）
        std::vector<double> ref_psi_s;
        std::vector<double> pred_psi_b_seq;    ///< 本次预测方位角序列（N 个，不含当前）
        std::vector<double> pred_psi_s_seq;
    };

    /// @param comm        机器人通信（只读 getStrictPose()；可为 nullptr，此时 step 返回 valid=false）
    /// @param params      动力学参数
    /// @param mpc_options MPC 求解器配置（dt / N / 限幅 / 权重 / refinement）
    /// @param options     积分补偿配置（默认不积分）
    DualYawMpcController(com::RobotCommunication* comm,
                         const dm::Params& params,
                         const MPCController::Options& mpc_options,
                         const Options& options);

    /// 同上，积分补偿默认关闭（两个增益为 0）。
    DualYawMpcController(com::RobotCommunication* comm,
                         const dm::Params& params,
                         const MPCController::Options& mpc_options)
        : DualYawMpcController(comm, params, mpc_options, Options{}) {}

    DualYawMpcController(const DualYawMpcController&) = delete;
    DualYawMpcController& operator=(const DualYawMpcController&) = delete;

    /// 单点目标模式：压入延迟缓冲后求解（目标为世界系方位角）。
    Result step(double target_psi_b, double target_psi_s, bool integral_enable = false);

    /// 整序列模式：用传入序列替换延迟缓冲后求解（取前 N 个，不足用最后一个补齐）。
    Result step(const std::vector<double>& target_psi_b_buf,
                const std::vector<double>& target_psi_s_buf,
                bool integral_enable = false);

    // ------------------------------------------------------------------
    // 只读访问 / 配置
    // ------------------------------------------------------------------
    double integralB() const { return integral_b_; }
    double integralS() const { return integral_s_; }

    MPCController& mpc() { return mpc_; }
    const MPCController& mpc() const { return mpc_; }

    /// 覆盖预测用的基座角加速度（rad/s²，整段预测内常值；默认 0）。
    void setBaseAngularAcceleration(double ddtheta_c) { base_ddtheta_c_ = ddtheta_c; }
    double baseAngularAcceleration() const { return base_ddtheta_c_; }

    /// 清空延迟目标缓冲、积分值与 MPC 热启动状态。
    void reset();

private:
    /// 公共求解：读状态 + 构造参考序列 + MPC + 积分补偿。
    Result solve(bool integral_enable);

    /// 把一条目标序列写入延迟缓冲（最多 N 个，不足用最后一个值补齐到 N 个）。
    static void fillBuffer(std::vector<double>& buf,
                           const std::vector<double>& seq,
                           int n);

    com::RobotCommunication* comm_;
    int                      n_;
    MPCController            mpc_;
    Options                  options_;

    std::vector<double> target_buf_b_;  ///< 延迟目标序列（最旧在前，最多 N 个）
    std::vector<double> target_buf_s_;

    double base_ddtheta_c_ = 0.0;  ///< 基座角加速度（预测用）

    // 积分补偿状态（两轴独立）
    double integral_b_ = 0.0;
    double integral_s_ = 0.0;
    double prev_pred_psi_b_ = 0.0;
    double prev_pred_psi_s_ = 0.0;
    bool   has_prev_pred_ = false;
};

}  // namespace mpc
}  // namespace tcbs
