#pragma once

// tcbs::RobotController —— 一体化控制封装（双级 yaw）
//
// 与参考工程 /home/huhu233/rm2027/TorqueController 的 tcs::RobotController 形式
// 一致：内部完成
//   - 通信接口建立（com::RobotCommunication：MCU + IMU 串口 + 严格反解包构建器）
//   - MPC 控制封装建立（mpc::McuMpcController：双级 yaw MPC + 后台发送线程，
//     周期由构造参数 mpc_loop_period 指定）
//
// 外部用法：实例化一个 RobotController 即可完全控制——
//   - getState()：统一获取一个按来源分组的结构体（MCU / IMU 原始数据、
//     严格反解包、MPC 状态；MPC 额外含最新一次运算的参考序列与预测序列）
//   - set()：直通 McuMpcController::set（设置发送参数 + 两个 yaw 的 MPC 目标）
//
// 关于严格反解包：本工程暂无参考工程的 FusionFilter/yaw 状态估计器，
// com::RobotCommunication 提供的世界系输出就是 FullStrictPoseBuilder::StrictPose，
// 因此 State::strict 直接使用该类型（而非另立同构结构），避免与其实现漂移。

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "Communications.hpp"
#include "mcu_mpc_controller.hpp"
#include "params.hpp"

namespace tcbs {

class RobotController {
public:
    // ── 按来源分组的数据结构 ──

    // MCU 原始数据（经 McuDataPreprocessor 预处理；yaw 通道已按标定映射）
    struct McuData {
        bool   valid = false;
        float  bullet_velocity = 0.0f;
        float  pitch_angle = 0.0f;        // 已标定（关节角语义）
        double yaw_big_angle = 0.0;       // 大 yaw 关节角（多圈连续）
        float  yaw_big_omega = 0.0f;
        float  yaw_small_angle = 0.0f;    // 小 yaw 关节角（相对大 yaw）
        float  yaw_small_omega = 0.0f;
        float  chassis_imu_yaw = 0.0f;    // 底盘自身 IMU 的 yaw（0 ~ 2π）
        float  chassis_imu_omega = 0.0f;  // 底盘 yaw 角速度
        uint8_t mark = 0, color = 0, auto_aim_switch = 0;
        uint8_t yaw_big_temperature = 0, yaw_small_temperature = 0;
        uint8_t mcu2_seq = 0;             // MCU2 新样本序号（值保持时不变）
    };

    // IMU 原始数据
    struct ImuData {
        bool   valid = false;
        float  gx = 0.0f, gy = 0.0f, gz = 0.0f;
        float  ax = 0.0f, ay = 0.0f, az = 0.0f;
        double euler_yaw = 0.0, euler_pitch = 0.0, euler_roll = 0.0;
        uint32_t dt_one_tenth_ms = 0;
    };

    // MPC 状态（= McuMpcController::State，另附世界系预测与目标）
    struct MpcData {
        double yaw_big_target_angle = 0.0;     // 下发的关节系目标角（一步后预测）
        double yaw_big_target_velocity = 0.0;
        double yaw_big_torque = 0.0;
        double yaw_small_target_angle = 0.0;
        double yaw_small_target_velocity = 0.0;
        double yaw_small_torque = 0.0;
        double pred_psi_b = 0.0;               // 世界系预测方位角
        double pred_psi_s = 0.0;
        double delayed_target_b = 0.0;         // 当前参考（延迟 dt*N 步的目标）
        double delayed_target_s = 0.0;
        double set_target_b = 0.0;             // 最近一次 set() 下发的（已 wrap 的）目标
        double set_target_s = 0.0;
        double integral_b = 0.0;               // 积分补偿累积量（两轴独立）
        double integral_s = 0.0;
        double loop_fps = 0.0;                 // 后台 loop 真实帧率（帧/秒）
        uint64_t ticks_since_set = 0;          // 距上一次 set() 已过去的 loop 次数
        std::vector<double> ref_psi_b_seq;     // 最新一次参考序列（N 个）
        std::vector<double> ref_psi_s_seq;
        std::vector<double> pred_psi_b_seq;    // 最新一次预测序列（N 个，不含当前）
        std::vector<double> pred_psi_s_seq;
    };

    // 统一状态：按来源分组
    struct State {
        McuData mcu;
        ImuData imu;
        com::FullStrictPoseBuilder::StrictPose strict;  // 严格反解包（独立输出）
        MpcData mpc;
    };

    // 控制模式：单目标（默认）/ 序列。
    // 构造时选定模式后，调用另一种模式的 set 接口会抛出 std::runtime_error。
    enum class Mode { SINGLE = 0, SEQUENCE = 1 };

    /// @param imu_location     IMU 安装位置（构型，决定严格反解的运动学链）
    /// @param params           动力学参数
    /// @param mpc_options      MPC 求解器配置（dt / N / 限幅 / 权重 / refinement，
    ///                         以及 use_gravity：默认 false = 模型里没有重力项，
    ///                         实测 gx/gy 照常出现在 getState().strict 里但不进模型）
    /// @param wrapper_options  积分补偿配置（两轴各自增益）
    /// @param mpc_loop_period  McuMpcController 后台 loop 周期 [s]（无默认值）
    /// @param mcu_linear_params MCU 数据线性映射标定参数
    /// @param sequence_mode    true 选择序列模式
    RobotController(com::FullStrictPoseBuilder::ImuLocation imu_location,
                    const dm::Params& params,
                    const mpc::MPCController::Options& mpc_options,
                    const mpc::DualYawMpcController::Options& wrapper_options,
                    double mpc_loop_period,
                    const com::McuDataPreprocessor::LinearParams& mcu_linear_params,
                    bool sequence_mode = false);
    ~RobotController();

    RobotController(const RobotController&) = delete;
    RobotController& operator=(const RobotController&) = delete;

    /// 统一获取：MCU / IMU 原始数据 + 严格反解包 + MPC 状态（含参考/预测序列与积分值）
    State getState();

    /// 直通 McuMpcController::set（单目标模式）。
    /// 序列模式下调用本接口抛出 std::runtime_error。
    void set(bool auto_aim_enable, bool yaw_torque_only_mode, double target_psi_b,
             double target_psi_s, double pitch_target_angle, bool fire,
             bool integral_enable);

    /// 直通 McuMpcController 序列版 set。单目标模式下调用本接口抛出 std::runtime_error。
    void set(bool auto_aim_enable, bool yaw_torque_only_mode,
             const std::vector<double>& target_psi_b_seq,
             const std::vector<double>& target_psi_s_seq,
             const std::vector<double>& pitch_seq,
             const std::vector<bool>& fire_seq, bool integral_enable);

    Mode mode() const { return sequence_mode_ ? Mode::SEQUENCE : Mode::SINGLE; }

    // ── 内部子模块的只读访问（标定/调试用）──
    const com::RobotCommunication& comm() const { return comm_; }
    com::RobotCommunication& comm() { return comm_; }

private:
    bool sequence_mode_ = false;
    com::RobotCommunication comm_;
    mpc::McuMpcController mcu_mpc_;
};

}  // namespace tcbs
