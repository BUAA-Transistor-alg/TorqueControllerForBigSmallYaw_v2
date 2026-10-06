// FullStrictPoseBuilder.h — 「全量严格反解数据包」构建器
//
// ── 来源与定位 ──
//   本类的**用法**仿照 /home/huhu233/rm2027/TorqueController 中 `YawChassisFusion fusion_`
//   在 `tcs::RobotCommunication` 里的接法（见该仓库 include/tcs/communication/Communications.hpp）:
//     · 作为 RobotCommunication 的成员，默认构造；
//     · IMU 回调里喂 onImu(...)；MCU 回调里喂 onMcu(...)（喂的是**预处理后**的数据）；
//     · 对外提供只读的 strictPose()（严格反解包）。
//   数据结构则对齐 TorqueControllerForBigSmallYaw 的 `dual_yaw::StrictPose`
//   （反解输入快照 + 底盘/各环节姿态与方位角 + 重力分量），故名「全量」。
//
// ── 现状 ──
//   onImu / onMcu 把打包好的样本存进成员变量（从未传入过的样本保持全 0）；
//   strictPose() 在锁内取一致快照后**做完整反解**：
//     · 按安装构型由 IMU 欧拉角与关节角反解底盘姿态 R_chassis → chassis_euler_*；
//     · 再把它投影到纯绕 Z 的旋转（projectToZRotation）得到 chassis_azimuth，
//       以及旋转平面内的重力分量 projX / projY → gx = projX·g、gy = projY·g
//       （底盘水平时二者为 0，倾斜时 |(gx,gy)| = g·sin(倾角)）；
//     · big_azimuth = chassis_azimuth + θ_b，small_azimuth = big_azimuth + θ_s；
//     · 各环节方位角速度由 IMU 陀螺 gz/gy 与关节角速度按构型链式合成。
//   角度由 atan2 / asin 给出，天然落在 (−π, π]（pitch ∈ [−π/2, π/2]）。
//
//   ★ 唯一的例外是 chassis_azimuth：它被**累计圈数解卷绕成多圈连续量**（见 StrictPose
//     的字段注释与 unwrapChassisAzimuth），因为下游按「多圈世界方位角」使用它。其余
//     atan2/asin 输出（chassis_euler_*）保持 (−π, π] 不变——那两个是给变换树用的
//     包裹欧拉角，语义不同，禁止顺手"一起解卷绕"。
//
// ── 约定 ──
//   · 与 `dual_yaw::StrictPose` 一致: **没有 valid 标志，始终可读**；
//     所需数据缺失时以历史值或 0 参与计算。
//   · 与 RobotTfTree 一致的 ZXY 约定: R = Rz(yaw)·Rx(pitch)·Ry(roll)。
//   · 运动学链（IMU 安装位置可配置，语义见下方 imu_location）:
//       ON_BIG_YAW (0): R_world_imu = R_world_chassis · Rz(θ_b) · R_A_IMU
//       ON_HEAD    (1): R_world_imu = R_world_chassis · Rz(θ_b) · Rz(θ_s) · Rx(θ_p) · R_H_IMU
#ifndef TCBS_COM_FULL_STRICT_POSE_BUILDER_H
#define TCBS_COM_FULL_STRICT_POSE_BUILDER_H

#include <mutex>

namespace tcbs {
namespace com {

class FullStrictPoseBuilder {
public:
    // 严格反解数据包（全量版）: 既给出反解结果，也带齐"反解时用到的每一项数据"，
    // 使外部只依赖这一个包就能重构整车姿态。
    struct StrictPose {
        // ── 正解所需角度 ──
        double yaw_big_angle = 0.0;      // θ_b（大 yaw 关节角）
        double yaw_small_angle = 0.0;    // θ_s（小 yaw 关节角）
        double pitch_angle = 0.0;        // θ_p（pitch 关节角）
        // ── 反解结果 ──
        // chassis_euler_*：ZXY 欧拉角，**包裹在 (−π, π]**（变换树/流水线按包裹值使用）。
        double chassis_euler_yaw = 0.0, chassis_euler_pitch = 0.0, chassis_euler_roll = 0.0;
        // ── 各环节的世界方位角 ──
        // ★ 三者都是**多圈连续量**（不是 (−π,π] 包裹值）：
        //     chassis_azimuth = 累计圈数解卷绕后的底盘方位角；
        //     big_azimuth     = chassis_azimuth + θ_b（θ_b 是电控给的多圈关节角）。
        //   mpc::DualYawMpcController::measure() 正是按此口径合成
        //   psi_b = chassis_azimuth + θ_b，若这里给包裹值，底盘每转一圈 psi_b 就跳 ±2π，
        //   MPC 代价会出现整圈残差（每圈一个力矩脉冲）。
        double chassis_azimuth = 0.0;
        double big_azimuth = 0.0;
        double small_azimuth = 0.0;
        // ── 重力在旋转平面内的分量 ──
        double gx = 0.0;
        double gy = 0.0;
        // ── 角速度 ──
        double small_motor_omega = 0.0;
        double small_azimuth_omega = 0.0;
        double big_motor_omega = 0.0;
        double big_azimuth_omega = 0.0;
        double chassis_omega = 0.0;
    };

    // ── 输入样本（由 RobotCommunication 在回调里打包后传入）──
    // IMU 高频样本（不再需要陀螺 gx/gy/gz）。
    struct ImuSample {
        double euler_yaw = 0.0;     // 世界←IMU 的欧拉角（rad, ZXY）
        double euler_pitch = 0.0;
        double euler_roll = 0.0;
        double gx = 0.0;
        double gy = 0.0;
        double gz = 0.0;
    };

    // MCU 低频样本（喂**预处理后**的数据；不再需要 mcu2_seq）。
    struct McuSample {
        double yaw_big_angle = 0.0;     // 大 yaw 关节角 θ_b（链路低频且可能被保持）
        double yaw_big_omega = 0.0;     // 大 yaw 关节角速度
        double yaw_small_angle = 0.0;   // 小 yaw 关节角 θ_s（实时可信）
        double yaw_small_omega = 0.0;   // 小 yaw 关节角速度
        double pitch_angle = 0.0;       // pitch 关节角 θ_p
        double chassis_imu_yaw = 0.0;   // 底盘自身 IMU 的 yaw
        double chassis_imu_omega = 0.0; // 底盘 yaw 角速度
    };

    enum class ImuLocation { ON_BIG_YAW = 0, ON_HEAD = 1 };

    FullStrictPoseBuilder(ImuLocation imu_location = ImuLocation::ON_HEAD) {imu_location_ = imu_location;}

    FullStrictPoseBuilder(const FullStrictPoseBuilder&) = delete;
    FullStrictPoseBuilder& operator=(const FullStrictPoseBuilder&) = delete;

    // 高频路径：每个 IMU 包调用一次，保存最新样本（线程安全）。
    void onImu(const ImuSample& sample);

    // 低频路径：每个 MCU 包调用一次（喂**预处理后**的数据），保存最新样本（线程安全）。
    void onMcu(const McuSample& sample);

    // 读取严格反解数据包（线程安全：回调线程写、主线程读；始终可读，无 valid 标志）。
    // 反解输入快照取自最近缓存的样本；样本从未传入过时对应字段为 0。
    StrictPose strictPose() const;


private:
    // ── 底盘方位角的**累计圈数解卷绕**（见 .cpp 里的实现与说明）──
    //   matToEulerZXY 用 atan2 给出的 azimuth 落在 (−π, π]：底盘每转一圈就跳 ±2π。
    //   这里以「上一拍输出」为参考、把 ±2π 的跳变吸收进 turn，使输出成为多圈连续量。
    //   语义：顺序滤波器。必须与样本快照同序调用；重复喂同一个快照时增量为 0（幂等），
    //   因此多线程各自调用 strictPose() 也不会把圈数累坏。
    double unwrapChassisAzimuth(double wrapped) const;

    mutable std::mutex mtx_;   // 保护内部状态（回调线程写、主线程读）
    ImuSample imu_;            // 最近一次 IMU 样本（从未传入则全 0）
    McuSample mcu_;            // 最近一次 MCU 样本（从未传入则全 0）

    ImuLocation imu_location_;

    double g_ = 9.81;

    // 解卷绕状态（独立锁：与上面的样本快照互不阻塞）
    mutable std::mutex azimuth_mtx_;
    mutable bool   chassis_unwrap_init_ = false;  // 是否已锁存首个样本
    mutable double chassis_azimuth_last_ = 0.0;   // 上一拍输出的多圈值（rad）
    mutable double chassis_azimuth_turn_ = 0.0;   // 累计圈数偏移（2π 的整数倍，rad）
};

}  // namespace com
}  // namespace tcbs

#endif // TCBS_COM_FULL_STRICT_POSE_BUILDER_H
