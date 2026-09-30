// FullStrictPoseBuilder.h — 「全量严格反解数据包」构建器（骨架；具体处理内容待实现）
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
// ── 现状（重要）──
//   ★ **具体处理内容暂时留空**: onImu / onMcu 只把打包好的样本存进成员变量
//     （从未传入过的样本保持全 0）；strictPose() 在锁内用这两个成员填充
//     StrictPose 的反解输入快照。反解算法（底盘姿态、方位角、重力分量）尚未实现，
//     相关字段暂为 0。填空时只需改本类的 .cpp，RobotCommunication 一侧无需再动。
//
// ── 约定 ──
//   · 与 `dual_yaw::StrictPose` 一致: **没有 valid 标志，始终可读**；
//     所需数据缺失时以历史值或 0 参与计算，角度最终 wrap 到 (−π, π]（wrap 待实现）。
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
        double chassis_euler_yaw = 0.0, chassis_euler_pitch = 0.0, chassis_euler_roll = 0.0;
        // ── 各环节的世界方位角 ──
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

    FullStrictPoseBuilder() = default;

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
    mutable std::mutex mtx_;   // 保护内部状态（回调线程写、主线程读）
    ImuSample imu_;            // 最近一次 IMU 样本（从未传入则全 0）
    McuSample mcu_;            // 最近一次 MCU 样本（从未传入则全 0）

    double g_ = 9.81;
};

}  // namespace com
}  // namespace tcbs

#endif // TCBS_COM_FULL_STRICT_POSE_BUILDER_H
