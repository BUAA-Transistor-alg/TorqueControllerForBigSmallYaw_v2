#pragma once

// tcbs::mpc::angle —— 角度「整圈对齐」工具（纯头文件、无外部依赖，便于单独单测）
//
// ── 为什么需要它（★ 这是一次真实故障的修复件，改之前请读完）──
//   严格反解包 FullStrictPoseBuilder::StrictPose 里的方位角由 atan2 给出，落在 (−π, π]
//   （单圈）。而 mpc::DualYawMpcController::measure() 按
//       psi_b = chassis_azimuth + theta_b
//   合成世界方位角，其中 theta_b 是电控上报的**多圈**关节角。两者「圈」的口径不一致时，
//   底盘每转一圈 psi_b 就跳 ±2π；而参考序列只在 McuMpcController::set()（= 上位机每帧
//   下发一次）按当时的实测值对齐过一次，100 Hz 的拍之间**不再对齐** ⇒ 底盘跨过 ±π 到
//   下一帧之间，MPC 代价里出现 w_psi_b·(2π)² ≈ 39.5 的整圈残差（正常跟踪时 ~4e-4）
//   ⇒ 每个底盘整圈打出一个恒定幅值的力矩脉冲（大 yaw 抖一下）。
//
//   修复分两层，本文件是第 2 层：
//     1) FullStrictPoseBuilder 累计圈数，把 chassis_azimuth 变成**多圈连续量**（根因修复）；
//     2) 本文件：MPC **每拍**把参考整体平移整数圈、对齐到当拍实测世界方位角（兜底）。
//        即使将来还有别的环节引入整圈表示差，代价里也不会再出现 2π 量级残差。
//
// ── 语义约定 ──
//   · 所有函数只做 **2π 的整数倍平移**：绝不改变序列的形状（斜率、曲率、各点相对关系），
//     也绝不改变传进来的小偏差（跟踪误差原样保留）；
//   · 两个角本来就相差 < π 时是**恒等变换**（正常运行路径零副作用、零分配）。
#ifndef TCBS_MPC_ANGLE_WRAP_HPP
#define TCBS_MPC_ANGLE_WRAP_HPP

#include <cmath>
#include <vector>

namespace tcbs {
namespace mpc {

/// 2π 的整数倍平移量（rad）。
inline double turnShift(double v, double ref) {
    constexpr double kTwoPi = 2.0 * 3.14159265358979323846;
    return kTwoPi * std::round((ref - v) / kTwoPi);
}

/// 把角 v 平移**整数圈**到与 ref 相差最小的等效角：结果 ≡ v (mod 2π) 且 |结果 − ref| ≤ π。
/// v 与 ref 本来就相差 < π 时返回 v 本身（恒等）。
inline double wrapToNearest(double v, double ref) {
    return v + turnShift(v, ref);
}

/// 把整条参考序列**整体**平移同样的整数圈，使首元素落在与 ref 最近的同圈。
/// 序列为空 ⇒ 空操作；首元素与 ref 相差 < π ⇒ 恒等（序列逐元素原样返回）。
inline void alignSeqToNearest(std::vector<double>& seq, double ref) {
    if (seq.empty()) return;
    const double shift = turnShift(seq.front(), ref);
    if (shift == 0.0) return;
    for (double& v : seq) v += shift;
}

}  // namespace mpc
}  // namespace tcbs

#endif  // TCBS_MPC_ANGLE_WRAP_HPP
