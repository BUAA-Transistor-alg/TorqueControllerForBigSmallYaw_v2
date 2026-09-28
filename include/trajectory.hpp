#pragma once

// 有限多步序列仿真 + 标量损失对每一步控制力矩的解析梯度（离散伴随法）。
//
// 与 Simulator 的区别：
//   * 一次调用处理整条 K 步序列，正向轨迹与反向伴随在同一个函数内完成；
//   * 反向只传播 4 维协态，不构造 K x 4 x 2K 的灵敏度张量；
//   * 正向每个 RK4 子步的阶段雅可比 (Jx, Ju) 在同一次三角运算中算出并被记录，
//     反向直接复用，因此**灵敏度传播不引入任何额外的 sin/cos/tanh**。
//
// 运动学与动力学约定与 include/dynamics.hpp 完全一致：
//   psi_b = theta_c + theta_b
//   psi_s = theta_c + theta_b + theta_s
//   psi'_b = theta'_c + theta'_b      psi'_s = theta'_c + theta'_b + theta'_s
//
// 损失（时间均值形式，k = 0 .. K-1，共六项）：
//   L = (1/2K) * sum_k [ w1*(psi_b[k]-psi_b*[k])^2
//                      + w2*(psi_s[k]-psi_s*[k])^2
//                      + w3*(psi'_b[k]-psi'_b*[k])^2
//                      + w4*(psi'_s[k]-psi'_s*[k])^2
//                      + w5*tau_b[k]^2
//                      + w6*tau_s[k]^2 ]
//   目标序列为 nullptr 时该项退化为对 0 的偏差（即仅惩罚幅值）。
//
// 约定与假设：
//   * theta_c 序列为外部给定、不参与优化的已知量；每步内基座角加速度常值，
//     位置按 theta_c(tau) = theta_c0 + dtheta_c*tau + 0.5*ddtheta_c*tau^2 外推。
//   * 两个控制力矩在每一步内为常值（零阶保持）。
//   * 步长 dt 相同，每步内做 kTrajectoryRefinement 个 RK4 子步。
//   * 力矩序列按 AoS 存放：tau[i*2+0] = tau_b[i]，tau[i*2+1] = tau_s[i]。

#include <cstddef>

#include "params.hpp"
#include "state.hpp"

namespace tcbss {

// ---------------------------------------------------------------------------
// 常量
// ---------------------------------------------------------------------------

/// 每步内固定的 RK4 子步数（固定容量，避免运行期分配）。
inline constexpr int kTrajectoryRefinement = 4;

/// 配置合法性检查。返回 nullptr 表示合法，否则返回静态错误字符串。
const char* validateTrajectoryConfig(const Params& p, std::size_t num_steps, double dt);

// ---------------------------------------------------------------------------
// 损失规格
// ---------------------------------------------------------------------------

/// 六项的权重与目标序列。
/// 目标序列长度均应为 num_steps；传 nullptr 表示该项目标恒为 0。
struct LossSpec {
    double w_psi_b = 0.0;
    double w_psi_s = 0.0;
    double w_dpsi_b = 0.0;
    double w_dpsi_s = 0.0;
    double w_tau_b = 0.0;
    double w_tau_s = 0.0;

    const double* target_psi_b = nullptr;
    const double* target_psi_s = nullptr;
    const double* target_dpsi_b = nullptr;
    const double* target_dpsi_s = nullptr;
};

// ---------------------------------------------------------------------------
// 正向轨迹记录与工作缓冲（调用方分配一次，可跨多次调用复用）
// ---------------------------------------------------------------------------

/// 正向轨迹记录：每个 RK4 子步一条。
///
/// 只记录反向真正需要的量：
///   * 子步起始状态（损失的状态梯度注入需要它）；
///   * 四个阶段的 Jx / Ju（反向 VJP 直接消费）。
/// 反向不需要阶段导数 k_i，也不需要 Phi/Psi，因此都不保存、不计算
/// （除非显式打开 with_step_jacobians）。
struct TrajectoryRecord {
    State x;                                  ///< 子步起始状态
    double Jx[kTrajectoryRefinement][4][4];    ///< 各阶段 d(f)/d(x)，行主序
    double Ju[kTrajectoryRefinement][4][2];    ///< 各阶段 d(f)/d(u)
    double Phi[4][4];                          ///< 仅当 with_step_jacobians=true 时填充
    double Psi[4][2];                          ///< 仅当 with_step_jacobians=true 时填充
};

/// 调用方一次性分配、反复复用的缓冲。
struct TrajectoryWorkspace {
    std::size_t num_steps = 0;             ///< 已分配的步数 K
    std::size_t num_substeps = 0;          ///< = K * kTrajectoryRefinement
    TrajectoryRecord* records = nullptr;   ///< 长度 num_substeps
    State* X = nullptr;                    ///< 长度 K，各步末状态
    double* dLdX = nullptr;                ///< 长度 K*4，损失对各步末状态的梯度

    /// 分配（仅当步数变化时重新分配）。返回 false 表示分配失败。
    bool resize(std::size_t steps);

    /// 释放全部缓冲。
    void release();

    ~TrajectoryWorkspace() { release(); }

    TrajectoryWorkspace() = default;
    TrajectoryWorkspace(const TrajectoryWorkspace&) = delete;
    TrajectoryWorkspace& operator=(const TrajectoryWorkspace&) = delete;
};

// ---------------------------------------------------------------------------
// 核心运算
// ---------------------------------------------------------------------------

/// 仅前向：计算损失值，并可导出四个全局量序列。
/// @param tau               长度 K*2；为 nullptr 时全部步使用 (tau_b_fixed, tau_s_fixed)
/// @param out_psi_b 等      长度 K 的输出序列，可为 nullptr
double computeTrajectoryLoss(const Params& p,
                             double theta_c0,
                             double dtheta_c,
                             double ddtheta_c,
                             double dt,
                             std::size_t num_steps,
                             const double* tau,
                             double tau_b_fixed,
                             double tau_s_fixed,
                             const LossSpec& spec,
                             const State& x0,
                             double* out_psi_b,
                             double* out_psi_s,
                             double* out_dpsi_b,
                             double* out_dpsi_s);

/// 正向仿真 + 反向伴随，一次算完，返回损失值并写出 dL/dtau。
///
/// @param tau               长度 K*2，不可为 nullptr
/// @param ws                调用方缓冲；内部保证与 num_steps 匹配
/// @param grad_tau          输出，长度 K*2
/// @param out_final_state   输出最后一步之后的状态，可为 nullptr
/// @param with_step_jacobians 是否额外装配每子步的 Phi/Psi（仅当需要整步灵敏度时打开）
/// @return 损失值
double simulateAndGradient(const Params& p,
                           double theta_c0,
                           double dtheta_c,
                           double ddtheta_c,
                           double dt,
                           std::size_t num_steps,
                           const double* tau,
                           const LossSpec& spec,
                           const State& x0,
                           TrajectoryWorkspace& ws,
                           double* grad_tau,
                           State* out_final_state,
                           bool with_step_jacobians);

}  // namespace tcbss
