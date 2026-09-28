#pragma once

// 系统参数辨识模式：对 14 个动力学参数求损失的解析梯度。
//
// 与 include/trajectory.hpp 的区别（两者互不影响）：
//   * trajectory.hpp 求 dL/d(tau)：力矩是决策变量，用离散伴随反向传播；
//   * 本文件求 dL/dp：力矩序列是**已知输入**，14 个动力学参数是决策变量。
//     参数维数固定且与步数 K 无关，因此用前向灵敏度即可，且**不需要反向扫描**、
//     不需要保存历史状态：
//         dX_{k+1}/dp = Phi_k * dX_k/dp + Jp_k
//         dL/dp       = sum_k (dL/dX_k)^T * (dX_k/dp)
//     其中 Jp_k 只含当前步的局部信息（参数不出现在历史项里）。
//
// 损失（全局量 psi 的四项时间均值加权和，k = 0..K-1）：
//   L = (1/2K) sum_k [ w_pb (psi_b[k]-psi_b*[k])^2 + w_ps (psi_s[k]-psi_s*[k])^2
//                    + w_vb (dpsi_b[k]-dpsi_b*[k])^2 + w_vs (dpsi_s[k]-dpsi_s*[k])^2 ]
//   psi_b = theta_c + theta_b,  psi_s = theta_c + theta_b + theta_s
//   dpsi_b = dtheta_c + dtheta_b, dpsi_s = dtheta_c + dtheta_b + dtheta_s
//
// 被辨识参数（固定顺序，共 14 个）：
//   0 mb   1 Ib   2 Pbx  3 Pby  4 ms   5 Is   6 Psx  7 Psy
//   8 Dx   9 Dy  10 fbc 11 fbv 12 fsc 13 fsv
//   （kParamGradientCount / paramGradientNames() 给出权威值与顺序）
//
// 不在其中的量：
//   * gx / gy：重力矢量是**随采集数据一起给出的已知输入**（存在 Params 里
//     供正演使用），不是被辨识参数；
//   * lambda：平滑摩擦参数，固定常数。

#include <cstddef>

#include "params.hpp"
#include "state.hpp"

namespace tcbss {

/// refinement 为运行期参数（语义见 trajectory.hpp：每主步重复 refinement 次经典 RK4）。

/// 参与辨识的参数个数。
inline constexpr int kParamGradientCount = 14;

/// 参数名（用于 Python 侧校验顺序）。
const char* const* paramGradientNames();

/// 损失规格：四项权重 + 目标序列（长度均为 K；nullptr 表示目标恒为 0）。
struct ParamLossSpec {
    double w_psi_b = 0.0;
    double w_psi_s = 0.0;
    double w_dpsi_b = 0.0;
    double w_dpsi_s = 0.0;
    const double* target_psi_b = nullptr;
    const double* target_psi_s = nullptr;
    const double* target_dpsi_b = nullptr;
    const double* target_dpsi_s = nullptr;
};

/// 调用方分配一次、反复复用的缓冲。
struct ParamGradientWorkspace {
    std::size_t num_steps = 0;
    int refinement = 0;
    std::size_t num_substeps = 0;
    /// 各步末状态，长度 K（损失是在步末状态上定义的）。
    State* X = nullptr;
    /// 各子步起始状态，长度 K*refinement（损失梯度回传需要）。
    State* Xsub = nullptr;
    /// 各子步的四阶段状态雅可比 Jx（前向灵敏度传播需要），4*16 doubles/子步。
    double* Jx = nullptr;
    /// 损失对各子步末状态的梯度，4 doubles/子步。
    double* dLdX = nullptr;
    /// 单个子步的 4 个阶段缓冲（运行期分配，避免热路径反复构造 Eval）。
    void* evbuf = nullptr;

    bool resize(std::size_t steps, int refine);
    void release();
    ~ParamGradientWorkspace() { release(); }
    ParamGradientWorkspace() = default;
    ParamGradientWorkspace(const ParamGradientWorkspace&) = delete;
    ParamGradientWorkspace& operator=(const ParamGradientWorkspace&) = delete;
};

/// 配置校验；返回 nullptr 表示合法。
const char* validateParamGradientConfig(const Params& p, std::size_t num_steps, double dt,
                                        int refinement);

/// 仅正向：计算损失值，并可选导出四个全局量序列。
double computeParamGradientLoss(const Params& p,
                                double theta_c0,
                                double dtheta_c,
                                double ddtheta_c,
                                double dt,
                                int refinement,
                                std::size_t num_steps,
                                const double* tau,
                                const ParamLossSpec& spec,
                                const State& x0,
                                double* out_psi_b,
                                double* out_psi_s,
                                double* out_dpsi_b,
                                double* out_dpsi_s);

/// 正向 + 前向参数灵敏度，返回损失并写出 dL/dp（长度 kParamGradientCount）。
///
/// @param tau        长度 K*2 的已知力矩序列，不可为 nullptr
/// @param grad_p     输出，长度 kParamGradientCount
/// @param out_final_state 可为 nullptr
double computeParamGradient(const Params& p,
                            double theta_c0,
                            double dtheta_c,
                            double ddtheta_c,
                            double dt,
                            int refinement,
                            std::size_t num_steps,
                            const double* tau,
                            const ParamLossSpec& spec,
                            const State& x0,
                            ParamGradientWorkspace& ws,
                            double* grad_p,
                            State* out_final_state);

}  // namespace tcbss
