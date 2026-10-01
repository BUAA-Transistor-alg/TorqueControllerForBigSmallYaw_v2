#pragma once

// 批量（SoA）参数辨识模式：一次对 **一批** 序列求损失与 dL/dp。
//
// 与 include/dm/param_gradient.hpp 的关系：
//   * 数学定义、损失、基座外推约定、被辨识参数顺序**完全一致**（见那边文件头）；
//   * 那边一次只处理一条序列；本文件处理 B 条，且**所有数组以样本维为最内层**
//     （SoA：同一样本的不同量不连续，不同样本的同一量连续），从而可以
//     1) 对样本维做 SIMD 向量化；2) 把样本切成若干块交给多个线程。
//   * 单序列入口、trajectory 的力矩梯度、Simulator **都不受影响**。
//
// ---------------------------------------------------------------------------
// 内存布局（所有数组都按 "量 × 样本" 的扁平 SoA 存放，样本维 b 连续）
// ---------------------------------------------------------------------------
//   params[q*B + b]      q = 0..16，顺序同 Params 字段：
//                        mb Ib Pbx Pby ms Is Psx Psy Dx Dy gx gy fbc fbv fsc fsv lambda
//   x0[i*B + b]          i = 0..3：theta_b dtheta_b theta_s dtheta_s
//   base[j*B + b]        j = 0..2：theta_c0 dtheta_c ddtheta_c
//   tau[(2k+c)*B + b]    k = 0..K-1，c = 0/1：tau_b / tau_s
//   target_*(k*B + b)    k = 0..K-1；指针为 nullptr 表示该项目标恒为 0
//   weights[q*B + b]     q = 0..3：w_psi_b w_psi_s w_dpsi_b w_dpsi_s
//   out_loss[b]                     长度 B
//   out_grad[j*B + b]    j = 0..13：14 个可辨识参数（顺序同 paramGradientNames()）
//
// 同一批内所有样本共享 num_steps / dt / refinement。

#include <cstddef>

#include "param_gradient.hpp"
#include "params.hpp"
#include "state.hpp"

namespace tcbs {
namespace dm {

/// params 每个样本的量个数（= Params 的 17 个字段）。
inline constexpr int kBatchParamCount = 17;

/// 内部缓冲。创建/扩容一次后可反复复用（跨 batch、跨 num_steps）。
///
/// 线程模型：`num_threads <= 0` 表示用 std::thread::hardware_concurrency()。
/// `lanes` 是每次 SIMD 分块处理的样本数上限（默认 16），越大 SIMD 越宽但
/// 中间量占用的缓冲越大。
struct ParamGradientBatchWorkspace {
    int refinement = 0;
    int max_batch = 0;
    std::size_t max_num_steps = 0;
    int num_threads = 1;
    int lanes = 16;

    struct Impl;
    Impl* impl = nullptr;

    ParamGradientBatchWorkspace() = default;
    ~ParamGradientBatchWorkspace();
    ParamGradientBatchWorkspace(const ParamGradientBatchWorkspace&) = delete;
    ParamGradientBatchWorkspace& operator=(const ParamGradientBatchWorkspace&) = delete;

    /// 分配 / 复用内部缓冲。返回 false 表示参数非法或分配失败。
    bool resize(int refine, int batch, std::size_t steps, int threads, int lanes_);

    /// 释放全部缓冲。
    void release();
};

/// 批量正向 + 前向参数灵敏度。
///
/// @param params     长度 17*B，见上面的布局说明
/// @param x0         长度 4*B
/// @param base       长度 3*B
/// @param tau        长度 2*K*B
/// @param target_*   长度 K*B，可为 nullptr（该项目标恒为 0）
/// @param weights    长度 4*B
/// @param batch      B，必须 >= 1 且 <= ws.max_batch
/// @param num_steps  K，必须 >= 1 且 <= ws.max_num_steps
/// @param out_loss   输出，长度 B
/// @param out_grad   输出，长度 14*B
///
/// 失败抛 std::invalid_argument / std::runtime_error。
void computeParamGradientBatch(const double* params,
                               const double* x0,
                               const double* base,
                               const double* tau,
                               const double* target_psi_b,
                               const double* target_psi_s,
                               const double* target_dpsi_b,
                               const double* target_dpsi_s,
                               const double* weights,
                               int batch,
                               std::size_t num_steps,
                               double dt,
                               ParamGradientBatchWorkspace& ws,
                               double* out_loss,
                               double* out_grad);

}  // namespace dm
}  // namespace tcbs
