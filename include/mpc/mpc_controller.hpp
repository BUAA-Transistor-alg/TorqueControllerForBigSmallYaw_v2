#pragma once

// tcbs::mpc::MPCController —— 双连杆（大/小 yaw）力矩 MPC 求解器（Ceres 梯度法）。
//
// 与参考工程 /home/huhu233/rm2027/TorqueController 的 tcs::MPCController 形式对应，
// 但内层模型换成 dm 的双连杆动力学 + RK4，且**不做自动求导**：
//
//   1) 内层 loss 与 dL/dτ 复用 dm::simulateAndGradient 的**解析离散伴随**
//      （include/dm/trajectory.hpp）——力矩是决策量、反向只传播 4 维协态。
//      因此 Ceres 侧用 ceres::FirstOrderFunction + GradientProblemSolver
//      （只需 cost + gradient），而不是 ceres::Problem + 自动求导代价函数。
//
//   2) 优化变量是**预 tanh 力矩的增量序列** Δx（长度 2N，AoS：Δx[2k] 大 yaw、
//      Δx[2k+1] 小 yaw，k = 0..N-1）。两个通道各自累加得到预 tanh 力矩 x：
//          x_c[k] = x_c^prev + Σ_{j=0..k} Δx_c[j]
//      其中 x_c^prev 是上一步**实际施加**的预 tanh 力矩（首步为 0），
//      再经 tanh 重参数化做**软限幅**：
//          τ_c[k] = max_torque_c · tanh(x_c[k])      ⇒ |τ_c| < max_torque_c
//
//   3) 总 loss = 内层 loss + 外层 L2 惩罚（大 / 小 yaw **各自一套权重**）：
//          F(Δx) = L_inner(τ)                     // dm::LossSpec 的六项均值形式
//                + w_x_b  · mean_k( x_b[k]² )     // 大 yaw 预 tanh 值均方
//                + w_x_s  · mean_k( x_s[k]² )     // 小 yaw 预 tanh 值均方
//                + w_dx_b · mean_k( Δx_b[k]² )    // 大 yaw 增量（相邻两步差值）均方
//                + w_dx_s · mean_k( Δx_s[k]² )    // 小 yaw 增量均方
//      每个 mean_k(·) 都是**该通道自己的** N 步时间均值：mean_k(·) = (1/N)·Σ_{k=0..N-1}(·)。
//      即每个权重只管一个通道、按该通道的步数 N 归一，两通道互不影响。
//
//   4) 变化率**不再做范围限制**（参考工程里 max_torque_rate 的参数硬边界已去掉），
//      限速完全由 w_dx_b / w_dx_s 的 L2 惩罚 + tanh 软限幅实现；也没有任何参数上下界。
//
// 内层 loss 的参考量沿用 dm::LossSpec 语义（世界系绝对量）：
//   psi_b = theta_c + theta_b,  psi_s = theta_c + theta_b + theta_s
//   dpsi_b = dtheta_c + dtheta_b, dpsi_s = dtheta_c + dtheta_b + dtheta_s
// 基座运动（theta_c / dtheta_c / ddtheta_c）由 trajectory 内部按等角加速度外推，
// 调用方**不要**在参考序列里再叠加底盘补偿。

#include <cstddef>
#include <string>
#include <vector>

#include "params.hpp"
#include "simulator.hpp"
#include "state.hpp"
#include "trajectory.hpp"

namespace tcbs {
namespace mpc {

class MPCController {
public:
    // ------------------------------------------------------------------
    // 配置（除物理参数外全部集中在这里，均带默认值）
    // ------------------------------------------------------------------
    struct Options {
        double dt = 0.0;         ///< 控制周期 [s]（= 预测步长），必须 > 0

        // 每个控制周期内的 RK4 子步数，[1, 4096]。
        //
        // ★ 稳定性要求（重要）：子步长 h = dt/refinement 必须小到能让 RK4 稳定地
        //   线性化库仑摩擦的刚性项。摩擦对速度的导数最大为 fbv + fbc·λ（λ 通常 100），
        //   经质量矩阵逆放大后，要求 h·(fbv+fbc·λ)·‖M⁻¹‖ ≲ 2.785（RK4 实轴稳定界）。
        //   不满足时：**正向轨迹仍然正常**，但解析伴随（dm::simulateAndGradient）
        //   会被数值不稳定逐子步放大成垃圾梯度 —— Ceres 随即以"参数容差"在
        //   0~2 次迭代后返回初值，表现为**输出恒为 0 力矩（控制器看起来是死的）**。
        //   实测（dt=0.01、λ=100、fbc=fsc=0.1、fbv=fsv=0.05）：
        //     refinement=4  → 每子步传播子 σ_max=5.11、整段连乘 10^56.7、梯度完全错误
        //     refinement>=6 → σ_max=1.000、梯度与有限差分吻合到 1e-8
        //   底盘水平/重力≈0 且关节接近静止时最容易踩到（此时 1-tanh²(λω) 取最大）。
        int    refinement = 1;

        int    N = 0;            ///< 预测步数，>= 1

        double max_torque_b = 0.0;  ///< 大 yaw 软限幅（|τ_b| < 该值），必须 > 0
        double max_torque_s = 0.0;  ///< 小 yaw 软限幅（|τ_s| < 该值），必须 > 0

        // ---- 内层 loss 权重（dm::LossSpec 语义，均为时间均值形式）----
        double w_psi_b = 0.0;    ///< 大 yaw 世界方位角跟踪
        double w_psi_s = 0.0;    ///< 小 yaw（头部）世界方位角跟踪
        double w_dpsi_b = 0.0;   ///< 大 yaw 世界角速度跟踪
        double w_dpsi_s = 0.0;   ///< 小 yaw 世界角速度跟踪
        double w_tau_b = 0.0;    ///< 大 yaw 力矩幅值
        double w_tau_s = 0.0;    ///< 小 yaw 力矩幅值

        // ---- 外层 L2 惩罚权重（作用在预 tanh 量上，大小 yaw 各一份）----
        // 注意：w_x_b / w_x_s 建议**必须 > 0**。tanh 在饱和区 dτ/dx = τ_max·(1-tanh²x)
        // 指数趋零，若完全不惩罚 x，x 会漂到饱和区（|x| ≳ 30）使解析梯度数值上变成 0，
        // 梯度法将无法退出饱和（Ceres 会以 0 次迭代直接返回）。w_x_*·mean(x²) 正是把 x
        // 约束在 tanh 线性区、保住梯度的机制；w_dx_* 只惩罚增量，无法约束 x 的常数漂移。
        double w_x_b = 0.0;      ///< 大 yaw mean(x_b²)，x 为预 tanh 力矩（建议 > 0，见上）
        double w_x_s = 0.0;      ///< 小 yaw mean(x_s²)，同上
        double w_dx_b = 0.0;     ///< 大 yaw mean(Δx_b²)，Δx 为预 tanh 力矩的相邻两步差值
        double w_dx_s = 0.0;     ///< 小 yaw mean(Δx_s²)，同上

        int max_iter = 50;       ///< Ceres LBFGS 最大迭代次数

        /// 是否把 Params::gx / gy（旋转平面内的重力分量）真正送进动力学模型。
        ///
        /// **默认 true = 使用重力**：每步由上层调 setGravity() 把实测 gx/gy 写进模型，
        /// 重力矩因此进入预测，控制器才能真正"知道"要扛多少重力。
        ///
        /// false 时求解器内部**恒按 (0,0) 处理**：构造时清零、setParams()
        /// 写入的重力也清零、setGravity() 无论收到什么值都按 0 存（实测值传进来
        /// 也不会进入模型）。模型里完全没有重力项，重力矩只能靠反馈/积分扛
        /// —— 这是"不用重力"的对照实验，需要显式打开。
        bool use_gravity = true;
    };

    // ------------------------------------------------------------------
    // 参考（目标）序列：世界系绝对量，长度要么为空（= 目标 0），要么 >= N。
    // 只取前 N 个；空表示该项目标恒为 0（配合对应权重即惩罚幅值）。
    // ------------------------------------------------------------------
    struct Reference {
        std::vector<double> psi_b;    ///< 目标 psi_b*（长度 N 或空）
        std::vector<double> psi_s;    ///< 目标 psi_s*（长度 N 或空）
        std::vector<double> dpsi_b;   ///< 目标 dpsi_b*（长度 N 或空）
        std::vector<double> dpsi_s;   ///< 目标 dpsi_s*（长度 N 或空）
    };

    // ------------------------------------------------------------------
    // 单步求解结果
    // ------------------------------------------------------------------
    struct Result {
        bool   ok = false;       ///< Ceres 是否给出可用解（否则回退到热启动初值）
        double loss = 0.0;       ///< 最终总 loss F(Δx)

        double torque_b = 0.0;   ///< 第一步大 yaw 力矩 [N·m]
        double torque_s = 0.0;   ///< 第一步小 yaw 力矩 [N·m]

        /// 施加第一步力矩、经过一个控制周期后的状态（用 dm::Simulator 积分，与
        /// 内层模型一致）。
        dm::State predicted{};

        /// 一个控制周期后的世界方位角 / 角速度（= 预测序列的第 0 项）。
        double pred_psi_b = 0.0;
        double pred_psi_s = 0.0;
        double pred_dpsi_b = 0.0;
        double pred_dpsi_s = 0.0;

        /// 优化出的完整力矩序列（长度 N）。
        std::vector<double> tau_b_seq;
        std::vector<double> tau_s_seq;

        /// 最优力矩下的预测世界量序列（长度 N，均为各步**步末**的值）。
        std::vector<double> pred_psi_b_seq;
        std::vector<double> pred_psi_s_seq;
        std::vector<double> pred_dpsi_b_seq;
        std::vector<double> pred_dpsi_s_seq;
    };

    /// @param params  动力学 / 重力 / 摩擦参数
    /// @param options 控制周期、步数、限幅、权重等（见 Options）
    /// @throw std::invalid_argument 配置非法（dt <= 0、N < 1、refinement 越界、
    ///        限幅 <= 0、权重 < 0、dm 配置校验失败）
    MPCController(const dm::Params& params, const Options& options);

    MPCController(const MPCController&) = delete;
    MPCController& operator=(const MPCController&) = delete;

    // ------------------------------------------------------------------
    // 求解一步：返回第一步力矩（滚动时域只在外部施加第一步）
    //
    // @param x0        当前两个广义坐标及其速度
    // @param theta_c   当前基座角（trajectory 内部按其等角加速度外推整段预测）
    // @param dtheta_c  当前基座角速度
    // @param ddtheta_c 基座角加速度（整段预测内常值）
    // @param reference 世界系参考序列（见 Reference）
    // ------------------------------------------------------------------
    Result step(const dm::State& x0,
                double theta_c,
                double dtheta_c,
                double ddtheta_c,
                const Reference& reference);

    /// 清空热启动状态（上一步最优序列 / 上一步实际施加的预 tanh 力矩）。
    void reset();

    // ------------------------------------------------------------------
    // 运行期参数更新
    //
    // 用途：底盘俯仰/横滚变化时，旋转平面内的重力分量 (gx,gy) 会随之改变
    // （|(gx,gy)| = g·sin(倾角)），上层每步用 FullStrictPoseBuilder 反解出的
    // StrictPose::gx/gy 调 setGravity() 即可，无需重建求解器。
    // 目标值/权重/步长等配置不受影响，热启动序列与工作缓冲保持有效。
    // ------------------------------------------------------------------

    /// 运行期替换全部动力学参数（会重新做 dm 配置校验，非法则抛异常）。
    /// Options::use_gravity = false 时，写入的 gx/gy 会被清零（模型无重力项）；
    /// 默认 true，写入的实测重力会真正进入模型。
    void setParams(const dm::Params& params);

    /// 运行期更新旋转平面内的重力分量（只改 Params::gx / gy）。
    /// 默认（use_gravity = true）按实测值存、模型会用它；
    /// 显式关掉对照模式（use_gravity = false）时这里恒按 (0,0) 存。
    void setGravity(double gx, double gy);

    // ------------------------------------------------------------------
    // 只读访问
    // ------------------------------------------------------------------
    const dm::Params& params() const { return params_; }
    const Options&    options() const { return opt_; }
    int    horizon()     const { return opt_.N; }
    int    numParameters() const { return 2 * opt_.N; }
    double maxTorqueB()  const { return opt_.max_torque_b; }
    double maxTorqueS()  const { return opt_.max_torque_s; }

    /// 上一次 step 的最优预 tanh 力矩序列（长度 2N，AoS）。
    const std::vector<double>& lastX() const { return x_prev_; }
    /// 上一次 Ceres 的结束信息（调试用）。
    const std::string& lastSolverMessage() const { return last_message_; }
    /// 上一次 Ceres 的迭代次数。
    int lastIterations() const { return last_iterations_; }
    /// 上一次 Ceres 报告的初始目标值（热启动初值处的 F）。
    double lastInitialCost() const { return initial_cost_; }
    /// 上一次 Ceres 报告的最终目标值（收敛后的 F，与 Result::loss 一致）。
    double lastFinalCost() const { return final_cost_; }

    // ------------------------------------------------------------------
    // 目标函数（供 ceres::FirstOrderFunction 调用；也便于数值差分校验）
    //
    // 返回 F(Δx)；gradient 非空时写出 ∂F/∂Δx（长度 2N）。
    // 注意：它读取上一次 step() 传入的 x0 / 基座量 / 参考序列，因此**只在
    // step() 内部或紧随其后**有意义。
    // ------------------------------------------------------------------
    double evaluate(const double* delta, double* gradient) const;

private:
    /// 由增量序列重建每步的预 tanh 力矩 x 与限幅后的力矩 τ（长度均 2N）。
    void buildTorque(const double* delta, double* x_out, double* tau_out) const;

    /// 校验参考序列长度（空或 >= N）。
    void validateReference(const Reference& reference) const;

    /// 按当前 ref_ 组装 dm::LossSpec。
    dm::LossSpec makeLossSpec() const;

    dm::Params params_;
    Options    opt_;

    // ---- 本次求解的输入（step() 写入，evaluate() 只读）----
    dm::State   x0_{};
    double      theta_c_ = 0.0;
    double      dtheta_c_ = 0.0;
    double      ddtheta_c_ = 0.0;
    Reference   ref_;

    // ---- 热启动 ----
    std::vector<double> x_prev_;        ///< 上一次最优预 tanh 序列（长度 2N）
    double x_applied_b_ = 0.0;          ///< 上一步实际施加的预 tanh 力矩（大 yaw）
    double x_applied_s_ = 0.0;          ///< 上一步实际施加的预 tanh 力矩（小 yaw）

    // ---- 复用缓冲（Ceres 会反复调用 evaluate，避免热路径反复分配）----
    mutable dm::TrajectoryWorkspace ws_;          ///< 正向记录 + 反向协态缓冲
    mutable std::vector<double> x_scratch_;       ///< 预 tanh 序列，长度 2N
    mutable std::vector<double> tau_scratch_;     ///< 力矩序列，长度 2N
    mutable std::vector<double> grad_tau_;        ///< dL_inner/dτ，长度 2N
    mutable std::vector<double> grad_x_;          ///< dF/dx，长度 2N

    /// 一步预测用的积分器（与内层 trajectory 的积分方式一致）。
    dm::Simulator one_step_sim_;

    // ---- 求解统计 ----
    std::string last_message_;
    int         last_iterations_ = 0;
    double      initial_cost_ = -1.0;
    double      final_cost_ = -1.0;
};

}  // namespace mpc
}  // namespace tcbs
