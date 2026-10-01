#include "mpc_controller.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <ceres/ceres.h>
#include <ceres/gradient_problem.h>
#include <ceres/gradient_problem_solver.h>

namespace tcbs {
namespace mpc {
namespace {

// ============================================================================
// Ceres 适配层：唯一职责是把 MPCController::evaluate 的解析 cost/gradient
// 转交给 GradientProblemSolver。
//
// 这里**没有**任何自动求导 / 数值微分：梯度由 dm::simulateAndGradient 的
// 离散伴随 + tanh / 累加的解析链式法则给出。
// ============================================================================
class MPCObjective : public ceres::FirstOrderFunction {
public:
    explicit MPCObjective(const MPCController* owner) : owner_(owner) {}

    bool Evaluate(const double* const parameters,
                  double* cost,
                  double* gradient) const override {
        *cost = owner_->evaluate(parameters, gradient);
        return true;
    }

    int NumParameters() const override { return owner_->numParameters(); }

private:
    const MPCController* owner_;
};

/// Options::use_gravity = false（对照模式）时把参数里的重力清零：模型里没有重力项，
/// 调用方传实测 gx/gy 也不会进入动力学。构造 / setParams / setGravity 三处
/// 写重力的地方都要过这一道（params_ 与 one_step_sim_ 必须拿到同一份）。
dm::Params withGravityOption(const dm::Params& params, bool use_gravity) {
    dm::Params out = params;
    if (!use_gravity) {
        out.gx = 0.0;
        out.gy = 0.0;
    }
    return out;
}

}  // namespace

// ----------------------------------------------------------------------------
// 构造 / 重置
// ----------------------------------------------------------------------------
MPCController::MPCController(const dm::Params& params, const Options& options)
    : params_(withGravityOption(params, options.use_gravity)),
      opt_(options),
      // 单步预测积分器：与内层 trajectory 用同样的 dt / refinement。
      // 若 dt/refinement 非法，这里会先抛出 std::invalid_argument。
      // 用 params_（已按 use_gravity 处理过）保证与求解路径同一份重力。
      one_step_sim_(params_, options.dt, options.refinement) {
    if (!(opt_.dt > 0.0)) {
        throw std::invalid_argument("MPCController: dt must be > 0");
    }
    if (opt_.N < 1) {
        throw std::invalid_argument("MPCController: N must be >= 1");
    }
    if (opt_.refinement < dm::kRefinementMin || opt_.refinement > dm::kRefinementMax) {
        throw std::invalid_argument("MPCController: refinement out of range");
    }
    if (!(opt_.max_torque_b > 0.0) || !(opt_.max_torque_s > 0.0)) {
        throw std::invalid_argument("MPCController: max_torque must be > 0");
    }
    const double weights[] = {opt_.w_psi_b, opt_.w_psi_s, opt_.w_dpsi_b, opt_.w_dpsi_s,
                              opt_.w_tau_b, opt_.w_tau_s, opt_.w_x,      opt_.w_dx};
    for (double w : weights) {
        if (!(w >= 0.0)) {
            throw std::invalid_argument("MPCController: weights must be >= 0");
        }
    }
    if (opt_.max_iter < 1) {
        throw std::invalid_argument("MPCController: max_iter must be >= 1");
    }
    if (const char* err = dm::validateTrajectoryConfig(params_, opt_.N, opt_.dt,
                                                       opt_.refinement)) {
        throw std::invalid_argument(std::string("MPCController: ") + err);
    }

    const int m = numParameters();
    x_scratch_.assign(static_cast<std::size_t>(m), 0.0);
    tau_scratch_.assign(static_cast<std::size_t>(m), 0.0);
    grad_tau_.assign(static_cast<std::size_t>(m), 0.0);
    grad_x_.assign(static_cast<std::size_t>(m), 0.0);
    if (!ws_.resize(static_cast<std::size_t>(opt_.N), opt_.refinement)) {
        throw std::runtime_error("MPCController: failed to allocate trajectory workspace");
    }
    reset();
}

void MPCController::reset() {
    x_prev_.assign(static_cast<std::size_t>(numParameters()), 0.0);
    x_applied_b_ = 0.0;
    x_applied_s_ = 0.0;
    ref_ = Reference{};
}

void MPCController::setParams(const dm::Params& params) {
    if (const char* err = dm::validateTrajectoryConfig(params, opt_.N, opt_.dt,
                                                       opt_.refinement)) {
        throw std::invalid_argument(std::string("MPCController::setParams: ") + err);
    }
    params_ = withGravityOption(params, opt_.use_gravity);
}

void MPCController::setGravity(double gx, double gy) {
    // use_gravity = false（对照模式）时模型里没有重力项：这里恒按 (0,0) 存；
    // 默认 true 时按实测值存、真正进入模型，
    // 调用方照常传实测值也不会进入动力学。
    if (!opt_.use_gravity) {
        gx = 0.0;
        gy = 0.0;
    }
    // 只改重力两项：gx/gy 不参与 validateTrajectoryConfig 的检查（它只看 dt/K/refinement/lambda），
    // 因此不必重新校验。
    params_.gx = gx;
    params_.gy = gy;
}

// ----------------------------------------------------------------------------
// 内部工具
// ----------------------------------------------------------------------------
void MPCController::buildTorque(const double* delta, double* x_out, double* tau_out) const {
    const int n = opt_.N;
    double xb = x_applied_b_;
    double xs = x_applied_s_;
    for (int k = 0; k < n; ++k) {
        xb += delta[2 * k + 0];
        xs += delta[2 * k + 1];
        x_out[2 * k + 0] = xb;
        x_out[2 * k + 1] = xs;
        // τ = τ_max·tanh(x)：软限幅，|τ| < τ_max，且处处光滑可导。
        tau_out[2 * k + 0] = opt_.max_torque_b * std::tanh(xb);
        tau_out[2 * k + 1] = opt_.max_torque_s * std::tanh(xs);
    }
}

void MPCController::validateReference(const Reference& reference) const {
    const std::size_t n = static_cast<std::size_t>(opt_.N);
    const auto check = [n](const std::vector<double>& v, const char* name) {
        if (!v.empty() && v.size() < n) {
            throw std::invalid_argument(std::string("MPCController: reference ") + name +
                                        " length must be >= N (or empty for target 0)");
        }
    };
    check(reference.psi_b, "psi_b");
    check(reference.psi_s, "psi_s");
    check(reference.dpsi_b, "dpsi_b");
    check(reference.dpsi_s, "dpsi_s");
}

dm::LossSpec MPCController::makeLossSpec() const {
    dm::LossSpec spec;
    spec.w_psi_b = opt_.w_psi_b;
    spec.w_psi_s = opt_.w_psi_s;
    spec.w_dpsi_b = opt_.w_dpsi_b;
    spec.w_dpsi_s = opt_.w_dpsi_s;
    spec.w_tau_b = opt_.w_tau_b;
    spec.w_tau_s = opt_.w_tau_s;
    // 空序列 -> nullptr，dm 内部按"目标恒为 0"处理（只惩罚幅值）。
    spec.target_psi_b = ref_.psi_b.empty() ? nullptr : ref_.psi_b.data();
    spec.target_psi_s = ref_.psi_s.empty() ? nullptr : ref_.psi_s.data();
    spec.target_dpsi_b = ref_.dpsi_b.empty() ? nullptr : ref_.dpsi_b.data();
    spec.target_dpsi_s = ref_.dpsi_s.empty() ? nullptr : ref_.dpsi_s.data();
    return spec;
}

// ----------------------------------------------------------------------------
// 目标函数：内层解析 loss/梯度 + 外层 L2 惩罚
//
//   F(Δx) = L_inner(τ(x)) + w_x·mean(x²) + w_dx·mean(Δx²)
//
// 梯度（链式）：
//   ∂F/∂x_i    = ∂L_inner/∂τ_i · τ_max·(1 - tanh²(x_i)) + 2·w_x/(2N)·x_i
//   ∂F/∂Δx_c[k] = Σ_{j≥k} ∂F/∂x_c[j] + 2·w_dx/(2N)·Δx_c[k]
// （x_c[k] 对 Δx_c[j] 的导数在 j ≤ k 时为 1，故外层是对 Δx 的反向累加）
// ----------------------------------------------------------------------------
double MPCController::evaluate(const double* delta, double* gradient) const {
    const int m = numParameters();
    const double inv_m = 1.0 / static_cast<double>(m);

    buildTorque(delta, x_scratch_.data(), tau_scratch_.data());

    // ---- 内层：正向轨迹 + 反向伴随（解析 dL/dτ）----
    // gradient == nullptr 时 dm 侧跳过反向扫描，只做一次正向。
    const dm::LossSpec spec = makeLossSpec();
    double loss = dm::simulateAndGradient(
        params_, theta_c_, dtheta_c_, ddtheta_c_, opt_.dt, opt_.refinement,
        static_cast<std::size_t>(opt_.N), tau_scratch_.data(), spec, x0_, ws_,
        (gradient != nullptr) ? grad_tau_.data() : nullptr, nullptr, false);

    // ---- 外层 L2 惩罚 ----
    double sum_x2 = 0.0;
    double sum_d2 = 0.0;
    for (int i = 0; i < m; ++i) {
        sum_x2 += x_scratch_[i] * x_scratch_[i];
        sum_d2 += delta[i] * delta[i];
    }
    loss += opt_.w_x * inv_m * sum_x2 + opt_.w_dx * inv_m * sum_d2;

    // ---- 解析梯度 ----
    if (gradient != nullptr) {
        for (int i = 0; i < m; ++i) {
            const double limit = (i % 2 == 0) ? opt_.max_torque_b : opt_.max_torque_s;
            const double th = std::tanh(x_scratch_[i]);
            grad_x_[i] = grad_tau_[i] * limit * (1.0 - th * th) +
                         opt_.w_x * 2.0 * inv_m * x_scratch_[i];
        }
        double acc_b = 0.0;
        double acc_s = 0.0;
        for (int k = opt_.N - 1; k >= 0; --k) {
            acc_b += grad_x_[2 * k + 0];
            acc_s += grad_x_[2 * k + 1];
            gradient[2 * k + 0] = acc_b + opt_.w_dx * 2.0 * inv_m * delta[2 * k + 0];
            gradient[2 * k + 1] = acc_s + opt_.w_dx * 2.0 * inv_m * delta[2 * k + 1];
        }
    }
    return loss;
}

// ----------------------------------------------------------------------------
// 求解一步
// ----------------------------------------------------------------------------
MPCController::Result MPCController::step(const dm::State& x0,
                                          double theta_c,
                                          double dtheta_c,
                                          double ddtheta_c,
                                          const Reference& reference) {
    validateReference(reference);

    x0_ = x0;
    theta_c_ = theta_c;
    dtheta_c_ = dtheta_c;
    ddtheta_c_ = ddtheta_c;
    ref_ = reference;

    const int m = numParameters();
    const int n = opt_.N;

    // ---- 初值（热启动）：上一步最优 x 序列整体左移一步，末项保持 ----
    std::vector<double> x_init(static_cast<std::size_t>(m), 0.0);
    if (static_cast<int>(x_prev_.size()) == m) {
        for (int k = 0; k + 1 < n; ++k) {
            x_init[2 * k + 0] = x_prev_[2 * (k + 1) + 0];
            x_init[2 * k + 1] = x_prev_[2 * (k + 1) + 1];
        }
        x_init[2 * (n - 1) + 0] = x_prev_[2 * (n - 1) + 0];
        x_init[2 * (n - 1) + 1] = x_prev_[2 * (n - 1) + 1];
    }
    std::vector<double> delta_init(static_cast<std::size_t>(m), 0.0);
    delta_init[0] = x_init[0] - x_applied_b_;
    delta_init[1] = x_init[1] - x_applied_s_;
    for (int k = 1; k < n; ++k) {
        delta_init[2 * k + 0] = x_init[2 * k + 0] - x_init[2 * (k - 1) + 0];
        delta_init[2 * k + 1] = x_init[2 * k + 1] - x_init[2 * (k - 1) + 1];
    }

    // ---- Ceres 梯度法求解（无参数边界）----
    std::vector<double> delta = delta_init;
    ceres::GradientProblem problem(new MPCObjective(this));
    ceres::GradientProblemSolver::Options solver_options;
    solver_options.line_search_direction_type = ceres::LBFGS;
    solver_options.line_search_type = ceres::WOLFE;
    solver_options.max_num_iterations = opt_.max_iter;
    solver_options.function_tolerance = 1e-12;
    solver_options.gradient_tolerance = 1e-12;
    solver_options.parameter_tolerance = 1e-12;
    solver_options.logging_type = ceres::SILENT;
    solver_options.minimizer_progress_to_stdout = false;
    ceres::GradientProblemSolver::Summary summary;
    ceres::Solve(solver_options, problem, delta.data(), &summary);

    last_message_ = summary.message;
    last_iterations_ = static_cast<int>(summary.iterations.size());
    initial_cost_ = summary.initial_cost;
    final_cost_ = summary.final_cost;

    // 求解不可用时回退到热启动初值：比直接给 0 力矩平滑，且始终有界。
    const bool ok = summary.IsSolutionUsable();
    const double* delta_used = ok ? delta.data() : delta_init.data();

    Result result;
    result.ok = ok;

    // ---- 重建 x / τ ----
    // 注意：以下 loss / 预测必须用**本步求解时**的累加基准 x_applied_*，
    // 因此要放在更新 x_applied_* 之前（否则 delta 会被叠加到新的基准上）。
    buildTorque(delta_used, x_scratch_.data(), tau_scratch_.data());
    result.tau_b_seq.resize(static_cast<std::size_t>(n));
    result.tau_s_seq.resize(static_cast<std::size_t>(n));
    for (int k = 0; k < n; ++k) {
        result.tau_b_seq[k] = tau_scratch_[2 * k + 0];
        result.tau_s_seq[k] = tau_scratch_[2 * k + 1];
    }
    result.torque_b = result.tau_b_seq[0];
    result.torque_s = result.tau_s_seq[0];

    // ---- 最终总 loss（含外层惩罚）----
    result.loss = evaluate(delta_used, nullptr);

    // ---- 最优力矩下的预测世界量序列（同一次正向，供上层显示/积分补偿）----
    result.pred_psi_b_seq.assign(static_cast<std::size_t>(n), 0.0);
    result.pred_psi_s_seq.assign(static_cast<std::size_t>(n), 0.0);
    result.pred_dpsi_b_seq.assign(static_cast<std::size_t>(n), 0.0);
    result.pred_dpsi_s_seq.assign(static_cast<std::size_t>(n), 0.0);
    dm::computeTrajectoryLoss(params_, theta_c_, dtheta_c_, ddtheta_c_, opt_.dt,
                              opt_.refinement, static_cast<std::size_t>(n),
                              tau_scratch_.data(), 0.0, 0.0, makeLossSpec(), x0_,
                              result.pred_psi_b_seq.data(), result.pred_psi_s_seq.data(),
                              result.pred_dpsi_b_seq.data(),
                              result.pred_dpsi_s_seq.data());
    result.pred_psi_b = result.pred_psi_b_seq[0];
    result.pred_psi_s = result.pred_psi_s_seq[0];
    result.pred_dpsi_b = result.pred_dpsi_b_seq[0];
    result.pred_dpsi_s = result.pred_dpsi_s_seq[0];

    // 一个控制周期后的状态（与内层模型同一积分方式）。
    one_step_sim_.setState(x0_);
    result.predicted = one_step_sim_.step(result.torque_b, result.torque_s, theta_c_,
                                          dtheta_c_, ddtheta_c_);

    // ---- 保存热启动状态 ----
    // 下一步从"本次实际施加的第一步预 tanh 值"继续累加。
    if (static_cast<int>(x_prev_.size()) != m) {
        x_prev_.assign(static_cast<std::size_t>(m), 0.0);
    }
    for (int k = 0; k < n; ++k) {
        x_prev_[2 * k + 0] = x_scratch_[2 * k + 0];
        x_prev_[2 * k + 1] = x_scratch_[2 * k + 1];
    }
    x_applied_b_ = x_prev_[0];
    x_applied_s_ = x_prev_[1];

    return result;
}

}  // namespace mpc
}  // namespace tcbs
