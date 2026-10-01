#include "capi_dm.h"

#include <exception>
#include <new>
#include <string>

#include "params.hpp"
#include "simulator.hpp"
#include "param_gradient.hpp"
#include "param_gradient_batch.hpp"
#include "state.hpp"
#include "trajectory.hpp"

namespace tcbs {
namespace dm {

namespace {

// 线程局部的错误信息，供 tcbs_last_error() 返回。
thread_local std::string g_last_error;

void setError(const char* msg) {
    g_last_error = (msg != nullptr) ? msg : "unknown error";
}

tcbs::dm::Simulator* asSimulator(TcbsSimulator* sim) {
    return reinterpret_cast<tcbs::dm::Simulator*>(sim);
}

const tcbs::dm::Simulator* asSimulator(const TcbsSimulator* sim) {
    return reinterpret_cast<const tcbs::dm::Simulator*>(sim);
}

tcbs::dm::Params toParams(const TcbsParams& c) {
    return tcbs::dm::Params(c.mb, c.Ib, c.Pbx, c.Pby,
                         c.ms, c.Is, c.Psx, c.Psy,
                         c.Dx, c.Dy, c.gx, c.gy,
                         c.fbc, c.fbv, c.fsc, c.fsv, c.lambda_);
}

TcbsParams toCParams(const tcbs::dm::Params& p) {
    TcbsParams c;
    c.mb = p.mb;
    c.Ib = p.Ib;
    c.Pbx = p.Pbx;
    c.Pby = p.Pby;
    c.ms = p.ms;
    c.Is = p.Is;
    c.Psx = p.Psx;
    c.Psy = p.Psy;
    c.Dx = p.Dx;
    c.Dy = p.Dy;
    c.gx = p.gx;
    c.gy = p.gy;
    c.fbc = p.fbc;
    c.fbv = p.fbv;
    c.fsc = p.fsc;
    c.fsv = p.fsv;
    c.lambda_ = p.lambda;
    return c;
}

tcbs::dm::State toState(const TcbsState& c) {
    return tcbs::dm::State{c.theta_b, c.dtheta_b, c.theta_s, c.dtheta_s};
}

TcbsState toCState(const tcbs::dm::State& s) {
    TcbsState c;
    c.theta_b = s.theta_b;
    c.dtheta_b = s.dtheta_b;
    c.theta_s = s.theta_s;
    c.dtheta_s = s.dtheta_s;
    return c;
}


/// 轨迹求解器的 C++ 侧持有点：参数 + 可复用缓冲。
struct TrajectoryHolder {
    tcbs::dm::Params params;
    int refinement;
    tcbs::dm::TrajectoryWorkspace workspace;

    TrajectoryHolder(const tcbs::dm::Params& p, int r) : params(p), refinement(r) {}
};

TrajectoryHolder* asTrajectory(TcbsTrajectory* t) {
    return reinterpret_cast<TrajectoryHolder*>(t);
}

const TrajectoryHolder* asTrajectory(const TcbsTrajectory* t) {
    return reinterpret_cast<const TrajectoryHolder*>(t);
}

tcbs::dm::LossSpec makeSpec(double w_psi_b,
                         double w_psi_s,
                         double w_dpsi_b,
                         double w_dpsi_s,
                         double w_tau_b,
                         double w_tau_s,
                         const double* target_psi_b,
                         const double* target_psi_s,
                         const double* target_dpsi_b,
                         const double* target_dpsi_s) {
    tcbs::dm::LossSpec s;
    s.w_psi_b = w_psi_b;
    s.w_psi_s = w_psi_s;
    s.w_dpsi_b = w_dpsi_b;
    s.w_dpsi_s = w_dpsi_s;
    s.w_tau_b = w_tau_b;
    s.w_tau_s = w_tau_s;
    s.target_psi_b = target_psi_b;
    s.target_psi_s = target_psi_s;
    s.target_dpsi_b = target_dpsi_b;
    s.target_dpsi_s = target_dpsi_s;
    return s;
}



/// 参数梯度求解器的 C++ 侧持有点。
struct ParamGradientHolder {
    tcbs::dm::Params params;
    int refinement;
    tcbs::dm::ParamGradientWorkspace workspace;

    ParamGradientHolder(const tcbs::dm::Params& p, int r) : params(p), refinement(r) {}
};

ParamGradientHolder* asParamGradient(TcbsTrajectory* t) {
    return reinterpret_cast<ParamGradientHolder*>(t);
}

const ParamGradientHolder* asParamGradient(const TcbsTrajectory* t) {
    return reinterpret_cast<const ParamGradientHolder*>(t);
}

/// 批量参数梯度句柄的 C++ 侧持有点。
struct ParamGradientBatchHolder {
    tcbs::dm::ParamGradientBatchWorkspace ws;
};

ParamGradientBatchHolder* asBatch(TcbsParamGradientBatch* h) {
    return reinterpret_cast<ParamGradientBatchHolder*>(h);
}

const ParamGradientBatchHolder* asBatch(const TcbsParamGradientBatch* h) {
    return reinterpret_cast<const ParamGradientBatchHolder*>(h);
}

tcbs::dm::ParamLossSpec makeParamSpec(double w_psi_b,
                                   double w_psi_s,
                                   double w_dpsi_b,
                                   double w_dpsi_s,
                                   const double* target_psi_b,
                                   const double* target_psi_s,
                                   const double* target_dpsi_b,
                                   const double* target_dpsi_s) {
    tcbs::dm::ParamLossSpec s;
    s.w_psi_b = w_psi_b;
    s.w_psi_s = w_psi_s;
    s.w_dpsi_b = w_dpsi_b;
    s.w_dpsi_s = w_dpsi_s;
    s.target_psi_b = target_psi_b;
    s.target_psi_s = target_psi_s;
    s.target_dpsi_b = target_dpsi_b;
    s.target_dpsi_s = target_dpsi_s;
    return s;
}


}  // namespace

extern "C" {

const char* tcbs_last_error(void) {
    return g_last_error.c_str();
}

TcbsSimulator* tcbs_create(const TcbsParams* params, double dt, int refinement) {
    if (params == nullptr) {
        setError("tcbs_create: params is NULL");
        return nullptr;
    }

    try {
        g_last_error.clear();
        auto* sim = new tcbs::dm::Simulator(toParams(*params), dt, refinement);
        return reinterpret_cast<TcbsSimulator*>(sim);
    } catch (const std::exception& e) {
        setError(e.what());
        return nullptr;
    } catch (...) {
        setError("tcbs_create: unknown error");
        return nullptr;
    }
}

void tcbs_destroy(TcbsSimulator* sim) {
    delete asSimulator(sim);
}

void tcbs_set_state(TcbsSimulator* sim, TcbsState state) {
    if (sim == nullptr) {
        setError("tcbs_set_state: sim is NULL");
        return;
    }
    asSimulator(sim)->setState(toState(state));
}

void tcbs_set_generalized(TcbsSimulator* sim,
                           double theta_b,
                           double dtheta_b,
                           double theta_s,
                           double dtheta_s) {
    if (sim == nullptr) {
        setError("tcbs_set_generalized: sim is NULL");
        return;
    }
    asSimulator(sim)->setState(theta_b, dtheta_b, theta_s, dtheta_s);
}

void tcbs_set_theta_b(TcbsSimulator* sim, double theta_b, double dtheta_b) {
    if (sim == nullptr) {
        setError("tcbs_set_theta_b: sim is NULL");
        return;
    }
    asSimulator(sim)->setThetaB(theta_b, dtheta_b);
}

void tcbs_set_theta_s(TcbsSimulator* sim, double theta_s, double dtheta_s) {
    if (sim == nullptr) {
        setError("tcbs_set_theta_s: sim is NULL");
        return;
    }
    asSimulator(sim)->setThetaS(theta_s, dtheta_s);
}

void tcbs_get_state(const TcbsSimulator* sim, TcbsState* out) {
    if (sim == nullptr || out == nullptr) {
        setError("tcbs_get_state: sim or out is NULL");
        return;
    }
    *out = toCState(asSimulator(sim)->state());
}

double tcbs_get_dt(const TcbsSimulator* sim) {
    if (sim == nullptr) {
        setError("tcbs_get_dt: sim is NULL");
        return 0.0;
    }
    return asSimulator(sim)->dt();
}

int tcbs_get_refinement(const TcbsSimulator* sim) {
    if (sim == nullptr) {
        setError("tcbs_get_refinement: sim is NULL");
        return 0;
    }
    return asSimulator(sim)->refinement();
}

void tcbs_get_params(const TcbsSimulator* sim, TcbsParams* out) {
    if (sim == nullptr || out == nullptr) {
        setError("tcbs_get_params: sim or out is NULL");
        return;
    }
    *out = toCParams(asSimulator(sim)->params());
}

void tcbs_step(TcbsSimulator* sim,
                double Tb,
                double Ts,
                double theta_c,
                double dtheta_c,
                double ddtheta_c,
                TcbsState* out) {
    if (sim == nullptr) {
        setError("tcbs_step: sim is NULL");
        return;
    }

    const tcbs::dm::State next =
        asSimulator(sim)->step(Tb, Ts, theta_c, dtheta_c, ddtheta_c);

    if (out != nullptr) {
        *out = toCState(next);
    }
}

/* ==================================================================== */
/* 有限多步轨迹 + 离散伴随梯度                                          */
TcbsTrajectory* tcbs_trajectory_create(const TcbsParams* params, int refinement) {
    if (params == nullptr) {
        setError("tcbs_trajectory_create: params is NULL");
        return nullptr;
    }
    try {
        g_last_error.clear();
        auto* holder = new TrajectoryHolder(toParams(*params), refinement);
        return reinterpret_cast<TcbsTrajectory*>(holder);
    } catch (const std::exception& e) {
        setError(e.what());
        return nullptr;
    } catch (...) {
        setError("tcbs_trajectory_create: unknown error");
        return nullptr;
    }
}

void tcbs_trajectory_destroy(TcbsTrajectory* t) {
    delete asTrajectory(t);
}

double tcbs_trajectory_loss(const TcbsTrajectory* t,
                             double theta_c0,
                             double dtheta_c,
                             double ddtheta_c,
                             double dt,
                             int refinement,
                             size_t num_steps,
                             const double* tau,
                             double tau_b_fixed,
                             double tau_s_fixed,
                             const TcbsState* x0,
                             double w_psi_b,
                             double w_psi_s,
                             double w_dpsi_b,
                             double w_dpsi_s,
                             double w_tau_b,
                             double w_tau_s,
                             const double* target_psi_b,
                             const double* target_psi_s,
                             const double* target_dpsi_b,
                             const double* target_dpsi_s,
                             double* out_psi_b,
                             double* out_psi_s,
                             double* out_dpsi_b,
                             double* out_dpsi_s) {
    if (t == nullptr) {
        setError("tcbs_trajectory_loss: trajectory is NULL");
        return -1.0;
    }
    if (x0 == nullptr) {
        setError("tcbs_trajectory_loss: x0 is NULL");
        return -1.0;
    }
    const auto* holder = asTrajectory(t);
    const char* err =
        tcbs::dm::validateTrajectoryConfig(holder->params, num_steps, dt, refinement);
    if (err != nullptr) {
        setError(err);
        return -1.0;
    }

    try {
        const tcbs::dm::LossSpec spec = makeSpec(w_psi_b, w_psi_s, w_dpsi_b, w_dpsi_s, w_tau_b,
                                              w_tau_s, target_psi_b, target_psi_s,
                                              target_dpsi_b, target_dpsi_s);
        return tcbs::dm::computeTrajectoryLoss(holder->params, theta_c0, dtheta_c, ddtheta_c, dt,
                                            refinement, num_steps, tau, tau_b_fixed, tau_s_fixed, spec,
                                            toState(*x0), out_psi_b, out_psi_s, out_dpsi_b,
                                            out_dpsi_s);
    } catch (const std::exception& e) {
        setError(e.what());
        return -1.0;
    } catch (...) {
        setError("tcbs_trajectory_loss: unknown error");
        return -1.0;
    }
}

double tcbs_trajectory_gradient(TcbsTrajectory* t,
                                 double theta_c0,
                                 double dtheta_c,
                                 double ddtheta_c,
                                 double dt,
                                 int refinement,
                                 size_t num_steps,
                                 const double* tau,
                                 const TcbsState* x0,
                                 double w_psi_b,
                                 double w_psi_s,
                                 double w_dpsi_b,
                                 double w_dpsi_s,
                                 double w_tau_b,
                                 double w_tau_s,
                                 const double* target_psi_b,
                                 const double* target_psi_s,
                                 const double* target_dpsi_b,
                                 const double* target_dpsi_s,
                                 double* grad_tau,
                                 TcbsState* out_final_state) {
    if (t == nullptr) {
        setError("tcbs_trajectory_gradient: trajectory is NULL");
        return -1.0;
    }
    if (x0 == nullptr || tau == nullptr || grad_tau == nullptr) {
        setError("tcbs_trajectory_gradient: x0 / tau / grad_tau must not be NULL");
        return -1.0;
    }
    auto* holder = asTrajectory(t);
    const char* err =
        tcbs::dm::validateTrajectoryConfig(holder->params, num_steps, dt, refinement);
    if (err != nullptr) {
        setError(err);
        return -1.0;
    }

    try {
        const tcbs::dm::LossSpec spec = makeSpec(w_psi_b, w_psi_s, w_dpsi_b, w_dpsi_s, w_tau_b,
                                              w_tau_s, target_psi_b, target_psi_s,
                                              target_dpsi_b, target_dpsi_s);
        tcbs::dm::State final_state{};
        const double loss = tcbs::dm::simulateAndGradient(
            holder->params, theta_c0, dtheta_c, ddtheta_c, dt, refinement, num_steps, tau, spec,
            toState(*x0), holder->workspace, grad_tau,
            (out_final_state != nullptr) ? &final_state : nullptr,
            /*with_step_jacobians=*/false);
        if (out_final_state != nullptr) *out_final_state = toCState(final_state);
        return loss;
    } catch (const std::exception& e) {
        setError(e.what());
        return -1.0;
    } catch (...) {
        setError("tcbs_trajectory_gradient: unknown error");
        return -1.0;
    }
}


/* ==================================================================== */
/* 系统参数辨识模式                                                     */
int tcbs_param_gradient_count(void) {
    return tcbs::dm::kParamGradientCount;
}

const char* tcbs_param_gradient_name(int index) {
    if (index < 0 || index >= tcbs::dm::kParamGradientCount) return "";
    return tcbs::dm::paramGradientNames()[index];
}

TcbsTrajectory* tcbs_param_gradient_create(const TcbsParams* params, int refinement) {
    if (params == nullptr) {
        setError("tcbs_param_gradient_create: params is NULL");
        return nullptr;
    }
    try {
        g_last_error.clear();
        return reinterpret_cast<TcbsTrajectory*>(
            new ParamGradientHolder(toParams(*params), refinement));
    } catch (const std::exception& e) {
        setError(e.what());
        return nullptr;
    } catch (...) {
        setError("tcbs_param_gradient_create: unknown error");
        return nullptr;
    }
}

void tcbs_param_gradient_destroy(TcbsTrajectory* t) {
    delete asParamGradient(t);
}

int tcbs_param_gradient_set_params(TcbsTrajectory* t, const TcbsParams* params) {
    if (t == nullptr) {
        setError("tcbs_param_gradient_set_params: trajectory is NULL");
        return 0;
    }
    if (params == nullptr) {
        setError("tcbs_param_gradient_set_params: params is NULL");
        return 0;
    }
    try {
        g_last_error.clear();
        asParamGradient(t)->params = toParams(*params);
        return 1;
    } catch (const std::exception& e) {
        setError(e.what());
        return 0;
    } catch (...) {
        setError("tcbs_param_gradient_set_params: unknown error");
        return 0;
    }
}

void tcbs_param_gradient_get_params(const TcbsTrajectory* t, TcbsParams* out) {
    if (t == nullptr || out == nullptr) {
        setError("tcbs_param_gradient_get_params: trajectory or out is NULL");
        return;
    }
    *out = toCParams(asParamGradient(t)->params);
}

double tcbs_param_gradient_loss(const TcbsTrajectory* t,
                                 double theta_c0,
                                 double dtheta_c,
                                 double ddtheta_c,
                                 double dt,
                                 int refinement,
                                 size_t num_steps,
                                 const double* tau,
                                 const TcbsState* x0,
                                 double w_psi_b,
                                 double w_psi_s,
                                 double w_dpsi_b,
                                 double w_dpsi_s,
                                 const double* target_psi_b,
                                 const double* target_psi_s,
                                 const double* target_dpsi_b,
                                 const double* target_dpsi_s,
                                 double* out_psi_b,
                                 double* out_psi_s,
                                 double* out_dpsi_b,
                                 double* out_dpsi_s) {
    if (t == nullptr) {
        setError("tcbs_param_gradient_loss: trajectory is NULL");
        return -1.0;
    }
    if (x0 == nullptr || tau == nullptr) {
        setError("tcbs_param_gradient_loss: x0 / tau must not be NULL");
        return -1.0;
    }
    const auto* holder = asParamGradient(t);
    if (const char* err = tcbs::dm::validateParamGradientConfig(holder->params, num_steps, dt, refinement)) {
        setError(err);
        return -1.0;
    }
    try {
        const tcbs::dm::ParamLossSpec spec = makeParamSpec(
            w_psi_b, w_psi_s, w_dpsi_b, w_dpsi_s, target_psi_b, target_psi_s,
            target_dpsi_b, target_dpsi_s);
        return tcbs::dm::computeParamGradientLoss(holder->params, theta_c0, dtheta_c, ddtheta_c,
                                              dt, refinement, num_steps, tau, spec, toState(*x0),
                                              out_psi_b, out_psi_s, out_dpsi_b, out_dpsi_s);
    } catch (const std::exception& e) {
        setError(e.what());
        return -1.0;
    } catch (...) {
        setError("tcbs_param_gradient_loss: unknown error");
        return -1.0;
    }
}

double tcbs_param_gradient_run(TcbsTrajectory* t,
                                double theta_c0,
                                double dtheta_c,
                                double ddtheta_c,
                                double dt,
                                int refinement,
                                size_t num_steps,
                                const double* tau,
                                const TcbsState* x0,
                                double w_psi_b,
                                double w_psi_s,
                                double w_dpsi_b,
                                double w_dpsi_s,
                                const double* target_psi_b,
                                const double* target_psi_s,
                                const double* target_dpsi_b,
                                const double* target_dpsi_s,
                                double* grad_p,
                                TcbsState* out_final_state) {
    if (t == nullptr) {
        setError("tcbs_param_gradient_run: trajectory is NULL");
        return -1.0;
    }
    if (x0 == nullptr || tau == nullptr || grad_p == nullptr) {
        setError("tcbs_param_gradient_run: x0 / tau / grad_p must not be NULL");
        return -1.0;
    }
    auto* holder = asParamGradient(t);
    if (const char* err = tcbs::dm::validateParamGradientConfig(holder->params, num_steps, dt, refinement)) {
        setError(err);
        return -1.0;
    }
    try {
        const tcbs::dm::ParamLossSpec spec = makeParamSpec(
            w_psi_b, w_psi_s, w_dpsi_b, w_dpsi_s, target_psi_b, target_psi_s,
            target_dpsi_b, target_dpsi_s);
        tcbs::dm::State final_state{};
        const double loss = tcbs::dm::computeParamGradient(
            holder->params, theta_c0, dtheta_c, ddtheta_c, dt, refinement, num_steps, tau, spec,
            toState(*x0), holder->workspace, grad_p,
            (out_final_state != nullptr) ? &final_state : nullptr);
        if (out_final_state != nullptr) *out_final_state = toCState(final_state);
        return loss;
    } catch (const std::exception& e) {
        setError(e.what());
        return -1.0;
    } catch (...) {
        setError("tcbs_param_gradient_run: unknown error");
        return -1.0;
    }
}

// ---------------------------------------------------------------------------
// 批量（SoA）参数梯度
// ---------------------------------------------------------------------------
TcbsParamGradientBatch* tcbs_param_gradient_batch_create(int refinement, int max_batch,
                                                         size_t max_num_steps,
                                                         int num_threads, int lanes) {
    if (refinement < tcbs::dm::kRefinementMin ||
        refinement > tcbs::dm::kRefinementMax) {
        setError("tcbs_param_gradient_batch_create: refinement out of range");
        return nullptr;
    }
    if (max_batch < 1 || max_num_steps < 1) {
        setError("tcbs_param_gradient_batch_create: max_batch / max_num_steps must be >= 1");
        return nullptr;
    }
    try {
        g_last_error.clear();
        auto* holder = new ParamGradientBatchHolder();
        const int ln = (lanes <= 0) ? 16 : lanes;
        if (!holder->ws.resize(refinement, max_batch, max_num_steps, num_threads, ln)) {
            delete holder;
            setError("tcbs_param_gradient_batch_create: workspace allocation failed");
            return nullptr;
        }
        return reinterpret_cast<TcbsParamGradientBatch*>(holder);
    } catch (const std::exception& e) {
        setError(e.what());
        return nullptr;
    } catch (...) {
        setError("tcbs_param_gradient_batch_create: unknown error");
        return nullptr;
    }
}

void tcbs_param_gradient_batch_destroy(TcbsParamGradientBatch* h) { delete asBatch(h); }

int tcbs_param_gradient_batch_threads(const TcbsParamGradientBatch* h) {
    if (h == nullptr) {
        setError("tcbs_param_gradient_batch_threads: handle is NULL");
        return 0;
    }
    return asBatch(h)->ws.num_threads;
}

int tcbs_param_gradient_batch_lanes(const TcbsParamGradientBatch* h) {
    if (h == nullptr) {
        setError("tcbs_param_gradient_batch_lanes: handle is NULL");
        return 0;
    }
    return asBatch(h)->ws.lanes;
}

int tcbs_param_gradient_batch_run(TcbsParamGradientBatch* h,
                                  int batch,
                                  size_t num_steps,
                                  double dt,
                                  const double* params,
                                  const double* x0,
                                  const double* base,
                                  const double* tau,
                                  const double* target_psi_b,
                                  const double* target_psi_s,
                                  const double* target_dpsi_b,
                                  const double* target_dpsi_s,
                                  const double* weights,
                                  double* out_loss,
                                  double* out_grad) {
    if (h == nullptr) {
        setError("tcbs_param_gradient_batch_run: handle is NULL");
        return 0;
    }
    try {
        g_last_error.clear();
        tcbs::dm::computeParamGradientBatch(params, x0, base, tau, target_psi_b,
                                            target_psi_s, target_dpsi_b, target_dpsi_s,
                                            weights, batch, num_steps, dt, asBatch(h)->ws,
                                            out_loss, out_grad);
        return 1;
    } catch (const std::exception& e) {
        setError(e.what());
        return 0;
    } catch (...) {
        setError("tcbs_param_gradient_batch_run: unknown error");
        return 0;
    }
}

}  // extern "C"

}  // namespace dm
}  // namespace tcbs
