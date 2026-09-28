#include "tcbss_capi.h"

#include <exception>
#include <new>
#include <string>

#include "params.hpp"
#include "simulator.hpp"
#include "state.hpp"
#include "trajectory.hpp"

namespace {

// 线程局部的错误信息，供 tcbss_last_error() 返回。
thread_local std::string g_last_error;

void setError(const char* msg) {
    g_last_error = (msg != nullptr) ? msg : "unknown error";
}

tcbss::Simulator* asSimulator(TcbssSimulator* sim) {
    return reinterpret_cast<tcbss::Simulator*>(sim);
}

const tcbss::Simulator* asSimulator(const TcbssSimulator* sim) {
    return reinterpret_cast<const tcbss::Simulator*>(sim);
}

tcbss::Params toParams(const TcbssParams& c) {
    return tcbss::Params(c.mb, c.Ib, c.Pbx, c.Pby,
                         c.ms, c.Is, c.Psx, c.Psy,
                         c.Dx, c.Dy, c.gx, c.gy,
                         c.fbc, c.fbv, c.fsc, c.fsv, c.lambda_);
}

TcbssParams toCParams(const tcbss::Params& p) {
    TcbssParams c;
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

tcbss::State toState(const TcbssState& c) {
    return tcbss::State{c.theta_b, c.dtheta_b, c.theta_s, c.dtheta_s};
}

TcbssState toCState(const tcbss::State& s) {
    TcbssState c;
    c.theta_b = s.theta_b;
    c.dtheta_b = s.dtheta_b;
    c.theta_s = s.theta_s;
    c.dtheta_s = s.dtheta_s;
    return c;
}


/// 轨迹求解器的 C++ 侧持有点：参数 + 可复用缓冲。
struct TrajectoryHolder {
    tcbss::Params params;
    tcbss::TrajectoryWorkspace workspace;

    explicit TrajectoryHolder(const tcbss::Params& p) : params(p) {}
};

TrajectoryHolder* asTrajectory(TcbssTrajectory* t) {
    return reinterpret_cast<TrajectoryHolder*>(t);
}

const TrajectoryHolder* asTrajectory(const TcbssTrajectory* t) {
    return reinterpret_cast<const TrajectoryHolder*>(t);
}

tcbss::LossSpec makeSpec(double w_psi_b,
                         double w_psi_s,
                         double w_dpsi_b,
                         double w_dpsi_s,
                         double w_tau_b,
                         double w_tau_s,
                         const double* target_psi_b,
                         const double* target_psi_s,
                         const double* target_dpsi_b,
                         const double* target_dpsi_s) {
    tcbss::LossSpec s;
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


}  // namespace

extern "C" {

const char* tcbss_last_error(void) {
    return g_last_error.c_str();
}

TcbssSimulator* tcbss_create(const TcbssParams* params, double dt, int refinement) {
    if (params == nullptr) {
        setError("tcbss_create: params is NULL");
        return nullptr;
    }

    try {
        g_last_error.clear();
        auto* sim = new tcbss::Simulator(toParams(*params), dt, refinement);
        return reinterpret_cast<TcbssSimulator*>(sim);
    } catch (const std::exception& e) {
        setError(e.what());
        return nullptr;
    } catch (...) {
        setError("tcbss_create: unknown error");
        return nullptr;
    }
}

void tcbss_destroy(TcbssSimulator* sim) {
    delete asSimulator(sim);
}

void tcbss_set_state(TcbssSimulator* sim, TcbssState state) {
    if (sim == nullptr) {
        setError("tcbss_set_state: sim is NULL");
        return;
    }
    asSimulator(sim)->setState(toState(state));
}

void tcbss_set_generalized(TcbssSimulator* sim,
                           double theta_b,
                           double dtheta_b,
                           double theta_s,
                           double dtheta_s) {
    if (sim == nullptr) {
        setError("tcbss_set_generalized: sim is NULL");
        return;
    }
    asSimulator(sim)->setState(theta_b, dtheta_b, theta_s, dtheta_s);
}

void tcbss_set_theta_b(TcbssSimulator* sim, double theta_b, double dtheta_b) {
    if (sim == nullptr) {
        setError("tcbss_set_theta_b: sim is NULL");
        return;
    }
    asSimulator(sim)->setThetaB(theta_b, dtheta_b);
}

void tcbss_set_theta_s(TcbssSimulator* sim, double theta_s, double dtheta_s) {
    if (sim == nullptr) {
        setError("tcbss_set_theta_s: sim is NULL");
        return;
    }
    asSimulator(sim)->setThetaS(theta_s, dtheta_s);
}

void tcbss_get_state(const TcbssSimulator* sim, TcbssState* out) {
    if (sim == nullptr || out == nullptr) {
        setError("tcbss_get_state: sim or out is NULL");
        return;
    }
    *out = toCState(asSimulator(sim)->state());
}

double tcbss_get_dt(const TcbssSimulator* sim) {
    if (sim == nullptr) {
        setError("tcbss_get_dt: sim is NULL");
        return 0.0;
    }
    return asSimulator(sim)->dt();
}

int tcbss_get_refinement(const TcbssSimulator* sim) {
    if (sim == nullptr) {
        setError("tcbss_get_refinement: sim is NULL");
        return 0;
    }
    return asSimulator(sim)->refinement();
}

void tcbss_get_params(const TcbssSimulator* sim, TcbssParams* out) {
    if (sim == nullptr || out == nullptr) {
        setError("tcbss_get_params: sim or out is NULL");
        return;
    }
    *out = toCParams(asSimulator(sim)->params());
}

void tcbss_step(TcbssSimulator* sim,
                double Tb,
                double Ts,
                double theta_c,
                double dtheta_c,
                double ddtheta_c,
                TcbssState* out) {
    if (sim == nullptr) {
        setError("tcbss_step: sim is NULL");
        return;
    }

    const tcbss::State next =
        asSimulator(sim)->step(Tb, Ts, theta_c, dtheta_c, ddtheta_c);

    if (out != nullptr) {
        *out = toCState(next);
    }
}

/* ==================================================================== */
/* 有限多步轨迹 + 离散伴随梯度                                          */
TcbssTrajectory* tcbss_trajectory_create(const TcbssParams* params) {
    if (params == nullptr) {
        setError("tcbss_trajectory_create: params is NULL");
        return nullptr;
    }
    try {
        g_last_error.clear();
        auto* holder = new TrajectoryHolder(toParams(*params));
        return reinterpret_cast<TcbssTrajectory*>(holder);
    } catch (const std::exception& e) {
        setError(e.what());
        return nullptr;
    } catch (...) {
        setError("tcbss_trajectory_create: unknown error");
        return nullptr;
    }
}

void tcbss_trajectory_destroy(TcbssTrajectory* t) {
    delete asTrajectory(t);
}

double tcbss_trajectory_loss(const TcbssTrajectory* t,
                             double theta_c0,
                             double dtheta_c,
                             double ddtheta_c,
                             double dt,
                             size_t num_steps,
                             const double* tau,
                             double tau_b_fixed,
                             double tau_s_fixed,
                             const TcbssState* x0,
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
        setError("tcbss_trajectory_loss: trajectory is NULL");
        return -1.0;
    }
    if (x0 == nullptr) {
        setError("tcbss_trajectory_loss: x0 is NULL");
        return -1.0;
    }
    const auto* holder = asTrajectory(t);
    const char* err =
        tcbss::validateTrajectoryConfig(holder->params, num_steps, dt);
    if (err != nullptr) {
        setError(err);
        return -1.0;
    }

    try {
        const tcbss::LossSpec spec = makeSpec(w_psi_b, w_psi_s, w_dpsi_b, w_dpsi_s, w_tau_b,
                                              w_tau_s, target_psi_b, target_psi_s,
                                              target_dpsi_b, target_dpsi_s);
        return tcbss::computeTrajectoryLoss(holder->params, theta_c0, dtheta_c, ddtheta_c, dt,
                                            num_steps, tau, tau_b_fixed, tau_s_fixed, spec,
                                            toState(*x0), out_psi_b, out_psi_s, out_dpsi_b,
                                            out_dpsi_s);
    } catch (const std::exception& e) {
        setError(e.what());
        return -1.0;
    } catch (...) {
        setError("tcbss_trajectory_loss: unknown error");
        return -1.0;
    }
}

double tcbss_trajectory_gradient(TcbssTrajectory* t,
                                 double theta_c0,
                                 double dtheta_c,
                                 double ddtheta_c,
                                 double dt,
                                 size_t num_steps,
                                 const double* tau,
                                 const TcbssState* x0,
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
                                 TcbssState* out_final_state) {
    if (t == nullptr) {
        setError("tcbss_trajectory_gradient: trajectory is NULL");
        return -1.0;
    }
    if (x0 == nullptr || tau == nullptr || grad_tau == nullptr) {
        setError("tcbss_trajectory_gradient: x0 / tau / grad_tau must not be NULL");
        return -1.0;
    }
    auto* holder = asTrajectory(t);
    const char* err =
        tcbss::validateTrajectoryConfig(holder->params, num_steps, dt);
    if (err != nullptr) {
        setError(err);
        return -1.0;
    }

    try {
        const tcbss::LossSpec spec = makeSpec(w_psi_b, w_psi_s, w_dpsi_b, w_dpsi_s, w_tau_b,
                                              w_tau_s, target_psi_b, target_psi_s,
                                              target_dpsi_b, target_dpsi_s);
        tcbss::State final_state{};
        const double loss = tcbss::simulateAndGradient(
            holder->params, theta_c0, dtheta_c, ddtheta_c, dt, num_steps, tau, spec,
            toState(*x0), holder->workspace, grad_tau,
            (out_final_state != nullptr) ? &final_state : nullptr,
            /*with_step_jacobians=*/false);
        if (out_final_state != nullptr) *out_final_state = toCState(final_state);
        return loss;
    } catch (const std::exception& e) {
        setError(e.what());
        return -1.0;
    } catch (...) {
        setError("tcbss_trajectory_gradient: unknown error");
        return -1.0;
    }
}

}  // extern "C"
