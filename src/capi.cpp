#include "tcbss_capi.h"

#include <exception>
#include <new>
#include <string>

#include "params.hpp"
#include "simulator.hpp"

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

}  // extern "C"
