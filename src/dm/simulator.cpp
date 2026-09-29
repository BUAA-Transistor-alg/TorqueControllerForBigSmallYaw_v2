#include "simulator.hpp"

#include <stdexcept>

#include "dynamics.hpp"
#include "rk4.hpp"

namespace tcbs {
namespace dm {

Simulator::Simulator(const Params& params, double dt, int refinement)
    : params_(params),
      dt_(dt),
      refinement_(refinement),
      state_{0.0, 0.0, 0.0, 0.0} {
    if (!(dt_ > 0.0)) {
        throw std::invalid_argument("Simulator: dt must be > 0");
    }
    if (refinement_ < 1) {
        throw std::invalid_argument("Simulator: refinement must be >= 1");
    }
}

void Simulator::setState(const State& state) {
    state_ = state;
}

void Simulator::setState(double theta_b, double dtheta_b, double theta_s, double dtheta_s) {
    state_.theta_b = theta_b;
    state_.dtheta_b = dtheta_b;
    state_.theta_s = theta_s;
    state_.dtheta_s = dtheta_s;
}

void Simulator::setThetaB(double theta_b, double dtheta_b) {
    state_.theta_b = theta_b;
    state_.dtheta_b = dtheta_b;
}

void Simulator::setThetaS(double theta_s, double dtheta_s) {
    state_.theta_s = theta_s;
    state_.dtheta_s = dtheta_s;
}

State Simulator::step(double Tb,
                      double Ts,
                      double theta_c,
                      double dtheta_c,
                      double ddtheta_c) {
    const double h = dt_ / static_cast<double>(refinement_);

    // 基座按等角加速度外推：tau 为本步内部的相对时间。
    const auto deriv = [&](double tau, const State& y) -> StateDerivative {
        const double tc = theta_c + dtheta_c * tau + 0.5 * ddtheta_c * tau * tau;
        const double dtc = dtheta_c + ddtheta_c * tau;
        return computeDerivative(params_, Tb, Ts, tc, dtc, ddtheta_c, y);
    };

    for (int i = 0; i < refinement_; ++i) {
        state_ = rk4Step(deriv, state_, static_cast<double>(i) * h, h);
    }

    return state_;
}

}  // namespace dm
}  // namespace tcbs
