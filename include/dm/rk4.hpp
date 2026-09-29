#pragma once

// 经典四阶龙格-库塔（RK4）单步积分器。
//
// 该积分器与具体模型解耦：只要 Deriv 可调用对象满足
//     StateDerivative deriv(double t, const State& y)
// 即可复用。

#include "state.hpp"

namespace tcbs {
namespace dm {

/// 在区间 [t, t + h] 上对 y 做一次 RK4 推进，返回区间末端状态。
template <typename Deriv>
State rk4Step(const Deriv& deriv, const State& y, double t, double h) {
    const StateDerivative k1 = deriv(t, y);
    const StateDerivative k2 = deriv(t + 0.5 * h, advance(y, k1, 0.5 * h));
    const StateDerivative k3 = deriv(t + 0.5 * h, advance(y, k2, 0.5 * h));
    const StateDerivative k4 = deriv(t + h, advance(y, k3, h));

    const double w = h / 6.0;

    State out;
    out.theta_b = y.theta_b +
                  w * (k1.dtheta_b + 2.0 * k2.dtheta_b + 2.0 * k3.dtheta_b + k4.dtheta_b);
    out.dtheta_b = y.dtheta_b +
                   w * (k1.ddtheta_b + 2.0 * k2.ddtheta_b + 2.0 * k3.ddtheta_b + k4.ddtheta_b);
    out.theta_s = y.theta_s +
                  w * (k1.dtheta_s + 2.0 * k2.dtheta_s + 2.0 * k3.dtheta_s + k4.dtheta_s);
    out.dtheta_s = y.dtheta_s +
                   w * (k1.ddtheta_s + 2.0 * k2.ddtheta_s + 2.0 * k3.ddtheta_s + k4.ddtheta_s);
    return out;
}

}  // namespace dm
}  // namespace tcbs
