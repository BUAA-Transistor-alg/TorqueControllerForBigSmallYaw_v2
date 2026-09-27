#pragma once

// 广义坐标状态及状态导数。
//
// 两个广义坐标：
//   theta_b : 广义坐标 1（对应连杆 b 的相对角）
//   theta_s : 广义坐标 2（对应连杆 s 的相对角）

namespace tcbss {

/// 两个广义坐标的当前位置与速度。
struct State {
    double theta_b;   ///< 广义坐标 1 角度 [rad]
    double dtheta_b;  ///< 广义坐标 1 角速度 [rad/s]
    double theta_s;   ///< 广义坐标 2 角度 [rad]
    double dtheta_s;  ///< 广义坐标 2 角速度 [rad/s]
};

/// 状态对时间的导数，字段与 State 一一对应。
struct StateDerivative {
    double dtheta_b;   ///< d(theta_b)/dt
    double ddtheta_b;  ///< d(dtheta_b)/dt
    double dtheta_s;   ///< d(theta_s)/dt
    double ddtheta_s;  ///< d(dtheta_s)/dt
};

/// 返回 y + s * k，用于 RK4 的中间状态推进。
inline State advance(const State& y, const StateDerivative& k, double s) {
    State out;
    out.theta_b  = y.theta_b  + s * k.dtheta_b;
    out.dtheta_b = y.dtheta_b + s * k.ddtheta_b;
    out.theta_s  = y.theta_s  + s * k.dtheta_s;
    out.dtheta_s = y.dtheta_s + s * k.ddtheta_s;
    return out;
}

}  // namespace tcbss
