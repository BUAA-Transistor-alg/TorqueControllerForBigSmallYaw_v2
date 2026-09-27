#include "dynamics.hpp"

#include <cmath>

namespace tcbss {
namespace {

/// 行列式奇异判据（理论上正定系统不会触发，工程上保留安全检查）。
constexpr double kDetEpsilon = 1e-12;

/// 连杆 b 对关节 b 的惯量：mb*|Pb|^2 + Ib。
inline double linkBInertia(const Params& p) {
    return p.mb * (p.Pbx * p.Pbx + p.Pby * p.Pby) + p.Ib;
}

/// 连杆 s 对关节 s 的惯量：ms*|Ps|^2 + Is。
inline double linkSInertia(const Params& p) {
    return p.ms * (p.Psx * p.Psx + p.Psy * p.Psy) + p.Is;
}

/// 关节 s 相对关节 b 的偏移带来的平动惯量：ms*|D|^2。
inline double jointOffsetInertia(const Params& p) {
    return p.ms * (p.Dx * p.Dx + p.Dy * p.Dy);
}

}  // namespace

Kinematics computeKinematics(const Params& p,
                             double theta_c,
                             double theta_b,
                             double theta_s) {
    Kinematics kin;

    kin.psi_b = theta_c + theta_b;
    kin.psi_s = theta_c + theta_b + theta_s;

    kin.sin_psi_b = std::sin(kin.psi_b);
    kin.cos_psi_b = std::cos(kin.psi_b);
    kin.sin_psi_s = std::sin(kin.psi_s);
    kin.cos_psi_s = std::cos(kin.psi_s);

    kin.sin_theta_s = std::sin(theta_s);
    kin.cos_theta_s = std::cos(theta_s);

    // h 为两个连杆之间的关节耦合项（只与 theta_s 有关）。
    // h = Dx*Psx*cos(ts) + Dy*Psy*cos(ts) + Dy*Psx*sin(ts) - Dx*Psy*sin(ts)
    kin.h = p.Dx * p.Psx * kin.cos_theta_s + p.Dy * p.Psy * kin.cos_theta_s +
            p.Dy * p.Psx * kin.sin_theta_s - p.Dx * p.Psy * kin.sin_theta_s;

    // dh/dtheta_s
    kin.dh_dtheta_s = -p.Dx * p.Psx * kin.sin_theta_s - p.Dy * p.Psy * kin.sin_theta_s +
                      p.Dy * p.Psx * kin.cos_theta_s - p.Dx * p.Psy * kin.cos_theta_s;

    return kin;
}

MassMatrix computeMassMatrix(const Params& p, const Kinematics& kin) {
    // 由正向运动学导出（q = [theta_b, theta_s]）：
    //   M11 = mb|Pb|^2 + Ib + ms|D|^2 + (ms|Ps|^2 + Is) + 2*ms*h
    //   M12 =                (ms|Ps|^2 + Is) +     ms*h
    //   M22 =                (ms|Ps|^2 + Is)
    const double Ib_ = linkBInertia(p);
    const double Is_ = linkSInertia(p);
    const double Id_ = jointOffsetInertia(p);

    MassMatrix m;
    m.M11 = Ib_ + Id_ + Is_ + 2.0 * p.ms * kin.h;
    m.M12 = Is_ + p.ms * kin.h;
    m.M22 = Is_;
    return m;
}

void computeCoriolis(const Params& p,
                     const Kinematics& kin,
                     double dpsi_b,
                     double dtheta_s,
                     double& C1,
                     double& C2) {
    // 由拉格朗日方程展开（M 只依赖 theta_s，故所有科氏项系数正比于 ms*dh/dtheta_s）：
    //   C1 = ms*(dh/dtheta_s) * dtheta_s * (2*dpsi_b + dtheta_s)
    //   C2 = -ms*(dh/dtheta_s) * dpsi_b^2
    const double coupling = p.ms * kin.dh_dtheta_s;

    C1 = coupling * dtheta_s * (2.0 * dpsi_b + dtheta_s);
    C2 = -coupling * dpsi_b * dpsi_b;
}

void computeGravity(const Params& p, const Kinematics& kin, double& G1, double& G2) {
    // 连杆 s 的重力项：它既通过 psi_s 进入 G2，也通过 psi_b（关节 s 随连杆 b 转动）
    // 进入 G1。
    const double gs_sin = p.ms * (p.gx * p.Psx + p.gy * p.Psy);
    const double gs_cos = p.ms * (p.gx * p.Psy - p.gy * p.Psx);

    G2 = gs_sin * kin.sin_psi_s + gs_cos * kin.cos_psi_s;

    const double gb_sin =
        p.gx * (p.mb * p.Pbx + p.ms * p.Dx) + p.gy * (p.mb * p.Pby + p.ms * p.Dy);
    const double gb_cos =
        p.gx * (p.mb * p.Pby + p.ms * p.Dy) - p.gy * (p.mb * p.Pbx + p.ms * p.Dx);

    G1 = gb_sin * kin.sin_psi_b + gb_cos * kin.cos_psi_b + G2;
}

void computeGeneralizedForces(const Params& p,
                              double Tb,
                              double Ts,
                              double dtheta_b,
                              double dtheta_s,
                              double& Qb,
                              double& Qs) {
    Qb = Tb - p.fbv * dtheta_b - p.fbc * std::tanh(p.lambda * dtheta_b);
    Qs = Ts - p.fsv * dtheta_s - p.fsc * std::tanh(p.lambda * dtheta_s);
}

bool solveAccelerations(const MassMatrix& m,
                        double F1,
                        double F2,
                        double& ddtheta_b,
                        double& ddtheta_s) {
    const double det = m.M11 * m.M22 - m.M12 * m.M12;

    if (std::abs(det) < kDetEpsilon) {
        ddtheta_b = 0.0;
        ddtheta_s = 0.0;
        return false;
    }

    ddtheta_b = (F1 * m.M22 - F2 * m.M12) / det;
    ddtheta_s = (m.M11 * F2 - m.M12 * F1) / det;
    return true;
}

void computeAccelerations(const Params& p,
                          double Tb,
                          double Ts,
                          double theta_c,
                          double dtheta_c,
                          double ddtheta_c,
                          const State& state,
                          double& ddtheta_b,
                          double& ddtheta_s) {
    const Kinematics kin = computeKinematics(p, theta_c, state.theta_b, state.theta_s);
    const MassMatrix mass = computeMassMatrix(p, kin);

    const double dpsi_b = dtheta_c + state.dtheta_b;

    double C1 = 0.0;
    double C2 = 0.0;
    computeCoriolis(p, kin, dpsi_b, state.dtheta_s, C1, C2);

    double G1 = 0.0;
    double G2 = 0.0;
    computeGravity(p, kin, G1, G2);

    double Qb = 0.0;
    double Qs = 0.0;
    computeGeneralizedForces(p, Tb, Ts, state.dtheta_b, state.dtheta_s, Qb, Qs);

    // ddtheta_c 作为已知的基座角加速度分配到方程右端：
    // 第 1 式系数为 M11，第 2 式系数为 M12（= 质量矩阵中 theta_c 与广义坐标的耦合列）。
    const double F1 = Qb - C1 - G1 - mass.M11 * ddtheta_c;
    const double F2 = Qs - C2 - G2 - mass.M12 * ddtheta_c;

    solveAccelerations(mass, F1, F2, ddtheta_b, ddtheta_s);
}

StateDerivative computeDerivative(const Params& p,
                                  double Tb,
                                  double Ts,
                                  double theta_c,
                                  double dtheta_c,
                                  double ddtheta_c,
                                  const State& state) {
    StateDerivative d;
    d.dtheta_b = state.dtheta_b;
    d.dtheta_s = state.dtheta_s;

    computeAccelerations(p, Tb, Ts, theta_c, dtheta_c, ddtheta_c, state, d.ddtheta_b,
                         d.ddtheta_s);

    return d;
}

}  // namespace tcbss
