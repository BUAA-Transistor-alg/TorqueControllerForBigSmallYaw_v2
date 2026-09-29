#pragma once

// 双连杆平面系统的动力学模型。
//
// 相对 dynamic_model/ 中最初的参考实现，本模型修正了三处缺项，使其与正向运动学
// 完全自洽（并且能量守恒）：
//   1. 质量矩阵 M11 / M12 补上连杆 s 绕关节 s 的惯量 (ms*|Ps|^2 + Is)；
//   2. 重力项 G1 补上连杆 s 的重力贡献 G2（它通过 psi_b 的旋转同样作用于 theta_b）；
//   3. 科氏项 C1 修正为 ms*(dh/dtheta_s)*dtheta_s*(2*dpsi_b + dtheta_s)。
// 详细推导与数值验证见 README.md。
//
// 运动学约定：
//   psi_b = theta_c + theta_b
//   psi_s = theta_c + theta_b + theta_s
// 其中 theta_c 为外部输入（基座偏航角），theta_b / theta_s 为两个广义坐标。
// 关节 s 位于 R(psi_b)*(Dx, Dy)，连杆 s 质心位于关节 s + R(psi_s)*(Psx, Psy)。
//
// 动力学方程标准形式（q = [theta_b, theta_s]^T）：
//   M(theta_s) * qdd + C(qd) + G(q) = Q - M_col * ddtheta_c
// 展开为：
//   M11*ddtheta_b + M12*ddtheta_s = F1
//   M12*ddtheta_b + M22*ddtheta_s = F2
//   F1 = Qb - C1 - G1 - M11*ddtheta_c
//   F2 = Qs - C2 - G2 - M12*ddtheta_c

#include "params.hpp"
#include "state.hpp"

namespace tcbs {
namespace dm {

/// 与角度有关的运动学中间量。
struct Kinematics {
    double psi_b;          ///< 连杆 b 绝对角
    double psi_s;          ///< 连杆 s 绝对角
    double sin_psi_b;
    double cos_psi_b;
    double sin_psi_s;
    double cos_psi_s;
    double sin_theta_s;
    double cos_theta_s;
    double h;              ///< 关节耦合项 ms*h 出现在 M11/M12 中
    double dh_dtheta_s;    ///< h 对 theta_s 的偏导数
};

/// 对称质量矩阵。
struct MassMatrix {
    double M11;
    double M12;
    double M22;
};

/// 计算运动学中间量。
Kinematics computeKinematics(const Params& p,
                             double theta_c,
                             double theta_b,
                             double theta_s);

/// 计算质量矩阵。注意 M 只与 theta_s 有关，与 theta_c 无关。
MassMatrix computeMassMatrix(const Params& p, const Kinematics& kin);

/// 计算科里奥利力 / 离心力项。
/// @param dpsi_b    连杆 b 的绝对角速度 dtheta_c + dtheta_b
/// @param dtheta_s  广义坐标 2 的角速度
void computeCoriolis(const Params& p,
                     const Kinematics& kin,
                     double dpsi_b,
                     double dtheta_s,
                     double& C1,
                     double& C2);

/// 计算重力项（G1 含连杆 s 的贡献，G2 为连杆 s 自身的贡献）。
void computeGravity(const Params& p,
                    const Kinematics& kin,
                    double& G1,
                    double& G2);

/// 计算广义力（驱动力矩 + 摩擦）。
void computeGeneralizedForces(const Params& p,
                              double Tb,
                              double Ts,
                              double dtheta_b,
                              double dtheta_s,
                              double& Qb,
                              double& Qs);

/// 求解 2x2 线性方程组。奇异时输出 0 并返回 false。
bool solveAccelerations(const MassMatrix& m,
                        double F1,
                        double F2,
                        double& ddtheta_b,
                        double& ddtheta_s);

/// 由当前状态与外部输入计算两个广义坐标的角加速度。
void computeAccelerations(const Params& p,
                          double Tb,
                          double Ts,
                          double theta_c,
                          double dtheta_c,
                          double ddtheta_c,
                          const State& state,
                          double& ddtheta_b,
                          double& ddtheta_s);

/// 由当前状态与外部输入计算完整状态导数（供积分器使用）。
StateDerivative computeDerivative(const Params& p,
                                  double Tb,
                                  double Ts,
                                  double theta_c,
                                  double dtheta_c,
                                  double ddtheta_c,
                                  const State& state);

}  // namespace dm
}  // namespace tcbs
