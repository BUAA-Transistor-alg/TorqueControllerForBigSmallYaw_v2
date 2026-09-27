#pragma once

// 使用上述动力学模型 + RK4 积分的定步长仿真器。
//
// 约定：
//   * 构造时给出全部物理参数、单步 dt 及细化倍数（细化倍数 = 每个 dt 内的
//     RK4 子步数）；没有任何默认参数。
//   * 两个广义坐标的位置与速度可通过 setState / setThetaB / setThetaS 直接设置。
//   * step() 输入两个驱动力矩与 theta_c 的位置、速度、加速度，返回按参数与
//     这些输入量演化 dt 之后的两个广义坐标及其速度。

#include "params.hpp"
#include "state.hpp"

namespace tcbss {

class Simulator {
public:
    /// @param params     全部物理 / 重力 / 摩擦参数
    /// @param dt         单步仿真时间 [s]，必须 > 0
    /// @param refinement 细化倍数（每个 dt 内的 RK4 子步数），必须 >= 1
    /// @throw std::invalid_argument dt <= 0 或 refinement < 1
    Simulator(const Params& params, double dt, int refinement);

    Simulator(const Simulator&) = default;
    Simulator& operator=(const Simulator&) = default;

    // ------------------------------------------------------------------
    // 状态设置：直接设置当前两个广义坐标的位置与速度
    // ------------------------------------------------------------------
    void setState(const State& state);
    void setState(double theta_b, double dtheta_b, double theta_s, double dtheta_s);
    void setThetaB(double theta_b, double dtheta_b);
    void setThetaS(double theta_s, double dtheta_s);

    // ------------------------------------------------------------------
    // 只读访问
    // ------------------------------------------------------------------
    const State& state() const { return state_; }
    const Params& params() const { return params_; }
    double dt() const { return dt_; }
    int refinement() const { return refinement_; }

    /// 每个细化子步的时间长度 dt / refinement。
    double substepDt() const { return dt_ / static_cast<double>(refinement_); }

    // ------------------------------------------------------------------
    // 仿真一步
    //
    // @param Tb        关节 b 驱动力矩
    // @param Ts        关节 s 驱动力矩
    // @param theta_c   本步起始时刻的基座角度
    // @param dtheta_c  本步起始时刻的基座角速度
    // @param ddtheta_c 本步起始时刻的基座角加速度
    // @return 演化 dt 后的两个广义坐标及其速度
    //
    // 在 RK4 子步内，基座按等角加速度外推：
    //   theta_c(tau) = theta_c + dtheta_c*tau + 0.5*ddtheta_c*tau^2
    //   dtheta_c(tau) = dtheta_c + ddtheta_c*tau
    //   ddtheta_c(tau) = ddtheta_c
    // 其中 tau 为本步内部的相对时间，取值范围 [0, dt]。
    // ------------------------------------------------------------------
    State step(double Tb,
               double Ts,
               double theta_c,
               double dtheta_c,
               double ddtheta_c);

private:
    Params params_;
    double dt_;
    int refinement_;
    State state_;
};

}  // namespace tcbss
