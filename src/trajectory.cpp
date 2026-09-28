#include "trajectory.hpp"

#include <cmath>
#include <cstdlib>

// sincos 是 GNU 扩展：一次调用同时得到 sin 与 cos，比分别调用快近一倍，
// 而本模块每步要算 3 组、每步 4 个子步、每子步 4 个阶段，是最热的地方。
#if defined(__GNUC__) && !defined(__clang__)
#define TCBSS_HAVE_SINCOS 1
#endif

namespace tcbss {
namespace {

constexpr double kDetEpsilon = 1e-12;

inline void sincos3(double a, double b, double c, double& sa, double& ca, double& sb,
                    double& cb, double& sc, double& cc) {
#if TCBSS_HAVE_SINCOS
    ::sincos(a, &sa, &ca);
    ::sincos(b, &sb, &cb);
    ::sincos(c, &sc, &cc);
#else
    sa = std::sin(a);
    ca = std::cos(a);
    sb = std::sin(b);
    cb = std::cos(b);
    sc = std::sin(c);
    cc = std::cos(c);
#endif
}

/// 一次动力学求值的全部产物。
/// 关键的时间优化：前向导数与两组雅可比在同一次三角运算中一并得出，
/// 因此灵敏度传播不引入任何额外的 sin/cos/tanh。
struct Eval {
    StateDerivative dx;
    double Jx[4][4];   ///< x = [theta_b, dtheta_b, theta_s, dtheta_s]
    double Ju[4][2];   ///< u = [Tb, Ts]
    bool solvable;
};

/// 与 src/dynamics.cpp 完全一致的参数派生常量（只依赖 Params）。
struct Derived {
    double I_B;
    double I_S;
    double I_D;
    double gs_sin;
    double gs_cos;
    double gb_sin;
    double gb_cos;
};

inline Derived makeDerived(const Params& p) {
    Derived d;
    d.I_B = p.mb * (p.Pbx * p.Pbx + p.Pby * p.Pby) + p.Ib;
    d.I_S = p.ms * (p.Psx * p.Psx + p.Psy * p.Psy) + p.Is;
    d.I_D = p.ms * (p.Dx * p.Dx + p.Dy * p.Dy);
    d.gs_sin = p.ms * (p.gx * p.Psx + p.gy * p.Psy);
    d.gs_cos = p.ms * (p.gx * p.Psy - p.gy * p.Psx);
    d.gb_sin = p.gx * (p.mb * p.Pbx + p.ms * p.Dx) + p.gy * (p.mb * p.Pby + p.ms * p.Dy);
    d.gb_cos = p.gx * (p.mb * p.Pby + p.ms * p.Dy) - p.gy * (p.mb * p.Pbx + p.ms * p.Dx);
    return d;
}

/// 前向导数 + 状态雅可比 + 输入雅可比，一次算完。
inline void evaluate(const Params& p,
                     const Derived& d,
                     const State& x,
                     double Tb,
                     double Ts,
                     double tc,
                     double dtc,
                     double ddtc,
                     Eval& out) {
    const double tb = x.theta_b;
    const double dtb = x.dtheta_b;
    const double ts = x.theta_s;
    const double dts = x.dtheta_s;

    const double psi_b = tc + tb;
    const double psi_s = psi_b + ts;

    double sb, cb, ss, cs, st, ct;
    sincos3(psi_b, psi_s, ts, sb, cb, ss, cs, st, ct);

    // h 与 dh/dtheta_s；二阶导 d2h/dtheta_s^2 = -h
    const double h =
        p.Dx * p.Psx * ct + p.Dy * p.Psy * ct + p.Dy * p.Psx * st - p.Dx * p.Psy * st;
    const double dhd =
        -p.Dx * p.Psx * st - p.Dy * p.Psy * st + p.Dy * p.Psx * ct - p.Dx * p.Psy * ct;

    const double M11 = d.I_B + d.I_D + d.I_S + 2.0 * p.ms * h;
    const double M12 = d.I_S + p.ms * h;
    const double M22 = d.I_S;

    // 科氏项，dpsi_b = dtc + dtb
    const double dpb = dtc + dtb;
    const double cpl = p.ms * dhd;
    const double C1 = cpl * dts * (2.0 * dpb + dts);
    const double C2 = -cpl * dpb * dpb;

    // 重力项
    const double G2 = d.gs_sin * ss + d.gs_cos * cs;
    const double G1 = d.gb_sin * sb + d.gb_cos * cb + G2;

    // 广义力（含平滑摩擦）
    const double th_tb = std::tanh(p.lambda * dtb);
    const double th_ts = std::tanh(p.lambda * dts);
    const double Qb = Tb - p.fbv * dtb - p.fbc * th_tb;
    const double Qs = Ts - p.fsv * dts - p.fsc * th_ts;

    const double F1 = Qb - C1 - G1 - M11 * ddtc;
    const double F2 = Qs - C2 - G2 - M12 * ddtc;

    const double det = M11 * M22 - M12 * M12;
    const double invdet = 1.0 / det;
    const double Minv00 = M22 * invdet;
    const double Minv01 = -M12 * invdet;
    const double Minv11 = M11 * invdet;

    const double qdd_b = Minv00 * F1 + Minv01 * F2;
    const double qdd_s = Minv01 * F1 + Minv11 * F2;

    out.dx.dtheta_b = dtb;    // d(theta_b)/dt
    out.dx.ddtheta_b = qdd_b; // d(dtheta_b)/dt
    out.dx.dtheta_s = dts;
    out.dx.ddtheta_s = qdd_s;

    // ---------------- 状态雅可比 ----------------
    // 重力：G2 经 psi_s 同时依赖 theta_b 与 theta_s
    const double dG2 = d.gs_sin * cs - d.gs_cos * ss;
    const double dG1_b = d.gb_sin * cb - d.gb_cos * sb + dG2;

    // 对 theta_b：dM/dtheta_b = 0，无惯性修正
    const double dF1_dtb = -dG1_b;
    const double dF2_dtb = -dG2;

    // 对 theta_s：M、重力、科氏、基座耦合 (-dM*ddtc) 全变，含惯性修正 -dM/dtheta_s*qdd
    const double dM11 = 2.0 * p.ms * dhd;
    const double dM12 = p.ms * dhd;
    const double dC1_dts = -p.ms * h * dts * (2.0 * dpb + dts);
    const double dC2_dts = p.ms * h * dpb * dpb;
    const double dF1_ds = -dC1_dts - dG2 - dM11 * ddtc;
    const double dF2_ds = -dC2_dts - dG2 - dM12 * ddtc;
    const double corr1 = dM11 * qdd_b + dM12 * qdd_s;
    const double corr2 = dM12 * qdd_b;

    // 对 dtheta_b：科氏 + 摩擦
    const double dC1_ddtb = cpl * dts * 2.0;
    const double dC2_ddtb = -cpl * 2.0 * dpb;
    const double dQb_ddtb = -p.fbv - p.fbc * p.lambda * (1.0 - th_tb * th_tb);
    const double dF1_ddtb = -dC1_ddtb + dQb_ddtb;
    const double dF2_ddtb = -dC2_ddtb;

    // 对 dtheta_s：C2 与 dts 无关
    const double dC1_ddts = cpl * (2.0 * dpb + 2.0 * dts);
    const double dQs_ddts = -p.fsv - p.fsc * p.lambda * (1.0 - th_ts * th_ts);
    const double dF1_ddts = -dC1_ddts;
    const double dF2_ddts = dQs_ddts;

    out.Jx[0][0] = 0.0;
    out.Jx[0][1] = 1.0;
    out.Jx[0][2] = 0.0;
    out.Jx[0][3] = 0.0;

    out.Jx[1][0] = Minv00 * dF1_dtb + Minv01 * dF2_dtb;
    out.Jx[1][1] = Minv00 * dF1_ddtb + Minv01 * dF2_ddtb;
    out.Jx[1][2] = Minv00 * (dF1_ds - corr1) + Minv01 * (dF2_ds - corr2);
    out.Jx[1][3] = Minv00 * dF1_ddts + Minv01 * dF2_ddts;

    out.Jx[2][0] = 0.0;
    out.Jx[2][1] = 0.0;
    out.Jx[2][2] = 0.0;
    out.Jx[2][3] = 1.0;

    out.Jx[3][0] = Minv01 * dF1_dtb + Minv11 * dF2_dtb;
    out.Jx[3][1] = Minv01 * dF1_ddtb + Minv11 * dF2_ddtb;
    out.Jx[3][2] = Minv01 * (dF1_ds - corr1) + Minv11 * (dF2_ds - corr2);
    out.Jx[3][3] = Minv01 * dF1_ddts + Minv11 * dF2_ddts;

    // ---------------- 输入雅可比 ----------------
    // 力矩线性进入 Q，故 d(qdd)/du = M^{-1} 的两行
    out.Ju[0][0] = 0.0;
    out.Ju[0][1] = 0.0;
    out.Ju[1][0] = Minv00;
    out.Ju[1][1] = Minv01;
    out.Ju[2][0] = 0.0;
    out.Ju[2][1] = 0.0;
    out.Ju[3][0] = Minv01;
    out.Ju[3][1] = Minv11;

    out.solvable = std::abs(det) >= kDetEpsilon;
}

/// 基座（theta_c）在相对时刻 tau 的位置与角速度，等角加速度外推。
inline void baseAt(double tc0, double dtc0, double ddtc, double tau, double& tc, double& dtc) {
    tc = tc0 + dtc0 * tau + 0.5 * ddtc * tau * tau;
    dtc = dtc0 + ddtc * tau;
}

inline void matmul4(const double A[4][4], const double B[4][4], double out[4][4]) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            out[i][j] = A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j] +
                        A[i][3] * B[3][j];
        }
    }
}

/// RK4 组合：x += (h/6)(k1 + 2 k2 + 2 k3 + k4)。
/// StateDerivative 的字段名沿用状态名：dtheta_b 即 d(theta_b)/dt，ddtheta_b 即其导数。
template <typename EvalT>
inline void rk4Combine(State& x, const EvalT& e1, const EvalT& e2, const EvalT& e3,
                       const EvalT& e4, double h) {
    const double w = h / 6.0;
    x.theta_b += w * (e1.dx.dtheta_b + 2.0 * e2.dx.dtheta_b + 2.0 * e3.dx.dtheta_b +
                      e4.dx.dtheta_b);
    x.dtheta_b += w * (e1.dx.ddtheta_b + 2.0 * e2.dx.ddtheta_b + 2.0 * e3.dx.ddtheta_b +
                       e4.dx.ddtheta_b);
    x.theta_s += w * (e1.dx.dtheta_s + 2.0 * e2.dx.dtheta_s + 2.0 * e3.dx.dtheta_s +
                      e4.dx.dtheta_s);
    x.dtheta_s += w * (e1.dx.ddtheta_s + 2.0 * e2.dx.ddtheta_s + 2.0 * e3.dx.ddtheta_s +
                       e4.dx.ddtheta_s);
}

inline double sq(double v) { return v * v; }

/// 六项损失中某一项的目标值（序列为空则取 0）。
inline double targetAt(const double* seq, std::size_t k) {
    return (seq != nullptr) ? seq[k] : 0.0;
}

/// 把"损失对第 k 步末状态的梯度"累加到 gL（不含力矩项）。
/// psi_b = tc + theta_b, psi_s = tc + theta_b + theta_s
/// dpsi_b = dtc + dtheta_b, dpsi_s = dtc + dtheta_b + dtheta_s
inline void accumulateStateLossGradient(const LossSpec& w,
                                        double tc,
                                        double dtc,
                                        const State& x,
                                        std::size_t k,
                                        double inv_k,
                                        double* gL) {
    const double r1 = (tc + x.theta_b) - targetAt(w.target_psi_b, k);
    const double r2 = (tc + x.theta_b + x.theta_s) - targetAt(w.target_psi_s, k);
    const double d1 = (dtc + x.dtheta_b) - targetAt(w.target_dpsi_b, k);
    const double d2 = (dtc + x.dtheta_b + x.dtheta_s) - targetAt(w.target_dpsi_s, k);

    // theta_b 同时进入 r1 与 r2；dtheta_b 同时进入 d1 与 d2
    gL[0] += inv_k * (w.w_psi_b * r1 + w.w_psi_s * r2);
    gL[1] += inv_k * (w.w_dpsi_b * d1 + w.w_dpsi_s * d2);
    gL[2] += inv_k * (w.w_psi_s * r2);
    gL[3] += inv_k * (w.w_dpsi_s * d2);
}

/// 第 k 步的标量损失（六项，已含 1/(2K)）。
inline double stepLoss(const LossSpec& w,
                       double tc,
                       double dtc,
                       const State& x,
                       std::size_t k,
                       double Tb,
                       double Ts,
                       double inv_k) {
    const double r1 = (tc + x.theta_b) - targetAt(w.target_psi_b, k);
    const double r2 = (tc + x.theta_b + x.theta_s) - targetAt(w.target_psi_s, k);
    const double d1 = (dtc + x.dtheta_b) - targetAt(w.target_dpsi_b, k);
    const double d2 = (dtc + x.dtheta_b + x.dtheta_s) - targetAt(w.target_dpsi_s, k);

    return 0.5 * inv_k *
           (w.w_psi_b * sq(r1) + w.w_psi_s * sq(r2) + w.w_dpsi_b * sq(d1) +
            w.w_dpsi_s * sq(d2) + w.w_tau_b * sq(Tb) + w.w_tau_s * sq(Ts));
}

/// 单个 RK4 子步的正向推进（不记录雅可比），供纯损失计算复用。
inline void advanceSubstep(const Params& p,
                           const Derived& d,
                           State& x,
                           double Tb,
                           double Ts,
                           double tc0,
                           double dtc0,
                           double ddtc,
                           double t_base,
                           double h) {
    double tc_s, dtc_s;
    Eval e1, e2, e3, e4;
    baseAt(tc0, dtc0, ddtc, t_base, tc_s, dtc_s);
    evaluate(p, d, x, Tb, Ts, tc_s, dtc_s, ddtc, e1);
    State x2 = advance(x, e1.dx, 0.5 * h);
    baseAt(tc0, dtc0, ddtc, t_base + 0.5 * h, tc_s, dtc_s);
    evaluate(p, d, x2, Tb, Ts, tc_s, dtc_s, ddtc, e2);
    State x3 = advance(x, e2.dx, 0.5 * h);
    evaluate(p, d, x3, Tb, Ts, tc_s, dtc_s, ddtc, e3);
    State x4 = advance(x, e3.dx, h);
    baseAt(tc0, dtc0, ddtc, t_base + h, tc_s, dtc_s);
    evaluate(p, d, x4, Tb, Ts, tc_s, dtc_s, ddtc, e4);

    rk4Combine(x, e1, e2, e3, e4, h);
}

}  // namespace

const char* validateTrajectoryConfig(const Params& p, std::size_t num_steps, double dt) {
    if (!(dt > 0.0)) return "dt must be > 0";
    if (num_steps < 1) return "num_steps must be >= 1";
    if (!(p.lambda > 0.0)) return "lambda must be > 0";
    return nullptr;
}

bool TrajectoryWorkspace::resize(std::size_t steps) {
    if (steps == num_steps && records != nullptr) return true;
    release();
    num_steps = steps;
    num_substeps = steps * static_cast<std::size_t>(kTrajectoryRefinement);
    records = static_cast<TrajectoryRecord*>(std::calloc(num_substeps, sizeof(TrajectoryRecord)));
    X = static_cast<State*>(std::calloc(steps, sizeof(State)));
    dLdX = static_cast<double*>(std::calloc(steps * 4, sizeof(double)));
    if (records == nullptr || X == nullptr || dLdX == nullptr) {
        release();
        return false;
    }
    return true;
}

void TrajectoryWorkspace::release() {
    std::free(records);
    std::free(X);
    std::free(dLdX);
    records = nullptr;
    X = nullptr;
    dLdX = nullptr;
    num_steps = 0;
    num_substeps = 0;
}

double simulateAndGradient(const Params& p,
                           double theta_c0,
                           double dtheta_c,
                           double ddtheta_c,
                           double dt,
                           std::size_t num_steps,
                           const double* tau,
                           const LossSpec& w,
                           const State& x0,
                           TrajectoryWorkspace& ws,
                           double* grad_tau,
                           State* out_final_state,
                           bool with_step_jacobians) {
    const Derived d = makeDerived(p);
    const double h = dt / static_cast<double>(kTrajectoryRefinement);
    const double inv_k = 1.0 / static_cast<double>(num_steps);

    if (tau == nullptr || !ws.resize(num_steps)) {
        if (grad_tau != nullptr) {
            for (std::size_t i = 0; i < num_steps * 2; ++i) grad_tau[i] = 0.0;
        }
        if (out_final_state != nullptr) *out_final_state = x0;
        return 0.0;
    }

    // =====================================================================
    // 1) 正向扫描：推进状态、累计损失、记录每个子步的阶段雅可比
    // =====================================================================
    double loss = 0.0;
    State x = x0;
    for (std::size_t k = 0; k < num_steps; ++k) {
        const double Tb = tau[2 * k + 0];
        const double Ts = tau[2 * k + 1];
        const double t_base = static_cast<double>(k) * dt;
        const std::size_t rbase = k * static_cast<std::size_t>(kTrajectoryRefinement);

        for (int s = 0; s < kTrajectoryRefinement; ++s) {
            TrajectoryRecord& rec = ws.records[rbase + static_cast<std::size_t>(s)];
            rec.x = x;

            double tc_s, dtc_s;
            Eval e1, e2, e3, e4;

            baseAt(theta_c0, dtheta_c, ddtheta_c, t_base, tc_s, dtc_s);
            evaluate(p, d, x, Tb, Ts, tc_s, dtc_s, ddtheta_c, e1);
            State x2 = advance(x, e1.dx, 0.5 * h);

            baseAt(theta_c0, dtheta_c, ddtheta_c, t_base + 0.5 * h, tc_s, dtc_s);
            evaluate(p, d, x2, Tb, Ts, tc_s, dtc_s, ddtheta_c, e2);
            State x3 = advance(x, e2.dx, 0.5 * h);

            evaluate(p, d, x3, Tb, Ts, tc_s, dtc_s, ddtheta_c, e3);
            State x4 = advance(x, e3.dx, h);

            baseAt(theta_c0, dtheta_c, ddtheta_c, t_base + h, tc_s, dtc_s);
            evaluate(p, d, x4, Tb, Ts, tc_s, dtc_s, ddtheta_c, e4);

            // 记录四阶段的 Jx / Ju（反向 VJP 直接消费）
            const Eval* ev[4] = {&e1, &e2, &e3, &e4};
            for (int i = 0; i < 4; ++i) {
                for (int a = 0; a < 4; ++a) {
                    for (int b = 0; b < 4; ++b) rec.Jx[i][a][b] = ev[i]->Jx[a][b];
                    rec.Ju[i][a][0] = ev[i]->Ju[a][0];
                    rec.Ju[i][a][1] = ev[i]->Ju[a][1];
                }
            }

            if (with_step_jacobians) {
                // A2 = I + h/2 J1, A3 = I + h/2 J2 A2, A4 = I + h J3 A3
                // Phi = I + (h/6)(J1 + 2 J2 A2 + 2 J3 A3 + J4 A4)
                // Psi = (h/6)(Ju1 + 2 Ju2 + 2 Ju3 + Ju4)      （u 在步内恒定）
                const double hh = 0.5 * h;
                double A2[4][4], A3[4][4], tmp[4][4];
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        A2[i][j] = (i == j ? 1.0 : 0.0) + hh * rec.Jx[0][i][j];
                    }
                }
                matmul4(rec.Jx[1], A2, tmp);
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) A3[i][j] = (i == j ? 1.0 : 0.0) + hh * tmp[i][j];
                }
                double acc[4][4];
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) acc[i][j] = rec.Jx[0][i][j] + rec.Jx[3][i][j];
                }
                matmul4(rec.Jx[1], A2, tmp);
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) acc[i][j] += 2.0 * tmp[i][j];
                }
                matmul4(rec.Jx[2], A3, tmp);
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        rec.Phi[i][j] =
                            (i == j ? 1.0 : 0.0) + (h / 6.0) * (acc[i][j] + 2.0 * tmp[i][j]);
                    }
                }
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 2; ++j) {
                        rec.Psi[i][j] = (h / 6.0) * (rec.Ju[0][i][j] + 2.0 * rec.Ju[1][i][j] +
                                                     2.0 * rec.Ju[2][i][j] + rec.Ju[3][i][j]);
                    }
                }
            }

            // 推进子步
            rk4Combine(x, e1, e2, e3, e4, h);
        }

        // 第 k 步末状态、损失与状态梯度
        ws.X[k] = x;
        double tc_end, dtc_end;
        baseAt(theta_c0, dtheta_c, ddtheta_c, t_base + dt, tc_end, dtc_end);
        loss += stepLoss(w, tc_end, dtc_end, x, k, Tb, Ts, inv_k);
        double* gL = ws.dLdX + 4 * k;
        gL[0] = gL[1] = gL[2] = gL[3] = 0.0;
        accumulateStateLossGradient(w, tc_end, dtc_end, x, k, inv_k, gL);
    }

    // =====================================================================
    // 2) 反向扫描：协态递推 + 逐步力矩梯度（纯 matvec，无新动力学求值）
    //
    //    反向直接用整子步 Jacobian 的显式形式，避免手推多阶段权重出错：
    //      y' = y + w1 k1 + w2 k2 + w2 k3 + w1 k4,  w1 = h/6, w2 = 2h/6
    //      dy'/dy = A4,  A2 = I + (h/2)J1, A3 = I + (h/2)J2 A2, A4 = I + h J3 A3
    //      dy'/du = Psi = (h/6)(Ju1 + 2 Ju2 + 2 Ju3 + Ju4)
    //    两者全部由正向已算出的 Jx/Ju 组装，不引入任何新的三角函数。
    // =====================================================================
    if (grad_tau != nullptr) {
        const double w1 = h / 6.0;
        const double w2 = 2.0 * h / 6.0;
        const double hh = 0.5 * h;

        double lambda[4] = {0.0, 0.0, 0.0, 0.0};
        for (std::size_t kk = num_steps; kk-- > 0;) {
            const double* gL = ws.dLdX + 4 * kk;
            lambda[0] += gL[0];
            lambda[1] += gL[1];
            lambda[2] += gL[2];
            lambda[3] += gL[3];

            double du0 = 0.0, du1 = 0.0;
            const std::size_t rbase = kk * static_cast<std::size_t>(kTrajectoryRefinement);
            for (int s = kTrajectoryRefinement; s-- > 0;) {
                const TrajectoryRecord& rec = ws.records[rbase + static_cast<std::size_t>(s)];

                // 组 A2, A3, A4（与正向同一套公式，纯算术）
                double A2[4][4], A3[4][4], A4[4][4], tmp[4][4];
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        A2[i][j] = (i == j ? 1.0 : 0.0) + hh * rec.Jx[0][i][j];
                    }
                }
                matmul4(rec.Jx[1], A2, tmp);
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) A3[i][j] = (i == j ? 1.0 : 0.0) + hh * tmp[i][j];
                }
                matmul4(rec.Jx[2], A3, tmp);
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) A4[i][j] = (i == j ? 1.0 : 0.0) + h * tmp[i][j];
                }

                // dL/dy = A4^T lambda
                double dY[4];
                for (int i = 0; i < 4; ++i) {
                    double acc = 0.0;
                    for (int a = 0; a < 4; ++a) acc += A4[a][i] * lambda[a];
                    dY[i] = acc;
                }

                // dL/du += Psi^T lambda
                for (int a = 0; a < 4; ++a) {
                    du0 += w1 * (rec.Ju[0][a][0] + rec.Ju[3][a][0]) * lambda[a] +
                           w2 * (rec.Ju[1][a][0] + rec.Ju[2][a][0]) * lambda[a];
                    du1 += w1 * (rec.Ju[0][a][1] + rec.Ju[3][a][1]) * lambda[a] +
                           w2 * (rec.Ju[1][a][1] + rec.Ju[2][a][1]) * lambda[a];
                }

                for (int i = 0; i < 4; ++i) lambda[i] = dY[i];
            }

            const double Tb = tau[2 * kk + 0];
            const double Ts = tau[2 * kk + 1];
            grad_tau[2 * kk + 0] = du0 + inv_k * w.w_tau_b * Tb;
            grad_tau[2 * kk + 1] = du1 + inv_k * w.w_tau_s * Ts;
        }
    }

    if (out_final_state != nullptr) *out_final_state = x;
    return loss;
}

double computeTrajectoryLoss(const Params& p,
                             double theta_c0,
                             double dtheta_c,
                             double ddtheta_c,
                             double dt,
                             std::size_t num_steps,
                             const double* tau,
                             double tau_b_fixed,
                             double tau_s_fixed,
                             const LossSpec& spec,
                             const State& x0,
                             double* out_psi_b,
                             double* out_psi_s,
                             double* out_dpsi_b,
                             double* out_dpsi_s) {
    const Derived d = makeDerived(p);
    const double h = dt / static_cast<double>(kTrajectoryRefinement);
    const double inv_k = 1.0 / static_cast<double>(num_steps);

    State x = x0;
    double loss = 0.0;
    for (std::size_t k = 0; k < num_steps; ++k) {
        const double Tb = (tau != nullptr) ? tau[2 * k + 0] : tau_b_fixed;
        const double Ts = (tau != nullptr) ? tau[2 * k + 1] : tau_s_fixed;
        const double t_base = static_cast<double>(k) * dt;

        for (int s = 0; s < kTrajectoryRefinement; ++s) {
            advanceSubstep(p, d, x, Tb, Ts, theta_c0, dtheta_c, ddtheta_c, t_base, h);
        }

        double tc_end, dtc_end;
        baseAt(theta_c0, dtheta_c, ddtheta_c, t_base + dt, tc_end, dtc_end);
        const double psi_b = tc_end + x.theta_b;
        const double psi_s = tc_end + x.theta_b + x.theta_s;
        const double dpsi_b = dtc_end + x.dtheta_b;
        const double dpsi_s = dtc_end + x.dtheta_b + x.dtheta_s;

        if (out_psi_b != nullptr) out_psi_b[k] = psi_b;
        if (out_psi_s != nullptr) out_psi_s[k] = psi_s;
        if (out_dpsi_b != nullptr) out_dpsi_b[k] = dpsi_b;
        if (out_dpsi_s != nullptr) out_dpsi_s[k] = dpsi_s;

        const double e1v = psi_b - targetAt(spec.target_psi_b, k);
        const double e2v = psi_s - targetAt(spec.target_psi_s, k);
        const double e3v = dpsi_b - targetAt(spec.target_dpsi_b, k);
        const double e4v = dpsi_s - targetAt(spec.target_dpsi_s, k);
        loss += 0.5 * inv_k *
                (spec.w_psi_b * sq(e1v) + spec.w_psi_s * sq(e2v) + spec.w_dpsi_b * sq(e3v) +
                 spec.w_dpsi_s * sq(e4v) + spec.w_tau_b * sq(Tb) + spec.w_tau_s * sq(Ts));
    }
    return loss;
}

}  // namespace tcbss
