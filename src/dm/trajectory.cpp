#include "trajectory.hpp"

#include <cmath>
#include <cstdlib>

// sincos 是 GNU 扩展：一次调用同时得到 sin 与 cos。
#if defined(__GNUC__) && !defined(__clang__)
#define TCBS_HAVE_SINCOS 1
#endif

namespace tcbs {
namespace dm {
namespace {

constexpr double kDetEpsilon = 1e-12;

inline void sincos3(double a, double b, double c, double& sa, double& ca, double& sb,
                    double& cb, double& sc, double& cc) {
#if TCBS_HAVE_SINCOS
    ::sincos(a, &sa, &ca);
    ::sincos(b, &sb, &cb);
    ::sincos(c, &sc, &cc);
#else
    sa = std::sin(a); ca = std::cos(a);
    sb = std::sin(b); cb = std::cos(b);
    sc = std::sin(c); cc = std::cos(c);
#endif
}

/// 一次动力学求值：前向导数 + 状态雅可比 + 输入雅可比。
struct Eval {
    StateDerivative dx;
    double Jx[4][4];   ///< x = [theta_b, dtheta_b, theta_s, dtheta_s]
    double Ju[4][2];   ///< u = [Tb, Ts]
};

struct Derived {
    double I_B, I_S, I_D;
    double gs_sin, gs_cos, gb_sin, gb_cos;
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

inline void evaluate(const Params& p,
                     const Derived& d,
                     const State& x,
                     double Tb,
                     double Ts,
                     double tc,
                     double dtc,
                     double ddtc,
                     Eval& out) {
    const double tb = x.theta_b, dtb = x.dtheta_b;
    const double ts = x.theta_s, dts = x.dtheta_s;

    const double psi_b = tc + tb;
    const double psi_s = psi_b + ts;

    double sb, cb, ss, cs, st, ct;
    sincos3(psi_b, psi_s, ts, sb, cb, ss, cs, st, ct);

    const double h = p.Dx * p.Psx * ct + p.Dy * p.Psy * ct + p.Dy * p.Psx * st -
                     p.Dx * p.Psy * st;
    const double dhd = -p.Dx * p.Psx * st - p.Dy * p.Psy * st + p.Dy * p.Psx * ct -
                       p.Dx * p.Psy * ct;

    const double M11 = d.I_B + d.I_D + d.I_S + 2.0 * p.ms * h;
    const double M12 = d.I_S + p.ms * h;
    const double M22 = d.I_S;

    const double dpb = dtc + dtb;
    const double cpl = p.ms * dhd;
    const double C1 = cpl * dts * (2.0 * dpb + dts);
    const double C2 = -cpl * dpb * dpb;

    const double G2 = d.gs_sin * ss + d.gs_cos * cs;
    const double G1 = d.gb_sin * sb + d.gb_cos * cb + G2;

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

    out.dx.dtheta_b = dtb;
    out.dx.ddtheta_b = qdd_b;
    out.dx.dtheta_s = dts;
    out.dx.ddtheta_s = qdd_s;

    const double dG2 = d.gs_sin * cs - d.gs_cos * ss;
    const double dG1_b = d.gb_sin * cb - d.gb_cos * sb + dG2;
    const double dM11 = 2.0 * p.ms * dhd;
    const double dM12 = p.ms * dhd;
    const double dC1_dts = -p.ms * h * dts * (2.0 * dpb + dts);
    const double dC2_dts = p.ms * h * dpb * dpb;
    const double dF1_dtb = -dG1_b;
    const double dF2_dtb = -dG2;
    const double dF1_ds = -dC1_dts - dG2 - dM11 * ddtc;
    const double dF2_ds = -dC2_dts - dG2 - dM12 * ddtc;
    const double corr1 = dM11 * qdd_b + dM12 * qdd_s;
    const double corr2 = dM12 * qdd_b;
    const double dC1_ddtb = cpl * dts * 2.0;
    const double dC2_ddtb = -cpl * 2.0 * dpb;
    const double dQb_ddtb = -p.fbv - p.fbc * p.lambda * (1.0 - th_tb * th_tb);
    const double dC1_ddts = cpl * (2.0 * dpb + 2.0 * dts);
    const double dQs_ddts = -p.fsv - p.fsc * p.lambda * (1.0 - th_ts * th_ts);

    out.Jx[0][0] = 0.0; out.Jx[0][1] = 1.0; out.Jx[0][2] = 0.0; out.Jx[0][3] = 0.0;
    out.Jx[1][0] = Minv00 * dF1_dtb + Minv01 * dF2_dtb;
    out.Jx[1][1] = Minv00 * (-dC1_ddtb + dQb_ddtb) + Minv01 * (-dC2_ddtb);
    out.Jx[1][2] = Minv00 * (dF1_ds - corr1) + Minv01 * (dF2_ds - corr2);
    out.Jx[1][3] = Minv00 * (-dC1_ddts) + Minv01 * dQs_ddts;
    out.Jx[2][0] = 0.0; out.Jx[2][1] = 0.0; out.Jx[2][2] = 0.0; out.Jx[2][3] = 1.0;
    out.Jx[3][0] = Minv01 * dF1_dtb + Minv11 * dF2_dtb;
    out.Jx[3][1] = Minv01 * (-dC1_ddtb + dQb_ddtb) + Minv11 * (-dC2_ddtb);
    out.Jx[3][2] = Minv01 * (dF1_ds - corr1) + Minv11 * (dF2_ds - corr2);
    out.Jx[3][3] = Minv01 * (-dC1_ddts) + Minv11 * dQs_ddts;

    out.Ju[0][0] = 0.0; out.Ju[0][1] = 0.0;
    out.Ju[1][0] = Minv00; out.Ju[1][1] = Minv01;
    out.Ju[2][0] = 0.0; out.Ju[2][1] = 0.0;
    out.Ju[3][0] = Minv01; out.Ju[3][1] = Minv11;
}

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

inline double sq(double v) { return v * v; }

inline double targetAt(const double* seq, std::size_t k) {
    return (seq != nullptr) ? seq[k] : 0.0;
}

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
    gL[0] = inv_k * (w.w_psi_b * r1 + w.w_psi_s * r2);
    gL[1] = inv_k * (w.w_dpsi_b * d1 + w.w_dpsi_s * d2);
    gL[2] = inv_k * (w.w_psi_s * r2);
    gL[3] = inv_k * (w.w_dpsi_s * d2);
}

inline double stepLoss(const LossSpec& w,
                       double tc,
                       double dtc,
                       const State& x,
                       std::size_t k,
                       double inv_k,
                       double Tb,
                       double Ts) {
    const double r1 = (tc + x.theta_b) - targetAt(w.target_psi_b, k);
    const double r2 = (tc + x.theta_b + x.theta_s) - targetAt(w.target_psi_s, k);
    const double d1 = (dtc + x.dtheta_b) - targetAt(w.target_dpsi_b, k);
    const double d2 = (dtc + x.dtheta_b + x.dtheta_s) - targetAt(w.target_dpsi_s, k);
    return 0.5 * inv_k *
           (w.w_psi_b * sq(r1) + w.w_psi_s * sq(r2) + w.w_dpsi_b * sq(d1) +
            w.w_dpsi_s * sq(d2) + w.w_tau_b * sq(Tb) + w.w_tau_s * sq(Ts));
}

/// 经典 4 阶段 RK4 的一个子步，与 include/rk4.hpp 的 rk4Step 完全同构：
///   k1 = f(t0, x),  k2 = f(t0+h/2, x + h/2 k1),
///   k3 = f(t0+h/2, x + h/2 k2),  k4 = f(t0+h, x + h k3)
///   x' = x + (h/6)(k1 + 2 k2 + 2 k3 + k4)
inline void rk4Classic(const Params& p,
                       const Derived& d,
                       State& x,
                       double Tb,
                       double Ts,
                       double tc0,
                       double dtc0,
                       double ddtc,
                       double t0,
                       double h,
                       Eval* ev) {
    double tc, dtc;
    baseAt(tc0, dtc0, ddtc, t0, tc, dtc);
    evaluate(p, d, x, Tb, Ts, tc, dtc, ddtc, ev[0]);
    const State x2 = advance(x, ev[0].dx, 0.5 * h);
    baseAt(tc0, dtc0, ddtc, t0 + 0.5 * h, tc, dtc);
    evaluate(p, d, x2, Tb, Ts, tc, dtc, ddtc, ev[1]);
    const State x3 = advance(x, ev[1].dx, 0.5 * h);
    evaluate(p, d, x3, Tb, Ts, tc, dtc, ddtc, ev[2]);
    const State x4 = advance(x, ev[2].dx, h);
    baseAt(tc0, dtc0, ddtc, t0 + h, tc, dtc);
    evaluate(p, d, x4, Tb, Ts, tc, dtc, ddtc, ev[3]);

    const double w = h / 6.0;
    x.theta_b += w * (ev[0].dx.dtheta_b + 2.0 * ev[1].dx.dtheta_b +
                      2.0 * ev[2].dx.dtheta_b + ev[3].dx.dtheta_b);
    x.dtheta_b += w * (ev[0].dx.ddtheta_b + 2.0 * ev[1].dx.ddtheta_b +
                       2.0 * ev[2].dx.ddtheta_b + ev[3].dx.ddtheta_b);
    x.theta_s += w * (ev[0].dx.dtheta_s + 2.0 * ev[1].dx.dtheta_s +
                      2.0 * ev[2].dx.dtheta_s + ev[3].dx.dtheta_s);
    x.dtheta_s += w * (ev[0].dx.ddtheta_s + 2.0 * ev[1].dx.ddtheta_s +
                       2.0 * ev[2].dx.ddtheta_s + ev[3].dx.ddtheta_s);
}

/// 一个子步的整步 Jacobian 与输入 Jacobian（供反向使用）：
///   A2 = I + (h/2) J1, A3 = I + (h/2) J2 A2, A4m = I + h J3 A3
///   dy'/dy = W = I + (h/6)(J1 + 2 J2 A2 + 2 J3 A3 + J4 A4m)
///   dy'/du = Psi = (h/6)(Ju1 + 2 Ju2 + 2 Ju3 + Ju4)
///
/// 注意：dy'/dy **不是** A4m。A4m 只是第 4 阶段那一路的链式因子，必须按
/// RK4 的 4 阶段权重 (1,2,2,1) 加权求和才等于整步传播子；直接用 A4m 会让
/// 反向只剩 O(h^3) 精度（正演是 O(h^5)），长轨迹上累积成可见梯度误差。
inline void stepJacobians(const Eval* ev, double h, double W[4][4], double Psi[4][2]) {
    double A2[4][4], A3[4][4], A4m[4][4], tmp[4][4];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) A2[i][j] = (i == j ? 1.0 : 0.0) + 0.5 * h * ev[0].Jx[i][j];
    }
    matmul4(ev[1].Jx, A2, tmp);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) A3[i][j] = (i == j ? 1.0 : 0.0) + 0.5 * h * tmp[i][j];
    }
    matmul4(ev[2].Jx, A3, tmp);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) A4m[i][j] = (i == j ? 1.0 : 0.0) + h * tmp[i][j];
    }
    const double w1 = h / 6.0, w2 = 2.0 * h / 6.0;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            double s = (i == j ? 1.0 : 0.0) + w1 * ev[0].Jx[i][j];
            for (int c = 0; c < 4; ++c) {
                s += w2 * (ev[1].Jx[i][c] * A2[c][j] + ev[2].Jx[i][c] * A3[c][j]) +
                     w1 * ev[3].Jx[i][c] * A4m[c][j];
            }
            W[i][j] = s;
        }
    }
    // dy'/du 同样必须带阶段耦合：τ 进入全部 4 个阶段，且阶段状态依赖 τ，
    //   du1 = Ju1
    //   du2 = Ju2 + (h/2) Jx2 du1
    //   du3 = Ju3 + (h/2) Jx3 du2
    //   du4 = Ju4 +  h    Jx4 du3
    //   Psi = (h/6)(du1 + 2 du2 + 2 du3 + du4)
    // 直接写 (Ju1+2Ju2+2Ju3+Ju4) 会漏掉上面那些耦合项，只剩 O(h^2) 精度。
    double du1[4][2], du2[4][2], du3[4][2], du4[4][2];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 2; ++j) du1[i][j] = ev[0].Ju[i][j];
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 2; ++j) {
            double a = ev[1].Ju[i][j];
            for (int c = 0; c < 4; ++c) a += 0.5 * h * ev[1].Jx[i][c] * du1[c][j];
            du2[i][j] = a;
        }
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 2; ++j) {
            double a = ev[2].Ju[i][j];
            for (int c = 0; c < 4; ++c) a += 0.5 * h * ev[2].Jx[i][c] * du2[c][j];
            du3[i][j] = a;
        }
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 2; ++j) {
            double a = ev[3].Ju[i][j];
            for (int c = 0; c < 4; ++c) a += h * ev[3].Jx[i][c] * du3[c][j];
            du4[i][j] = a;
        }
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 2; ++j) {
            Psi[i][j] = w1 * (du1[i][j] + du4[i][j]) + w2 * (du2[i][j] + du3[i][j]);
        }
    }
}

}  // namespace

const char* validateTrajectoryConfig(const Params& p,
                                     std::size_t num_steps,
                                     double dt,
                                     int refinement) {
    if (!(dt > 0.0)) return "dt must be > 0";
    if (num_steps < 1) return "num_steps must be >= 1";
    if (refinement < kRefinementMin || refinement > kRefinementMax) {
        return "refinement out of range";
    }
    if (!(p.lambda > 0.0)) return "lambda must be > 0";
    return nullptr;
}

bool TrajectoryWorkspace::resize(std::size_t steps, int refine) {
    if (steps == num_steps && refine == refinement && records != nullptr) return true;
    release();
    num_steps = steps;
    refinement = refine;
    num_substeps = steps * static_cast<std::size_t>(refine);
    records = static_cast<TrajectoryRecord*>(std::calloc(num_substeps, sizeof(TrajectoryRecord)));
    // 每子步：Jx 4 阶段 * 16 = 64；Ju 4 阶段 * 8 = 32；Phi 16；Psi 8
    const std::size_t per_sub = 64 + 32 + 24;
    stage_data = static_cast<double*>(std::calloc(num_substeps * per_sub, sizeof(double)));
    X = static_cast<State*>(std::calloc(steps, sizeof(State)));
    dLdX = static_cast<double*>(std::calloc(steps * 4, sizeof(double)));
    if (records == nullptr || stage_data == nullptr || X == nullptr || dLdX == nullptr) {
        release();
        return false;
    }
    for (std::size_t i = 0; i < num_substeps; ++i) {
        double* base = stage_data + i * per_sub;
        records[i].Jx = base;
        records[i].Ju = base + 64;
        records[i].phi_psi = base + 96;
    }
    return true;
}

void TrajectoryWorkspace::release() {
    std::free(records);
    std::free(stage_data);
    std::free(X);
    std::free(dLdX);
    records = nullptr;
    stage_data = nullptr;
    X = nullptr;
    dLdX = nullptr;
    num_steps = 0;
    refinement = 0;
    num_substeps = 0;
}

double simulateAndGradient(const Params& p,
                           double theta_c0,
                           double dtheta_c,
                           double ddtheta_c,
                           double dt,
                           int refinement,
                           std::size_t num_steps,
                           const double* tau,
                           const LossSpec& w,
                           const State& x0,
                           TrajectoryWorkspace& ws,
                           double* grad_tau,
                           State* out_final_state,
                           bool with_step_jacobians) {
    if (validateTrajectoryConfig(p, num_steps, dt, refinement) != nullptr || tau == nullptr ||
        !ws.resize(num_steps, refinement)) {
        if (grad_tau != nullptr) {
            for (std::size_t i = 0; i < num_steps * 2; ++i) grad_tau[i] = 0.0;
        }
        if (out_final_state != nullptr) *out_final_state = x0;
        return 0.0;
    }
    (void)with_step_jacobians;   // Phi/Psi 总是计算（反向需要）

    const Derived d = makeDerived(p);
    const double h = dt / static_cast<double>(refinement);
    const double inv_k = 1.0 / static_cast<double>(num_steps);

    Eval ev[4];

    // =====================================================================
    // 1) 正向扫描：每个主步重复 refinement 次经典 RK4 子步
    // =====================================================================
    double loss = 0.0;
    State x = x0;
    for (std::size_t k = 0; k < num_steps; ++k) {
        const double Tb = tau[2 * k + 0];
        const double Ts = tau[2 * k + 1];
        const double t_base = static_cast<double>(k) * dt;
        const std::size_t rbase = k * static_cast<std::size_t>(refinement);

        for (int s = 0; s < refinement; ++s) {
            TrajectoryRecord& rec = ws.records[rbase + static_cast<std::size_t>(s)];
            rec.x = x;
            // 与 Simulator::step 一致：第 s 个子步的起点时间用 s*h（相对本主步起点）
            rk4Classic(p, d, x, Tb, Ts, theta_c0, dtheta_c, ddtheta_c,
                       static_cast<double>(s) * h, h, ev);
            for (int st = 0; st < 4; ++st) {
                for (int a = 0; a < 4; ++a) {
                    rec.Ju[(st * 4 + a) * 2 + 0] = ev[st].Ju[a][0];
                    rec.Ju[(st * 4 + a) * 2 + 1] = ev[st].Ju[a][1];
                    for (int b = 0; b < 4; ++b) {
                        rec.Jx[((st * 4 + a) * 4) + b] = ev[st].Jx[a][b];
                    }
                }
            }
            double W[4][4], Psi[4][2];
            stepJacobians(ev, h, W, Psi);
            for (int a = 0; a < 4; ++a) {
                for (int b = 0; b < 4; ++b) rec.phi_psi[a * 4 + b] = W[a][b];
                rec.phi_psi[16 + a * 2 + 0] = Psi[a][0];
                rec.phi_psi[16 + a * 2 + 1] = Psi[a][1];
            }
        }

        ws.X[k] = x;
        double tc_end, dtc_end;
        baseAt(theta_c0, dtheta_c, ddtheta_c, t_base + dt, tc_end, dtc_end);
        loss += stepLoss(w, tc_end, dtc_end, x, k, inv_k, Tb, Ts);
        accumulateStateLossGradient(w, tc_end, dtc_end, x, k, inv_k, ws.dLdX + 4 * k);
    }

    // =====================================================================
    // 2) 反向扫描：协态递推（子步逆序）
    // =====================================================================
    if (grad_tau != nullptr) {
        double lambda[4] = {0.0, 0.0, 0.0, 0.0};
        for (std::size_t kk = num_steps; kk-- > 0;) {
            const double* gL = ws.dLdX + 4 * kk;
            lambda[0] += gL[0];
            lambda[1] += gL[1];
            lambda[2] += gL[2];
            lambda[3] += gL[3];

            double du0 = 0.0, du1 = 0.0;
            const std::size_t rbase = kk * static_cast<std::size_t>(refinement);
            for (int s = refinement; s-- > 0;) {
                const TrajectoryRecord& rec = ws.records[rbase + static_cast<std::size_t>(s)];
                const double* Phi = rec.phi_psi;
                const double* Psi = rec.phi_psi + 16;
                double dY[4];
                for (int i = 0; i < 4; ++i) {
                    double acc = 0.0;
                    for (int a = 0; a < 4; ++a) acc += Phi[a * 4 + i] * lambda[a];
                    dY[i] = acc;
                }
                for (int a = 0; a < 4; ++a) {
                    du0 += Psi[a * 2 + 0] * lambda[a];
                    du1 += Psi[a * 2 + 1] * lambda[a];
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
                             int refinement,
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
    if (validateTrajectoryConfig(p, num_steps, dt, refinement) != nullptr) return 0.0;

    const Derived d = makeDerived(p);
    const double h = dt / static_cast<double>(refinement);
    const double inv_k = 1.0 / static_cast<double>(num_steps);
    Eval ev[4];

    State x = x0;
    double loss = 0.0;
    for (std::size_t k = 0; k < num_steps; ++k) {
        const double Tb = (tau != nullptr) ? tau[2 * k + 0] : tau_b_fixed;
        const double Ts = (tau != nullptr) ? tau[2 * k + 1] : tau_s_fixed;
        const double t_base = static_cast<double>(k) * dt;

        for (int s = 0; s < refinement; ++s) {
            rk4Classic(p, d, x, Tb, Ts, theta_c0, dtheta_c, ddtheta_c,
                       static_cast<double>(s) * h, h, ev);
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
        loss += stepLoss(spec, tc_end, dtc_end, x, k, inv_k, Tb, Ts);
    }
    return loss;
}

}  // namespace dm
}  // namespace tcbs
