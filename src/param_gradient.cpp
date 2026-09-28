#include "param_gradient.hpp"

#include <cmath>
#include <cstdlib>

// 与 trajectory.cpp 相同的 sincos 优化（热区每子步 3 组三角）。
#if defined(__GNUC__) && !defined(__clang__)
#define TCBSS_PARAM_HAVE_SINCOS 1
#endif

namespace tcbss {
namespace {

constexpr double kDetEpsilon = 1e-12;

inline void sincos3(double a, double b, double c, double& sa, double& ca, double& sb,
                    double& cb, double& sc, double& cc) {
#if TCBSS_PARAM_HAVE_SINCOS
    ::sincos(a, &sa, &ca);
    ::sincos(b, &sb, &cb);
    ::sincos(c, &sc, &cc);
#else
    sa = std::sin(a); ca = std::cos(a);
    sb = std::sin(b); cb = std::cos(b);
    sc = std::sin(c); cc = std::cos(c);
#endif
}

/// 参数索引（与头文件注释、paramGradientNames() 严格一致）。
enum ParamIndex {
    kMb = 0, kIb, kPbx, kPby,
    kMs, kIs, kPsx, kPsy,
    kDx, kDy, kGx, kGy,
    kFbc, kFbv, kFsc, kFsv
};

constexpr int NP = kParamGradientCount;

/// 一次动力学求值：前向导数 + 状态雅可比 Jx(4x4) + 参数雅可比 Jp(4xNP)。
struct Eval {
    StateDerivative dx;
    double Jx[4][4];
    double Jp[4][NP];
};

/// 只算加速度（不含任何雅可比），供参数方向的局部差分使用。
/// 与 evaluate() 的前向部分逐行一致，保证差分与解析共享同一模型。
inline void accelOnly(const Params& p,
                      const State& x,
                      double Tb,
                      double Ts,
                      double tc,
                      double dtc,
                      double ddtc,
                      double& qdd_b,
                      double& qdd_s) {
    const double tb = x.theta_b, dtb = x.dtheta_b;
    const double ts = x.theta_s, dts = x.dtheta_s;
    const double psi_b = tc + tb;
    const double psi_s = psi_b + ts;

    double sb, cb, ss, cs, st, ct;
    sincos3(psi_b, psi_s, ts, sb, cb, ss, cs, st, ct);
    (void)sb; (void)cb;

    const double I_B = p.mb * (p.Pbx * p.Pbx + p.Pby * p.Pby) + p.Ib;
    const double I_S = p.ms * (p.Psx * p.Psx + p.Psy * p.Psy) + p.Is;
    const double I_D = p.ms * (p.Dx * p.Dx + p.Dy * p.Dy);
    const double h = p.Dx * p.Psx * ct + p.Dy * p.Psy * ct + p.Dy * p.Psx * st -
                     p.Dx * p.Psy * st;
    const double dhd = -p.Dx * p.Psx * st - p.Dy * p.Psy * st + p.Dy * p.Psx * ct -
                       p.Dx * p.Psy * ct;
    const double M11 = I_B + I_D + I_S + 2.0 * p.ms * h;
    const double M12 = I_S + p.ms * h;
    const double M22 = I_S;

    const double dpb = dtc + dtb;
    const double cpl = p.ms * dhd;
    const double C1 = cpl * dts * (2.0 * dpb + dts);
    const double C2 = -cpl * dpb * dpb;

    const double gs_sin = p.ms * (p.gx * p.Psx + p.gy * p.Psy);
    const double gs_cos = p.ms * (p.gx * p.Psy - p.gy * p.Psx);
    const double G2 = gs_sin * ss + gs_cos * cs;
    const double gb_sin = p.gx * (p.mb * p.Pbx + p.ms * p.Dx) +
                          p.gy * (p.mb * p.Pby + p.ms * p.Dy);
    const double gb_cos = p.gx * (p.mb * p.Pby + p.ms * p.Dy) -
                          p.gy * (p.mb * p.Pbx + p.ms * p.Dx);
    const double G1 = gb_sin * sb + gb_cos * cb + G2;

    const double Qb = Tb - p.fbv * dtb - p.fbc * std::tanh(p.lambda * dtb);
    const double Qs = Ts - p.fsv * dts - p.fsc * std::tanh(p.lambda * dts);

    const double F1 = Qb - C1 - G1 - M11 * ddtc;
    const double F2 = Qs - C2 - G2 - M12 * ddtc;
    const double det = M11 * M22 - M12 * M12;
    qdd_b = (F1 * M22 - F2 * M12) / det;
    qdd_s = (M11 * F2 - M12 * F1) / det;
}

/// 按参数索引构造扰动后的 Params（用于局部差分）。
inline Params paramsWithDelta(const Params& p, int idx, double delta) {
    double v[16] = {p.mb,  p.Ib,  p.Pbx, p.Pby, p.ms,  p.Is,  p.Psx, p.Psy,
                    p.Dx,  p.Dy,  p.gx,  p.gy,  p.fbc, p.fbv, p.fsc, p.fsv};
    v[idx] += delta;
    return Params(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
                  v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15], p.lambda);
}

/// 返回该参数的解析偏导是否已经验证过。
/// 已验证者用闭式；未验证者退回局部中心差分（正确但更慢）。
inline bool analyticPartialVerified(int idx) {
    (void)idx;
    return true;   // 全部 16 个参数的解析偏导均已用 FD 逐项核对
}

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

/// 单个参数对中间量的偏导。h / dhd 是 h 与 dh/dtheta_s 的参数偏导。
struct ParamPartials {
    double dh = 0.0, dhd = 0.0;
    double d_IB = 0.0, d_IS = 0.0, d_ID = 0.0;
    double d_gs_sin = 0.0, d_gs_cos = 0.0, d_gb_sin = 0.0, d_gb_cos = 0.0;
    double d_fbc = 0.0, d_fbv = 0.0, d_fsc = 0.0, d_fsv = 0.0;
    double d_ms = 0.0;
};

inline void paramPartials(const Params& p, double ct, double st, int idx,
                          ParamPartials& q) {
    switch (idx) {
        case kMb:
            // I_B = mb*(Pbx^2+Pby^2) + Ib  =>  dI_B/dmb = Pb^2
            q.d_IB = p.Pbx * p.Pbx + p.Pby * p.Pby;
            q.d_gb_sin = p.gx * p.Pbx + p.gy * p.Pby;
            q.d_gb_cos = p.gx * p.Pby - p.gy * p.Pbx;
            break;
        case kIb:
            q.d_IB = 1.0;
            break;
        case kPbx:
            // h / dhd 只依赖 D 与 Ps，不依赖 Pb —— 故 d h/d Pb* = 0
            q.d_IB = 2.0 * p.mb * p.Pbx;
            // gb_sin = gx(mb Pbx + ms Dx) + gy(mb Pby + ms Dy) => +gx*mb
            // gb_cos = gx(mb Pby + ms Dy) - gy(mb Pbx + ms Dx) => -gy*mb
            q.d_gb_sin = p.gx * p.mb;
            q.d_gb_cos = -p.gy * p.mb;
            break;
        case kPby:
            q.d_IB = 2.0 * p.mb * p.Pby;
            q.d_gb_sin = p.gy * p.mb;
            q.d_gb_cos = p.gx * p.mb;
            break;
        case kMs:
            q.d_ms = 1.0;
            // 表格核对值：I_S=Ps^2, I_D=D^2, gs_sin=gx Psx+gy Psy,
            // gs_cos=gx Psy-gy Psx, gb_sin=gx Dx+gy Dy, gb_cos=gy Dx-gx Dy
            q.d_IS = p.Psx * p.Psx + p.Psy * p.Psy;
            q.d_ID = p.Dx * p.Dx + p.Dy * p.Dy;
            q.d_gs_sin = p.gx * p.Psx + p.gy * p.Psy;
            q.d_gs_cos = p.gx * p.Psy - p.gy * p.Psx;
            q.d_gb_sin = p.gx * p.Dx + p.gy * p.Dy;
            q.d_gb_cos = p.gx * p.Dy - p.gy * p.Dx;
            break;
        case kIs:
            q.d_IS = 1.0;
            break;
        case kPsx:
            // h   = Dx cos + Dy sin 形式：dh/dPsx   = Dx*ct - Dy*st
            // dhd = -Dx sin + Dy cos 形式：d dhd/dPsx = -Dx*st - Dy*ct
            // h   = Psx(Dx ct + Dy st) + Psy(Dy ct - Dx st)
            // dhd = Psx(-Dx st + Dy ct) + Psy(-Dy st - Dx ct)
            q.dh = p.Dx * ct + p.Dy * st;
            q.dhd = -p.Dx * st + p.Dy * ct;
            q.d_IS = 2.0 * p.ms * p.Psx;
            q.d_gs_sin = p.ms * p.gx;
            q.d_gs_cos = -p.ms * p.gy;    // 表格：gs_cos = gx Psy - gy Psx
            break;
        case kPsy:
            q.dh = p.Dy * ct - p.Dx * st;
            q.dhd = -p.Dy * st - p.Dx * ct;
            q.d_IS = 2.0 * p.ms * p.Psy;
            q.d_gs_sin = p.ms * p.gy;     // 表格：gs_sin = gx Psx + gy Psy
            q.d_gs_cos = p.ms * p.gx;
            break;
        case kDx:
            q.dh = -p.Psy * st + p.Psx * ct;
            q.dhd = -p.Psy * ct - p.Psx * st;
            q.d_ID = 2.0 * p.ms * p.Dx;
            q.d_gb_sin = p.gx * p.ms;
            q.d_gb_cos = -p.gy * p.ms;
            break;
        case kDy:
            q.dh = p.Psy * ct + p.Psx * st;
            q.dhd = -p.Psy * st + p.Psx * ct;
            q.d_ID = 2.0 * p.ms * p.Dy;
            q.d_gb_sin = p.gy * p.ms;
            q.d_gb_cos = p.gx * p.ms;
            break;
        case kGx:
            q.d_gs_sin = p.ms * p.Psx;
            q.d_gs_cos = p.ms * p.Psy;
            q.d_gb_sin = p.mb * p.Pbx + p.ms * p.Dx;
            q.d_gb_cos = p.mb * p.Pby + p.ms * p.Dy;
            break;
        case kGy:
            q.d_gs_sin = p.ms * p.Psy;
            q.d_gs_cos = -p.ms * p.Psx;
            q.d_gb_sin = p.mb * p.Pby + p.ms * p.Dy;
            q.d_gb_cos = -(p.mb * p.Pbx + p.ms * p.Dx);
            break;
        case kFbc:
            q.d_fbc = 1.0;
            break;
        case kFbv:
            q.d_fbv = 1.0;
            break;
        case kFsc:
            q.d_fsc = 1.0;
            break;
        case kFsv:
            q.d_fsv = 1.0;
            break;
        default:
            break;
    }
}

/// 一次求值：前向导数 + 状态雅可比 + 参数雅可比，三者共用同一批三角函数。
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

    // ---------------- 状态雅可比 ----------------
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

    // ---------------- 参数雅可比 ----------------
    // d(qdd)/dp = Minv * (dF/dp - dM/dp * qdd)
    // 解析式只对已验证的参数使用；其余用局部中心差分（复用 accelOnly，
    // 保证与解析共享同一模型，代价是每参数 2 次额外加速度求值）。
    for (int i = 0; i < NP; ++i) {
        if (!analyticPartialVerified(i)) {
            const double base_i[16] = {p.mb,  p.Ib,  p.Pbx, p.Pby, p.ms,  p.Is,
                                       p.Psx, p.Psy, p.Dx,  p.Dy,  p.gx,  p.gy,
                                       p.fbc, p.fbv, p.fsc, p.fsv};
            const double scale = std::fabs(base_i[i]) > 1.0 ? std::fabs(base_i[i]) : 1.0;
            const double hstep = 1e-7 * scale;
            double bp, bs, mp, ms_;
            accelOnly(paramsWithDelta(p, i, hstep), x, Tb, Ts, tc, dtc, ddtc, bp, bs);
            accelOnly(paramsWithDelta(p, i, -hstep), x, Tb, Ts, tc, dtc, ddtc, mp, ms_);
            out.Jp[0][i] = 0.0;
            out.Jp[1][i] = (bp - mp) / (2.0 * hstep);
            out.Jp[2][i] = 0.0;
            out.Jp[3][i] = (bs - ms_) / (2.0 * hstep);
            continue;
        }
        ParamPartials q;
        paramPartials(p, ct, st, i, q);

        // M11 = I_B + I_D + I_S + 2*ms*h  =>  dM11 = dI_B+dI_D+dI_S + 2*h*d_ms + 2*ms*dh
        // M12 = I_S + ms*h                =>  dM12 = dI_S + h*d_ms + ms*dh
        // （2*h*d_ms 与 h*d_ms 两项：h 不随 ms 变，但 ms 是 h 的系数，容易漏）
        const double dM11p = q.d_IB + q.d_ID + q.d_IS + 2.0 * h * q.d_ms + 2.0 * p.ms * q.dh;
        const double dM12p = q.d_IS + h * q.d_ms + p.ms * q.dh;
        const double dM22p = q.d_IS;

        // d(ms*dhd)/dp = dms*dhd + ms*d(dhd)
        const double d_cpl = q.d_ms * dhd + p.ms * q.dhd;
        const double dC1p = d_cpl * dts * (2.0 * dpb + dts);
        const double dC2p = -d_cpl * dpb * dpb;

        const double dG2p = q.d_gs_sin * ss + q.d_gs_cos * cs;
        const double dG1p = q.d_gb_sin * sb + q.d_gb_cos * cb + dG2p;

        const double dQbp = -q.d_fbv * dtb - q.d_fbc * th_tb;
        const double dQsp = -q.d_fsv * dts - q.d_fsc * th_ts;

        const double dF1p = dQbp - dC1p - dG1p - dM11p * ddtc;
        const double dF2p = dQsp - dC2p - dG2p - dM12p * ddtc;

        const double e1 = dF1p - (dM11p * qdd_b + dM12p * qdd_s);
        const double e2 = dF2p - (dM12p * qdd_b + dM22p * qdd_s);

        out.Jp[0][i] = 0.0;
        out.Jp[1][i] = Minv00 * e1 + Minv01 * e2;
        out.Jp[2][i] = 0.0;
        out.Jp[3][i] = Minv01 * e1 + Minv11 * e2;
    }
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

inline double targetAt(const double* seq, std::size_t k) {
    return (seq != nullptr) ? seq[k] : 0.0;
}

inline void accumulateLossGrad(const ParamLossSpec& w,
                               double tc,
                               double dtc,
                               const State& x,
                               std::size_t k,
                               double inv_k,
                               double* g) {
    const double r1 = (tc + x.theta_b) - targetAt(w.target_psi_b, k);
    const double r2 = (tc + x.theta_b + x.theta_s) - targetAt(w.target_psi_s, k);
    const double d1 = (dtc + x.dtheta_b) - targetAt(w.target_dpsi_b, k);
    const double d2 = (dtc + x.dtheta_b + x.dtheta_s) - targetAt(w.target_dpsi_s, k);
    g[0] += inv_k * (w.w_psi_b * r1 + w.w_psi_s * r2);
    g[1] += inv_k * (w.w_dpsi_b * d1 + w.w_dpsi_s * d2);
    g[2] += inv_k * (w.w_psi_s * r2);
    g[3] += inv_k * (w.w_dpsi_s * d2);
}

inline double stepLoss(const ParamLossSpec& w,
                       double tc,
                       double dtc,
                       const State& x,
                       std::size_t k,
                       double inv_k) {
    const double r1 = (tc + x.theta_b) - targetAt(w.target_psi_b, k);
    const double r2 = (tc + x.theta_b + x.theta_s) - targetAt(w.target_psi_s, k);
    const double d1 = (dtc + x.dtheta_b) - targetAt(w.target_dpsi_b, k);
    const double d2 = (dtc + x.dtheta_b + x.dtheta_s) - targetAt(w.target_dpsi_s, k);
    return 0.5 * inv_k * (w.w_psi_b * r1 * r1 + w.w_psi_s * r2 * r2 +
                          w.w_dpsi_b * d1 * d1 + w.w_dpsi_s * d2 * d2);
}

const char* const kNames[NP] = {
    "mb", "Ib", "Pbx", "Pby", "ms", "Is", "Psx", "Psy",
    "Dx", "Dy", "gx", "gy", "fbc", "fbv", "fsc", "fsv"};

/// 单个 RK4 子步：给定起始状态与灵敏度 S，返回末端状态与末端灵敏度。
/// 用"四阶段 dk_i/dp 的显式链式"，与 dY/dy = A4 完全同构。
inline void rk4StepWithSens(const Params& p,
                            const Derived& d,
                            const State& x,
                            const double S[4][NP],
                            double Tb,
                            double Ts,
                            double tc0,
                            double dtc0,
                            double ddtc,
                            double t_base,
                            double h,
                            State& x_next,
                            double S_next[4][NP]) {
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

    // --- 状态推进 ---
    const double w1 = h / 6.0, w2 = 2.0 * h / 6.0;
    x_next.theta_b = x.theta_b + w1 * (e1.dx.dtheta_b + e4.dx.dtheta_b) +
                     w2 * (e2.dx.dtheta_b + e3.dx.dtheta_b);
    x_next.dtheta_b = x.dtheta_b + w1 * (e1.dx.ddtheta_b + e4.dx.ddtheta_b) +
                      w2 * (e2.dx.ddtheta_b + e3.dx.ddtheta_b);
    x_next.theta_s = x.theta_s + w1 * (e1.dx.dtheta_s + e4.dx.dtheta_s) +
                     w2 * (e2.dx.dtheta_s + e3.dx.dtheta_s);
    x_next.dtheta_s = x.dtheta_s + w1 * (e1.dx.ddtheta_s + e4.dx.ddtheta_s) +
                      w2 * (e2.dx.ddtheta_s + e3.dx.ddtheta_s);

    // --- 灵敏度：先求各阶段的 dk_i/dp ---
    //   dk1/dp = Jp1
    //   dk2/dp = Jp2 + (h/2) J1 dk1/dp
    //   dk3/dp = Jp3 + (h/2) J2 dk2/dp
    //   dk4/dp = Jp4 +  h    J3 dk3/dp
    // 再 dY'/dp = (h/6)(dk1 + 2 dk2 + 2 dk3 + dk4) = A4 * S + Jp_eff
    double dk[4][4][NP];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < NP; ++j) dk[0][i][j] = e1.Jp[i][j];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < NP; ++j) {
            double a = e2.Jp[i][j];
            for (int c = 0; c < 4; ++c) a += (0.5 * h) * e1.Jx[i][c] * dk[0][c][j];
            dk[1][i][j] = a;
        }
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < NP; ++j) {
            double a = e3.Jp[i][j];
            for (int c = 0; c < 4; ++c) a += (0.5 * h) * e2.Jx[i][c] * dk[1][c][j];
            dk[2][i][j] = a;
        }
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < NP; ++j) {
            double a = e4.Jp[i][j];
            for (int c = 0; c < 4; ++c) a += h * e3.Jx[i][c] * dk[2][c][j];
            dk[3][i][j] = a;
        }
    }

    // 整步：Phi 用状态雅可比链（与 dY/dy 一致），Jp_eff = (h/6)(dk1+2dk2+2dk3+dk4)
    const double hh = 0.5 * h;
    double A2[4][4], A3[4][4], A4m[4][4], tmp[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) A2[i][j] = (i == j ? 1.0 : 0.0) + hh * e1.Jx[i][j];
    matmul4(e2.Jx, A2, tmp);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) A3[i][j] = (i == j ? 1.0 : 0.0) + hh * tmp[i][j];
    matmul4(e3.Jx, A3, tmp);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) A4m[i][j] = (i == j ? 1.0 : 0.0) + h * tmp[i][j];

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < NP; ++j) {
            const double jp_eff = w1 * (dk[0][i][j] + dk[3][i][j]) +
                                  w2 * (dk[1][i][j] + dk[2][i][j]);
            double acc = 0.0;
            for (int a = 0; a < 4; ++a) acc += A4m[i][a] * S[a][j];
            S_next[i][j] = acc + jp_eff;
        }
    }
}

}  // namespace

const char* const* paramGradientNames() { return kNames; }

const char* validateParamGradientConfig(const Params& p, std::size_t num_steps, double dt) {
    if (!(dt > 0.0)) return "dt must be > 0";
    if (num_steps < 1) return "num_steps must be >= 1";
    if (!(p.lambda > 0.0)) return "lambda must be > 0";
    return nullptr;
}

bool ParamGradientWorkspace::resize(std::size_t steps) {
    if (steps == num_steps && X != nullptr) return true;
    release();
    num_steps = steps;
    num_substeps = steps * static_cast<std::size_t>(kParamRefinement);
    X = static_cast<State*>(std::calloc(steps, sizeof(State)));
    Xsub = static_cast<State*>(std::calloc(num_substeps, sizeof(State)));
    Jx = static_cast<double*>(std::calloc(num_substeps * 16, sizeof(double)));
    dLdX = static_cast<double*>(std::calloc(num_substeps * 4, sizeof(double)));
    if (X == nullptr || Xsub == nullptr || Jx == nullptr || dLdX == nullptr) {
        release();
        return false;
    }
    return true;
}

void ParamGradientWorkspace::release() {
    std::free(X); std::free(Xsub); std::free(Jx); std::free(dLdX);
    X = nullptr; Xsub = nullptr; Jx = nullptr; dLdX = nullptr;
    num_steps = 0;
    num_substeps = 0;
}

double computeParamGradientLoss(const Params& p,
                                double theta_c0,
                                double dtheta_c,
                                double ddtheta_c,
                                double dt,
                                std::size_t num_steps,
                                const double* tau,
                                const ParamLossSpec& spec,
                                const State& x0,
                                double* out_psi_b,
                                double* out_psi_s,
                                double* out_dpsi_b,
                                double* out_dpsi_s) {
    const Derived d = makeDerived(p);
    const double h = dt / static_cast<double>(kParamRefinement);
    const double inv_k = 1.0 / static_cast<double>(num_steps);

    State x = x0;
    double loss = 0.0;
    for (std::size_t k = 0; k < num_steps; ++k) {
        const double Tb = tau[2 * k + 0];
        const double Ts = tau[2 * k + 1];
        const double t_base = static_cast<double>(k) * dt;

        for (int s = 0; s < kParamRefinement; ++s) {
            double Sdummy[4][NP] = {};
            State x_next;
            rk4StepWithSens(p, d, x, Sdummy, Tb, Ts, theta_c0, dtheta_c, ddtheta_c, t_base, h,
                            x_next, Sdummy);
            x = x_next;
        }

        double tc_end, dtc_end;
        baseAt(theta_c0, dtheta_c, ddtheta_c, t_base + dt, tc_end, dtc_end);
        const double psi_b = tc_end + x.theta_b;
        const double psi_s = tc_end + x.theta_b + x.theta_s;
        const double dpsi_b = dtc_end + x.dtheta_b;
        const double dpsi_s = dtc_end + x.dtheta_b + x.dtheta_s;
        if (out_psi_b) out_psi_b[k] = psi_b;
        if (out_psi_s) out_psi_s[k] = psi_s;
        if (out_dpsi_b) out_dpsi_b[k] = dpsi_b;
        if (out_dpsi_s) out_dpsi_s[k] = dpsi_s;

        loss += stepLoss(spec, tc_end, dtc_end, x, k, inv_k);
    }
    return loss;
}

double computeParamGradient(const Params& p,
                            double theta_c0,
                            double dtheta_c,
                            double ddtheta_c,
                            double dt,
                            std::size_t num_steps,
                            const double* tau,
                            const ParamLossSpec& spec,
                            const State& x0,
                            ParamGradientWorkspace& ws,
                            double* grad_p,
                            State* out_final_state) {
    const Derived d = makeDerived(p);
    const double h = dt / static_cast<double>(kParamRefinement);
    const double inv_k = 1.0 / static_cast<double>(num_steps);

    if (tau == nullptr || !ws.resize(num_steps)) {
        if (grad_p != nullptr) {
            for (int i = 0; i < NP; ++i) grad_p[i] = 0.0;
        }
        if (out_final_state != nullptr) *out_final_state = x0;
        return 0.0;
    }

    // S = dX_k/dp，随步推进；不需要保存历史
    double S[4][NP] = {};
    double grad[NP] = {};

    double loss = 0.0;
    State x = x0;
    for (std::size_t k = 0; k < num_steps; ++k) {
        const double Tb = tau[2 * k + 0];
        const double Ts = tau[2 * k + 1];
        const double t_base = static_cast<double>(k) * dt;

        for (int s = 0; s < kParamRefinement; ++s) {
            double Snext[4][NP];
            State x_next;
            rk4StepWithSens(p, d, x, S, Tb, Ts, theta_c0, dtheta_c, ddtheta_c, t_base, h,
                            x_next, Snext);
            x = x_next;
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < NP; ++j) S[i][j] = Snext[i][j];
        }

        ws.X[k] = x;
        double tc_end, dtc_end;
        baseAt(theta_c0, dtheta_c, ddtheta_c, t_base + dt, tc_end, dtc_end);
        loss += stepLoss(spec, tc_end, dtc_end, x, k, inv_k);

        double g[4] = {0.0, 0.0, 0.0, 0.0};
        accumulateLossGrad(spec, tc_end, dtc_end, x, k, inv_k, g);
        for (int j = 0; j < NP; ++j) {
            double acc = 0.0;
            for (int i = 0; i < 4; ++i) acc += g[i] * S[i][j];
            grad[j] += acc;
        }
    }

    if (grad_p != nullptr) {
        for (int j = 0; j < NP; ++j) grad_p[j] = grad[j];
    }
    if (out_final_state != nullptr) *out_final_state = x;
    return loss;
}

}  // namespace tcbss
