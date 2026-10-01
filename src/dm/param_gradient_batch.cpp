// 批量（SoA）参数辨识模式的实现：见 include/dm/param_gradient_batch.hpp 的布局说明。
//
// 实现要点：
//   * 所有输入/输出都是 SoA（样本维 b 连续），因此对样本维的循环天然可 SIMD；
//   * 每个样本的轨迹互相独立 ⇒ 按样本切块交给多个 std::thread；
//   * 热路径 evaluateBatchLanes 分两遍：
//       1) 三角函数 / tanh 在**非向量化**的前置小循环里算好存进 mid（libm 调用
//          无法自动向量化）；
//       2) 其余全是纯算术，按 lane 循环 + `#pragma omp simd`；参数雅可比按
//          "参数索引在外、lane 在内" 展开，每个参数的公式都是一个干净的向量循环。
//   * 数学与 include/dm/param_gradient.hpp 的标量实现逐行对应；批量结果必须与
//     它对得上（见 tools/param_gradient_batch_check.cpp）。

#include "param_gradient_batch.hpp"

#include "trajectory.hpp"  // kRefinementMin / kRefinementMax

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(__GNUC__) || defined(__clang__)
#define TCBS_RESTRICT __restrict__
#else
#define TCBS_RESTRICT
#endif

// 与 param_gradient.cpp 相同的 sincos 优化。
#if defined(__GNUC__) && !defined(__clang__)
#define TCBS_PGB_HAVE_SINCOS 1
#endif

namespace tcbs {
namespace dm {
namespace {

constexpr int NP = kParamGradientCount;   // 14 个可辨识参数
constexpr int NQ = kBatchParamCount;      // 17 个 Params 字段

// params 的 SoA 下标（顺序同 Params 字段）
enum {
    qMb = 0, qIb, qPbx, qPby, qMs, qIs, qPsx, qPsy,
    qDx, qDy, qGx, qGy, qFbc, qFbv, qFsc, qFsv, qLambda
};

// 可辨识参数下标（顺序同 paramGradientNames()）
enum {
    iMb = 0, iIb, iPbx, iPby, iMs, iIs, iPsx, iPsy,
    iDx, iDy, iFbc, iFbv, iFsc, iFsv
};

// evaluateBatchLanes 的中间量槽位（单位 = cap）
enum {
    mH = 0, mDhd, mSs, mCs, mSb, mCb, mCt, mSt,
    mDpb, mDts, mDtb, mThTb, mThTs,
    mQddB, mQddS, mMinv00, mMinv01, mMinv11, mDdtc, NMID
};

// 每个 stage 的 Eval 缓冲布局（单位 = cap）
constexpr int eDx = 0;      // 4  * cap
constexpr int eJx = 4;      // 16 * cap
constexpr int eJp = 20;     // 4*NP * cap
constexpr int eStride = 20 + 4 * NP;   // = 76

inline void sincos3(double a, double b, double c, double& sa, double& ca, double& sb,
                    double& cb, double& sc, double& cc) {
#if TCBS_PGB_HAVE_SINCOS
    ::sincos(a, &sa, &ca);
    ::sincos(b, &sb, &cb);
    ::sincos(c, &sc, &cc);
#else
    sa = std::sin(a); ca = std::cos(a);
    sb = std::sin(b); cb = std::cos(b);
    sc = std::sin(c); cc = std::cos(c);
#endif
}

/// 由某个参数的部分导数写 Jp 的两行非零项（0/2 行恒为 0）。
/// 与标量实现 evaluate() 的参数雅可比段逐行一致。
inline void writeJp(const double* TCBS_RESTRICT mid, int cap, int b, double ms,
                    double d_h, double d_dhd,
                    double d_IB, double d_IS, double d_ID,
                    double d_gss, double d_gsc, double d_gbs, double d_gbc,
                    double d_fbc, double d_fbv, double d_fsc, double d_fsv,
                    double d_ms,
                    double* TCBS_RESTRICT jp, int i) {
    const double h     = mid[mH * cap + b];
    const double dhdv  = mid[mDhd * cap + b];
    const double dts   = mid[mDts * cap + b];
    const double dtb   = mid[mDtb * cap + b];
    const double dpb   = mid[mDpb * cap + b];
    const double th_tb = mid[mThTb * cap + b];
    const double th_ts = mid[mThTs * cap + b];
    const double qdd_b = mid[mQddB * cap + b];
    const double qdd_s = mid[mQddS * cap + b];
    const double Mi00  = mid[mMinv00 * cap + b];
    const double Mi01  = mid[mMinv01 * cap + b];
    const double Mi11  = mid[mMinv11 * cap + b];
    const double ss    = mid[mSs * cap + b];
    const double cs    = mid[mCs * cap + b];
    const double sb    = mid[mSb * cap + b];
    const double cb    = mid[mCb * cap + b];
    const double ddtc  = mid[mDdtc * cap + b];

    const double dM11p = d_IB + d_ID + d_IS + 2.0 * h * d_ms + 2.0 * ms * d_h;
    const double dM12p = d_IS + h * d_ms + ms * d_h;
    const double dM22p = d_IS;
    const double d_cpl = d_ms * dhdv + ms * d_dhd;
    const double dC1p = d_cpl * dts * (2.0 * dpb + dts);
    const double dC2p = -d_cpl * dpb * dpb;
    const double dG2p = d_gss * ss + d_gsc * cs;
    const double dG1p = d_gbs * sb + d_gbc * cb + dG2p;
    const double dQbp = -d_fbv * dtb - d_fbc * th_tb;
    const double dQsp = -d_fsv * dts - d_fsc * th_ts;
    const double dF1p = dQbp - dC1p - dG1p - dM11p * ddtc;
    const double dF2p = dQsp - dC2p - dG2p - dM12p * ddtc;
    const double e1 = dF1p - (dM11p * qdd_b + dM12p * qdd_s);
    const double e2 = dF2p - (dM12p * qdd_b + dM22p * qdd_s);

    jp[(0 * NP + i) * cap + b] = 0.0;
    jp[(1 * NP + i) * cap + b] = Mi00 * e1 + Mi01 * e2;
    jp[(2 * NP + i) * cap + b] = 0.0;
    jp[(3 * NP + i) * cap + b] = Mi01 * e1 + Mi11 * e2;
}

/// 一次批量求值：前向导数 dx(4) + 状态雅可比 Jx(16) + 参数雅可比 Jp(4*NP)。
/// 所有输入指针都已偏移到**本分块起点**，并以全局 batch 大小作为 stride。
void evaluateBatchLanes(const double* const* TCBS_RESTRICT pq,
                        const double* const* TCBS_RESTRICT px,
                        const double* TCBS_RESTRICT pTc,
                        const double* TCBS_RESTRICT pDtc,
                        const double* TCBS_RESTRICT pDdtc,
                        const double* TCBS_RESTRICT pTb,
                        const double* TCBS_RESTRICT pTs,
                        int n, int cap,
                        double* TCBS_RESTRICT mid,
                        double* TCBS_RESTRICT ev) {
    const double* TCBS_RESTRICT pMb = pq[qMb];
    const double* TCBS_RESTRICT pIb = pq[qIb];
    const double* TCBS_RESTRICT pPbx = pq[qPbx];
    const double* TCBS_RESTRICT pPby = pq[qPby];
    const double* TCBS_RESTRICT pMs = pq[qMs];
    const double* TCBS_RESTRICT pIs = pq[qIs];
    const double* TCBS_RESTRICT pPsx = pq[qPsx];
    const double* TCBS_RESTRICT pPsy = pq[qPsy];
    const double* TCBS_RESTRICT pDx = pq[qDx];
    const double* TCBS_RESTRICT pDy = pq[qDy];
    const double* TCBS_RESTRICT pGx = pq[qGx];
    const double* TCBS_RESTRICT pGy = pq[qGy];
    const double* TCBS_RESTRICT pFbc = pq[qFbc];
    const double* TCBS_RESTRICT pFbv = pq[qFbv];
    const double* TCBS_RESTRICT pFsc = pq[qFsc];
    const double* TCBS_RESTRICT pFsv = pq[qFsv];
    const double* TCBS_RESTRICT pLam = pq[qLambda];

    const double* TCBS_RESTRICT xtb = px[0];
    const double* TCBS_RESTRICT xdtb = px[1];
    const double* TCBS_RESTRICT xts = px[2];
    const double* TCBS_RESTRICT xdts = px[3];

    // ---- 前置（非 SIMD）：三角函数 + tanh ----
    for (int b = 0; b < n; ++b) {
        const double psib = pTc[b] + xtb[b];
        const double psis = psib + xts[b];
        double sb, cb, ss, cs, st, ct;
        sincos3(psib, psis, xts[b], sb, cb, ss, cs, st, ct);
        mid[mSb * cap + b] = sb;
        mid[mCb * cap + b] = cb;
        mid[mSs * cap + b] = ss;
        mid[mCs * cap + b] = cs;
        mid[mSt * cap + b] = st;
        mid[mCt * cap + b] = ct;
        mid[mThTb * cap + b] = std::tanh(pLam[b] * xdtb[b]);
        mid[mThTs * cap + b] = std::tanh(pLam[b] * xdts[b]);
    }

    // ---- 第一遍（SIMD）：前向 + 状态雅可比 + 中间量 ----
#pragma omp simd
    for (int b = 0; b < n; ++b) {
        const double mb_ = pMb[b], Ib_ = pIb[b];
        const double Pbx_ = pPbx[b], Pby_ = pPby[b];
        const double ms_ = pMs[b], Is_ = pIs[b];
        const double Psx_ = pPsx[b], Psy_ = pPsy[b];
        const double Dx_ = pDx[b], Dy_ = pDy[b];
        const double gx_ = pGx[b], gy_ = pGy[b];
        const double fbc_ = pFbc[b], fbv_ = pFbv[b];
        const double fsc_ = pFsc[b], fsv_ = pFsv[b];
        const double lam_ = pLam[b];
        const double dtb = xdtb[b], dts = xdts[b];
        const double ddtc = pDdtc[b];
        const double sb = mid[mSb * cap + b], cb = mid[mCb * cap + b];
        const double ss = mid[mSs * cap + b], cs = mid[mCs * cap + b];
        const double st = mid[mSt * cap + b], ct = mid[mCt * cap + b];

        const double I_B = mb_ * (Pbx_ * Pbx_ + Pby_ * Pby_) + Ib_;
        const double I_S = ms_ * (Psx_ * Psx_ + Psy_ * Psy_) + Is_;
        const double I_D = ms_ * (Dx_ * Dx_ + Dy_ * Dy_);
        const double h = Dx_ * Psx_ * ct + Dy_ * Psy_ * ct + Dy_ * Psx_ * st -
                         Dx_ * Psy_ * st;
        const double dhd = -Dx_ * Psx_ * st - Dy_ * Psy_ * st + Dy_ * Psx_ * ct -
                           Dx_ * Psy_ * ct;
        const double M11 = I_B + I_D + I_S + 2.0 * ms_ * h;
        const double M12 = I_S + ms_ * h;
        const double M22 = I_S;

        const double dpb = pDtc[b] + dtb;
        const double cpl = ms_ * dhd;
        const double C1 = cpl * dts * (2.0 * dpb + dts);
        const double C2 = -cpl * dpb * dpb;

        const double gs_sin = ms_ * (gx_ * Psx_ + gy_ * Psy_);
        const double gs_cos = ms_ * (gx_ * Psy_ - gy_ * Psx_);
        const double G2 = gs_sin * ss + gs_cos * cs;
        const double gb_sin = gx_ * (mb_ * Pbx_ + ms_ * Dx_) + gy_ * (mb_ * Pby_ + ms_ * Dy_);
        const double gb_cos = gx_ * (mb_ * Pby_ + ms_ * Dy_) - gy_ * (mb_ * Pbx_ + ms_ * Dx_);
        const double G1 = gb_sin * sb + gb_cos * cb + G2;

        const double th_tb = mid[mThTb * cap + b];
        const double th_ts = mid[mThTs * cap + b];
        const double Qb = pTb[b] - fbv_ * dtb - fbc_ * th_tb;
        const double Qs = pTs[b] - fsv_ * dts - fsc_ * th_ts;

        const double F1 = Qb - C1 - G1 - M11 * ddtc;
        const double F2 = Qs - C2 - G2 - M12 * ddtc;
        const double det = M11 * M22 - M12 * M12;
        const double invdet = 1.0 / det;
        const double Mi00 = M22 * invdet;
        const double Mi01 = -M12 * invdet;
        const double Mi11 = M11 * invdet;
        const double qdd_b = Mi00 * F1 + Mi01 * F2;
        const double qdd_s = Mi01 * F1 + Mi11 * F2;

        mid[mH * cap + b] = h;
        mid[mDhd * cap + b] = dhd;
        mid[mDpb * cap + b] = dpb;
        mid[mDts * cap + b] = dts;
        mid[mDtb * cap + b] = dtb;
        mid[mQddB * cap + b] = qdd_b;
        mid[mQddS * cap + b] = qdd_s;
        mid[mMinv00 * cap + b] = Mi00;
        mid[mMinv01 * cap + b] = Mi01;
        mid[mMinv11 * cap + b] = Mi11;
        mid[mDdtc * cap + b] = ddtc;

        ev[(eDx + 0) * cap + b] = dtb;
        ev[(eDx + 1) * cap + b] = qdd_b;
        ev[(eDx + 2) * cap + b] = dts;
        ev[(eDx + 3) * cap + b] = qdd_s;

        // ---- 状态雅可比（与标量 evaluate 逐行一致）----
        const double dG2 = gs_sin * cs - gs_cos * ss;
        const double dG1_b = gb_sin * cb - gb_cos * sb + dG2;
        const double dM11 = 2.0 * ms_ * dhd;
        const double dM12 = ms_ * dhd;
        const double dC1_dts = -ms_ * h * dts * (2.0 * dpb + dts);
        const double dC2_dts = ms_ * h * dpb * dpb;
        const double dF1_dtb = -dG1_b;
        const double dF2_dtb = -dG2;
        const double dF1_ds = -dC1_dts - dG2 - dM11 * ddtc;
        const double dF2_ds = -dC2_dts - dG2 - dM12 * ddtc;
        const double corr1 = dM11 * qdd_b + dM12 * qdd_s;
        const double corr2 = dM12 * qdd_b;
        const double dC1_ddtb = cpl * dts * 2.0;
        const double dC2_ddtb = -cpl * 2.0 * dpb;
        const double dQb_ddtb = -fbv_ - fbc_ * lam_ * (1.0 - th_tb * th_tb);
        const double dC1_ddts = cpl * (2.0 * dpb + 2.0 * dts);
        const double dQs_ddts = -fsv_ - fsc_ * lam_ * (1.0 - th_ts * th_ts);

        double* Jx = ev + eJx * cap;
        Jx[(0 * 4 + 0) * cap + b] = 0.0;
        Jx[(0 * 4 + 1) * cap + b] = 1.0;
        Jx[(0 * 4 + 2) * cap + b] = 0.0;
        Jx[(0 * 4 + 3) * cap + b] = 0.0;
        Jx[(1 * 4 + 0) * cap + b] = Mi00 * dF1_dtb + Mi01 * dF2_dtb;
        Jx[(1 * 4 + 1) * cap + b] = Mi00 * (-dC1_ddtb + dQb_ddtb) + Mi01 * (-dC2_ddtb);
        Jx[(1 * 4 + 2) * cap + b] = Mi00 * (dF1_ds - corr1) + Mi01 * (dF2_ds - corr2);
        Jx[(1 * 4 + 3) * cap + b] = Mi00 * (-dC1_ddts) + Mi01 * dQs_ddts;
        Jx[(2 * 4 + 0) * cap + b] = 0.0;
        Jx[(2 * 4 + 1) * cap + b] = 0.0;
        Jx[(2 * 4 + 2) * cap + b] = 0.0;
        Jx[(2 * 4 + 3) * cap + b] = 1.0;
        Jx[(3 * 4 + 0) * cap + b] = Mi01 * dF1_dtb + Mi11 * dF2_dtb;
        Jx[(3 * 4 + 1) * cap + b] = Mi01 * (-dC1_ddtb + dQb_ddtb) + Mi11 * (-dC2_ddtb);
        Jx[(3 * 4 + 2) * cap + b] = Mi01 * (dF1_ds - corr1) + Mi11 * (dF2_ds - corr2);
        Jx[(3 * 4 + 3) * cap + b] = Mi01 * (-dC1_ddts) + Mi11 * dQs_ddts;
    }

    // ---- 第二遍（SIMD）：14 个参数的 Jp ----
    double* Jp = ev + eJp * cap;
    const double* TCBS_RESTRICT ct = mid + mCt * cap;
    const double* TCBS_RESTRICT st = mid + mSt * cap;

    for (int i = 0; i < NP; ++i) {
        switch (i) {
            case iMb:
#pragma omp simd
                for (int b = 0; b < n; ++b) {
                    const double Pbx_ = pPbx[b], Pby_ = pPby[b];
                    const double gx_ = pGx[b], gy_ = pGy[b];
                    writeJp(mid, cap, b, pMs[b], 0.0, 0.0,
                            Pbx_ * Pbx_ + Pby_ * Pby_, 0.0, 0.0,
                            0.0, 0.0, gx_ * Pbx_ + gy_ * Pby_, gx_ * Pby_ - gy_ * Pbx_,
                            0.0, 0.0, 0.0, 0.0, 0.0, Jp, i);
                }
                break;
            case iIb:
#pragma omp simd
                for (int b = 0; b < n; ++b)
                    writeJp(mid, cap, b, pMs[b], 0.0, 0.0, 1.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, Jp, i);
                break;
            case iPbx:
#pragma omp simd
                for (int b = 0; b < n; ++b) {
                    const double mb_ = pMb[b], Pbx_ = pPbx[b];
                    const double gx_ = pGx[b], gy_ = pGy[b];
                    writeJp(mid, cap, b, pMs[b], 0.0, 0.0,
                            2.0 * mb_ * Pbx_, 0.0, 0.0,
                            0.0, 0.0, gx_ * mb_, -gy_ * mb_,
                            0.0, 0.0, 0.0, 0.0, 0.0, Jp, i);
                }
                break;
            case iPby:
#pragma omp simd
                for (int b = 0; b < n; ++b) {
                    const double mb_ = pMb[b], Pby_ = pPby[b];
                    const double gx_ = pGx[b], gy_ = pGy[b];
                    writeJp(mid, cap, b, pMs[b], 0.0, 0.0,
                            2.0 * mb_ * Pby_, 0.0, 0.0,
                            0.0, 0.0, gy_ * mb_, gx_ * mb_,
                            0.0, 0.0, 0.0, 0.0, 0.0, Jp, i);
                }
                break;
            case iMs:
#pragma omp simd
                for (int b = 0; b < n; ++b) {
                    const double Psx_ = pPsx[b], Psy_ = pPsy[b];
                    const double Dx_ = pDx[b], Dy_ = pDy[b];
                    const double gx_ = pGx[b], gy_ = pGy[b];
                    writeJp(mid, cap, b, pMs[b], 0.0, 0.0,
                            0.0, Psx_ * Psx_ + Psy_ * Psy_, Dx_ * Dx_ + Dy_ * Dy_,
                            gx_ * Psx_ + gy_ * Psy_, gx_ * Psy_ - gy_ * Psx_,
                            gx_ * Dx_ + gy_ * Dy_, gx_ * Dy_ - gy_ * Dx_,
                            0.0, 0.0, 0.0, 0.0, 1.0, Jp, i);
                }
                break;
            case iIs:
#pragma omp simd
                for (int b = 0; b < n; ++b)
                    writeJp(mid, cap, b, pMs[b], 0.0, 0.0, 0.0, 1.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, Jp, i);
                break;
            case iPsx:
#pragma omp simd
                for (int b = 0; b < n; ++b) {
                    const double ms_ = pMs[b], Psx_ = pPsx[b];
                    const double Dx_ = pDx[b], Dy_ = pDy[b];
                    const double gx_ = pGx[b], gy_ = pGy[b];
                    const double ct_ = ct[b], st_ = st[b];
                    writeJp(mid, cap, b, ms_,
                            Dx_ * ct_ + Dy_ * st_, -Dx_ * st_ + Dy_ * ct_,
                            0.0, 2.0 * ms_ * Psx_, 0.0,
                            ms_ * gx_, -ms_ * gy_, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, Jp, i);
                }
                break;
            case iPsy:
#pragma omp simd
                for (int b = 0; b < n; ++b) {
                    const double ms_ = pMs[b], Psy_ = pPsy[b];
                    const double Dx_ = pDx[b], Dy_ = pDy[b];
                    const double gx_ = pGx[b], gy_ = pGy[b];
                    const double ct_ = ct[b], st_ = st[b];
                    writeJp(mid, cap, b, ms_,
                            Dy_ * ct_ - Dx_ * st_, -Dy_ * st_ - Dx_ * ct_,
                            0.0, 2.0 * ms_ * Psy_, 0.0,
                            ms_ * gy_, ms_ * gx_, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, Jp, i);
                }
                break;
            case iDx:
#pragma omp simd
                for (int b = 0; b < n; ++b) {
                    const double ms_ = pMs[b], Psx_ = pPsx[b], Psy_ = pPsy[b];
                    const double Dx_ = pDx[b];
                    const double gx_ = pGx[b], gy_ = pGy[b];
                    const double ct_ = ct[b], st_ = st[b];
                    writeJp(mid, cap, b, ms_,
                            -Psy_ * st_ + Psx_ * ct_, -Psy_ * ct_ - Psx_ * st_,
                            0.0, 0.0, 2.0 * ms_ * Dx_,
                            0.0, 0.0, gx_ * ms_, -gy_ * ms_,
                            0.0, 0.0, 0.0, 0.0, 0.0, Jp, i);
                }
                break;
            case iDy:
#pragma omp simd
                for (int b = 0; b < n; ++b) {
                    const double ms_ = pMs[b], Psx_ = pPsx[b], Psy_ = pPsy[b];
                    const double Dy_ = pDy[b];
                    const double gx_ = pGx[b], gy_ = pGy[b];
                    const double ct_ = ct[b], st_ = st[b];
                    writeJp(mid, cap, b, ms_,
                            Psy_ * ct_ + Psx_ * st_, -Psy_ * st_ + Psx_ * ct_,
                            0.0, 0.0, 2.0 * ms_ * Dy_,
                            0.0, 0.0, gy_ * ms_, gx_ * ms_,
                            0.0, 0.0, 0.0, 0.0, 0.0, Jp, i);
                }
                break;
            case iFbc:
#pragma omp simd
                for (int b = 0; b < n; ++b)
                    writeJp(mid, cap, b, pMs[b], 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, Jp, i);
                break;
            case iFbv:
#pragma omp simd
                for (int b = 0; b < n; ++b)
                    writeJp(mid, cap, b, pMs[b], 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, Jp, i);
                break;
            case iFsc:
#pragma omp simd
                for (int b = 0; b < n; ++b)
                    writeJp(mid, cap, b, pMs[b], 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, Jp, i);
                break;
            case iFsv:
            default:
#pragma omp simd
                for (int b = 0; b < n; ++b)
                    writeJp(mid, cap, b, pMs[b], 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, Jp, i);
                break;
        }
    }
}

/// 每个线程的临时缓冲。所有数组的 lane 维步长都是 cap。
struct Scratch {
    std::vector<double> X;       ///< 4*cap，当前状态（SoA）
    std::vector<double> S;       ///< 4*NP*cap，dX/dp
    std::vector<double> Snext;   ///< 4*NP*cap
    std::vector<double> dk;      ///< 4 * 4*NP*cap（dk1..dk4）
    std::vector<double> ev;      ///< 4 * eStride*cap（4 个 stage 的 Eval）
    std::vector<double> mid;     ///< NMID*cap
    std::vector<double> adv;     ///< 3 * 4*cap（x2/x3/x4）
    std::vector<double> tc;      ///< cap
    std::vector<double> dtc;     ///< cap
    std::vector<double> g;       ///< 4*cap
    std::vector<double> loss;    ///< cap
    std::vector<double> grad;    ///< NP*cap

    void alloc(int lanes) {
        const std::size_t L = static_cast<std::size_t>(lanes);
        X.assign(4 * L, 0.0);
        S.assign(static_cast<std::size_t>(4 * NP) * L, 0.0);
        Snext.assign(static_cast<std::size_t>(4 * NP) * L, 0.0);
        dk.assign(static_cast<std::size_t>(4 * 4 * NP) * L, 0.0);
        ev.assign(static_cast<std::size_t>(4 * eStride) * L, 0.0);
        mid.assign(static_cast<std::size_t>(NMID) * L, 0.0);
        adv.assign(static_cast<std::size_t>(12) * L, 0.0);
        tc.assign(L, 0.0);
        dtc.assign(L, 0.0);
        g.assign(4 * L, 0.0);
        loss.assign(L, 0.0);
        grad.assign(static_cast<std::size_t>(NP) * L, 0.0);
    }
};

/// 基座量按等角加速度外推到时刻 t（每 lane）。
inline void fillBase(double* TCBS_RESTRICT tc, double* TCBS_RESTRICT dtc,
                     const double* TCBS_RESTRICT pTc0,
                     const double* TCBS_RESTRICT pDtc,
                     const double* TCBS_RESTRICT pDdtc,
                     double t, int n) {
#pragma omp simd
    for (int b = 0; b < n; ++b) {
        const double dtc0 = pDtc[b], ddtc = pDdtc[b];
        tc[b] = pTc0[b] + dtc0 * t + 0.5 * ddtc * t * t;
        dtc[b] = dtc0 + ddtc * t;
    }
}

/// 单个子步的 RK4 + 参数灵敏度（批量 lane 版），与标量 rk4StepWithSens 一致。
inline void rk4StepSensBatch(const double* const* pq,
                             const double* TCBS_RESTRICT pTc0,
                             const double* TCBS_RESTRICT pDtc,
                             const double* TCBS_RESTRICT pDdtc,
                             const double* TCBS_RESTRICT pTb,
                             const double* TCBS_RESTRICT pTs,
                             double t0, double h, int n, int cap, Scratch& s) {
    double* TCBS_RESTRICT X = s.X.data();
    double* TCBS_RESTRICT S = s.S.data();
    double* TCBS_RESTRICT ev = s.ev.data();
    double* TCBS_RESTRICT mid = s.mid.data();
    double* TCBS_RESTRICT adv = s.adv.data();
    double* TCBS_RESTRICT dk = s.dk.data();

    const double* px[4] = {X + 0 * cap, X + 1 * cap, X + 2 * cap, X + 3 * cap};
    const double* px2[4] = {adv + 0 * cap, adv + 1 * cap, adv + 2 * cap, adv + 3 * cap};
    const double* px3[4] = {adv + 4 * cap, adv + 5 * cap, adv + 6 * cap, adv + 7 * cap};
    const double* px4[4] = {adv + 8 * cap, adv + 9 * cap, adv + 10 * cap, adv + 11 * cap};

    // stage 0
    fillBase(s.tc.data(), s.dtc.data(), pTc0, pDtc, pDdtc, t0, n);
    evaluateBatchLanes(pq, px, s.tc.data(), s.dtc.data(), pDdtc, pTb, pTs, n, cap, mid, ev);
#pragma omp simd
    for (int b = 0; b < n; ++b)
        for (int i = 0; i < 4; ++i)
            adv[i * cap + b] = X[i * cap + b] + 0.5 * h * ev[(eDx + i) * cap + b];

    // stage 1
    fillBase(s.tc.data(), s.dtc.data(), pTc0, pDtc, pDdtc, t0 + 0.5 * h, n);
    evaluateBatchLanes(pq, px2, s.tc.data(), s.dtc.data(), pDdtc, pTb, pTs, n, cap, mid,
                       ev + static_cast<std::size_t>(eStride) * cap);
#pragma omp simd
    for (int b = 0; b < n; ++b)
        for (int i = 0; i < 4; ++i)
            adv[(4 + i) * cap + b] =
                X[i * cap + b] +
                0.5 * h * ev[(eStride + eDx + i) * cap + b];

    // stage 2
    fillBase(s.tc.data(), s.dtc.data(), pTc0, pDtc, pDdtc, t0 + 0.5 * h, n);
    evaluateBatchLanes(pq, px3, s.tc.data(), s.dtc.data(), pDdtc, pTb, pTs, n, cap, mid,
                       ev + static_cast<std::size_t>(2 * eStride) * cap);
#pragma omp simd
    for (int b = 0; b < n; ++b)
        for (int i = 0; i < 4; ++i)
            adv[(8 + i) * cap + b] =
                X[i * cap + b] +
                h * ev[(2 * eStride + eDx + i) * cap + b];

    // stage 3
    fillBase(s.tc.data(), s.dtc.data(), pTc0, pDtc, pDdtc, t0 + h, n);
    evaluateBatchLanes(pq, px4, s.tc.data(), s.dtc.data(), pDdtc, pTb, pTs, n, cap, mid,
                       ev + static_cast<std::size_t>(3 * eStride) * cap);

    // 状态推进：X' = X + (h/6)(dx0 + 2dx1 + 2dx2 + dx3)
    const double w1 = h / 6.0, w2 = 2.0 * h / 6.0;
#pragma omp simd
    for (int b = 0; b < n; ++b)
        for (int i = 0; i < 4; ++i) {
            const double d0 = ev[(eDx + i) * cap + b];
            const double d1 = ev[(eStride + eDx + i) * cap + b];
            const double d2 = ev[(2 * eStride + eDx + i) * cap + b];
            const double d3 = ev[(3 * eStride + eDx + i) * cap + b];
            X[i * cap + b] += w1 * (d0 + d3) + w2 * (d1 + d2);
        }

    // ---- 参数灵敏度前传（与标量实现逐项对应）----
    double* TCBS_RESTRICT dk1 = dk;
    double* TCBS_RESTRICT dk2 = dk + static_cast<std::size_t>(4 * NP) * cap;
    double* TCBS_RESTRICT dk3 = dk + static_cast<std::size_t>(8 * NP) * cap;
    double* TCBS_RESTRICT dk4 = dk + static_cast<std::size_t>(12 * NP) * cap;
    double* TCBS_RESTRICT Sn = s.Snext.data();

    const double* Jp0 = ev + eJp * cap;
    const double* Jp1 = ev + static_cast<std::size_t>(eStride + eJp) * cap;
    const double* Jp2 = ev + static_cast<std::size_t>(2 * eStride + eJp) * cap;
    const double* Jp3 = ev + static_cast<std::size_t>(3 * eStride + eJp) * cap;
    const double* Jx0 = ev + eJx * cap;
    const double* Jx1 = ev + static_cast<std::size_t>(eStride + eJx) * cap;
    const double* Jx2 = ev + static_cast<std::size_t>(2 * eStride + eJx) * cap;
    const double* Jx3 = ev + static_cast<std::size_t>(3 * eStride + eJx) * cap;

    // 四遍必须分开：dk2 依赖**全部** dk1[c][j]（c 为状态分量），
    // 融合成一个 (i,j) 循环会读到尚未计算的 dk1 行。
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < NP; ++j) {
            const int o = (i * NP + j) * cap;
#pragma omp simd
            for (int b = 0; b < n; ++b) {
                double a = Jp0[o + b];
                for (int c = 0; c < 4; ++c)
                    a += Jx0[(i * 4 + c) * cap + b] * S[(c * NP + j) * cap + b];
                dk1[o + b] = a;
            }
        }
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < NP; ++j) {
            const int o = (i * NP + j) * cap;
#pragma omp simd
            for (int b = 0; b < n; ++b) {
                double a = Jp1[o + b];
                for (int c = 0; c < 4; ++c)
                    a += Jx1[(i * 4 + c) * cap + b] *
                         (S[(c * NP + j) * cap + b] + 0.5 * h * dk1[(c * NP + j) * cap + b]);
                dk2[o + b] = a;
            }
        }
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < NP; ++j) {
            const int o = (i * NP + j) * cap;
#pragma omp simd
            for (int b = 0; b < n; ++b) {
                double a = Jp2[o + b];
                for (int c = 0; c < 4; ++c)
                    a += Jx2[(i * 4 + c) * cap + b] *
                         (S[(c * NP + j) * cap + b] + 0.5 * h * dk2[(c * NP + j) * cap + b]);
                dk3[o + b] = a;
            }
        }
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < NP; ++j) {
            const int o = (i * NP + j) * cap;
#pragma omp simd
            for (int b = 0; b < n; ++b) {
                double a = Jp3[o + b];
                for (int c = 0; c < 4; ++c)
                    a += Jx3[(i * 4 + c) * cap + b] *
                         (S[(c * NP + j) * cap + b] + h * dk3[(c * NP + j) * cap + b]);
                dk4[o + b] = a;
            }
        }
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < NP; ++j) {
            const int o = (i * NP + j) * cap;
#pragma omp simd
            for (int b = 0; b < n; ++b)
                Sn[o + b] = S[o + b] + w1 * (dk1[o + b] + dk4[o + b]) +
                            w2 * (dk2[o + b] + dk3[o + b]);
        }
    }
    std::copy(Sn, Sn + static_cast<std::size_t>(4 * NP) * cap, S);
}

/// 一批样本在 [b0, b1) 上的完整计算（一个线程负责这一段）。
void runRange(const double* TCBS_RESTRICT params,
              const double* TCBS_RESTRICT x0,
              const double* TCBS_RESTRICT base,
              const double* TCBS_RESTRICT tau,
              const double* const* TCBS_RESTRICT tgt,
              const double* TCBS_RESTRICT weights,
              int B, int b0, int b1, std::size_t K, double dt, int refinement,
              int cap, double* TCBS_RESTRICT out_loss, double* TCBS_RESTRICT out_grad,
              Scratch& s) {
    const double h = dt / static_cast<double>(refinement);
    const double inv_k = 1.0 / static_cast<double>(K);

    for (int start = b0; start < b1; start += cap) {
        const int n = std::min(cap, b1 - start);
        const std::size_t off = static_cast<std::size_t>(start);

        const double* pq[NQ];
        for (int q = 0; q < NQ; ++q)
            pq[q] = params + static_cast<std::size_t>(q) * B + off;
        const double* px0[4];
        for (int i = 0; i < 4; ++i)
            px0[i] = x0 + static_cast<std::size_t>(i) * B + off;
        const double* pbase[3];
        for (int j = 0; j < 3; ++j)
            pbase[j] = base + static_cast<std::size_t>(j) * B + off;
        const double* pwt[4];
        for (int q = 0; q < 4; ++q)
            pwt[q] = weights + static_cast<std::size_t>(q) * B + off;
        const double* pt[4];
        for (int q = 0; q < 4; ++q)
            pt[q] = (tgt[q] != nullptr) ? (tgt[q] + off) : nullptr;

        // 初始化状态与灵敏度
        for (int i = 0; i < 4; ++i)
            std::copy(px0[i], px0[i] + n, s.X.begin() + static_cast<std::size_t>(i) * cap);
        std::fill(s.S.begin(),
                  s.S.begin() + static_cast<std::size_t>(4 * NP) * cap, 0.0);
        std::fill(s.loss.begin(), s.loss.begin() + n, 0.0);
        std::fill(s.grad.begin(),
                  s.grad.begin() + static_cast<std::size_t>(NP) * cap, 0.0);

        for (std::size_t k = 0; k < K; ++k) {
            const double* pTb = tau + (2 * k) * B + off;
            const double* pTs = tau + (2 * k + 1) * B + off;
            const double t_base = static_cast<double>(k) * dt;

            for (int sub = 0; sub < refinement; ++sub) {
                const double t0 = t_base + static_cast<double>(sub) * h;
                rk4StepSensBatch(pq, pbase[0], pbase[1], pbase[2], pTb, pTs, t0, h, n, cap, s);
            }

            // 步末基座量（损失在这些量上定义）
            fillBase(s.tc.data(), s.dtc.data(), pbase[0], pbase[1], pbase[2],
                     t_base + dt, n);
#pragma omp simd
            for (int b = 0; b < n; ++b) {
                const double tce = s.tc[b], dtce = s.dtc[b];
                const double theta_b = s.X[0 * cap + b], dtheta_b = s.X[1 * cap + b];
                const double theta_s = s.X[2 * cap + b], dtheta_s = s.X[3 * cap + b];
                const double r1 = (tce + theta_b) - (pt[0] ? pt[0][k * B + b] : 0.0);
                const double r2 = (tce + theta_b + theta_s) - (pt[1] ? pt[1][k * B + b] : 0.0);
                const double d1 = (dtce + dtheta_b) - (pt[2] ? pt[2][k * B + b] : 0.0);
                const double d2 = (dtce + dtheta_b + dtheta_s) -
                                  (pt[3] ? pt[3][k * B + b] : 0.0);
                const double wa = pwt[0][b], wb = pwt[1][b];
                const double wc = pwt[2][b], wd = pwt[3][b];
                s.loss[b] += 0.5 * inv_k * (wa * r1 * r1 + wb * r2 * r2 +
                                            wc * d1 * d1 + wd * d2 * d2);
                s.g[0 * cap + b] = inv_k * (wa * r1 + wb * r2);
                s.g[1 * cap + b] = inv_k * (wc * d1 + wd * d2);
                s.g[2 * cap + b] = inv_k * (wb * r2);
                s.g[3 * cap + b] = inv_k * (wd * d2);
            }
            for (int j = 0; j < NP; ++j) {
                double* TCBS_RESTRICT gj = s.grad.data() + static_cast<std::size_t>(j) * cap;
                const double* TCBS_RESTRICT S = s.S.data();
#pragma omp simd
                for (int b = 0; b < n; ++b) {
                    gj[b] += s.g[0 * cap + b] * S[(0 * NP + j) * cap + b] +
                             s.g[1 * cap + b] * S[(1 * NP + j) * cap + b] +
                             s.g[2 * cap + b] * S[(2 * NP + j) * cap + b] +
                             s.g[3 * cap + b] * S[(3 * NP + j) * cap + b];
                }
            }
        }

        for (int b = 0; b < n; ++b) out_loss[off + b] = s.loss[b];
        for (int j = 0; j < NP; ++j)
            std::copy(s.grad.begin() + static_cast<std::size_t>(j) * cap,
                      s.grad.begin() + static_cast<std::size_t>(j) * cap + n,
                      out_grad + static_cast<std::size_t>(j) * B + off);
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// 工作区
// ---------------------------------------------------------------------------
struct ParamGradientBatchWorkspace::Impl {
    std::vector<Scratch> scratch;
};

ParamGradientBatchWorkspace::~ParamGradientBatchWorkspace() { release(); }

void ParamGradientBatchWorkspace::release() {
    delete impl;
    impl = nullptr;
    refinement = 0;
    max_batch = 0;
    max_num_steps = 0;
    num_threads = 1;
    lanes = 16;
}

bool ParamGradientBatchWorkspace::resize(int refine, int batch, std::size_t steps,
                                         int threads, int lanes_) {
    if (refine < kRefinementMin || refine > kRefinementMax) return false;
    if (batch < 1 || steps < 1) return false;
    if (lanes_ < 1) return false;
    if (threads <= 0) {
        const unsigned hc = std::thread::hardware_concurrency();
        threads = (hc == 0) ? 1 : static_cast<int>(hc);
    }
    if (threads < 1) threads = 1;
    if (lanes_ > 512) lanes_ = 512;

    if (impl != nullptr && refinement == refine && num_threads == threads &&
        lanes == lanes_) {
        max_batch = std::max(max_batch, batch);
        max_num_steps = std::max(max_num_steps, steps);
        return true;
    }

    release();
    try {
        Impl* p = new Impl();
        p->scratch.resize(static_cast<std::size_t>(threads));
        for (auto& sc : p->scratch) sc.alloc(lanes_);
        impl = p;
    } catch (...) {
        release();
        return false;
    }
    refinement = refine;
    max_batch = batch;
    max_num_steps = steps;
    num_threads = threads;
    lanes = lanes_;
    return true;
}

// ---------------------------------------------------------------------------
// 批量计算
// ---------------------------------------------------------------------------
void computeParamGradientBatch(const double* params,
                               const double* x0,
                               const double* base,
                               const double* tau,
                               const double* target_psi_b,
                               const double* target_psi_s,
                               const double* target_dpsi_b,
                               const double* target_dpsi_s,
                               const double* weights,
                               int batch,
                               std::size_t num_steps,
                               double dt,
                               ParamGradientBatchWorkspace& ws,
                               double* out_loss,
                               double* out_grad) {
    if (!(dt > 0.0)) throw std::invalid_argument("dt must be > 0");
    if (batch < 1) throw std::invalid_argument("batch must be >= 1");
    if (num_steps < 1) throw std::invalid_argument("num_steps must be >= 1");
    if (params == nullptr || x0 == nullptr || base == nullptr || tau == nullptr ||
        weights == nullptr || out_loss == nullptr || out_grad == nullptr) {
        throw std::invalid_argument("null pointer");
    }
    if (ws.impl == nullptr) throw std::invalid_argument("workspace not created");
    if (batch > ws.max_batch || num_steps > ws.max_num_steps) {
        throw std::invalid_argument("batch / num_steps exceeds workspace capacity");
    }

    const double* tgt[4] = {target_psi_b, target_psi_s, target_dpsi_b, target_dpsi_s};
    const int nt = std::min(ws.num_threads, batch);
    const int cap = ws.lanes;
    auto& scratch = ws.impl->scratch;

    if (nt <= 1) {
        runRange(params, x0, base, tau, tgt, weights, batch, 0, batch, num_steps, dt,
                 ws.refinement, cap, out_loss, out_grad, scratch[0]);
        return;
    }

    std::vector<std::thread> pool;
    pool.reserve(static_cast<std::size_t>(nt));
    for (int t = 0; t < nt; ++t) {
        const int b0 = static_cast<int>((static_cast<long long>(batch) * t) / nt);
        const int b1 = static_cast<int>((static_cast<long long>(batch) * (t + 1)) / nt);
        pool.emplace_back([&, t, b0, b1]() {
            runRange(params, x0, base, tau, tgt, weights, batch, b0, b1, num_steps, dt,
                     ws.refinement, cap, out_loss, out_grad, scratch[t]);
        });
    }
    for (auto& th : pool) th.join();
}

}  // namespace dm
}  // namespace tcbs
