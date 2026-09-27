// 离线验证程序（放在 build/ 下，不参与 CMake 构建，验证后删除）
//
// 校验对象：src/ 中解析实现的动力学 + RK4 仿真器。
// 独立 oracle：由正向运动学数值导出完整 3x3 质量矩阵、势函数梯度、Christoffel 符号。
//
// [1] 与 dynamic_model 参考实现逐点对拍
// [2] RK4 细化倍数收敛性
// [3] 单大步(高细化) vs 多小步(theta_c 解析推进)
// [4] 解析 3x3 质量矩阵（含 theta_c 耦合列）vs 运动学数值质量矩阵
// [5] 解析重力项 vs 势函数数值梯度
// [6] 解析科氏项 vs 数值 Christoffel 符号
// [7] 端到端：含 theta_c 运动的加速度 vs 数值 oracle 全模型
// [8] 能量守恒 + 自由演化有界性

#include <cmath>
#include <cstdio>
#include <random>

#include "dynamics.hpp"
#include "rk4.hpp"
#include "simulator.hpp"
#include "state.hpp"

// 参考模型（已修正）：仅用于对拍
#include "../dynamic_model/deepseek_cpp_20260927_1a755b.cpp"

namespace {

struct V2 { double x, y; };

V2 rot(V2 v, double a) {
    const double c = std::cos(a), s = std::sin(a);
    return V2{c * v.x - s * v.y, s * v.x + c * v.y};
}

// ---------------- 独立数值 oracle ----------------
class NumModel {
public:
    explicit NumModel(const tcbss::Params& pp) : p(pp) {}

    tcbss::Params p;

    V2 comB(double thc, double tb, double ts) const {
        return rot(V2{p.Pbx, p.Pby}, thc + tb);
    }
    V2 comS(double thc, double tb, double ts) const {
        const V2 j = rot(V2{p.Dx, p.Dy}, thc + tb);
        const V2 r = rot(V2{p.Psx, p.Psy}, thc + tb + ts);
        return V2{j.x + r.x, j.y + r.y};
    }

    double potential(double thc, double tb, double ts) const {
        const V2 b = comB(thc, tb, ts);
        const V2 s = comS(thc, tb, ts);
        return -(p.gx * (p.mb * b.x + p.ms * s.x) + p.gy * (p.mb * b.y + p.ms * s.y));
    }

    // q = (theta_c, theta_b, theta_s)
    // 数值版：位置对 q 用中心差分（用于验证质量矩阵本身）
    void mass3(double thc, double tb, double ts, double M[3][3]) const {
        const double e = 1e-6;
        V2 jb[3], js[3];
        for (int k = 0; k < 3; ++k) {
            double a1 = thc, b1 = tb, c1 = ts, a0 = thc, b0 = tb, c0 = ts;
            if (k == 0) { a1 += e; a0 -= e; }
            else if (k == 1) { b1 += e; b0 -= e; }
            else { c1 += e; c0 -= e; }
            const V2 pb1 = comB(a1, b1, c1), pb0 = comB(a0, b0, c0);
            const V2 ps1 = comS(a1, b1, c1), ps0 = comS(a0, b0, c0);
            jb[k] = V2{(pb1.x - pb0.x) / (2 * e), (pb1.y - pb0.y) / (2 * e)};
            js[k] = V2{(ps1.x - ps0.x) / (2 * e), (ps1.y - ps0.y) / (2 * e)};
        }
        assemble(jb, js, M);
    }

    // 解析版：速度 = R(psi) * J * r，J = [[0,-1],[1,0]]（用于后续求 dM/dq）
    void mass3Exact(double thc, double tb, double ts, double M[3][3]) const {
        const double psib = thc + tb;
        const double psis = thc + tb + ts;
        const V2 Jpb = rot(V2{-p.Pby, p.Pbx}, psib);
        const V2 Jd = rot(V2{-p.Dy, p.Dx}, psib);
        const V2 JPs = rot(V2{-p.Psy, p.Psx}, psis);

        V2 jb[3], js[3];
        jb[0] = Jpb; jb[1] = Jpb; jb[2] = V2{0.0, 0.0};
        js[0] = V2{Jd.x + JPs.x, Jd.y + JPs.y};
        js[1] = js[0];
        js[2] = JPs;
        assemble(jb, js, M);
    }

    void assemble(const V2 jb[3], const V2 js[3], double M[3][3]) const {
        // 角度对 q 的偏导：psi_b = thc+tb, psi_s = thc+tb+ts
        const double dpsi_b[3] = {1.0, 1.0, 0.0};
        const double dpsi_s[3] = {1.0, 1.0, 1.0};
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                M[i][j] = p.mb * (jb[i].x * jb[j].x + jb[i].y * jb[j].y) +
                          p.Ib * dpsi_b[i] * dpsi_b[j] +
                          p.ms * (js[i].x * js[j].x + js[i].y * js[j].y) +
                          p.Is * dpsi_s[i] * dpsi_s[j];
            }
        }
    }

    // 完整 3 自由度模型：M qdd + C = Q - G（Q 只在 tb/ts 上有驱动力矩）
    void accelerations(double Tb, double Ts, double thc, double dthc, double ddthc,
                       double tb, double dtb, double ts, double dts,
                       double& ddtb, double& ddts) const {
        const double e = 1e-5;
        double M[3][3];
        mass3Exact(thc, tb, ts, M);

        // dM/dq_k（M 只依赖 tb, ts；k=0 时为零）
        double dM[3][3][3];
        for (int k = 1; k < 3; ++k) {
            double Mp[3][3], Mm[3][3];
            if (k == 1) { mass3Exact(thc, tb + e, ts, Mp); mass3Exact(thc, tb - e, ts, Mm); }
            else { mass3Exact(thc, tb, ts + e, Mp); mass3Exact(thc, tb, ts - e, Mm); }
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) dM[i][j][k] = (Mp[i][j] - Mm[i][j]) / (2 * e);
        }
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) dM[i][j][0] = 0.0;

        double G[3];
        G[0] = (potential(thc + e, tb, ts) - potential(thc - e, tb, ts)) / (2 * e);
        G[1] = (potential(thc, tb + e, ts) - potential(thc, tb - e, ts)) / (2 * e);
        G[2] = (potential(thc, tb, ts + e) - potential(thc, tb, ts - e)) / (2 * e);

        const double qd[3] = {dthc, dtb, dts};
        double C[3] = {0.0, 0.0, 0.0};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k)
                    C[i] += 0.5 * (dM[i][j][k] + dM[i][k][j] - dM[j][k][i]) * qd[j] * qd[k];

        // 摩擦与驱动力（与 src 中一致）
        const double Qb = Tb - p.fbv * dtb - p.fbc * std::tanh(p.lambda * dtb);
        const double Qs = Ts - p.fsv * dts - p.fsc * std::tanh(p.lambda * dts);

        // 两个广义坐标方程：M_rr qdd_r = Q_r - G_r - M_rc*ddthc - C_r
        const double r0 = Qb - G[1] - M[1][0] * ddthc - C[1];
        const double r1 = Qs - G[2] - M[2][0] * ddthc - C[2];
        const double det = M[1][1] * M[2][2] - M[1][2] * M[2][1];
        ddtb = (r0 * M[2][2] - r1 * M[1][2]) / det;
        ddts = (M[1][1] * r1 - M[1][2] * r0) / det;
    }

    double fullEnergy(double thc, double dthc, double tb, double dtb, double ts, double dts) const {
        double M[3][3];
        mass3Exact(thc, tb, ts, M);
        const double qd[3] = {dthc, dtb, dts};
        double ke = 0.0;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) ke += 0.5 * qd[i] * M[i][j] * qd[j];
        return ke + potential(thc, tb, ts);
    }
};

tcbss::Params makeMine() {
    return tcbss::Params(
        /*mb*/ 1.3, /*Ib*/ 0.045, /*Pbx*/ 0.17, /*Pby*/ -0.03,
        /*ms*/ 0.6, /*Is*/ 0.021, /*Psx*/ 0.12, /*Psy*/ 0.04,
        /*Dx*/ 0.31, /*Dy*/ -0.05, /*gx*/ 0.4, /*gy*/ -9.81,
        /*fbc*/ 0.07, /*fbv*/ 0.11, /*fsc*/ 0.05, /*fsv*/ 0.09, /*lambda*/ 100.0);
}

tcbss::Params makeFrictionless() {
    return tcbss::Params(1.3, 0.045, 0.17, -0.03,
                         0.6, 0.021, 0.12, 0.04,
                         0.31, -0.05, 0.4, -9.81,
                         0.0, 0.0, 0.0, 0.0, 100.0);
}

::Params makeRef(const tcbss::Params& p, double Tb, double Ts) {
    ::Params r;
    r.mb = p.mb;   r.Ib = p.Ib;   r.Pbx = p.Pbx; r.Pby = p.Pby;
    r.ms = p.ms;   r.Is = p.Is;   r.Psx = p.Psx; r.Psy = p.Psy;
    r.Dx = p.Dx;   r.Dy = p.Dy;   r.gx = p.gx;   r.gy = p.gy;
    r.Tb = Tb;     r.Ts = Ts;
    r.fbc = p.fbc; r.fbv = p.fbv; r.fsc = p.fsc; r.fsv = p.fsv; r.lambda = p.lambda;
    return r;
}

double stateErr(const tcbss::State& a, const tcbss::State& b) {
    return std::fmax(std::fmax(std::fabs(a.theta_b - b.theta_b), std::fabs(a.dtheta_b - b.dtheta_b)),
                     std::fmax(std::fabs(a.theta_s - b.theta_s), std::fabs(a.dtheta_s - b.dtheta_s)));
}

// ---------------- [1] ----------------
int testReferenceMatch() {
    const tcbss::Params mine = makeMine();
    std::mt19937_64 rng(20260927ULL);
    std::uniform_real_distribution<double> ang(-3.2, 3.2), vel(-6.0, 6.0), acc(-25.0, 25.0), tor(-8.0, 8.0);

    double max_err = 0.0;
    for (int i = 0; i < 300000; ++i) {
        const double thc = ang(rng), dthc = vel(rng), ddthc = acc(rng);
        const tcbss::State s{ang(rng), vel(rng), ang(rng), vel(rng)};
        const double Tb = tor(rng), Ts = tor(rng);
        double am = 0.0, bm = 0.0, ar = 0.0, br = 0.0;
        tcbss::computeAccelerations(mine, Tb, Ts, thc, dthc, ddthc, s, am, bm);
        ::computeAccelerations(makeRef(mine, Tb, Ts), thc, dthc, ddthc,
                               s.theta_b, s.dtheta_b, s.theta_s, s.dtheta_s, ar, br);
        max_err = std::fmax(max_err, std::fabs(am - ar));
        max_err = std::fmax(max_err, std::fabs(bm - br));
    }
    std::printf("[1] 与修正后参考实现对拍: 300000 组, 最大误差 = %.3e  -> %s\n",
                max_err, (max_err < 1e-12) ? "PASS" : "FAIL");
    return (max_err < 1e-12) ? 0 : 1;
}

// ---------------- [2] ----------------
int testRefinementConvergence() {
    const tcbss::Params p = makeMine();
    const NumModel num(p);
    const double T = 0.5, dt = 0.01;
    const double thc0 = 0.3, dthc = 1.1, ddthc = -2.0;
    const double Tb = 1.7, Ts = -0.9;
    const tcbss::State y0{-1.2, 0.8, 0.6, -1.4};

    tcbss::State y_ref = y0;
    {
        const double h = 1.0e-6;
        const long n = std::lround(T / h);
        for (long i = 0; i < n; ++i) {
            const double tau = i * h;
            const auto f = [&](double t, const tcbss::State& s) -> tcbss::StateDerivative {
                const double d = t - tau;
                const double tc = thc0 + dthc * (tau + d) + 0.5 * ddthc * (tau + d) * (tau + d);
                const double dtc = dthc + ddthc * (tau + d);
                double ab = 0.0, as = 0.0;
                num.accelerations(Tb, Ts, tc, dtc, ddthc, s.theta_b, s.dtheta_b, s.theta_s, s.dtheta_s, ab, as);
                return tcbss::StateDerivative{s.dtheta_b, ab, s.dtheta_s, as};
            };
            y_ref = tcbss::rk4Step(f, y_ref, tau, h);
        }
    }

    bool ok = true;
    std::printf("[2] 细化倍数收敛性 (dt=%.3g, T=%.3g, 基准 h=1e-6 RK4 + 数值 oracle):\n", dt, T);
    for (int refinement : {1, 2, 4, 8, 16, 32}) {
        tcbss::Simulator sim(p, dt, refinement);
        sim.setState(y0);
        const long n = std::lround(T / dt);
        for (long i = 0; i < n; ++i) {
            const double tau = i * dt;
            sim.step(Tb, Ts, thc0 + dthc * tau + 0.5 * ddthc * tau * tau, dthc + ddthc * tau, ddthc);
        }
        const double err = stateErr(sim.state(), y_ref);
        std::printf("    refinement = %2d (h=%.3e): 与基准差 = %.3e\n", refinement, dt / refinement, err);
        if (refinement >= 32 && err > 1e-6) ok = false;
    }
    std::printf("    -> %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// ---------------- [3] ----------------
int testSingleVsMultiStep() {
    const tcbss::Params p = makeMine();
    const double thc0 = -0.4, dthc0 = 0.9, ddthc = -3.0;
    const double Tb = 0.8, Ts = 1.3;
    const tcbss::State y0{0.9, -0.5, -1.1, 0.7};
    const double T = 0.05;

    tcbss::Simulator a(p, T, 400);
    a.setState(y0);
    const tcbss::State ya = a.step(Tb, Ts, thc0, dthc0, ddthc);

    const int n = 250;
    const double dt = T / n;
    tcbss::Simulator b(p, dt, 1);
    b.setState(y0);
    for (int i = 0; i < n; ++i) {
        const double t = i * dt;
        b.step(Tb, Ts, thc0 + dthc0 * t + 0.5 * ddthc * t * t, dthc0 + ddthc * t, ddthc);
    }
    const double err = stateErr(ya, b.state());
    std::printf("[3] 单大步(细化400, h=%.3e) vs 250小步(h=%.3e): 最大差 = %.3e  -> %s\n",
                T / 400.0, dt, err, (err < 1e-6) ? "PASS" : "FAIL");
    return (err < 1e-6) ? 0 : 1;
}

// ---------------- [4][5][6] 解析 vs 数值 oracle ----------------
int testAnalyticVsNumeric() {
    const tcbss::Params p = makeMine();
    const NumModel num(p);

    std::mt19937_64 rng(11ULL);
    std::uniform_real_distribution<double> ang(-3.1, 3.1), vel(-7.0, 7.0), acc(-20.0, 20.0), tor(-6.0, 6.0);

    double maxM = 0.0, maxG = 0.0, maxC = 0.0, maxA = 0.0;
    for (int it = 0; it < 4000; ++it) {
        const double thc = ang(rng), tb = ang(rng), ts = ang(rng);
        const double dthc = vel(rng), dtb = vel(rng), dts = vel(rng), ddthc = acc(rng);
        const double Tb = tor(rng), Ts = tor(rng);

        // [4] 质量矩阵
        double M[3][3];
        num.mass3(thc, tb, ts, M);
        const tcbss::Kinematics kin = tcbss::computeKinematics(p, thc, tb, ts);
        const tcbss::MassMatrix ma = tcbss::computeMassMatrix(p, kin);
        maxM = std::fmax(maxM, std::fabs(M[0][0] - ma.M11));
        maxM = std::fmax(maxM, std::fabs(M[0][1] - ma.M11));
        maxM = std::fmax(maxM, std::fabs(M[1][1] - ma.M11));
        maxM = std::fmax(maxM, std::fabs(M[0][2] - ma.M12));
        maxM = std::fmax(maxM, std::fabs(M[1][2] - ma.M12));
        maxM = std::fmax(maxM, std::fabs(M[2][2] - ma.M22));

        // [5] 重力项
        double G1 = 0.0, G2 = 0.0;
        tcbss::computeGravity(p, kin, G1, G2);
        const double e = 1e-6;
        const double G1n = (num.potential(thc, tb + e, ts) - num.potential(thc, tb - e, ts)) / (2 * e);
        const double G2n = (num.potential(thc, tb, ts + e) - num.potential(thc, tb, ts - e)) / (2 * e);
        maxG = std::fmax(maxG, std::fabs(G1 - G1n));
        maxG = std::fmax(maxG, std::fabs(G2 - G2n));

        // [6] 科氏项（数值 Christoffel）
        const double ee = 1e-5;
        double dM[3][3][3] = {};
        for (int k = 1; k < 3; ++k) {
            double Mp[3][3], Mm[3][3];
            if (k == 1) { num.mass3Exact(thc, tb + ee, ts, Mp); num.mass3Exact(thc, tb - ee, ts, Mm); }
            else { num.mass3Exact(thc, tb, ts + ee, Mp); num.mass3Exact(thc, tb, ts - ee, Mm); }
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) dM[i][j][k] = (Mp[i][j] - Mm[i][j]) / (2 * ee);
        }
        const double qd[3] = {dthc, dtb, dts};
        double Cn[3] = {0.0, 0.0, 0.0};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k)
                    Cn[i] += 0.5 * (dM[i][j][k] + dM[i][k][j] - dM[j][k][i]) * qd[j] * qd[k];
        double C1 = 0.0, C2 = 0.0;
        tcbss::computeCoriolis(p, kin, dthc + dtb, dts, C1, C2);
        maxC = std::fmax(maxC, std::fabs(C1 - Cn[1]));
        maxC = std::fmax(maxC, std::fabs(C2 - Cn[2]));

        // [7] 端到端加速度
        const tcbss::State s{tb, dtb, ts, dts};
        double ab = 0.0, as = 0.0, abn = 0.0, asn = 0.0;
        tcbss::computeAccelerations(p, Tb, Ts, thc, dthc, ddthc, s, ab, as);
        num.accelerations(Tb, Ts, thc, dthc, ddthc, tb, dtb, ts, dts, abn, asn);
        maxA = std::fmax(maxA, std::fabs(ab - abn));
        maxA = std::fmax(maxA, std::fabs(as - asn));
    }

    const bool ok = (maxM < 1e-8) && (maxG < 1e-8) && (maxC < 1e-7) && (maxA < 1e-7);
    std::printf("[4] 质量矩阵(含 theta_c 耦合列) 解析 vs 数值: 最大差 = %.3e\n", maxM);
    std::printf("[5] 重力项 解析 vs 数值势梯度:            最大差 = %.3e\n", maxG);
    std::printf("[6] 科氏项 解析 vs 数值 Christoffel:        最大差 = %.3e\n", maxC);
    std::printf("[7] 端到端加速度(含 theta_c 运动) vs oracle: 最大差 = %.3e  -> %s\n",
                maxA, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// ---------------- [8] 能量守恒 + 有界性 ----------------
int testEnergy() {
    const tcbss::Params p = makeFrictionless();
    const NumModel num(p);
    const double theta_c = 0.0;

    tcbss::Simulator sim(p, 1e-3, 8);
    tcbss::State s{-M_PI / 2.0 + 0.15, 0.0, 0.0, 0.0};
    sim.setState(s);
    const double E0 = num.fullEnergy(theta_c, 0.0, s.theta_b, s.dtheta_b, s.theta_s, s.dtheta_s);
    double max_dE = 0.0, max_rate = 0.0;
    for (int i = 0; i < 20000; ++i) {
        s = sim.step(0.0, 0.0, theta_c, 0.0, 0.0);
        const double E = num.fullEnergy(theta_c, 0.0, s.theta_b, s.dtheta_b, s.theta_s, s.dtheta_s);
        max_dE = std::fmax(max_dE, std::fabs(E - E0));
        max_rate = std::fmax(max_rate, std::fmax(std::fabs(s.dtheta_b), std::fabs(s.dtheta_s)));
    }
    const bool ok = (max_dE < 1e-3) && (max_rate < 6.0);
    std::printf("[8] 零力矩零摩擦自由演化 20s: |dtheta|max=%.4f, 能量最大漂移=%.3e (E0=%.6f)  -> %s\n",
                max_rate, max_dE, E0, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

}  // namespace

int main() {
    int failures = 0;
    failures += testReferenceMatch();
    failures += testRefinementConvergence();
    failures += testSingleVsMultiStep();
    failures += testAnalyticVsNumeric();
    failures += testEnergy();
    std::printf("\n失败项: %d\n", failures);
    return failures == 0 ? 0 : 1;
}
