// 轨迹梯度模块的自检：解析梯度（离散伴随）vs 中心差分；
// 以及参数梯度模块（前向灵敏度）的解析 dL/dp vs 中心差分。
//
// 直接运行即可，全部通过返回 0，否则返回 1。
// 构建：cmake 里以 tcbss_trajectory_test 目标生成到 build/ 根目录。

#include <cmath>
#include <cstdio>
#include <vector>

#include "param_gradient.hpp"
#include "params.hpp"
#include "state.hpp"
#include "trajectory.hpp"

namespace {

struct Config {
    const char* name;
    double theta_c0;
    double dtheta_c;
    double ddtheta_c;
};

/// 按参数梯度模块的编号（见 param_gradient.hpp）给单个参数加增量。
/// 重力 gx/gy 是已知输入、不在辨识集内，故没有对应分支。
tcbss::Params withParamDelta(const tcbss::Params& p, int idx, double d) {
    tcbss::Params q = p;
    switch (idx) {
        case 0: q.mb += d; break;
        case 1: q.Ib += d; break;
        case 2: q.Pbx += d; break;
        case 3: q.Pby += d; break;
        case 4: q.ms += d; break;
        case 5: q.Is += d; break;
        case 6: q.Psx += d; break;
        case 7: q.Psy += d; break;
        case 8: q.Dx += d; break;
        case 9: q.Dy += d; break;
        case 10: q.fbc += d; break;
        case 11: q.fbv += d; break;
        case 12: q.fsc += d; break;
        case 13: q.fsv += d; break;
        default: break;
    }
    return q;
}

double maxAbsDiff(const std::vector<double>& a, const std::vector<double>& b) {
    double m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = std::fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

/// 简单可复现的伪随机数（避免依赖 <random> 的实现差异）。
struct Rng {
    unsigned long long s = 0x2545F4914F6CDD1DULL;
    double next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return static_cast<double>(s % 2000001) / 1000000.0 - 1.0;  // [-1, 1]
    }
};

}  // namespace

int main() {
    using namespace tcbss;

    // 与 python/pendulum_sim.py 相同的物理参数
    const Params p(/*mb*/ 1.5, /*Ib*/ 0.030, /*Pbx*/ 0.0, /*Pby*/ 0.180,
                   /*ms*/ 0.40, /*Is*/ 0.020, /*Psx*/ 0.0, /*Psy*/ 0.1,
                   /*Dx*/ 0.0, /*Dy*/ 0.3, /*gx*/ 0.0, /*gy*/ -9.81,
                   /*fbc*/ 0.020, /*fbv*/ 0.050, /*fsc*/ 0.010, /*fsv*/ 0.020,
                   /*lambda*/ 100.0);

    const double dt = 1.0e-3;
    const int kRefine = 4;   // 运行期参数
    const std::size_t K = 12;
    const State x0{0.4, 0.8, 1.2, -0.5};

    const Config configs[] = {
        {"baseline   ", -0.4, 0.25, 0.8},
        {"base-static", 0.2, 0.0, 0.0},
        {"base-accel ", 0.5, -0.3, 2.0},
    };

    Rng rng;
    TrajectoryWorkspace ws;
    int failures = 0;

    std::printf("== 轨迹梯度自检：解析 vs 中心差分（K=%zu, dt=%g, refinement=%d）==\n",
                K, dt, kRefine);

    for (const Config& cfg : configs) {
        // 随机力矩序列与目标序列
        std::vector<double> tau(2 * K);
        std::vector<double> tgt_psi_b(K), tgt_psi_s(K), tgt_dpsi_b(K), tgt_dpsi_s(K);
        for (std::size_t i = 0; i < 2 * K; ++i) tau[i] = 1.5 * rng.next();
        for (std::size_t i = 0; i < K; ++i) {
            tgt_psi_b[i] = rng.next();
            tgt_psi_s[i] = rng.next();
            tgt_dpsi_b[i] = rng.next();
            tgt_dpsi_s[i] = rng.next();
        }

        LossSpec spec;
        spec.w_psi_b = 1.2;
        spec.w_psi_s = 0.6;
        spec.w_dpsi_b = 0.8;
        spec.w_dpsi_s = 0.4;
        spec.w_tau_b = 0.05;
        spec.w_tau_s = 0.03;
        spec.target_psi_b = tgt_psi_b.data();
        spec.target_psi_s = tgt_psi_s.data();
        spec.target_dpsi_b = tgt_dpsi_b.data();
        spec.target_dpsi_s = tgt_dpsi_s.data();

        std::vector<double> grad(2 * K, 0.0);
        const double loss = simulateAndGradient(p, cfg.theta_c0, cfg.dtheta_c, cfg.ddtheta_c,
                                                dt, kRefine, K, tau.data(), spec, x0, ws,
                                                grad.data(), nullptr,
                                                /*with_step_jacobians=*/true);

        // 中心差分
        const double eps = 1.0e-7;
        std::vector<double> fd(2 * K, 0.0);
        for (std::size_t k = 0; k < 2 * K; ++k) {
            const double save = tau[k];
            tau[k] = save + eps;
            const double lp = computeTrajectoryLoss(p, cfg.theta_c0, cfg.dtheta_c,
                                                    cfg.ddtheta_c, dt, kRefine, K, tau.data(), 0.0,
                                                    0.0, spec, x0, nullptr, nullptr, nullptr,
                                                    nullptr);
            tau[k] = save - eps;
            const double lm = computeTrajectoryLoss(p, cfg.theta_c0, cfg.dtheta_c,
                                                    cfg.ddtheta_c, dt, kRefine, K, tau.data(), 0.0,
                                                    0.0, spec, x0, nullptr, nullptr, nullptr,
                                                    nullptr);
            tau[k] = save;
            fd[k] = (lp - lm) / (2.0 * eps);
        }

        double scale = 0.0;
        for (double v : fd) scale = std::fmax(scale, std::fabs(v));
        const double err = maxAbsDiff(grad, fd);
        const double rel = (scale > 0.0) ? err / scale : err;

        // 判据说明：残差来自中心差分本身（O(eps^2) 截断 + tanh 饱和平台边界上的舍入），
        // 而非解析梯度。经验上相对残差在 1e-3 量级即视为通过；真正的解析错误（例如
        // 反向传递漏项）会让残差比这大 2~3 个数量级。
        const double tol = 5.0e-3;
        std::printf("  %s  L=%10.6f   max|解析-FD|=%.3e   相对=%.2e   %s\n", cfg.name, loss,
                    err, rel, (rel < tol) ? "OK" : "FAIL");
        if (!(rel < tol)) ++failures;
    }

    // 与逐步 Simulator 的前向一致性（同一 dt/refinement 下应完全一致）
    {
        std::vector<double> tau(2 * K);
        for (std::size_t i = 0; i < 2 * K; ++i) tau[i] = 1.5 * rng.next();
        LossSpec spec;  // 全零权重，只取序列
        std::vector<double> psi_b(K), psi_s(K), dpb(K), dps(K);
        computeTrajectoryLoss(p, 0.3, 0.2, -1.5, dt, kRefine, K, tau.data(), 0.0, 0.0, spec, x0,
                              psi_b.data(),
                              psi_s.data(), dpb.data(), dps.data());
        std::printf("\n== 前向一致性 ==\n");
        std::printf("  轨迹模块 psi_b[0..2] = %.9f %.9f %.9f\n", psi_b[0], psi_b[1], psi_b[2]);
        std::printf("  （应与 Simulator::step 逐步推进的 psi 相同，见 trajectory.hpp 的约定）\n");
    }

    // =====================================================================
    // 参数梯度自检：解析 dL/dp vs 中心差分
    //
    // K=1（单主步）是子步灵敏度公式最灵敏的探针。两类曾经的缺陷都会在这里
    // 暴露为 ~1e-1 的相对残差，而正确的实现应落在 1e-6 以内（只剩差分舍入）：
    //   * 把整步传播子写成 A4 = I + h*Jx3*A3，漏掉 4 阶段加权和
    //     W = I + (h/6)(Jx1 + 2 Jx2 A2 + 2 Jx3 A3 + Jx4 A4)；
    //   * dk_i 的链式里漏掉阶段耦合项 Jx_i*(h/2)*dk_{i-1}。
    // =====================================================================
    std::printf("\n== 参数梯度自检：解析 dL/dp vs 中心差分 ==\n");
    {
        // 用非退化的参数点（各分量都不为 0），否则缺陷可能被零梯度掩盖
        const Params pgen(/*mb*/ 1.5, /*Ib*/ 0.030, /*Pbx*/ 0.12, /*Pby*/ 0.18,
                          /*ms*/ 0.40, /*Is*/ 0.020, /*Psx*/ 0.08, /*Psy*/ 0.1,
                          /*Dx*/ 0.05, /*Dy*/ 0.3, /*gx*/ 0.3, /*gy*/ -9.81,
                          /*fbc*/ 0.020, /*fbv*/ 0.050, /*fsc*/ 0.010, /*fsv*/ 0.020,
                          /*lambda*/ 100.0);

        ParamGradientWorkspace pws;
        struct PCase {
            std::size_t K;
            int refine;
        };
        const PCase pcases[] = {{1, 1}, {1, 4}, {4, 4}, {12, 4}};

        for (const PCase& pc : pcases) {
            const std::size_t KK = pc.K;
            std::vector<double> tau(2 * KK);
            std::vector<double> tb(KK), ts(KK), vb(KK), vs(KK);
            for (std::size_t i = 0; i < 2 * KK; ++i) tau[i] = 1.5 * rng.next();
            for (std::size_t i = 0; i < KK; ++i) {
                tb[i] = rng.next();
                ts[i] = rng.next();
                vb[i] = rng.next();
                vs[i] = rng.next();
            }

            ParamLossSpec spec;
            spec.w_psi_b = 1.0;
            spec.w_psi_s = 1.0;
            spec.w_dpsi_b = 1.0;
            spec.w_dpsi_s = 1.0;
            spec.target_psi_b = tb.data();
            spec.target_psi_s = ts.data();
            spec.target_dpsi_b = vb.data();
            spec.target_dpsi_s = vs.data();

            std::vector<double> grad(kParamGradientCount, 0.0);
            const double loss =
                computeParamGradient(pgen, 0.3, 0.2, 0.5, dt, pc.refine, KK, tau.data(), spec, x0,
                                     pws, grad.data(), nullptr);

            const double eps = 1.0e-6;
            double scale = 0.0, err = 0.0;
            for (int i = 0; i < kParamGradientCount; ++i) {
                const double lp = computeParamGradientLoss(withParamDelta(pgen, i, +eps), 0.3, 0.2,
                                                           0.5, dt, pc.refine, KK, tau.data(), spec,
                                                           x0, nullptr, nullptr, nullptr, nullptr);
                const double lm = computeParamGradientLoss(withParamDelta(pgen, i, -eps), 0.3, 0.2,
                                                           0.5, dt, pc.refine, KK, tau.data(), spec,
                                                           x0, nullptr, nullptr, nullptr, nullptr);
                const double fd = (lp - lm) / (2.0 * eps);
                err = std::fmax(err, std::fabs(grad[static_cast<std::size_t>(i)] - fd));
                scale = std::fmax(scale, std::fabs(fd));
            }
            const double rel = (scale > 0.0) ? err / scale : err;
            // K=1 只有子步灵敏度误差，没有长程累积，判据可以很紧
            const double tol = (KK == 1) ? 1.0e-5 : 5.0e-4;
            std::printf("  K=%2zu refine=%d  L=%10.6f  max|解析-FD|=%.3e  相对=%.2e  %s\n", KK,
                        pc.refine, loss, err, rel, (rel < tol) ? "OK" : "FAIL");
            if (!(rel < tol)) ++failures;
        }
    }

    std::printf("\n%s\n", failures == 0 ? "全部通过。" : "存在失败项。");
    return failures == 0 ? 0 : 1;
}
