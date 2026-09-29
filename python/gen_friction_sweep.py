"""匀速旋转摩擦辨识采集：让关节 b 以多个恒定角速度旋转，测稳态平均控制力矩。

为什么单独做这个实验
    主数据集为了满足 θ_s ≤ ±30°、两关节速度 ≤3 rad/s 等机械/电控约束，运动被压得
    很温和；用"含噪速度差分求加速度"的回归去解摩擦时，条件数极差（rank 12/17、
    cond ~1e18），摩擦系数根本解不出来（实测 fbc 会解成负值）。而摩擦本身是可以
    用一个**条件数极好**的独立实验直接测出来的。

原理（与库内模型严格对应）
    θ_s 被控制器按住（θ̇_s≈0、θ̈_s≈0），θ_b 以恒定 ω 旋转（θ̈_b=0），基座静止
    （ddθ_c=0）。此时两个广义坐标的运动方程都退化成力矩平衡：

        Eq_b:  M11·θ̈_b + M12·θ̈_s + C1 + G1 + M11·ddθ_c = Qb
        稳态下左边只剩 G1，而 Qb = Tb − fbv·θ̇_b − fbc·tanh(λ·θ̇_b)，于是
            Tb = G1(ψ_b) + fbv·ω + fbc·tanh(λ·ω)
        其中 G1 = gb_sin·sinψ_b + gb_cos·cosψ_b + gs_sin·sinψ_s + gs_cos·cosψ_s，
        gb_* / gs_* 由**已经辨识出来的动力学参数**给出。

    把 G1 用测得的 sin/cos 均值精确减掉后，剩下的就是二维线性问题：
        ⟨Tb⟩ − ⟨G1⟩ = fbv·⟨ω⟩ + fbc·⟨tanh(λω)⟩
    列向量只有 ω 与 tanh(λω) 两列，条件数随 ω 跨越几个数量级而变化 → 可同时分开
    库仑项与黏滞项。

输出 data/friction/sweep.npz：每条实验一行（ω、平均力矩、G1 所需的三角量均值等）。

用法::

    python3 python/gen_friction_sweep.py            # 默认速度表 × 若干重力
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from tcbss import Params, Simulator, State  # noqa: E402
import identify_params as ip  # noqa: E402
import generate_id_dataset as g  # noqa: E402

# 速度表：正负都有、跨越约 3 个数量级，覆盖 tanh(λω) 的线性段(ω≲0.03)与饱和段
OMEGA_LIST = [-3.0, -1.0, -0.3, -0.1, -0.05, -0.02, -0.01, -0.005,
              0.005, 0.01, 0.02, 0.05, 0.1, 0.3, 1.0, 3.0]

N_SETTLE = 300          # 稳定段步数（3 s）
N_MEAS = 300            # 测量段步数（3 s）
KPV = 2.0               # 关节 b 速度环 P [N*m/(rad/s)]
KIV = 1.5               # 关节 b 速度环 I [N*m/(rad*s)]
KPS_HS = 30.0           # 关节 s 位置环 P（把 s 牢牢按住）
KDS_HS = 2.0            # 关节 s 位置环 D
INT_CLAMP = 5.0         # 积分限幅（等效力矩上限）


def run_one(P: Params, omega_ref: float, theta_b0: float,
            rng: np.random.Generator) -> dict | None:
    """一次匀速旋转实验，返回测量段的统计量；失败（未收敛/超限位）返回 None。"""
    n = N_SETTLE + N_MEAS
    Tb = np.zeros(n)
    ts_hist = np.zeros(n)
    thb = np.zeros(n)
    dthb = np.zeros(n)
    dths = np.zeros(n)
    tsc = np.zeros(n)
    integ = 0.0
    with Simulator(P, g.DT, g.REFINEMENT) as sim:
        sim.set_state(State(theta_b=theta_b0, dtheta_b=0.0, theta_s=0.0, dtheta_s=0.0))
        for k in range(n):
            st = sim.state
            e = omega_ref - st.dtheta_b
            integ = float(np.clip(integ + e * g.DT, -INT_CLAMP, INT_CLAMP))
            ub = KPV * e + KIV * integ
            us = -KPS_HS * st.theta_s - KDS_HS * st.dtheta_s
            ub = float(np.clip(ub, -g.TAU_B_MAX, g.TAU_B_MAX))
            us = float(np.clip(us, -g.TAU_S_MAX, g.TAU_S_MAX))
            Tb[k] = ub
            ts_hist[k] = us
            thb[k] = st.theta_b
            dthb[k] = st.dtheta_b
            dths[k] = st.dtheta_s
            tsc[k] = st.theta_s
            sim.step(ub, us, 0.0, 0.0, 0.0)   # 基座静止

    m = slice(N_SETTLE, n)                      # 只用测量段
    omega = float(dthb[m].mean())
    # 稳态判据：速度波动要小；θ_s 也要真的被按住；不能贴限幅
    if abs(dthb[m].std()) > 0.02 * max(abs(omega_ref), 0.02):
        return None
    if np.abs(tsc[m]).max() > np.radians(g.THETA_S_TARGET_DEG):
        return None
    if np.abs(Tb[m]).max() >= 0.999 * g.TAU_B_MAX:
        return None
    if abs(omega) < 0.2 * abs(omega_ref):
        return None

    psi_b = thb[m]                              # θ_c = 0（基座静止），ψ_b = θ_b
    psi_s = thb[m] + tsc[m]
    return dict(
        omega_ref=float(omega_ref),
        omega=omega,
        tanh_omega=float(np.tanh(ip.LAMBDA * omega)),
        tb_mean=float(Tb[m].mean()),
        ts_mean=float(ts_hist[m].mean()),
        mean_sin_psi_b=float(np.sin(psi_b).mean()),
        mean_cos_psi_b=float(np.cos(psi_b).mean()),
        mean_sin_psi_s=float(np.sin(psi_s).mean()),
        mean_cos_psi_s=float(np.cos(psi_s).mean()),
        theta_s_max_deg=float(np.degrees(np.abs(tsc[m]).max())),
        dtheta_s_rms=float(np.sqrt(np.mean(dths[m] ** 2))),
        # 残余科氏项 C1 的量级（θ̇_s≠0 时会污染平衡式，取出来监控）
        coriolis_proxy=float(np.mean(dths[m] * (2.0 * dthb[m] + dths[m]))),
        dtheta_b_std=float(dthb[m].std()),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(REPO / "data" / "friction"))
    ap.add_argument("--n-gravity", type=int, default=4, help="每个速度用几个重力样本")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"匀速旋转摩擦辨识采集：{len(OMEGA_LIST)} 个速度 × {args.n_gravity} 个重力样本")
    print(f"  速度表 [rad/s]: {OMEGA_LIST}")
    print(f"  每段 {N_SETTLE} 步稳定 + {N_MEAS} 步测量 (dt={g.DT}s, 共 {(N_SETTLE+N_MEAS)*g.DT:.1f}s)")

    rows = []
    tried = failed = 0
    for omega_ref in OMEGA_LIST:
        for j in range(args.n_gravity):
            rng = np.random.default_rng(args.seed + 1000 * j + int(round(omega_ref * 1000)))
            gx, gy, alpha = g.sample_gravity(rng)
            _tr = ip.load_truth(ip.DATA_DIR / "truth_params.txt")
            P = Params(**{n: _tr[n] for n in ip.PARAM_NAMES},
                       gx=gx, gy=gy, lambda_=g.TRUE_PARAMS["lambda_"])
            theta_b0 = float(rng.uniform(-np.pi, np.pi))
            tried += 1
            r = run_one(P, omega_ref, theta_b0, rng)
            if r is None:
                failed += 1
                continue
            r.update(gx=gx, gy=gy, alpha_deg=alpha, theta_b0=theta_b0)
            rows.append(r)
        print(f"  ω={omega_ref:+.3f}: 累计成功 {len(rows)}/{tried}（失败 {failed}）")

    if not rows:
        print("没有成功的实验")
        return 1
    keys = sorted(rows[0].keys())
    np.savez_compressed(out_dir / "sweep.npz",
                        **{k: np.array([r[k] for r in rows], dtype=np.float64) for k in keys})
    print(f"\n完成：{len(rows)} 条 -> {out_dir / 'sweep.npz'}")
    print(f"  |ω| 范围 [{np.abs([r['omega'] for r in rows]).min():.4f}, "
          f"{np.abs([r['omega'] for r in rows]).max():.3f}] rad/s")
    print(f"  θ_s 峰值中位 {np.median([r['theta_s_max_deg'] for r in rows]):.2f}°  "
          f"θ̇_s RMS 中位 {np.median([r['dtheta_s_rms'] for r in rows]):.2e} rad/s")
    print(f"  速度波动(测量段 σ) 中位 {np.median([r['dtheta_b_std'] for r in rows]):.2e} rad/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
