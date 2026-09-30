"""匀速旋转摩擦辨识采集（用 env 仿真环境接口实现）。

原理（与库内模型严格对应）
    θ_s 被控制器按住（θ̇_s≈0、θ̈_s≈0），θ_b 以恒定 ω 旋转（θ̈_b=0），基座恒为 0。
    此时两坐标的运动方程都退化成力矩平衡，关节 b 那条只剩 G1：
        Tb = G1(ψ_b) + fbv·ω + fbc·tanh(λ·ω)
    用已辨识的动力学参数把 G1 用测得的 sin/cos 均值精确减掉，剩下就是 (ω, tanh(λω))
    的二维线性最小二乘——条件数 ~2，比主回归好十几个数量级，摩擦因此能真正解出来。

与主数据采集相同的架构
  * 全程只构造 **一个** SimEnv；
  * 环境不支持重置：每个速度开始前靠控制力矩把状态带过去（PI 速度环天然完成）；
  * 动力学参数（含等效重力）构造时固定 → 一次运行 = 一个斜坡倾角；
    多个倾角用 --append 追加到同一个 sweep.npz。

用法::

    python3 python/gen_friction_sweep.py --alpha-deg 0
    python3 python/gen_friction_sweep.py --alpha-deg 12 --append
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import sim_config as cfg                                  # noqa: E402
from env import SimEnv                                      # noqa: E402


def run_one(env: SimEnv, omega_ref: float) -> dict | None:
    """在**当前状态**上做一次匀速旋转实验，返回测量段统计量；不达标返回 None。

    控制器：
      * 关节 b：速度环 PI（KPV/KIV）——从当前速度收敛到 omega_ref，不用重置；
      * 关节 s：位置环 PD（KPS/KDS）把 s 牢牢按在 0。
    """
    n = cfg.FS_N_SETTLE + cfg.FS_N_MEAS
    Tb = np.zeros(n); Ts = np.zeros(n)
    thb = np.zeros(n); dthb = np.zeros(n); dths = np.zeros(n); ths = np.zeros(n)
    integ = 0.0
    for k in range(n):
        st = env.state()                # 带测量噪声
        e = omega_ref - st.dtheta_b
        integ = float(np.clip(integ + e * env.dt, -cfg.FS_INT_CLAMP, cfg.FS_INT_CLAMP))
        ub = cfg.FS_KPV * e + cfg.FS_KIV * integ
        us = -cfg.FS_KPS * st.theta_s - cfg.FS_KDS * st.dtheta_s
        ub = float(np.clip(ub, -cfg.TAU_B_MAX, cfg.TAU_B_MAX))
        us = float(np.clip(us, -cfg.TAU_S_MAX, cfg.TAU_S_MAX))
        Tb[k] = ub; Ts[k] = us          # 记录发出去的值（环境内部叠执行器噪声）
        thb[k] = st.theta_b; dthb[k] = st.dtheta_b
        ths[k] = st.theta_s; dths[k] = st.dtheta_s
        env.step(ub, us)

    m = slice(cfg.FS_N_SETTLE, n)
    omega = float(dthb[m].mean())
    # 稳态判据：测量速度本身带噪，所以比较"前半均值 vs 后半均值"与均值的标准误
    half = cfg.FS_N_MEAS // 2
    v1 = dthb[cfg.FS_N_SETTLE:cfg.FS_N_SETTLE + half].mean()
    v2 = dthb[cfg.FS_N_SETTLE + half:].mean()
    se = max(cfg.SIGMA_VEL, 1e-6) / np.sqrt(half) + 0.02 * abs(omega_ref)
    if abs(v1 - v2) > 4.0 * se:
        return None
    if np.abs(ths[m]).max() > np.radians(cfg.THETA_S_TARGET_DEG):
        return None
    if np.abs(Tb[m]).max() >= 0.999 * cfg.TAU_B_MAX:
        return None
    if abs(omega) < 0.2 * abs(omega_ref):
        return None

    psi_b = thb[m]                       # 基座恒为 0 → ψ_b = θ_b
    psi_s = thb[m] + ths[m]
    return dict(
        omega_ref=float(omega_ref), omega=omega,
        tanh_omega=float(np.tanh(cfg.LAMBDA * omega)),
        tb_mean=float(Tb[m].mean()), ts_mean=float(Ts[m].mean()),
        mean_sin_psi_b=float(np.sin(psi_b).mean()),
        mean_cos_psi_b=float(np.cos(psi_b).mean()),
        mean_sin_psi_s=float(np.sin(psi_s).mean()),
        mean_cos_psi_s=float(np.cos(psi_s).mean()),
        theta_s_max_deg=float(np.degrees(np.abs(ths[m]).max())),
        dtheta_s_rms=float(np.sqrt(np.mean(dths[m] ** 2))),
        coriolis_proxy=float(np.mean(dths[m] * (2.0 * dthb[m] + dths[m]))),
        dtheta_b_std=float(dthb[m].std()),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(cfg.DATA_DIR_FRICTION / "sweep.npz"))
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--zero-gravity", action="store_true",
                    help="把等效重力强制设为 0（水平面）；不给则在倾角范围内随机")
    ap.add_argument("--gravity-seed", type=int, default=None,
                    help="重力抽样的随机种子；不给则每次运行都不同")
    ap.add_argument("--no-noise", dest="noise", action="store_false",
                    help="关闭仿真环境里的噪声")
    ap.add_argument("--sigma-pos", type=float, default=cfg.SIGMA_POS)
    ap.add_argument("--sigma-vel", type=float, default=cfg.SIGMA_VEL)
    ap.add_argument("--sigma-tau", type=float, default=cfg.SIGMA_TAU)
    ap.add_argument("--append", action="store_true",
                    help="把本次结果追加到已有 sweep.npz（多个重力会话拼接）")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"匀速旋转摩擦采集: 数据 -> {out_path}")
    print(f"  速度表 [rad/s]: {cfg.OMEGA_LIST}")
    print(f"  每段 {cfg.FS_N_SETTLE} 步稳定 + {cfg.FS_N_MEAS} 步测量 "
          f"(dt={cfg.DT}s, 共 {(cfg.FS_N_SETTLE+cfg.FS_N_MEAS)*cfg.DT:.1f}s)")

    rows = []
    tried = failed = 0
    rng = np.random.default_rng(args.seed)
    # 整个程序运行期间只构造一次环境；重力由环境随机确定、只读
    env = SimEnv(zero_gravity=args.zero_gravity, noise=args.noise,
                 sigma_pos=args.sigma_pos, sigma_vel=args.sigma_vel,
                 sigma_tau=args.sigma_tau, seed=args.gravity_seed)
    gx, gy = env.gravity
    alpha = env.gravity_alpha_deg
    print(f"  等效重力（构造时随机确定，不可设置）: α={alpha:.3f}°  "
          f"(gx,gy)=({gx:.4f},{gy:.4f})  |g|={np.hypot(gx, gy):.4f} m/s²")
    with env:
        for omega_ref in cfg.OMEGA_LIST:
            tried += 1
            r = run_one(env, omega_ref)
            if r is None:
                failed += 1
                continue
            r.update(gx=gx, gy=gy, alpha_deg=float(alpha))
            rows.append(r)
            print(f"  ω={omega_ref:+.3f}: ok  ⟨Tb⟩={r['tb_mean']:+.5f} N·m  "
                  f"⟨ω⟩={r['omega']:+.4f}  max|θs|={r['theta_s_max_deg']:.3f}°")

    if not rows:
        print("本次没有成功的实验")
        return 1

    old = {}
    if args.append and out_path.exists():
        with np.load(out_path) as d:
            old = {k: np.asarray(d[k]) for k in d.files}
    keys = sorted(rows[0].keys())
    new = {k: np.array([r[k] for r in rows], dtype=np.float64) for k in keys}
    merged = {}
    for k in keys:
        merged[k] = np.concatenate([old[k], new[k]]) if k in old else new[k]
    np.savez_compressed(out_path, **merged)

    print(f"\n完成：本次 {len(rows)}/{tried} 条成功（失败 {failed}），"
          f"累计 {len(merged['omega'])} 条 -> {out_path}")
    print(f"  |ω| 范围 [{np.abs(merged['omega']).min():.4f}, "
          f"{np.abs(merged['omega']).max():.3f}] rad/s")
    print(f"  θ_s 峰值中位 {np.median(merged['theta_s_max_deg']):.3f}°  "
          f"θ̇_s RMS 中位 {np.median(merged['dtheta_s_rms']):.2e} rad/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
