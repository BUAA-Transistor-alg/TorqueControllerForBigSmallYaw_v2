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
    每次运行写出独立的 sweep_<启动时间戳>.npz，多个倾角直接多次运行即可，
    辨识时把同一类别目录下所有 sweep_*.npz 合并使用（不再需要 --append）。

用法::

    python3 python/gen_friction_sweep.py --alpha-deg 0 --category friction_simA
    python3 python/gen_friction_sweep.py --alpha-deg 12 --category friction_simA

    # 真实硬件（RealEnv）：状态来自通信、力矩下发给 MCU、不加噪声、忙等精确帧控制
    python3 python/gen_friction_sweep.py --real
    python3 python/gen_friction_sweep.py --real --category friction_real
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import sim_config as cfg                                  # noqa: E402
from capi import ImuLocation                                # noqa: E402
from env import RealEnv, SimEnv                             # noqa: E402


def run_one(env, omega_ref: float, vel_tol_frac: float = 0.02,
            vel_tol_abs: float = 0.0) -> dict | None:
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
    # 稳态判据：前半均值 vs 后半均值，容差 = 测量速度噪声的标准误 + 相对项 + 绝对项。
    # 仿真环境 sigma_vel 就是注入的噪声；真实环境 sigma=(0,0,0)，靠 --fs-vel-tol-* 给裕度
    # （真实编码器有量化/滞后，容差太紧会把所有点都判失败）。
    half = cfg.FS_N_MEAS // 2
    v1 = dthb[cfg.FS_N_SETTLE:cfg.FS_N_SETTLE + half].mean()
    v2 = dthb[cfg.FS_N_SETTLE + half:].mean()
    se = max(env.sigma[1], 1e-6) / np.sqrt(half) + vel_tol_frac * abs(omega_ref) + vel_tol_abs
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
    ap.add_argument("--out", type=str, default=None,
                    help="输出 npz 的路径；时间戳会插进文件名，如 --out data/x/sweep.npz "
                         "-> data/x/sweep_<时间戳>.npz。缺省时按 --category / "
                         "data/friction[_real] 解析")
    ap.add_argument("--category", type=str, default=None,
                    help="类别名（= 目录名）：输出到 data/<类别>/sweep_<时间戳>.npz")
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

    # ---- 真实硬件环境（RealEnv）----
    ap.add_argument("--real", action="store_true",
                    help="用真实硬件环境：状态来自通信严格反解、力矩下发给 MCU、"
                         "不加噪声、perf_counter_ns+忙等精确帧控制。"
                         "此时 --zero-gravity/--gravity-seed/--no-noise/--sigma-* 均无效")
    ap.add_argument("--imu-location", choices=("head", "big_yaw"), default="head",
                    help="[--real] IMU 安装构型（决定严格反解的运动学链）")
    ap.add_argument("--spin-us", type=float, default=300.0,
                    help="[--real] 每帧末尾纯自旋等待的时长 [µs]（越大越准越费 CPU）")
    ap.add_argument("--tau-b-max", type=float, default=cfg.TAU_B_MAX,
                    help="[--real] 关节 b 下发前硬限幅 [N·m]")
    ap.add_argument("--tau-s-max", type=float, default=cfg.TAU_S_MAX,
                    help="[--real] 关节 s 下发前硬限幅 [N·m]")
    ap.add_argument("--gravity-settle-s", type=float, default=0.3,
                    help="[--real] 会话开始对反解重力取平均的时长 [s]")
    ap.add_argument("--ready-timeout-s", type=float, default=5.0,
                    help="[--real] 等待 MCU+IMU 首个有效样本的超时 [s]")
    ap.add_argument("--hold-pitch", action="store_true",
                    help="[--real] 保持当前实测 pitch（不扰动）；不给则把 pitch 目标压到 "
                         "--pitch-target-deg")
    ap.add_argument("--pitch-target-deg", type=float, default=0.0,
                    help="[--real] 非 --hold-pitch 时下发的固定 pitch 目标角 [°]（默认 0）")
    ap.add_argument("--fs-vel-tol-frac", type=float, default=0.02,
                    help="稳态判据的相对容差（真实编码器有量化/滞后，建议放宽到 ~0.1）")
    ap.add_argument("--fs-vel-tol-abs", type=float, default=0.0,
                    help="稳态判据的绝对容差 [rad/s]（真机建议给一点，如 0.02）")
    args = ap.parse_args()

    stamp = cfg.run_stamp()          # 本次运行的启动时间戳
    if args.out is not None:
        _o = Path(args.out)
        out_path = _o.parent / f"{_o.stem}_{stamp}{_o.suffix or '.npz'}"
    else:
        base = (cfg.category_dir(args.category) if args.category
                else (cfg.DATA_DIR_FRICTION_REAL if args.real else cfg.DATA_DIR_FRICTION))
        out_path = base / f"sweep_{stamp}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"匀速旋转摩擦采集: {'真实硬件' if args.real else '仿真'} 数据 -> {out_path}")
    print(f"  速度表 [rad/s]: {cfg.OMEGA_LIST}")
    print(f"  每段 {cfg.FS_N_SETTLE} 步稳定 + {cfg.FS_N_MEAS} 步测量 "
          f"(dt={cfg.DT}s, 共 {(cfg.FS_N_SETTLE+cfg.FS_N_MEAS)*cfg.DT:.1f}s)")
    print(f"  稳态判据: |v1-v2| <= 4*(σ_v/√n + {args.fs_vel_tol_frac:g}*|ω_ref| "
          f"+ {args.fs_vel_tol_abs:g})")

    rows = []
    tried = failed = 0
    rng = np.random.default_rng(args.seed)
    # 整个程序运行期间只构造一次环境
    if args.real:
        print(f"  pitch: {'保持当前实测角' if args.hold_pitch else f'目标 {args.pitch_target_deg:g}°'}")
        env = RealEnv(
            dt=cfg.DT,
            imu_location=(ImuLocation.ON_HEAD if args.imu_location == "head"
                          else ImuLocation.ON_BIG_YAW),
            tau_b_max=args.tau_b_max, tau_s_max=args.tau_s_max,
            spin_us=args.spin_us, gravity_settle_s=args.gravity_settle_s,
            hold_pitch=args.hold_pitch,
            pitch_target=float(np.radians(args.pitch_target_deg)),
            ready_timeout_s=args.ready_timeout_s)
    else:
        env = SimEnv(zero_gravity=args.zero_gravity, noise=args.noise,
                     sigma_pos=args.sigma_pos, sigma_vel=args.sigma_vel,
                     sigma_tau=args.sigma_tau, seed=args.gravity_seed)
    gx, gy = env.gravity
    alpha = env.gravity_alpha_deg
    print(f"  等效重力（{'反解得到' if args.real else '构造时随机确定'}，不可设置）: "
          f"α={alpha:.3f}°  (gx,gy)=({gx:.4f},{gy:.4f})  |g|={np.hypot(gx, gy):.4f} m/s²")
    with env:
        for omega_ref in cfg.OMEGA_LIST:
            tried += 1
            r = run_one(env, omega_ref, args.fs_vel_tol_frac, args.fs_vel_tol_abs)
            if r is None:
                failed += 1
                continue
            r.update(gx=gx, gy=gy, alpha_deg=float(alpha),
                     real=np.float64(1.0 if args.real else 0.0))
            rows.append(r)
            print(f"  ω={omega_ref:+.3f}: ok  ⟨Tb⟩={r['tb_mean']:+.5f} N·m  "
                  f"⟨ω⟩={r['omega']:+.4f}  max|θs|={r['theta_s_max_deg']:.3f}°")
        if args.real:
            print(f"  帧统计: frames={env.frame_count} late={env.late_count} "
                  f"平均周期={env.mean_period*1e3:.3f}ms 最大={env.max_period*1e3:.3f}ms")

    if not rows:
        print("本次没有成功的实验")
        return 1

    keys = sorted(rows[0].keys())
    data = {k: np.array([r[k] for r in rows], dtype=np.float64) for k in keys}
    np.savez_compressed(out_path, **data)

    print(f"\n完成：本次 {len(rows)}/{tried} 条成功（失败 {failed}）-> {out_path}")
    print(f"  |ω| 范围 [{np.abs(data['omega']).min():.4f}, "
          f"{np.abs(data['omega']).max():.3f}] rad/s")
    print(f"  θ_s 峰值中位 {np.median(data['theta_s_max_deg']):.3f}°  "
          f"θ̇_s RMS 中位 {np.median(data['dtheta_s_rms']):.2e} rad/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
