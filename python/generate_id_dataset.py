"""生成系统辨识用的仿真数据集（用 env 仿真环境接口实现）。

与旧版的区别（按需求改）：
  * 全程只构造 **一个** SimEnv 实例，所有轨迹复用它；
  * 环境不支持重置，因此每条轨迹开始前用**控制力矩把状态控回**所需初值；
  * 动力学参数（含等效重力）在构造时固定 → **一次运行 = 一个斜坡倾角下的采集会话**，
    不同重力靠多次运行（配 --start-index 追加）拼接成完整数据集；
  * 力矩噪声直接加到**实际施加**的力矩上（记录力矩 = 实际施加力矩，数据自洽）。

流程（对每条轨迹）：
    0. 等效重力由命令行给出（--alpha-deg / --g-dir-deg；不给则按配置随机抽一次）
    1. 控回初值：θ_b 控到随机目标、θ_s 控到 0、两角速度控到 0
    2. 闭环 PD 跟踪参考轨迹产生激励；θ_s 参考按 ±15° 设计，关节 b 自由激励
    3. 限位验收：max|θ_s| > 35° 丢弃，> 30° 则收缩参考幅值重试
    4. 状态曲线加测量噪声后写出 npz

用法（在仓库根执行）::

    # 水平面（平面内重力 = 0）
    python3 python/generate_id_dataset.py --num 20 --alpha-deg 0 --start-index 0
    # 倾角 12°、平面内方向 90°
    python3 python/generate_id_dataset.py --num 20 --alpha-deg 12 --start-index 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import sim_config as cfg                                  # noqa: E402
from capi import State                                 # noqa: E402
from env import SimEnv                                      # noqa: E402

DATA_DIR = cfg.DATA_DIR_SIM


# ===========================================================================
# 无重置时的"控回初值"控制器
# ===========================================================================
def reposition(env: SimEnv, theta_b_target: float) -> tuple[bool, int, State]:
    """用控制力矩把状态控回 (θ_b=target, θ̇_b=0, θ_s=0, θ̇_s=0)。

    环境不支持重置，只能这样回去。两个关键点：
      * 目标点做**限速斜坡**（REPOS_V_MAX），否则起步力矩长期饱和、积分绕死，
        收敛会慢到几千步（实测 1980 步）；
      * **条件积分**抗饱和：只在力矩未饱和时累加积分。
    θ_b 取最近的等价目标，避免绕整圈。
    返回 (是否收敛, 用了多少步, 最终状态)。
    """
    st = env.state()
    tgt = theta_b_target + 2.0 * np.pi * round((st.theta_b - theta_b_target) / (2.0 * np.pi))
    ref = st.theta_b
    ib = is_ = 0.0
    for i in range(cfg.REPOS_MAX_STEPS):
        st = env.state()
        # 目标点限速逼近
        ref += float(np.clip(tgt - ref, -cfg.REPOS_V_MAX * env.dt, cfg.REPOS_V_MAX * env.dt))
        e_b, e_s = ref - st.theta_b, -st.theta_s
        g1, g2 = env.gravity_torque(st.theta_b, st.theta_s)   # 环境内部算的重力前馈
        Tb_raw = g1 + cfg.REPOS_KP * e_b + cfg.REPOS_KI * ib - cfg.REPOS_KD * st.dtheta_b
        Ts_raw = g2 + cfg.REPOS_KP * e_s + cfg.REPOS_KI * is_ - cfg.REPOS_KD * st.dtheta_s
        Tb = float(np.clip(Tb_raw, -cfg.TAU_B_MAX, cfg.TAU_B_MAX))
        Ts = float(np.clip(Ts_raw, -cfg.TAU_S_MAX, cfg.TAU_S_MAX))
        # 条件积分：只在未饱和时累加
        if Tb == Tb_raw:
            ib = float(np.clip(ib + e_b * env.dt, -cfg.REPOS_INT_CLAMP, cfg.REPOS_INT_CLAMP))
        if Ts == Ts_raw:
            is_ = float(np.clip(is_ + e_s * env.dt, -cfg.REPOS_INT_CLAMP, cfg.REPOS_INT_CLAMP))
        st1 = env.step(Tb, Ts)
        if (abs(tgt - st1.theta_b) < cfg.REPOS_TOL_RAD and abs(st1.dtheta_b) < cfg.REPOS_TOL_VEL
                and abs(st1.theta_s) < cfg.REPOS_TOL_RAD
                and abs(st1.dtheta_s) < cfg.REPOS_TOL_VEL):
            return True, i + 1, st1
    return False, cfg.REPOS_MAX_STEPS, env.state()


# ===========================================================================
# 参考轨迹（多正弦，位置与速度双重归一化）
# ===========================================================================
def _multisine(rng, lo, hi, amp_pos, v_cap, K, dt, n=4):
    t = np.arange(K) * dt
    y = np.zeros(K)
    dy = np.zeros(K)
    for _ in range(n):
        f = rng.uniform(lo, hi)
        a = rng.uniform(0.3, 1.0)
        p = rng.uniform(0.0, 2.0 * np.pi)
        y += a * np.sin(2.0 * np.pi * f * t + p)
        dy += a * 2.0 * np.pi * f * np.cos(2.0 * np.pi * f * t + p)
    sc = min(amp_pos / max(np.abs(y).max(), 1e-12), v_cap / max(np.abs(dy).max(), 1e-12))
    ramp = np.ones(K)
    nr = max(1, int(round(cfg.SHAPE_RAMP_TIME / dt)))
    s = np.linspace(0.0, 1.0, nr)
    sm = s * s * (3.0 - 2.0 * s)
    ramp[:nr] *= sm
    ramp[-nr:] *= sm[::-1]
    return y * sc * ramp, dy * sc * ramp


def _velocity_barrier(v: float, v_max: float, k: float,
                      beta: float = cfg.V_BARRIER_BETA) -> float:
    """连续速度障碍（softplus 平滑）：超出限幅时给正比于超出量的刹车力矩。"""
    def sp(x):
        z = beta * x
        if z > 30.0:
            return x
        if z < -30.0:
            return 0.0
        return float(np.log1p(np.exp(z)) / beta)
    return -k * sp(v - v_max) + k * sp(-v - v_max)


# ===========================================================================
# 在环境上跑一条激励轨迹（在线闭环 + 噪声注入实际力矩）
# ===========================================================================
def run_excitation(env: SimEnv, rng, ts_ref_deg: float, tb_ref_rad: float) -> dict:
    """从**当前状态**出发跑一条激励轨迹。

    力矩噪声直接加到实际施加的力矩上，因此"记录力矩 = 实际施加力矩"，
    记录的状态就是该力矩的真实响应（数据自洽，不需要再复现一遍）。
    """
    K, dt = cfg.NUM_STEPS, env.dt
    yb, dyb = _multisine(rng, cfg.TB_FREQ_LO, cfg.TB_FREQ_HI, tb_ref_rad,
                         cfg.V_REF_MARGIN * cfg.V_MAX_B, K, dt)
    ys, dys = _multisine(rng, cfg.TS_FREQ_LO, cfg.TS_FREQ_HI, np.radians(ts_ref_deg),
                         cfg.V_REF_MARGIN * cfg.V_MAX_S, K, dt)
    st0 = env.state()
    tb0, ts0 = st0.theta_b, st0.theta_s

    tb_cmd = np.zeros(K); ts_cmd = np.zeros(K)
    tb_app = np.zeros(K); ts_app = np.zeros(K)
    psi_b = np.zeros(K); psi_s = np.zeros(K)
    dpsi_b = np.zeros(K); dpsi_s = np.zeros(K)

    for k in range(K):
        st = env.state()
        ub = cfg.KP_B * ((tb0 + yb[k]) - st.theta_b) + cfg.KD_B * (dyb[k] - st.dtheta_b)
        us = cfg.KP_S * ((ts0 + ys[k]) - st.theta_s) + cfg.KD_S * (dys[k] - st.dtheta_s)
        ub += _velocity_barrier(st.dtheta_b, cfg.V_MAX_B, cfg.V_BARRIER_GAIN * cfg.TAU_B_MAX)
        us += _velocity_barrier(st.dtheta_s, cfg.V_MAX_S, cfg.V_BARRIER_GAIN * cfg.TAU_S_MAX)
        ub = float(np.clip(ub, -cfg.TAU_B_MAX, cfg.TAU_B_MAX))
        us = float(np.clip(us, -cfg.TAU_S_MAX, cfg.TAU_S_MAX))
        tb_cmd[k], ts_cmd[k] = ub, us
        # 力矩噪声加在【实际施加】上（真实驱动器就是这样）
        ab = float(np.clip(ub + rng.normal(0.0, cfg.SIGMA_TAU), -cfg.TAU_B_MAX, cfg.TAU_B_MAX))
        as_ = float(np.clip(us + rng.normal(0.0, cfg.SIGMA_TAU), -cfg.TAU_S_MAX, cfg.TAU_S_MAX))
        tb_app[k], ts_app[k] = ab, as_
        st1 = env.step(ab, as_)
        # 基座恒为 0，所以 psi_b = θ_b，psi_s = θ_b + θ_s
        psi_b[k] = st1.theta_b
        psi_s[k] = st1.theta_b + st1.theta_s
        dpsi_b[k] = st1.dtheta_b
        dpsi_s[k] = st1.dtheta_b + st1.dtheta_s

    return dict(tb_cmd=tb_cmd, ts_cmd=ts_cmd, tb_app=tb_app, ts_app=ts_app,
                psi_b=psi_b, psi_s=psi_s, dpsi_b=dpsi_b, dpsi_s=dpsi_s, x0=st0)


# ===========================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=cfg.DEFAULT_NUM,
                    help="本次会话生成多少条轨迹")
    ap.add_argument("--out", type=str, default=str(DATA_DIR))
    ap.add_argument("--seed", type=int, default=cfg.DEFAULT_SEED)
    ap.add_argument("--start-index", type=int, default=0,
                    help="本会话第一条的编号（多次运行拼接数据集时用）")
    ap.add_argument("--zero-gravity", action="store_true",
                    help="把等效重力强制设为 0（水平面）；不给则在倾角范围内随机")
    ap.add_argument("--gravity-seed", type=int, default=None,
                    help="重力抽样的随机种子；不给则每次运行都不同")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"数据目录: {out_dir}   会话轨迹数: {args.num}   起始编号: {args.start_index}")
    print(f"采样: dt={cfg.DT}s ({1/cfg.DT:.0f}Hz)  K={cfg.NUM_STEPS}  "
          f"refinement={cfg.REFINEMENT}")
    print(f"约束: θ_s 目标 ≤{cfg.THETA_S_TARGET_DEG:.0f}° / 硬限 "
          f"{cfg.THETA_S_LIMIT_DEG:.0f}°；速度 ±{cfg.V_MAX_B:.1f}/±{cfg.V_MAX_S:.1f} rad/s")

    # ---- 整个程序运行期间只构造一次环境；重力由环境构造时随机确定、只读 ----
    env = SimEnv(zero_gravity=args.zero_gravity, seed=args.gravity_seed)
    gx, gy = env.gravity
    alpha = env.gravity_alpha_deg
    print(f"等效重力（环境构造时随机确定，不可设置）: 倾角 α={alpha:.3f}°  "
          f"(gx,gy)=({gx:.4f},{gy:.4f})  |g_inplane|={np.hypot(gx, gy):.4f} m/s²"
          f"  （上界 9.81·sin20° = {cfg.G_INPLANE_MAX:.3f}）")

    idx = 0
    rejected = 0
    consec_fail = 0
    ts_all, vb_all, vs_all = [], [], []
    over_b = over_s = 0

    with env:
        while idx < args.num:
            case_no = args.start_index + idx
            case_rng = np.random.default_rng(args.seed + 7919 * (case_no + 1))
            tb_target = float(case_rng.uniform(-np.pi, np.pi))
            ok, nrep, _ = reposition(env, tb_target)
            if not ok:
                consec_fail += 1
                print(f"  case {case_no}: 控回初值未收敛（{nrep} 步），跳过"
                      f"（连续失败 {consec_fail}）")
                if consec_fail >= 5:
                    print("连续 5 次控回失败，提前终止本会话")
                    break
                rejected += 1
                continue
            consec_fail = 0

            # 逐步收缩参考幅值，直到 max|θ_s| 进入目标范围
            best = None
            for k_try in range(cfg.MAX_ATTEMPTS):
                sc = cfg.SHRINK ** k_try
                rec = run_excitation(env, case_rng, cfg.TS_REF_DEG * sc, cfg.TB_REF_RAD * sc)
                ts_deg = float(np.degrees(np.abs(rec["psi_s"] - rec["psi_b"]).max()))
                if ts_deg <= cfg.THETA_S_TARGET_DEG:
                    best = (rec, ts_deg)
                    break
                if ts_deg <= cfg.THETA_S_LIMIT_DEG and best is None:
                    best = (rec, ts_deg)
                reposition(env, tb_target)      # 未达标：控回初值再试
            if best is None:
                rejected += 1
                continue
            rec, ts_deg = best

            psi_b, psi_s = rec["psi_b"], rec["psi_s"]
            dpsi_b, dpsi_s = rec["dpsi_b"], rec["dpsi_s"]
            tau_b, tau_s = rec["tb_app"], rec["ts_app"]
            ts_all.append(ts_deg)
            v_b = np.abs(dpsi_b)
            v_s = np.abs(dpsi_s - dpsi_b)
            vb_all.append(float(v_b.max()))
            vs_all.append(float(v_s.max()))
            over_b += int((v_b > cfg.V_MAX_B).sum())
            over_s += int((v_s > cfg.V_MAX_S).sum())

            # 状态曲线加测量噪声（力矩噪声已加在实际施加上）
            psi_b_n = psi_b + case_rng.normal(0.0, cfg.SIGMA_POS, cfg.NUM_STEPS)
            psi_s_n = psi_s + case_rng.normal(0.0, cfg.SIGMA_POS, cfg.NUM_STEPS)
            dpsi_b_n = dpsi_b + case_rng.normal(0.0, cfg.SIGMA_VEL, cfg.NUM_STEPS)
            dpsi_s_n = dpsi_s + case_rng.normal(0.0, cfg.SIGMA_VEL, cfg.NUM_STEPS)

            np.savez_compressed(
                out_dir / f"case_{case_no:04d}.npz",
                theta_c0=np.float64(0.0), dtheta_c=np.float64(0.0),
                ddtheta_c=np.float64(0.0),
                x0_theta_b=np.float64(rec["x0"].theta_b),
                x0_dtheta_b=np.float64(rec["x0"].dtheta_b),
                x0_theta_s=np.float64(rec["x0"].theta_s),
                x0_dtheta_s=np.float64(rec["x0"].dtheta_s),
                dt=np.float64(cfg.DT), num_steps=np.int64(cfg.NUM_STEPS),
                refinement=np.int64(cfg.REFINEMENT), seed=np.int64(args.seed),
                case_index=np.int64(case_no),
                weights=np.array([cfg.W_PSI_B, cfg.W_PSI_S,
                                  cfg.W_DPSI_B, cfg.W_DPSI_S], dtype=np.float64),
                noise_sigma=np.array([cfg.SIGMA_POS, cfg.SIGMA_VEL, cfg.SIGMA_TAU],
                                     dtype=np.float64),
                gx=np.float64(gx), gy=np.float64(gy),
                gravity_alpha_deg=np.float64(alpha),
                theta_s_max_deg=np.float64(ts_deg),
                v_max=np.array([cfg.V_MAX_B, cfg.V_MAX_S], dtype=np.float64),
                max_dtheta_b=np.float64(vb_all[-1]),
                max_dtheta_s=np.float64(vs_all[-1]),
                # 实测（状态含测量噪声；力矩 = 实际施加，本身已含噪）
                psi_b=psi_b_n, psi_s=psi_s_n, dpsi_b=dpsi_b_n, dpsi_s=dpsi_s_n,
                tau_b=tau_b, tau_s=tau_s,
                # 无测量噪声的参考（力矩为未加噪的指令值）
                psi_b_clean=psi_b, psi_s_clean=psi_s,
                dpsi_b_clean=dpsi_b, dpsi_s_clean=dpsi_s,
                tau_b_clean=rec["tb_cmd"], tau_s_clean=rec["ts_cmd"],
            )
            if idx % 10 == 0 or idx == args.num - 1:
                print(f"  [{idx + 1:4d}/{args.num}] case_{case_no:04d}  α={alpha:5.2f}°  "
                      f"max|θs|={ts_deg:5.1f}°  控回用 {nrep:4d} 步  "
                      f"|Tb|max={np.abs(tau_b).max():.2f} |Ts|max={np.abs(tau_s).max():.2f}")
            idx += 1

    n_samp = max(1, idx * cfg.NUM_STEPS)
    if ts_all:
        a = np.array(ts_all)
        print(f"\nθ_s 峰值: 中位 {np.median(a):.1f}°  最大 {a.max():.1f}°  "
              f"超 {cfg.THETA_S_TARGET_DEG:.0f}° 的条数 "
              f"{int((a > cfg.THETA_S_TARGET_DEG).sum())}/{a.size}")
        print(f"速度限幅 ±{cfg.V_MAX_B:.1f}/{cfg.V_MAX_S:.1f}: max|θ̇b| 中位 "
              f"{np.median(vb_all):.2f} 越界 {100.0*over_b/n_samp:.2f}%；max|θ̇s| 中位 "
              f"{np.median(vs_all):.2f} 越界 {100.0*over_s/n_samp:.2f}%")
    print(f"完成：本次生成 {idx} 条 -> {out_dir}（跳过 {rejected} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
