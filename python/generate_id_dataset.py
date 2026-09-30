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

    # 真实硬件（RealEnv）：状态来自通信严格反解、力矩下发给 MCU、不加噪声、
    # perf_counter_ns + 忙等精确帧控制；数据默认写到 data/sim_real
    python3 python/generate_id_dataset.py --real --num 20 --start-index 0

之后用同一套辨识脚本读真实数据::

    python3 python/identify_params.py --data data/sim_real \
        --friction-sweep data/friction_real/sweep.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import sim_config as cfg                                  # noqa: E402
from capi import ImuLocation                              # noqa: E402
from capi import State as _State                          # noqa: E402
from capi import State                                 # noqa: E402
from env import RealEnv, SimEnv                             # noqa: E402

DATA_DIR = cfg.DATA_DIR_SIM


# ===========================================================================
# 无重置时的"控回初值"控制器
# ===========================================================================
def reposition(env, theta_b_target: float,
               max_steps: int = cfg.REPOS_MAX_STEPS,
               gravity_ff: bool = True) -> tuple[bool, int, State]:
    """用控制力矩把状态控回 (θ_b=target, θ̇_b=0, θ_s=0, θ̇_s=0)。

    环境不支持重置，只能这样回去。两个关键点：
      * 目标点做**限速斜坡**（REPOS_V_MAX），否则起步力矩长期饱和、积分绕死，
        收敛会慢到几千步（实测 1980 步）；
      * **条件积分**抗饱和：只在力矩未饱和时累加积分。
    θ_b 取最近的等价目标，避免绕整圈。

    :param gravity_ff: 是否叠加环境算出的重力前馈 (G1, G2)。False 时纯靠
                       PI/PD 通过误差把重力扛住（可用 --no-gravity-ff 开关）。
    返回 (是否收敛, 用了多少步, 最终状态)。
    """
    st_raw = env.state()
    tgt = theta_b_target + 2.0 * np.pi * round(
        (st_raw.theta_b - theta_b_target) / (2.0 * np.pi))
    ref = st_raw.theta_b
    ib = is_ = 0.0
    # 测量带噪：控制与收敛判据都用一阶低通后的估计
    filt = st_raw
    # 收敛容差按环境实际噪声自适应：低通后的残余 σ = σ·sqrt(a/(2-a))
    a0 = cfg.CTRL_LPF_ALPHA
    eps_eff = env.sigma[0] * np.sqrt(a0 / (2.0 - a0)) if env.noise else 0.0
    tol = max(cfg.REPOS_TOL_RAD, cfg.REPOS_TOL_SIGMA * eps_eff)
    for i in range(max_steps):
        m = env.state()
        a = cfg.CTRL_LPF_ALPHA
        filt = _State(theta_b=filt.theta_b + a * (m.theta_b - filt.theta_b),
                      dtheta_b=filt.dtheta_b + a * (m.dtheta_b - filt.dtheta_b),
                      theta_s=filt.theta_s + a * (m.theta_s - filt.theta_s),
                      dtheta_s=filt.dtheta_s + a * (m.dtheta_s - filt.dtheta_s))
        st = filt
        # 目标点限速逼近
        ref += float(np.clip(tgt - ref, -cfg.REPOS_V_MAX * env.dt, cfg.REPOS_V_MAX * env.dt))
        e_b, e_s = ref - st.theta_b, -st.theta_s
        # 重力前馈（环境内部算的 G1/G2）；--no-gravity-ff 时置 0，做对照实验
        if gravity_ff:
            g1, g2 = env.gravity_torque(st.theta_b, st.theta_s)
        else:
            g1 = g2 = 0.0
        Tb_raw = g1 + cfg.REPOS_KP_B * e_b + cfg.REPOS_KI_B * ib - cfg.REPOS_KD_B * st.dtheta_b
        Ts_raw = g2 + cfg.REPOS_KP_S * e_s + cfg.REPOS_KI_S * is_ - cfg.REPOS_KD_S * st.dtheta_s
        Tb = float(np.clip(Tb_raw, -cfg.TAU_B_MAX, cfg.TAU_B_MAX))
        Ts = float(np.clip(Ts_raw, -cfg.TAU_S_MAX, cfg.TAU_S_MAX))
        # 条件积分：只在未饱和时累加
        if Tb == Tb_raw:
            ib = float(np.clip(ib + e_b * env.dt, -cfg.REPOS_INT_CLAMP, cfg.REPOS_INT_CLAMP))
        if Ts == Ts_raw:
            is_ = float(np.clip(is_ + e_s * env.dt, -cfg.REPOS_INT_CLAMP, cfg.REPOS_INT_CLAMP))
        env.step(Tb, Ts)
        # 收敛判据基于滤波估计；起始状态 x0 另用长窗均值，避免把测量噪声当初值
        m2 = env.state()
        a = cfg.CTRL_LPF_ALPHA
        filt = _State(theta_b=filt.theta_b + a * (m2.theta_b - filt.theta_b),
                      dtheta_b=filt.dtheta_b + a * (m2.dtheta_b - filt.dtheta_b),
                      theta_s=filt.theta_s + a * (m2.theta_s - filt.theta_s),
                      dtheta_s=filt.dtheta_s + a * (m2.dtheta_s - filt.dtheta_s))
        if (abs(tgt - filt.theta_b) < tol and abs(filt.dtheta_b) < cfg.REPOS_TOL_VEL
                and abs(filt.theta_s) < tol and abs(filt.dtheta_s) < cfg.REPOS_TOL_VEL):
            return True, i + 1, _average_state(env, cfg.X0_AVG)
    return False, max_steps, _average_state(env, cfg.X0_AVG)


def _average_state(env, n: int) -> _State:
    """在当前状态附近多采几次取平均，作为对真实状态的估计（用于记录 x0）。"""
    acc = []
    for _ in range(max(1, n)):
        acc.append(env.state())
    k = 1.0 / len(acc)
    return _State(theta_b=sum(s.theta_b for s in acc) * k,
                  dtheta_b=sum(s.dtheta_b for s in acc) * k,
                  theta_s=sum(s.theta_s for s in acc) * k,
                  dtheta_s=sum(s.dtheta_s for s in acc) * k)


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
def run_excitation(env, rng, ts_ref_deg: float, tb_ref_rad: float) -> dict:
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

    fb = fs = None          # D 项用的滤波速度（测量带噪，直接微分会把噪声放大成力矩噪声）
    for k in range(K):
        st = env.state()
        a = cfg.CTRL_LPF_ALPHA
        fb = st.dtheta_b if fb is None else fb + a * (st.dtheta_b - fb)
        fs = st.dtheta_s if fs is None else fs + a * (st.dtheta_s - fs)
        ub = cfg.KP_B * ((tb0 + yb[k]) - st.theta_b) + cfg.KD_B * (dyb[k] - fb)
        us = cfg.KP_S * ((ts0 + ys[k]) - st.theta_s) + cfg.KD_S * (dys[k] - fs)
        ub += _velocity_barrier(st.dtheta_b, cfg.V_MAX_B, cfg.V_BARRIER_GAIN * cfg.TAU_B_MAX)
        us += _velocity_barrier(st.dtheta_s, cfg.V_MAX_S, cfg.V_BARRIER_GAIN * cfg.TAU_S_MAX)
        ub = float(np.clip(ub, -cfg.TAU_B_MAX, cfg.TAU_B_MAX))
        us = float(np.clip(us, -cfg.TAU_S_MAX, cfg.TAU_S_MAX))
        tb_cmd[k], ts_cmd[k] = ub, us
        # 记录"发出去的力矩"；环境内部自己叠执行器噪声
        tb_app[k], ts_app[k] = ub, us
        st1 = env.step(ub, us)
        # 基座恒为 0；这里记的是 env.step **返回的测量值**（ψ_b=θ_b, ψ_s=θ_b+θ_s）
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
    ap.add_argument("--out", type=str, default=None,
                    help="输出目录；缺省时 --real 用 data/sim_real，否则用 data/sim")
    ap.add_argument("--seed", type=int, default=cfg.DEFAULT_SEED)
    ap.add_argument("--start-index", type=int, default=0,
                    help="本会话第一条的编号（多次运行拼接数据集时用）")
    ap.add_argument("--zero-gravity", action="store_true",
                    help="把等效重力强制设为 0（水平面）；不给则在倾角范围内随机")
    ap.add_argument("--gravity-seed", type=int, default=None,
                    help="重力抽样的随机种子；不给则每次运行都不同")
    ap.add_argument("--no-noise", dest="noise", action="store_false",
                    help="关闭仿真环境里的噪声（执行器 + 传感器）")
    ap.add_argument("--sigma-pos", type=float, default=cfg.SIGMA_POS,
                    help="传给环境的传感器角度噪声 σ [rad]")
    ap.add_argument("--sigma-vel", type=float, default=cfg.SIGMA_VEL,
                    help="传给环境的传感器角速度噪声 σ [rad/s]")
    ap.add_argument("--sigma-tau", type=float, default=cfg.SIGMA_TAU,
                    help="传给环境的执行器力矩噪声 σ [N·m]")

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
    ap.add_argument("--max-repos-steps", type=int, default=cfg.REPOS_MAX_STEPS,
                    help="[--real] 控回初值的最多步数（真机收敛慢时调大）")
    ap.add_argument("--no-gravity-ff", dest="gravity_ff", action="store_false",
                    help="关闭控回初值里的重力前馈（对照实验用；默认开启）")
    ap.add_argument("--hold-pitch", action="store_true",
                    help="[--real] 保持当前实测 pitch（不扰动）；不给则把 pitch 目标压到 "
                         "--pitch-target-deg")
    ap.add_argument("--pitch-target-deg", type=float, default=0.0,
                    help="[--real] 非 --hold-pitch 时下发的固定 pitch 目标角 [°]（默认 0）")
    args = ap.parse_args()

    if args.out is None:
        args.out = str(cfg.DATA_DIR_SIM_REAL if args.real else DATA_DIR)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"数据目录: {out_dir}   会话轨迹数: {args.num}   起始编号: {args.start_index}   "
          f"数据源: {'真实硬件' if args.real else '仿真'}")
    print(f"采样: dt={cfg.DT}s ({1/cfg.DT:.0f}Hz)  K={cfg.NUM_STEPS}  "
          f"refinement={cfg.REFINEMENT}（回放用）")
    print(f"约束: θ_s 目标 ≤{cfg.THETA_S_TARGET_DEG:.0f}° / 硬限 "
          f"{cfg.THETA_S_LIMIT_DEG:.0f}°；速度 ±{cfg.V_MAX_B:.1f}/±{cfg.V_MAX_S:.1f} rad/s")
    if not args.real:
        print(f"噪声（由仿真环境在接口内施加）: {'启用' if args.noise else '关闭'}  "
              f"σ_pos={args.sigma_pos:g} rad  σ_vel={args.sigma_vel:g} rad/s  "
              f"σ_tau={args.sigma_tau:g} N·m")

    # ---- 整个程序运行期间只构造一次环境 ----
    if args.real:
        print(f"pitch: {'保持当前实测角' if args.hold_pitch else f'目标 {args.pitch_target_deg:g}°'}")
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
    print(f"等效重力（{'反解得到' if args.real else '环境构造时随机确定'}，不可设置）: "
          f"倾角 α={alpha:.3f}°  (gx,gy)=({gx:.4f},{gy:.4f})  "
          f"|g_inplane|={np.hypot(gx, gy):.4f} m/s²"
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
            ok, nrep, _ = reposition(env, tb_target, args.max_repos_steps,
                                     gravity_ff=args.gravity_ff)
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
                # 测量带噪：验收用短窗均值，避免把噪声尖峰当成越限
                ths = rec["psi_s"] - rec["psi_b"]
                w = max(1, cfg.ACCEPT_AVG)
                ths_s = np.convolve(ths, np.ones(w) / w, mode="valid")
                ts_deg = float(np.degrees(np.abs(ths_s).max()))
                if ts_deg <= cfg.THETA_S_TARGET_DEG:
                    best = (rec, ts_deg)
                    break
                if ts_deg <= cfg.THETA_S_LIMIT_DEG and best is None:
                    best = (rec, ts_deg)
                reposition(env, tb_target, args.max_repos_steps,
                           gravity_ff=args.gravity_ff)      # 未达标：控回初值再试
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
                # 关掉噪声时写 0，避免误以为数据带噪
                noise_enabled=np.int64(1 if (args.noise and not args.real) else 0),
                noise_sigma=np.array([args.sigma_pos, args.sigma_vel, args.sigma_tau],
                                     dtype=np.float64) if (args.noise and not args.real)
                else np.zeros(3, dtype=np.float64),
                # 数据来源：0 = 仿真，1 = 真实硬件（RealEnv）
                real=np.int64(1 if args.real else 0),
                gx=np.float64(gx), gy=np.float64(gy),
                gravity_alpha_deg=np.float64(alpha),
                theta_s_max_deg=np.float64(ts_deg),
                v_max=np.array([cfg.V_MAX_B, cfg.V_MAX_S], dtype=np.float64),
                max_dtheta_b=np.float64(vb_all[-1]),
                max_dtheta_s=np.float64(vs_all[-1]),
                # 实测：状态 = 接口返回的测量值；力矩 = 发出去的指令值
                psi_b=psi_b, psi_s=psi_s, dpsi_b=dpsi_b, dpsi_s=dpsi_s,
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

    if args.real:
        print(f"\n帧统计: frames={env.frame_count}  late={env.late_count}（落后重对齐的帧）  "
              f"平均周期={env.mean_period*1e3:.3f}ms  最大={env.max_period*1e3:.3f}ms  "
              f"(目标 {cfg.DT*1e3:.3f}ms)")
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
