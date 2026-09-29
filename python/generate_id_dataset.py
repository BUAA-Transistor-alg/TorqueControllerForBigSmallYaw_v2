"""生成系统辨识用的仿真数据集。

流程（对每条轨迹）：
    0. 重力矢量在半径 9.81 的圆**内**独立均匀采样 —— 它是**已知输入**，
       随该条数据一起保存，不参与辨识（各条重力方向不同，也有利于把质量矩
       mb*P* / ms*P* 与转动惯量 Ib / Is 分开）。
    1. 随机初始状态：theta_c / theta_b / theta_s 在 [-pi, pi)，三个角速度在 [-1, 1]；
       ddtheta_c = 0（基座等速外推，本数据集里恒为 0）。
    2. 生成平滑控制力矩曲线：3~5 个带限正弦叠加 + 首尾 0.2 s 平滑起步/收尾，
       幅值限制为 |Tb| <= 3 N*m、|Ts| <= 1 N*m。
    3. 先给力矩曲线叠加高斯白噪，得到"实际施加"的力矩；用【该含噪力矩】做
       正向仿真（100 Hz，每主步 refinement 个 RK4 子步；refinement 是运行期
       参数，构造 Trajectory 时传入），保证记录的状态就是该力矩的真实响应。
    4. 再给状态曲线叠加高斯白噪，得到最终"实测"曲线（位置/速度/力矩都含噪）。

输出：
    data/sim/case_XXXX.npz     每条轨迹一个文件（含该条自己的重力）
    data/sim/truth_params.txt  被辨识的 14 个真实参数（不含重力与 lambda_）

每个 npz 内含（除被辨识的动力学参数外，复现该曲线所需的全部信息）：
    theta_c0, dtheta_c, ddtheta_c, x0_*, weights_*, dt, num_steps, refinement,
    seed, case_index, noise_*,
    gx, gy   —— 该条数据的重力矢量（**已知输入**，不参与辨识）
以及序列：psi_b/psi_s/dpsi_b/dpsi_s 与 tau_b/tau_s（noisy 为实测，clean 为无噪声参考）

用法（在仓库根执行）::

    python3 python/generate_id_dataset.py                    # 默认 120 条
    python3 python/generate_id_dataset.py --num 120 --out data/sim
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # 本文件所在目录：python/
REPO = HERE.parent                              # 仓库根
sys.path.insert(0, str(HERE))

from tcbss import Params, Simulator, State, Trajectory, TrajectoryLoss  # noqa: E402

# ===========================================================================
# 1. 真实动力学参数
#    被辨识的 14 个量写进 truth_params.txt 供辨识脚本对比；
#    重力 gx/gy 是**每条数据各自随机**的已知输入，不进 truth_params.txt。
# ===========================================================================
TRUE_PARAMS = dict(
    mb=1.5, Ib=0.030, Pbx=0.12, Pby=0.18,
    ms=0.40, Is=0.020, Psx=0.08, Psy=0.10,
    Dx=0.05, Dy=0.30,
    fbc=0.020, fbv=0.050, fsc=0.010, fsv=0.020,
    lambda_=100.0,
)
# 与库的 kParamGradientNames 一致（重力不在其中）
PARAM_NAMES = ["mb", "Ib", "Pbx", "Pby", "ms", "Is", "Psx", "Psy",
               "Dx", "Dy", "fbc", "fbv", "fsc", "fsv"]

# ===========================================================================
# 等效重力：摆平面整体放在斜坡上，起作用的只是重力【在摆平面内】的分量
#   * 平面水平时重力垂直于摆平面 -> 平面内等效重力 = 0（大部分采集时间是这种）
#   * 斜坡倾角 α <= RAMP_MAX_DEG 时，平面内等效重力 = 9.81*sin(α) <= 3.36 m/s²，
#     方向在平面内任意
# 所以 |gx,gy| 的上界是 9.81*sin(20°)=3.36，而**不是** 9.81。
# ===========================================================================
GRAVITY_MAG = 9.81                      # 真实重力加速度大小
RAMP_MAX_DEG = 20.0                     # 斜坡最大倾角
G_INPLANE_MAX = GRAVITY_MAG * np.sin(np.radians(RAMP_MAX_DEG))   # ≈ 3.355
FLAT_PROB = 0.6                         # "大部分时间放在平面上"：平面内重力=0 的比例
ALPHA_MIN_DEG = 1.0                     # 倾斜时的最小倾角（避免退化的极小重力）


def sample_gravity(rng: np.random.Generator) -> tuple[float, float, float]:
    """采样摆平面内的等效重力矢量。

    返回 (gx, gy, alpha_deg)：
      * 以 FLAT_PROB 的概率放在水平面上 -> (0, 0, 0)
      * 否则倾角 α ~ U(ALPHA_MIN_DEG, RAMP_MAX_DEG)，平面内等效重力
        = 9.81*sin(α)，方向在平面内均匀
    """
    if rng.random() < FLAT_PROB:
        return 0.0, 0.0, 0.0
    alpha = np.radians(rng.uniform(ALPHA_MIN_DEG, RAMP_MAX_DEG))
    mag = GRAVITY_MAG * np.sin(alpha)
    phi = rng.uniform(0.0, 2.0 * np.pi)
    return float(mag * np.cos(phi)), float(mag * np.sin(phi)), float(np.degrees(alpha))


# ===========================================================================
# 2. 数据设置
# ===========================================================================
DT = 0.01                 # 100 Hz
DURATION = 3.0            # s
NUM_STEPS = int(round(DURATION / DT))   # 300
# 每主步的 RK4 子步数：refinement 是**运行期参数**，在构造 Trajectory 时传入
# （见 python/tcbss.py 的 Trajectory(params, refinement=...)），不是编译期常量。
REFINEMENT = 4

W_PSI_B = 1.0
W_PSI_S = 1.0
W_DPSI_B = 1.0
W_DPSI_S = 1.0

SIGMA_POS = 1.0e-2        # rad
SIGMA_VEL = 5.0e-2        # rad/s
SIGMA_TAU = 5.0e-3        # N*m

TAU_B_MAX = 3.0           # N*m
TAU_S_MAX = 1.0           # N*m
RAMP_TIME = 0.2           # s，首尾平滑时间
N_SIN_MIN, N_SIN_MAX = 3, 5
FREQ_MIN, FREQ_MAX = 0.2, 2.0   # Hz

# ---------------------------------------------------------------------------
# θ_s 机械限位约束
#   机构 θ_s 机械限位 ±45°；超过 ±35° 电控会介入降力矩，于是"记录的发送力矩"
#   不再等于"实际施加力矩"，该条数据对辨识就失效了。因此：
#     * 设计目标：整条轨迹 max|θ_s| <= THETA_S_TARGET_DEG（留 5° 余量）
#     * 硬约束：  实测 max|θ_s| > THETA_S_LIMIT_DEG 直接丢弃该条
#   注意 θ_s 是关节 s 相对关节 b 的角，**不受基座 θ_c 影响**，但会被关节 b 的
#   加速度通过惯性耦合 M12*qdd_b 拖着走，所以必须闭环稳定（见 make_excitation）。
# ---------------------------------------------------------------------------
THETA_S_LIMIT_DEG = 35.0
THETA_S_TARGET_DEG = 30.0
THETA_S0_DEG = 5.0            # 初始 θ_s 抖动（不再整圈随机）
DTHETA_S0_MAX = 0.2           # 初始 θ̇_s [rad/s]
TS_REF_DEG = 15.0             # θ_s 参考轨迹幅值
TB_REF_RAD = 0.8              # θ_b 参考轨迹幅值
KP_B, KD_B = 36.0, 3.4        # 关节 b 的 PD 增益
KP_S, KD_S = 30.0, 2.0        # 关节 s 的 PD 增益
MAX_ATTEMPTS = 60             # 单条数据最多尝试次数
SHRINK = 0.85                 # 每次失败后参考幅值的收缩系数
TS_FREQ_LO, TS_FREQ_HI = 0.5, 3.0   # θ_s 参考频带（避开 ~0.7Hz 固有频率）
TB_FREQ_LO, TB_FREQ_HI = 0.2, 1.5   # θ_b 参考频带

# ---------------------------------------------------------------------------
# 两关节速度限幅（采集时的保护）
#   参考轨迹按【位置和速度双重归一化】设计，先天不超速；控制器里再加一层**连续
#   速度障碍**：超出限幅时给出正比于超出量的刹车力矩。用 softplus 平滑，处处可导、
#   没有开关跳变（原来用的是 `min(T,0)` 式的单边钳位，会在限幅点跳变、喂出抖振）。
#   超限**不丢弃**该条数据，只统计越界比例。
# ---------------------------------------------------------------------------
V_MAX_B = 3.0                 # 关节 b 速度限幅 [rad/s]
V_MAX_S = 3.0                 # 关节 s 速度限幅 [rad/s]
V_REF_MARGIN = 0.8            # 参考速度只用到限幅的 80%，给跟踪误差留余量
V_BARRIER_GAIN = 0.5          # 障碍增益 = GAIN * TAU_*_MAX / (1 rad/s 超出量)
V_BARRIER_BETA = 20.0         # softplus 锐度；越大越接近硬限幅，越小越平滑


def _soft_relu(x: float, beta: float) -> float:
    """(1/beta)*log(1+exp(beta*x))：beta→∞ 时趋近 max(x,0)，且处处可导。"""
    z = beta * x
    if z > 30.0:
        return float(x)
    if z < -30.0:
        return 0.0
    return float(np.log1p(np.exp(z)) / beta)


def velocity_barrier(v: float, v_max: float, k: float,
                     beta: float = V_BARRIER_BETA) -> float:
    """连续速度障碍：|v| 超过 v_max 时输出正比于超出量的【刹车】力矩。

        barrier(v) = -k*softplus(v - v_max) + k*softplus(-v - v_max)

    |v| < v_max 时两项都≈0（softplus 在负半轴指数衰减）；v > v_max 时 ≈ -k(v-v_max)
    即反向刹车；v < -v_max 时 ≈ +k(-v-v_max)。全程连续可导，没有跳变。
    """
    return -k * _soft_relu(v - v_max, beta) + k * _soft_relu(-v - v_max, beta)


def make_torque_curve(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """生成一对平滑力矩曲线，幅值受 TAU_*_MAX 限制，首尾平滑归零。"""
    t = np.arange(NUM_STEPS) * DT
    n_sin = rng.integers(N_SIN_MIN, N_SIN_MAX + 1)
    freqs = rng.uniform(FREQ_MIN, FREQ_MAX, size=n_sin)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_sin)
    amps = rng.uniform(0.3, 1.0, size=n_sin)

    ramp = np.ones(NUM_STEPS)
    n_ramp = max(1, int(round(RAMP_TIME / DT)))
    s = np.linspace(0.0, 1.0, n_ramp)
    smooth = s * s * (3.0 - 2.0 * s)          # smoothstep
    ramp[:n_ramp] *= smooth
    ramp[-n_ramp:] *= smooth[::-1]

    curve = np.zeros((2, NUM_STEPS))
    for i in range(2):
        y = np.zeros(NUM_STEPS)
        for a, f, p in zip(amps, freqs * (1.0 + 0.15 * i), phases + 0.7 * i):
            y += a * np.sin(2.0 * np.pi * f * t + p)
        y *= ramp
        peak = np.max(np.abs(y))
        limit = TAU_B_MAX if i == 0 else TAU_S_MAX
        if peak > 0.0:
            y *= limit / peak
        curve[i] = y
    return curve[0], curve[1]


def make_excitation(params: Params, x0: State, theta_c0: float,
                    rng: np.random.Generator,
                    ts_ref_deg: float = TS_REF_DEG,
                    tb_ref_rad: float = TB_REF_RAD,
                    v_max_b: float = V_MAX_B,
                    v_max_s: float = V_MAX_S
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """用 PD 跟踪预设计的参考轨迹来生成激励力矩（记录的就是实际施加的力矩）。

    为什么必须闭环：θ_s 会被关节 b 的加速度通过惯性耦合 M12*qdd_b 硬拖着走。
    实测纯开环随机正弦下，即使 Ts 幅值降到满幅的 1/10，max|θ_s| 仍有 816°；
    把 Tb 也压到 1/10 才降到 140°，都远达不到 ±30°。闭环后参考轨迹按限位设计，
    实测 max|θ_s| 可压到 25~30°。

    参考轨迹（位置 + 速度**双重归一化**，先天满足速度限幅）：
      * θ_b^d：0.2~1.5 Hz 多正弦，位置幅值 tb_ref_rad、速度幅值 ≤ V_REF_MARGIN*v_max_b
      * θ_s^d：0.5~3.0 Hz 多正弦，位置幅值 ts_ref_deg、速度幅值 ≤ V_REF_MARGIN*v_max_s
    力矩 = PD(参考 − 实际) + 连续速度障碍，再经 TAU_*_MAX 限幅，**原样记录**：
    记录力矩即实际施加力矩，对辨识来说仍是合法的已知输入。

    超速不丢弃数据：参考已按上限设计，控制器再用连续速度障碍（softplus 平滑，
    无开关跳变）给出正比于超出量的刹车力矩。

    返回 (tau_b, tau_s, theta_s_ref, dtheta_s_ref)。
    """
    t = np.arange(NUM_STEPS) * DT

    def multisine(lo: float, hi: float, amp_pos: float, v_cap: float, n: int = 4):
        """多正弦位置参考：同时按位置幅值 amp_pos 和速度幅值 v_cap 归一化。"""
        y = np.zeros(NUM_STEPS)
        dy = np.zeros(NUM_STEPS)
        for _ in range(n):
            f = rng.uniform(lo, hi)
            a = rng.uniform(0.3, 1.0)
            p = rng.uniform(0.0, 2.0 * np.pi)
            y += a * np.sin(2.0 * np.pi * f * t + p)
            dy += a * 2.0 * np.pi * f * np.cos(2.0 * np.pi * f * t + p)
        sc = min(amp_pos / max(np.abs(y).max(), 1e-12),
                 v_cap / max(np.abs(dy).max(), 1e-12))
        return y * sc, dy * sc

    yb, dyb = multisine(TB_FREQ_LO, TB_FREQ_HI, tb_ref_rad, V_REF_MARGIN * v_max_b)
    ys, dys = multisine(TS_FREQ_LO, TS_FREQ_HI, np.radians(ts_ref_deg),
                        V_REF_MARGIN * v_max_s)

    tb = np.zeros(NUM_STEPS)
    ts = np.zeros(NUM_STEPS)
    with Simulator(params, DT, REFINEMENT) as sim:
        sim.set_state(State(theta_b=x0.theta_b + yb[0], dtheta_b=x0.dtheta_b + dyb[0],
                            theta_s=x0.theta_s + ys[0], dtheta_s=x0.dtheta_s + dys[0]))
        for k in range(NUM_STEPS):
            st = sim.state
            ub = KP_B * ((x0.theta_b + yb[k]) - st.theta_b) + KD_B * (dyb[k] - st.dtheta_b)
            us = KP_S * ((x0.theta_s + ys[k]) - st.theta_s) + KD_S * (dys[k] - st.dtheta_s)
            # 连续速度障碍：超出限幅时叠加正比于超出量的刹车力矩（无开关跳变）
            ub += velocity_barrier(st.dtheta_b, v_max_b, V_BARRIER_GAIN * TAU_B_MAX)
            us += velocity_barrier(st.dtheta_s, v_max_s, V_BARRIER_GAIN * TAU_S_MAX)
            tb[k] = np.clip(ub, -TAU_B_MAX, TAU_B_MAX)
            ts[k] = np.clip(us, -TAU_S_MAX, TAU_S_MAX)
            sim.step(tb[k], ts[k], theta_c0, 0.0, 0.0)
    return tb, ts, x0.theta_s + ys, dys


def simulate(params: Params, x0: State, tau_b: np.ndarray, tau_s: np.ndarray,
             theta_c0: float, dtheta_c: float, ddtheta_c: float,
             refinement: int = REFINEMENT
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """用库的正向仿真生成四个全局量序列（refinement 运行期传入）。"""
    tau = np.column_stack([tau_b, tau_s]).astype(np.float64)
    spec = TrajectoryLoss(w_psi_b=W_PSI_B, w_psi_s=W_PSI_S,
                          w_dpsi_b=W_DPSI_B, w_dpsi_s=W_DPSI_S)
    with Trajectory(params, refinement=refinement) as tj:
        _, psi_b, psi_s, dpsi_b, dpsi_s = tj.loss(
            theta_c0, dtheta_c, ddtheta_c, DT, tau, x0, spec, return_sequences=True)
    return psi_b, psi_s, dpsi_b, dpsi_s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=120, help="轨迹条数")
    ap.add_argument("--out", type=str, default=str(REPO / "data" / "sim"))
    ap.add_argument("--seed", type=int, default=20240607)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"数据目录: {out_dir}")
    print(f"轨迹条数: {args.num}   dt={DT}s ({1/DT:.0f}Hz)   时长={DURATION}s   "
          f"K={NUM_STEPS}   记录 refinement={REFINEMENT}")
    print(f"等效重力: 摆平面内分量；水平面内为 0（概率 {FLAT_PROB:.0%}），"
          f"倾斜时 α~U({ALPHA_MIN_DEG},{RAMP_MAX_DEG})°，|g|=9.81·sinα ≤ {G_INPLANE_MAX:.3f} m/s²")
    print(f"θ_s 限位: 设计目标 ≤{THETA_S_TARGET_DEG:.0f}°，硬约束 ≤{THETA_S_LIMIT_DEG:.0f}°"
          f"（超出丢弃），激励用 PD 闭环（Ts 参考 ±{TS_REF_DEG:.0f}°）")
    print(f"速度限幅: 关节 b/s 均 ±{V_MAX_B:.1f}/±{V_MAX_S:.1f} rad/s（软限幅，超限不丢弃）")

    idx = 0
    attempts = 0
    max_attempts_total = args.num * MAX_ATTEMPTS
    rejected_limit = 0
    ts_max_all = []
    vb_all, vs_all = [], []
    over_b = over_s = 0
    while idx < args.num and attempts < max_attempts_total:
        attempts += 1
        rng = np.random.default_rng(args.seed + attempts)

        # --- 本条数据的等效重力（已知输入）与完整参数 ---
        gx, gy, alpha_deg = sample_gravity(rng)
        params = Params(**TRUE_PARAMS, gx=gx, gy=gy)

        theta_c0 = float(rng.uniform(-np.pi, np.pi))
        dtheta_c = 0.0
        ddtheta_c = 0.0

        # 逐次尝试：每次失败就把参考幅值收缩，直到 max|θ_s| 进入目标范围
        accepted = None
        fallback = None
        for k_try in range(MAX_ATTEMPTS):
            sc = SHRINK ** k_try
            x0 = State(
                theta_b=float(rng.uniform(-np.pi, np.pi)),
                dtheta_b=float(rng.uniform(-1.0, 1.0)),
                # θ_s 不再整圈随机：从 0 附近起，避免一起步就超机械限位
                theta_s=float(np.radians(rng.uniform(-THETA_S0_DEG, THETA_S0_DEG))),
                dtheta_s=float(rng.uniform(-DTHETA_S0_MAX, DTHETA_S0_MAX)),
            )
            # --- 闭环生成激励力矩（记录力矩 = 实际施加力矩）---
            tau_b_clean, tau_s_clean, ts_ref, _ = make_excitation(
                params, x0, theta_c0, rng,
                ts_ref_deg=TS_REF_DEG * sc, tb_ref_rad=TB_REF_RAD * sc)
            # 力矩测量噪声 -> 作为【实际施加】的力矩
            tau_b = tau_b_clean + rng.normal(0.0, SIGMA_TAU, NUM_STEPS)
            tau_s = tau_s_clean + rng.normal(0.0, SIGMA_TAU, NUM_STEPS)
            # 用【实际施加】的力矩开环复现，记录的状态才是该力矩的真实响应（数据自洽）
            psi_b, psi_s, dpsi_b, dpsi_s = simulate(
                params, x0, tau_b, tau_s, theta_c0, dtheta_c, ddtheta_c, REFINEMENT)
            ts_deg = float(np.degrees(np.abs(psi_s - psi_b).max()))
            if ts_deg <= THETA_S_TARGET_DEG:
                accepted = (x0, tau_b, tau_s, tau_b_clean, tau_s_clean, ts_ref,
                            psi_b, psi_s, dpsi_b, dpsi_s, ts_deg)
                break
            if ts_deg <= THETA_S_LIMIT_DEG and fallback is None:
                fallback = (x0, tau_b, tau_s, tau_b_clean, tau_s_clean, ts_ref,
                            psi_b, psi_s, dpsi_b, dpsi_s, ts_deg)
        else:
            rejected_limit += 1

        if accepted is None:
            # 目标(30°)没达到；若有满足硬约束(35°)的候选就退而用之，否则丢弃该条
            if fallback is None:
                continue
            accepted = fallback

        (x0, tau_b, tau_s, tau_b_clean, tau_s_clean, ts_ref,
         psi_b, psi_s, dpsi_b, dpsi_s, ts_deg) = accepted
        ts_max_all.append(ts_deg)
        seed = args.seed + attempts

        # --- 速度限幅统计（超限不丢弃，只看越界比例）---
        v_b = np.abs(dpsi_b)
        v_s = np.abs(dpsi_s - dpsi_b)
        v_b_max = float(v_b.max())
        v_s_max = float(v_s.max())
        vb_all.append(v_b_max)
        vs_all.append(v_s_max)
        over_b += int((v_b > V_MAX_B).sum())
        over_s += int((v_s > V_MAX_S).sum())

        # --- 状态曲线叠加高斯白噪 ---
        psi_b_n = psi_b + rng.normal(0.0, SIGMA_POS, NUM_STEPS)
        psi_s_n = psi_s + rng.normal(0.0, SIGMA_POS, NUM_STEPS)
        dpsi_b_n = dpsi_b + rng.normal(0.0, SIGMA_VEL, NUM_STEPS)
        dpsi_s_n = dpsi_s + rng.normal(0.0, SIGMA_VEL, NUM_STEPS)

        np.savez_compressed(
            out_dir / f"case_{idx:04d}.npz",
            # 复现所需信息
            theta_c0=np.float64(theta_c0),
            dtheta_c=np.float64(dtheta_c),
            ddtheta_c=np.float64(ddtheta_c),
            x0_theta_b=np.float64(x0.theta_b),
            x0_dtheta_b=np.float64(x0.dtheta_b),
            x0_theta_s=np.float64(x0.theta_s),
            x0_dtheta_s=np.float64(x0.dtheta_s),
            dt=np.float64(DT),
            num_steps=np.int64(NUM_STEPS),
            refinement=np.int64(REFINEMENT),
            seed=np.int64(seed),
            case_index=np.int64(idx),
            weights=np.array([W_PSI_B, W_PSI_S, W_DPSI_B, W_DPSI_S], dtype=np.float64),
            noise_sigma=np.array([SIGMA_POS, SIGMA_VEL, SIGMA_TAU], dtype=np.float64),
            # 重力矢量：随采集数据一起保存的**已知输入**，不是被辨识参数
            gx=np.float64(gx),
            gy=np.float64(gy),
            # 摆平面倾斜角（deg）：0 表示水平面（平面内重力为 0）
            gravity_alpha_deg=np.float64(alpha_deg),
            # θ_s 限位相关（供追溯/筛选；辨识不使用）
            theta_s_max_deg=np.float64(ts_deg),
            theta_s_ref=np.asarray(ts_ref, dtype=np.float64),
            # 速度限幅（已知约束；辨识不使用）
            v_max=np.array([V_MAX_B, V_MAX_S], dtype=np.float64),
            max_dtheta_b=np.float64(v_b_max),
            max_dtheta_s=np.float64(v_s_max),
            # 实测序列（含噪）
            psi_b=psi_b_n.astype(np.float64),
            psi_s=psi_s_n.astype(np.float64),
            dpsi_b=dpsi_b_n.astype(np.float64),
            dpsi_s=dpsi_s_n.astype(np.float64),
            # 实际施加的力矩（含噪，与上面仿真所用完全一致）
            tau_b=tau_b.astype(np.float64),
            tau_s=tau_s.astype(np.float64),
            # 无噪声参考（仅用于分析，辨识不应使用）
            psi_b_clean=psi_b.astype(np.float64),
            psi_s_clean=psi_s.astype(np.float64),
            dpsi_b_clean=dpsi_b.astype(np.float64),
            dpsi_s_clean=dpsi_s.astype(np.float64),
            tau_b_clean=tau_b_clean.astype(np.float64),
            tau_s_clean=tau_s_clean.astype(np.float64),
        )

        if idx % 20 == 0 or idx == args.num - 1:
            print(f"  [{idx + 1:4d}/{args.num}] α={alpha_deg:5.1f}°  "
                  f"|Tb|max={np.abs(tau_b).max():.2f} |Ts|max={np.abs(tau_s).max():.2f}  "
                  f"max|θs|={ts_deg:5.1f}°  max|θ̇b|={v_b_max:4.2f} max|θ̇s|={v_s_max:4.2f}  "
                  f"psi_b范围={psi_b.max() - psi_b.min():.2f} "
                  f"psi_s范围={psi_s.max() - psi_s.min():.2f}")
        idx += 1

    ts_arr = np.array(ts_max_all) if ts_max_all else np.zeros(0)
    if ts_arr.size:
        print(f"θ_s 峰值统计: 中位 {np.median(ts_arr):.1f}°，最大 {ts_arr.max():.1f}°，"
              f"超过 {THETA_S_TARGET_DEG:.0f}°（退回硬约束内）的条数 "
              f"{int((ts_arr > THETA_S_TARGET_DEG).sum())}/{ts_arr.size}")
    if vb_all:
        n_samp = idx * NUM_STEPS
        print(f"速度限幅(±{V_MAX_B:.1f}/{V_MAX_S:.1f} rad/s): "
              f"max|θ̇b| 中位 {np.median(vb_all):.2f}、超限样本 {over_b}/{n_samp} "
              f"({100.0*over_b/n_samp:.2f}%)；max|θ̇s| 中位 {np.median(vs_all):.2f}、"
              f"超限样本 {over_s}/{n_samp} ({100.0*over_s/n_samp:.2f}%)")
    print(f"总尝试 {attempts} 次生成 {idx} 条；其中 {rejected_limit} 次连 {THETA_S_LIMIT_DEG:.0f}° "
          f"都没达到已丢弃")

    # --- 保存真实参数 ---
    with open(out_dir / "truth_params.txt", "w", encoding="utf-8") as f:
        f.write("# 真实动力学参数（供辨识脚本对比）\n")
        f.write(f"# dt={DT} num_steps={NUM_STEPS} refinement={REFINEMENT} "
                f"duration={DURATION}s\n")
        f.write(f"# noise_sigma: pos={SIGMA_POS} vel={SIGMA_VEL} tau={SIGMA_TAU}\n")
        f.write(f"# weights: {W_PSI_B} {W_PSI_S} {W_DPSI_B} {W_DPSI_S}\n")
        for name in PARAM_NAMES:
            f.write(f"{name} {TRUE_PARAMS[name]:.10g}\n")
        f.write(f"lambda_ {TRUE_PARAMS['lambda_']:.10g}\n")

    print(f"完成：{args.num} 条 -> {out_dir}")
    print(f"真实参数 -> {out_dir / 'truth_params.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
