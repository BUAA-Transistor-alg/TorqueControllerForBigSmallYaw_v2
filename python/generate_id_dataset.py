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

from tcbss import Params, State, Trajectory, TrajectoryLoss  # noqa: E402

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

# 重力矢量：每条数据在半径 GRAVITY_RADIUS 的圆**内**均匀采样（按面积均匀）。
GRAVITY_RADIUS = 9.81


def sample_gravity(rng: np.random.Generator) -> tuple[float, float]:
    """在半径 GRAVITY_RADIUS 的圆内均匀采样重力矢量。

    按面积均匀：r = R*sqrt(u)（若取 r = R*u 会过度集中于圆心）。
    """
    r = GRAVITY_RADIUS * np.sqrt(rng.uniform(0.0, 1.0))
    phi = rng.uniform(0.0, 2.0 * np.pi)
    return float(r * np.cos(phi)), float(r * np.sin(phi))


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
    print(f"重力: 每条数据在半径 {GRAVITY_RADIUS} 的圆内独立均匀采样")

    for idx in range(args.num):
        seed = args.seed + idx
        rng = np.random.default_rng(seed)

        # --- 本条数据的重力（已知输入）与完整参数 ---
        gx, gy = sample_gravity(rng)
        params = Params(**TRUE_PARAMS, gx=gx, gy=gy)

        # --- 初始状态 ---
        theta_c0 = float(rng.uniform(-np.pi, np.pi))
        dtheta_c = 0.0
        ddtheta_c = 0.0
        x0 = State(
            theta_b=float(rng.uniform(-np.pi, np.pi)),
            dtheta_b=float(rng.uniform(-1.0, 1.0)),
            theta_s=float(rng.uniform(-np.pi, np.pi)),
            dtheta_s=float(rng.uniform(-1.0, 1.0)),
        )

        # --- 力矩曲线：先叠加高斯白噪，作为【实际施加】的力矩 ---
        tau_b_clean, tau_s_clean = make_torque_curve(rng)
        tau_b = tau_b_clean + rng.normal(0.0, SIGMA_TAU, NUM_STEPS)
        tau_s = tau_s_clean + rng.normal(0.0, SIGMA_TAU, NUM_STEPS)

        # 用【实际施加】的力矩仿真，记录的状态才是该力矩的真实响应，数据自洽。
        # 若反之（用干净力矩仿真、却把含噪力矩当精确输入交给辨识），噪声会被
        # 混沌工况指数放大：实测 41/120 条 psi 偏移 >0.05 rad、最大 10.6 rad，
        # 真值参数处 loss 从 2.6e-3 涨到 33，辨识必然发散。
        psi_b, psi_s, dpsi_b, dpsi_s = simulate(
            params, x0, tau_b, tau_s, theta_c0, dtheta_c, ddtheta_c, REFINEMENT)

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
            span_b = tau_b.max() - tau_b.min()
            span_s = tau_s.max() - tau_s.min()
            print(f"  [{idx + 1:4d}/{args.num}] |Tb|max={np.abs(tau_b).max():.2f} "
                  f"|Ts|max={np.abs(tau_s).max():.2f}  "
                  f"psi_b范围={psi_b.max() - psi_b.min():.2f} "
                  f"psi_s范围={psi_s.max() - psi_s.min():.2f}")

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
