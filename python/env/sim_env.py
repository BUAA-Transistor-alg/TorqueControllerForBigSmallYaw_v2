"""仿真环境封装：参数从配置文件读取、重力构造时随机确定、无重置、基座状态恒为 0。

对外只暴露两个操作接口
----------------------
    state()          -> State    立即获取当前状态
    step(Tb, Ts)     -> State    输入两个**给电控的指令值**（[-1, +1]），推进一个 dt

只读信息（构造后固定，不能设置）
--------------------------------
    gravity          -> (gx, gy) 本次会话的等效重力
    noise / sigma                本次会话是否启用噪声、各自的 σ
    dt / refinement              采样周期与每主步 RK4 子步数

设计约束（按需求固定）
----------------------
  * **动力学参数直接从 sim_config 读取**，调用方不接触真实参数；
  * **重力在构造时自动随机选取**：默认在摆平面倾角 [0, RAMP_MAX_DEG] 内随机
    （等效重力模长 = 9.81·sinα），平面内方向也随机；构造参数 zero_gravity=True
    时强制完全为 0。构造后**只能获取、不能修改**。
  * **不支持重置状态**：需要回到某个状态，只能靠控制力矩把它控回去；
  * 构造后的初始状态：θ_b = θ_s = 0，θ̇_b = θ̇_s = 0；
  * 基座状态恒为 0：θ_c = θ̇_c = θ̈_c = 0。

因此一个程序运行期间只构造一次该环境、后续一直复用同一个实例。

用法::

    env = SimEnv()                  # 重力随机（倾角 0~20°）
    env = SimEnv(zero_gravity=True) # 重力恒为 0
    print(env.gravity)              # 只能获取
    st = env.step(Tb, Ts)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sim_config as cfg                       # noqa: E402
from capi import Params, Simulator, State      # noqa: E402


def sample_gravity(rng: np.random.Generator | None = None,
                   zero: bool = False) -> tuple[float, float, float]:
    """抽一组摆平面内的等效重力。

    倾角 α ~ U(0, RAMP_MAX_DEG)，平面内方向 φ ~ U(0, 2π)；
    等效重力模长 = GRAVITY_MAG·sin(α) ≤ GRAVITY_MAG·sin(RAMP_MAX_DEG)。
    zero=True 时直接返回 (0, 0, 0)。

    返回 (gx, gy, alpha_deg)。
    """
    if zero:
        return 0.0, 0.0, 0.0
    r = rng if rng is not None else np.random.default_rng()
    alpha = float(r.uniform(cfg.ALPHA_MIN_DEG, cfg.RAMP_MAX_DEG))
    mag = cfg.GRAVITY_MAG * math.sin(math.radians(alpha))
    phi = float(r.uniform(0.0, 2.0 * math.pi))
    return mag * math.cos(phi), mag * math.sin(phi), alpha


class SimEnv:
    """双连杆仿真环境（基座静止、参数来自配置文件、重力随机但固定、不可重置）。

    噪声模型（构造时用 noise 开关控制，默认开）：
      * **执行器侧**：step(Tb, Ts) 内部把输入**指令值**叠加 σ_tau 的高斯噪声后再送进
        仿真；外部拿到的记录值就是那个输入，等价于"记录值上叠了噪声"。
        ★ 输入是"下发给电控的指令值"（协议恒在 [-1, +1]）；真实物理力矩由模型内部的
        通道增益换算：物理力矩 = kb·Tb、ks·Ts（见 sim_config.DEFAULT_PARAMS）。
      * **传感器侧**：state()/step() 返回的都是**叠加了测量噪声**的状态
        （角 σ_pos、角速度 σ_vel），内部仿真用的是无噪声的真实状态。
      采集脚本因此只需原样记录接口返回的状态、以及自己发出去的力矩，不自己做任何加噪。
    """

    def __init__(self, zero_gravity: bool = False,
                 noise: bool = cfg.NOISE_ENABLE,
                 sigma_pos: float = cfg.SIGMA_POS,
                 sigma_vel: float = cfg.SIGMA_VEL,
                 sigma_tau: float = cfg.SIGMA_TAU,
                 dt: float = cfg.DT,
                 refinement: int = cfg.REFINEMENT,
                 seed: int | None = None,
                 params_override: dict | None = None):
        # ---- 动力学参数直接从配置文件读取（调用方不接触真实参数）----
        # params_override 只用于"换一套真值造数据"（例如改 kb/ks 后重新采集），
        # 键必须已存在于 DEFAULT_PARAMS，避免拼错时静默生效。
        d = dict(cfg.DEFAULT_PARAMS)
        if params_override:
            bad = set(params_override) - set(d)
            if bad:
                raise ValueError(f"SimEnv: 未知的参数覆盖 {sorted(bad)}；"
                                 f"可选 {sorted(d)}")
            d.update({k: float(v) for k, v in params_override.items()})
        rng = np.random.default_rng(seed)
        gx, gy, alpha = sample_gravity(rng, zero=zero_gravity)
        self._gravity = (float(gx), float(gy))
        self._gravity_alpha_deg = float(alpha)
        self._params = Params(mb=d["mb"], Ib=d["Ib"], Pbx=d["Pbx"], Pby=d["Pby"],
                              ms=d["ms"], Is=d["Is"], Psx=d["Psx"], Psy=d["Psy"],
                              Dx=d["Dx"], Dy=d["Dy"], gx=gx, gy=gy,
                              fbc=d["fbc"], fbv=d["fbv"], fsc=d["fsc"], fsv=d["fsv"],
                              lambda_=d["lambda_"], kb=d.get("kb", 1.0), ks=d.get("ks", 1.0))
        self._dt = float(dt)
        self._refinement = int(refinement)
        self._noise = bool(noise)
        self._sigma = (float(sigma_pos), float(sigma_vel), float(sigma_tau))
        self._nrng = np.random.default_rng(seed)
        self._sim = Simulator(self._params, self._dt, self._refinement)
        # 初始状态：角度与角速度均为 0
        self._sim.set_state(State(theta_b=0.0, dtheta_b=0.0,
                                  theta_s=0.0, dtheta_s=0.0))

    # ---------------- 内部：传感器加噪 ----------------
    def _measured(self, st: State) -> State:
        if not self._noise:
            return st
        sp, sv, _ = self._sigma
        return State(theta_b=st.theta_b + self._nrng.normal(0.0, sp),
                     dtheta_b=st.dtheta_b + self._nrng.normal(0.0, sv),
                     theta_s=st.theta_s + self._nrng.normal(0.0, sp),
                     dtheta_s=st.dtheta_s + self._nrng.normal(0.0, sv))

    # ---------------- 对外操作接口 ----------------
    def state(self) -> State:
        """立即获取当前状态（启用噪声时为**测量值**）。"""
        return self._measured(self._sim.state)

    def step(self, Tb: float, Ts: float) -> State:
        """输入两个**指令值**（[-1, +1]），推进一个 dt，返回推进后的状态（启用噪声时为**测量值**）。

        内部按 "输入 + σ_tau 高斯噪声" 作为下发的指令推进（并做指令限幅），真实物理
        力矩 = kb/ks × 该指令值；基座状态恒为 0（theta_c = dtheta_c = ddtheta_c = 0）。
        """
        ab, as_ = float(Tb), float(Ts)
        if self._noise:
            # σ_tau 的口径是**物理力矩** [N·m]（见 sim_config 的 SIGMA_TAU 注释："直接加到
            # 实际施加上"）。而物理力矩 = kb/ks × 指令值，所以命令侧要**除以 k**，
            # 否则 kb=4 时实际力矩噪声会被放大成 0.02 N·m（比摩擦还大），
            # 让轨迹在 3 s 内发散、辨识退化成拟合噪声。
            sp, sv, s_tau = self._sigma
            ab = float(np.clip(ab + self._nrng.normal(0.0, s_tau / max(abs(self._params.kb), 1e-12)),
                               -cfg.TAU_B_MAX, cfg.TAU_B_MAX))
            as_ = float(np.clip(as_ + self._nrng.normal(0.0, s_tau / max(abs(self._params.ks), 1e-12)),
                                -cfg.TAU_S_MAX, cfg.TAU_S_MAX))
        return self._measured(self._sim.step(ab, as_, 0.0, 0.0, 0.0))

    # ---------------- 只读信息（构造后不可修改） ----------------
    @property
    def sigma(self) -> tuple[float, float, float]:
        """本次会话的噪声 σ (pos, vel, tau)。只读。"""
        return self._sigma

    @property
    def noise(self) -> bool:
        """本次会话是否启用噪声（构造时定，只读）。"""
        return self._noise

    @property
    def gravity(self) -> tuple[float, float]:
        """本次会话的等效重力 (gx, gy)。只能获取，不能设置。"""
        return self._gravity

    @property
    def gravity_alpha_deg(self) -> float:
        """本次会话的摆平面倾角 [°]（zero_gravity 时为 0）。只读。"""
        return self._gravity_alpha_deg

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def refinement(self) -> int:
        return self._refinement

    # ---------------- 控制器需要的模型量（不暴露参数本身） ----------------
    def gravity_torque(self, theta_b: float | None = None,
                       theta_s: float | None = None) -> tuple[float, float]:
        """给定状态下的静态重力矩，**已换算成"下发给电控的指令值"单位**。

        静止时物理力矩要扛住重力，即 κ = G；而 κ = k·(指令值) ⇒ 指令值 = G/k。
        所以这里返回 (G1/kb, G2/ks)，可以直接作为前馈叠加到 step() 的输入上
        （step 收的就是指令值）。放在环境里是为了让采集脚本做控制时不必接触
        真实参数、也不必自己换算增益。

        参数不给则用当前状态。
        """
        st = self._sim.state
        tb = st.theta_b if theta_b is None else float(theta_b)
        ts = st.theta_s if theta_s is None else float(theta_s)
        p = self._params
        gb_sin = p.gx * (p.mb * p.Pbx + p.ms * p.Dx) + p.gy * (p.mb * p.Pby + p.ms * p.Dy)
        gb_cos = p.gx * (p.mb * p.Pby + p.ms * p.Dy) - p.gy * (p.mb * p.Pbx + p.ms * p.Dx)
        gs_sin = p.ms * (p.gx * p.Psx + p.gy * p.Psy)
        gs_cos = p.ms * (p.gx * p.Psy - p.gy * p.Psx)
        psi_b, psi_s = tb, tb + ts
        G2 = gs_sin * math.sin(psi_s) + gs_cos * math.cos(psi_s)
        G1 = gb_sin * math.sin(psi_b) + gb_cos * math.cos(psi_b) + G2
        return G1 / p.kb, G2 / p.ks

    # ---------------- 生命周期 ----------------
    def close(self) -> None:
        self._sim.close()

    def __enter__(self) -> "SimEnv":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (f"SimEnv(gravity=({self._gravity[0]:.4f}, {self._gravity[1]:.4f}), "
                f"alpha={self._gravity_alpha_deg:.3f}°, dt={self._dt}, "
                f"refinement={self._refinement})")
