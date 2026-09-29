"""仿真环境封装：参数从配置文件读取、重力构造时随机确定、无重置、基座状态恒为 0。

对外只暴露两个操作接口
----------------------
    state()          -> State    立即获取当前状态
    step(Tb, Ts)     -> State    输入两个控制力矩，推进一个 dt，返回推进后的状态

只读信息（构造后固定，不能设置）
--------------------------------
    gravity          -> (gx, gy) 本次会话的等效重力
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
    """双连杆仿真环境（基座静止、参数来自配置文件、重力随机但固定、不可重置）。"""

    def __init__(self, zero_gravity: bool = False,
                 dt: float = cfg.DT,
                 refinement: int = cfg.REFINEMENT,
                 seed: int | None = None):
        # ---- 动力学参数直接从配置文件读取（调用方不接触真实参数）----
        d = cfg.DEFAULT_PARAMS
        rng = np.random.default_rng(seed)
        gx, gy, alpha = sample_gravity(rng, zero=zero_gravity)
        self._gravity = (float(gx), float(gy))
        self._gravity_alpha_deg = float(alpha)
        self._params = Params(mb=d["mb"], Ib=d["Ib"], Pbx=d["Pbx"], Pby=d["Pby"],
                              ms=d["ms"], Is=d["Is"], Psx=d["Psx"], Psy=d["Psy"],
                              Dx=d["Dx"], Dy=d["Dy"], gx=gx, gy=gy,
                              fbc=d["fbc"], fbv=d["fbv"], fsc=d["fsc"], fsv=d["fsv"],
                              lambda_=d["lambda_"])
        self._dt = float(dt)
        self._refinement = int(refinement)
        self._sim = Simulator(self._params, self._dt, self._refinement)
        # 初始状态：角度与角速度均为 0
        self._sim.set_state(State(theta_b=0.0, dtheta_b=0.0,
                                  theta_s=0.0, dtheta_s=0.0))

    # ---------------- 对外操作接口 ----------------
    def state(self) -> State:
        """立即获取当前状态。"""
        return self._sim.state

    def step(self, Tb: float, Ts: float) -> State:
        """输入两个控制力矩，推进一个 dt，返回推进后的状态。

        基座状态恒为 0，因此这里固定传 theta_c = dtheta_c = ddtheta_c = 0。
        """
        return self._sim.step(float(Tb), float(Ts), 0.0, 0.0, 0.0)

    # ---------------- 只读信息（构造后不可修改） ----------------
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
        """给定状态下的静态重力矩 (G1, G2)；不给状态则用当前状态。

        静止时 Q = G，所以这就是"扛住重力"所需的前馈力矩。放在环境里是为了让
        采集脚本在做控制时不必接触真实参数。
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
        return G1, G2

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
