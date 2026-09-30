"""真实硬件环境封装：与 SimEnv 同接口，但状态来自真实通信、力矩真的发给 MCU。

对外接口与 SimEnv **完全一致**（duck-typing），因此
`generate_id_dataset.py` / `gen_friction_sweep.py` 只换构造那一行即可复用：

    state()          -> State    立即获取当前**缓存**状态（不推进时间）
    step(Tb, Ts)     -> State    下发力矩，等到下一帧边界，再采样并返回新状态

只读信息
--------
    gravity          -> (gx, gy)  本会话开始时的等效重力（由严格反解得到）
    gravity_alpha_deg             对应的摆平面倾角 [°]
    noise            -> False     真实环境**不加任何噪声**
    sigma            -> (0,0,0)
    dt                           控制周期 [s]
    refinement                   回放用（环境本身不做积分）；与 cfg.REFINEMENT 一致

与 SimEnv 的关键差异
--------------------
1. **不加噪声**：状态就是通信解出来的值；`noise=False`、`sigma=(0,0,0)`。
   采集脚本里"控回初值"的收敛容差会退化成 `cfg.REPOS_TOL_RAD`。
2. **精确帧控制**：用 `time.perf_counter_ns()` 做节拍基准，`step()` 末尾忙等到
   下一帧边界 `t0 + k·dt`。等待策略是"远则 sleep、近则自旋"（默认最后 300µs 自旋），
   既避免纯 `sleep` 的毫秒级抖动，也不长时间占满 CPU。落后于节拍时**重新对齐**
   并计入 `late_count`，不做补偿性连跑（否则会突然发出一串密帧）。
3. **状态缓存**：`state()` 返回上一次采样结果，同一帧内多次调用完全一致
   （与 SimEnv 语义一致，控制律里一帧读两次不会看到不同样本）。
4. **安全**：
   * 每个 `step` 的力矩都会先按 `tau_b_max/tau_s_max` 硬限幅再下发；
   * `close()` 会先发一帧**零力矩**再关通信，避免最后一条力矩被一直保持；
   * 连续 `max_data_gap_frames` 帧收不到有效数据则抛 `RuntimeError`（默认 100 帧 = 1s），
     避免拿过期反馈继续闭环。

用法::

    with RealEnv(dt=0.01) as env:
        st = env.state()
        st = env.step(0.2, -0.1)
"""

from __future__ import annotations

import math
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sim_config as cfg                                              # noqa: E402
from capi import (ImuLocation, McuSendPacket, RobotCommunication,     # noqa: E402
                  State, YawMode)


class RealEnv:
    """真实双级 yaw 环境（状态来自严格反解包，力矩经 MCU 下发；不加噪声）。"""

    def __init__(self,
                 dt: float = cfg.DT,
                 imu_location: ImuLocation = ImuLocation.ON_HEAD,
                 linear_params=None,
                 tau_b_max: float = cfg.TAU_B_MAX,
                 tau_s_max: float = cfg.TAU_S_MAX,
                 spin_us: float = 300.0,
                 auto_aim_enable: int = 1,
                 fire: int = 0,
                 hold_pitch: bool = True,
                 ready_timeout_s: float = 5.0,
                 gravity_settle_s: float = 0.3,
                 max_data_gap_frames: int = 100,
                 nominal_params: dict | None = None):
        """
        :param dt:               控制周期 [s]（= 帧控制节拍、数据集里的 dt）
        :param imu_location:     IMU 安装位置（构型），决定严格反解的运动学链
        :param linear_params:    MCU 线性映射标定参数；None = 用 C++ 侧标定默认值
        :param tau_b_max/tau_s_max: 下发前的力矩硬限幅 [N·m]
        :param spin_us:          帧末自旋等待的时长 [µs]（越大越准、越费 CPU）
        :param auto_aim_enable:  发送包里的自瞄总开关
        :param fire:             发送包里的火控位
        :param hold_pitch:       True 时把当前实测 pitch 角回填为 pitch 目标（不扰动 pitch）
        :param ready_timeout_s:  等待 MCU + IMU 首个有效样本的超时 [s]
        :param gravity_settle_s: 会话开始时对反解重力取平均的时长 [s]（0 = 只用首样本）
        :param max_data_gap_frames: 连续无有效数据的容忍帧数，超过抛异常（<=0 关闭）
        :param nominal_params:   算重力前馈用的标称动力学参数；None = cfg.DEFAULT_PARAMS
        """
        if not (dt > 0.0):
            raise ValueError(f"dt 必须 > 0，收到 {dt}")
        self._dt = float(dt)
        self._dt_ns = int(round(self._dt * 1e9))
        self._spin_ns = int(round(max(0.0, spin_us) * 1e3))
        self._tau_b_max = float(tau_b_max)
        self._tau_s_max = float(tau_s_max)
        self._auto_aim_enable = int(auto_aim_enable)
        self._fire = int(fire)
        self._hold_pitch = bool(hold_pitch)
        self._max_data_gap = int(max_data_gap_frames)
        self._p = dict(cfg.DEFAULT_PARAMS if nominal_params is None else nominal_params)

        # ---- 通信（构造即启动 MCU / IMU 串口线程）----
        self._comm = RobotCommunication(imu_location, linear_params)

        # ---- 状态缓存 / 帧节拍 ----
        self._state = State(theta_b=0.0, dtheta_b=0.0, theta_s=0.0, dtheta_s=0.0)
        self._pose = None
        self._pitch_target = 0.0
        self._next_ns = time.perf_counter_ns()
        self._frame_count = 0
        self._late_count = 0
        self._gap_count = 0
        self._periods_ns: deque[int] = deque(maxlen=200)
        self._last_boundary_ns = self._next_ns

        # ---- 等硬件就绪 → 取本会话的等效重力 ----
        self._wait_ready(ready_timeout_s)
        # 先采一次真实状态，拿到当前 pitch 角，避免首帧把 pitch 指令发成 0
        self._sample()
        self._settle_gravity(gravity_settle_s)

    # ================= 内部 =================
    def _wait_ready(self, timeout_s: float) -> None:
        """等到 MCU 与 IMU 都有过有效样本。"""
        t_end = time.perf_counter() + max(0.0, timeout_s)
        while True:
            d = self._comm.get_latest_data()
            if d.mcu_valid and d.imu_valid:
                return
            if time.perf_counter() >= t_end:
                raise RuntimeError(
                    f"RealEnv: {timeout_s:g}s 内没有同时收到 MCU 与 IMU 的有效数据 "
                    f"(mcu_valid={d.mcu_valid}, imu_valid={d.imu_valid})；"
                    "检查串口/接线/供电，或用 tcbs_pitch_calibration / test_serial 确认链路")
            time.sleep(0.01)

    def _sample(self) -> None:
        """读一次严格反解包，刷新状态缓存与 pitch 目标。"""
        sp = self._comm.get_strict_pose()
        self._pose = sp
        self._state = State(theta_b=sp.yaw_big_angle, dtheta_b=sp.big_motor_omega,
                            theta_s=sp.yaw_small_angle, dtheta_s=sp.small_motor_omega)
        if self._hold_pitch:
            self._pitch_target = float(sp.pitch_angle)

    def _packet(self, tb: float, ts: float) -> McuSendPacket:
        """构造发送包（两轴都用"仅力矩"模式，目标角/速度不参与）。"""
        return McuSendPacket(
            auto_aim_enable=self._auto_aim_enable,
            fire=self._fire,
            pitch_target_angle=self._pitch_target,
            yaw_big_mode=int(YawMode.TORQUE_ONLY),
            yaw_big_target_angle=0.0,
            yaw_big_target_velocity=0.0,
            yaw_big_torque=float(tb),
            yaw_small_mode=int(YawMode.TORQUE_ONLY),
            yaw_small_target_angle=0.0,
            yaw_small_target_velocity=0.0,
            yaw_small_torque=float(ts),
        )

    def _send(self, tb: float, ts: float) -> None:
        """把两个力矩下发给 MCU。"""
        self._comm.send_to_mcu(self._packet(tb, ts))

    def _busy_wait(self, target_ns: int) -> None:
        """等到 perf_counter_ns() >= target_ns：远则 sleep、近则自旋。"""
        while True:
            remain = target_ns - time.perf_counter_ns()
            if remain <= 0:
                return
            if remain > self._spin_ns:
                time.sleep((remain - self._spin_ns) * 1e-9)
            else:
                while time.perf_counter_ns() < target_ns:
                    pass
                return

    def _pace(self) -> None:
        """推进到下一帧边界；已落后则重新对齐（不做补偿性连跑）。"""
        self._next_ns += self._dt_ns
        now = time.perf_counter_ns()
        if self._next_ns <= now:
            self._late_count += 1
            self._next_ns = now + self._dt_ns
        else:
            self._busy_wait(self._next_ns)
        self._periods_ns.append(self._next_ns - self._last_boundary_ns)
        self._last_boundary_ns = self._next_ns

    def _check_data_alive(self) -> None:
        if self._max_data_gap <= 0:
            return
        d = self._comm.get_latest_data()
        if d.mcu_valid and d.imu_valid:
            self._gap_count = 0
            return
        self._gap_count += 1
        if self._gap_count >= self._max_data_gap:
            raise RuntimeError(
                f"RealEnv: 连续 {self._gap_count} 帧没有有效数据"
                f"(mcu={d.mcu_valid}, imu={d.imu_valid})，已中止以免用过期反馈继续闭环")

    def _settle_gravity(self, seconds: float) -> None:
        """会话开始：在若干帧上对反解出的 (gx,gy) 取平均（顺带下发零力矩）。"""
        n = max(1, int(round(seconds / self._dt)))
        self._next_ns = time.perf_counter_ns()
        self._last_boundary_ns = self._next_ns
        gx = gy = 0.0
        for _ in range(n):
            self._send(0.0, 0.0)
            self._pace()
            self._sample()
            self._check_data_alive()
            gx += float(self._pose.gx)
            gy += float(self._pose.gy)
        self._gravity = (gx / n, gy / n)
        mag = math.hypot(*self._gravity)
        ratio = min(1.0, max(-1.0, mag / cfg.GRAVITY_MAG))
        self._gravity_alpha_deg = float(math.degrees(math.asin(ratio)))

    # ================= 对外操作接口（与 SimEnv 一致） =================
    def state(self) -> State:
        """立即获取当前状态（**缓存值**，同一帧内多次调用一致；不推进时间）。"""
        return self._state

    def step(self, Tb: float, Ts: float) -> State:
        """下发 (Tb, Ts) → 等到下一帧边界 → 采样并返回新状态。

        力矩先按 tau_b_max/tau_s_max 硬限幅；两轴都用"仅力矩"模式下发。
        """
        tb = float(np.clip(float(Tb), -self._tau_b_max, self._tau_b_max))
        ts = float(np.clip(float(Ts), -self._tau_s_max, self._tau_s_max))
        self._send(tb, ts)
        self._pace()
        self._sample()
        self._check_data_alive()
        self._frame_count += 1
        return self._state

    # ================= 只读信息 =================
    @property
    def gravity(self) -> tuple[float, float]:
        """本会话开始时的等效重力 (gx, gy)（严格反解结果的平均）。只读。"""
        return self._gravity

    def gravity_now(self) -> tuple[float, float]:
        """当前这一帧反解出的 (gx, gy)（会话中底盘姿态变化时用它）。"""
        if self._pose is None:
            return self._gravity
        return float(self._pose.gx), float(self._pose.gy)

    @property
    def gravity_alpha_deg(self) -> float:
        """等效重力对应的摆平面倾角 [°]（= asin(|g|/9.81)）。只读。"""
        return self._gravity_alpha_deg

    @property
    def noise(self) -> bool:
        """真实环境不加噪声，恒为 False（供采集脚本判断收敛容差）。"""
        return False

    @property
    def sigma(self) -> tuple[float, float, float]:
        """真实环境不加噪声，恒为 (0,0,0)。"""
        return (0.0, 0.0, 0.0)

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def refinement(self) -> int:
        """回放该数据集时用的 RK4 子步数（环境本身不做积分）。"""
        return int(cfg.REFINEMENT)

    @property
    def tau_max(self) -> tuple[float, float]:
        return (self._tau_b_max, self._tau_s_max)

    # ---- 帧节拍诊断 ----
    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def late_count(self) -> int:
        """落后于节拍、被迫重新对齐的帧数。"""
        return self._late_count

    @property
    def mean_period(self) -> float:
        """最近 200 帧的平均实际周期 [s]（无样本时返回 dt）。"""
        if not self._periods_ns:
            return self._dt
        return float(np.mean(self._periods_ns)) * 1e-9

    @property
    def max_period(self) -> float:
        """最近 200 帧里的最大实际周期 [s]。"""
        if not self._periods_ns:
            return self._dt
        return float(max(self._periods_ns)) * 1e-9

    # ================= 控制器需要的模型量 =================
    def gravity_torque(self, theta_b: float | None = None,
                       theta_s: float | None = None) -> tuple[float, float]:
        """给定状态下的静态重力矩 (G1, G2)；不给状态则用当前状态。

        与 SimEnv 同一公式，但重力用**本会话开始时反解出的 (gx, gy)**（即记录进
        数据集的同一组值，保证控制前馈与辨识模型一致）；动力学参数用标称值
        （cfg.DEFAULT_PARAMS）。
        """
        st = self._state
        tb = st.theta_b if theta_b is None else float(theta_b)
        ts = st.theta_s if theta_s is None else float(theta_s)
        p = self._p
        gx, gy = self._gravity
        gb_sin = gx * (p["mb"] * p["Pbx"] + p["ms"] * p["Dx"]) + gy * (p["mb"] * p["Pby"] + p["ms"] * p["Dy"])
        gb_cos = gx * (p["mb"] * p["Pby"] + p["ms"] * p["Dy"]) - gy * (p["mb"] * p["Pbx"] + p["ms"] * p["Dx"])
        gs_sin = p["ms"] * (gx * p["Psx"] + gy * p["Psy"])
        gs_cos = p["ms"] * (gx * p["Psy"] - gy * p["Psx"])
        psi_b, psi_s = tb, tb + ts
        G2 = gs_sin * math.sin(psi_s) + gs_cos * math.cos(psi_s)
        G1 = gb_sin * math.sin(psi_b) + gb_cos * math.cos(psi_b) + G2
        return G1, G2

    # ================= 生命周期 =================
    def close(self) -> None:
        """先发一帧零力矩（避免最后一条力矩被保持），再关闭通信。"""
        comm = getattr(self, "_comm", None)
        if comm is None:
            return
        self._comm = None
        try:
            self._pitch_target = float(self._pose.pitch_angle) if self._pose else 0.0
            comm.send_to_mcu(self._packet(0.0, 0.0))
        except Exception:
            pass
        try:
            comm.stop()
        except Exception:
            pass
        try:
            comm.close()
        except Exception:
            pass

    def __enter__(self) -> "RealEnv":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (f"RealEnv(gravity=({self._gravity[0]:.4f}, {self._gravity[1]:.4f}), "
                f"alpha={self._gravity_alpha_deg:.3f}°, dt={self._dt}, "
                f"frames={self._frame_count}, late={self._late_count}, "
                f"period={self.mean_period * 1e3:.3f}ms)")
