"""libtcbs.so 的 ctypes 包装。

对应 C 接口：include/capi/capi_dm.h，实现对 src/dm 中 C++ 仿真核心的调用。

典型用法::

    from capi import Params, State, Simulator

    params = Params(
        mb=1.5, Ib=0.03, Pbx=0.18, Pby=0.0,
        ms=0.4, Is=0.008, Psx=0.10, Psy=0.0,
        Dx=0.30, Dy=0.0, gx=0.0, gy=-9.81,
        fbc=0.02, fbv=0.05, fsc=0.01, fsv=0.02, lambda_=100.0,
    )

    with Simulator(params, dt=1e-3, refinement=4) as sim:
        sim.set_state(State(theta_b=-1.5708, dtheta_b=0.0, theta_s=0.0, dtheta_s=0.0))
        st = sim.step(Tb=0.0, Ts=0.0, theta_c=0.0, dtheta_c=0.0, ddtheta_c=0.0)

单位：角度 rad，角速度 rad/s，角加速度 rad/s^2，力矩 N*m，时间 s。
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

__all__ = ["Params", "State", "Simulator", "Trajectory", "TrajectoryLoss",
           "ParamGradient", "ParamLossSpec", "PARAM_GRADIENT_NAMES", "library_path"]


# ---------------------------------------------------------------------------
# 与 include/capi/capi_dm.h 一一对应的 C 结构体
# ---------------------------------------------------------------------------
class _CParams(ctypes.Structure):
    _fields_ = [
        ("mb", ctypes.c_double),
        ("Ib", ctypes.c_double),
        ("Pbx", ctypes.c_double),
        ("Pby", ctypes.c_double),
        ("ms", ctypes.c_double),
        ("Is", ctypes.c_double),
        ("Psx", ctypes.c_double),
        ("Psy", ctypes.c_double),
        ("Dx", ctypes.c_double),
        ("Dy", ctypes.c_double),
        ("gx", ctypes.c_double),
        ("gy", ctypes.c_double),
        ("fbc", ctypes.c_double),
        ("fbv", ctypes.c_double),
        ("fsc", ctypes.c_double),
        ("fsv", ctypes.c_double),
        ("lambda_", ctypes.c_double),
        # 控制力矩通道增益：物理力矩 = kb/ks × 下发给电控的指令值
        ("kb", ctypes.c_double),
        ("ks", ctypes.c_double),
    ]


class _CState(ctypes.Structure):
    _fields_ = [
        ("theta_b", ctypes.c_double),
        ("dtheta_b", ctypes.c_double),
        ("theta_s", ctypes.c_double),
        ("dtheta_s", ctypes.c_double),
    ]


# ---------------------------------------------------------------------------
# 动态库查找与加载
# ---------------------------------------------------------------------------
def library_path() -> str:
    """返回 libtcbs 动态库路径。

    查找顺序：环境变量 TCBS_LIB -> 项目 build/ 目录下的平台对应文件名。
    """
    here = Path(__file__).resolve().parent
    candidates = []
    env = os.environ.get("TCBS_LIB")
    if env:
        candidates.append(Path(env))
    build_dir = here.parent.parent / "build"
    candidates += [
        build_dir / "libtcbs.so",
        build_dir / "libtcbs.dylib",
        build_dir / "tcbs.dll",
    ]
    for cand in candidates:
        if cand.is_file():
            return str(cand)
    raise FileNotFoundError(
        "找不到 libtcbs 动态库，请先在项目根目录执行 ./build.sh 编译。\n"
        f"已查找: {', '.join(str(c) for c in candidates)}"
    )


def _load_library() -> ctypes.CDLL:
    lib = ctypes.CDLL(library_path())

    lib.tcbs_create.restype = ctypes.c_void_p
    lib.tcbs_create.argtypes = [ctypes.POINTER(_CParams), ctypes.c_double, ctypes.c_int]

    lib.tcbs_destroy.restype = None
    lib.tcbs_destroy.argtypes = [ctypes.c_void_p]

    lib.tcbs_last_error.restype = ctypes.c_char_p
    lib.tcbs_last_error.argtypes = []

    lib.tcbs_set_state.restype = None
    lib.tcbs_set_state.argtypes = [ctypes.c_void_p, _CState]

    lib.tcbs_set_generalized.restype = None
    lib.tcbs_set_generalized.argtypes = [
        ctypes.c_void_p,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
    ]

    lib.tcbs_set_theta_b.restype = None
    lib.tcbs_set_theta_b.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]

    lib.tcbs_set_theta_s.restype = None
    lib.tcbs_set_theta_s.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]

    lib.tcbs_get_state.restype = None
    lib.tcbs_get_state.argtypes = [ctypes.c_void_p, ctypes.POINTER(_CState)]

    lib.tcbs_get_dt.restype = ctypes.c_double
    lib.tcbs_get_dt.argtypes = [ctypes.c_void_p]

    lib.tcbs_get_refinement.restype = ctypes.c_int
    lib.tcbs_get_refinement.argtypes = [ctypes.c_void_p]

    lib.tcbs_get_params.restype = None
    lib.tcbs_get_params.argtypes = [ctypes.c_void_p, ctypes.POINTER(_CParams)]

    lib.tcbs_step.restype = None
    lib.tcbs_step.argtypes = [
        ctypes.c_void_p,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.POINTER(_CState),
    ]

    # ---- 轨迹梯度接口 ----
    lib.tcbs_trajectory_create.restype = ctypes.c_void_p
    lib.tcbs_trajectory_create.argtypes = [ctypes.POINTER(_CParams), ctypes.c_int]

    lib.tcbs_trajectory_destroy.restype = None
    lib.tcbs_trajectory_destroy.argtypes = [ctypes.c_void_p]

    _dbl_p = ctypes.POINTER(ctypes.c_double)

    lib.tcbs_trajectory_loss.restype = ctypes.c_double
    lib.tcbs_trajectory_loss.argtypes = [
        ctypes.c_void_p,                                  # t
        ctypes.c_double, ctypes.c_double, ctypes.c_double,  # theta_c0, dtheta_c, ddtheta_c
        ctypes.c_double,                                  # dt
        ctypes.c_int,                                     # refinement
        ctypes.c_size_t,                                  # num_steps
        _dbl_p,                                           # tau
        ctypes.c_double, ctypes.c_double,                 # tau_b_fixed, tau_s_fixed
        ctypes.POINTER(_CState),                          # x0
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,  # w1..w4
        ctypes.c_double, ctypes.c_double,                 # w5, w6
        _dbl_p, _dbl_p, _dbl_p, _dbl_p,                   # targets
        _dbl_p, _dbl_p, _dbl_p, _dbl_p,                   # outputs
    ]

    lib.tcbs_trajectory_gradient.restype = ctypes.c_double
    lib.tcbs_trajectory_gradient.argtypes = [
        ctypes.c_void_p,                                  # t
        ctypes.c_double, ctypes.c_double, ctypes.c_double,  # theta_c0, dtheta_c, ddtheta_c
        ctypes.c_double,                                  # dt
        ctypes.c_int,                                     # refinement
        ctypes.c_size_t,                                  # num_steps
        _dbl_p,                                           # tau
        ctypes.POINTER(_CState),                          # x0
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,  # w1..w4
        ctypes.c_double, ctypes.c_double,                 # w5, w6
        _dbl_p, _dbl_p, _dbl_p, _dbl_p,                   # targets
        _dbl_p,                                           # grad_tau
        ctypes.POINTER(_CState),                          # out_final_state
    ]

    # ---- 参数辨识模式 ----
    lib.tcbs_param_gradient_count.restype = ctypes.c_int
    lib.tcbs_param_gradient_count.argtypes = []

    lib.tcbs_param_gradient_name.restype = ctypes.c_char_p
    lib.tcbs_param_gradient_name.argtypes = [ctypes.c_int]

    lib.tcbs_param_gradient_create.restype = ctypes.c_void_p
    lib.tcbs_param_gradient_create.argtypes = [ctypes.POINTER(_CParams), ctypes.c_int]

    lib.tcbs_param_gradient_destroy.restype = None
    lib.tcbs_param_gradient_destroy.argtypes = [ctypes.c_void_p]

    lib.tcbs_param_gradient_set_params.restype = ctypes.c_int
    lib.tcbs_param_gradient_set_params.argtypes = [ctypes.c_void_p,
                                                    ctypes.POINTER(_CParams)]

    lib.tcbs_param_gradient_get_params.restype = None
    lib.tcbs_param_gradient_get_params.argtypes = [ctypes.c_void_p,
                                                    ctypes.POINTER(_CParams)]

    lib.tcbs_param_gradient_loss.restype = ctypes.c_double
    lib.tcbs_param_gradient_loss.argtypes = [
        ctypes.c_void_p,
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_int, ctypes.c_size_t,
        _dbl_p, ctypes.POINTER(_CState),
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        _dbl_p, _dbl_p, _dbl_p, _dbl_p,
        _dbl_p, _dbl_p, _dbl_p, _dbl_p,
    ]

    lib.tcbs_param_gradient_run.restype = ctypes.c_double
    lib.tcbs_param_gradient_run.argtypes = [
        ctypes.c_void_p,
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_int, ctypes.c_size_t,
        _dbl_p, ctypes.POINTER(_CState),
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        _dbl_p, _dbl_p, _dbl_p, _dbl_p,
        _dbl_p, ctypes.POINTER(_CState),
    ]

    # ---- 批量（SoA）参数梯度 ----
    lib.tcbs_param_gradient_batch_create.restype = ctypes.c_void_p
    lib.tcbs_param_gradient_batch_create.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_size_t, ctypes.c_int, ctypes.c_int]

    lib.tcbs_param_gradient_batch_destroy.restype = None
    lib.tcbs_param_gradient_batch_destroy.argtypes = [ctypes.c_void_p]

    lib.tcbs_param_gradient_batch_threads.restype = ctypes.c_int
    lib.tcbs_param_gradient_batch_threads.argtypes = [ctypes.c_void_p]

    lib.tcbs_param_gradient_batch_lanes.restype = ctypes.c_int
    lib.tcbs_param_gradient_batch_lanes.argtypes = [ctypes.c_void_p]

    lib.tcbs_param_gradient_batch_run.restype = ctypes.c_int
    lib.tcbs_param_gradient_batch_run.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int, ctypes.c_size_t, ctypes.c_double,
        _dbl_p, _dbl_p, _dbl_p, _dbl_p,
        _dbl_p, _dbl_p, _dbl_p, _dbl_p,
        _dbl_p, _dbl_p, _dbl_p,
    ]

    return lib


_lib = _load_library()


def _last_error() -> str:
    raw = _lib.tcbs_last_error()
    return raw.decode("utf-8", errors="replace") if raw else "unknown error"


# ---------------------------------------------------------------------------
# Python 侧数据结构：全部字段必填，无默认值
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Params:
    """系统参数。所有字段必填，不提供默认值。"""

    # 连杆 b
    mb: float      # 质量 [kg]
    Ib: float      # 绕质心转动惯量 [kg*m^2]
    Pbx: float     # 质心局部坐标 x [m]
    Pby: float     # 质心局部坐标 y [m]

    # 连杆 s
    ms: float      # 质量 [kg]
    Is: float      # 绕质心转动惯量 [kg*m^2]
    Psx: float     # 质心局部坐标 x [m]
    Psy: float     # 质心局部坐标 y [m]

    # 关节偏移与重力场
    Dx: float      # 关节偏移 x [m]
    Dy: float      # 关节偏移 y [m]
    gx: float      # 重力场加速度 x [m/s^2]
    gy: float      # 重力场加速度 y [m/s^2]

    # 摩擦
    fbc: float
    fbv: float
    fsc: float
    fsv: float
    lambda_: float

    # 控制力矩通道增益：物理力矩 = kb/ks × 下发给电控的指令值。
    # 带默认值 1.0 = 旧行为（指令值即物理力矩）；凡是描述真机/真机数据的路径
    # 都必须显式给出（Sentry1 实测 kb = 4）。
    kb: float = 1.0
    ks: float = 1.0

    def _to_c(self) -> _CParams:
        return _CParams(
            mb=self.mb, Ib=self.Ib, Pbx=self.Pbx, Pby=self.Pby,
            ms=self.ms, Is=self.Is, Psx=self.Psx, Psy=self.Psy,
            Dx=self.Dx, Dy=self.Dy, gx=self.gx, gy=self.gy,
            fbc=self.fbc, fbv=self.fbv, fsc=self.fsc, fsv=self.fsv,
            lambda_=self.lambda_, kb=self.kb, ks=self.ks,
        )

    @classmethod
    def _from_c(cls, c: _CParams) -> "Params":
        return cls(
            mb=c.mb, Ib=c.Ib, Pbx=c.Pbx, Pby=c.Pby,
            ms=c.ms, Is=c.Is, Psx=c.Psx, Psy=c.Psy,
            Dx=c.Dx, Dy=c.Dy, gx=c.gx, gy=c.gy,
            fbc=c.fbc, fbv=c.fbv, fsc=c.fsc, fsv=c.fsv,
            lambda_=c.lambda_, kb=c.kb, ks=c.ks,
        )


@dataclass(frozen=True)
class State:
    """两个广义坐标的位置与速度。所有字段必填。"""

    theta_b: float
    dtheta_b: float
    theta_s: float
    dtheta_s: float

    def _to_c(self) -> _CState:
        return _CState(
            theta_b=self.theta_b,
            dtheta_b=self.dtheta_b,
            theta_s=self.theta_s,
            dtheta_s=self.dtheta_s,
        )

    @classmethod
    def _from_c(cls, c: _CState) -> "State":
        return cls(
            theta_b=c.theta_b,
            dtheta_b=c.dtheta_b,
            theta_s=c.theta_s,
            dtheta_s=c.dtheta_s,
        )


# ---------------------------------------------------------------------------
# 仿真器
# ---------------------------------------------------------------------------
class Simulator:
    """C++ 仿真器的 Python 句柄。

    :param params:     全部系统参数
    :param dt:         单步仿真时间 [s]，必须 > 0
    :param refinement: 细化倍数（每个 dt 内的 RK4 子步数），必须 >= 1
    """

    def __init__(self, params: Params, dt: float, refinement: int):
        c_params = params._to_c()
        handle = _lib.tcbs_create(ctypes.byref(c_params), float(dt), int(refinement))
        if not handle:
            raise RuntimeError(f"tcbs_create 失败: {_last_error()}")
        self._handle = handle
        self._dt = float(dt)
        self._refinement = int(refinement)

    # -- 生命周期 ---------------------------------------------------------
    def close(self) -> None:
        if getattr(self, "_handle", None):
            _lib.tcbs_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self) -> "Simulator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _require_handle(self) -> ctypes.c_void_p:
        if not getattr(self, "_handle", None):
            raise RuntimeError("Simulator 已关闭")
        return self._handle

    # -- 状态设置 ---------------------------------------------------------
    def set_state(self, state: State) -> None:
        """直接设置当前两个广义坐标的位置与速度。"""
        _lib.tcbs_set_state(self._require_handle(), state._to_c())

    def set_generalized(
        self, theta_b: float, dtheta_b: float, theta_s: float, dtheta_s: float
    ) -> None:
        """直接设置当前两个广义坐标的位置与速度（分量形式）。"""
        _lib.tcbs_set_generalized(
            self._require_handle(), float(theta_b), float(dtheta_b),
            float(theta_s), float(dtheta_s),
        )

    def set_theta_b(self, theta_b: float, dtheta_b: float) -> None:
        """单独设置广义坐标 1 的位置与速度。"""
        _lib.tcbs_set_theta_b(self._require_handle(), float(theta_b), float(dtheta_b))

    def set_theta_s(self, theta_s: float, dtheta_s: float) -> None:
        """单独设置广义坐标 2 的位置与速度。"""
        _lib.tcbs_set_theta_s(self._require_handle(), float(theta_s), float(dtheta_s))

    # -- 读取 -------------------------------------------------------------
    @property
    def state(self) -> State:
        out = _CState()
        _lib.tcbs_get_state(self._require_handle(), ctypes.byref(out))
        return State._from_c(out)

    @property
    def dt(self) -> float:
        return _lib.tcbs_get_dt(self._require_handle())

    @property
    def refinement(self) -> int:
        return _lib.tcbs_get_refinement(self._require_handle())

    @property
    def params(self) -> Params:
        out = _CParams()
        _lib.tcbs_get_params(self._require_handle(), ctypes.byref(out))
        return Params._from_c(out)

    @property
    def substep_dt(self) -> float:
        """RK4 子步长度 dt / refinement。"""
        return self.dt / self.refinement

    # -- 仿真 -------------------------------------------------------------
    def step(
        self,
        Tb: float,
        Ts: float,
        theta_c: float,
        dtheta_c: float,
        ddtheta_c: float,
    ) -> State:
        """推进一个 dt，返回演化后的两个广义坐标及其速度。

        :param Tb:        关节 b 驱动力矩 [N*m]
        :param Ts:        关节 s 驱动力矩 [N*m]
        :param theta_c:   本步起始时刻的基座角度 [rad]
        :param dtheta_c:  本步起始时刻的基座角速度 [rad/s]
        :param ddtheta_c: 本步起始时刻的基座角加速度 [rad/s^2]
        """
        out = _CState()
        _lib.tcbs_step(
            self._require_handle(),
            float(Tb), float(Ts), float(theta_c), float(dtheta_c), float(ddtheta_c),
            ctypes.byref(out),
        )
        return State._from_c(out)

    def __repr__(self) -> str:
        return (
            f"Simulator(dt={self._dt:g}, refinement={self._refinement}, "
            f"substep_dt={self._dt / self._refinement:g})"
        )


# ---------------------------------------------------------------------------
# 有限多步轨迹 + 损失对每一步力矩的解析梯度
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrajectoryLoss:
    """六项加权的损失定义（时间均值形式）。

        L = (1/2K) * sum_k [ w_psi_b  * (psi_b[k]  - target_psi_b[k])^2
                           + w_psi_s  * (psi_s[k]  - target_psi_s[k])^2
                           + w_dpsi_b * (dpsi_b[k] - target_dpsi_b[k])^2
                           + w_dpsi_s * (dpsi_s[k] - target_dpsi_s[k])^2
                           + w_tau_b  * tau_b[k]^2
                           + w_tau_s  * tau_s[k]^2 ]

    其中 psi_b = theta_c + theta_b，psi_s = theta_c + theta_b + theta_s，
    dpsi_b = dtheta_c + dtheta_b，dpsi_s = dtheta_c + dtheta_b + dtheta_s。

    目标序列为 None 时该项退化为惩罚幅值（目标恒为 0）。
    序列长度必须与 num_steps 一致。
    """

    w_psi_b: float = 0.0
    w_psi_s: float = 0.0
    w_dpsi_b: float = 0.0
    w_dpsi_s: float = 0.0
    w_tau_b: float = 0.0
    w_tau_s: float = 0.0

    target_psi_b: Sequence[float] | None = None
    target_psi_s: Sequence[float] | None = None
    target_dpsi_b: Sequence[float] | None = None
    target_dpsi_s: Sequence[float] | None = None


def _dbl_array(seq) -> "np.ndarray":
    """把序列（或 None）转成连续 double 数组。"""
    if seq is None:
        return None
    arr = np.ascontiguousarray(seq, dtype=np.float64)
    return arr


def _dbl_ptr(arr):
    """把 numpy double 数组转成 ctypes 指针；None -> NULL。"""
    if arr is None:
        return ctypes.cast(None, ctypes.POINTER(ctypes.c_double))
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


class Trajectory:
    """有限多步轨迹求解器：一次调用完成正向仿真与反向伴随。

    缓冲在句柄内复用，因此对同一 batch 反复调用（例如优化迭代）不会有额外分配。

    :param params: 全部系统参数

    用法::

        traj = Trajectory(PARAMS)
        loss = traj.loss(theta_c0=0.0, dtheta_c=0.0, ddtheta_c=0.0,
                         dt=1e-3, num_steps=200, tau=tau, x0=x0, spec=spec)
        loss, grad = traj.gradient(...)      # grad 形状 (K, 2)
    """

    def __init__(self, params: Params, refinement: int = 4):
        c_params = params._to_c()
        handle = _lib.tcbs_trajectory_create(ctypes.byref(c_params), int(refinement))
        if not handle:
            raise RuntimeError(f"tcbs_trajectory_create 失败: {_last_error()}")
        self._handle = handle
        self._params = params
        self._refinement = int(refinement)

    # -- 生命周期 ---------------------------------------------------------
    def close(self) -> None:
        if getattr(self, "_handle", None):
            _lib.tcbs_trajectory_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self) -> "Trajectory":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _require_handle(self) -> ctypes.c_void_p:
        if not getattr(self, "_handle", None):
            raise RuntimeError("Trajectory 已关闭")
        return self._handle

    @property
    def params(self) -> Params:
        return self._params

    # -- 接口 -------------------------------------------------------------
    def loss(
        self,
        theta_c0: float,
        dtheta_c: float,
        ddtheta_c: float,
        dt: float,
        tau: "np.ndarray",
        x0: State,
        spec: TrajectoryLoss,
        return_sequences: bool = False,
    ):
        """只做正向仿真并返回损失值。

        :param tau: 形状 (K, 2) 的力矩序列（tau[k] = [Tb, Ts]），或形状 (2,) 的常值力矩
        :param return_sequences: True 时返回 (loss, psi_b, psi_s, dpsi_b, dpsi_s)
        """
        tau_arr, num_steps = self._prepare_tau(tau)
        x0_c = x0._to_c()

        outs = []
        for name in ("target_psi_b", "target_psi_s", "target_dpsi_b", "target_dpsi_s"):
            outs.append(_dbl_array(getattr(spec, name)))
        self._check_target_lengths(outs, num_steps)

        if return_sequences:
            buf = [np.zeros(num_steps, dtype=np.float64) for _ in range(4)]
            buf_ptr = [b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for b in buf]
        else:
            buf = None
            buf_ptr = [ctypes.cast(None, ctypes.POINTER(ctypes.c_double))] * 4

        value = _lib.tcbs_trajectory_loss(
            self._require_handle(),
            float(theta_c0), float(dtheta_c), float(ddtheta_c), float(dt),
            ctypes.c_int(self._refinement),
            ctypes.c_size_t(num_steps),
            _dbl_ptr(tau_arr),
            0.0, 0.0,
            ctypes.byref(x0_c),
            float(spec.w_psi_b), float(spec.w_psi_s),
            float(spec.w_dpsi_b), float(spec.w_dpsi_s),
            float(spec.w_tau_b), float(spec.w_tau_s),
            _dbl_ptr(outs[0]), _dbl_ptr(outs[1]), _dbl_ptr(outs[2]), _dbl_ptr(outs[3]),
            buf_ptr[0], buf_ptr[1], buf_ptr[2], buf_ptr[3],
        )
        if value < 0.0:
            raise RuntimeError(f"tcbs_trajectory_loss 失败: {_last_error()}")
        if return_sequences:
            return (value, buf[0], buf[1], buf[2], buf[3])
        return value

    def gradient(
        self,
        theta_c0: float,
        dtheta_c: float,
        ddtheta_c: float,
        dt: float,
        tau: "np.ndarray",
        x0: State,
        spec: TrajectoryLoss,
        return_final_state: bool = False,
    ):
        """正向 + 反向伴随，返回 (loss, grad)。

        :param tau: 形状 (K, 2) 的力矩序列
        :param return_final_state: True 时额外返回最后一步之后的状态
        :return: (loss, grad)，grad 形状 (K, 2)，grad[k] = [dL/dTb_k, dL/dTs_k]
        """
        tau_arr, num_steps = self._prepare_tau(tau)
        if tau_arr.ndim != 2:
            raise ValueError("gradient 需要形状 (K, 2) 的力矩序列")
        x0_c = x0._to_c()

        targets = [_dbl_array(getattr(spec, n)) for n in
                   ("target_psi_b", "target_psi_s", "target_dpsi_b", "target_dpsi_s")]
        self._check_target_lengths(targets, num_steps)

        grad = np.zeros((num_steps, 2), dtype=np.float64)
        final_c = _CState()

        value = _lib.tcbs_trajectory_gradient(
            self._require_handle(),
            float(theta_c0), float(dtheta_c), float(ddtheta_c), float(dt),
            ctypes.c_int(self._refinement),
            ctypes.c_size_t(num_steps),
            _dbl_ptr(tau_arr),
            ctypes.byref(x0_c),
            float(spec.w_psi_b), float(spec.w_psi_s),
            float(spec.w_dpsi_b), float(spec.w_dpsi_s),
            float(spec.w_tau_b), float(spec.w_tau_s),
            _dbl_ptr(targets[0]), _dbl_ptr(targets[1]),
            _dbl_ptr(targets[2]), _dbl_ptr(targets[3]),
            _dbl_ptr(grad),
            ctypes.byref(final_c),
        )
        if value < 0.0:
            raise RuntimeError(f"tcbs_trajectory_gradient 失败: {_last_error()}")
        if return_final_state:
            return value, grad, State._from_c(final_c)
        return value, grad

    # -- 内部 -------------------------------------------------------------
    @staticmethod
    def _prepare_tau(tau):
        """返回 (连续数组, num_steps)。支持 (K,2) 序列；常值力矩请用 loss() 的 fixed 参数。"""
        arr = np.ascontiguousarray(tau, dtype=np.float64)
        if arr.ndim == 1:
            if arr.shape[0] != 2:
                raise ValueError("常值力矩必须是长度 2 的序列 [Tb, Ts]")
            raise ValueError(
                "loss() 的常值力矩请传 (K,2) 的序列；或用 tau_b_fixed/tau_s_fixed 参数"
            )
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("tau 形状必须是 (K, 2)")
        return arr, arr.shape[0]

    @staticmethod
    def _check_target_lengths(targets, num_steps):
        for i, t in enumerate(targets):
            if t is not None and t.size != num_steps:
                raise ValueError(
                    f"目标序列 {i} 长度为 {t.size}，应为 num_steps={num_steps}"
                )

    def __repr__(self) -> str:
        return f"Trajectory({self._params!r})"



# ---------------------------------------------------------------------------
# 系统参数辨识模式：对 14 个动力学参数求损失的解析梯度（gx/gy 是已知输入，不辨识）
# ---------------------------------------------------------------------------
PARAM_GRADIENT_NAMES = tuple(
    _lib.tcbs_param_gradient_name(i).decode()
    for i in range(_lib.tcbs_param_gradient_count())
)
"""参与辨识的参数名，顺序与 C 接口严格一致（含 kb / ks）。"""

NPARAM = len(PARAM_GRADIENT_NAMES)
"""被辨识参数个数（= C 接口的 tcbs_param_gradient_count()，当前为 16）。"""


@dataclass(frozen=True)
class ParamLossSpec:
    """全局量 psi 的四项时间均值加权损失。

        L = (1/2K) sum_k [ w_psi_b  (psi_b  - psi_b*)^2
                         + w_psi_s  (psi_s  - psi_s*)^2
                         + w_dpsi_b (dpsi_b - dpsi_b*)^2
                         + w_dpsi_s (dpsi_s - dpsi_s*)^2 ]

    其中 psi_b = theta_c + theta_b，psi_s = theta_c + theta_b + theta_s，
    dpsi_b = dtheta_c + dtheta_b，dpsi_s = dtheta_c + dtheta_b + dtheta_s。
    目标序列为 None 时该项退化为惩罚幅值。序列长度必须等于 num_steps。
    """

    w_psi_b: float = 0.0
    w_psi_s: float = 0.0
    w_dpsi_b: float = 0.0
    w_dpsi_s: float = 0.0
    target_psi_b: Sequence[float] | None = None
    target_psi_s: Sequence[float] | None = None
    target_dpsi_b: Sequence[float] | None = None
    target_dpsi_s: Sequence[float] | None = None


class ParamGradient:
    """参数辨识求解器：力矩序列为已知输入，对动力学参数求损失梯度。

    与 Trajectory 完全独立（不同句柄、不同函数），互不影响。
    内部用前向灵敏度，不需要反向扫描，也不保存历史状态。

    :param params: 全部系统参数（作为求导点，也是正演所用参数）

    用法::

        pg = ParamGradient(PARAMS)
        loss, grad = pg.gradient(theta_c0, dtheta_c, ddtheta_c, dt, tau, x0, spec)
        # grad 形状 (16,)，顺序见 PARAM_GRADIENT_NAMES（含 kb/ks）
    """

    def __init__(self, params: Params, refinement: int = 4):
        c_params = params._to_c()
        handle = _lib.tcbs_param_gradient_create(ctypes.byref(c_params), int(refinement))
        if not handle:
            raise RuntimeError(f"tcbs_param_gradient_create 失败: {_last_error()}")
        self._handle = handle
        self._params = params
        self._refinement = int(refinement)

    def close(self) -> None:
        if getattr(self, "_handle", None):
            _lib.tcbs_param_gradient_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self) -> "ParamGradient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _require_handle(self) -> ctypes.c_void_p:
        if not getattr(self, "_handle", None):
            raise RuntimeError("ParamGradient 已关闭")
        return self._handle

    @property
    def params(self) -> Params:
        """当前求导点参数（从 C 句柄读回，反映 set_params 的结果）。"""
        c = _CParams()
        _lib.tcbs_param_gradient_get_params(self._require_handle(), ctypes.byref(c))
        return Params._from_c(c)

    def set_params(self, params: Params) -> None:
        """就地更新求导点参数（不重新分配缓冲）。

        **优化循环里每次更新参数后都必须调用**，否则 loss/梯度会一直停留
        在创建时的参数点上，表现为"loss 不下降、且与学习率无关"。
        """
        c_params = params._to_c()
        ok = _lib.tcbs_param_gradient_set_params(self._require_handle(),
                                                  ctypes.byref(c_params))
        if not ok:
            raise RuntimeError(f"tcbs_param_gradient_set_params 失败: {_last_error()}")
        self._params = params

    @staticmethod
    def _prepare(tau, num_steps=None):
        arr = np.ascontiguousarray(tau, dtype=np.float64)
        if arr.ndim == 1 and arr.shape[0] == 2 and num_steps is not None:
            arr = np.tile(arr, (num_steps, 1))          # 常值力矩展开
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("tau 形状必须是 (K, 2) 或长度 2 的常值序列")
        return arr, arr.shape[0]

    @staticmethod
    def _targets(spec, K):
        out = []
        for name in ("target_psi_b", "target_psi_s", "target_dpsi_b", "target_dpsi_s"):
            seq = getattr(spec, name)
            a = _dbl_array(seq)
            if a is not None and a.size != K:
                raise ValueError(f"{name} 长度为 {a.size}，应为 K={K}")
            out.append(a)
        return out

    def loss(self, theta_c0, dtheta_c, ddtheta_c, dt, tau, x0, spec,
             return_sequences: bool = False):
        """仅前向：返回损失值；return_sequences=True 时额外返回四个全局量序列。"""
        tau_arr, K = self._prepare(tau)
        t = self._targets(spec, K)
        x0_c = x0._to_c()
        if return_sequences:
            buf = [np.zeros(K, dtype=np.float64) for _ in range(4)]
            bufp = [b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for b in buf]
        else:
            buf = None
            bufp = [ctypes.cast(None, ctypes.POINTER(ctypes.c_double))] * 4
        val = _lib.tcbs_param_gradient_loss(
            self._require_handle(),
            float(theta_c0), float(dtheta_c), float(ddtheta_c), float(dt),
            ctypes.c_int(self._refinement),
            ctypes.c_size_t(K), _dbl_ptr(tau_arr), ctypes.byref(x0_c),
            float(spec.w_psi_b), float(spec.w_psi_s),
            float(spec.w_dpsi_b), float(spec.w_dpsi_s),
            _dbl_ptr(t[0]), _dbl_ptr(t[1]), _dbl_ptr(t[2]), _dbl_ptr(t[3]),
            bufp[0], bufp[1], bufp[2], bufp[3],
        )
        if val < 0.0:
            raise RuntimeError(f"tcbs_param_gradient_loss 失败: {_last_error()}")
        return (val, *buf) if return_sequences else val

    def gradient(self, theta_c0, dtheta_c, ddtheta_c, dt, tau, x0, spec,
                 return_final_state: bool = False):
        """正向 + 前向参数灵敏度，返回 (loss, grad)，grad 形状 (16,)。"""
        tau_arr, K = self._prepare(tau)
        t = self._targets(spec, K)
        x0_c = x0._to_c()
        grad = np.zeros(len(PARAM_GRADIENT_NAMES), dtype=np.float64)
        final_c = _CState()
        val = _lib.tcbs_param_gradient_run(
            self._require_handle(),
            float(theta_c0), float(dtheta_c), float(ddtheta_c), float(dt),
            ctypes.c_int(self._refinement),
            ctypes.c_size_t(K), _dbl_ptr(tau_arr), ctypes.byref(x0_c),
            float(spec.w_psi_b), float(spec.w_psi_s),
            float(spec.w_dpsi_b), float(spec.w_dpsi_s),
            _dbl_ptr(t[0]), _dbl_ptr(t[1]), _dbl_ptr(t[2]), _dbl_ptr(t[3]),
            _dbl_ptr(grad), ctypes.byref(final_c),
        )
        if val < 0.0:
            raise RuntimeError(f"tcbs_param_gradient_run 失败: {_last_error()}")
        if return_final_state:
            return val, grad, State._from_c(final_c)
        return val, grad

    def __repr__(self) -> str:
        return f"ParamGradient({self._params!r})"


class ParamGradientBatch:
    """批量参数梯度：一次对 **一批** 序列求 (loss[b], dL/dp[b])。

    数学与 :class:`ParamGradient` 完全一致，但一次处理 B 条互相独立的序列，
    并且所有数组都是 **SoA**：样本维 b 最连续（不同样本的同一个量相邻），
    底层用 SIMD + 多线程并行。

    与 :class:`ParamGradient` 的区别：``ParamGradient`` 的求导点参数写在句柄里
    （``set_params``），而这里参数是**每次调用一起传进来的**（因为同一批里
    每条序列的 gx/gy 可以不同）。

    :param refinement:   每个 dt 内的 RK4 子步数
    :param max_batch:    允许的最大批大小 B（缓冲按此分配）
    :param max_num_steps: 允许的最大步数 K
    :param num_threads:  <= 0 表示用硬件并发
    :param lanes:        每个 SIMD 分块处理的样本数上限

    用法::

        pgb = ParamGradientBatch(refinement=16, max_batch=120, max_num_steps=300)
        loss, grad = pgb.run(params, x0, base, tau, targets, weights, dt)
        # loss: (B,)   grad: (16, B)

    数组形状（均为 float64）：
        params  (19, B)  顺序同 Params 字段（mb..lambda, kb, ks）
        x0      (4,  B)  theta_b dtheta_b theta_s dtheta_s
        base    (3,  B)  theta_c0 dtheta_c ddtheta_c
        tau     (K, 2, B)  —— 即 (2k+c) 行、样本维连续
        targets 长度 4 的序列，每项 (K, B) 或 None（目标恒 0）
        weights (4,  B)  w_psi_b w_psi_s w_dpsi_b w_dpsi_s
    """

    def __init__(self, refinement: int, max_batch: int, max_num_steps: int,
                 num_threads: int = 0, lanes: int = 16):
        handle = _lib.tcbs_param_gradient_batch_create(
            int(refinement), int(max_batch), ctypes.c_size_t(max_num_steps),
            int(num_threads), int(lanes))
        if not handle:
            raise RuntimeError(f"tcbs_param_gradient_batch_create 失败: {_last_error()}")
        self._handle = handle
        self._refinement = int(refinement)
        self._max_batch = int(max_batch)
        self._max_num_steps = int(max_num_steps)

    def close(self) -> None:
        if getattr(self, "_handle", None):
            _lib.tcbs_param_gradient_batch_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self) -> "ParamGradientBatch":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _require_handle(self) -> ctypes.c_void_p:
        if not getattr(self, "_handle", None):
            raise RuntimeError("ParamGradientBatch 已关闭")
        return self._handle

    @property
    def num_threads(self) -> int:
        """实际使用的线程数。"""
        return int(_lib.tcbs_param_gradient_batch_threads(self._require_handle()))

    @property
    def lanes(self) -> int:
        """实际使用的 SIMD 分块宽度。"""
        return int(_lib.tcbs_param_gradient_batch_lanes(self._require_handle()))

    def run(self, params, x0, base, tau, targets, weights, dt):
        """计算一批序列的 (loss, grad)。

        返回 ``(loss, grad)``：``loss`` 形状 (B,)，``grad`` 形状 (16, B)。
        失败抛 RuntimeError。
        """
        pr = np.ascontiguousarray(params, dtype=np.float64)
        x0a = np.ascontiguousarray(x0, dtype=np.float64)
        ba = np.ascontiguousarray(base, dtype=np.float64)
        ta = np.ascontiguousarray(tau, dtype=np.float64)
        wa = np.ascontiguousarray(weights, dtype=np.float64)
        if pr.ndim != 2 or pr.shape[0] != 19:
            raise ValueError(f"params 形状应为 (19, B)，收到 {pr.shape}")
        B = pr.shape[1]
        if x0a.shape != (4, B):
            raise ValueError(f"x0 形状应为 (4, {B})，收到 {x0a.shape}")
        if ba.shape != (3, B):
            raise ValueError(f"base 形状应为 (3, {B})，收到 {ba.shape}")
        if ta.ndim != 3 or ta.shape[1] != 2 or ta.shape[2] != B:
            raise ValueError(f"tau 形状应为 (K, 2, {B})，收到 {ta.shape}")
        K = ta.shape[0]
        if wa.shape != (4, B):
            raise ValueError(f"weights 形状应为 (4, {B})，收到 {wa.shape}")
        if B > self._max_batch:
            raise ValueError(f"batch={B} 超过 max_batch={self._max_batch}")
        if K > self._max_num_steps:
            raise ValueError(f"num_steps={K} 超过 max_num_steps={self._max_num_steps}")

        tarr = []
        if len(targets) != 4:
            raise ValueError("targets 必须是长度 4 的序列")
        for name, t in zip(("target_psi_b", "target_psi_s",
                            "target_dpsi_b", "target_dpsi_s"), targets):
            if t is None:
                tarr.append(None)
                continue
            a = np.ascontiguousarray(t, dtype=np.float64)
            if a.shape != (K, B):
                raise ValueError(f"{name} 形状应为 ({K}, {B})，收到 {a.shape}")
            tarr.append(a)

        loss = np.zeros(B, dtype=np.float64)
        grad = np.zeros(NPARAM * B, dtype=np.float64)
        ok = _lib.tcbs_param_gradient_batch_run(
            self._require_handle(),
            ctypes.c_int(B), ctypes.c_size_t(K), ctypes.c_double(dt),
            _dbl_ptr(pr), _dbl_ptr(x0a), _dbl_ptr(ba), _dbl_ptr(ta),
            _dbl_ptr(tarr[0]), _dbl_ptr(tarr[1]), _dbl_ptr(tarr[2]), _dbl_ptr(tarr[3]),
            _dbl_ptr(wa), _dbl_ptr(loss), _dbl_ptr(grad),
        )
        if not ok:
            raise RuntimeError(f"tcbs_param_gradient_batch_run 失败: {_last_error()}")
        return loss, grad.reshape(NPARAM, B)

    def __repr__(self) -> str:
        return (f"ParamGradientBatch(refinement={self._refinement}, "
                f"max_batch={self._max_batch}, max_num_steps={self._max_num_steps})")
