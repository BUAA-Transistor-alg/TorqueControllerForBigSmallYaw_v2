"""libtcbss.so 的 ctypes 包装。

对应 C 接口：include/tcbss_capi.h，实现对 src/ 中 C++ 仿真器的调用。

典型用法::

    from tcbss import Params, State, Simulator

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

__all__ = ["Params", "State", "Simulator", "Trajectory", "TrajectoryLoss", "library_path"]


# ---------------------------------------------------------------------------
# 与 include/tcbss_capi.h 一一对应的 C 结构体
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
    """返回 libtcbss 动态库路径。

    查找顺序：环境变量 TCBSS_LIB -> 项目 build/ 目录下的平台对应文件名。
    """
    here = Path(__file__).resolve().parent
    candidates = []
    env = os.environ.get("TCBSS_LIB")
    if env:
        candidates.append(Path(env))
    build_dir = here.parent / "build"
    candidates += [
        build_dir / "libtcbss.so",
        build_dir / "libtcbss.dylib",
        build_dir / "tcbss.dll",
    ]
    for cand in candidates:
        if cand.is_file():
            return str(cand)
    raise FileNotFoundError(
        "找不到 libtcbss 动态库，请先在项目根目录执行 ./build.sh 编译。\n"
        f"已查找: {', '.join(str(c) for c in candidates)}"
    )


def _load_library() -> ctypes.CDLL:
    lib = ctypes.CDLL(library_path())

    lib.tcbss_create.restype = ctypes.c_void_p
    lib.tcbss_create.argtypes = [ctypes.POINTER(_CParams), ctypes.c_double, ctypes.c_int]

    lib.tcbss_destroy.restype = None
    lib.tcbss_destroy.argtypes = [ctypes.c_void_p]

    lib.tcbss_last_error.restype = ctypes.c_char_p
    lib.tcbss_last_error.argtypes = []

    lib.tcbss_set_state.restype = None
    lib.tcbss_set_state.argtypes = [ctypes.c_void_p, _CState]

    lib.tcbss_set_generalized.restype = None
    lib.tcbss_set_generalized.argtypes = [
        ctypes.c_void_p,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
    ]

    lib.tcbss_set_theta_b.restype = None
    lib.tcbss_set_theta_b.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]

    lib.tcbss_set_theta_s.restype = None
    lib.tcbss_set_theta_s.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]

    lib.tcbss_get_state.restype = None
    lib.tcbss_get_state.argtypes = [ctypes.c_void_p, ctypes.POINTER(_CState)]

    lib.tcbss_get_dt.restype = ctypes.c_double
    lib.tcbss_get_dt.argtypes = [ctypes.c_void_p]

    lib.tcbss_get_refinement.restype = ctypes.c_int
    lib.tcbss_get_refinement.argtypes = [ctypes.c_void_p]

    lib.tcbss_get_params.restype = None
    lib.tcbss_get_params.argtypes = [ctypes.c_void_p, ctypes.POINTER(_CParams)]

    lib.tcbss_step.restype = None
    lib.tcbss_step.argtypes = [
        ctypes.c_void_p,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.POINTER(_CState),
    ]

    # ---- 轨迹梯度接口 ----
    lib.tcbss_trajectory_create.restype = ctypes.c_void_p
    lib.tcbss_trajectory_create.argtypes = [ctypes.POINTER(_CParams)]

    lib.tcbss_trajectory_destroy.restype = None
    lib.tcbss_trajectory_destroy.argtypes = [ctypes.c_void_p]

    _dbl_p = ctypes.POINTER(ctypes.c_double)

    lib.tcbss_trajectory_loss.restype = ctypes.c_double
    lib.tcbss_trajectory_loss.argtypes = [
        ctypes.c_void_p,                                  # t
        ctypes.c_double, ctypes.c_double, ctypes.c_double,  # theta_c0, dtheta_c, ddtheta_c
        ctypes.c_double,                                  # dt
        ctypes.c_size_t,                                  # num_steps
        _dbl_p,                                           # tau
        ctypes.c_double, ctypes.c_double,                 # tau_b_fixed, tau_s_fixed
        ctypes.POINTER(_CState),                          # x0
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,  # w1..w4
        ctypes.c_double, ctypes.c_double,                 # w5, w6
        _dbl_p, _dbl_p, _dbl_p, _dbl_p,                   # targets
        _dbl_p, _dbl_p, _dbl_p, _dbl_p,                   # outputs
    ]

    lib.tcbss_trajectory_gradient.restype = ctypes.c_double
    lib.tcbss_trajectory_gradient.argtypes = [
        ctypes.c_void_p,                                  # t
        ctypes.c_double, ctypes.c_double, ctypes.c_double,  # theta_c0, dtheta_c, ddtheta_c
        ctypes.c_double,                                  # dt
        ctypes.c_size_t,                                  # num_steps
        _dbl_p,                                           # tau
        ctypes.POINTER(_CState),                          # x0
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,  # w1..w4
        ctypes.c_double, ctypes.c_double,                 # w5, w6
        _dbl_p, _dbl_p, _dbl_p, _dbl_p,                   # targets
        _dbl_p,                                           # grad_tau
        ctypes.POINTER(_CState),                          # out_final_state
    ]

    return lib


_lib = _load_library()


def _last_error() -> str:
    raw = _lib.tcbss_last_error()
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

    def _to_c(self) -> _CParams:
        return _CParams(
            mb=self.mb, Ib=self.Ib, Pbx=self.Pbx, Pby=self.Pby,
            ms=self.ms, Is=self.Is, Psx=self.Psx, Psy=self.Psy,
            Dx=self.Dx, Dy=self.Dy, gx=self.gx, gy=self.gy,
            fbc=self.fbc, fbv=self.fbv, fsc=self.fsc, fsv=self.fsv,
            lambda_=self.lambda_,
        )

    @classmethod
    def _from_c(cls, c: _CParams) -> "Params":
        return cls(
            mb=c.mb, Ib=c.Ib, Pbx=c.Pbx, Pby=c.Pby,
            ms=c.ms, Is=c.Is, Psx=c.Psx, Psy=c.Psy,
            Dx=c.Dx, Dy=c.Dy, gx=c.gx, gy=c.gy,
            fbc=c.fbc, fbv=c.fbv, fsc=c.fsc, fsv=c.fsv,
            lambda_=c.lambda_,
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
        handle = _lib.tcbss_create(ctypes.byref(c_params), float(dt), int(refinement))
        if not handle:
            raise RuntimeError(f"tcbss_create 失败: {_last_error()}")
        self._handle = handle
        self._dt = float(dt)
        self._refinement = int(refinement)

    # -- 生命周期 ---------------------------------------------------------
    def close(self) -> None:
        if getattr(self, "_handle", None):
            _lib.tcbss_destroy(self._handle)
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
        _lib.tcbss_set_state(self._require_handle(), state._to_c())

    def set_generalized(
        self, theta_b: float, dtheta_b: float, theta_s: float, dtheta_s: float
    ) -> None:
        """直接设置当前两个广义坐标的位置与速度（分量形式）。"""
        _lib.tcbss_set_generalized(
            self._require_handle(), float(theta_b), float(dtheta_b),
            float(theta_s), float(dtheta_s),
        )

    def set_theta_b(self, theta_b: float, dtheta_b: float) -> None:
        """单独设置广义坐标 1 的位置与速度。"""
        _lib.tcbss_set_theta_b(self._require_handle(), float(theta_b), float(dtheta_b))

    def set_theta_s(self, theta_s: float, dtheta_s: float) -> None:
        """单独设置广义坐标 2 的位置与速度。"""
        _lib.tcbss_set_theta_s(self._require_handle(), float(theta_s), float(dtheta_s))

    # -- 读取 -------------------------------------------------------------
    @property
    def state(self) -> State:
        out = _CState()
        _lib.tcbss_get_state(self._require_handle(), ctypes.byref(out))
        return State._from_c(out)

    @property
    def dt(self) -> float:
        return _lib.tcbss_get_dt(self._require_handle())

    @property
    def refinement(self) -> int:
        return _lib.tcbss_get_refinement(self._require_handle())

    @property
    def params(self) -> Params:
        out = _CParams()
        _lib.tcbss_get_params(self._require_handle(), ctypes.byref(out))
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
        _lib.tcbss_step(
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

    def __init__(self, params: Params):
        c_params = params._to_c()
        handle = _lib.tcbss_trajectory_create(ctypes.byref(c_params))
        if not handle:
            raise RuntimeError(f"tcbss_trajectory_create 失败: {_last_error()}")
        self._handle = handle
        self._params = params

    # -- 生命周期 ---------------------------------------------------------
    def close(self) -> None:
        if getattr(self, "_handle", None):
            _lib.tcbss_trajectory_destroy(self._handle)
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

        value = _lib.tcbss_trajectory_loss(
            self._require_handle(),
            float(theta_c0), float(dtheta_c), float(ddtheta_c), float(dt),
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
            raise RuntimeError(f"tcbss_trajectory_loss 失败: {_last_error()}")
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

        value = _lib.tcbss_trajectory_gradient(
            self._require_handle(),
            float(theta_c0), float(dtheta_c), float(ddtheta_c), float(dt),
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
            raise RuntimeError(f"tcbss_trajectory_gradient 失败: {_last_error()}")
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

