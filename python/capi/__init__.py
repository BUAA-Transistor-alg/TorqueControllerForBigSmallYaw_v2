"""libtcbs.so 的 ctypes 包装（C 接口封装）。

对应 C 头文件 include/capi/capi_dm.h，实现对 src/dm 中 C++ 仿真核心的调用；
C++ 内容位于命名空间 tcbs::dm，C 符号前缀为 tcbs_。

用法::

    from capi import Params, State, Simulator
"""

from .capi_dm import (  # noqa: F401
    PARAM_GRADIENT_NAMES,
    ParamGradient,
    ParamLossSpec,
    Params,
    Simulator,
    State,
    Trajectory,
    TrajectoryLoss,
    library_path,
)

__all__ = ["Params", "State", "Simulator", "Trajectory", "TrajectoryLoss",
           "ParamGradient", "ParamLossSpec", "PARAM_GRADIENT_NAMES", "library_path"]
