"""libtcbs.so 的 ctypes 包装（C 接口封装）。

对应 C 头文件：
  * include/capi/capi_dm.h  —— src/dm 的 C++ 仿真核心（命名空间 tcbs::dm）
  * include/capi/capi_com.h —— src/com 的通信封装（命名空间 tcbs::com）

C 符号前缀统一为 tcbs_。

用法::

    from capi import Params, State, Simulator
    from capi import RobotCommunication, ImuLocation, McuSendPacket
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
from .capi_com import (  # noqa: F401
    ImuLocation,
    ImuReceivePacket,
    ImuSendPacket,
    LatestData,
    LinearParams,
    McuReceivePacket,
    McuSendPacket,
    RobotCommunication,
    StrictPose,
    YawMode,
)

__all__ = ["Params", "State", "Simulator", "Trajectory", "TrajectoryLoss",
           "ParamGradient", "ParamLossSpec", "PARAM_GRADIENT_NAMES", "library_path",
           # 通信模块（capi_com）
           "RobotCommunication", "ImuLocation", "YawMode", "LinearParams",
           "McuSendPacket", "McuReceivePacket", "ImuSendPacket", "ImuReceivePacket",
           "LatestData", "StrictPose"]
