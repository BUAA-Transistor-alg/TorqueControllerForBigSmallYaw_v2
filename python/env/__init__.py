"""环境封装包。

两个环境对外**接口完全一致**（duck-typing），采集脚本只换构造那一行：

  * ``SimEnv``  —— 仿真环境：动力学在 C++ 侧跑，重力构造时随机固定，可选加噪。
  * ``RealEnv`` —— 真实硬件环境：状态来自 RobotCommunication 的严格反解包，
    力矩真的下发给 MCU；**不加噪声**，用 ``time.perf_counter_ns()`` + 忙等做精确帧控制。

两者都只有两个操作接口：``state()``（立即取当前状态）与 ``step(Tb, Ts)``（走一步），
外加只读的 ``gravity`` / ``dt`` / ``refinement`` / ``noise`` / ``sigma``。
"""

from .sim_env import SimEnv, sample_gravity  # noqa: F401
from .real_env import RealEnv                # noqa: F401

__all__ = ["SimEnv", "RealEnv", "sample_gravity"]
