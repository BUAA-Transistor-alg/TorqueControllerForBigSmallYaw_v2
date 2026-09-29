"""仿真环境封装包。

SimEnv 对外只有两个操作接口：state()（立即取当前状态）与 step(Tb, Ts)（走一步），
外加只读的 gravity / dt / refinement。动力学参数从 sim_config 读取，重力在构造时
随机确定（zero_gravity=True 则恒为 0），构造后不可修改、不可重置。
"""

from .sim_env import SimEnv, sample_gravity  # noqa: F401

__all__ = ["SimEnv", "sample_gravity"]
