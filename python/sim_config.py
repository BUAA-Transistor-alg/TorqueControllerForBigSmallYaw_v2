"""仿真 / 数据采集的全部默认参数（配置集中在此，脚本仍可用命令行覆盖）。

分四块：
  1. 仿真内核默认值（dt / refinement / 默认动力学参数）
  2. 工况与机械约束（等效重力、θ_s 限位、两关节速度限幅、力矩范围 ±1）
  3. 激励设计（参考轨迹频带/幅值、PD 增益、噪声 sigma）
  4. 采集会话设置（轨迹条数、时长、各类实验的专用参数）与输出目录
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ===========================================================================
# 1. 仿真内核默认值
# ===========================================================================
DT = 0.01                        # 采样周期 [s]（100 Hz）
DURATION = 3.0                   # 单条轨迹时长 [s]
NUM_STEPS = int(round(DURATION / DT))   # 300
REFINEMENT = 4                   # 每个主步的经典 RK4 子步数（运行期参数）

# 平滑摩擦常数：**已知的固定模型常数**，不是被辨识量（控制器/采集脚本可以用它）
LAMBDA = 100.0

# 默认动力学参数：既是造数据的真值，也是辨识脚本的对比基准
# （只有 sim_config 与辨识脚本会读它；两个采集脚本不碰它，参数一律由 SimEnv 内部读取）
DEFAULT_PARAMS = dict(
    mb=1.5, Ib=0.030, Pbx=0.12, Pby=0.18,
    ms=0.40, Is=0.020, Psx=0.08, Psy=0.10,
    Dx=0.05, Dy=0.30,
    fbc=0.020, fbv=0.050, fsc=0.010, fsv=0.020,
    lambda_=LAMBDA,
    # 等效重力的默认值（实际每次采集由调用方按摆平面倾角给出）
    gx=0.0, gy=0.0,
)

# 参与辨识的参数（与 C 接口 tcbs_param_gradient_name 一致；重力是已知输入、不在其中）
PARAM_NAMES = ["mb", "Ib", "Pbx", "Pby", "ms", "Is", "Psx", "Psy",
               "Dx", "Dy", "fbc", "fbv", "fsc", "fsv"]

# ===========================================================================
# 2. 工况与机械约束
# ===========================================================================
# ---- 等效重力：摆平面整体放在斜坡上，起作用的只有重力在摆平面内的分量 ----
#   水平面 -> 平面内重力 = 0；倾角 α 时 |g| = 9.81·sin(α)
GRAVITY_MAG = 9.81               # 真实重力加速度 [m/s²]
RAMP_MAX_DEG = 20.0              # 斜坡最大倾角 [°]
G_INPLANE_MAX = GRAVITY_MAG * 0.3420201433256687   # = 9.81·sin(20°) ≈ 3.355
ALPHA_MIN_DEG = 0.0             # 随机抽倾角时的下界
# 注：重力不再按概率抽"水平面"，而是由 SimEnv(zero_gravity=True) 显式指定为 0

# ---- θ_s 机械限位：±45° 硬限位；超 ±35° 电控降力矩，记录力矩≠实际力矩 ----
THETA_S_LIMIT_DEG = 35.0         # 硬约束：超过则丢弃该条
THETA_S_TARGET_DEG = 30.0        # 设计目标（留 5° 余量）
THETA_S0_DEG = 5.0               # 每条轨迹初始 θ_s 抖动 [°]

# ---- 两关节速度限幅 ----
V_MAX_B = 10.0                    # 关节 b 速度限幅 [rad/s]
V_MAX_S = 10.0                    # 关节 s 速度限幅 [rad/s]
V_REF_MARGIN = 0.8               # 参考速度只用到限幅的 80%
V_BARRIER_GAIN = 0.0             # 连续速度障碍增益 = GAIN·TAU_*_MAX/(1 rad/s 超出)
V_BARRIER_BETA = 0.5            # 障碍的 softplus 锐度

# ---- 力矩范围：两关节统一为 [-1, +1] [N·m] ----
TAU_B_MAX = 1.0                  # 关节 b 力矩上限 [N·m]
TAU_S_MAX = 1.0                  # 关节 s 力矩上限 [N·m]

# ===========================================================================
# 3. 激励设计与测量噪声
# ===========================================================================
# 参考轨迹频带：θ_s 取在关节固有频率(~0.7Hz)之上，避免共振放大
TB_FREQ_LO, TB_FREQ_HI = 0.2, 1.5      # θ_b 参考频带 [Hz]
TS_FREQ_LO, TS_FREQ_HI = 0.5, 3.0      # θ_s 参考频带 [Hz]
TS_REF_DEG = 15.0                      # θ_s 参考幅值 [°]
TB_REF_RAD = 0.8                       # θ_b 参考幅值 [rad]

KP_B, KD_B = 0.5, 0.01                 # 关节 b 位置环 PD 增益
KP_S, KD_S = 5.0, 0.1                 # 关节 s 位置环 PD 增益

MAX_ATTEMPTS = 60                      # 单条数据最多尝试次数（收缩参考幅值）
SHRINK = 0.85                          # 每次失败后参考幅值收缩系数
SHAPE_RAMP_TIME = 0.2                  # [s] 多正弦首尾平滑时间

NOISE_ENABLE = True                    # 仿真环境默认是否启用噪声（构造时可覆盖）
SIGMA_POS = 1.0e-2                     # 状态角测量噪声 σ [rad]
SIGMA_VEL = 5.0e-2                     # 状态角速度测量噪声 σ [rad/s]
SIGMA_TAU = 5.0e-3                     # 力矩噪声 σ [N·m]（直接加到实际施加上）

# 辨识损失权重（四项时间均值）
W_PSI_B = W_PSI_S = W_DPSI_B = W_DPSI_S = 1.0

# ===========================================================================
# 4. 采集会话设置
# ===========================================================================
DEFAULT_NUM = 120                # 默认轨迹条数
DEFAULT_SEED = 20240607

# ---- 无重置时的"控回初值"控制器（两关节各自一份，便于分开整定）----
REPOS_KP_B, REPOS_KD_B, REPOS_KI_B = 1.0, 0.0, 0.1    # 关节 b 位置 PI + 速度 D
REPOS_KP_S, REPOS_KD_S, REPOS_KI_S = 1.0, 0.0, 0.1    # 关节 s 位置 PI + 速度 D
REPOS_V_MAX = 1.5                # 目标点限速斜坡 [rad/s]（避免起步力矩饱和）
REPOS_TOL_SIGMA = 3.0            # 收敛容差 = max(REPOS_TOL_RAD, SIGMA·滤波后残余σ)
# 注意：噪声在仿真环境里，控制/判据读到的都是**带噪测量值**。因此：
#   * 控制回路对测量状态做一阶低通（CTRL_LPF_ALPHA），否则 D 项会把噪声放大成力矩噪声；
#   * 收敛判据必须基于滤波后的估计，容差不能低于测量噪声（否则永远判不了收敛）；
#   * 验收用的 θ_s 先做短窗均值（ACCEPT_AVG），x0 用较长窗口均值（X0_AVG）。
CTRL_LPF_ALPHA = 1.0            # 控制用状态的一阶低通系数（越小越平滑）
ACCEPT_AVG = 5                   # 验收 θ_s 的短窗均值点数
X0_AVG = 20                      # 起始状态（x0）的均值点数
REPOS_TOL_RAD = 0.05           # 角度收敛容差 [rad]
REPOS_TOL_VEL = 0.2           # 速度收敛容差 [rad/s]
REPOS_MAX_STEPS = 3000           # 最多用多少步把状态控回去
REPOS_INT_CLAMP = 1.0            # 积分限幅

# ---- 匀速旋转摩擦实验 ----
OMEGA_LIST = [-3.0, -1.0, -0.3, -0.1, -0.05, -0.02, -0.01, -0.005,
              0.005, 0.01, 0.02, 0.05, 0.1, 0.3, 1.0, 3.0]
FS_N_SETTLE = 600                # 稳定段步数
FS_N_MEAS = 300                  # 测量段步数
FS_KPV, FS_KIV = 0.01, 0.1        # 关节 b 速度环 PI 增益
FS_KPS, FS_KDS = 1.0, 0.1       # 关节 s 位置环 PD 增益（把 s 牢牢按住）
FS_INT_CLAMP = 50.0               # 速度环积分限幅

# ---- 输出目录 ----
DATA_DIR_SIM = REPO / "data" / "sim"
DATA_DIR_FRICTION = REPO / "data" / "friction"
DATA_DIR_IDENTIFY = REPO / "data" / "identify"
# 真实硬件采集（RealEnv）用的独立目录，避免与仿真数据混在一起
DATA_DIR_SIM_REAL = REPO / "data" / "sim_real"
DATA_DIR_FRICTION_REAL = REPO / "data" / "friction_real"


def run_stamp() -> str:
    """程序启动时的时间戳，用于给采集 npz 命名（多次运行互不覆盖）。

    每个采集脚本在 main() 开头调用一次；同一次运行写出的所有文件共用同一个戳。
    """
    return time.strftime("%Y%m%d_%H%M%S")


def category_dir(name: str) -> Path:
    """类别名 -> 目录 ``data/<name>``。

    类别即目录名，沿用现有布局：``--category simA`` → ``data/simA``，
    ``--category friction_simA`` → ``data/friction_simA``。
    """
    return REPO / "data" / name
