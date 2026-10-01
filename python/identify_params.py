"""辨识 data/sim/ 的动力学参数。

两步走：
  1) **代数初始化**（algebraic_init）：把运动方程对一组"聚合量"线性化后做闭式
     最小二乘。这是凸问题、有唯一解、**完全不依赖初值**，所以"随机初值差 2~3 个
     数量级"对它没有任何影响。它直接把 12 个可辨识组合解到 ~5%~10%。
  2) **课程 + 梯度精修**：用库的解析梯度（ParamGradient，前向灵敏度）+ torch Adam，
     拟合窗口从 K=5 逐步拉长到全窗口 K=300，把组合误差压到 ~0.1%、loss 压到噪声底。

梯度来源：ParamGradient（对 14 个动力学参数的前向灵敏度解析梯度，已用有限差分验证
到 1e-9，见 tests/test_trajectory_gradient.cpp）。torch 只负责参数容器与 Adam 更新。

被辨识参数（14 个）：
    mb, Ib, Pbx, Pby, ms, Is, Psx, Psy, Dx, Dy, fbc, fbv, fsc, fsv
已知量（不辨识，随数据给出）：
    gx, gy —— 重力矢量，**每条数据各自在半径 9.81 的圆内随机采样**并存在该条
              npz 里（见 generate_id_dataset.py）；求导时按条设置
    lambda_ —— 固定的平滑摩擦常数

重要：**并非 14 个参数都能辨识**。动力学只通过这些聚合量依赖参数：
    I_B=Ib+mb*Pb^2, I_S=Is+ms*Ps^2, I_D=ms*D^2, ms*h, ms*dhd,
    gs_*=ms*(gx*Psx+gy*Psy), gb_*=gx*(mb*Pbx+ms*Dx)+gy*(mb*Pby+ms*Dy), 4 个摩擦
把它们当自由未知数后方程对它们严格线性；线性系统里有 5 组共线列，等价于**恰好
2 个不可辨识方向**（mb 与 ms 的标度）。因此 mb、Ib、Psx、Psy、Pbx、Pby 的**数值
本身没有意义**，参数可以沿平坦谷滑很远而 loss 完全不变；只有 IDENTIFIABLE 里那
12 个组合是有意义的评价指标。绘图里参数曲线对不可辨识方向会明显偏离真值，属正常。

reparameterization（把 14 个量映射到 14 个 O(1) 的 theta）：
    mb, Ib, ms, Is, fbc, fbv, fsc, fsv  > 0   -> theta = log(p)
    (Pbx,Pby), (Psx,Psy), (Dx,Dy)             -> theta = (log r, phi), p = (r cos phi, r sin phi)

batch 调度（按需求实现）：
    * 每个 epoch 内数据不重复；一个 epoch 走完所有数据后重新打乱
    * batch = 8；若最后剩 n 个，则分给前面倒数 n 个 batch 各加 1
    * 若 n 超过前面的 batch 数，多出的部分在"尽可能平均、且尽量靠后"的前提下均摊
    * 若 batch_size > 数据总数，则直接全量计算

短窗口滑窗（K < full_K 的课程阶段，按需求实现）：
    * 拟合窗口长度为 K 时，不再固定只取每条数据开头的 K 步，而是以 K/2 为步长
      在整条数据上滑动长度为 K 的窗口：起点 s = 0, K/2, K, …（要求 s+K <= full_K），
      若最后一个窗口没贴到序列末尾再补一个贴尾窗口 s = full_K-K，
      于是一条数据取出多个样本；
    * 所有 (数据, 窗口) 样本汇总成一个**样本池**，每个 step 从池子里取 batch 个
      样本（上面的打乱/分批规则作用在池子上，池子走完算一个 epoch）；
    * s = 0 的样本就是"数据开头的 K 步"，与旧行为完全一致；K == full_K 的阶段
      （整段数据）只有 s=0 一个窗口，自然不做滑窗。
    * 窗口初值：窗口从**全局时刻 s*dt 的状态**出发。s=0 用 npz 里记录的 x0；
      s>0 用该时刻的实测状态反解（θ_b = psi_b[s-1] − θ_c(s·dt)，
      θ̇_b = dpsi_b[s-1] − θ̇_c(s·dt)，θ_s = psi_s[s-1] − psi_b[s-1]，θ̇_s 同理；
      psi_b[i] 是第 i 步**末**（时刻 (i+1)dt）的测量，故时刻 s*dt 对应下标 s-1）。
      基座量按等角加速度外推到 s*dt，因此窗口内的基座运动与原序列完全一致。
      这只是把积分起点平移，动力学方程、损失形式、每条样本的长度都不变；
      s-1 不在本窗口的 target 里，其测量噪声与本窗口残差独立，故不引入系统偏差。

逐参数冻结（--freeze-params，按需求实现）：
    * 在**代数最小二乘初始化之后**固定被选参数的值，前 n 个**全局** Adam step 内
      不优化，第 n 步起解冻；n 每个参数单独配置：
      ``--freeze-params "fbc:3000,fbv:3000,Dx:2000"``。
    * 省略步数 = 整个训练全程冻结（``"Dx"``）；步数给 0（或负数）= 不冻结，
      方便在长命令里临时关掉某一项。
    * 冻结值 = 初始化最小二乘解出的值；冻结期间该参数在数值上**完全不动**
      （opt.step() 之后精确写回），同时它的梯度分量被扣掉，Adam 的动量/二阶矩
      不会被它污染。
    * (Pbx,Pby)、(Psx,Psy)、(Dx,Dy) 在 theta 里是 (log r, phi)，可以**只冻结其中
      一个**：写回时用"冻结分量的固定值 + 另一分量的当前值"重建 theta。

外部已知参数（--known-params）：
    * ``--known-params "Dx=0.05,Dy=0.30"`` 把量得到的机械尺寸当已知量：代数最小二乘
      里 A+B = Dx·P + Dy·Q2、C−D = Dy·P − Dx·Q2 被解析折进 P、Q2 两列，
      设计矩阵 12 列 → 10 列（cond 79.6 → 58.7，P/Q2 的相对标准误改善 2~5 倍），
      训练时默认**全程冻结**在给定值上。
    * 也可给 ``mb``/``ms``（两个不可辨识标度）：跳过 (mb,ms) 网格搜索，把平坦谷
      彻底固定；``Dx,Dy`` 与 ``mb,ms`` 都固定后问题是 10 维、满秩、无零空间。
    * 摩擦 ``fbc/fbv/fsc/fsv`` 同样可以给：该列搬到右端丢掉，已知值优先于
      ``--friction-sweep`` 的结果。
    * ``Pbx/Pby/Psx/Psy/Ib/Is`` **不能**单独给定（它们只以 mb·Pbx、ms·Dx、
      Is+ms·Ps² 这类聚合形式进入方程），给了解析器会直接报错并说明原因。
    * 已知参数默认全程冻结；想让某个值只当初值、训练时放开，写
      ``--freeze-params "Dx:0"``（不冻结）或 ``"Dx:2000"``（先冻后放）。
    * **注意精度**：固定后拟合无法再纠正它，测量误差直接变成不可消除的偏差。
      本仓库数据实测（噪声底 2.6e-3）：Dx/Dy 偏差 0.1mm → loss 1.00×底；
      0.3mm → ~1.1×；1mm → 1.4~2.2×；3mm → 3.7~11.7×。亚毫米级固定才划算。

用法（在仓库根执行）::

    python3 python/identify_params.py                       # 代数初值 + 课程 + 10000 步
    python3 python/identify_params.py --init random --init-orders 2,3   # 随机初值对照
    python3 python/identify_params.py \
        --freeze-params "fbc:3000,fbv:3000,Dx"              # 逐参数冻结（Dx 全程冻结）
    python3 python/identify_params.py \
        --known-params "Dx=0.05,Dy=0.30"                    # 机械尺寸已知，10 列回归
    python3 python/identify_params.py \
        --known-params "Dx=0.05,Dy=0.30,mb=1.5,ms=0.4"      # 连同质量标度一起固定
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent          # 本文件所在目录：python/
REPO = HERE.parent                              # 仓库根
sys.path.insert(0, str(HERE))

from capi import (Params, State, ParamGradient, ParamGradientBatch,  # noqa: E402
                   ParamLossSpec, PARAM_GRADIENT_NAMES)
import sim_config as cfg  # noqa: E402

DATA_DIR = REPO / "data" / "sim"

# ===========================================================================
# 参数分类与取值范围（用于随机初始化；"不离谱"的物理量级）
# ===========================================================================
# 重参数化：把 14 个被辨识参数映射到 14 个无量纲 / 角度变量 theta
#   * 8 个恒正标量      -> theta = log(p)        (mb,Ib,ms,Is,fbc,fbv,fsc,fsv)
#   * 3 组有符号参数对  -> theta = (log r, phi)  ((Pbx,Pby),(Psx,Psy),(Dx,Dy))
#         p = (r cos phi, r sin phi)
#
# 为什么要用极坐标：直接对有符号参数做加法参数化时，同一组内两个分量可以相差
# 一个量级（例如 (Dx,Dy)），Adam 的统一步长会让小的那个一步跳很多、大的几乎不动。
# 换成 (log r, phi) 后 14 个 theta 分量都是 O(1)，单一学习率对所有参数都合适。
#
# 重力 gx/gy **不在**被辨识参数里：它是随采集数据一起保存的已知输入
# （每条 npz 内的 gx/gy），只参与正演，不求导。
# ===========================================================================
POSITIVE = {"mb", "Ib", "ms", "Is", "fbc", "fbv", "fsc", "fsv"}
PARAM_NAMES = list(PARAM_GRADIENT_NAMES)
assert set(PARAM_NAMES) == POSITIVE | {"Pbx", "Pby", "Psx", "Psy", "Dx", "Dy"}, PARAM_NAMES

# theta 分量 -> 物理量的布局（下标与 PARAM_GRADIENT_NAMES 对齐）
SCALAR_AT = {0: "mb", 1: "Ib", 4: "ms", 5: "Is", 10: "fbc", 11: "fbv", 12: "fsc", 13: "fsv"}
PAIR_AT = {2: ("Pbx", "Pby"), 6: ("Psx", "Psy"), 8: ("Dx", "Dy")}
_PARAM_INDEX = {n: i for i, n in enumerate(PARAM_NAMES)}

# 已知量：重力来自数据；lambda_ 是固定的平滑摩擦常数
LAMBDA = 100.0


def make_params(case, p_ident: dict) -> Params:
    """被辨识参数 + 该条数据的已知重力 + 固定 lambda_ -> 完整 Params。"""
    return Params(**p_ident, gx=case.gx, gy=case.gy, lambda_=LAMBDA)


# 随机初始化的范围
#   每个物理参数乘以 10^(±o)，o 在 [MIN, MAX] 内随机（符号随机）
#   默认即每个参数偏离真值 **2~3 个数量级**（×100 ~ ×1000，或 ÷100 ~ ÷1000）
# 这么大的初值差距下，直接对 3 s 全窗口做局部优化是不可行的（见文件末尾说明），
# 必须靠短窗口课程学习一步步拉回来。
LOG10_ORDERS_MIN = 2.0     # 至少 2 个数量级
LOG10_ORDERS_MAX = 3.0

# ===========================================================================
# 可辨识组合（12 个）
#
# 动力学只通过以下几个"聚合量"依赖参数：
#   I_B=Ib+mb*Pb^2, I_S=Is+ms*Ps^2, I_D=ms*D^2, ms*h, ms*dhd,
#   gs_sin/cos=ms*(gx*Psx+gy*Psy) 等, gb_sin/cos=gx*(mb*Pbx+ms*Dx)+...
# 把它们当作自由未知数后，运动方程 M*qdd + C + G = Q 对它们是**严格线性**的
# （见 algebraic_init），因此可以直接做闭式最小二乘。
#
# 线性系统里有 5 组严格共线的列，对应恰好不可辨识的方向：
#   I_B 与 I_D、ms*Dx*Psx 与 ms*Dy*Psy、ms*Dy*Psx 与 -ms*Dx*Psy、
#   mb*Pbx 与 ms*Dx、mb*Pby 与 ms*Dy。
# 于是**恰好有 2 个不可辨识方向**（mb 与 ms 的标度），下面 12 个组合才是可辨识的。
# 注意 Dx、Dy 是可辨识的（由 ms(Dx*Psx+Dy*Psy) 与 ms(Dy*Psx-Dx*Psy) 非线性组合
# 即可解出），而 mb*Pbx、ms*Dx 单独都不可辨识——之前那张列表是错的。
# ===========================================================================
IDENTIFIABLE = [
    ("Is + ms*(Psx^2+Psy^2)", lambda p: p["Is"] + p["ms"] * (p["Psx"] ** 2 + p["Psy"] ** 2)),
    ("Ib + mb*(Pbx^2+Pby^2) + ms*(Dx^2+Dy^2)",
     lambda p: p["Ib"] + p["mb"] * (p["Pbx"] ** 2 + p["Pby"] ** 2)
     + p["ms"] * (p["Dx"] ** 2 + p["Dy"] ** 2)),
    ("ms*Psx",                lambda p: p["ms"] * p["Psx"]),
    ("ms*Psy",                lambda p: p["ms"] * p["Psy"]),
    ("Dx",                    lambda p: p["Dx"]),
    ("Dy",                    lambda p: p["Dy"]),
    ("mb*Pbx + ms*Dx",        lambda p: p["mb"] * p["Pbx"] + p["ms"] * p["Dx"]),
    ("mb*Pby + ms*Dy",        lambda p: p["mb"] * p["Pby"] + p["ms"] * p["Dy"]),
    ("fbc",                   lambda p: p["fbc"]),
    ("fbv",                   lambda p: p["fbv"]),
    ("fsc",                   lambda p: p["fsc"]),
    ("fsv",                   lambda p: p["fsv"]),
]


def combo_error(p: dict, ref: dict) -> float:
    """12 个可辨识组合相对 ref 的最大相对误差（单参数偏差没有意义，见上）。"""
    return max(abs(f(p) - f(ref)) / max(abs(f(ref)), 1e-12) for _, f in IDENTIFIABLE)


# ===========================================================================
# 数据
# ===========================================================================
class Case:
    __slots__ = ("theta_c0", "dtheta_c", "ddtheta_c", "x0", "dt", "K",
                 "psi_b", "psi_s", "dpsi_b", "dpsi_s", "tau", "seed", "gx", "gy",
                 "weights", "theta_c_seq", "dtheta_c_seq",
                 "xstate", "base_seq")

    def __init__(self, path: Path):
        d = np.load(path)
        self.theta_c0 = float(d["theta_c0"])
        self.dtheta_c = float(d["dtheta_c"])
        self.ddtheta_c = float(d["ddtheta_c"])
        self.x0 = State(float(d["x0_theta_b"]), float(d["x0_dtheta_b"]),
                        float(d["x0_theta_s"]), float(d["x0_dtheta_s"]))
        self.dt = float(d["dt"])
        self.K = int(d["num_steps"])
        self.seed = int(d["seed"])
        # 重力矢量：随数据一起保存的已知输入（不是被辨识参数）
        self.gx = float(d["gx"])
        self.gy = float(d["gy"])
        # 辨识 loss 的四项权重 [psi_b, psi_s, dpsi_b, dpsi_s]：随数据一起记录，
        # 采集时写的是当时的 cfg.W_PSI_*。缺该字段的老数据按全 1。
        # 命令行可用 --w-psi-* / --w-dpsi-* 覆盖（见 main）。
        w = d["weights"] if "weights" in d.files else (1.0, 1.0, 1.0, 1.0)
        self.weights = tuple(float(x) for x in np.asarray(w).ravel())
        self.psi_b = np.ascontiguousarray(d["psi_b"], dtype=np.float64)
        self.psi_s = np.ascontiguousarray(d["psi_s"], dtype=np.float64)
        self.dpsi_b = np.ascontiguousarray(d["dpsi_b"], dtype=np.float64)
        self.dpsi_s = np.ascontiguousarray(d["dpsi_s"], dtype=np.float64)
        self.tau = np.ascontiguousarray(np.column_stack([d["tau_b"], d["tau_s"]]),
                                        dtype=np.float64)
        self.theta_c_seq, self.dtheta_c_seq = theta_c_trajectory(self)
        # ---- 滑窗样本用的整段边界量（下标 j = 全局时刻 j*dt，j = 0..K）----
        # xstate[j]：时刻 j*dt 的状态 (θ_b, θ̇_b, θ_s, θ̇_s)。
        #   j=0 用数据里记录的 x0；j>=1 用实测下标 j-1 的测量反解
        #   （ψ_b = θ_c + θ_b，ψ_s = θ_c + θ_b + θ_s）。
        self.xstate = np.empty((self.K + 1, 4), dtype=np.float64)
        self.xstate[0] = (self.x0.theta_b, self.x0.dtheta_b,
                          self.x0.theta_s, self.x0.dtheta_s)
        self.xstate[1:, 0] = self.psi_b - self.theta_c_seq
        self.xstate[1:, 1] = self.dpsi_b - self.dtheta_c_seq
        self.xstate[1:, 2] = self.psi_s - self.psi_b
        self.xstate[1:, 3] = self.dpsi_s - self.dpsi_b
        # base_seq[j]：时刻 j*dt 的基座 (θ_c, θ̇_c, θ̈_c)，等角加速度外推。
        # 把窗口起点平移到 s 后，C 侧就以 base_seq[s] 为"序列起点"的基座量。
        t = np.arange(self.K + 1) * self.dt
        self.base_seq = np.column_stack([
            self.theta_c0 + self.dtheta_c * t + 0.5 * self.ddtheta_c * t * t,
            self.dtheta_c + self.ddtheta_c * t,
            np.full(self.K + 1, self.ddtheta_c, dtype=np.float64)])

    def state_at(self, s: int) -> State:
        """全局时刻 s*dt 的（实测反解）状态，作为滑窗样本的初值 x0。"""
        return State(float(self.xstate[s, 0]), float(self.xstate[s, 1]),
                     float(self.xstate[s, 2]), float(self.xstate[s, 3]))

    def base_at(self, s: int) -> tuple[float, float, float]:
        """全局时刻 s*dt 的基座 (θ_c, θ̇_c, θ̈_c)。"""
        return (float(self.base_seq[s, 0]), float(self.base_seq[s, 1]),
                float(self.base_seq[s, 2]))


def theta_c_trajectory(c: Case) -> tuple[np.ndarray, np.ndarray]:
    """基座序列（与库内的等角加速度外推一致），长度 K，对应每个主步末。"""
    t = (np.arange(c.K) + 1) * c.dt
    tc = c.theta_c0 + c.dtheta_c * t + 0.5 * c.ddtheta_c * t * t
    dtc = c.dtheta_c + c.ddtheta_c * t
    return tc, dtc


def load_truth(path: Path) -> dict:
    truth = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            k, v = line.split()
            truth[k] = float(v)
    return truth


# ===========================================================================
# batch 调度
# ===========================================================================
def distribute_batches(n_items: int, batch_size: int, rng: np.random.Generator):
    """返回一批 batch 的索引数组列表（每个 epoch 用一次）。

    规则：
      * batch_size >= n_items：一个 batch 含全部
      * 否则先均分，剩 n 个分给前面倒数 n 个 batch 各 +1
      * 若 n 超过前面的 batch 数，多出的均摊到全部 batch（尽量平均、尽量靠后）
    """
    perm = rng.permutation(n_items)
    if batch_size >= n_items:
        return [perm]
    n_full = n_items // batch_size
    rem = n_items % batch_size
    sizes = [batch_size] * n_full
    if rem == 0:
        pass
    elif rem <= n_full:
        # 从倒数第 rem 个开始各 +1
        for j in range(n_full - rem, n_full):
            sizes[j] += 1
    else:
        # 多出的部分均摊：先全部 +1，再处理余数（尽量靠后）
        extra_each = rem // n_full
        left = rem % n_full
        for j in range(n_full):
            sizes[j] += extra_each
        for j in range(n_full - left, n_full):
            sizes[j] += 1
    out, pos = [], 0
    for sz in sizes:
        out.append(perm[pos:pos + sz])
        pos += sz
    return out


# ===========================================================================
# 参数容器（torch）
# ===========================================================================
class ParamSpec:
    """theta <-> Params 的极坐标重参数化、随机初始化与链式法则。

    布局见文件顶部 SCALAR_AT / PAIR_AT 的说明。
    """

    def __init__(self, truth: dict, rng: np.random.Generator,
                 orders: tuple[float, float] = (LOG10_ORDERS_MIN, LOG10_ORDERS_MAX),
                 device="cpu"):
        self.names = PARAM_NAMES
        self.truth = np.array([truth[n] for n in self.names], dtype=np.float64)

        # 逐个【物理参数】做乘性扰动，倍数为 10^(±orders 区间内的随机数)，符号随机。
        # 这样"初值差几个数量级"是字面含义（例如 mb 猜成 0.005 或 450）。
        p0 = self.truth.copy()
        for i in range(len(PARAM_NAMES)):
            o = rng.uniform(orders[0], orders[1])
            p0[i] *= 10.0 ** (o if rng.random() < 0.5 else -o)
        theta0 = ParamSpec.to_theta(p0)
        self.theta = torch.tensor(theta0, dtype=torch.float64, requires_grad=True)
        self.device = device

    # ---------------- theta <-> 物理参数 ----------------
    @staticmethod
    def to_params(theta_np: np.ndarray) -> dict:
        p = {n: float(np.exp(theta_np[i])) for i, n in SCALAR_AT.items()}
        for i, (nx, ny) in PAIR_AT.items():
            r, phi = float(np.exp(theta_np[i])), float(theta_np[i + 1])
            p[nx] = r * np.cos(phi)
            p[ny] = r * np.sin(phi)
        return p

    @staticmethod
    def to_theta(p) -> np.ndarray:
        get = (lambda n: float(p[n])) if isinstance(p, dict) else (lambda n: float(p[_PARAM_INDEX[n]]))
        th = np.zeros(len(PARAM_NAMES))
        for i, n in SCALAR_AT.items():
            th[i] = np.log(get(n))
        for i, (nx, ny) in PAIR_AT.items():
            px, py = get(nx), get(ny)
            th[i] = 0.5 * np.log(px * px + py * py)
            th[i + 1] = np.arctan2(py, px)
        return th

    @staticmethod
    def dtheta_from_dp(grad_p: np.ndarray, theta_np: np.ndarray) -> np.ndarray:
        """物理参数梯度 dL/dp -> theta 梯度 dL/dtheta（极坐标链式）。"""
        out = np.zeros(len(PARAM_NAMES))
        for i, n in SCALAR_AT.items():
            out[i] = grad_p[_PARAM_INDEX[n]] * np.exp(theta_np[i])
        for i, (nx, ny) in PAIR_AT.items():
            r, phi = np.exp(theta_np[i]), theta_np[i + 1]
            px, py = r * np.cos(phi), r * np.sin(phi)
            gx_, gy_ = grad_p[_PARAM_INDEX[nx]], grad_p[_PARAM_INDEX[ny]]
            out[i] = gx_ * px + gy_ * py            # dL/d(log r)
            out[i + 1] = -gx_ * py + gy_ * px       # dL/dphi
        return out

    # ---------------- torch 侧 ----------------
    def params_from_tensor(self) -> dict:
        with torch.no_grad():
            return ParamSpec.to_params(self.theta.detach().cpu().numpy())

    def theta_np(self) -> np.ndarray:
        return self.theta.detach().cpu().numpy().copy()

    def to_params_np(self, theta_np: np.ndarray) -> dict:
        return ParamSpec.to_params(theta_np)

    def seed_values(self) -> dict:
        return self.params_from_tensor()


# ===========================================================================
# 逐参数冻结："初始化最小二乘后先冻结，第 n 个 step 起才优化"，n 每个参数单独配
# ===========================================================================
def parse_freeze_params(spec: str) -> dict[str, float]:
    """解析 ``--freeze-params`` 的 "参数:步数,..." 表。

    * ``name:n``：前 n 个**全局** Adam step 内冻结，第 n 步起参与优化。
      为了能在长命令里临时关掉某一项，``n<=0`` 视为不冻结。
    * ``name``（省略步数）：整个训练全程冻结，从不解冻（记作 inf）。

    ``n<=0`` 的条目会以 0 保留在返回值里（``frozen_now`` 自然永不命中它）：
    ``--known-params`` 需要区分"表里没提到这个参数"（known 参数默认全程冻结）
    与"明确写了 0 = 不冻结"（此时只借用的已知值当初值，训练时放开）。
    """
    out: dict[str, float] = {}
    for part in spec.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            name, n_s = (s.strip() for s in part.split(":", 1))
            n = int(float(n_s))
            out[name] = float(max(0, n))    # 0 / 负数 = 不冻结（保留条目，见 docstring）
        else:
            out[part] = math.inf    # 省略步数 = 全程冻结
    for name in out:
        if name not in _PARAM_INDEX:
            raise ValueError(f"--freeze-params 里的未知参数 {name!r}；"
                             f"可选: {', '.join(PARAM_NAMES)}")
    # 按 PARAM_NAMES 的顺序返回，打印/绘图稳定
    return {n: out[n] for n in PARAM_NAMES if n in out}


# ===========================================================================
# 外部已知参数（--known-params）：把量得到的物理量当已知量喂进去
#
# 不是所有参数都能"单独"给定。12 列回归里能被解析消掉的只有下面这几种：
#   * Dx, Dy       —— 机械尺寸（关节 s 相对关节 b 的偏移）。两者必须同时给：
#                     A+B = Dx·P + Dy·Q2、C−D = Dy·P − Dx·Q2，只有 Dx、Dy 都已知
#                     才能把这两列折进 P、Q2（这也是 12 列 → 10 列的关键）。
#   * fbc..fsv     —— 摩擦，直接把该列搬到右端并丢掉。
#   * mb, ms       —— 两个不可辨识标度。它们不出现在任何一列里，给定时只用来
#                     跳过 (mb, ms) 的网格搜索，从而把 2 维平坦谷彻底固定。
# Pbx/Pby/Psx/Psy/Ib/Is 不能单独给定：它们只以 mb·Pbx、ms·Dx、Is+ms·Ps² 这类
# 聚合形式进入方程，单独固定某一个而不知道质量标度时方程无法自洽。
# ===========================================================================
KNOWN_OK = {"Dx", "Dy", "mb", "ms", "fbc", "fbv", "fsc", "fsv"}
KNOWN_UNSUPPORTED = {"Pbx", "Pby", "Psx", "Psy", "Ib", "Is"}


def parse_known_params(spec: str) -> dict[str, float]:
    """解析 ``--known-params`` 的 "参数=值,..." 表。

    例: ``--known-params "Dx=0.05,Dy=0.30"``（两个机械尺寸）。
    语义见上面 KNOWN_OK 的说明；给不支持的参数会直接抛 ValueError。
    """
    out: dict[str, float] = {}
    for part in spec.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"--known-params 的 {part!r} 缺少 '='；应为 参数=值")
        name, val_s = (s.strip() for s in part.split("=", 1))
        if name not in _PARAM_INDEX:
            raise ValueError(f"--known-params 里的未知参数 {name!r}；"
                             f"可选: {', '.join(sorted(KNOWN_OK))}")
        if name in KNOWN_UNSUPPORTED:
            raise ValueError(
                f"{name!r} 不能单独作为已知量：它只以 mb·Pbx / ms·Dx / Is+ms·Ps² "
                f"这类聚合形式进入方程，单独固定而不给出质量标度时方程无法自洽。"
                f"可用的已知参数: {', '.join(sorted(KNOWN_OK))}")
        try:
            v = float(val_s)
        except ValueError:
            raise ValueError(f"--known-params 里 {name} 的值 {val_s!r} 不是数") from None
        if not math.isfinite(v):
            raise ValueError(f"--known-params 里 {name} 的值 {val_s!r} 不是有限数")
        if name in POSITIVE and v <= 0.0:
            raise ValueError(f"--known-params 里 {name} 必须为正（重参数化用 log），"
                             f"收到 {v:g}")
        out[name] = v
    if ("Dx" in out) != ("Dy" in out):
        raise ValueError("Dx 与 Dy 必须同时给出：12 列回归中 A+B / C−D 只有 "
                         "Dx、Dy 都已知时才能解析折进 P、Q2（否则无法单独消去）")
    return out


class FreezeSchedule:
    """逐参数"先冻结、第 n 步起解冻"的调度（每个参数的 n 单独配置）。

    在**代数最小二乘初始化之后**捕获冻结值（``capture``），随后每个 Adam step：
      1) ``apply_grad``：把 dL/dtheta 里会改变"当前仍冻结"参数的分量扣掉
         （标量直接清零；参数对 (Pbx,Pby)/(Psx,Psy)/(Dx,Dy) 的 theta 是
         (log r, phi)，只冻结其中一个时按约束方向做正交投影），
         这样 Adam 的动量/二阶矩不会被冻结参数污染；
      2) ``restore``：``opt.step()`` 之后把冻结参数**精确写回**初始化值
         （成对参数用"冻结分量的固定值 + 另一分量的当前值"重建 theta），
         保证冻结期间该参数数值上完全不动。
    """

    def __init__(self, until: dict[str, float]):
        self.until = dict(until)
        self.values: dict[str, float] = {}
        self.theta0: np.ndarray | None = None

    def __bool__(self) -> bool:
        return bool(self.until)

    def capture(self, ps: "ParamSpec") -> None:
        """固定冻结值：必须在最小二乘初始化之后、第一次更新之前调用。

        同时记下 theta 本身：标量参数与"整对冻结"的参数对在写回时可以做到
        **位级精确**（直接还原 theta 分量），不必经过 exp/log 往返。
        """
        self.theta0 = ps.theta_np()
        p = ps.seed_values()
        self.values = {n: float(p[n]) for n in self.until}

    def frozen_now(self, gstep: int) -> set[str]:
        """第 gstep 个全局 step（0-based）开始时仍处于冻结的参数。"""
        return {n for n, u in self.until.items() if gstep < u}

    def apply_grad(self, dtheta: np.ndarray, theta_np: np.ndarray,
                   gstep: int) -> np.ndarray:
        frozen = self.frozen_now(gstep)
        if not frozen:
            return dtheta
        g = np.array(dtheta, dtype=np.float64, copy=True)
        # 标量参数：对应的 theta 分量直接清零（Adam 更新恒为 0）
        for i, n in SCALAR_AT.items():
            if n in frozen:
                g[i] = 0.0
        # 参数对：如果只冻结其中一个，把梯度投影到"该分量不变"的方向上
        p = None
        for i, (nx, ny) in PAIR_AT.items():
            fx, fy = nx in frozen, ny in frozen
            if fx and fy:
                g[i] = g[i + 1] = 0.0
            elif fx or fy:
                if p is None:
                    p = ParamSpec.to_params(theta_np)
                nm, other = (nx, ny) if fx else (ny, nx)
                # c(theta) = p_nm；∂p_nx/∂φ = -p_ny、∂p_ny/∂φ = +p_nx
                sgn = 1.0 if nm == nx else -1.0
                gc = np.array([p[nm], -sgn * p[other]])
                nrm = float(gc @ gc)
                if nrm > 0.0:
                    g[i:i + 2] -= (float(g[i:i + 2] @ gc) / nrm) * gc
        return g

    def restore(self, ps: "ParamSpec", gstep: int) -> None:
        """opt.step() 之后调用：把仍冻结的参数写回初始化值。"""
        frozen = self.frozen_now(gstep)
        if not frozen:
            return
        th = ps.theta_np()
        p = None
        for i, n in SCALAR_AT.items():
            if n in frozen:
                th[i] = self.theta0[i]          # 位级精确
        for i, (nx, ny) in PAIR_AT.items():
            fx, fy = nx in frozen, ny in frozen
            if fx and fy:
                th[i], th[i + 1] = self.theta0[i], self.theta0[i + 1]   # 位级精确
            elif fx or fy:
                # 只冻结其中一个：用它固定的物理值 + 另一个的当前值重建 theta
                if p is None:
                    p = ps.params_from_tensor()
                px = self.values[nx] if fx else p[nx]
                py = self.values[ny] if fy else p[ny]
                th[i] = 0.5 * np.log(px * px + py * py)
                th[i + 1] = np.arctan2(py, px)
        with torch.no_grad():
            ps.theta.copy_(torch.tensor(th, dtype=torch.float64))

    def label(self, n: str) -> str:
        u = self.until[n]
        if u <= 0:
            return "不冻结"
        return f"前 {int(u)} 步冻结" if np.isfinite(u) else "全程冻结"


# ===========================================================================
# 单条数据的 loss / 梯度（支持只取 [s, s+K) 这段窗口，课程学习 + 滑窗用）
# ===========================================================================
def case_spec(case: Case, K: int, s: int = 0) -> ParamLossSpec:
    """目标为该条实测序列的 [s, s+K) 段；四项权重取自 case.weights。

    case.weights = [w_psi_b, w_psi_s, w_dpsi_b, w_dpsi_s]：
    psi 是位置项、dpsi 是速度项，b/s 分别对应大/小 yaw。
    s 是窗口起点（主步下标）；s=0 即"数据开头的 K 步"（旧行为）。
    """
    w_psi_b, w_psi_s, w_dpsi_b, w_dpsi_s = case.weights
    sl = slice(s, s + K)
    return ParamLossSpec(w_psi_b=w_psi_b, w_psi_s=w_psi_s,
                         w_dpsi_b=w_dpsi_b, w_dpsi_s=w_dpsi_s,
                         target_psi_b=case.psi_b[sl], target_psi_s=case.psi_s[sl],
                         target_dpsi_b=case.dpsi_b[sl], target_dpsi_s=case.dpsi_s[sl])


def case_loss_and_grad(pg: ParamGradient, case: Case,
                       K: int, s: int = 0) -> tuple[float, np.ndarray]:
    """返回 [s, s+K) 这段窗口上的 (loss, dL/dp)（对物理参数，不是 theta）。

    求导点必须已经由调用方通过 ``pg.set_params(...)`` 设好——句柄里存着参数，
    这里不接收 params，以免出现"传了参数但句柄没更新"的静默错配。

    tau 必须和 target 一起截断到窗口：C++ 侧的步数取自 tau 的行数。
    窗口起点 s>0 时，积分从该时刻的实测状态（case.xstate[s]）出发，基座量也
    平移到该时刻（case.base_seq[s]）——等价于把整段序列的 [s, s+K) 段单独拎出来。
    """
    tc0, dtc, ddtc = case.base_at(s)
    loss, grad = pg.gradient(tc0, dtc, ddtc, case.dt,
                             case.tau[s:s + K], case.state_at(s), case_spec(case, K, s))
    return float(loss), np.asarray(grad, dtype=np.float64)


def case_loss(pg: ParamGradient, case: Case, K: int, s: int = 0) -> float:
    """只要 [s, s+K) 窗口上的 loss。同样要求句柄参数已经设好。"""
    tc0, dtc, ddtc = case.base_at(s)
    return float(pg.loss(tc0, dtc, ddtc, case.dt,
                         case.tau[s:s + K], case.state_at(s), case_spec(case, K, s)))


# ===========================================================================
# 批量（SoA）参数梯度：一次算完一个 batch 的 loss 之和与 dL/dp 之和
#
# 与上面逐条调用 case_loss_and_grad 完全等价（数值一致到 1e-13 量级），
# 但底层是 SoA + SIMD + 多线程，所以同样的 batch 只用一次 C 调用。
# ===========================================================================
def pack_batch_data(cases: list["Case"], lam: float) -> dict:
    """把各条数据的（整段）数组拼成整块（按样本取窗口时只做一次 fancy indexing）。

    * x0      (N, 4)
    * base    (N, 3)   theta_c0 / dtheta_c / ddtheta_c
    * xstate  (N, K+1, 4)  各主步边界时刻 j*dt 的状态（滑窗样本的初值）
    * base_seq(N, K+1, 3)  各主步边界时刻 j*dt 的基座量（滑窗样本的序列起点基座）
    * tau     (N, K, 2)
    * tgt     长度 4，各 (N, K)
    * gx/gy   (N,)
    * weights (N, 4)
    * dt / lambda 标量
    """
    return {
        "dt": cases[0].dt,
        "lambda": float(lam),
        "x0": np.array([[c.x0.theta_b, c.x0.dtheta_b, c.x0.theta_s, c.x0.dtheta_s]
                        for c in cases], dtype=np.float64),
        "base": np.array([[c.theta_c0, c.dtheta_c, c.ddtheta_c] for c in cases],
                         dtype=np.float64),
        "xstate": np.stack([c.xstate for c in cases]).astype(np.float64, copy=False),
        "base_seq": np.stack([c.base_seq for c in cases]).astype(np.float64, copy=False),
        "tau": np.stack([c.tau for c in cases]).astype(np.float64, copy=False),
        "tgt": [np.stack([c.psi_b for c in cases]).astype(np.float64, copy=False),
                np.stack([c.psi_s for c in cases]).astype(np.float64, copy=False),
                np.stack([c.dpsi_b for c in cases]).astype(np.float64, copy=False),
                np.stack([c.dpsi_s for c in cases]).astype(np.float64, copy=False)],
        "gx": np.array([c.gx for c in cases], dtype=np.float64),
        "gy": np.array([c.gy for c in cases], dtype=np.float64),
        "weights": np.array([c.weights for c in cases], dtype=np.float64),
    }


def batch_loss_and_grad(pgb: ParamGradientBatch, win_case, win_start, K: int,
                        params_now: dict, data: dict,
                        loss_scale: float) -> tuple[float, np.ndarray]:
    """一个 batch 的 (loss 之和, dL/dp 之和 × loss_scale)，grad 形状 (14,)。

    ``win_case`` / ``win_start`` 是这个 batch 里各样本的（数据下标, 窗口起点 s）：
    样本覆盖该条数据的 [s, s+K) 段，初值取该时刻的实测状态 ``xstate[:, s]``，
    基座起点取 ``base_seq[:, s]``（等价于把整段序列的窗口段单独拎出来）。

    ``params_now`` 是 14 个被辨识参数（同一批共享）；每条数据的 gx/gy 不同，
    按样本从 ``data`` 里取。
    """
    ci = np.asarray(win_case, dtype=np.intp)
    st = np.asarray(win_start, dtype=np.intp)
    B = ci.size
    cols = st[:, None] + np.arange(K, dtype=np.intp)[None, :]        # (B, K)
    # params: (17, B)，顺序同 Params 字段（mb..Dy | gx gy | fbc..fsv | lambda）
    pr = np.empty((17, B), dtype=np.float64)
    pr[:10, :] = np.array([params_now[n] for n in PARAM_NAMES[:10]],
                          dtype=np.float64)[:, None]
    pr[10, :] = data["gx"][ci]
    pr[11, :] = data["gy"][ci]
    pr[12:16, :] = np.array([params_now[n] for n in PARAM_NAMES[10:]],
                            dtype=np.float64)[:, None]
    pr[16, :] = data["lambda"]
    loss, grad = pgb.run(
        pr,
        data["xstate"][ci, st].T,                     # (4, B)
        data["base_seq"][ci, st].T,                   # (3, B)
        data["tau"][ci[:, None], cols].transpose(1, 2, 0),   # (K, 2, B)
        [t[ci[:, None], cols].T for t in data["tgt"]],  # 各 (K, B)
        data["weights"][ci].T,                        # (4, B)
        data["dt"],
    )
    return float(loss.sum()), grad.sum(axis=1) * loss_scale


# ===========================================================================
# 短窗口滑窗：长度为 K 的窗口在长度 full_K 的数据上的所有起点
# ===========================================================================
def window_starts(full_K: int, K: int, stride: int) -> np.ndarray:
    """返回长度 K 的窗口在长度 full_K 序列上的全部起点（升序、互不重复）。

    起点 0, stride, 2*stride, …（要求 s + K <= full_K）；若最后一个窗口没有贴到
    序列末尾（s + K < full_K），再补一个贴尾窗口 full_K - K，保证尾部数据也被用到。
    stride = K//2（K/2 的滑窗步长）时，full_K 被 K 整除的情况下一般不触发补齐。
    """
    stride = max(1, int(stride))
    starts = list(range(0, full_K - K + 1, stride))
    tail = full_K - K
    if starts[-1] != tail:
        starts.append(tail)
    return np.asarray(starts, dtype=np.intp)


# ===========================================================================
# 代数初始化：把运动方程对"聚合量"线性化后做闭式最小二乘
#
# 运动方程（见 src/dm/param_gradient.cpp 的 evaluate）：
#     M(θ_s)·q̈ + C(q̇,θ_s) + G(θ) + M·ddθ_c = Q(τ, q̇)
#     M11 = I_B + I_D + I_S + 2·ms·h,  M12 = I_S + ms·h,  M22 = I_S
#     ms·h   = A·ct + B·ct + C·st − D·st
#     ms·dhd = −A·st − B·st + C·ct − D·ct
#     G2 = (gx·P + gy·Q2)·ss + (gx·Q2 − gy·P)·cs
#     G1 = gb_sin·sb + gb_cos·cb + G2,  gb_sin = gx(R+U) + gy(S+V), …
#     Qb = Tb − fbv·ḃb − fbc·tanh(λ·ḃb)，Qs 同理
# 未知数（17 个，均以一次幂出现）：
#     I_B, I_S, I_D, A=ms·Dx·Psx, B=ms·Dy·Psy, C=ms·Dy·Psx, D=ms·Dx·Psy,
#     P=ms·Psx, Q2=ms·Psy, R=mb·Pbx, S=mb·Pby, U=ms·Dx, V=ms·Dy, fbc…fsv
# q̈ 由实测角速度差分（先做滑动平均降噪）得到。于是这是**凸问题、闭式解**，
# 完全不依赖初值——这正是"初值差几个数量级"也能做出来的原因。
# ===========================================================================
_LAM = LAMBDA


def _smooth(y: np.ndarray, w: int = 7) -> np.ndarray:
    k = np.ones(w) / w
    return np.convolve(np.pad(y, (w // 2, w // 2), mode="edge"), k, mode="valid")


def _regression_block(c: Case) -> tuple[np.ndarray, np.ndarray]:
    """单条数据贡献 2K 行，列为 **12 个可辨识组合**（列满秩，条件数好）。

    为什么不用 17 个"聚合量"直接回归：那 17 列里有 5 组**严格共线**（I_B≡I_D、
    A≡B、C≡−D、R≡U、S≡V），设计矩阵 rank 只有 12/17；而 θ_s 被限制在 ±30° 后
    cos θ_s 几乎不变，列之间进一步接近共线，条件数实测 1.4e18 → 摩擦解出来是
    负值。把 5 组共线列解析地合并成下面 12 列后，条件数降到 ~73，结果才可信。

    列顺序：0 I_BD  1 I_S  2 A+B  3 C−D  4 P  5 Q2  6 R+U  7 S+V
            8 fbc   9 fbv  10 fsc 11 fsv
    """
    tc, dtc = c.theta_c_seq, c.dtheta_c_seq
    ddtc = c.ddtheta_c
    ts = c.psi_s - c.psi_b
    dtb = _smooth(c.dpsi_b - dtc)
    dts = _smooth(c.dpsi_s - c.dpsi_b)
    ddtb = np.gradient(dtb, c.dt)
    ddts = np.gradient(dts, c.dt)

    ss, cs = np.sin(c.psi_s), np.cos(c.psi_s)
    sb, cb = np.sin(c.psi_b), np.cos(c.psi_b)
    st, ct = np.sin(ts), np.cos(ts)
    gx, gy = c.gx, c.gy
    Tb, Ts = c.tau[:, 0], c.tau[:, 1]
    n = c.K
    R = np.zeros((2 * n, 12))
    y = np.zeros(2 * n)
    dc = dts * (2.0 * (dtc + dtb) + dts)          # C1 的公共因子
    for k in range(n):
        rb = np.zeros(12)
        rs = np.zeros(12)
        # --- 第 1 行：M11·ddtb + M12·ddts + C1 + G1 + M11·ddtc − Qb = 0
        rb[0] += ddtb[k] + ddtc                        # I_BD = I_B + I_D
        rb[1] += ddtb[k] + ddts[k] + ddtc              # I_S
        rb[2] += ct[k] * (2.0 * ddtb[k] + ddts[k] + 2.0 * ddtc) - st[k] * dc[k]   # A+B
        rb[3] += st[k] * (2.0 * ddtb[k] + ddts[k] + 2.0 * ddtc) + ct[k] * dc[k]   # C−D
        rb[6] += gx * sb[k] - gy * cb[k]               # R+U
        rb[7] += gy * sb[k] + gx * cb[k]               # S+V
        rb[4] += gx * ss[k] - gy * cs[k]               # P
        rb[5] += gy * ss[k] + gx * cs[k]               # Q2
        rb[9] += dtb[k]
        rb[8] += np.tanh(_LAM * dtb[k])
        y[2 * k] = Tb[k]
        # --- 第 2 行：M12·ddtb + M22·ddts + C2 + G2 + M12·ddtc − Qs = 0
        rs[1] += ddtb[k] + ddts[k] + ddtc
        rs[2] += ct[k] * (ddtb[k] + ddtc) + st[k] * dtb[k] ** 2                    # A+B
        rs[3] += st[k] * (ddtb[k] + ddtc) - ct[k] * dtb[k] ** 2                    # C−D
        rs[4] += gx * ss[k] - gy * cs[k]
        rs[5] += gy * ss[k] + gx * cs[k]
        rs[11] += dts[k]
        rs[10] += np.tanh(_LAM * dts[k])
        y[2 * k + 1] = Ts[k]
        R[2 * k] = rb
        R[2 * k + 1] = rs
    return R, y


def algebraic_init(cases: list[Case], nominal=(1.0, 1.0),
                   friction: tuple[float, float, float, float] | None = None,
                   known: dict[str, float] | None = None
                   ) -> tuple[dict, dict]:
    """在测量数据上做闭式最小二乘，解出可辨识组合，再回代成物理参数。

    friction 若给出 (fbc, fbv, fsc, fsv) 则**直接采用**（来自匀速旋转实验），
    否则用回归自己的结果。注意 (mb, ms) 的可行性判据只看 Ib/Is，与摩擦无关——
    因此摩擦解坏掉也不会像以前那样让整个初始化抛异常。

    known 是外部已知参数（``--known-params``，见 parse_known_params）：
      * ``Dx``/``Dy``：把 A+B = Dx·P + Dy·Q2、C−D = Dy·P − Dx·Q2 折进 P、Q2 两列，
        设计矩阵 12 列 → 10 列（这是固定精确机械尺寸时唯一正确的降维方式）；
      * ``fbc``..``fsv``：该列搬到右端并丢掉，其余组合由"扣除已知摩擦后的力矩"估计；
      * ``mb``/``ms``：跳过对应方向的网格搜索，直接用给定值（2 个都给 = 平坦谷消失）。
    known 里的摩擦优先于 ``friction`` 参数。
    """
    known = dict(known or {})
    unknown = [n for n in known if n not in KNOWN_OK]
    if unknown:
        raise ValueError(f"algebraic_init 收到不支持的已知参数: {', '.join(unknown)}")

    R = np.vstack([_regression_block(c)[0] for c in cases])
    y = np.concatenate([_regression_block(c)[1] for c in cases])

    keep = np.ones(R.shape[1], dtype=bool)
    if "Dx" in known:
        # A+B 与 C−D 都是 P=ms·Psx、Q2=ms·Psy 的已知系数线性组合：折进去并丢掉这两列
        R[:, 4] = R[:, 4] + known["Dx"] * R[:, 2] + known["Dy"] * R[:, 3]
        R[:, 5] = R[:, 5] + known["Dy"] * R[:, 2] - known["Dx"] * R[:, 3]
        keep[2] = keep[3] = False
    for j, n in ((8, "fbc"), (9, "fbv"), (10, "fsc"), (11, "fsv")):
        if n in known:
            y = y - R[:, j] * known[n]      # 已知摩擦搬到右端
            keep[j] = False

    xs = np.zeros(R.shape[1])
    xs[keep], *_ = np.linalg.lstsq(R[:, keep], y, rcond=None)
    for j, n in ((8, "fbc"), (9, "fbv"), (10, "fsc"), (11, "fsv")):
        if n in known:                      # 折掉的那几列回填已知值，便于下面统一取用
            xs[j] = known[n]

    I_BD, I_S = xs[0], xs[1]
    AB, CmD = xs[2], xs[3]
    P, Q2 = xs[4], xs[5]
    RU, SV = xs[6], xs[7]
    if friction is None:
        friction = (xs[8], xs[9], xs[10], xs[11])
    else:
        # known 优先于 sweep 给出的摩擦
        friction = tuple(known.get(n, friction[i])
                         for i, n in enumerate(("fbc", "fbv", "fsc", "fsv")))
    fbc, fbv, fsc, fsv = friction
    if "Dx" in known:
        Dx, Dy = known["Dx"], known["Dy"]       # 直接采用测量值，不再从 P/Q2 回解
    else:
        det = P * P + Q2 * Q2
        Dx = (P * AB - Q2 * CmD) / det           # Dx、Dy 由聚合量非线性组合解出
        Dy = (Q2 * AB + P * CmD) / det

    def back(mb: float, ms: float) -> dict:
        I_D = ms * (Dx * Dx + Dy * Dy)
        I_B = I_BD - I_D
        Pbx = (RU - ms * Dx) / mb
        Pby = (SV - ms * Dy) / mb
        return dict(mb=mb, Ib=I_B - mb * (Pbx ** 2 + Pby ** 2), Pbx=Pbx, Pby=Pby,
                    ms=ms, Is=I_S - (P * P + Q2 * Q2) / ms, Psx=P / ms, Psy=Q2 / ms,
                    Dx=Dx, Dy=Dy, fbc=fbc, fbv=fbv, fsc=fsc, fsv=fsv)

    def feasible(p: dict) -> bool:
        # 可行性只看动力学参数；摩擦由独立实验给出，不参与这个判据
        v = np.array([p["mb"], p["Ib"], p["Pbx"], p["Pby"],
                      p["ms"], p["Is"], p["Psx"], p["Psy"], p["Dx"], p["Dy"]])
        return bool(np.isfinite(v).all() and p["mb"] > 0 and p["ms"] > 0
                    and p["Ib"] > 1e-6 and p["Is"] > 1e-6)

    mb_grid = ([known["mb"]] if "mb" in known
               else np.geomspace(0.02, 100.0, 140))
    ms_grid = ([known["ms"]] if "ms" in known
               else np.geomspace(0.02, 100.0, 140))
    best, best_cost = None, np.inf
    for mb in mb_grid:
        for ms in ms_grid:
            p = back(mb, ms)
            if not feasible(p):
                continue
            cost = abs(np.log(mb / nominal[0])) + abs(np.log(ms / nominal[1]))
            if cost < best_cost:
                best, best_cost = p, cost
    if best is None:
        if "mb" in known or "ms" in known:
            raise RuntimeError(
                f"代数初始化在给定 mb={known.get('mb', '自由')} ms={known.get('ms', '自由')} "
                f"下不可行（Ib 或 Is ≤ 0）。请检查 --known-params 的数值，或不要固定质量标度")
        raise RuntimeError("代数初始化找不到可行的 (mb, ms)，数据可能异常")

    for n, v in known.items():                  # 已知量按给定值精确回填（免去往返误差）
        best[n] = float(v)
    info = dict(mb=best["mb"], ms=best["ms"], combo_err=None, known=dict(known))
    return best, info


def _select_joint(d: dict, want_s: bool) -> None:
    """按 ``joint`` 列就地过滤（1=关节 s 小 yaw；缺列/0=关节 b 大 yaw）。

    两种采集分开成两个类别目录时不会混，但同一个目录里放了两类文件、
    或者把 s 数据指给了大 yaw 的选项时，必须给出干净的报错而不是算出一堆垃圾。
    """
    j = d.get("joint")
    if j is None:
        if want_s:
            raise ValueError(
                "这份数据的 npz 里没有 joint=1 标记，不是小 yaw 匀速往返数据；"
                "大 yaw 数据请用 --friction-sweep / --friction-category")
        return
    mask = (j > 0.5) if want_s else (j <= 0.5)
    if not mask.any():
        raise ValueError(
            ("找不到 joint=1（小 yaw）的行；" if want_s
             else "找不到 joint=0（大 yaw）的行；")
            + "检查是否把两类数据指反了")
    for k in list(d):
        d[k] = np.asarray(d[k])[mask]


def _G2_of(row: dict, p_dyn: dict, key_sin: str = "mean_sin_psi_s",
           key_cos: str = "mean_cos_psi_s") -> float:
    """按已辨识参数算某个窗口的平均重力矩 G2（关节 s 方程里那一项）。"""
    ms, Psx, Psy = p_dyn["ms"], p_dyn["Psx"], p_dyn["Psy"]
    gx = float(row["gx"]) if "gx" in row else 0.0
    gy = float(row["gy"]) if "gy" in row else 0.0
    gs_sin = ms * (gx * Psx + gy * Psy)
    gs_cos = ms * (gx * Psy - gy * Psx)
    return gs_sin * float(row[key_sin]) + gs_cos * float(row[key_cos])


def _M22_of(p_dyn: dict) -> float:
    """关节 s 的等效惯量 M22 = Is + ms|Ps|²（与 θ_s 无关）。"""
    return p_dyn["Is"] + p_dyn["ms"] * (p_dyn["Psx"] ** 2 + p_dyn["Psy"] ** 2)


def _M12_of(p_dyn: dict, theta_s: float) -> float:
    """耦合惯量 M12 = Is_ + ms·h(θ_s)（h 见 dynamics.cpp 的 computeKinematics）。"""
    Dx, Dy, Psx, Psy = p_dyn["Dx"], p_dyn["Dy"], p_dyn["Psx"], p_dyn["Psy"]
    c, s = np.cos(theta_s), np.sin(theta_s)
    h = Dx * Psx * c + Dy * Psy * c + Dy * Psx * s - Dx * Psy * s
    return _M22_of(p_dyn) + p_dyn["ms"] * h


def fit_friction_sweep_s(sweep_path, p_dyn: dict | None = None,
                         dyn_correct: bool = True,
                         drop_first: int = 0, drop_last: int = 0,
                         vmin: float | None = None, vmax: float | None = None,
                         verbose: bool = False,
                         list_only: bool = False) -> tuple[float, float, dict]:
    """用小 yaw（关节 s）匀速往返数据独立拟合 (fsc, fsv)。

    稳态关系（θ̈_b≈0、θ̈_s≈0、θ̇_b≈0、基座静止，见 gen_friction_sweep_s.py）：
        Ts = G2(ψ_s) + fsv·θ̇_s + fsc·tanh(λ·θ̇_s)

    数据里每个速度有正、反两趟，把它们配对做差：
        ΔTs = fsv·Δθ̇_s + fsc·Δtanh(λθ̇_s) + [M22·Δθ̈_s + M12·Δθ̈_b + ΔC2 + ΔG2]
    两趟扫过同一以 0 为中心**对称**的位置窗口 ⇒ **ΔG2 一阶精确抵消**（这是本方法
    的核心，也是它比"用辨识参数直接扣 G2"稳的原因：G2 常比摩擦大一个量级）。

    ``dyn_correct=True``（默认）时再用 p_dyn 扣掉残余的 M22·Δθ̈_s、M12·Δθ̈_b 与 ΔG2：
    实机大 yaw 被机械抱死时这些项本来就接近 0（扣不扣几乎无差），但仿真里大 yaw
    只靠位置环抱、M12 又大，不扣会偏 100%+（实测）。ΔC2 = −ms·h'·θ̇_b² 未扣
    （需要逐样本的 h'·θ̇_b²，采集未记录；θ̇_b 已被阈值限制在很小的范围）。

    速度点的取舍（按数据里**实际存在的 |ω|** 从小到大排序后）：
      * ``drop_first``/``drop_last``：丢掉最慢的 n 个 / 最快的 m 个速度点；
      * ``vmin``/``vmax``：只保留 |ω| 落在 [vmin, vmax] 内的速度点；
      两者可叠加。``verbose=True`` 会先把数据里全部 |ω| 和实际使用的 |ω| 打出来。

    sweep_path 可以是单个 npz 也可以是类别目录（合并其中全部 sweep_*.npz）。
    返回 (fsc, fsv, 诊断信息)；诊断里带"纯差分（完全不依赖已辨识参数）"的结果，
    以及 ``all_omega`` / ``used_omega`` / ``dropped_omega`` 三个列表。
    """
    d, sweep_files = _load_sweeps(sweep_path)
    _select_joint(d, want_s=True)
    n = int(np.asarray(d["omega"]).size)
    if n == 0:
        raise ValueError(f"{sweep_path} 里没有小 yaw 数据行")

    # joint 过滤后每列都已对齐；逐"趟"打包成 dict 列表
    rows = [{k: float(np.asarray(v)[i]) for k, v in d.items()} for i in range(n)]

    def mp(rs, k):
        return float(np.mean([r[k] for r in rs]))

    def pair_delta(pp: list[dict], pm: list[dict]) -> dict:
        """一对（正向若干趟 vs 反向若干趟）的差分量。"""
        d_om = mp(pp, "omega") - mp(pm, "omega")
        d_th = mp(pp, "tanh_omega") - mp(pm, "tanh_omega")
        d_ts = mp(pp, "ts_mean") - mp(pm, "ts_mean")
        corr = 0.0
        if dyn_correct and p_dyn is not None:
            corr += _M22_of(p_dyn) * (mp(pp, "ddtheta_s_mean") - mp(pm, "ddtheta_s_mean"))
            m12 = 0.5 * (_M12_of(p_dyn, np.radians(mp(pp, "theta_s_mean_deg")))
                         + _M12_of(p_dyn, np.radians(mp(pm, "theta_s_mean_deg"))))
            if "ddtheta_b_mean" in pp[0]:
                corr += m12 * (mp(pp, "ddtheta_b_mean") - mp(pm, "ddtheta_b_mean"))
            corr += _G2_of(pp[0], p_dyn) - _G2_of(pm[0], p_dyn)   # 窗口没完全对齐时的残余
        return dict(d_omega=d_om, d_tanh=d_th, d_ts=d_ts, corr=corr,
                    d_b=mp(pp, "theta_b_mean_deg") - mp(pm, "theta_b_mean_deg"),
                    omega=0.5 * abs(d_om), n_p=len(pp), n_m=len(pm))

    def _fmt(vs):
        return "  ".join(f"{x:.4g}" for x in vs) if len(vs) else "（无）"

    def select_speeds(pairs_in: list[dict]) -> tuple[list[dict], dict]:
        """按 |ω| 从小到大排序后做范围/首尾剔除，并返回打印用的信息。"""
        ordered = sorted(pairs_in, key=lambda q: q["omega"])
        all_w = [q["omega"] for q in ordered]
        keep = np.ones(len(ordered), dtype=bool)
        if drop_first > 0:
            keep[:int(drop_first)] = False
        if drop_last > 0:
            keep[len(ordered) - int(drop_last):] = False
        if vmin is not None:
            keep &= np.array([q["omega"] >= vmin for q in ordered])
        if vmax is not None:
            keep &= np.array([q["omega"] <= vmax for q in ordered])
        used = [q for q, k in zip(ordered, keep) if k]
        sel = dict(all_omega=all_w, used_omega=[q["omega"] for q in used],
                   dropped_omega=[q["omega"] for q, k in zip(ordered, keep) if not k],
                   keep=[int(k) for k in keep])
        if verbose:
            how = []
            if drop_first:
                how.append(f"丢最慢 {drop_first} 个")
            if drop_last:
                how.append(f"丢最快 {drop_last} 个")
            if vmin is not None:
                how.append(f"|ω| ≥ {vmin:g}")
            if vmax is not None:
                how.append(f"|ω| ≤ {vmax:g}")
            print(f"  数据里的全部 |ω|（{len(all_w)} 个）: {_fmt(all_w)}")
            print(f"  取舍规则: {' + '.join(how) if how else '不筛，全用'}")
            print(f"  实际使用（{len(sel['used_omega'])} 个）: {_fmt(sel['used_omega'])}")
            if sel["dropped_omega"]:
                print(f"  剔除（{len(sel['dropped_omega'])} 个）: {_fmt(sel['dropped_omega'])}")
        if len(used) < 2:
            raise ValueError(
                f"筛选后只剩 {len(used)} 个速度点（全部 {len(all_w)} 个：{_fmt(all_w)}），"
                f"至少需要 2 个才能拟合 fsc/fsv")
        return used, sel

    dirn = np.sign(np.asarray(d["direction"])) if "direction" in d else np.zeros(n)
    pid = np.asarray(d["pair_id"]).astype(int) if "pair_id" in d else np.zeros(n, dtype=int)
    has_pair = bool((dirn > 0).any() and (dirn < 0).any())
    pairs = []
    if has_pair:
        for p in sorted({int(x) for x in pid}):
            idx = np.where(pid == p)[0]
            pp = [rows[i] for i in idx if dirn[i] > 0]
            pm = [rows[i] for i in idx if dirn[i] < 0]
            if pp and pm:
                pairs.append(pair_delta(pp, pm))

    sel_info = dict(all_omega=[], used_omega=[], dropped_omega=[], keep=[])
    if pairs:
        pairs, sel_info = select_speeds(pairs)

    if pairs and list_only:                    # 只列速度点、不拟合（--list）
        return float("nan"), float("nan"), dict(
            n=n, pairs=len(pairs), files=len(sweep_files), names=sweep_files,
            mode="list", fsc_raw=float("nan"), fsv_raw=float("nan"),
            residual=float("nan"), **sel_info)

    if not pairs:
        # 没有配对信息（旧数据）：退回"直接用 p_dyn 扣 G2"的单趟拟合
        if p_dyn is None:
            raise ValueError(
                f"{sweep_path} 里没有 pair_id/direction 配对列，单趟拟合又需要已辨识参数")
        A = np.column_stack([np.asarray(d["omega"]), np.asarray(d["tanh_omega"])])
        y = np.asarray(d["ts_mean"]) - np.array([_G2_of({k: float(np.asarray(v)[i])
                                                         for k, v in d.items()}, p_dyn)
                                                 for i in range(n)])
        (fsv, fsc), *_ = np.linalg.lstsq(A, y, rcond=None)
        info = dict(n=n, pairs=0, files=len(sweep_files), names=sweep_files,
                    mode="single", fsc_raw=float(fsc), fsv_raw=float(fsv),
                    omega_abs_min=float(np.abs(A[:, 0]).min()),
                    omega_abs_max=float(np.abs(A[:, 0]).max()),
                    residual=float(np.std(y - A @ np.array([fsv, fsc]))),
                    **sel_info)
        return float(fsc), float(fsv), info

    A = np.array([[q["d_omega"], q["d_tanh"]] for q in pairs])
    y_raw = np.array([q["d_ts"] for q in pairs])
    y = y_raw - np.array([q["corr"] for q in pairs])

    # 稳健化：先解一次，剔除修正后仍明显离群的对，再解一次。
    # 一对坏数据（大 yaw 没被真正抱死的那一趟）就能把 fsv 拽偏 50%+（实测），
    # 阈值 = max(4·1.4826·MAD, 5e-3 N·m)——5e-3 就是力矩测量/量化噪声量级。
    dropped: list[int] = []
    if len(y) >= 5:
        c0 = np.linalg.lstsq(A, y, rcond=None)[0]
        r0 = y - A @ c0
        mad = 1.4826 * float(np.median(np.abs(r0 - np.median(r0))))
        keep = np.abs(r0 - np.median(r0)) <= max(4.0 * mad, 5e-3)
        if 3 <= int(keep.sum()) < len(y):
            dropped = [int(i) for i in np.where(~keep)[0]]
            A, y = A[keep], y[keep]

    (fsv, fsc), *_ = np.linalg.lstsq(A, y, rcond=None)
    (fsv_raw, fsc_raw), *_ = np.linalg.lstsq(
        np.array([[q["d_omega"], q["d_tanh"]] for q in pairs]), y_raw, rcond=None)
    resid = y - A @ np.array([fsv, fsc])
    info = dict(n=n, pairs=len(pairs), files=len(sweep_files), names=sweep_files,
                mode="diff", fsc_raw=float(fsc_raw), fsv_raw=float(fsv_raw),
                dropped=dropped, kept=int(len(y)), **sel_info,
                residual=float(np.std(resid)),
                cond=float(np.linalg.cond(A)),
                omega_abs_min=float(min(q["omega"] for q in pairs)),
                omega_abs_max=float(max(q["omega"] for q in pairs)),
                d_ts_max=float(np.abs(y_raw).max()))
    return float(fsc), float(fsv), info


def parse_stages(spec: str) -> list[tuple[int, float]]:
    """解析 "K:权重,K:权重,..." 形式的课程表，权重归一化到和为 1。"""
    out: list[tuple[int, float]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        k_s, w_s = part.split(":")
        out.append((int(k_s), float(w_s)))
    if not out:
        raise ValueError(f"课程表为空: {spec!r}")
    total_w = sum(w for _, w in out)
    return [(k, w / total_w) for k, w in out]


def _load_sweeps(path) -> tuple[dict, list[str]]:
    """加载摩擦 sweep 数据（单文件或目录）。

    path 是目录时，合并其中**全部** ``sweep_*.npz``（没有匹配则退回 ``*.npz``），
    用来把同一类别下多次运行的结果拼起来。列取各文件的并集，缺失的列按该文件的
    行数补 0（旧文件可能没有 real 等较新的列）。非数值列忽略。

    返回 (列字典, 用到的文件名列表)。
    """
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("sweep_*.npz"))
        if not files:
            files = sorted(p.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"目录里没有找到 sweep npz: {p}")
    else:
        files = [p]

    per_file: list[dict] = []
    for f in files:
        with np.load(f) as z:
            per_file.append({k: np.asarray(z[k]) for k in z.files})

    keys: set[str] = set()
    for d in per_file:
        keys.update(d.keys())

    def _n(d: dict) -> int:
        return int(np.asarray(d["omega"]).size)

    merged: dict[str, np.ndarray] = {}
    for k in sorted(keys):
        cols = []
        for d in per_file:
            v = d.get(k)
            if v is None:
                cols.append(np.zeros(_n(d), dtype=np.float64))
            elif v.ndim == 0:
                cols.append(np.full(_n(d), float(v), dtype=np.float64))
            elif np.issubdtype(v.dtype, np.number):
                cols.append(np.asarray(v, dtype=np.float64).ravel())
            else:
                cols.append(np.zeros(_n(d), dtype=np.float64))
        merged[k] = np.concatenate(cols) if cols else np.zeros(0)
    return merged, [f.name for f in files]


def fit_friction_sweep(sweep_path, p_dyn: dict) -> tuple[float, float, dict]:
    """用匀速旋转实验的稳态数据拟合关节 b 的两个摩擦系数。

    稳态关系（θ̈_b=θ̈_s=0、θ̇_s≈0、基座静止，见 gen_friction_sweep.py 的推导）：
        Tb = G1(ψ_b) + fbv·ω + fbc·tanh(λ·ω)
    用 p_dyn（已辨识的动力学参数）把 G1 精确减掉，剩下的就是 (ω, tanh(λω)) 的二维
    线性最小二乘——实测残差 ~3e-4 N·m、条件数 ~2，比主回归（cond 1e18）好十几个
    数量级，所以摩擦能真正解出来。

    sweep_path 可以是单个 npz，也可以是**类别目录**（合并其中全部 sweep_*.npz，
    即该类别下所有采集运行）。

    返回 (fbc, fbv, 诊断信息)。
    """
    d, sweep_files = _load_sweeps(sweep_path)
    _select_joint(d, want_s=False)
    ms, Psx, Psy = p_dyn["ms"], p_dyn["Psx"], p_dyn["Psy"]
    mb, Pbx, Pby = p_dyn["mb"], p_dyn["Pbx"], p_dyn["Pby"]
    Dx, Dy = p_dyn["Dx"], p_dyn["Dy"]
    P, Q2 = ms * Psx, ms * Psy
    R, S = mb * Pbx, mb * Pby
    U, V = ms * Dx, ms * Dy
    gx, gy = d["gx"], d["gy"]
    gb_sin = gx * (R + U) + gy * (S + V)
    gb_cos = gx * (S + V) - gy * (R + U)
    gs_sin = gx * P + gy * Q2
    gs_cos = gx * Q2 - gy * P
    G1 = (gb_sin * d["mean_sin_psi_b"] + gb_cos * d["mean_cos_psi_b"]
          + gs_sin * d["mean_sin_psi_s"] + gs_cos * d["mean_cos_psi_s"])
    yv = d["tb_mean"] - G1
    A = np.column_stack([d["omega"], d["tanh_omega"]])
    (fbv, fbc), *_ = np.linalg.lstsq(A, yv, rcond=None)
    resid = yv - A @ np.array([fbv, fbc])
    info = dict(n=int(len(yv)), files=len(sweep_files), names=sweep_files,
                residual=float(np.std(resid)),
                cond=float(np.linalg.cond(A)),
                omega_abs_min=float(np.abs(d["omega"]).min()),
                omega_abs_max=float(np.abs(d["omega"]).max()))
    return float(fbc), float(fbv), info


# 默认课程：窗口从 20 步逐步拉长到 300 步，权重集中在长窗口。
# 有了代数初值（--init algebraic，默认）之后，K=5/10 这类极小窗口已无必要——
# 它们信息量不足、只会让参数在平坦谷里漂；K=20 起就够（加速度已可观测）。
# 若改用随机初值（--init random），可自行加回小窗口，但实测那样也救不了 2 个数量级。
# DEFAULT_STAGES = "20:0.08,50:0.10,100:0.12,200:0.20,300:0.50"
DEFAULT_STAGES = "20:0.2,50:0.2,100:0.2,200:0.2,300:0.2"


def report_init(init_p: dict, ref: dict, ref_ident: dict, has_truth: bool,
                init_kind: str, out_dir: Path,
                cases: list["Case"], plot_pick, refinement: int) -> None:
    """梯度优化开始前：打印初值并画两张图（参数对比 + 位置曲线对比）。

    ``--init algebraic``（默认）时 ``init_p`` 就是闭式最小二乘（algebraic_init）解出的初值。
    保存到 ``out_dir``：
      * ``init_algebraic.png``：左 = 14 个参数绝对值（symlog）初值 vs 真值/标称；
        右 = 12 个可辨识组合的相对误差；
      * ``init_trajectory_algebraic.png``：``plot_pick`` 指定的几条数据上，
        用初值参数前向仿真的 ψ_b/ψ_s 位置曲线 vs 实测（有真值再叠真值曲线）。
    随机初值时文件名里是 ``random``。
    """
    base = "真值" if has_truth else "标称"
    kind = "代数最小二乘解" if init_kind == "algebraic" else "随机初值"
    # 图里一律用英文：matplotlib 默认字体没有中文字形，中文会渲染成方框
    base_en = "truth" if has_truth else "nominal"
    kind_en = "algebraic LSQ" if init_kind == "algebraic" else "random"
    print(f"\n=== 梯度优化前的初值（{kind}，基准：{base}）===")
    if has_truth:
        print(f"{'参数':>8} {'真值':>14} {'初值':>14} {'相对误差':>12}")
        for n in PARAM_NAMES:
            r = (init_p[n] - ref[n]) / max(abs(ref[n]), 1e-12)
            print(f"{n:>8} {ref[n]:>14.6g} {init_p[n]:>14.6g} {r:>12.2e}")
    else:
        print(f"{'参数':>8} {'初值':>14}")
        for n in PARAM_NAMES:
            print(f"{n:>8} {init_p[n]:>14.6g}")
    print(f"可辨识组合最大相对误差（基准 {base}）: {combo_error(init_p, ref_ident):.4g}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = list(PARAM_NAMES)
        refv = np.array([ref[n] for n in names], dtype=float)
        initv = np.array([init_p[n] for n in names], dtype=float)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

        # 左：14 个参数的绝对值（symlog 同时容纳正负与跨数量级）
        ax = axes[0]
        y = np.arange(len(names))
        h = 0.38
        ax.barh(y + h / 2, initv, height=h, color="tab:blue",
                label=f"initial ({kind_en})")
        ax.barh(y - h / 2, refv, height=h, color="k", alpha=0.45, label=base_en)
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xscale("symlog", linthresh=1e-3)
        ax.axvline(0.0, color="k", lw=0.8)
        ax.set_xlabel("parameter value (symlog)")
        ax.set_title(f"14 parameters: initial vs {base_en}")
        ax.legend(fontsize=8)
        ax.grid(True, axis="x", alpha=0.3)

        # 右：12 个可辨识组合的相对误差
        labels = [lab for lab, _ in IDENTIFIABLE]
        refc = np.array([fn(ref) for _, fn in IDENTIFIABLE], dtype=float)
        initc = np.array([fn(init_p) for _, fn in IDENTIFIABLE], dtype=float)
        relc = (initc - refc) / np.maximum(np.abs(refc), 1e-12)
        ax = axes[1]
        yy = np.arange(len(labels))
        ax.barh(yy, relc,
                color=["tab:red" if abs(v) > 0.05 else "tab:green" for v in relc])
        ax.set_yticks(yy)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0.0, color="k", lw=0.8)
        ax.set_xlabel(f"relative error (initial - {base_en}) / |{base_en}|")
        ax.set_title(f"12 identifiable combinations (max |rel| = {np.max(np.abs(relc)):.3g})")
        ax.grid(True, axis="x", alpha=0.3)

        fig.suptitle("Initial parameters BEFORE gradient optimization", fontsize=13)
        p = out_dir / ("init_algebraic.png" if init_kind == "algebraic"
                       else "init_random.png")
        fig.savefig(p, dpi=110)
        plt.close(fig)
        print(f"初值图已保存: {p}")

        # ---- 位置曲线对比：用初值参数前向仿真 vs 实测（有真值再叠一条真值）----
        import contextlib
        pick = np.atleast_1d(plot_pick)
        fig, axes = plt.subplots(len(pick), 2, figsize=(14, 3.2 * len(pick)),
                                 constrained_layout=True)
        axes = np.atleast_2d(axes)
        with contextlib.ExitStack() as stack:
            pg_i = stack.enter_context(
                ParamGradient(make_params(cases[0], init_p), refinement=refinement))
            pg_t = (stack.enter_context(
                        ParamGradient(make_params(cases[0], ref_ident),
                                      refinement=refinement))
                    if has_truth else None)
            for row, ci in enumerate(pick):
                c = cases[int(ci)]
                # 每条曲线都必须用该条数据自己的已知重力
                pg_i.set_params(make_params(c, init_p))
                _, *si = pg_i.loss(c.theta_c0, c.dtheta_c, c.ddtheta_c, c.dt, c.tau,
                                   c.x0, case_spec(c, c.K), return_sequences=True)
                st = None
                if pg_t is not None:
                    pg_t.set_params(make_params(c, ref_ident))
                    _, *st = pg_t.loss(c.theta_c0, c.dtheta_c, c.ddtheta_c, c.dt, c.tau,
                                       c.x0, case_spec(c, c.K), return_sequences=True)
                t = (np.arange(c.K) + 1) * c.dt
                for col, (meas, lab) in enumerate(((c.psi_b, "psi_b"), (c.psi_s, "psi_s"))):
                    ax = axes[row][col]
                    ax.plot(t, meas, ".", ms=1.5, alpha=0.4, label="measured")
                    if st is not None:
                        ax.plot(t, st[col], lw=1.2, label=f"{base_en} params")
                    ax.plot(t, si[col], lw=1.2, ls="--", label=f"initial ({kind_en})")
                    ax.set_title(f"case {int(ci)}: {lab}", fontsize=10)
                    ax.grid(True, alpha=0.3)
                    if row == 0:
                        ax.legend(fontsize=7)
        fig.suptitle("Measured vs INITIAL-parameter position trajectories "
                     "(before gradient optimization)", fontsize=13)
        p2 = out_dir / ("init_trajectory_algebraic.png" if init_kind == "algebraic"
                        else "init_trajectory_random.png")
        fig.savefig(p2, dpi=110)
        plt.close(fig)
        print(f"初值位置曲线已保存: {p2}")
    except ImportError:
        print("（未安装 matplotlib，跳过初值绘图）")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", type=str, default=None,
                    help="类别名（= 目录名）：读 data/<类别>/ 下**全部** case_*.npz"
                         "（多次采集运行的时间戳文件会一起用上）；被 --data 覆盖")
    ap.add_argument("--data", type=str, default=None,
                    help="数据目录（优先于 --category）；缺省时用 --category 或 data/sim")
    ap.add_argument("--steps", type=int, default=10000,
                    help="总优化步数，按 --stages 的权重分配到各阶段")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-mode", choices=("batch", "loop"), default="batch",
                    help="梯度精修用哪种参数梯度接口：batch=批量 SoA（默认，"
                         "样本维 SIMD + 多线程）；loop=逐条调用单序列接口（原实现，"
                         "用于对照/调试）")
    ap.add_argument("--batch-threads", type=int, default=0,
                    help="批量梯度的线程数；0 = 用硬件并发")
    ap.add_argument("--batch-lanes", type=int, default=16,
                    help="批量梯度每个 SIMD 分块处理的样本数上限")
    ap.add_argument("--lr", type=float, default=0.0075,
                    help="Adam 学习率；theta 空间极窄的稳定区间，见文件末尾说明")
    ap.add_argument("--loss-scale", type=float, default=1.0,
                    help="对均值 loss 乘的系数（Adam 会归一化梯度，影响很小）")
    # ---- loss 四项权重：位置(psi)与速度(dpsi) × 大/小 yaw(b/s) ----
    # 不给就用数据里记录的 weights（采集时由 sim_config.W_PSI_* 写入）
    ap.add_argument("--w-psi-b", type=float, default=None,
                    help="位置项 ψ_b 权重；默认用数据里记录的 weights")
    ap.add_argument("--w-psi-s", type=float, default=None,
                    help="位置项 ψ_s 权重；默认用数据里记录的 weights")
    ap.add_argument("--w-dpsi-b", type=float, default=None,
                    help="速度项 dψ_b 权重；默认用数据里记录的 weights")
    ap.add_argument("--w-dpsi-s", type=float, default=None,
                    help="速度项 dψ_s 权重；默认用数据里记录的 weights")
    ap.add_argument("--no-full-batch-final", dest="full_batch_final", action="store_false",
                    help="最后一个课程阶段也只用 batch（默认改为全量，否则组合误差停在 ~3%%）")
    ap.add_argument("--stages", type=str, default=DEFAULT_STAGES,
                    help='短窗口课程表 "K:权重,..."，窗口逐步拉长到全窗口')
    ap.add_argument("--no-curriculum", dest="curriculum", action="store_false",
                    help="关闭课程学习：只用全窗口 K 从头训到底")
    ap.add_argument("--window-stride", type=float, default=0.5,
                    help="短窗口阶段的滑窗步长，以该阶段的 K 为单位（默认 0.5 = K/2）；"
                         "只在 K < full_K 的阶段生效")
    ap.add_argument("--no-sliding-windows", dest="sliding_windows",
                    action="store_false",
                    help="关闭短窗口滑窗：K < full_K 的阶段仍只用每条数据开头的 K 步"
                         "（旧行为，用于对照）")
    ap.add_argument("--eval-every", type=int, default=100,
                    help="每隔多少步在全窗口全量数据上评估一次 loss")
    ap.add_argument("--init", choices=("algebraic", "random"), default="algebraic",
                    help="algebraic=先用闭式最小二乘解出聚合量（与初值无关，推荐）；"
                         "random=用随机初值（用于对照）")
    ap.add_argument("--mass-nominal", type=float, nargs=2, default=(1.0, 1.0),
                    metavar=("MB", "MS"),
                    help="代数初始化中两个不可辨识标度 mb/ms 的标称值（不影响 loss）；"
                         "若 --known-params 给了 mb/ms，则该项被覆盖")
    ap.add_argument("--friction-category", type=str, default=None,
                    help="摩擦类别名（= 目录名）：读 data/<名字>/ 下**全部** sweep_*.npz；"
                         "被 --friction-sweep 覆盖")
    ap.add_argument("--friction-sweep", type=str, default=None,
                    help="匀速旋转摩擦实验数据：可以是单个 npz，也可以是目录"
                         "（合并其中全部 sweep_*.npz）；缺省时用 --friction-category "
                         "或 data/friction。用它独立拟合 fbc/fbv，关节 s 取一半")
    ap.add_argument("--no-friction-sweep", dest="use_friction_sweep",
                    action="store_false",
                    help="不用匀速旋转实验，摩擦仍由主回归给出（实测会解成负值）")
    ap.add_argument("--friction-category-s", type=str, default=None,
                    help="**小 yaw** 摩擦类别名（= 目录名）：读 data/<名字>/ 下全部 "
                         "sweep_*.npz（gen_friction_sweep_s.py 的产物）；"
                         "被 --friction-sweep-s 覆盖")
    ap.add_argument("--friction-sweep-s", type=str, default=None,
                    help="**小 yaw** 匀速往返数据：单个 npz 或目录；缺省用 "
                         "--friction-category-s 或 data/friction_s。用它**独立**拟合 "
                         "fsc/fsv（正反趟差分，不依赖大 yaw 摩擦）；找不到就回退成"
                         "『大 yaw 摩擦的一半』")
    ap.add_argument("--no-friction-sweep-s", dest="use_friction_sweep_s",
                    action="store_false",
                    help="不用小 yaw 数据，fsc/fsv 直接取大 yaw 的一半（旧行为）")
    ap.add_argument("--fs-s-vmin", type=float, default=None,
                    help="小 yaw 独立拟合时，只用 |ω| ≥ 该值的数据点 [rad/s]")
    ap.add_argument("--fs-s-vmax", type=float, default=None,
                    help="小 yaw 独立拟合时，只用 |ω| ≤ 该值的数据点 [rad/s]")
    ap.add_argument("--fs-s-drop-first", type=int, default=0,
                    help="小 yaw 独立拟合时丢掉最慢的 n 个速度点（按数据里实际存在的 |ω| 排序）")
    ap.add_argument("--fs-s-drop-last", type=int, default=0,
                    help="小 yaw 独立拟合时丢掉最快的 m 个速度点")
    ap.add_argument("--fs-s-no-dyn", dest="fs_s_dyn", action="store_false",
                    help="小 yaw 差分拟合**不扣** M22·Δθ̈_s + M12·Δθ̈_b + ΔG2，"
                         "完全不依赖已辨识参数（纯差分；大 yaw 被机械抱死时两者几乎无差）")
    ap.add_argument("--init-orders", type=str, default=f"{LOG10_ORDERS_MIN},{LOG10_ORDERS_MAX}",
                    help='--init random 时，初值偏离真值的数量级区间 "min,max"')
    ap.add_argument("--freeze-params", type=str, default="",
                    help='逐参数冻结表 "参数:步数,..."：在代数最小二乘初始化之后固定'
                         '这些参数，前 n 个全局 Adam step 内不优化，第 n 步起解冻；'
                         'n 每个参数单独配置。省略步数 = 全程冻结，步数给 0 = 不冻结。'
                         '例: --freeze-params "fbc:3000,fbv:3000,Dx" 。'
                         f"可选参数: {', '.join(PARAM_NAMES)}")
    ap.add_argument("--known-params", type=str, default="",
                    help='外部已知参数表 "参数=值,..."：代数初始化时当已知量用'
                         '（Dx,Dy 解析折掉 A+B/C−D 两列；摩擦搬到右端；mb,ms 直接'
                         '固定标度、跳过网格搜索），并默认全程冻结在给定值上。'
                         '例: --known-params "Dx=0.05,Dy=0.30"。'
                         '若想让某个已知值只当初值、训练时放开，'
                         '用 --freeze-params "Dx:0"（或给个有限步数先冻后放）。'
                         f"可用的已知参数: {', '.join(sorted(KNOWN_OK))}")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=str, default=None,
                    help="结果目录；缺省时 --category 给定时用 data/identify_<类别>，"
                         "否则用 data/identify")
    args = ap.parse_args()
    # 冻结表 / 已知参数表尽早校验，参数写错时给干净的报错而不是后面的 traceback
    try:
        freeze_until = parse_freeze_params(args.freeze_params)
        known = parse_known_params(args.known_params)
    except ValueError as e:
        ap.error(str(e))
    # 已知参数默认全程冻结；表里明确写了 0（= 不冻结）或有限步数时按表来
    for n in known:
        freeze_until.setdefault(n, math.inf)

    def _stamp_of(name: str) -> str:
        """从 case_<时间戳>_<序号>.npz 里解析出时间戳（旧的无戳文件返回 unknown）。"""
        parts = Path(name).stem.split("_")
        return f"{parts[1]}_{parts[2]}" if len(parts) >= 4 else "unknown"

    data_dir = (Path(args.data) if args.data
                else (cfg.category_dir(args.category) if args.category else DATA_DIR))
    files = sorted(data_dir.glob("case_*.npz"))
    if not files:
        print(f"未找到数据: {data_dir}")
        return 1
    stamps = sorted({_stamp_of(f.name) for f in files})
    truth_path = data_dir / "truth_params.txt"
    has_truth = truth_path.exists()
    truth = load_truth(truth_path) if has_truth else None
    # 没有真值文件（例如真机采集的数据）时：随机初值/组合对比需要一组数值基准，
    # 退回配置里的标称参数。它只参与计算，不被当成"真值"画出来或打印对比。
    ref = truth if has_truth else dict(cfg.DEFAULT_PARAMS)
    out_dir = (Path(args.out) if args.out
               else (cfg.category_dir(f"identify_{args.category}") if args.category
                     else REPO / "data" / "identify"))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"数据类别: {args.category or data_dir}   目录: {data_dir}   文件: {len(files)} 个"
          f"（{len(stamps)} 次运行: {', '.join(stamps)}）")
    if has_truth:
        print(f"真值文件: {truth_path}")
    else:
        print(f"真值文件: 无（{truth_path} 不存在）——不画真值曲线、不做真值对比")

    cases = [Case(f) for f in files]
    # loss 四项权重可用命令行覆盖；只给其中几项时，其余仍用数据里记录的值
    w_override = (args.w_psi_b, args.w_psi_s, args.w_dpsi_b, args.w_dpsi_s)
    if any(v is not None for v in w_override):
        for c in cases:
            c.weights = tuple(cur if v is None else float(v)
                              for v, cur in zip(w_override, c.weights))
    N = len(cases)
    dt = cases[0].dt
    # 拟合用的 refinement（RK4 子步数）是**拟合侧的配置**，固定取 cfg.REFINEMENT，
    # 不读每条 npz 里记录的值（采集时写进文件的只作数据溯源，不参与拟合）。
    refinement = int(cfg.REFINEMENT)
    # 位置曲线对比用的固定抽样：用独立生成器，既不影响训练用的 rng（批次调度），
    # 又让"初值图"和"最终辨识图"画的是同一批数据，方便前后对比。
    plot_pick = np.random.default_rng(args.seed + 2024).choice(
        N, size=min(3, N), replace=False)
    if len({c.dt for c in cases}) != 1:
        print("各条数据的 dt 不一致，暂不支持")
        return 1
    g_range = (min(min(c.gx, c.gy) for c in cases), max(max(c.gx, c.gy) for c in cases))
    print(f"数据: {N} 条   dt={dt}s ({1/dt:.0f}Hz)   K={cases[0].K}   "
          f"refinement={refinement}（取自 cfg.REFINEMENT）")
    wsrc = "命令行覆盖" if any(v is not None for v in w_override) else "取自数据"
    print(f"loss 权重（{wsrc}）: 位置 psi_b={cases[0].weights[0]:g} "
          f"psi_s={cases[0].weights[1]:g}   速度 dpsi_b={cases[0].weights[2]:g} "
          f"dpsi_s={cases[0].weights[3]:g}")
    print(f"被辨识参数: {len(PARAM_NAMES)} 个（重力 gx/gy 为已知输入，取值跨度 "
          f"[{g_range[0]:.2f}, {g_range[1]:.2f}]）")
    print(f"优化: {args.steps} 步, batch={args.batch}, Adam lr={args.lr}")
    effective_bs = min(args.batch, N)
    print(f"每 epoch batch 数 ≈ {max(1, N // effective_bs)}   "
          f"(数据 {N} 条 / batch {effective_bs})")

    full_K = cases[0].K
    # 批量梯度的整块数据：预先拼一次，训练时只按 batch 取子集
    batch_data = pack_batch_data(cases, LAMBDA) if args.grad_mode == "batch" else None

    # ------------------------------------------------------------------
    # 短窗口样本池：K < full_K 的阶段把每条数据的滑窗都取出来当样本
    # ------------------------------------------------------------------
    def stage_pool(K_s: int) -> tuple[np.ndarray, np.ndarray, int]:
        """该阶段的样本池，返回 (数据下标, 窗口起点, 每条数据的窗口数)。

        * K_s == full_K（整段数据）或 --no-sliding-windows：每条数据只有
          起点 0 的一个样本，等价于旧行为；
        * 否则以 round(K_s * --window-stride)（默认 K_s/2）为步长滑窗。
        """
        if not args.sliding_windows or K_s >= full_K:
            return (np.arange(N, dtype=np.intp), np.zeros(N, dtype=np.intp), 1)
        stride = max(1, int(round(K_s * args.window_stride)))
        st = window_starts(full_K, K_s, stride)
        ci = np.repeat(np.arange(N, dtype=np.intp), st.size)
        return ci, np.tile(st, N), int(st.size)

    # 课程表：(拟合窗口 K, 该阶段步数)。K 按权重分摊总步数，并逐步拉长到 full_K。
    if args.curriculum:
        plan = [(min(k, full_K), max(1, int(round(args.steps * w))))
                for k, w in parse_stages(args.stages)]
        plan.sort()
        plan[-1] = (full_K, plan[-1][1])       # 最后阶段一定用全窗口
    else:
        plan = [(full_K, args.steps)]
    total_steps = sum(n for _, n in plan)

    # 组合误差/随机初值的数值基准：有真值用真值，没有就用标称（见上面 ref 的说明）
    truth_ident = {n: ref[n] for n in PARAM_NAMES}
    rng = np.random.default_rng(args.seed)
    if args.init == "algebraic":
        # ---- 摩擦：大 yaw 用匀速旋转、小 yaw 用匀速往返，先各自独立拟合再回代 ----
        friction = None
        p_dyn = None
        sweep_path = (Path(args.friction_sweep) if args.friction_sweep
                      else (cfg.category_dir(args.friction_category)
                            if args.friction_category
                            else REPO / "data" / "friction"))
        if args.use_friction_sweep and sweep_path.exists():
            p_dyn, _ = algebraic_init(cases, nominal=tuple(args.mass_nominal),
                                      friction=(0.0, 0.0, 0.0, 0.0), known=known)
            fbc, fbv, finfo = fit_friction_sweep(sweep_path, p_dyn)
            print(f"匀速旋转实验拟合大 yaw 摩擦（{finfo['files']} 个 sweep 文件，"
                  f"{finfo['n']} 个速度点，"
                  f"|ω| {finfo['omega_abs_min']:.4f}~{finfo['omega_abs_max']:.2f} rad/s，"
                  f"残差 {finfo['residual']:.2e} N·m，cond {finfo['cond']:.1f}）:")
            print(f"  fbc={fbc:.5f}  fbv={fbv:.5f}"
                  + (f"   真值 {truth['fbc']:.5f} / {truth['fbv']:.5f}"
                     f"   （相对误差 {abs(fbc-truth['fbc'])/truth['fbc']:.2%} / "
                     f"{abs(fbv-truth['fbv'])/truth['fbv']:.2%}）" if has_truth else ""))
            fsc, fsv = 0.5 * fbc, 0.5 * fbv            # 缺省回退：小 yaw 取一半
            print(f"  关节 s 缺省取一半: fsc={fsc:.5f} fsv={fsv:.5f}"
                  + (f"   真值 {truth['fsc']:.5f} / {truth['fsv']:.5f}"
                     if has_truth else ""))

            # ---- 小 yaw：独立拟合（正反趟差分），优先于"取一半" ----
            s_path = (Path(args.friction_sweep_s) if args.friction_sweep_s
                      else (cfg.category_dir(args.friction_category_s)
                            if args.friction_category_s
                            else cfg.DATA_DIR_FRICTION_S))
            if args.use_friction_sweep_s and s_path.exists():
                if p_dyn is None:
                    p_dyn, _ = algebraic_init(cases, nominal=tuple(args.mass_nominal),
                                              friction=(0.0, 0.0, 0.0, 0.0), known=known)
                try:
                    fsc2, fsv2, sinfo = fit_friction_sweep_s(
                        s_path, p_dyn, dyn_correct=args.fs_s_dyn,
                        drop_first=args.fs_s_drop_first, drop_last=args.fs_s_drop_last,
                        vmin=args.fs_s_vmin, vmax=args.fs_s_vmax, verbose=True)
                except ValueError as e:
                    print(f"（小 yaw 数据 {s_path} 不可用：{e}）")
                else:
                    fsc, fsv = fsc2, fsv2
                    how = ("正反趟差分 + p_dyn 扣 M22·Δθ̈_s/M12·Δθ̈_b/ΔG2"
                           if (args.fs_s_dyn and sinfo["mode"] == "diff")
                           else "单趟 + p_dyn 扣 G2" if sinfo["mode"] == "single"
                           else "纯正反趟差分（不用任何已辨识参数）")
                    print(f"小 yaw 匀速往返拟合关节 s 摩擦（{sinfo['files']} 个文件，"
                          f"{sinfo['n']} 趟，{sinfo['pairs']} 对，"
                          f"|ω| {sinfo['omega_abs_min']:.4f}~{sinfo['omega_abs_max']:.2f} "
                          f"rad/s，残差 {sinfo['residual']:.2e} N·m，{how}）:")
                    print(f"  fsc={fsc:.5f}  fsv={fsv:.5f}"
                          + (f"   真值 {truth['fsc']:.5f} / {truth['fsv']:.5f}"
                             f"   （相对误差 {abs(fsc-truth['fsc'])/truth['fsc']:.2%} / "
                             f"{abs(fsv-truth['fsv'])/truth['fsv']:.2%}）"
                             if has_truth else ""))
                    if sinfo.get("dropped"):
                        print(f"  稳健剔除 {len(sinfo['dropped'])} 对离群（修正后残差 "
                              f"> 4·MAD 且 > 5e-3 N·m），留 {sinfo['kept']} 对")
                    print(f"  对照·纯差分（不扣任何动力学项）: "
                          f"fsc={sinfo['fsc_raw']:.5f} fsv={sinfo['fsv_raw']:.5f}"
                          f"    → 两者差得越多说明大 yaw 没被真正固定住"
                          f"（θ̈_b 残差被 M12 放大）")
            elif args.use_friction_sweep_s:
                print(f"（未找到小 yaw 数据 {s_path}，fsc/fsv 仍取大 yaw 的一半）")
            friction = (fbc, fbv, fsc, fsv)
        elif args.use_friction_sweep:
            print(f"（未找到 {sweep_path}，摩擦改用主回归结果）")
        p_init, ainfo = algebraic_init(cases, nominal=tuple(args.mass_nominal),
                                      friction=friction, known=known)
        ps = ParamSpec(ref, rng, orders=(0.0, 0.0))
        ps.theta = torch.tensor(
            ParamSpec.to_theta([p_init[n] for n in PARAM_NAMES]),
            dtype=torch.float64, requires_grad=True)
        n_col = 12 - (2 if "Dx" in known else 0) - sum(
            1 for n in ("fbc", "fbv", "fsc", "fsv") if n in known)
        scale = ("mb/ms 为已知值（跳过网格搜索）" if "mb" in known and "ms" in known
                 else f"不可辨识标度取 mb={ainfo['mb']:.4g} ms={ainfo['ms']:.4g}")
        print(f"代数初始化（闭式最小二乘，与初值无关，设计矩阵 {n_col} 列）: {scale}")
    else:
        init_orders = tuple(float(v) for v in args.init_orders.split(","))
        ps = ParamSpec(ref, rng, orders=init_orders)
        if known:       # 随机初值也要把已知量写进去（随后按冻结表固定）
            p_rand = ps.seed_values()
            p_rand.update(known)
            with torch.no_grad():
                ps.theta.copy_(torch.tensor(
                    ParamSpec.to_theta([p_rand[n] for n in PARAM_NAMES]),
                    dtype=torch.float64))
        print(f"随机初始化: 每个参数乘 10^(±[{init_orders[0]}, {init_orders[1]}])")
    if known:
        print("已知参数（外部给定，初始化时当已知量、训练中按冻结表固定）: "
              + "，".join(f"{n}={known[n]:.6g}" for n in PARAM_NAMES if n in known))

    # 梯度优化开始前：先把初值（algebraic 时 = 闭式最小二乘解）打印并画一遍图
    init0 = ps.seed_values()
    report_init(init0, ref, truth_ident, has_truth, args.init, out_dir,
                cases, plot_pick, refinement)

    # 逐参数冻结：冻结值就是**初始化（最小二乘 / 已知量注入）之后的值**，第一次更新前抓取
    freeze = FreezeSchedule(freeze_until)
    if freeze:
        freeze.capture(ps)
        active = [n for n in PARAM_NAMES if freeze.until.get(n, 0.0) != 0.0]
        if active:
            print("\n冻结计划（初始化之后生效，每个参数单独配置解冻步；冻结期间数值完全不动）:")
            for n in active:
                print(f"  {n:>5}: {freeze.label(n)}，冻结值 {freeze.values[n]:.6g}")

    init_combo = combo_error(init0, truth_ident)
    init_spread = max(abs(np.log10(abs(init0[n]) / abs(ref[n])))
                      for n in PARAM_NAMES)
    print(f"优化: 共 {total_steps} 步, batch={args.batch}, Adam lr={args.lr}"
          + ("  + 短窗口课程学习" if args.curriculum else "  （无课程学习）"))
    if args.curriculum:
        print("课程表: " + " -> ".join(f"K={k}({n}步)" for k, n in plan))
        if args.sliding_windows:
            slide = [f"K={k}: {N * window_starts(full_K, k, max(1, int(round(k * args.window_stride)))).size} 个样本"
                     for k, _ in plan if k < full_K]
            if slide:
                print("短窗口滑窗（步长 = %.2f×K，每个阶段的样本池）: " % args.window_stride
                      + "，".join(slide))
        else:
            print("短窗口滑窗: 已关闭（--no-sliding-windows），只用每条数据开头的 K 步")
    # 单参数偏差没有意义（mb/ms 两条方向严格不可辨识，参数可沿平坦谷滑很远而
    # loss 完全不变），所以初值好坏只看可辨识组合误差。
    print(f"初值: 可辨识组合最大相对误差={init_combo:.4g}（基准："
          f"{'真值' if has_truth else '标称参数'}），"
          f"单参数最大偏离={init_spread:.2f} 个数量级（不可辨识方向，仅供参考）")
    if has_truth:
        ref_vals = []
        with ParamGradient(make_params(cases[0], truth_ident), refinement=refinement) as pg_ref:
            for c in cases:
                pg_ref.set_params(make_params(c, truth_ident))
                ref_vals.append(case_loss(pg_ref, c, full_K))
        ref_loss = float(np.mean(ref_vals))
        print(f"真值参数下的全窗口平均 loss（噪声底）≈ {ref_loss:.6e}")
    else:
        ref_loss = None      # 无真值：不画噪声底线、不打印真值对比

    hist_loss, hist_theta, hist_K = [], [], []
    hist_eval_step, hist_eval_loss = [], []
    gstep = 0

    # 学习率必须随窗口一起变：损失对参数的敏感度（以及刚度）随积分时长急剧增长，
    # 所以短窗口能承受大得多的步长；而初值差 2~3 个数量级（|Δθ|≈19）时，只有
    # 短窗口阶段的大步长才可能把它拉回来。这里按 lr ∝ 1/K 给每个阶段定步长，
    # 并在阶段内做余弦衰减（warm restart）：阶段开始时步子大、结束时落定。
    def stage_lr_for(K_s: int) -> float:
        return args.lr * 1 # (full_K / max(1, K_s))

    with ParamGradient(make_params(cases[0], ps.seed_values()),
                       refinement=refinement) as pg:
        pgb = None
        if batch_data is not None:
            pgb = ParamGradientBatch(refinement=refinement, max_batch=max(args.batch*2, N),
                                     max_num_steps=full_K,
                                     num_threads=args.batch_threads,
                                     lanes=args.batch_lanes)
            print(f"参数梯度接口: 批量 SoA（threads={pgb.num_threads} lanes={pgb.lanes}，"
                  f"B≤{N}，K≤{full_K}）")
        else:
            print("参数梯度接口: 逐条单序列（--grad-mode loop）")
        for stage_i, (K_s, n_steps) in enumerate(plan):
            lr_s = stage_lr_for(K_s)
            # 最后一个阶段用全量数据：batch=8 的梯度噪声会让组合误差停在 ~3%
            # （实测末段 5000 步完全不再改善），全量后能到 ~0.1%。
            bs = N if (args.full_batch_final and stage_i == len(plan) - 1) else args.batch
            # 样本池：K_s < full_K 时用滑窗把一条数据摊成多个样本
            pool_case, pool_start, n_win = stage_pool(K_s)
            W = int(pool_case.size)
            opt = torch.optim.Adam([ps.theta], lr=lr_s)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, n_steps), eta_min=0.0) #lr_s * 0.15)
            win_lab = (f"滑窗 {n_win} 个/条 × {N} 条 = {W} 个样本，"
                       f"步长 {max(1, int(round(K_s * args.window_stride)))} 步"
                       if n_win > 1 else f"每条数据 1 段（数据开头 {K_s} 步）")
            print(f"\n--- 课程阶段: 拟合窗口 K={K_s} / {full_K}，{win_lab}，"
                  f"预算 {n_steps} 步，batch={bs}，"
                  f"每 epoch {max(1, W // min(bs, W))} 个 batch，"
                  f"lr {lr_s:.3g} -> {0.0:.3g} ---")
            epoch_order = None
            batch_cursor = 0

            for _ in range(n_steps):
                if epoch_order is None or batch_cursor >= len(epoch_order):
                    epoch_order = distribute_batches(W, bs, rng)
                    batch_cursor = 0
                sel = epoch_order[batch_cursor]
                batch_cursor += 1
                bcase = pool_case[sel]
                bstart = pool_start[sel]

                params_now = ps.params_from_tensor()

                # 定期在【全窗口 × 全量数据】上评估：batch+短窗口的 loss 看不出真实进度
                if gstep % args.eval_every == 0 or gstep == total_steps - 1:
                    tot = 0.0
                    for c in cases:
                        pg.set_params(make_params(c, params_now))
                        tot += case_loss(pg, c, full_K)
                    hist_eval_step.append(gstep)
                    hist_eval_loss.append(tot / N)

                if pgb is not None:
                    # 批量 SoA：一个 C 调用算完整个 batch 的 loss 与 dL/dp
                    loss_sum, grad_sum = batch_loss_and_grad(
                        pgb, bcase, bstart, K_s, params_now, batch_data,
                        args.loss_scale)
                else:
                    loss_sum = 0.0
                    grad_sum = np.zeros(len(PARAM_NAMES))
                    for i, s_i in zip(bcase, bstart):
                        c = cases[int(i)]
                        # 重力是每条数据自带的已知量：连同当前被辨识参数一起写给句柄。
                        # 不调用 set_params 的话 loss/梯度会一直停在创建时的参数点上，
                        # 表现为 loss 不下降且与学习率无关。
                        pg.set_params(make_params(c, params_now))
                        loss, g = case_loss_and_grad(pg, c, K_s, int(s_i))
                        loss_sum += loss
                        grad_sum += g * args.loss_scale

                # 物理参数梯度 -> theta 梯度（重参数化的链式法则）
                theta_np = ps.theta_np()
                dtheta = ParamSpec.dtheta_from_dp(grad_sum, theta_np)
                # 仍处于冻结期的参数：先扣掉会改变它的梯度分量（不污染 Adam 动量），
                # 再在 opt.step() 后精确写回冻结值（见 FreezeSchedule）。
                if freeze:
                    dtheta = freeze.apply_grad(dtheta, theta_np, gstep)
                hist_loss.append(loss_sum / len(sel))
                hist_theta.append(theta_np)
                hist_K.append(K_s)

                ps.theta.grad = torch.tensor(dtheta, dtype=torch.float64)
                opt.step()
                if freeze:
                    freeze.restore(ps, gstep)
                sched.step()
                opt.zero_grad(set_to_none=True)
                gstep += 1

                if gstep % 500 == 0 or gstep == total_steps:
                    cur = ps.seed_values()
                    combo_lab = "组合误差" if has_truth else "组合偏离(标称)"
                    frozen_lab = ""
                    if freeze:
                        fz = freeze.frozen_now(gstep)
                        frozen_lab = ("  冻结=" + ",".join(n for n in PARAM_NAMES if n in fz)
                                      if fz else "  冻结=无")
                    print(f"  step {gstep:6d}  K={K_s:>3}  batch loss={hist_loss[-1]:.4e}  "
                          f"{combo_lab}={combo_error(cur, truth_ident):.3e}  "
                          f"lr={sched.get_last_lr()[0]:.2e}{frozen_lab}")

        if pgb is not None:
            pgb.close()

    # ---------------------------------------------------------------------
    # 结果
    # ---------------------------------------------------------------------
    hist_loss = np.array(hist_loss)
    hist_theta = np.array(hist_theta)
    hist_K = np.array(hist_K)
    # 课程阶段的切换点（用于在图上标注窗口变化）
    stage_bounds, stage_ks = [], []
    for i, kk in enumerate(hist_K):
        if i == 0 or kk != hist_K[i - 1]:
            stage_bounds.append(i)
            stage_ks.append(int(kk))
    est = ps.seed_values()

    # 训练结束后再在全窗口全量数据上评估一次（上面的定期评估发生在最后一次更新之前）
    with ParamGradient(make_params(cases[0], est), refinement=refinement) as pg_fin:
        fin_vals = []
        for c in cases:
            pg_fin.set_params(make_params(c, est))
            fin_vals.append(case_loss(pg_fin, c, full_K))
    final_loss = float(np.mean(fin_vals))

    print("\n=== 辨识结果（注意：mb/ms 两个标度严格不可辨识，单参数数值本身无意义）===")
    init_params = ps.to_params_np(hist_theta[0])   # 第一次更新前的初值
    if has_truth:
        print(f"{'参数':>8} {'真值':>14} {'初值':>14} {'估计':>14} {'相对误差':>12}")
        for n in PARAM_NAMES:
            r = (est[n] - truth[n]) / max(abs(truth[n]), 1e-12)
            print(f"{n:>8} {truth[n]:>14.6g} {init_params[n]:>14.6g} {est[n]:>14.6g} {r:>12.2e}")
    else:
        print(f"{'参数':>8} {'初值':>14} {'估计':>14}")
        for n in PARAM_NAMES:
            print(f"{n:>8} {init_params[n]:>14.6g} {est[n]:>14.6g}")

    combo_init = combo_error(init_params, truth_ident)
    combo_final = combo_error(est, truth_ident)
    base = "真值" if has_truth else "标称参数"
    print(f"\n可辨识组合最大相对误差（基准 {base}）: "
          f"初值 {combo_init:.4g} -> 终值 {combo_final:.4g}")
    if has_truth:
        print(f"全窗口平均 loss: 初值 {hist_eval_loss[0]:.4e} -> 终值 {final_loss:.4e}"
              f"  真值底 {ref_loss:.4e}（比值 {final_loss / ref_loss:.2f}）")
    else:
        print(f"全窗口平均 loss: 初值 {hist_eval_loss[0]:.4e} -> 终值 {final_loss:.4e}")

    # ---------------------------------------------------------------------
    # 绘图
    # ---------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 1) 14 个被辨识参数曲线
        fig, axes = plt.subplots(4, 4, figsize=(15, 10), constrained_layout=True)
        for k in range(len(PARAM_NAMES), axes.size):   # 14 个参数，关掉多余空轴
            axes.flatten()[k].axis("off")
        for k, n in enumerate(PARAM_NAMES):
            ax = axes[k // 4][k % 4]
            vals = np.array([ps.to_params_np(th)[n] for th in hist_theta])
            ax.plot(vals, lw=1.0, label="estimated")
            if has_truth:
                ax.axhline(truth[n], color="k", ls="--", lw=1.0, label="truth")
            # 冻结区间（橙色阴影）：这段时间该参数不参与优化，曲线应为直线
            if freeze and n in freeze.until:
                u = freeze.until[n]
                end = min(u, total_steps) if np.isfinite(u) else total_steps
                ax.axvspan(0, end, color="tab:orange", alpha=0.10,
                           label="frozen (no update)")
                if np.isfinite(u):
                    ax.axvline(u, color="tab:orange", lw=0.9, ls=":")
            # 课程阶段的窗口切换点（参数曲线在这里会有明显的斜率变化）
            for b in stage_bounds:
                ax.axvline(b, color="tab:red", lw=0.6, alpha=0.35)
            ax.set_yscale("symlog", linthresh=1e-3)   # 跨多个数量级且可能为负
            if freeze and n in freeze.until:
                u = freeze.until[n]
                ax.set_title(f"{n}  [frozen < {int(u)}]" if np.isfinite(u)
                             else f"{n}  [frozen]", fontsize=10)
            else:
                ax.set_title(n, fontsize=10)
            ax.grid(True, alpha=0.3)
            if k == 0:
                ax.legend(fontsize=7)
        fig.suptitle(
            f"Parameter identification ({len(PARAM_NAMES)} params, gravity known, "
            f"init={args.init}, curriculum + Adam)  —  "
            f"mb/ms scale is not identifiable; see the combinations figure",
            fontsize=12)
        p1 = out_dir / "params.png"
        fig.savefig(p1, dpi=110)
        plt.close(fig)

        # 2) loss + 可辨识组合
        n_combo = len(IDENTIFIABLE)
        fig, axes = plt.subplots(1 + (n_combo + 2) // 3, 3, figsize=(15, 4 + 3 * ((n_combo + 2) // 3)),
                                 constrained_layout=True)
        axes = np.atleast_2d(axes)
        axes[0][0].semilogy(hist_eval_step, hist_eval_loss, "-o", ms=3, lw=1.4,
                            label="full-window (K=%d) loss, all data" % full_K)
        axes[0][0].semilogy(range(len(hist_loss)), hist_loss, lw=0.6, alpha=0.35,
                            label="batch loss (current stage windows)")
        if ref_loss is not None:
            axes[0][0].axhline(ref_loss, color="k", ls="--", lw=1.0,
                               label="truth-param floor (noise)")
        for b, kk in zip(stage_bounds, stage_ks):
            axes[0][0].axvline(b, color="tab:red", lw=0.8, alpha=0.4)
            axes[0][0].annotate(f"K={kk}", (b, 0.0), xycoords=("data", "axes fraction"),
                                xytext=(2, 2), textcoords="offset points",
                                fontsize=6, color="tab:red")
        # 各参数的解冻步（橙色点线）：与左侧参数曲线上的阴影一致
        if freeze:
            for j, u in enumerate(sorted({u for u in freeze.until.values()
                                          if np.isfinite(u) and u > 0})):
                axes[0][0].axvline(u, color="tab:orange", lw=0.9, ls=":", alpha=0.9)
                axes[0][0].annotate("unfreeze", (u, 1.0),
                                    xycoords=("data", "axes fraction"),
                                    xytext=(2, -9 - 8 * (j % 3)), textcoords="offset points",
                                    fontsize=6, color="tab:orange")
        axes[0][0].legend(fontsize=6)
        axes[0][0].set_xlabel("Adam step")
        axes[0][0].set_title("loss")
        axes[0][0].grid(True, alpha=0.3)
        axes[0][1].axis("off")
        axes[0][2].axis("off")
        for k, (label, fn) in enumerate(IDENTIFIABLE):
            ax = axes[1 + k // 3][k % 3]
            hist_vals = []
            for th in hist_theta:
                hist_vals.append(fn(ps.to_params_np(th)))
            ax.plot(hist_vals, lw=1.0, label="estimated")
            if has_truth:
                ax.axhline(fn(truth), color="k", ls="--", lw=1.0, label="truth")
            for b in stage_bounds:
                ax.axvline(b, color="tab:red", lw=0.6, alpha=0.35)
            ax.set_title(label, fontsize=9)
            ax.grid(True, alpha=0.3)
            if k == 0:
                ax.legend(fontsize=7)
        fig.suptitle("Loss and structurally identifiable combinations", fontsize=13)
        p2 = out_dir / "loss_and_identifiable.png"
        fig.savefig(p2, dpi=110)
        plt.close(fig)

        # 3) 随机 3 条数据：真实参数 vs 辨识参数，在相同控制下的轨迹
        #    没有真值文件时只画 measured vs identified（不画 truth 曲线）
        import contextlib
        pick = plot_pick
        fig, axes = plt.subplots(len(pick), 2, figsize=(14, 3.2 * len(pick)),
                                 constrained_layout=True)
        axes = np.atleast_2d(axes)
        with contextlib.ExitStack() as stack:
            pg_e = stack.enter_context(
                ParamGradient(make_params(cases[0], est), refinement=refinement))
            pg_t = (stack.enter_context(
                        ParamGradient(make_params(cases[0], truth_ident),
                                      refinement=refinement))
                    if has_truth else None)
            for row, ci in enumerate(pick):
                c = cases[int(ci)]
                # 每条曲线都必须用该条数据自己的已知重力
                pg_e.set_params(make_params(c, est))
                _, *se = pg_e.loss(c.theta_c0, c.dtheta_c, c.ddtheta_c, c.dt, c.tau,
                                   c.x0, case_spec(c, full_K), return_sequences=True)
                st = None
                if pg_t is not None:
                    pg_t.set_params(make_params(c, truth_ident))
                    _, *st = pg_t.loss(c.theta_c0, c.dtheta_c, c.ddtheta_c, c.dt, c.tau,
                                       c.x0, case_spec(c, full_K), return_sequences=True)
                t = (np.arange(c.K) + 1) * c.dt
                for col, (meas, lab) in enumerate(((c.psi_b, "psi_b"), (c.psi_s, "psi_s"))):
                    ax = axes[row][col]
                    ax.plot(t, meas, ".", ms=1.5, alpha=0.4, label="measured")
                    if st is not None:
                        ax.plot(t, st[col], lw=1.2, label="truth params")
                    ax.plot(t, se[col], lw=1.2, ls="--", label="identified")
                    ax.set_title(f"case {int(ci)}: {lab}", fontsize=10)
                    ax.grid(True, alpha=0.3)
                    if row == 0:
                        ax.legend(fontsize=7)
        fig.suptitle("Same control input: "
                     + ("truth vs identified trajectory" if has_truth
                        else "measured vs identified trajectory"), fontsize=13)
        p3 = out_dir / "trajectories.png"
        fig.savefig(p3, dpi=110)
        plt.close(fig)
        print(f"\n图已保存: {p1}\n          {p2}\n          {p3}")
    except ImportError:
        print("（未安装 matplotlib，跳过绘图）")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ===========================================================================
# 为什么是"代数初值 + 课程学习"（实测依据，改动前请先读）
# ===========================================================================
# 本数据集是 3 s @ 100 Hz 的强激励轨迹，系统工作在**混沌区**：把真值参数扰动
# 1e-6，末态 |Δpsi| 平均从 4.6e-11 放大到 1.2e-6（约 2.6e4 倍）。这带来两个硬约束：
#
# 1) 局部梯度法的收敛盆极窄。实测"同时按尺度 eps 扰动全部 theta"后的平均 loss
#    （真值处 = 噪声底 2.61e-3）：
#       eps=1e-4 -> 2.626e-3      eps=1e-3 -> 2.661e-3     eps=1e-2 -> 6.08e-3
#       eps=0.03 -> 3.16e-2       eps=0.1  -> 2.54e-1     eps=0.35 -> 3.48
#    实测把初值整体拉到 2~3 个数量级（每个参数 ×10^±[2,3]）后，**只靠短窗口课程
#    学习是救不回来的**：10000 步、窗口 K=5→300，loss 仍停在真值底的 1.4e4 倍，
#    组合误差 770。原因是短窗口信息量不足（K=1 时加速度项 0.5*dt^2*qdd 远小于观测
#    噪声，参数会在平坦谷里漂到 1e5 倍而 loss 仍在地板附近），而长窗口的盆又太窄。
#    实测能力阶梯（3000 步课程）：初值 0.5 个数量级 -> 组合误差 92%；
#    1.4 个数量级 -> loss 9178x；1.9 个数量级 -> 11040x；2.8 个数量级 -> 13531x。
#
# 2) 所以真正的解法是**第 0 步用代数方法**（algebraic_init）：把方程对聚合量线性化
#    后闭式最小二乘。它是凸问题、与初值完全无关，一步就把 12 个可辨识组合解到
#    5%~10%，于是"初值差几个数量级"这件事被彻底绕开。实测（8000 步课程精修）：
#       代数初值:  组合误差 9.2e-2,  loss 2.97     (真值底 2.61e-3)
#       K=50 后:   组合误差 3.2e-2,  loss 8.8e-3
#       K=200 后:  组合误差 3.1e-3,  loss 2.65e-3  (1.02x)
#       K=300 后:  组合误差 1.2e-3,  loss 2.607e-3 (1.00x)
#    其中 Dx、Dy、fbc、fbv、fsc、fsv 恢复到 1e-5~1e-3。
#
# 3) lr 必须随窗口变：损失对参数的刚度随积分时长急剧增长，短窗口能承受大得多的
#    步长。默认按 lr ∝ 1/K（以全窗口 lr=5e-4 为基准），并在每个阶段内做余弦衰减
#    （warm restart）。实测 lr=0.02 时即使从 eps=0.02 出发 loss 也会从 1.84 炸到 43。
#
# 结论：先前"14 个参数都恢复到 0.2%~3%"的说法是**误导性的**——mb/ms 那 2 条方向
# 严格不可辨识，当初始点在真值附近时梯度根本不移动它们，"恢复"只是"没动过"。
# 正确的评价指标只有 IDENTIFIABLE 里那 12 个组合。
# 解析梯度本身没有问题，已用有限差分逐分量验证到 1e-9
# （见 tests/test_trajectory_gradient.cpp）。
