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

用法（在仓库根执行）::

    python3 python/identify_params.py                       # 代数初值 + 课程 + 10000 步
    python3 python/identify_params.py --init random --init-orders 2,3   # 随机初值对照
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent          # 本文件所在目录：python/
REPO = HERE.parent                              # 仓库根
sys.path.insert(0, str(HERE))

from tcbss import (Params, State, ParamGradient, ParamLossSpec,  # noqa: E402
                   PARAM_GRADIENT_NAMES)

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
    __slots__ = ("theta_c0", "dtheta_c", "ddtheta_c", "x0", "dt", "K", "refinement",
                 "psi_b", "psi_s", "dpsi_b", "dpsi_s", "tau", "seed", "gx", "gy",
                 "theta_c_seq", "dtheta_c_seq")

    def __init__(self, path: Path):
        d = np.load(path)
        self.theta_c0 = float(d["theta_c0"])
        self.dtheta_c = float(d["dtheta_c"])
        self.ddtheta_c = float(d["ddtheta_c"])
        self.x0 = State(float(d["x0_theta_b"]), float(d["x0_dtheta_b"]),
                        float(d["x0_theta_s"]), float(d["x0_dtheta_s"]))
        self.dt = float(d["dt"])
        self.K = int(d["num_steps"])
        self.refinement = int(d["refinement"])
        self.seed = int(d["seed"])
        # 重力矢量：随数据一起保存的已知输入（不是被辨识参数）
        self.gx = float(d["gx"])
        self.gy = float(d["gy"])
        self.psi_b = np.ascontiguousarray(d["psi_b"], dtype=np.float64)
        self.psi_s = np.ascontiguousarray(d["psi_s"], dtype=np.float64)
        self.dpsi_b = np.ascontiguousarray(d["dpsi_b"], dtype=np.float64)
        self.dpsi_s = np.ascontiguousarray(d["dpsi_s"], dtype=np.float64)
        self.tau = np.ascontiguousarray(np.column_stack([d["tau_b"], d["tau_s"]]),
                                        dtype=np.float64)
        self.theta_c_seq, self.dtheta_c_seq = theta_c_trajectory(self)


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
# 单条数据的 loss / 梯度（支持只取前 K 步，课程学习用）
# ===========================================================================
def case_spec(case: Case, K: int) -> ParamLossSpec:
    """四项权重全 1、目标为该条实测序列的前 K 步。"""
    return ParamLossSpec(w_psi_b=1.0, w_psi_s=1.0, w_dpsi_b=1.0, w_dpsi_s=1.0,
                         target_psi_b=case.psi_b[:K], target_psi_s=case.psi_s[:K],
                         target_dpsi_b=case.dpsi_b[:K], target_dpsi_s=case.dpsi_s[:K])


def case_loss_and_grad(pg: ParamGradient, case: Case,
                       K: int) -> tuple[float, np.ndarray]:
    """返回前 K 步上的 (loss, dL/dp)（对物理参数，不是 theta）。

    求导点必须已经由调用方通过 ``pg.set_params(...)`` 设好——句柄里存着参数，
    这里不接收 params，以免出现"传了参数但句柄没更新"的静默错配。

    tau 必须和 target 一起截断到 K：C++ 侧的步数取自 tau 的行数。
    """
    loss, grad = pg.gradient(case.theta_c0, case.dtheta_c, case.ddtheta_c, case.dt,
                             case.tau[:K], case.x0, case_spec(case, K))
    return float(loss), np.asarray(grad, dtype=np.float64)


def case_loss(pg: ParamGradient, case: Case, K: int) -> float:
    """只要 loss（前 K 步）。同样要求句柄参数已经设好。"""
    return float(pg.loss(case.theta_c0, case.dtheta_c, case.ddtheta_c, case.dt,
                         case.tau[:K], case.x0, case_spec(case, K)))


# ===========================================================================
# 代数初始化：把运动方程对"聚合量"线性化后做闭式最小二乘
#
# 运动方程（见 src/param_gradient.cpp 的 evaluate）：
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
    """单条数据贡献 2K 行（每个采样点两行：两个广义坐标）。"""
    tc, dtc = c.theta_c_seq, c.dtheta_c_seq
    ddtc = c.ddtheta_c
    tb = c.psi_b - tc
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
    R = np.zeros((2 * n, 17))
    y = np.zeros(2 * n)
    dc = dts * (2.0 * (dtc + dtb) + dts)          # C1 的公共因子
    for k in range(n):
        rb = np.zeros(17)
        rs = np.zeros(17)
        # --- 第 1 行：M11·ddtb + M12·ddts + C1 + G1 + M11·ddtc − Qb = 0
        rb[0] += ddtb[k] + ddtc
        rb[1] += ddtb[k] + ddts[k] + ddtc
        rb[2] += ddtb[k] + ddtc
        mfacb = 2.0 * ddtb[k] + ddts[k] + 2.0 * ddtc
        for idx, coef in ((3, ct[k]), (4, ct[k]), (5, st[k]), (6, -st[k])):
            rb[idx] += coef * mfacb
        rb[3] += -st[k] * dc[k]
        rb[4] += -st[k] * dc[k]
        rb[5] += ct[k] * dc[k]
        rb[6] += -ct[k] * dc[k]
        rb[9] += gx * sb[k] - gy * cb[k]        # R
        rb[10] += gy * sb[k] + gx * cb[k]       # S
        rb[11] += gx * sb[k] - gy * cb[k]       # U
        rb[12] += gy * sb[k] + gx * cb[k]       # V
        rb[7] += gx * ss[k] - gy * cs[k]        # P
        rb[8] += gy * ss[k] + gx * cs[k]        # Q2
        rb[14] += dtb[k]
        rb[13] += np.tanh(_LAM * dtb[k])
        y[2 * k] = Tb[k]
        # --- 第 2 行：M12·ddtb + M22·ddts + C2 + G2 + M12·ddtc − Qs = 0
        rs[1] += ddtb[k] + ddts[k] + ddtc
        mfacs = ddtb[k] + ddtc
        for idx, coef in ((3, ct[k]), (4, ct[k]), (5, st[k]), (6, -st[k])):
            rs[idx] += coef * mfacs
        rs[3] += st[k] * dtb[k] ** 2            # C2 = (A st + B st − C ct + D ct)·ḃb²
        rs[4] += st[k] * dtb[k] ** 2
        rs[5] += -ct[k] * dtb[k] ** 2
        rs[6] += ct[k] * dtb[k] ** 2
        rs[7] += gx * ss[k] - gy * cs[k]
        rs[8] += gy * ss[k] + gx * cs[k]
        rs[16] += dts[k]
        rs[15] += np.tanh(_LAM * dts[k])
        y[2 * k + 1] = Ts[k]
        R[2 * k] = rb
        R[2 * k + 1] = rs
    return R, y


def algebraic_init(cases: list[Case], nominal=(1.0, 1.0)) -> tuple[dict, dict]:
    """在测量数据上做闭式最小二乘，解出 12 个可辨识聚合组合，再回代成物理参数。

    返回 (物理参数字典, 诊断信息)。两个不可辨识方向（mb、ms 的标度）取使参数
    全部可行的、且最接近 nominal 的值；它们不影响 loss，只影响参数沿平坦谷的落点。
    """
    R = np.vstack([_regression_block(c)[0] for c in cases])
    y = np.concatenate([_regression_block(c)[1] for c in cases])
    xs, *_ = np.linalg.lstsq(R, y, rcond=None)

    I_S = xs[1]
    I_BD = xs[0] + xs[2]
    AB = xs[3] + xs[4]
    CmD = xs[5] - xs[6]
    P, Q2 = xs[7], xs[8]
    RU = xs[9] + xs[11]
    SV = xs[10] + xs[12]
    fbc, fbv, fsc, fsv = xs[13], xs[14], xs[15], xs[16]
    det = P * P + Q2 * Q2

    def back(mb: float, ms: float) -> dict:
        # Dx、Dy 可由聚合量非线性组合解出（所以它们其实可辨识）
        Dx = (P * AB - Q2 * CmD) / det
        Dy = (Q2 * AB + P * CmD) / det
        I_D = ms * (Dx * Dx + Dy * Dy)
        I_B = I_BD - I_D
        Pbx = (RU - ms * Dx) / mb
        Pby = (SV - ms * Dy) / mb
        return dict(mb=mb, Ib=I_B - mb * (Pbx ** 2 + Pby ** 2), Pbx=Pbx, Pby=Pby,
                    ms=ms, Is=I_S - (P * P + Q2 * Q2) / ms, Psx=P / ms, Psy=Q2 / ms,
                    Dx=Dx, Dy=Dy, fbc=fbc, fbv=fbv, fsc=fsc, fsv=fsv)

    def feasible(p: dict) -> bool:
        v = np.array(list(p.values()), dtype=np.float64)
        return bool(np.isfinite(v).all() and p["mb"] > 0 and p["ms"] > 0
                    and p["Ib"] > 1e-6 and p["Is"] > 1e-6
                    and p["fbc"] > 0 and p["fbv"] > 0 and p["fsc"] > 0 and p["fsv"] > 0)

    best, best_cost = None, np.inf
    for mb in np.geomspace(0.02, 100.0, 140):
        for ms in np.geomspace(0.02, 100.0, 140):
            p = back(mb, ms)
            if not feasible(p):
                continue
            cost = abs(np.log(mb / nominal[0])) + abs(np.log(ms / nominal[1]))
            if cost < best_cost:
                best, best_cost = p, cost
    if best is None:
        raise RuntimeError("代数初始化找不到可行的 (mb, ms)，数据可能异常")
    info = dict(mb=best["mb"], ms=best["ms"], combo_err=None)
    return best, info


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


# 默认课程：窗口从 20 步逐步拉长到 300 步，权重集中在长窗口。
# 有了代数初值（--init algebraic，默认）之后，K=5/10 这类极小窗口已无必要——
# 它们信息量不足、只会让参数在平坦谷里漂；K=20 起就够（加速度已可观测）。
# 若改用随机初值（--init random），可自行加回小窗口，但实测那样也救不了 2 个数量级。
DEFAULT_STAGES = "20:0.08,50:0.10,100:0.12,200:0.20,300:0.50"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=str(DATA_DIR))
    ap.add_argument("--steps", type=int, default=10000,
                    help="总优化步数，按 --stages 的权重分配到各阶段")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5.0e-4,
                    help="Adam 学习率；theta 空间极窄的稳定区间，见文件末尾说明")
    ap.add_argument("--loss-scale", type=float, default=1.0,
                    help="对均值 loss 乘的系数（Adam 会归一化梯度，影响很小）")
    ap.add_argument("--no-full-batch-final", dest="full_batch_final", action="store_false",
                    help="最后一个课程阶段也只用 batch（默认改为全量，否则组合误差停在 ~3%%）")
    ap.add_argument("--stages", type=str, default=DEFAULT_STAGES,
                    help='短窗口课程表 "K:权重,..."，窗口逐步拉长到全窗口')
    ap.add_argument("--no-curriculum", dest="curriculum", action="store_false",
                    help="关闭课程学习：只用全窗口 K 从头训到底")
    ap.add_argument("--eval-every", type=int, default=100,
                    help="每隔多少步在全窗口全量数据上评估一次 loss")
    ap.add_argument("--init", choices=("algebraic", "random"), default="algebraic",
                    help="algebraic=先用闭式最小二乘解出聚合量（与初值无关，推荐）；"
                         "random=用随机初值（用于对照）")
    ap.add_argument("--mass-nominal", type=float, nargs=2, default=(1.0, 1.0),
                    metavar=("MB", "MS"),
                    help="代数初始化中两个不可辨识标度 mb/ms 的标称值（不影响 loss）")
    ap.add_argument("--init-orders", type=str, default=f"{LOG10_ORDERS_MIN},{LOG10_ORDERS_MAX}",
                    help='--init random 时，初值偏离真值的数量级区间 "min,max"')
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=str, default=str(REPO / "data" / "identify"))
    args = ap.parse_args()

    data_dir = Path(args.data)
    files = sorted(data_dir.glob("case_*.npz"))
    if not files:
        print(f"未找到数据: {data_dir}")
        return 1
    truth = load_truth(data_dir / "truth_params.txt")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [Case(f) for f in files]
    N = len(cases)
    dt = cases[0].dt
    # refinement 是每条数据自带的运行期参数（库构造时传入，不再是编译期常量）。
    # 本数据集里各条一致；若不一致就需要按 refinement 分组各自建句柄。
    refinements = {c.refinement for c in cases}
    if len(refinements) != 1:
        print(f"数据里的 refinement 不一致: {sorted(refinements)}；"
              f"请先按 refinement 分组再辨识")
        return 1
    refinement = cases[0].refinement
    if len({c.dt for c in cases}) != 1:
        print("各条数据的 dt 不一致，暂不支持")
        return 1
    g_range = (min(min(c.gx, c.gy) for c in cases), max(max(c.gx, c.gy) for c in cases))
    print(f"数据: {N} 条   dt={dt}s ({1/dt:.0f}Hz)   K={cases[0].K}   "
          f"refinement={refinement}（每条自带的运行期参数）")
    print(f"被辨识参数: {len(PARAM_NAMES)} 个（重力 gx/gy 为已知输入，取值跨度 "
          f"[{g_range[0]:.2f}, {g_range[1]:.2f}]）")
    print(f"优化: {args.steps} 步, batch={args.batch}, Adam lr={args.lr}")
    effective_bs = min(args.batch, N)
    print(f"每 epoch batch 数 ≈ {max(1, N // effective_bs)}   "
          f"(数据 {N} 条 / batch {effective_bs})")

    full_K = cases[0].K
    # 课程表：(拟合窗口 K, 该阶段步数)。K 按权重分摊总步数，并逐步拉长到 full_K。
    if args.curriculum:
        plan = [(min(k, full_K), max(1, int(round(args.steps * w))))
                for k, w in parse_stages(args.stages)]
        plan.sort()
        plan[-1] = (full_K, plan[-1][1])       # 最后阶段一定用全窗口
    else:
        plan = [(full_K, args.steps)]
    total_steps = sum(n for _, n in plan)

    truth_ident = {n: truth[n] for n in PARAM_NAMES}
    rng = np.random.default_rng(args.seed)
    if args.init == "algebraic":
        p_init, ainfo = algebraic_init(cases, nominal=tuple(args.mass_nominal))
        ps = ParamSpec(truth, rng, orders=(0.0, 0.0))
        ps.theta = torch.tensor(
            ParamSpec.to_theta([p_init[n] for n in PARAM_NAMES]),
            dtype=torch.float64, requires_grad=True)
        print(f"代数初始化（闭式最小二乘，与初值无关）: "
              f"不可辨识标度取 mb={ainfo['mb']:.4g} ms={ainfo['ms']:.4g}")
    else:
        init_orders = tuple(float(v) for v in args.init_orders.split(","))
        ps = ParamSpec(truth, rng, orders=init_orders)
        print(f"随机初始化: 每个参数乘 10^(±[{init_orders[0]}, {init_orders[1]}])")
    init_combo = combo_error(ps.seed_values(), truth_ident)
    init_spread = max(abs(np.log10(abs(ps.seed_values()[n]) / abs(truth[n])))
                      for n in PARAM_NAMES)
    print(f"优化: 共 {total_steps} 步, batch={args.batch}, Adam lr={args.lr}"
          + ("  + 短窗口课程学习" if args.curriculum else "  （无课程学习）"))
    if args.curriculum:
        print("课程表: " + " -> ".join(f"K={k}({n}步)" for k, n in plan))
    # 单参数偏差没有意义（mb/ms 两条方向严格不可辨识，参数可沿平坦谷滑很远而
    # loss 完全不变），所以初值好坏只看可辨识组合误差。
    print(f"初值: 可辨识组合最大相对误差={init_combo:.4g}，"
          f"单参数最大偏离={init_spread:.2f} 个数量级（不可辨识方向，仅供参考）")
    ref_vals = []
    with ParamGradient(make_params(cases[0], truth_ident), refinement=refinement) as pg_ref:
        for c in cases:
            pg_ref.set_params(make_params(c, truth_ident))
            ref_vals.append(case_loss(pg_ref, c, full_K))
    ref_loss = float(np.mean(ref_vals))
    print(f"真值参数下的全窗口平均 loss（噪声底）≈ {ref_loss:.6e}")

    hist_loss, hist_theta, hist_K = [], [], []
    hist_eval_step, hist_eval_loss = [], []
    gstep = 0

    # 学习率必须随窗口一起变：损失对参数的敏感度（以及刚度）随积分时长急剧增长，
    # 所以短窗口能承受大得多的步长；而初值差 2~3 个数量级（|Δθ|≈19）时，只有
    # 短窗口阶段的大步长才可能把它拉回来。这里按 lr ∝ 1/K 给每个阶段定步长，
    # 并在阶段内做余弦衰减（warm restart）：阶段开始时步子大、结束时落定。
    def stage_lr_for(K_s: int) -> float:
        return args.lr * (full_K / max(1, K_s))

    with ParamGradient(make_params(cases[0], ps.seed_values()),
                       refinement=refinement) as pg:
        for stage_i, (K_s, n_steps) in enumerate(plan):
            lr_s = stage_lr_for(K_s)
            # 最后一个阶段用全量数据：batch=8 的梯度噪声会让组合误差停在 ~3%
            # （实测末段 5000 步完全不再改善），全量后能到 ~0.1%。
            bs = N if (args.full_batch_final and stage_i == len(plan) - 1) else args.batch
            opt = torch.optim.Adam([ps.theta], lr=lr_s)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, n_steps), eta_min=lr_s * 0.15)
            print(f"\n--- 课程阶段: 拟合窗口 K={K_s} / {full_K}，"
                  f"预算 {n_steps} 步，batch={bs}，lr {lr_s:.3g} -> {lr_s * 0.15:.3g} ---")
            epoch_order = None
            batch_cursor = 0

            for _ in range(n_steps):
                if epoch_order is None or batch_cursor >= len(epoch_order):
                    epoch_order = distribute_batches(N, bs, rng)
                    batch_cursor = 0
                idxs = epoch_order[batch_cursor]
                batch_cursor += 1

                params_now = ps.params_from_tensor()

                # 定期在【全窗口 × 全量数据】上评估：batch+短窗口的 loss 看不出真实进度
                if gstep % args.eval_every == 0 or gstep == total_steps - 1:
                    tot = 0.0
                    for c in cases:
                        pg.set_params(make_params(c, params_now))
                        tot += case_loss(pg, c, full_K)
                    hist_eval_step.append(gstep)
                    hist_eval_loss.append(tot / N)

                loss_sum = 0.0
                grad_sum = np.zeros(len(PARAM_NAMES))
                for i in idxs:
                    c = cases[int(i)]
                    # 重力是每条数据自带的已知量：连同当前被辨识参数一起写给句柄。
                    # 不调用 set_params 的话 loss/梯度会一直停在创建时的参数点上，
                    # 表现为 loss 不下降且与学习率无关。
                    pg.set_params(make_params(c, params_now))
                    loss, g = case_loss_and_grad(pg, c, K_s)
                    loss_sum += loss
                    grad_sum += g * args.loss_scale

                # 物理参数梯度 -> theta 梯度（重参数化的链式法则）
                theta_np = ps.theta_np()
                dtheta = ParamSpec.dtheta_from_dp(grad_sum, theta_np)
                hist_loss.append(loss_sum / len(idxs))
                hist_theta.append(theta_np)
                hist_K.append(K_s)

                ps.theta.grad = torch.tensor(dtheta, dtype=torch.float64)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                gstep += 1

                if gstep % 500 == 0 or gstep == total_steps:
                    cur = ps.seed_values()
                    print(f"  step {gstep:6d}  K={K_s:>3}  batch loss={hist_loss[-1]:.4e}  "
                          f"组合误差={combo_error(cur, truth_ident):.3e}  "
                          f"lr={sched.get_last_lr()[0]:.2e}")

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
    print(f"{'参数':>8} {'真值':>14} {'初值':>14} {'估计':>14} {'相对误差':>12}")
    init_params = ps.to_params_np(hist_theta[0])   # 第一次更新前的初值
    for n in PARAM_NAMES:
        r = (est[n] - truth[n]) / max(abs(truth[n]), 1e-12)
        print(f"{n:>8} {truth[n]:>14.6g} {init_params[n]:>14.6g} {est[n]:>14.6g} {r:>12.2e}")

    combo_init = combo_error(init_params, truth_ident)
    combo_final = combo_error(est, truth_ident)
    print(f"\n可辨识组合最大相对误差: 初值 {combo_init:.4g} -> 终值 {combo_final:.4g}")
    print(f"全窗口平均 loss: 初值 {hist_eval_loss[0]:.4e} -> 终值 {final_loss:.4e}"
          f"  真值底 {ref_loss:.4e}（比值 {final_loss / ref_loss:.2f}）")

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
            ax.axhline(truth[n], color="k", ls="--", lw=1.0, label="truth")
            # 课程阶段的窗口切换点（参数曲线在这里会有明显的斜率变化）
            for b in stage_bounds:
                ax.axvline(b, color="tab:red", lw=0.6, alpha=0.35)
            ax.set_yscale("symlog", linthresh=1e-3)   # 跨多个数量级且可能为负
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
                            label="batch loss (current K)")
        axes[0][0].axhline(ref_loss, color="k", ls="--", lw=1.0,
                           label="truth-param floor (noise)")
        for b, kk in zip(stage_bounds, stage_ks):
            axes[0][0].axvline(b, color="tab:red", lw=0.8, alpha=0.4)
            axes[0][0].annotate(f"K={kk}", (b, 0.0), xycoords=("data", "axes fraction"),
                                xytext=(2, 2), textcoords="offset points",
                                fontsize=6, color="tab:red")
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
        pick = rng.choice(N, size=min(3, N), replace=False)
        fig, axes = plt.subplots(len(pick), 2, figsize=(14, 3.2 * len(pick)),
                                 constrained_layout=True)
        axes = np.atleast_2d(axes)
        with ParamGradient(make_params(cases[0], truth_ident),
                           refinement=refinement) as pg_t, \
             ParamGradient(make_params(cases[0], est),
                           refinement=refinement) as pg_e:
            for row, ci in enumerate(pick):
                c = cases[int(ci)]
                # 两条曲线必须用同一条数据的已知重力
                pg_t.set_params(make_params(c, truth_ident))
                pg_e.set_params(make_params(c, est))
                _, *st = pg_t.loss(c.theta_c0, c.dtheta_c, c.ddtheta_c, c.dt, c.tau,
                                   c.x0, case_spec(c, full_K), return_sequences=True)
                _, *se = pg_e.loss(c.theta_c0, c.dtheta_c, c.ddtheta_c, c.dt, c.tau,
                                   c.x0, case_spec(c, full_K), return_sequences=True)
                t = (np.arange(c.K) + 1) * c.dt
                axes[row][0].plot(t, c.psi_b, ".", ms=1.5, alpha=0.4, label="measured")
                axes[row][0].plot(t, st[0], lw=1.2, label="truth params")
                axes[row][0].plot(t, se[0], lw=1.2, ls="--", label="identified")
                axes[row][0].set_title(f"case {int(ci)}: psi_b", fontsize=10)
                axes[row][0].grid(True, alpha=0.3)
                if row == 0:
                    axes[row][0].legend(fontsize=7)
                axes[row][1].plot(t, c.psi_s, ".", ms=1.5, alpha=0.4, label="measured")
                axes[row][1].plot(t, st[1], lw=1.2, label="truth params")
                axes[row][1].plot(t, se[1], lw=1.2, ls="--", label="identified")
                axes[row][1].set_title(f"case {int(ci)}: psi_s", fontsize=10)
                axes[row][1].grid(True, alpha=0.3)
                if row == 0:
                    axes[row][1].legend(fontsize=7)
        fig.suptitle("Same control input: truth vs identified trajectory", fontsize=13)
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
