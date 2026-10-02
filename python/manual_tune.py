#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""手动标定参数（v2）—— PyQt5 GUI：左 3×2 曲线，右 文件翻页 + 参数滑块。

这是 v1 ``TorqueControllerForBigSmallYaw/python/scripts/identify_params/manual_tune.py``
在 v2（2-DOF：大 yaw θ_b + 小 yaw θ_s，带基座运动）上的移植版。

与 v1 的差异
------------
* **不需要翻转力矩**：v2 的数据里 τ_b/τ_s 就是模型口径，没有 τ 符号勾选框。
* **初始值来自代数最小二乘**，且按 ``commands.txt`` 最后一条指令的口径确定：
      category            Sentry1                       （data/Sentry1/case_*.npz）
      friction-category   friction_Sentry1              （大 yaw 摩擦扫频）
      friction-category-s friction_s_Sentry1            （小 yaw 匀速往返）
      known-params        Dx=0.0,Dy=0.07
      ignore-gravity      （gx/gy 一律按 0）
  即：``p_dyn = algebraic_init(cases, friction=(0,0,0,0), known)`` →
  ``fit_friction_sweep`` / ``fit_friction_sweep_s`` 独立拟合摩擦 →
  ``algebraic_init(cases, friction=(fbc,fbv,fsc,fsv), known)``。
  这条路径与 ``identify_params.py --init algebraic`` 完全一致。
* **是否使用重力**：只是一个勾选框，**只影响前向仿真曲线**（模型里放不放数据里
  记录的 gx/gy）。初始值始终按上面的 ``--ignore-gravity`` 口径算（勾选前后初值不变，
  也没有 `--init random` 那种随机初值）。
* 曲线画的是**绝对角 ψ_b = θ_c + θ_b、ψ_s = θ_c + θ_b + θ_s**（与辨识 loss 同一口径），
  右轴画该段记录的 τ_b / τ_s。

布局
----
    左侧:  3 行（当前组最多 3 个采样文件）× 2 列（ψ_b / ψ_s）；
           每格 **实线 = 实测 ψ**、**虚线 = 当前参数下的仿真 ψ**（从该段记录初值出发，
           用该段记录的 τ 前向积分，与辨识同一套 C++ 内核），右轴点线 = 记录力矩。
    右侧:  「上一组 / 下一组」按 3 个文件一组滚动；「前向仿真使用重力」勾选框；
           每个参数一行（名 → 滑块 → 数值框）。**正数参数在对数范围内调节**
           （默认 [1e-6, 500]，不够时自动按数量级扩张），实数参数（Pbx/Pby/Psx/Psy/
           Dx/Dy）线性调节。滑块与数值框双向同步，拖动时**实时**重算重绘。
           另有 载入… / 另存为… / 复位（复位 = 回到代数最小二乘初值）。

跑法::

    cd /home/huhu233/rm2027/TorqueControllerForBigSmallYaw_v2
    python3 python/manual_tune.py                       # 默认 Sentry1 + 最后一条指令口径
    python3 python/manual_tune.py --data data/Sentry1    # 直接给目录 / glob
    python3 python/manual_tune.py --use-gravity          # 打开时默认勾选"使用重力"

依赖: PyQt5 + matplotlib（Qt5Agg 后端）+ torch（identify_params 会 import）+ 已编译的
      build/libtcbs.so。
"""

from __future__ import annotations

import argparse
import glob as globmod
import os
import re
import sys
from pathlib import Path

import numpy as np

# ★ 允许直接按文件路径运行（与 identify_params / sim_config 同一套引导）
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _ensure_mpl_config_dir() -> None:
    """在 import matplotlib **之前**给它一个可写配置目录。

    本机的 ``~/.config/matplotlib`` 不可写，matplotlib 导入时会刷一屏
    "配置文件不可写入"/"created a temporary cache directory" 警告（每次启动
    都换一个新的临时目录，首次绘制也变慢）。这里只在默认目录确实不可写时，
    改用一个稳定的临时目录，警告与重复建缓存都没有了。
    """
    if os.environ.get("MPLCONFIGDIR"):
        return
    xdg = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    default = Path(xdg) / "matplotlib"
    probe = default if default.exists() else Path(xdg)
    if not os.access(probe, os.W_OK):
        import tempfile
        os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(),
                                                  "tcbs_mplconfig")


_ensure_mpl_config_dir()

import sim_config as cfg  # noqa: E402
from identify_params import (  # noqa: E402
    LAMBDA,
    PARAM_NAMES,
    POSITIVE,
    Case,
    algebraic_init,
    case_spec,
    fit_friction_sweep,
    fit_friction_sweep_s,
    parse_known_params,
)
from capi import ParamGradient, Params  # noqa: E402

try:
    from PyQt5 import QtCore, QtWidgets
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"[error] 需要 PyQt5: {exc}\n安装: pip install PyQt5") from exc

import matplotlib  # noqa: E402

try:
    matplotlib.use("Qt5Agg")
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"[error] 需要 matplotlib 的 Qt5Agg 后端: {exc}") from exc

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
GROUPSZ = 3                       # 一次显示 / 翻页的文件数（= 行数）
NROW = 3
# 正数参数的滑块范围（物理量）。★ 故意比物理量级宽：代数初值里 Ib 可能到 ~4e-5，
# 若下界只到 1e-4，set_value 会把它静默截断成 1e-4，滑块显示与真实参数就不一致了。
# 仍然不够时 ParamRow._expand 会按数量级自动扩张。
LOG_LO, LOG_HI = 1e-6, 500.0
POS_LO_FLOOR = 1e-12              # 正数参数滑块下界的下限（再小数值框就表示不出了）
# 实数参数（Pbx/Pby/Psx/Psy/Dx/Dy）的滑块范围: ±此值（同样按需自动扩张）
REAL_RANGE = 0.5
SLIDER_STEPS = 1000
COL_NAMES = ("psi_b (big yaw)", "psi_s (small yaw)")   # 只作 matplotlib 标题（英文）
TAU_LABELS = ("tau_b", "tau_s")

# 最后一条指令的口径（均可被命令行覆盖）
DEF_CATEGORY = "Sentry1"
DEF_FRICTION_CATEGORY = "friction_Sentry1"
DEF_FRICTION_CATEGORY_S = "friction_s_Sentry1"
DEF_KNOWN_PARAMS = "Dx=0.0,Dy=0.07"
# 控制力矩通道增益初值（物理力矩 = k × 下发给电控的指令值）。
# kb=4 是 Sentry1 实测口径（下发 1 ⇒ 4 N·m）；ks 未实测，取 1 只是约定
# ——它只决定参数的数值标度，模型响应只认 ks × 惯量参数这个乘积。
DEF_TORQUE_GAIN = "kb=4.0,ks=1.0"


def parse_torque_gain(spec: str) -> dict:
    """解析 ``--torque-gain`` 的 "kb=值,ks=值" 表；缺项用默认值补齐。"""
    out = {"kb": 4.0, "ks": 1.0}
    for part in str(spec).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"--torque-gain 的 {part!r} 缺少 '='；应为 kb=值 或 ks=值")
        name, val_s = (s.strip() for s in part.split("=", 1))
        if name not in out:
            raise ValueError(f"--torque-gain 里未知的键 {name!r}；只能是 kb / ks")
        try:
            v = float(val_s)
        except ValueError:
            raise ValueError(f"--torque-gain 里 {name} 的值 {val_s!r} 不是数") from None
        if not np.isfinite(v) or v == 0.0:
            raise ValueError(f"--torque-gain 里 {name} 必须是有限非零数（收到 {val_s!r}）；"
                             f"0 会让该轴完全没有力矩")
        out[name] = v
    return out


# ===========================================================================
# 文件收集 / 初始值（代数最小二乘，口径 = commands.txt 最后一条指令）
# ===========================================================================
def collect_case_files(spec) -> list:
    """把「目录 / glob / 逗号分隔路径」统一成 case npz 文件列表（按名字排序）。

    目录优先取 ``case_*.npz``，没有则退回 ``*.npz``（与 identify_params 一致）。
    """
    out = []
    for pat in str(spec).split(","):
        pat = pat.strip()
        if not pat:
            continue
        p = Path(pat)
        if p.is_dir():
            hit = sorted(p.glob("case_*.npz")) or sorted(p.glob("*.npz"))
        else:
            hit = sorted(Path(x) for x in globmod.glob(pat))
            if not hit and p.is_file():
                hit = [p]
        out.extend(hit)
    seen, res = set(), []
    for f in out:
        key = str(f)
        if key not in seen:
            seen.add(key)
            res.append(f)
    return res


def compute_algebraic_init(files, known, fric_big, fric_small, use_sweeps=True,
                           gains=None, log=print):
    """按 ``identify_params.py --init algebraic`` 的口径算初值。

    返回 ``(p_init, cases, info)``：
      * ``p_init``：**全部 16 个**参数的 dict（14 个物理量由代数最小二乘 + 摩擦
        扫频回代；kb/ks 不参与那个闭式回归，由 ``gains``/``known`` 直接给出）；
      * ``cases`` ：全部 ``Case`` 对象，**保留数据里原始 gx/gy**（供前向仿真用，
                    初始值本身按 g=0 求，见下）；
      * ``info``  ：摩擦拟合诊断（打印 / 状态栏用）。

    ★ 与最后一条指令一致：**初始值一律按 g=0 求**（``--ignore-gravity``），
      数据里记录的 gx/gy 只在"前向仿真使用重力"勾选时进入模型。
    """
    gains = {"kb": 4.0, "ks": 1.0} if gains is None else dict(gains)
    cases = [Case(f) for f in files]
    raw_g = [(c.gx, c.gy) for c in cases]
    info = {"n_files": len(cases), "friction": None}
    # kb/ks 是**力矩通道增益**（物理力矩 = k × 下发的指令值），属于模型的一部分。
    # 这里把 gains/known 指定的 k 交给 algebraic_init，由它按 k 缩放两行回归；
    # 解出的参数与这两个 k 处在同一标度，因此不能事后再覆盖 k（否则差一个倍数）。
    known_k = dict(known)
    for g in ("kb", "ks"):
        known_k.setdefault(g, gains[g])
    try:
        # g=0：等价 --ignore-gravity（此时必须给 Dx/Dy 才能分离 Dx/Dy/Psx/Psy）
        for c in cases:
            c.gx = 0.0
            c.gy = 0.0
        # ---- 先解一次动力学（带 0 摩擦），用来给摩擦扫频扣 G1/G2 ----
        p_dyn, ainfo_dyn = algebraic_init(cases, friction=(0.0, 0.0, 0.0, 0.0),
                                          known=known_k)
        log(f"[init] 力矩通道增益: kb={p_dyn['kb']:.6g} [{ainfo_dyn['kb_source']}]  "
            f"ks={p_dyn['ks']:.6g} [{ainfo_dyn['ks_source']}]")

        friction = None
        if use_sweeps and fric_big is not None and Path(fric_big).exists():
            fbc, fbv, ib = fit_friction_sweep(fric_big, p_dyn, ignore_gravity=True,
                                              kb=float(p_dyn["kb"]))
            fsc, fsv = 0.5 * fbc, 0.5 * fbv        # 缺省回退：小 yaw 取大 yaw 的一半
            info["friction"] = dict(fbc=fbc, fbv=fbv, info_big=ib, small=None)
            log(f"[init] 大 yaw 摩擦（{ib['files']} 个 sweep，{ib['n']} 个速度点，"
                f"残差 {ib['residual']:.2e} N·m）: fbc={fbc:.5g} fbv={fbv:.5g}")
            if fric_small is not None and Path(fric_small).exists():
                try:
                    fsc, fsv, ism = fit_friction_sweep_s(
                        fric_small, p_dyn, verbose=False, ignore_gravity=True,
                        ks=float(p_dyn["ks"]))
                except Exception as exc:
                    log(f"[warn] 小 yaw 摩擦拟合失败，改用大 yaw 的一半: {exc}")
                else:
                    info["friction"]["small"] = ism
                    log(f"[init] 小 yaw 摩擦（{ism.get('files')} 个文件，"
                        f"{ism.get('n')} 趟）: fsc={fsc:.5g} fsv={fsv:.5g}")
            friction = (fbc, fbv, fsc, fsv)
        elif use_sweeps:
            log(f"[warn] 未找到摩擦扫频目录 {fric_big}，摩擦改用主回归结果")

        p_init, ainfo = algebraic_init(cases, friction=friction, known=known_k)
        # kb/ks 已由 algebraic_init 采用（known_k 里的值），参数与它们同标度，不再覆盖。
        p_init = sanitize_params(p_init, log=log)
        info["mb"] = ainfo["mb"]
        info["ms"] = ainfo["ms"]
    finally:
        # 还原数据里记录的原始重力（前向仿真用）
        for c, (gx, gy) in zip(cases, raw_g):
            c.gx, c.gy = gx, gy

    log("[init] 代数最小二乘初值（与初值无关，g 一律按 0）: "
        + " ".join(f"{n}={p_init[n]:.5g}" for n in PARAM_NAMES))
    return p_init, cases, info


def sanitize_params(p: dict, log=print) -> dict:
    """把非有限值 / 解成非正的正数参数拉回可表示范围。

    正数参数（mb/Ib/ms/Is/fbc/fsv…）在 GUI 里是对数滑块，范围恒 >0；若代数最小二乘
    给出 ≤0（``--no-friction-sweep`` 时摩擦实测会解成负值）或 NaN，滑块只能显示成
    下界，会造成"显示的数和真正下发给模型的数不一致"。这里统一拉回 ``LOG_LO``。
    """
    bad = []
    for n in PARAM_NAMES:
        if not np.isfinite(float(p[n])):
            p[n] = 0.0
        if n in POSITIVE and float(p[n]) <= 0.0:
            p[n] = LOG_LO
            bad.append(n)
    if bad:
        log(f"[warn] 以下正数参数解出非正值，已置为 {LOG_LO:g}"
            f"（对数滑块可表示的最小值）: {', '.join(bad)}")
    return p


def _params_for(case: Case, p: dict, use_gravity: bool) -> Params:
    """把参数（含 kb/ks）+ 该条数据的重力（勾选时用记录值，否则 0）+ 固定 λ 拼成 Params。"""
    gx = float(case.gx) if use_gravity else 0.0
    gy = float(case.gy) if use_gravity else 0.0
    return Params(mb=p["mb"], Ib=p["Ib"], Pbx=p["Pbx"], Pby=p["Pby"],
                  ms=p["ms"], Is=p["Is"], Psx=p["Psx"], Psy=p["Psy"],
                  Dx=p["Dx"], Dy=p["Dy"], gx=gx, gy=gy,
                  fbc=p["fbc"], fbv=p["fbv"], fsc=p["fsc"], fsv=p["fsv"],
                  lambda_=LAMBDA, kb=p["kb"], ks=p["ks"])


def default_param_dict() -> dict:
    """没有数据时的兜底初值（cfg 里的标称参数）。"""
    return {n: float(cfg.DEFAULT_PARAMS[n]) for n in PARAM_NAMES}


# ===========================================================================
# 参数文件读写（手动标定的另存 / 载入）
# ===========================================================================
def write_params_file(path, p: dict, note: str = "") -> None:
    lines = ["# TorqueControllerForBigSmallYaw_v2 —— manual_tune 手动标定参数",
             f"# {note}" if note else "#",
             "# 单位: kg, kg*m^2, m, N*m/(rad/s), N*m；lambda_ 为固定常数"]
    for n in PARAM_NAMES:
        lines.append(f"{n} = {float(p[n]):.10g}")
    lines.append(f"lambda_ = {LAMBDA:g}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_params_file(path) -> dict:
    """读 ``名字 = 值``（也兼容空白分隔），只取 PARAM_NAMES 里的参数。"""
    vals = {}
    with open(path, "r", errors="replace") as fh:
        for ln in fh:
            m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*[=\s]\s*([-+0-9.eE]+)", ln)
            if not m:
                continue
            name, val = m.group(1), m.group(2)
            if name in PARAM_NAMES:
                vals[name] = float(val)
    if not vals:
        raise ValueError(f"{path} 里没有读到任何被辨识参数（需要 '名字 = 值' 格式）")
    return vals


# ===========================================================================
# 一行参数控件: 名字 + 滑块 + 数值框（正数用对数映射，实数用线性）
# ===========================================================================
class ParamRow(QtWidgets.QWidget):
    def __init__(self, index: int, value: float, changed, is_positive: bool):
        super().__init__()
        self.index = int(index)
        self.is_positive = bool(is_positive)
        self._changed = changed
        name = PARAM_NAMES[self.index]
        self.lo, self.hi = ((LOG_LO, LOG_HI) if self.is_positive
                            else (-REAL_RANGE, REAL_RANGE))
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(2, 1, 2, 1)
        self.lbl = QtWidgets.QLabel(f"{self.index:>2} {name}")
        self.lbl.setFixedWidth(150)
        lay.addWidget(self.lbl)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, SLIDER_STEPS)
        self.spin = QtWidgets.QDoubleSpinBox()
        # 12 位小数：Ib 实测在 1e-5 量级，对数滑块扩张后还会更小；若只有 8 位，
        # 1e-9 会被四舍五入成 0，正数参数就变成非法的 0 了。
        self.spin.setDecimals(12)
        self.spin.setRange(self.lo, self.hi)
        self.spin.setFixedWidth(160)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin)
        self._sync_guard = False
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)
        self._refresh_tooltip()
        self.set_value(value)

    # ── 范围（按需按数量级/倍数扩张，避免把初值静默截断）──
    def _refresh_tooltip(self) -> None:
        name = PARAM_NAMES[self.index]
        self.lbl.setToolTip(f"{name}（正数：滑块为对数、自动扩数量级）"
                            if self.is_positive else f"{name}（实数：线性滑块）")

    def _expand(self, v: float) -> None:
        """保证 v 落在 [lo, hi] 内；只扩不缩。"""
        v = float(v)
        if self.is_positive:
            if not np.isfinite(v) or v <= 0.0:
                return
            while v < self.lo and self.lo > POS_LO_FLOOR:
                self.lo = max(self.lo / 10.0, POS_LO_FLOOR)
            while v > self.hi:
                self.hi *= 10.0
        else:
            r = max(REAL_RANGE, 2.0 * abs(v)) if np.isfinite(v) else REAL_RANGE
            if r > 0.5 * (self.hi - self.lo):
                self.lo, self.hi = -r, r
        self.spin.setRange(self.lo, self.hi)
        self.spin.setSingleStep(max(1e-12, (self.hi - self.lo) / 2000.0))

    def _emit(self) -> None:
        """把当前值上报给窗口；正数参数不允许 ≤0（滑块/数值框可能四舍五入到 0）。"""
        v = self.value()
        if self.is_positive and (not np.isfinite(v) or v <= 0.0):
            v = self.lo
        self._changed(self.index, float(v))

    # ── 物理值 ↔ 滑块 ──
    def _to_slider(self, v: float) -> int:
        if self.is_positive:
            v = min(max(float(v), self.lo), self.hi)
            t = (np.log(v) - np.log(self.lo)) / (np.log(self.hi) - np.log(self.lo))
        else:
            t = (float(v) - self.lo) / (self.hi - self.lo)
        return int(round(min(max(t, 0.0), 1.0) * SLIDER_STEPS))

    def _to_value(self, pos: int) -> float:
        t = pos / float(SLIDER_STEPS)
        if self.is_positive:
            return float(np.exp(np.log(self.lo) + t * (np.log(self.hi) - np.log(self.lo))))
        return float(self.lo + t * (self.hi - self.lo))

    def value(self) -> float:
        return float(self.spin.value())

    def set_value(self, v: float) -> None:
        if self._sync_guard:
            return
        self._sync_guard = True
        try:
            self._expand(v)
            self.slider.setValue(self._to_slider(v))
            self.spin.setValue(float(np.clip(v, self.lo, self.hi)))
        finally:
            self._sync_guard = False

    def _from_slider(self, pos: int) -> None:
        if self._sync_guard:
            return
        self._sync_guard = True
        try:
            self.spin.setValue(self._to_value(pos))
        finally:
            self._sync_guard = False
        self._emit()

    def _from_spin(self, v: float) -> None:
        if self._sync_guard:
            return
        self._sync_guard = True
        try:
            self.slider.setValue(self._to_slider(v))
        finally:
            self._sync_guard = False
        self._emit()


# ===========================================================================
# 主窗口
# ===========================================================================
class ManualTuneWindow(QtWidgets.QMainWindow):
    def __init__(self, files, known: dict, fric_big, fric_small,
                 use_sweeps: bool = True, use_gravity: bool = True,
                 gains: dict | None = None):
        super().__init__()
        self.setWindowTitle("manual_tune —— 手动标定（实测 vs 仿真，v2 2-DOF）")
        self.known = dict(known)
        self.fric_big = fric_big
        self.fric_small = fric_small
        self.use_sweeps = bool(use_sweeps)
        # 控制力矩通道增益初值（kb/ks 两个滑块）
        self.gains = {"kb": 4.0, "ks": 1.0}
        if gains:
            self.gains.update({k: float(v) for k, v in gains.items()})

        self.files: list = []
        self.cases: list = []
        self.group = 0
        self.p = default_param_dict()          # 当前 14 个参数
        self.p_init = dict(self.p)             # 代数最小二乘初值（复位用）
        self.info = {}
        self._group_cases: list = []
        self._group_files: list = []
        self._kernels: list = []
        self._dirty = False
        self._backend = "未初始化"

        # ── 左: 3×2 棋盘图 ──
        self.fig = Figure(figsize=(11.5, 8.5), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.axes = self.fig.subplots(NROW, 2, squeeze=False)
        self.lines = {}
        self.axt = {}
        for r in range(NROW):
            for c in range(2):
                ax = self.axes[r][c]
                ln_m, = ax.plot([], [], color=f"C{c}", lw=1.3, label="measured psi")
                ln_s, = ax.plot([], [], color=f"C{c}", lw=1.3, ls="--",
                                label="simulated psi")
                axt = ax.twinx()
                ln_t, = axt.plot([], [], color="0.30", lw=1.0, ls=":",
                                 label=f"{TAU_LABELS[c]} (recorded, right)")
                self.lines[(r, c)] = (ln_m, ln_s, ln_t)
                self.axt[(r, c)] = axt
                ax.grid(True, alpha=0.3)
                if r == 0:
                    ax.set_title(COL_NAMES[c], fontsize=10)
                if c == 0:
                    ax.set_ylabel("psi [rad]", fontsize=9)
                if r == NROW - 1:
                    ax.set_xlabel("t [s]", fontsize=9)
                ax.tick_params(labelsize=8)
                axt.tick_params(labelsize=7, colors="0.35")
                axt.set_ylabel(TAU_LABELS[c] + " [N*m]", fontsize=8, color="0.35")
        self.axes[0][0].legend(handles=[self.lines[(0, 0)][0], self.lines[(0, 0)][1],
                                        self.lines[(0, 0)][2]], fontsize=7, loc="best")

        # ── 右: 数据翻页 + 重力开关 + 参数滑块 ──
        right = QtWidgets.QWidget()
        right.setFixedWidth(490)
        rlay = QtWidgets.QVBoxLayout(right)

        gb_data = QtWidgets.QGroupBox("数据")
        dl = QtWidgets.QVBoxLayout(gb_data)
        row = QtWidgets.QHBoxLayout()
        self.btn_open = QtWidgets.QPushButton("选择数据…")
        self.btn_prev = QtWidgets.QPushButton("◀ 上一组")
        self.btn_next = QtWidgets.QPushButton("下一组 ▶")
        for b in (self.btn_open, self.btn_prev, self.btn_next):
            row.addWidget(b)
        dl.addLayout(row)
        # ★ 重力开关只作用于**前向仿真**：**默认勾选**（用数据里记录的 gx/gy，
        #   与 MPC 的 use_gravity=true 默认一致）；取消勾选 = 模型里没有重力项，
        #   是"不用重力"的对照。初始值仍按 g=0 口径算好、不受勾选影响。
        self.chk_gravity = QtWidgets.QCheckBox("前向仿真使用重力（数据里的 gx/gy）")
        self.chk_gravity.setToolTip(
            "只影响画曲线时的正演模型（★ 默认勾选 = 使用重力）：\n"
            "  勾选 = Params.gx/gy 取该条数据记录的等效重力；\n"
            "  不勾 = gx=gy=0（模型里没有重力项，等价 --ignore-gravity 对照模式）。\n"
            "★ 初始值（代数最小二乘 + 摩擦扫频）始终按 g=0 口径计算，"
            "勾选与否都不会改变参数的初值。")
        self.chk_gravity.setChecked(bool(use_gravity))
        dl.addWidget(self.chk_gravity)
        self.lbl_group = QtWidgets.QLabel("")
        self.lbl_group.setWordWrap(True)
        dl.addWidget(self.lbl_group)
        rlay.addWidget(gb_data)

        gb_p = QtWidgets.QGroupBox("参数（正数=对数滑块；实数=线性滑块）")
        pl = QtWidgets.QVBoxLayout(gb_p)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QtWidgets.QWidget()
        il = QtWidgets.QVBoxLayout(inner)
        il.setSpacing(0)
        self.rows = []
        for i in range(len(PARAM_NAMES)):
            pr = ParamRow(i, float(self.p[PARAM_NAMES[i]]), self._on_param,
                          PARAM_NAMES[i] in POSITIVE)
            self.rows.append(pr)
            il.addWidget(pr)
        il.addStretch(1)
        scroll.setWidget(inner)
        pl.addWidget(scroll)
        rlay.addWidget(gb_p, 1)

        gb_out = QtWidgets.QGroupBox("参数文件")
        ol = QtWidgets.QHBoxLayout(gb_out)
        self.btn_load = QtWidgets.QPushButton("载入…")
        self.btn_save = QtWidgets.QPushButton("另存为…")
        self.btn_reset = QtWidgets.QPushButton("复位到初值")
        for b in (self.btn_load, self.btn_save, self.btn_reset):
            ol.addWidget(b)
        rlay.addWidget(gb_out)

        central = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(central)
        h.addWidget(self.canvas, 1)
        h.addWidget(right)
        self.setCentralWidget(central)

        self.status = self.statusBar()
        self.btn_open.clicked.connect(self.on_open)
        self.btn_prev.clicked.connect(lambda: self.shift_group(-1))
        self.btn_next.clicked.connect(lambda: self.shift_group(+1))
        self.btn_load.clicked.connect(self.on_load_params)
        self.btn_save.clicked.connect(self.on_save_params)
        self.btn_reset.clicked.connect(self.on_reset)
        self.chk_gravity.toggled.connect(lambda _c: self.redraw())

        # 拖动滑块时合并重算（30 ms 一次），避免每个像素都重画
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

        self.reload(files, first=True)
        self.resize(1560, 920)

    # ── 参数变化（合并刷新）──
    def _on_param(self, idx: int, value: float) -> None:
        self.p[PARAM_NAMES[int(idx)]] = float(value)
        self._dirty = True

    def _tick(self) -> None:
        if self._dirty:
            self._dirty = False
            self.redraw()

    def _sync_rows(self) -> None:
        for i, pr in enumerate(self.rows):
            pr.set_value(float(self.p[PARAM_NAMES[i]]))

    # ── 数据装载 / 初始值 ──
    def reload(self, files, first: bool = False) -> None:
        """重新收集数据、重算代数最小二乘初值、回到第 1 组。"""
        self.files = [Path(f) for f in files]
        if self.files:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            self.status.showMessage(f"正在读 {len(self.files)} 个文件并做代数最小二乘初始化…")
            QtWidgets.QApplication.processEvents()
            try:
                p_init, cases, info = compute_algebraic_init(
                    self.files, self.known, self.fric_big, self.fric_small,
                    use_sweeps=self.use_sweeps, gains=self.gains)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(
                    self, "初始化失败",
                    f"代数最小二乘初始化失败，改用 cfg.DEFAULT_PARAMS：\n{exc}")
                p_init, cases, info = default_param_dict(), [], {}
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
            self.cases = cases
            self.p_init = dict(p_init)
            # --known-params 与 --torque-gain 都可能给 kb/ks：known 优先（它是已知量），
            # 其余用 --torque-gain 的初值。两者都只是初值，之后可自由手调。
            for g in ("kb", "ks"):
                self.p_init[g] = float(self.known.get(g, self.gains[g]))
            self.p = dict(self.p_init)
            self.info = info
        else:
            self.cases = []
            self.p_init = default_param_dict()
            for g in ("kb", "ks"):
                self.p_init[g] = float(self.known.get(g, self.gains[g]))
            self.p = dict(self.p_init)
            self.info = {}
        self._sync_rows()
        self.group = 0
        self.load_group()
        if first:
            self.status.showMessage("就绪。拖动滑块实时重算；「复位到初值」= 回到代数最小二乘解。")

    def on_open(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "选择采样数据目录")
        if d:
            self.reload(collect_case_files(d))
        else:
            fs, _ = QtWidgets.QFileDialog.getOpenFileNames(
                self, "选择采样数据文件（按 3 个一组翻页）", "",
                "采样数据 (*.npz);;所有文件 (*)")
            if fs:
                self.reload(list(fs))

    def shift_group(self, step: int) -> None:
        n = len(self.cases)
        if n == 0:
            return
        last = max(0, ((n - 1) // GROUPSZ) * GROUPSZ)
        self.group = int(min(max(self.group + step * GROUPSZ, 0), last))
        self.load_group()

    def load_group(self) -> None:
        # 释放旧内核
        for pg in self._kernels:
            try:
                pg.close()
            except Exception:
                pass
        self._kernels = []
        self._group_files = list(self.files[self.group:self.group + GROUPSZ])
        self._group_cases = list(self.cases[self.group:self.group + GROUPSZ])
        for c in self._group_cases:
            try:
                pg = ParamGradient(_params_for(c, self.p, self.chk_gravity.isChecked()),
                                   refinement=int(cfg.REFINEMENT))
                self._backend = "C++ ParamGradient"
            except Exception as exc:                                     # pragma: no cover
                pg = None
                self._backend = f"不可用（{type(exc).__name__}: {exc}）"
            self._kernels.append(pg)
        n_grp = ((len(self.cases) - 1) // GROUPSZ + 1) if self.cases else 0
        g_idx = (self.group // GROUPSZ + 1) if self.cases else 0
        names = "\n".join("  " + os.path.basename(str(f)) for f in self.files[
            self.group:self.group + GROUPSZ]) or "  （未选择数据）"
        self.lbl_group.setText(f"第 {g_idx}/{max(1, n_grp)} 组"
                               f"（共 {len(self.cases)} 个文件）\n{names}")
        self.btn_prev.setEnabled(self.group > 0)
        self.btn_next.setEnabled(self.group + GROUPSZ < len(self.cases))
        self.redraw()

    # ── 仿真 + 画图 ──
    def _simulate(self) -> list:
        """返回 ``[(case, psi_sim[T,2], err_msg), ...]``：从各段记录初值出发前向积分。"""
        use_g = self.chk_gravity.isChecked()
        out = []
        for i, case in enumerate(self._group_cases):
            pg = self._kernels[i] if i < len(self._kernels) else None
            if pg is None:
                out.append((case, None, "无仿真内核"))
                continue
            try:
                pg.set_params(_params_for(case, self.p, use_g))
                _, psi_b, psi_s, *_ = pg.loss(
                    case.theta_c0, case.dtheta_c, case.ddtheta_c, case.dt,
                    case.tau, case.x0, case_spec(case, case.K), return_sequences=True)
                out.append((case, np.column_stack([psi_b, psi_s]), None))
            except Exception as exc:
                out.append((case, None, f"{type(exc).__name__}: {exc}"))
        return out

    def redraw(self) -> None:
        sims = self._simulate()
        rmses = []
        for r in range(NROW):
            for c in range(2):
                ax = self.axes[r][c]
                ln_m, ln_s, ln_t = self.lines[(r, c)]
                axt = self.axt[(r, c)]
                if r >= len(sims):
                    ln_m.set_data([], [])
                    ln_s.set_data([], [])
                    ln_t.set_data([], [])
                    if r == 0:
                        ax.set_title("(no data)", fontsize=9)
                    continue
                case, pred, err = sims[r]
                fpath = (self._group_files[r] if r < len(self._group_files)
                         else Path(f"case{r}.npz"))
                fname = os.path.basename(str(fpath))
                t = (np.arange(case.K) + 1) * case.dt
                meas = np.asarray(case.psi_b if c == 0 else case.psi_s, dtype=np.float64)
                if pred is None:
                    ln_m.set_data(t, meas)
                    ln_s.set_data([], [])
                    ln_t.set_data(t, np.asarray(case.tau[:, c], dtype=np.float64))
                    ax.set_xlim(float(t[0]), float(t[-1]) if case.K > 1 else 1.0)
                    m, mn = float(np.max(meas)), float(np.min(meas))
                    pad = max(1e-3, 0.1 * (m - mn))
                    ax.set_ylim(mn - pad, m + pad)
                    ax.set_title(f"{fname[:20]}  sim failed: {err}", fontsize=7)
                    continue
                pv = np.asarray(pred[:, c], dtype=np.float64)
                rmse = float(np.degrees(np.sqrt(np.mean((meas - pv) ** 2))))
                rmses.append(rmse)
                ln_m.set_data(t, meas)
                ln_s.set_data(t, pv)
                tau_k = np.asarray(case.tau[:, c], dtype=np.float64)
                ln_t.set_data(t, tau_k)
                tm, tn = float(np.max(tau_k)), float(np.min(tau_k))
                tpad = max(1e-6, 0.10 * (tm - tn))
                axt.set_ylim(tn - tpad, tm + tpad)
                axt.set_xlim(float(t[0]), float(t[-1]) if case.K > 1 else 1.0)
                ax.set_xlim(float(t[0]), float(t[-1]) if case.K > 1 else 1.0)
                m, mn = float(np.max(meas)), float(np.min(meas))
                pad = max(1e-3, 0.10 * (m - mn))
                ax.set_ylim(mn - pad, m + pad)
                title = f"{fname[:22]}  RMSE={rmse:.2f}deg"
                if float(np.max(pv)) > m + pad or float(np.min(pv)) < mn - pad:
                    title += "  [!]sim out of range"
                ax.set_title(title, fontsize=8)
        self.canvas.draw_idle()
        self._update_status(rmses)

    def _update_status(self, rmses) -> None:
        g = "开（数据 gx/gy）" if self.chk_gravity.isChecked() else "关（g=0）"
        mean = f"{np.mean(rmses):.2f}deg" if rmses else "--"
        txt = " ".join(f"{n}={self.p[n]:.4g}" for n in
                       ("mb", "Ib", "ms", "Is", "fbc", "fbv", "fsc", "fsv"))
        self.status.showMessage(
            f"RMSE(本组均值)={mean}  |  重力={g}  |  {txt}  |  后端: {self._backend}")

    # ── 参数文件 ──
    def on_load_params(self) -> None:
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "载入参数文件", "", "参数文本 (*.txt *.params);;所有文件 (*)")
        if not f:
            return
        try:
            vals = read_params_file(f)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "读取失败", str(exc))
            return
        for n, v in vals.items():
            self.p[n] = float(v)
        # 成对/正数约束：正数参数给 0 或负值时不下发（滑块无法表示）
        bad = [n for n in POSITIVE if self.p[n] <= 0.0]
        if bad:
            QtWidgets.QMessageBox.warning(
                self, "参数非法", f"以下正数参数 ≤ 0，已忽略该文件里的取值: {', '.join(bad)}")
            for n in bad:
                self.p[n] = float(self.p_init[n])
        self._sync_rows()
        self.redraw()
        self.status.showMessage(f"已载入 {f}", 5000)

    def on_save_params(self) -> None:
        f, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "保存参数", "manual_params.txt", "参数文本 (*.txt);;所有文件 (*)")
        if not f:
            return
        note = (f"手动标定（manual_tune.py）；重力前向仿真="
                f"{'on' if self.chk_gravity.isChecked() else 'off'}")
        write_params_file(f, self.p, note=note)
        self.status.showMessage(f"已写入 {f}", 5000)

    def on_reset(self) -> None:
        self.p = dict(self.p_init)
        self._sync_rows()
        self.redraw()
        self.status.showMessage("已复位到代数最小二乘初值", 4000)


# ===========================================================================
# 入口
# ===========================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="手动标定参数（v2，PyQt5 GUI）")
    ap.add_argument("--category", type=str, default=DEF_CATEGORY,
                    help=f"数据类别名（= data/<名> 目录）；默认 {DEF_CATEGORY}")
    ap.add_argument("--data", type=str, default=None,
                    help="数据目录 / glob（优先于 --category）")
    ap.add_argument("--friction-category", type=str, default=DEF_FRICTION_CATEGORY,
                    help=f"大 yaw 摩擦扫频类别；默认 {DEF_FRICTION_CATEGORY}")
    ap.add_argument("--friction-category-s", type=str, default=DEF_FRICTION_CATEGORY_S,
                    help=f"小 yaw 摩擦扫频类别；默认 {DEF_FRICTION_CATEGORY_S}")
    ap.add_argument("--no-friction-sweep", dest="use_sweeps", action="store_false",
                    help="初始值不用摩擦扫频，改由主回归给出（可能解成负值）")
    ap.add_argument("--known-params", type=str, default=DEF_KNOWN_PARAMS,
                    help=f'外部已知参数表 "参数=值,..."；默认 "{DEF_KNOWN_PARAMS}"')
    ap.add_argument("--torque-gain", type=str, default=DEF_TORQUE_GAIN,
                    help=f'控制力矩通道增益（物理力矩 = k × 下发给电控的指令值），'
                         f'"kb=值,ks=值"；默认 "{DEF_TORQUE_GAIN}"。'
                         f'★ 只决定参数与指令之间的标度，作为 kb/ks 两个滑块的初值，'
                         f'随时可手调；模型响应只认它们的比值 × 惯量参数。')
    ap.add_argument("--no-gravity", dest="use_gravity", action="store_false",
                    help="★ 默认「使用重力」（前向仿真用数据里记录的 gx/gy，与 MPC 的 "
                         "use_gravity=true 默认一致）；本开关取消勾选，做「不用重力」的对照。"
                         "初始值始终按 g=0 口径求，不受影响。")
    ap.add_argument("--use-gravity", dest="use_gravity", action="store_true",
                    help="冗余的显式打开（= 默认行为），便于脚本里写清楚")
    a, _ = ap.parse_known_args(argv)

    try:
        known = parse_known_params(a.known_params)
        gains = parse_torque_gain(a.torque_gain)
    except ValueError as exc:
        ap.error(str(exc))

    spec = a.data if a.data else cfg.category_dir(a.category)
    files = collect_case_files(spec)
    if not files:
        print(f"[warn] 没有找到采样数据: {spec}（可在 GUI 里点“选择数据…”）")
    fric_big = cfg.category_dir(a.friction_category) if a.friction_category else None
    fric_small = cfg.category_dir(a.friction_category_s) if a.friction_category_s else None

    app = QtWidgets.QApplication(sys.argv[:1])
    win = ManualTuneWindow(files, known, fric_big, fric_small,
                           use_sweeps=a.use_sweeps, use_gravity=a.use_gravity,
                           gains=gains)
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
