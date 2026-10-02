#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""等价参数工具 —— PyQt5 GUI：在不改变动力学的前提下改写参数。

为什么会有"等价参数"
--------------------
本工程的动力学（``src/dm/dynamics.cpp``）只通过下面这些**不变量组合**依赖 16 个
被辨识参数（记 ``Pb=(Pbx,Pby)`` 等）：

    A  = mb|Pb|^2 + Ib + ms|D|^2           (注意：是"和"，不是 mb|Pb|^2+Ib 与 ms|D|^2 各自)
    G  = ms|Ps|^2 + Is
    P1 = ms (Dx Psx + Dy Psy)
    P2 = ms (Dy Psx - Dx Psy)
    mu_s = ms * Ps                         (由重力 gs_sin/gs_cos 定)
    Sig  = mb * Pb + ms * D                (由重力 gb_sin/gb_cos 定)
    以及摩擦 fbc,fbv,fsc,fsv 与增益 kb,ks

而且整个方程组对**共同的标度** sigma 齐次：把上面所有量乘以 sigma，广义加速度
``(ddtheta_b, ddtheta_s)`` 完全不变。于是"同一个动力学"对应一整个参数族（规范轨道），
一般位形下是 **3 维**，自由参数可以取

    u = [log(sigma), log(ms'/ms), log(mb'/mb)]

显式公式（已在工程内做过符号验证）：

    mu_s' = sigma * mu_s
    mu_d' = (ms'/ms) * mu_d                 -> D' = D（关节偏置其实被不变量钉死）
    mu_b' = sigma * Sig - mu_d'
    Ps' = mu_s'/ms'   D' = mu_d'/ms'   Pb' = mu_b'/mb'
    Is' = sigma*G - |mu_s'|^2/ms'
    Ib' = sigma*A - |mu_b'|^2/mb' - |mu_d'|^2/ms'
    fbc'..ks' = sigma * (fbc..ks)

特殊情况：当 ``mu_s = ms*Ps = 0``（连杆 s 质心正好在关节 s 轴上）时，P1/P2 不再是
有效约束，``mu_d'`` 的方向与大小也变成自由的，等价类升到 **5 维**（本工具会自动识别）。

本工具做什么
------------
* 第一列：初始参数值。可从 ``identified_params/<会话>/params.txt`` 里按**任意列名**
  读取（表头里的所有列都会成为选项），也可以直接在表格里手动改。
* 第二列：修改后的值。**默认全空** = 不改。填了哪一行，就要求等价参数取该值。
* 第三列：实时算出的等价参数；第四列显示相对初始值的变化。
* 底部实时显示：剩余自由度（**消除之前**的）、约束秩、残差、物理可行性。

判定规则
--------
* 无约束            -> 自由度 = 等价类维数，等价参数 = 初始参数。
* 约束秩 r < 维数 n -> **欠定**：用"最小化修改前后参数差距平方和"消掉冗余，给出唯一解。
* 约束秩 r = n      -> **唯一确定**。
* 约束条数 > n      -> **过定**：相容则仍是唯一解，矛盾（残差 > 容差）则显示无解。
* 解出来在物理范围外（质量/惯量/摩擦为负、质量矩阵非正定）-> 单独提示具体违规项。

口径说明
--------
本工具的等价关系按**模型含重力**（gx/gy 为已知输入且非零）定义，这时 mu_s 与 Sig
是可观测的。若参数是在 ``--ignore-gravity``（gx=gy=0）口径下辨识的，则等价类更大
（一般位形 7 维），本工具给出的是其中一个 3 维子族——仍然是严格等价的，只是没有
覆盖全部自由度。

跑法::

    cd /home/huhu233/rm2027/TorqueControllerForBigSmallYaw_v2
    python3 python/equivalent_params.py
    python3 python/equivalent_params.py --params identified_params/Sentry1_v1/params.txt

依赖: PyQt5 + numpy + scipy（界面之外的部分不依赖 torch / 已编译库）。
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    import sim_config as _cfg
    PARAM_NAMES = list(_cfg.PARAM_NAMES)
    _DEFAULTS = dict(getattr(_cfg, "DEFAULT_PARAMS", {}))
except Exception:  # pragma: no cover - 独立运行时的兜底
    PARAM_NAMES = ["mb", "Ib", "Pbx", "Pby", "ms", "Is", "Psx", "Psy",
                   "Dx", "Dy", "fbc", "fbv", "fsc", "fsv", "kb", "ks"]
    _DEFAULTS = dict(mb=1.5, Ib=0.030, Pbx=0.12, Pby=0.18, ms=0.40, Is=0.020,
                     Psx=0.08, Psy=0.10, Dx=0.05, Dy=0.30, fbc=0.020, fbv=0.050,
                     fsc=0.010, fsv=0.020, kb=4.0, ks=1.0)

N = len(PARAM_NAMES)
IDX = {n: i for i, n in enumerate(PARAM_NAMES)}
(MB, IB, PBX, PBY, MS, IS, PSX, PSY, DX, DY,
 FBC, FBV, FSC, FSV, KB, KS) = range(16)
#: 物理上取正值的参数（用于给求解器加上下界与给出违规提示）
POSITIVE = (MB, IB, MS, IS, FBC, FBV, FSC, FSV)
#: 只由 sigma 决定的参数（约束它们时可以直接解出 sigma，用作求解种子）
SIGMA_ONLY = (FBC, FBV, FSC, FSV, KB, KS)

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    from scipy.optimize import least_squares, minimize
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"[error] 需要 scipy: {exc}\n安装: pip install scipy") from exc


# ===========================================================================
# 1. 核心数学
# ===========================================================================
U_CLIP = 30.0          # log 参数的取值夹紧（防止 exp 溢出）
FEAS_TOL = 1e-9        # 约束残差容差（绝对）


def orbit(p0: np.ndarray, u) -> np.ndarray:
    """由初始参数 p0 与自由参数 u 生成等价参数。

    u = [log sigma, log(ms'/ms), log(mb'/mb)]
    （退化位形 mu_s = 0 时再追加 [mu_dx', mu_dy']，长度 5）。
    """
    u = np.clip(np.asarray(u, dtype=float), -U_CLIP, U_CLIP)
    mb, Ib, Pbx, Pby = p0[0], p0[1], p0[2], p0[3]
    ms, Is, Psx, Psy = p0[4], p0[5], p0[6], p0[7]
    Dx, Dy = p0[8], p0[9]
    sigma = float(np.exp(u[0]))
    ms2 = ms * float(np.exp(u[1]))
    mb2 = mb * float(np.exp(u[2]))

    mu_s = np.array([ms * Psx, ms * Psy])
    mu_d = np.array([ms * Dx, ms * Dy])
    mu_b = np.array([mb * Pbx, mb * Pby])
    sig_mu = mu_b + mu_d
    A = mb * (Pbx ** 2 + Pby ** 2) + Ib + ms * (Dx ** 2 + Dy ** 2)
    G = ms * (Psx ** 2 + Psy ** 2) + Is

    mu_s2 = sigma * mu_s
    mu_d2 = (ms2 / ms) * mu_d if len(u) == 3 else np.array([u[3], u[4]])
    mu_b2 = sigma * sig_mu - mu_d2

    Ps2 = mu_s2 / ms2
    D2 = mu_d2 / ms2
    Pb2 = mu_b2 / mb2
    Is2 = sigma * G - float(mu_s2 @ mu_s2) / ms2
    Ib2 = sigma * A - float(mu_b2 @ mu_b2) / mb2 - float(mu_d2 @ mu_d2) / ms2
    return np.array([mb2, Ib2, Pb2[0], Pb2[1], ms2, Is2, Ps2[0], Ps2[1],
                     D2[0], D2[1], sigma * p0[10], sigma * p0[11],
                     sigma * p0[12], sigma * p0[13], sigma * p0[14],
                     sigma * p0[15]])


def free_param_count(p0: np.ndarray) -> int:
    """等价类维数（未加约束前的自由度）：一般 3，mu_s = 0 退化时 5。"""
    return 5 if (p0[MS] * p0[PSX] == 0.0 and p0[MS] * p0[PSY] == 0.0) else 3


def physical_violations(p: np.ndarray) -> list:
    """返回物理可行性违规项（空列表 = 可行）。"""
    viol = []
    mb, Ib, Pbx, Pby, ms, Is, Psx, Psy, Dx, Dy = p[:10]
    if not mb > 0:
        viol.append("mb <= 0")
    if not ms > 0:
        viol.append("ms <= 0")
    if Ib < 0:
        viol.append("Ib < 0")
    if Is < 0:
        viol.append("Is < 0")
    for i, nm in ((FBC, "fbc"), (FBV, "fbv"), (FSC, "fsc"), (FSV, "fsv")):
        if p[i] < 0:
            viol.append(f"{nm} < 0")
    A = mb * (Pbx ** 2 + Pby ** 2) + Ib + ms * (Dx ** 2 + Dy ** 2)
    G = ms * (Psx ** 2 + Psy ** 2) + Is
    if A <= 0:
        viol.append("绕关节 b 惯量 A <= 0")
    if G <= 0:
        viol.append("M22 = ms|Ps|^2 + Is <= 0")
    if A * G - ms ** 2 * (Dx ** 2 + Dy ** 2) * (Psx ** 2 + Psy ** 2) <= 0:
        viol.append("det(M) <= 0（质量矩阵非正定）")
    return viol


@dataclass
class SolveResult:
    ok: bool = False                 # 约束是否相容（有等价解）
    p_eq: np.ndarray = None          # 等价参数
    sigma: float = 1.0
    dof: int = 0                     # 剩余自由度（消除之前）
    rank: int = 0                    # 约束 Jacobian 的秩
    ncon: int = 0                    # 用户约束条数
    fiber: int = 3                   # 等价类维数
    residual: float = 0.0            # 约束残差（绝对值最大）
    violations: list = field(default_factory=list)
    message: str = ""
    degenerate: bool = False

    @property
    def feasible_physical(self) -> bool:
        return not self.violations

    @property
    def status(self) -> str:
        """'under' | 'unique' | 'inconsistent' | 'unphysical' | 'degenerate'"""
        if not self.ok:
            return "inconsistent"
        if self.violations:
            return "unphysical"
        if self.dof == 0:
            return "unique"
        return "under"


def solve_equivalent(p0, targets, metric: str = "abs",
                     tol: float = FEAS_TOL) -> SolveResult:
    """在等价类里找满足 targets（{参数下标: 目标值}）且改动平方和最小的参数。

    metric = 'abs'  -> 最小化 sum (p' - p)^2          （题干要求的"差距平方和"）
    metric = 'rel'  -> 最小化 sum ((p' - p)/|p|)^2    （量级悬殊时更均衡）
    """
    p0 = np.asarray(p0, dtype=float)
    tgt = {int(k): float(v) for k, v in targets.items()}
    tidx = sorted(tgt)
    tval = np.array([tgt[i] for i in tidx], dtype=float)
    k = len(tidx)
    n = free_param_count(p0)
    out = SolveResult(p_eq=p0.copy(), ncon=k, fiber=n, dof=n,
                      degenerate=(n == 5))
    if p0[MS] == 0.0 or p0[MB] == 0.0:
        out.message = "初始参数里 ms 或 mb 为 0，等价类退化，无法计算。"
        return out

    u0 = np.zeros(n)
    if n == 5:                       # 退化位形：把初始 mu_d 作为起点
        u0[3], u0[4] = p0[MS] * p0[DX], p0[MS] * p0[DY]

    w = np.ones(N) if metric == "abs" else 1.0 / np.maximum(np.abs(p0), 1e-3)

    def cfun(u):
        return orbit(p0, u)[tidx] - tval if k else np.zeros(0)

    def obj(u):
        return float(((w * (orbit(p0, u) - p0)) ** 2).sum())

    bounds = [(-U_CLIP, U_CLIP)] * 3 + ([(-1e6, 1e6)] * 2 if n == 5 else [])

    # ---- 种子：原点是可行的（自动满足不变量），再给每个"简单约束"一个代数种子 ----
    seeds = [u0]
    for i, v in zip(tidx, tval):
        if i in SIGMA_ONLY and p0[i] != 0.0 and v * p0[i] > 0:
            s = np.zeros(n); s[0] = np.log(abs(v / p0[i])); seeds.append(s)
        elif i == MB and v > 0:
            s = np.zeros(n); s[2] = np.log(v / p0[MB]); seeds.append(s)
        elif i == MS and v > 0:
            s = np.zeros(n); s[1] = np.log(v / p0[MS]); seeds.append(s)

    cands = []
    for s in seeds:
        try:
            r = minimize(obj, s, method="SLSQP", bounds=bounds,
                         constraints=[{"type": "eq", "fun": cfun}],
                         options={"maxiter": 300, "ftol": 1e-14})
            cands.append(r.x)
        except Exception:
            pass
    try:                             # 纯可行性兜底
        fit = least_squares(cfun, u0, method="trf", max_nfev=1500,
                            xtol=1e-15, ftol=1e-15, gtol=1e-15)
        cands.append(fit.x)
    except Exception:
        pass

    best = None
    for u in cands:
        if not np.all(np.isfinite(u)):
            continue
        pe = orbit(p0, u)
        if not np.all(np.isfinite(pe)):
            continue
        c = cfun(u)
        rr = float(np.max(np.abs(c))) if c.size else 0.0
        key = (round(rr, 10), obj(u))
        if best is None or key < best[0]:
            best = (key, u, rr)
    if best is None:
        out.message = "求解失败：没有产生有限候选解。"
        return out
    _, u, residual = best
    pe = orbit(p0, u)

    # ---- 秩 / 剩余自由度（用中心差分；失败时退回 matrix_rank） ----
    rank = 0
    if k:
        J = np.zeros((k, n))
        for i in range(n):
            du = np.zeros(n); du[i] = 1e-7
            J[:, i] = (orbit(p0, u + du)[tidx] - orbit(p0, u - du)[tidx]) / 2e-7
        J = np.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)
        nrm = np.linalg.norm(J, axis=1)
        keep = nrm > 1e-12
        if np.any(keep):
            try:
                sv = np.linalg.svd(J[keep] / nrm[keep][:, None], compute_uv=False)
                rank = int(np.sum(sv > 1e-8 * max(sv[0], 1e-30)))
            except np.linalg.LinAlgError:
                rank = int(np.linalg.matrix_rank(J))

    out.ok = residual < tol
    out.p_eq = pe
    out.sigma = float(np.exp(np.clip(u[0], -U_CLIP, U_CLIP)))
    out.rank = rank
    out.dof = max(n - rank, 0)
    out.residual = residual
    out.violations = physical_violations(pe)
    if not out.ok:
        out.message = (f"约束与等价类不相容：最大残差 {residual:.3e}"
                       f"（可能过定或该参数组合不可能同时达到）")
    return out


# ===========================================================================
# 2. 参数文件读取（表头里所有列都作为选项）
# ===========================================================================
def read_param_table(path) -> tuple:
    """读取 ``参数名 列1 列2 ...`` 形式的参数文件。

    返回 ``(列名列表, {参数名: [该行各列的值, ...]})``；缺少的列用 ``None`` 占位。
    同时兼容 ``名字 = 值`` 写法。
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    raw = [ln for ln in (l.strip() for l in text.splitlines()) if ln]

    header = None
    table = {}
    for ln in raw:
        if "=" in ln and ln.split("=")[0].strip() in IDX:
            name, _, val = ln.partition("=")
            try:
                table.setdefault(name.strip(), []).append(float(val.strip()))
            except ValueError:
                table.setdefault(name.strip(), []).append(None)
            continue
        toks = ln.split()
        if not toks:
            continue
        if toks[0] not in IDX:                      # 表头行
            if header is None:
                header = toks[1:] or None
            continue
        vals = []
        for t in toks[1:]:
            try:
                vals.append(float(t))
            except ValueError:
                vals.append(None)
        table[toks[0]] = vals

    ncol = max((len(v) for v in table.values()), default=0)
    if not header or len(header) < ncol:
        header = [f"列{i + 1}" for i in range(ncol)]
    table = {k: v + [None] * (ncol - len(v)) for k, v in table.items()}
    return header, table


def discover_param_files() -> list:
    """自动发现 ``identified_params/*/params.txt``。"""
    found, seen = [], set()
    for pat in ("identified_params/*/params.txt", "identified_params/**/params.txt"):
        for f in sorted(REPO.glob(pat)):
            if f not in seen:
                seen.add(f)
                found.append(f)
    return found


def defaults_vector() -> np.ndarray:
    return np.array([float(_DEFAULTS.get(n, 0.0)) for n in PARAM_NAMES], dtype=float)


# ===========================================================================
# 3. GUI
# ===========================================================================
try:
    from PyQt5 import QtCore, QtWidgets
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"[error] 需要 PyQt5: {exc}\n安装: pip install PyQt5") from exc

COLS = ("参数", "初始参数值", "修改后的值", "等价参数", "变化 (等价 − 初始)")
COL_EDIT_INIT, COL_TARGET, COL_EQ, COL_DELTA = 1, 2, 3, 4
_MANUAL = "（手动输入）"


def _fmt(v) -> str:
    if v is None:
        return ""
    try:
        return f"{float(v):.6g}"
    except (TypeError, ValueError):
        return str(v)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, params_file=None, column=None):
        super().__init__()
        self.setWindowTitle("等价参数工具 — 保持动力学不变地改写参数")
        self.resize(1020, 700)
        self._p_eq = None
        self._dirty = False

        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(180)          # 防抖：打字时不必每键都解一次
        self._timer.timeout.connect(self._recompute)

        self._build_ui()
        files = discover_param_files()
        if params_file is None and files:
            params_file = str(files[0])          # 默认载入第一个真实参数文件
        self._populate_sources(params_file)
        self._apply_initial(params_file, column)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        # --- 参数来源行 ---
        src = QtWidgets.QHBoxLayout()
        src.addWidget(QtWidgets.QLabel("参数来源"))
        self.src_combo = QtWidgets.QComboBox()
        self.src_combo.setMinimumWidth(340)
        self.src_combo.currentIndexChanged.connect(self._on_source_changed)
        src.addWidget(self.src_combo, 1)
        btn_browse = QtWidgets.QPushButton("浏览…")
        btn_browse.clicked.connect(self._on_browse)
        src.addWidget(btn_browse)
        src.addSpacing(8)
        src.addWidget(QtWidgets.QLabel("列"))
        self.col_combo = QtWidgets.QComboBox()
        self.col_combo.setMinimumWidth(120)
        src.addWidget(self.col_combo)
        btn_load = QtWidgets.QPushButton("载入")
        btn_load.clicked.connect(self._on_load)
        src.addWidget(btn_load)
        src.addSpacing(12)
        src.addWidget(QtWidgets.QLabel("误差度量"))
        self.metric_combo = QtWidgets.QComboBox()
        self.metric_combo.addItem("绝对平方和 Σ Δp²", "abs")
        self.metric_combo.addItem("相对平方和 Σ (Δp/|p|)²", "rel")
        self.metric_combo.currentIndexChanged.connect(self._schedule)
        src.addWidget(self.metric_combo)
        root.addLayout(src)

        # --- 表格 ---
        self.table = QtWidgets.QTableWidget(N, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        hh.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        for r, name in enumerate(PARAM_NAMES):
            it = QtWidgets.QTableWidgetItem(name)
            it.setFlags(QtCore.Qt.ItemIsEnabled)
            it.setTextAlignment(QtCore.Qt.AlignCenter)
            self.table.setItem(r, 0, it)
            for c in (COL_EDIT_INIT, COL_TARGET):
                self.table.setItem(r, c, QtWidgets.QTableWidgetItem(""))
            for c in (COL_EQ, COL_DELTA):
                cell = QtWidgets.QTableWidgetItem("")
                cell.setFlags(QtCore.Qt.ItemIsEnabled)
                cell.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                self.table.setItem(r, c, cell)
        self.table.itemChanged.connect(self._schedule)
        root.addWidget(self.table, 1)

        # --- 状态区 ---
        self.status = QtWidgets.QLabel("—")
        f = self.status.font(); f.setPointSize(f.pointSize() + 2); f.setBold(True)
        self.status.setFont(f)
        self.status.setWordWrap(True)
        root.addWidget(self.status)

        self.detail = QtWidgets.QLabel("—")
        self.detail.setWordWrap(True)
        root.addWidget(self.detail)

        hint = QtWidgets.QLabel(
            "提示：第二列留空 = 不改该参数；填写 = 要求等价参数取该值。"
            "等价关系按模型含重力（gx/gy≠0）定义。")
        hint.setStyleSheet("color:#666;")
        hint.setWordWrap(True)
        root.addWidget(hint)

        # --- 按钮行 ---
        btns = QtWidgets.QHBoxLayout()
        b_clear = QtWidgets.QPushButton("清除修改")
        b_clear.clicked.connect(self._clear_targets)
        btns.addWidget(b_clear)
        b_adopt = QtWidgets.QPushButton("等价参数 → 初始值")
        b_adopt.clicked.connect(self._adopt_equivalent)
        btns.addWidget(b_adopt)
        btns.addStretch(1)
        b_export = QtWidgets.QPushButton("导出等价参数…")
        b_export.clicked.connect(self._export)
        btns.addWidget(b_export)
        root.addLayout(btns)

    # ------------------------------------------------------ 数据源 / 载入
    def _populate_sources(self, select=None):
        self.src_combo.blockSignals(True)
        self.src_combo.clear()
        self.src_combo.addItem(_MANUAL, None)
        for f in discover_param_files():
            try:
                rel = f.relative_to(REPO)
            except ValueError:
                rel = f
            self.src_combo.addItem(str(rel), str(f))
        if select:
            idx = self.src_combo.findData(str(select))
            if idx < 0:
                self.src_combo.addItem(str(select), str(select))
                idx = self.src_combo.count() - 1
            self.src_combo.setCurrentIndex(idx)
        elif self.src_combo.count() > 1:
            self.src_combo.setCurrentIndex(1)
        self.src_combo.blockSignals(False)
        self._fill_columns(self.src_combo.currentData())

    def _fill_columns(self, file_str):
        self.col_combo.clear()
        if not file_str:
            self.col_combo.addItem("—", None)
            return
        try:
            header, _ = read_param_table(file_str)
        except Exception as exc:
            self.col_combo.addItem(f"读取失败: {exc}", None)
            return
        for i, h in enumerate(header):
            self.col_combo.addItem(str(h), i)
        idx = self.col_combo.count() - 1
        self.col_combo.setCurrentIndex(max(idx, 0))

    def _on_source_changed(self, _):
        self._fill_columns(self.src_combo.currentData())

    def _on_browse(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择参数文件", str(REPO / "identified_params"),
            "参数文件 (*.txt);;所有文件 (*)")
        if not path:
            return
        self.src_combo.addItem(path, path)
        self.src_combo.setCurrentIndex(self.src_combo.count() - 1)
        self._on_load()

    def _on_load(self):
        self._apply_initial(self.src_combo.currentData(),
                            self.col_combo.currentText())

    def _apply_initial(self, file_str, column):
        vec = None
        ci = None
        if file_str:
            try:
                header, table = read_param_table(file_str)
                if column is not None and column in header:
                    ci = header.index(column)
                else:
                    cur = self.col_combo.currentData()      # 跟随"列"下拉框
                    ci = cur if isinstance(cur, int) else 0
                vec = []
                for n in PARAM_NAMES:
                    row = table.get(n)
                    val = row[ci] if row and ci < len(row) else None
                    vec.append(val)
                if all(v is None for v in vec):
                    vec = None
            except Exception as exc:
                self._set_status(f"读取参数文件失败：{exc}", "error")
        if vec is None:
            if file_str is None:
                vec = defaults_vector()
            else:
                self._set_status("该列没有可用数值，保留当前初始值。", "warn")
                return
        if isinstance(ci, int) and 0 <= ci < self.col_combo.count():
            self.col_combo.blockSignals(True)
            self.col_combo.setCurrentIndex(ci)
            self.col_combo.blockSignals(False)
        self._set_initial_column(vec)
        self._recompute()

    def _set_initial_column(self, vec):
        self.table.blockSignals(True)
        for r, v in enumerate(vec):
            self.table.item(r, COL_EDIT_INIT).setText(_fmt(v))
        self.table.blockSignals(False)

    # ------------------------------------------------------------ 交互
    def _schedule(self, *_):
        self._timer.start()

    def _clear_targets(self):
        self.table.blockSignals(True)
        for r in range(N):
            self.table.item(r, COL_TARGET).setText("")
        self.table.blockSignals(False)
        self._recompute()

    def _adopt_equivalent(self):
        if self._p_eq is None:
            return
        self._set_initial_column(self._p_eq)
        self._clear_targets()

    def _export(self):
        if self._p_eq is None:
            self._set_status("当前没有可导出的等价参数。", "warn")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "导出等价参数", str(REPO / "equivalent_params.txt"),
            "文本 (*.txt)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(f"{'参数':>8s}{'等价值':>18s}\n")
                for n, v in zip(PARAM_NAMES, self._p_eq):
                    fh.write(f"{n:>8s}{v:>18.10g}\n")
        except Exception as exc:
            self._set_status(f"导出失败：{exc}", "error")
            return
        self._set_status(f"已导出到 {path}", "ok")

    # --------------------------------------------------------- 读取表格
    def _read_initial(self) -> np.ndarray:
        vals = []
        for r in range(N):
            txt = self.table.item(r, COL_EDIT_INIT).text().strip()
            if not txt:
                raise ValueError(f"初始参数 {PARAM_NAMES[r]} 为空")
            try:
                vals.append(float(txt))
            except ValueError:
                raise ValueError(f"初始参数 {PARAM_NAMES[r]} = “{txt}” 不是数值")
        return np.array(vals, dtype=float)

    def _read_targets(self) -> dict:
        out = {}
        for r in range(N):
            txt = self.table.item(r, COL_TARGET).text().strip()
            if not txt:
                continue
            try:
                out[r] = float(txt)
            except ValueError:
                raise ValueError(f"修改值 {PARAM_NAMES[r]} = “{txt}” 不是数值")
        return out

    def _set_eq_column(self, vec, p0):
        self.table.blockSignals(True)
        for r in range(N):
            self.table.item(r, COL_EQ).setText(_fmt(vec[r]) if vec is not None else "")
            if vec is None or p0 is None:
                self.table.item(r, COL_DELTA).setText("")
            else:
                d = float(vec[r]) - float(p0[r])
                self.table.item(r, COL_DELTA).setText("0" if d == 0 else f"{d:+.6g}")
        self.table.blockSignals(False)

    # ------------------------------------------------------------ 主计算
    def _recompute(self):
        try:
            p0 = self._read_initial()
            targets = self._read_targets()
        except ValueError as exc:
            self._p_eq = None
            self._set_eq_column(None, None)
            self._set_status(str(exc), "error")
            return
        metric = self.metric_combo.currentData() or "abs"
        try:
            res = solve_equivalent(p0, targets, metric=metric)
        except Exception as exc:                     # 求解器异常不应打断界面
            self._p_eq = None
            self._set_eq_column(None, None)
            self._set_status(f"求解异常：{exc}", "error")
            return
        self._p_eq = res.p_eq if res.ok else None
        self._set_eq_column(res.p_eq, p0)
        self._report(res)

    # ------------------------------------------------------------ 状态显示
    def _set_status(self, text, kind="info"):
        color = {"ok": "#0a7d2e", "warn": "#b35c00", "error": "#b00020",
                 "info": "#222", "under": "#0b5cad"}[kind]
        self.status.setStyleSheet(f"color:{color};")
        self.status.setText(text)

    def _report(self, res: SolveResult):
        n, k = res.fiber, res.ncon
        over = k > n
        pieces = [f"约束 {k} 条", f"约束秩 {res.rank}", f"等价类 {n} 维",
                  f"剩余自由度（消除前） {res.dof}", f"σ = {res.sigma:.6g}"]
        if res.degenerate:
            pieces.append("检测到 μs = ms·Ps = 0 退化位形")
        if over:
            pieces.append("约束条数已超过等价类维数（过定）")
        self.detail.setText("　|　".join(pieces) + f"　|　最大残差 {res.residual:.2e}")

        if not res.ok:
            self._set_status(f"✗ 无解：{res.message}", "error")
        elif res.violations:
            self._set_status("⚠ 有等价解，但物理不可行：" + "、".join(res.violations),
                             "warn")
            self.detail.setText(self.detail.text() +
                                "　|　提示：可换一个约束或调整初始参数")
        elif res.dof == 0:
            self._set_status("✓ 唯一确定：约束已把等价自由度全部钉死。", "ok")
        else:
            self._set_status(
                f"欠定：剩余自由度 {res.dof} → 已用最小平方和"
                f"（{'绝对' if (self.metric_combo.currentData() == 'abs') else '相对'}）"
                f"消除冗余，给出唯一解。", "under")


# ===========================================================================
# 4. 入口
# ===========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description="等价参数工具（PyQt5）")
    ap.add_argument("--params", default=None,
                    help="启动时载入的参数文件（默认识别 identified_params/*/params.txt）")
    ap.add_argument("--column", default=None, help="启动时使用哪一列（默认最后一列）")
    args = ap.parse_args(argv)

    try:
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    except Exception:
        pass
    app = QtWidgets.QApplication(sys.argv[:1])
    win = MainWindow(args.params, args.column)
    win.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
