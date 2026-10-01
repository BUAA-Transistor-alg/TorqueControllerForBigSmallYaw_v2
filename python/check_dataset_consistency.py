#!/usr/bin/env python3
"""检查数据集里"记录的角速度 dpsi"与"相邻两点角度差分"是否一致。

用法:
    python check_dataset_consistency.py data/Sentry1
    python check_dataset_consistency.py data/simA --plot out.png

结论（Sentry1 真机数据）：两者一致，但要注意**对齐约定**——
dpsi[k] 对应的是**后向差分** (psi[k] - psi[k-1]) / dt，不是前向差分。
激励频带内（0.2-3 Hz）增益 ~1.000、相位 ~0、相干 ~1；
高频段（>5 Hz）的差异来自编码器量化（关节 b 步长 2*pi/8192）被差分放大，
而 MCU 上报的 omega 更平滑。
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np


def _spectral_gain(files, joint, bands):
    """按频带汇总 |V| / |FD| 增益与相干。"""
    acc = {b: [0.0, 0.0, 0.0] for b in bands}
    for f in files:
        d = np.load(f)
        dt = float(d["dt"])
        psi = d["psi_" + joint].astype(float)
        v = d["dpsi_" + joint].astype(float)
        a = (psi[1:] - psi[:-1]) / dt
        b = v[1:]
        n = len(a)
        fr = np.fft.rfftfreq(n, dt)
        w = np.hanning(n)
        Fa, Fb = np.fft.rfft(a * w), np.fft.rfft(b * w)
        for lo, hi in bands:
            m = (fr >= lo) & (fr < hi)
            if not m.any():
                continue
            acc[(lo, hi)][0] += float(np.sum(np.abs(Fa[m]) ** 2))
            acc[(lo, hi)][1] += float(np.sum(np.abs(Fb[m]) ** 2))
            acc[(lo, hi)][2] += float(np.abs(np.sum(Fb[m] * np.conj(Fa[m]))))
    out = []
    for lo, hi in bands:
        sa, sb, sp = acc[(lo, hi)]
        if sa <= 0.0:
            continue
        out.append((lo, hi, float(np.sqrt(sb / sa)), float(sp / np.sqrt(sa * sb))))
    return out


def _best_shift(psi, v, dt, smax=5):
    """在 [-smax, smax] 内找最佳对齐：v[k] vs (psi[k+s+1]-psi[k+s])/dt。"""
    fd = (psi[1:] - psi[:-1]) / dt
    best = None
    for s in range(-smax, smax + 1):
        i0, i1 = max(0, -s), min(len(v), len(fd) - s)
        a, b = fd[i0 + s:i1 + s], v[i0:i1]
        if len(a) < 30:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        if best is None or r > best[1]:
            best = (s, r, float(np.dot(a, b) / np.dot(a, a)))
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default="data/Sentry1", help="数据目录（*.npz）")
    ap.add_argument("--plot", default=None, help="可选：输出对比图 PNG 路径")
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.dir) / "*.npz")))
    if not files:
        print(f"[错误] {args.dir} 下没有 .npz 文件")
        return 1
    print(f"目录: {args.dir}   文件数: {len(files)}")

    # ---- 逐条找最佳对齐 ----
    shifts = []
    for f in files:
        d = np.load(f)
        dt = float(d["dt"])
        for j in "bs":
            shifts.append(_best_shift(d["psi_" + j].astype(float),
                                      d["dpsi_" + j].astype(float), dt)[0])
    uniq, cnt = np.unique(shifts, return_counts=True)
    print("最佳对齐(样本偏移)分布:", dict(zip(uniq.tolist(), cnt.tolist())))

    # ---- 后向差分汇总 ----
    print("\n=== 汇总: dpsi[k] vs (psi[k]-psi[k-1])/dt ===")
    for j in "bs":
        A, B = [], []
        for f in files:
            d = np.load(f)
            dt = float(d["dt"])
            psi = d["psi_" + j].astype(float)
            A.append((psi[1:] - psi[:-1]) / dt)
            B.append(d["dpsi_" + j].astype(float)[1:])
        a, b = np.concatenate(A), np.concatenate(B)
        res = b - a
        vr = float(np.sqrt((b ** 2).mean()))
        rr = float(np.sqrt((res ** 2).mean()))
        print(f"  关节 {j}: 点数={len(a)}  相关={np.corrcoef(a, b)[0, 1]:.4f}  "
              f"增益={np.dot(a, b) / np.dot(a, a):.4f}  "
              f"MAE={np.abs(res).mean():.4f} rad/s  相对RMS={100 * rr / vr:.2f}%  "
              f"(vRMS={vr:.3f}, resMax={np.abs(res).max():.3f})")

    # ---- 频带增益 ----
    bands = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0),
             (3.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 50.0)]
    print("\n=== 频带增益 |V|/|FD| 与相干 ===")
    for j in "bs":
        print(f"  关节 {j}:")
        for lo, hi, g, c in _spectral_gain(files, j, bands):
            print(f"    {lo:5.1f}-{hi:5.1f} Hz  增益={g:.4f}  相干={c:.4f}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        d = np.load(files[len(files) // 2])
        dt = float(d["dt"])
        t = np.arange(int(d["num_steps"])) * dt
        fig, axes = plt.subplots(3, 2, figsize=(13, 9))
        for col, j in ((0, "b"), (1, "s")):
            psi = d["psi_" + j].astype(float)
            v = d["dpsi_" + j].astype(float)
            fd = (psi[1:] - psi[:-1]) / dt
            ax = axes[0, col]
            ax.plot(t[1:], fd, lw=1.6, color="tab:orange",
                    label=r"backward diff $(\psi_k-\psi_{k-1})/dt$")
            ax.plot(t[1:], v[1:], lw=1.0, ls="--", color="tab:blue",
                    label=r"recorded $d\psi_k$")
            ax.set_title(f"joint {j}: time domain")
            ax.set_xlabel("t [s]"); ax.set_ylabel("omega [rad/s]")
            ax.legend(fontsize=8); ax.grid(alpha=.3)
        fig.suptitle(f"dataset consistency check: {args.dir}")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=130)
        print(f"\n图已保存: {args.plot}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
