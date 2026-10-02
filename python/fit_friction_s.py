"""单独拟合小 yaw（关节 s）摩擦参数 fsc/fsv —— 从 identify_params.py 里摘出来的那一部分。

和 identify_params.py 的关系
    * 拟合算法**完全共用** ``identify_params.fit_friction_sweep_s``（正反趟差分 +
      可选 p_dyn 动力学修正），所以两边结果一模一样；
    * 区别只在于它是独立入口：不读主数据集、不训练，只做摩擦拟合，方便单独扫速度点。

速度点的取舍
    按数据里**实际存在的 |ω|** 从小到大排序后：
      * ``--drop-first N``  丢掉最慢的 N 个速度点；
      * ``--drop-last  M``  丢掉最快的 M 个速度点；
      * ``--vmin/--vmax``   只保留 |ω| 落在区间内的点。
    拟合前先把**全部 |ω|** 和**实际使用的 |ω|** 打出来（``--list`` 可以只看不拟合）。

``p_dyn``（用于扣残余的 M22·Δθ̈_s + M12·Δθ̈_b + ΔG2）有三个来源：
      * ``--category/--data``：从主数据集用代数最小二乘反解（和 identify_params 一致）；
      * ``--p-dyn "ms=...,Psx=...,..."``：直接给；
      * 都不给 / ``--no-dyn``：只做纯差分，不需要任何已辨识参数。

用法::

    # 只看数据里有哪些速度点
    python3 python/fit_friction_s.py --sweep data/friction_s_Sentry1 --list

    # 丢掉最慢 2 个、最快 1 个速度点后再拟合
    python3 python/fit_friction_s.py --sweep data/friction_s_Sentry1 \
        --drop-first 2 --drop-last 1

    # 只保留 |ω| ∈ [0.05, 3]，并用主数据集反解的 p_dyn 做动力学修正
    python3 python/fit_friction_s.py --sweep data/friction_s_Sentry1 \
        --category Sentry1 --vmin 0.05 --vmax 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import sim_config as cfg                                        # noqa: E402
import identify_params as ip                                    # noqa: E402


def parse_p_dyn(spec: str) -> dict:
    """解析 "ms=1.01,Psx=-0.0095,..." → dict（键名任意，喂给 p_dyn 用）。"""
    out: dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"--p-dyn 每项要写成 名字=值，收到 {part!r}")
        k, v = part.split("=", 1)
        out[k.strip()] = float(v)
    need = {"ms", "Psx", "Psy", "Is", "Dx", "Dy"}
    missing = need - set(out)
    if missing:
        raise ValueError(f"--p-dyn 缺这些键: {', '.join(sorted(missing))}"
                         f"（动力学修正要用 M22=Is+ms|Ps|²、M12=Is_+ms·h、G2）")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=str, default=None,
                    help="小 yaw 匀速往返数据：单个 npz 或类别目录（合并其中 sweep_*.npz）；"
                         "缺省 data/friction_s")
    ap.add_argument("--drop-first", type=int, default=0,
                    help="丢掉最慢的 N 个速度点（按数据里实际存在的 |ω| 升序）")
    ap.add_argument("--drop-last", type=int, default=0,
                    help="丢掉最快的 M 个速度点")
    ap.add_argument("--vmin", type=float, default=None,
                    help="只保留 |ω| ≥ 该值的数据点 [rad/s]")
    ap.add_argument("--vmax", type=float, default=None,
                    help="只保留 |ω| ≤ 该值的数据点 [rad/s]")
    ap.add_argument("--list", action="store_true",
                    help="只列出数据里的速度点与取舍结果，不拟合")
    # ---- p_dyn 的来源 ----
    ap.add_argument("--category", type=str, default=None,
                    help="主数据集类别名（= 目录名）：用它代数反解 p_dyn，和 identify_params 一致")
    ap.add_argument("--data", type=str, default=None,
                    help="主数据集目录（优先于 --category），里面需有 case_*.npz")
    ap.add_argument("--p-dyn", type=str, default=None,
                    help='直接给动力学参数 "ms=...,Psx=...,Psy=...,Is=...,Dx=...,Dy=..."')
    ap.add_argument("--no-dyn", dest="dyn_correct", action="store_false",
                    help="不做动力学修正（纯正反趟差分，完全不需要已辨识参数）")
    args = ap.parse_args()

    sweep_path = Path(args.sweep) if args.sweep else cfg.DATA_DIR_FRICTION_S
    if not sweep_path.exists():
        print(f"找不到小 yaw 数据: {sweep_path}")
        return 1

    # ---- 只列速度点（--list）：不拟合，也不需要 p_dyn ----
    if args.list:
        print(f"列出小 yaw 数据的速度点 {sweep_path}")
        try:
            _fsc, _fsv, info = ip.fit_friction_sweep_s(
                sweep_path, None, dyn_correct=False,
                drop_first=args.drop_first, drop_last=args.drop_last,
                vmin=args.vmin, vmax=args.vmax, verbose=True, list_only=True)
        except ValueError as e:
            print(f"列出失败：{e}")
            return 1
        print(f"\n{'|ω| [rad/s]':>14} {'用/丢':>6}")
        for w, k in zip(info["all_omega"], info["keep"]):
            print(f"{w:>14.4f} {'用' if k else '丢':>6}")
        return 0

    # ---- p_dyn ----
    p_dyn = None
    if args.dyn_correct:
        if args.p_dyn:
            try:
                p_dyn = parse_p_dyn(args.p_dyn)
            except ValueError as e:
                ap.error(str(e))
            print(f"p_dyn: 来自 --p-dyn（{len(p_dyn)} 个键）")
        elif args.data or args.category:
            data_dir = (Path(args.data) if args.data
                        else cfg.category_dir(args.category))
            files = sorted(data_dir.glob("case_*.npz"))
            if not files:
                print(f"主数据集里没有 case_*.npz: {data_dir}")
                return 1
            cases = [ip.Case(f) for f in files]
            p_dyn, _ = ip.algebraic_init(cases, friction=(0.0, 0.0, 0.0, 0.0))
            print(f"p_dyn: 由 {data_dir}（{len(files)} 条 case）代数反解得到")
        else:
            print("没有给 --category/--data/--p-dyn，自动退化为**纯差分**（不做动力学修正）\n")
            args.dyn_correct = False

    if p_dyn is not None:
        m22 = ip._M22_of(p_dyn)
        print(f"  M22=Is+ms|Ps|²={m22:.6g} kg·m²"
              f"   ms={p_dyn['ms']:.6g} |Ps|={np.hypot(p_dyn['Psx'], p_dyn['Psy']):.6g}"
              f"   Dx={p_dyn['Dx']:.6g} Dy={p_dyn['Dy']:.6g}")

    print(f"\n拟合小 yaw 摩擦 {sweep_path}")
    try:
        fsc, fsv, info = ip.fit_friction_sweep_s(
            sweep_path, p_dyn, dyn_correct=args.dyn_correct,
            drop_first=args.drop_first, drop_last=args.drop_last,
            vmin=args.vmin, vmax=args.vmax, verbose=True,
            # 数据里的 ts_mean 是指令值：用与 p_dyn 同标度的 ks 换算成物理力矩
            ks=(float(p_dyn["ks"]) if p_dyn is not None else 1.0))
    except ValueError as e:
        print(f"拟合失败：{e}")
        return 1

    how = ("正反趟差分 + p_dyn 扣 M22·Δθ̈_s/M12·Δθ̈_b/ΔG2" if (args.dyn_correct
           and info["mode"] == "diff") else
           "单趟 + p_dyn 扣 G2" if info["mode"] == "single" else "纯正反趟差分")
    print(f"\n结果（{info['files']} 个文件，{info['n']} 趟，{info['pairs']} 对，"
          f"用了 {len(info['used_omega'])} 个速度点，残差 {info['residual']:.3e} N·m，{how}）:")
    print(f"  fsc = {fsc:.6f}   fsv = {fsv:.6f}")
    print(f"  对照·纯差分（不扣任何动力学项）: fsc={info['fsc_raw']:.6f} "
          f"fsv={info['fsv_raw']:.6f}")

    # ---- 速度点逐个展示（用哪个、丢了哪个）----
    if info["mode"] == "diff" and info["keep"]:
        print(f"\n{'|ω| [rad/s]':>14} {'用/丢':>6}")
        for w, k in zip(info["all_omega"], info["keep"]):
            print(f"{w:>14.4f} {'用' if k else '丢':>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
