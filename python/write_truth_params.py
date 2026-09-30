"""把仿真真值参数写成 data/sim/truth_params.txt（供辨识脚本对比）。

单独成文件的原因：两个**采集脚本**不应接触真实参数（它们是可按同样逻辑跑在真机上的
采集流程），而"造数据用的真值"是仿真侧的东西，放在这里最清楚。

用法::

    python3 python/write_truth_params.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import sim_config as cfg  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=None,
                    help="输出目录（优先于 --category）；缺省时按 --category 或 data/sim")
    ap.add_argument("--category", type=str, default=None,
                    help="类别名（= 目录名）：写到 data/<类别>/truth_params.txt")
    args = ap.parse_args()

    out_dir = (Path(args.out) if args.out
               else (cfg.category_dir(args.category) if args.category else cfg.DATA_DIR_SIM))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "truth_params.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 真实动力学参数（供辨识脚本对比）\n")
        f.write(f"# dt={cfg.DT} num_steps={cfg.NUM_STEPS} refinement={cfg.REFINEMENT} "
                f"duration={cfg.DURATION}s\n")
        f.write(f"# noise_sigma: pos={cfg.SIGMA_POS} vel={cfg.SIGMA_VEL} tau={cfg.SIGMA_TAU}\n")
        f.write(f"# weights: {cfg.W_PSI_B} {cfg.W_PSI_S} {cfg.W_DPSI_B} {cfg.W_DPSI_S}\n")
        for n in cfg.PARAM_NAMES:
            f.write(f"{n} {cfg.DEFAULT_PARAMS[n]:.10g}\n")
        f.write(f"lambda_ {cfg.LAMBDA:.10g}\n")
    print(f"真实参数 -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
