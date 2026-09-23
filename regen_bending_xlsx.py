"""一次性刷新: 不重跑仿真, 直接从各动作已有 wrench.csv 重建带弯矩分量的 关节弯矩.xlsx。

用法: python regen_bending_xlsx.py [robot=a2]
"""

import csv
import sys
from pathlib import Path

import numpy as np

from quad_pipeline.config import ROBOTS
from quad_pipeline.motions import MOTIONS
from quad_pipeline.wrench import decompose, write_bending_xlsx


def main():
    robot = sys.argv[1] if len(sys.argv) > 1 else "a2"
    cfg = ROBOTS[robot]
    root = Path("results") / cfg.name
    for key, cls in MOTIONS.items():
        csv_path = root / key / "wrench.csv"
        if not csv_path.exists():
            print(f"跳过 {key}: 无 wrench.csv")
            continue
        with open(csv_path, encoding="utf-8") as f:
            r = csv.reader(f)
            next(r)
            A = np.array([[float(v) for v in row] for row in r])
        dec = decompose(A, cfg)
        out = root / key / "关节弯矩.xlsx"
        write_bending_xlsx(A, dec, cfg, f"{cfg.name} {cls.CN}", out)
        print(f"已刷新 {key}: {out}")


if __name__ == "__main__":
    main()
