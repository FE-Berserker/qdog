"""quad_pipeline — 四足机器狗 MuJoCo 动作仿真与关节载荷分析流水线。

用法见 run_pipeline.py 或 README.md。
"""

from .config import RobotConfig, ROBOTS, register
from .core import CaseSimBase, build_scene, run_sim, save_csv
from .motions import MOTIONS
from . import report, wrench, video, envelope
