"""quad_pipeline 机器人配置。

新增一个机器狗 = 新增一个 RobotConfig 预设并在 ROBOTS 注册, 流水线其余部分不变。
关节命名约定 (宇树系): 关节 "{leg}_{part}_joint", 执行器 "{leg}_{part}", 子体 "{leg}_{part}";
leg ∈ {FL, RL, FR, RR}, part ∈ {hip, thigh, calf}。其他品牌改 *_fmt 模板即可。
"""

from dataclasses import dataclass, field

WORKSPACE = None  # 由 core 初始化时注入 (流水线包所在目录的上一级)


@dataclass
class RobotConfig:
    name: str                              # 机器人标识 (输出目录名)
    model_xml: str                         # 机器人模型 xml (相对工作区或绝对路径)
    mesh_dir: str                          # 网格目录 (同上浮点)
    legs: tuple = ("FL", "RL", "FR", "RR")
    parts: tuple = ("hip", "thigh", "calf")
    joint_fmt: str = "{leg}_{part}_joint"
    act_fmt: str = "{leg}_{part}"
    body_fmt: str = "{leg}_{part}"
    foot_body_fmt: str = "{leg}_calf"      # 足端碰撞球所在体
    # 关节轴 (子体局部系, 用于弯矩分解): 宇树系 hip 绕 x, thigh/calf 绕 y
    part_axis: dict = field(default_factory=lambda: {
        "hip": (1.0, 0.0, 0.0), "thigh": (0.0, 1.0, 0.0), "calf": (0.0, 1.0, 0.0)})
    L1: float = 0.275                      # 大腿杆长 (髋->膝) m
    L2: float = 0.275                      # 小腿杆长 (膝->足球心) m
    foot_r: float = 0.032                  # 足底球半径 m
    r_max: float = 0.52                    # IK 最大腿长 m (受小腿关节限位约束)
    r_stand: float = 0.44                  # 站立腿长 m
    rated: tuple = (120.0, 120.0, 180.0)   # 各部位执行器额定力矩 N·m (hip, thigh, calf)
    mass: float = 0.0                      # 整机质量 kg; 0 -> 从模型自动求和


ROBOTS = {}


def register(cfg: RobotConfig):
    ROBOTS[cfg.name] = cfg
    return cfg


# ---- 宇树 A2 预设 ----
A2 = register(RobotConfig(
    name="a2",
    model_xml="unitree_robots/a2/a2.xml",
    mesh_dir="unitree_robots/a2/meshes/",
))
