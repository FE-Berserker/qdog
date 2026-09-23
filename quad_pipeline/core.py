"""quad_pipeline 核心: 场景构建 / 仿真基类 / 轨迹记录 / 通用工具。

轨迹列布局 (所有动作一致, 供视频/弯矩/包络复用):
  time | base qpos(7) | q(12) | v(12) | tq(12) 执行器力矩 | tg(12) 重力矩
       | tp(12) 被动 | tc(12) 约束 | ti(12) 惯性 | Fz(4) 单足反力
       | st(4) 支撑标志 | base_vz | base_vx | v_cmd
"""

import csv
from pathlib import Path

import mujoco
import numpy as np

from .config import RobotConfig

BASE = Path(__file__).resolve().parent          # quad_pipeline/
WORKSPACE = BASE.parent                          # 工作区根

G = 9.81

SCENE_TMPL = """<mujoco model="{model_name}">
  <include file="{model_xml_abs}"/>
  <compiler meshdir="{mesh_dir_abs}"/>
  <statistic center="0 0 0.4" extent="1.2"/>
  <visual>
    <global offwidth="1280" offheight="960"/>
    <headlight diffuse="0.4 0.4 0.4" ambient="0.4 0.4 0.4" specular="0 0 0"/>
    <map znear=".01" zfar="50"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".4 .6 .8" rgb2="0 0 0" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="5 5" reflectance=".2"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 .05" type="plane" material="grid" condim="3"/>
  </worldbody>
</mujoco>
"""


def _abs(p):
    p = Path(p)
    return str(p if p.is_absolute() else WORKSPACE / p)


def build_scene(cfg: RobotConfig, out_dir: Path, extra: str = "", name: str = "scene.xml"):
    """生成场景 xml (绝对路径引用模型与网格, 与结果目录位置无关)。

    extra: 追加在 </mujoco> 前的定制片段 (如堵转工况的 equality 焊接),
    非空时应给不同的 name 以免覆盖共享场景。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    xml = SCENE_TMPL.format(
        model_name=f"{cfg.name}_scene",
        model_xml_abs=_abs(cfg.model_xml).replace("\\", "/"),
        mesh_dir_abs=_abs(cfg.mesh_dir).replace("\\", "/"),
    )
    if extra:
        xml = xml.replace("</mujoco>", extra + "\n</mujoco>")
    scene = out_dir / name
    scene.write_text(xml, encoding="utf-8")
    return scene


def smoothstep(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * u)


class CaseSimBase:
    """12 关节 PD + 轨迹记录基类; 动作子类实现 init_pose() 与 step()。

    cfg: RobotConfig。model: 可注入传感器增强模型 (弯矩提取时重放用)。
    跌倒判定阈值 Z_LO/ATT_LIM 可在子类覆盖 (如前扑/跌落)。

    工况维度 (子类覆盖, 构成"工况 × 负载 × 地形"矩阵):
      PAYLOAD   躯干附加负载 kg, 按紧凑货载贴近躯干质心建模 (0=空载)
      SLOPE_DEG 等效坡度角 deg; 重力矢量倾斜法: 地面几何保持平地,
                重力向 -x 倾斜, 与上坡在力学上严格等价 (均匀无限坡面),
                视频中"平地 + 机身前倾"即爬坡姿态
    """

    Z_LO = 0.28
    ATT_LIM = 0.7
    PAYLOAD = 0.0
    SLOPE_DEG = 0.0

    def __init__(self, cfg: RobotConfig, model=None, scene=None):
        self.cfg = cfg
        if model is None:
            if scene is None:
                raise ValueError("CaseSimBase 需要 model 或 scene 之一")
            model = mujoco.MjModel.from_xml_path(str(scene))
        self.model = model
        self.data = None
        self._bind_model(model)

    def _bind_model(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        cfg, m = self.cfg, model
        self.joint_names = [cfg.joint_fmt.format(leg=lg, part=p)
                            for lg in cfg.legs for p in cfg.parts]
        self.jq = {n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                   for n in self.joint_names}
        self.jd = {n: m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                   for n in self.joint_names}
        self.act = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                         cfg.act_fmt.format(leg=n.split("_")[0], part=n.split("_")[1]))
                    for n in self.joint_names}
        self.foot_geoms = {}
        for lg in cfg.legs:
            foot = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                     cfg.foot_body_fmt.format(leg=lg))
            self.foot_geoms[lg] = [g for g in range(m.ngeom)
                                   if m.geom_bodyid[g] == foot
                                   and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
        self.trunk_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, cfg.trunk_body)
        self.ctrl_hi = m.actuator_ctrlrange[:, 1].copy()
        self.ctrl_lo = m.actuator_ctrlrange[:, 0].copy()
        self.jrange = {n: m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                       for n in self.joint_names}
        # PD 增益 (A2 调定; 其他机型必要时经 params 覆盖)
        self.kp_st = np.array([[120.0, 240.0, 400.0][i % 3] for i in range(12)])
        self.kd_st = np.array([[3.0, 8.0, 14.0][i % 3] for i in range(12)])
        self.kp_sw = np.array([[60.0, 110.0, 220.0][i % 3] for i in range(12)])
        self.kd_sw = np.array([[2.0, 4.0, 6.0][i % 3] for i in range(12)])
        if cfg.mass <= 0:
            cfg.mass = float(sum(m.body_mass))   # 基准自重 (不含附加负载)
        self._apply_case_conditions(m)
        self.t = 0.0
        self.stance = {lg: True for lg in cfg.legs}
        self.x_td_lat = {lg: 0.0 for lg in cfg.legs}
        self.x_lo_lat = {lg: 0.0 for lg in cfg.legs}
        self.vcmd_now = 0.0
        self.fallen = False
        self._twin = mujoco.MjData(m)

    def _apply_case_conditions(self, m):
        """按类属性施加负载与等效坡度; 普通仿真与 wrench 传感器模型重放共用。"""
        if self.PAYLOAD > 0.0:
            if self.trunk_bid < 0:
                raise ValueError(f"躯干体 {self.cfg.trunk_body!r} 不存在, 请检查 trunk_body 配置")
            m.body_mass[self.trunk_bid] += self.PAYLOAD
            # 惯量增量: 按边长 10 cm 的紧凑箱体 (I = m·s²/6), 重量效应为主, 此项为二阶修正
            m.body_inertia[self.trunk_bid] += self.PAYLOAD * 0.1 ** 2 / 6.0
        if self.SLOPE_DEG != 0.0:
            th = np.radians(self.SLOPE_DEG)
            m.opt.gravity[:] = (-G * np.sin(th), 0.0, -G * np.cos(th))

    # ---- 运动学 ----
    def ik2r(self, ux, uz):
        """平面 2R 逆运动学: 足端相对髋原点 (x 前, z 上) -> (thigh, calf)。"""
        L1, L2 = self.cfg.L1, self.cfg.L2
        r = min(max(float(np.hypot(ux, -uz)), 0.10), self.cfg.r_max)
        cos2 = (r * r - L1 * L1 - L2 * L2) / (2 * L1 * L2)
        th2 = -np.arccos(min(max(cos2, -1.0), 1.0))      # 膝盖弯曲分支 (calf<0)
        psi = np.arccos(min(max((r * r + L1 * L1 - L2 * L2) / (2 * r * L1), -1.0), 1.0))
        return psi - np.arctan2(ux, -uz), th2

    # ---- 状态查询 ----
    def foot_forces(self):
        f = {lg: 0.0 for lg in self.cfg.legs}
        d, m = self.data, self.model
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            for lg, gids in self.foot_geoms.items():
                if g1 in gids or g2 in gids:
                    force = np.zeros(6)
                    mujoco.mj_contactForce(m, d, i, force)
                    f[lg] += force[0]
        return f

    def grav_torque(self):
        """纯重力矩 (冻结速度前向计算)。
        qfrc_bias 按自由度索引 (浮动基座占前 6 个), 必须按各关节 DOF 地址取。"""
        self._twin.qpos[:] = self.data.qpos
        self._twin.qvel[:] = 0.0
        self._twin.act[:] = 0.0
        mujoco.mj_forward(self.model, self._twin)
        return [self._twin.qfrc_bias[self.jd[n]] for n in self.joint_names]

    def body_rpy(self):
        qw, qx, qy, qz = self.data.qpos[3:7]
        roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
        pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0))
        return roll, pitch

    # ---- 步态时钟足端目标 (trot/walk 类动作复用) ----
    def foot_target(self, lg, x_a, x_c, gait_t, duty, step_h, r_stance, phase0, t_start):
        t_gait = max(0.0, self.t - t_start)
        phi = ((t_gait / gait_t) + phase0.get(lg, 0.0)) % 1.0
        if phi < duty:
            u = phi / duty
            x = x_c + x_a * (1.0 - 2.0 * u)
            self.stance[lg] = True
            return x, -r_stance, True
        if self.stance[lg]:
            self.x_lo_lat[lg] = x_c - x_a
            self.x_td_lat[lg] = x_c + x_a
        u = (phi - duty) / (1.0 - duty)
        s = (2.0 * np.pi * u - np.sin(2.0 * np.pi * u)) / (2.0 * np.pi)
        x = self.x_lo_lat[lg] + (self.x_td_lat[lg] - self.x_lo_lat[lg]) * s
        z = -r_stance + step_h * 0.5 * (1.0 - np.cos(2.0 * np.pi * u))
        self.stance[lg] = False
        return x, z, False

    # ---- 底层 PD ----
    def apply_pd(self, q_tgt, kp, kd):
        d, m = self.data, self.model
        for i, n in enumerate(self.joint_names):
            lo, hi = self.jrange[n]
            q_tgt[i] = min(max(q_tgt[i], lo + 0.02), hi - 0.02)
        q = np.array([d.qpos[self.jq[n]] for n in self.joint_names])
        v = np.array([d.qvel[self.jd[n]] for n in self.joint_names])
        ctrl = kp * (q_tgt - q) - kd * v
        idx = np.array([self.act[n] for n in self.joint_names])
        ctrl = np.clip(ctrl, self.ctrl_lo[idx], self.ctrl_hi[idx])
        d.ctrl[idx] = ctrl
        mujoco.mj_step(m, d)
        self.t = d.time
        roll, pitch = self.body_rpy()
        if d.qpos[2] < self.Z_LO or abs(roll) > self.ATT_LIM or abs(pitch) > self.ATT_LIM:
            self.fallen = True

    # ---- 支撑标志: 步态类动作用时钟, 冲击类动作用接触力 ----
    def mark_stance_by_contact(self, ff, thr=5.0):
        for lg in self.cfg.legs:
            self.stance[lg] = ff[lg] > thr

    # ---- 轨迹记录 (统一列布局) ----
    def record(self, rec):
        d = self.data
        rec.append(d.time)
        for i in range(7):
            rec.append(d.qpos[i])
        for n in self.joint_names:
            rec.append(d.qpos[self.jq[n]])
        for n in self.joint_names:
            rec.append(d.qvel[self.jd[n]])
        for n in self.joint_names:
            rec.append(d.actuator_force[self.act[n]])
        grav = self.grav_torque()
        for i in range(12):
            rec.append(grav[i])
        for i in range(12):
            rec.append(d.qfrc_passive[i])
        for i in range(12):
            rec.append(d.qfrc_constraint[i])
        for i in range(12):
            rec.append(d.qfrc_bias[i] - grav[i])
        ff = self.foot_forces()
        for lg in self.cfg.legs:
            rec.append(ff[lg])
        for lg in self.cfg.legs:
            rec.append(1 if self.stance[lg] else 0)
        rec.append(d.qvel[2])
        rec.append(d.qvel[0])
        rec.append(self.vcmd_now)
        return rec

    def stand_pose(self, q0, r=None, x=0.0):
        cfg = self.cfg
        r = cfg.r_stand if r is None else r
        q0[2] = cfg.foot_r + r
        for lg in cfg.legs:
            th1, th2 = self.ik2r(x, -r)
            q0[self.jq[cfg.joint_fmt.format(leg=lg, part="thigh")]] = th1
            q0[self.jq[cfg.joint_fmt.format(leg=lg, part="calf")]] = th2
        return q0


def col_names(cfg: RobotConfig):
    jn = [cfg.joint_fmt.format(leg=lg, part=p) for lg in cfg.legs for p in cfg.parts]
    return (["time"] + [f"base_{s}" for s in ("x", "y", "z", "qw", "qx", "qy", "qz")]
            + [f"q_{n}" for n in jn] + [f"v_{n}" for n in jn] + [f"tq_{n}" for n in jn]
            + [f"tg_{n}" for n in jn] + [f"tp_{n}" for n in jn]
            + [f"tc_{n}" for n in jn] + [f"ti_{n}" for n in jn]
            + [f"Fz_{lg}" for lg in cfg.legs] + [f"st_{lg}" for lg in cfg.legs]
            + ["base_vz", "base_vx", "v_cmd"])


# 列索引
C_TQ, C_FZ, C_ST, C_VZ, C_VX, C_VCMD = 32, 92, 96, 100, 101, 102
C_BX, C_BY, C_BZ = 1, 2, 3


def save_csv(path, A, cfg):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(col_names(cfg))
        w.writerows([[float(v) for v in row] for row in A])


def run_sim(sim, t_end):
    """通用仿真主循环 -> 轨迹数组。"""
    sim.init_pose()
    recs = []
    for _ in range(int(t_end / sim.model.opt.timestep)):
        sim.step()
        row = []
        sim.record(row)
        recs.append(row)
        if sim.fallen:
            print(f"!! 跌倒/姿态超限, t={sim.t:.3f} 提前结束")
            break
    return np.array(recs)
