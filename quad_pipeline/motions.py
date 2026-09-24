"""quad_pipeline 动作库: 行走 / 奔跑 / 跳跃 / 前扑 / 跌落 + 载荷地形工况。

每个动作 = 一个 CaseSimBase 子类 (init_pose + step) + 元数据注册 (MOTIONS)。
参数均为 A2 实机调定值; 换机型时在子类 PARAMS 基础上派生覆盖即可。

各动作经实机仿真验证 (A2, 2026-09):
  walk  四节拍行走, 实测 0.39 m/s, 姿态峰峰值 2.3°/1.8°
  trot  对角小跑 v0.3, 实测 1.33 m/s (指令 1.5), 姿态峰峰值 8.0°/3.8°
  jump  原地跳跃, 起跳 vz≈2.0 m/s, 落地缓冲
  leap  前扑 v2, 起跳 vx=0.71/vz=1.58, 腾空 0.34 s 前进 0.24 m, 前腿落地缓冲
  drop  0.30 m 自由跌落, 触地 GRF 峰值 ≈ 4.7 kN (11.9 倍体重)

载荷/地形工况 (按设计工况矩阵扩展, 负载/坡度为类属性, 变体即新工况):
  walk_full   满载行走: 平地 walk + 25 kg 躯干负载 (A2 持续行走负载)
  slope       爬坡: 20° 等效上坡 (重力矢量倾斜法), 实测上行 0.17 m/s
  slope_full  满载爬坡: 15° + 25 kg, 实测上行 0.42 m/s —— 额定扭矩与热平衡
              的决定性持续工况 (坡度/速度受时钟步态扭矩预算限制, 见 DESC)
  stand_load  静站载重: 站立保持 + 100 kg (A2 站立最大负载), 结构静强度工况

异常/失效工况 (A2, 2026-09 实仿):
  push  侧向推搡: trot 中躯干 50 N×0.25 s 侧向冲击, 姿态峰峰值 9.0°/4.3°, 恢复
  fall  侧跌倒落: 0.30 m+25° 侧倾非受控跌落, 触地 GRF 峰值 1905 N (~4.8 倍体重)
  stall 关节堵转: FL 膝 equality 焊接卡滞+下蹲, FL_calf 持续饱和 180 N·m (执行器限幅)
"""

import mujoco
import numpy as np

from .core import CaseSimBase, smoothstep, G

MOTIONS = {}


def motion(key, cn, desc, t_end, win):
    """注册动作元数据。win: (t0, t1) 统计窗口。"""
    def deco(cls):
        cls.KEY, cls.CN, cls.DESC = key, cn, desc
        cls.T_END, cls.WIN = t_end, win
        MOTIONS[key] = cls
        return cls
    return deco


# ============================================================ 行走 (四节拍)
@motion("walk", "行走", "四节拍爬行 walk: FR->RL->FL->RR 依次迈步, 占空比0.80, 任意时刻>=3腿支撑",
        t_end=8.0, win=(2.8, 8.0))
class WalkMotion(CaseSimBase):
    # 步态平滑化 (原参数 GAIT_T 0.80 / DUTY 0.75 / V_MAX 0.6 实测: 前进速度呈
    # "落地尖峰 -> 滑移衰减" 的 5 Hz 锯齿 ±27%, 躯干上下跳动 24.5 mm, 观感一顿一顿)。
    # 三处联动改动: 提高步频 (每步扰动幅度更小) + 加大占空比 (同时支撑腿更多、
    # 载荷转移更平缓) + 提高摆动相刚度 SW_SCALE (摆动相在 (1-DUTY)·GAIT_T 内要前送
    # 2·x_a, 步频/占空比提高后摆动速度上升, 刚度不足则足端来不及到位、落地打滑)。
    # 实测: 空载 v 0.386→0.470, 速度脉动 ±27%→±18%, 躯干跳动 24.5→7.6 mm;
    # 满载 v 0.386→0.401, 脉动 ±27%→±17%, 跳动 24.5→10.8 mm。
    GAIT_T, DUTY, STEP_H = 0.65, 0.80, 0.06
    V_MAX, K_FB = 0.60, 0.15
    SW_SCALE = 2.0           # 摆动相刚度倍率
    T_SETTLE, T_RAMP = 0.8, 1.2
    K_PITCH, K_PITCHD = 0.10, 0.03
    K_ROLL, K_ROLLD = 0.10, 0.03
    R_ST = None              # 站姿腿长覆盖 (坡道蹲低降质心)
    XC_LIM = 0.12            # 速度反馈足端修正限幅 (坡道需加大抗后滑)
    XA_BOOST = 1.0           # 扫腿幅值前馈系数 (坡道补偿系统性速度缺口+重力拖拽)

    def __init__(self, cfg, model=None, scene=None):
        super().__init__(cfg, model=model, scene=scene)
        self.PHASE0 = {"FR": 0.0, "RL": 0.25, "FL": 0.5, "RR": 0.75}
        if self.SW_SCALE != 1.0:
            self.kp_sw = self.kp_sw * self.SW_SCALE
            self.kd_sw = self.kd_sw * float(np.sqrt(self.SW_SCALE))
        self.r_st = self.R_ST or cfg.r_stand
        self.x_grav = 0.0      # 足端支撑中心前馈偏移 (坡度补偿在子类设置)

    def v_cmd(self):
        if self.t < self.T_SETTLE:
            v = 0.0
        elif self.t < self.T_SETTLE + self.T_RAMP:
            v = self.V_MAX * smoothstep((self.t - self.T_SETTLE) / self.T_RAMP)
        else:
            v = self.V_MAX
        self.vcmd_now = v
        return v

    def init_pose(self):
        q0 = np.zeros(self.model.nq)
        self.stand_pose(q0, r=self.r_st, x=self.x_grav)
        self.data.qpos[:] = q0
        mujoco.mj_forward(self.model, self.data)

    def step(self):
        d = self.data
        v_x, vcmd = d.qvel[0], self.v_cmd()
        roll, pitch = self.body_rpy()
        roll_r, pitch_r = d.qvel[3], d.qvel[4]
        T_st = self.DUTY * self.GAIT_T
        x_a = float(np.clip(0.5 * max(vcmd, 0.05) * T_st * self.XA_BOOST, 0.02, 0.26))
        x_c = float(np.clip(self.K_FB * (v_x - vcmd), -self.XC_LIM, self.XC_LIM)) + self.x_grav
        q_tgt = np.zeros(12)
        kp = np.zeros(12)
        kd = np.zeros(12)
        for li, lg in enumerate(self.cfg.legs):
            x_f, z_f, st = self.foot_target(lg, x_a, x_c, self.GAIT_T, self.DUTY,
                                            self.STEP_H, self.r_st,
                                            self.PHASE0, self.T_SETTLE)
            depth = -z_f
            if st:
                dr = (self.K_PITCH * pitch + self.K_PITCHD * pitch_r) * (1.0 if lg.startswith("F") else -1.0)
                dr += (self.K_ROLL * roll + self.K_ROLLD * roll_r) * (-1.0 if lg[1] == "L" else 1.0)
                depth += float(np.clip(dr, -0.05, 0.05))
            depth = min(max(depth, 0.12), self.cfg.r_max - 0.02)
            th1, th2 = self.ik2r(x_f, -depth)
            q_tgt[li * 3 + 1] = th1
            q_tgt[li * 3 + 2] = th2
            kp[li * 3:li * 3 + 3] = self.kp_st[li * 3:li * 3 + 3] if st else self.kp_sw[li * 3:li * 3 + 3]
            kd[li * 3:li * 3 + 3] = self.kd_st[li * 3:li * 3 + 3] if st else self.kd_sw[li * 3:li * 3 + 3]
        self.apply_pd(q_tgt, kp, kd)


# ============================================================ 奔跑 (对角小跑 v0.3)
@motion("trot", "奔跑", "对角小跑 trot v0.3: 占空比0.55+扫腿幅值前馈补偿25%, 指令1.5 m/s",
        t_end=8.0, win=(2.8, 8.0))
class TrotMotion(WalkMotion):
    # trot 单独调定 (步频 0.45 / 占空比 0.55 的弹跳步态), 不继承 walk 的步态平滑改动
    SW_SCALE = 1.0
    GAIT_T, DUTY, STEP_H = 0.45, 0.55, 0.08
    V_MAX = 1.5
    XA_BOOST = 1.25          # 扫腿幅值前馈补偿 (系统性速度缺口约 25%)
    T_SETTLE, T_RAMP = 0.8, 1.4

    def __init__(self, cfg, model=None, scene=None):
        super().__init__(cfg, model=model, scene=scene)
        GROUP_A = ("FL", "RR")
        self.PHASE0 = {lg: (0.0 if lg in GROUP_A else 0.5) for lg in self.cfg.legs}

    def step(self):
        d = self.data
        v_x, vcmd = d.qvel[0], self.v_cmd()
        roll, pitch = self.body_rpy()
        roll_r, pitch_r = d.qvel[3], d.qvel[4]
        T_st = self.DUTY * self.GAIT_T
        x_a = float(np.clip(0.5 * max(vcmd, 0.05) * T_st * self.XA_BOOST, 0.02, 0.26))
        x_c = float(np.clip(self.K_FB * (v_x - vcmd), -0.12, 0.12))
        q_tgt = np.zeros(12)
        kp = np.zeros(12)
        kd = np.zeros(12)
        for li, lg in enumerate(self.cfg.legs):
            x_f, z_f, st = self.foot_target(lg, x_a, x_c, self.GAIT_T, self.DUTY,
                                            self.STEP_H, self.cfg.r_stand,
                                            self.PHASE0, self.T_SETTLE)
            depth = -z_f
            if st:
                dr = (self.K_PITCH * pitch + self.K_PITCHD * pitch_r) * (1.0 if lg.startswith("F") else -1.0)
                dr += (self.K_ROLL * roll + self.K_ROLLD * roll_r) * (-1.0 if lg[1] == "L" else 1.0)
                depth += float(np.clip(dr, -0.05, 0.05))
            depth = min(max(depth, 0.12), self.cfg.r_max - 0.02)
            th1, th2 = self.ik2r(x_f, -depth)
            q_tgt[li * 3 + 1] = th1
            q_tgt[li * 3 + 2] = th2
            kp[li * 3:li * 3 + 3] = self.kp_st[li * 3:li * 3 + 3] if st else self.kp_sw[li * 3:li * 3 + 3]
            kd[li * 3:li * 3 + 3] = self.kd_st[li * 3:li * 3 + 3] if st else self.kd_sw[li * 3:li * 3 + 3]
        self.apply_pd(q_tgt, kp, kd)


# ============================================================ 原地跳跃
@motion("jump", "原地跳跃", "下蹲蓄力->爆发蹬伸->腾空->落地缓冲->恢复站立",
        t_end=3.2, win=(0.0, 3.2))
class JumpMotion(CaseSimBase):
    Z_LO = 0.15           # 深蹲/缓冲时躯干可至 0.25 m 以下, 放宽跌倒阈值
    ATT_LIM = 0.9

    T_SETTLE, T_CROUCH, T_PUSH = 0.5, 0.9, 1.1
    R_STAND, R_CROUCH, R_PUSH, R_AIR = 0.42, 0.22, 0.50, 0.38
    T_ABSORB, T_RECOVER = 0.55, 0.8

    def __init__(self, cfg, model=None, scene=None):
        super().__init__(cfg, model=model, scene=scene)
        # 跳跃用更高刚度的 PD (与 A2 跳跃验证版一致)
        self.kp_st = np.array([[120.0, 260.0, 600.0][i % 3] for i in range(12)])
        self.kd_st = np.array([[3.0, 6.0, 10.0][i % 3] for i in range(12)])
        self.phase = "settle"
        self.events = {}

    def init_pose(self):
        q0 = np.zeros(self.model.nq)
        self.stand_pose(q0, r=self.R_STAND)
        self.data.qpos[:] = q0
        mujoco.mj_forward(self.model, self.data)

    def cmd_r(self):
        t, ph = self.t, self.phase
        if ph == "settle":
            return self.R_STAND
        if ph == "crouch":
            return self.R_STAND + (self.R_CROUCH - self.R_STAND) * smoothstep(
                (t - self.T_SETTLE) / (self.T_CROUCH - self.T_SETTLE))
        if ph == "push":
            return self.R_CROUCH + (self.R_PUSH - self.R_CROUCH) * smoothstep(
                (t - self.T_CROUCH) / (self.T_PUSH - self.T_CROUCH))
        if ph == "air":
            t_land = self.events.get("t_land_est")
            if t_land is not None and t >= t_land - 0.30:
                u = smoothstep((t - (t_land - 0.30)) / 0.30)
                return self.R_AIR + (self.R_PUSH - self.R_AIR) * u
            return self.R_AIR
        if ph == "absorb":
            return self.R_CROUCH
        if ph == "recover":
            t0 = self.events["t_touch"] + self.T_ABSORB
            return self.R_CROUCH + (self.R_STAND - self.R_CROUCH) * smoothstep((t - t0) / self.T_RECOVER)
        return self.R_STAND

    def step(self):
        d = self.data
        r_cmd = self.cmd_r()
        q_tgt = np.zeros(12)
        for li, lg in enumerate(self.cfg.legs):
            r_i = r_cmd
            # 触地段姿态配平: 俯仰/横滚误差 -> 前后/左右腿差动收缩
            # (鼻上仰 -> 前腿多伸把躯干下俯; 左侧抬起 -> 右腿多伸把躯干推平)
            roll, pitch = self.body_rpy()
            if self.phase != "air":
                e_bal = float(np.clip(0.045 * d.qvel[4] + 1.0 * pitch, -0.08, 0.08))
                e_roll = float(np.clip(0.045 * d.qvel[3] + 1.0 * roll, -0.08, 0.08))
                front = lg.startswith("F")
                left = lg.endswith("L")
                r_i += (e_bal if front else -e_bal) + (-e_roll if left else e_roll)
            # 空中段落地找平: 鼻上仰时前腿相对多伸, 争取四脚同时触地
            else:
                e_air = float(np.clip(0.45 * (-pitch), -0.08, 0.08))
                r_i += (e_air if lg.startswith("F") else -e_air)
            r_i = min(max(r_i, 0.10), self.cfg.r_max)
            th1, th2 = self.ik2r(0.0, -r_i)
            q_tgt[li * 3 + 1] = th1
            q_tgt[li * 3 + 2] = th2

        kd_scale = 2.5 if self.phase == "absorb" else 1.0
        self.apply_pd(q_tgt, self.kp_st, self.kd_st * kd_scale)

        # 事件检测
        ff = self.foot_forces()
        self.mark_stance_by_contact(ff)
        grounded = any(fv > 5.0 for fv in ff.values())
        if self.phase == "settle" and self.t >= self.T_SETTLE:
            self.phase = "crouch"
        elif self.phase == "crouch" and self.t >= self.T_CROUCH:
            self.phase = "push"
        elif self.phase == "push" and not grounded:
            self.phase = "air"
            self.events["t_takeoff"] = self.t
            self.events["v_takeoff"] = d.qvel[2]
            z_apex = d.qpos[2] + d.qvel[2] ** 2 / (2 * G)
            z_land = self.cfg.foot_r + self.R_PUSH
            dt_down = np.sqrt(max(2 * (z_apex - z_land), 0.0) / G)
            self.events["t_land_est"] = self.t + d.qvel[2] / G + dt_down
        elif self.phase == "air" and grounded and self.t > self.events.get("t_takeoff", 0) + 0.15:
            self.phase = "absorb"
            self.events["t_touch"] = self.t
        elif self.phase == "absorb" and self.t >= self.events["t_touch"] + self.T_ABSORB:
            self.phase = "recover"
            self.events["t_recover"] = self.t
        elif self.phase == "recover" and self.t >= self.events["t_recover"] + self.T_RECOVER:
            self.phase = "hold"


# ============================================================ 前扑跳跃 v2
@motion("leap", "前扑跳跃", "1.0 m/s起跑->深蹲偏置爆发蹬伸->腾空前扑->前腿先着地缓冲",
        t_end=5.0, win=(1.6, 4.0))
class LeapMotion(CaseSimBase):
    Z_LO = 0.18
    ATT_LIM = 1.0

    GAIT_T, DUTY, STEP_H, K_FB = 0.45, 0.55, 0.08, 0.15
    K_PITCH, K_PITCHD = 0.10, 0.03
    K_ROLL, K_ROLLD = 0.10, 0.03
    T_SETTLE, T_RUN, T_CROUCH, T_PUSH = 0.8, 1.6, 1.85, 2.05
    R_CROUCH = 0.26
    R_PUSH_HIND, R_PUSH_FORE = 0.50, 0.46
    R_AIR_FORE, X_AIR_FORE = 0.44, 0.12
    R_AIR_HIND = 0.32
    R_ABSORB = 0.33
    T_ABSORB, T_RECOVER = 0.45, 0.7
    V_RUN = 1.0

    def __init__(self, cfg, model=None, scene=None):
        super().__init__(cfg, model=model, scene=scene)
        GROUP_A = ("FL", "RR")
        self.PHASE0 = {lg: (0.0 if lg in GROUP_A else 0.5) for lg in cfg.legs}
        self.phase = "run"
        self.events = {}
        self._air_count = 0

    def init_pose(self):
        q0 = np.zeros(self.model.nq)
        self.stand_pose(q0)
        self.data.qpos[:] = q0
        mujoco.mj_forward(self.model, self.data)

    def _grounded3(self, fz_sum):
        if fz_sum < 20.0:
            self._air_count += 1
        else:
            self._air_count = 0
        return self._air_count >= 3

    def step(self):
        d = self.data
        v_x = d.qvel[0]
        roll, pitch = self.body_rpy()
        roll_r, pitch_r = d.qvel[3], d.qvel[4]
        fz_sum = sum(self.foot_forces().values())

        if self.phase == "run" and self.t >= self.T_RUN:
            self.phase = "crouch"
        elif self.phase == "crouch" and self.t >= self.T_CROUCH:
            self.phase = "push"
        elif self.phase == "push" and self.t >= self.T_PUSH and self._grounded3(fz_sum):
            self.phase = "air"
            self.events["t_takeoff"] = self.t
            self.events["vx_takeoff"] = v_x
            self.events["vz_takeoff"] = d.qvel[2]
        elif self.phase == "air" and fz_sum > 30.0:
            self.phase = "absorb"
            self.events["t_touch"] = self.t
        elif self.phase == "absorb" and self.t >= self.events["t_touch"] + self.T_ABSORB:
            self.phase = "recover"
            self.events["t_recover"] = self.t

        self.vcmd_now = self.V_RUN if self.phase == "run" else 0.0
        q_tgt = np.zeros(12)
        kp = np.zeros(12)
        kd = np.zeros(12)
        kd_scale = 1.0

        if self.phase == "run":
            T_st = self.DUTY * self.GAIT_T
            x_a = float(np.clip(0.5 * max(self.vcmd_now, 0.05) * T_st, 0.02, 0.26))
            x_c = float(np.clip(self.K_FB * (v_x - self.vcmd_now), -0.12, 0.12))
            for li, lg in enumerate(self.cfg.legs):
                x_f, z_f, st = self.foot_target(lg, x_a, x_c, self.GAIT_T, self.DUTY,
                                                self.STEP_H, self.cfg.r_stand,
                                                self.PHASE0, self.T_SETTLE)
                depth = -z_f
                if st:
                    dr = (self.K_PITCH * pitch + self.K_PITCHD * pitch_r) * (1.0 if lg.startswith("F") else -1.0)
                    dr += (self.K_ROLL * roll + self.K_ROLLD * roll_r) * (-1.0 if lg[1] == "L" else 1.0)
                    depth += float(np.clip(dr, -0.05, 0.05))
                depth = min(max(depth, 0.12), self.cfg.r_max - 0.02)
                th1, th2 = self.ik2r(x_f, -depth)
                q_tgt[li * 3 + 1] = th1
                q_tgt[li * 3 + 2] = th2
                kp[li * 3:li * 3 + 3] = self.kp_st[li * 3:li * 3 + 3] if st else self.kp_sw[li * 3:li * 3 + 3]
                kd[li * 3:li * 3 + 3] = self.kd_st[li * 3:li * 3 + 3] if st else self.kd_sw[li * 3:li * 3 + 3]
        else:
            for li, lg in enumerate(self.cfg.legs):
                is_front = lg.startswith("F")
                if self.phase == "crouch":
                    # 足端随身体前进而后扫, 避免摩擦刹车
                    u = smoothstep((self.t - self.T_RUN) / (self.T_CROUCH - self.T_RUN))
                    x_t = 0.06 - 0.12 * u
                    r_t = self.cfg.r_stand + (self.R_CROUCH - self.cfg.r_stand) * u
                elif self.phase == "push":
                    u = smoothstep((self.t - self.T_CROUCH) / (self.T_PUSH - self.T_CROUCH))
                    r_tgt = self.R_PUSH_HIND if not is_front else self.R_PUSH_FORE
                    x_t = -0.06 - 0.08 * u
                    r_t = self.R_CROUCH + (r_tgt - self.R_CROUCH) * u
                elif self.phase == "air":
                    if is_front:
                        x_t, r_t = self.X_AIR_FORE, self.R_AIR_FORE
                    else:
                        x_t, r_t = -0.05, self.R_AIR_HIND
                    kp[li * 3:li * 3 + 3] = self.kp_sw[li * 3:li * 3 + 3]
                    kd[li * 3:li * 3 + 3] = self.kd_sw[li * 3:li * 3 + 3]
                elif self.phase == "absorb":
                    x_t, r_t = 0.0, self.R_ABSORB
                    kd_scale = 2.5
                else:
                    u = smoothstep((self.t - self.events["t_recover"]) / self.T_RECOVER)
                    x_t, r_t = 0.0, self.R_ABSORB + (self.cfg.r_stand - self.R_ABSORB) * u

                if self.phase in ("crouch", "push", "absorb", "recover"):
                    kp[li * 3:li * 3 + 3] = self.kp_st[li * 3:li * 3 + 3]
                    kd[li * 3:li * 3 + 3] = self.kd_st[li * 3:li * 3 + 3] * kd_scale

                depth = min(max(r_t, 0.12), self.cfg.r_max - 0.02)
                th1, th2 = self.ik2r(x_t, -depth)
                q_tgt[li * 3 + 1] = th1
                q_tgt[li * 3 + 2] = th2

        self.apply_pd(q_tgt, kp, kd)
        # 支撑标志供视频/统计: 冲击阶段用接触力
        if self.phase != "run":
            self.mark_stance_by_contact(self.foot_forces())


# ============================================================ 跌落缓冲
@motion("drop", "跌落缓冲", "0.30 m 自由跌落, 四腿深蹲吸能缓冲, 极限冲击工况",
        t_end=2.5, win=(0.0, 2.0))
class DropMotion(CaseSimBase):
    Z_LO = 0.15
    ATT_LIM = 0.9

    DROP_H = 0.30
    R_ABSORB, T_ABSORB = 0.30, 0.5
    T_RECOVER = 0.7

    def __init__(self, cfg, model=None, scene=None):
        super().__init__(cfg, model=model, scene=scene)
        self.phase = "fall"
        self.t_touch = None

    def init_pose(self):
        q0 = np.zeros(self.model.nq)
        self.stand_pose(q0)
        q0[2] = self.cfg.foot_r + self.cfg.r_stand + self.DROP_H
        self.data.qpos[:] = q0
        mujoco.mj_forward(self.model, self.data)

    def step(self):
        ff = self.foot_forces()
        grounded = sum(ff.values()) > 20.0
        if self.phase == "fall" and grounded:
            self.phase = "absorb"
            self.t_touch = self.t
        elif self.phase == "absorb" and self.t >= self.t_touch + self.T_ABSORB:
            self.phase = "recover"
            self.t_rec = self.t

        if self.phase == "fall":
            r_cmd = self.cfg.r_stand
        elif self.phase == "absorb":
            r_cmd = self.R_ABSORB
        else:
            r_cmd = self.R_ABSORB + (self.cfg.r_stand - self.R_ABSORB) * smoothstep(
                (self.t - self.t_rec) / self.T_RECOVER)

        q_tgt = np.zeros(12)
        for li, lg in enumerate(self.cfg.legs):
            th1, th2 = self.ik2r(0.0, -r_cmd)
            q_tgt[li * 3 + 1] = th1
            q_tgt[li * 3 + 2] = th2
        kd_scale = 3.0 if self.phase == "absorb" else 1.0
        self.apply_pd(q_tgt, self.kp_st, self.kd_st * kd_scale)
        self.mark_stance_by_contact(ff)


# ============================================================ 载荷/地形工况
# 按设计工况矩阵扩展: 负载 (PAYLOAD) 与坡度 (SLOPE_DEG) 为类属性, 派生即新工况。
# 负载值取自 A2 规格书 (持续行走 25 kg / 站立最大 100 kg), 换机型在子类覆盖。
# 注意: 支撑相关节扭矩来自地面反力而非 segment 自重 (重力前馈在此处无效),
# 重载保持类工况需加大 PD 刚度以顶住 3 倍以上的整机重量。


@motion("walk_full", "满载行走",
        "平地四节拍行走 + 躯干满载 25 kg (A2 持续行走负载), 额定扭矩/热平衡工况",
        t_end=8.0, win=(2.8, 8.0))
class WalkFullMotion(WalkMotion):
    PAYLOAD = 25.0


@motion("slope", "爬坡",
        "20° 等效上坡四节拍行走 (重力矢量倾斜法), 实测上行 0.17 m/s; "
        "时钟步态在 30° 因执行器饱和无法持续上行, 故定 20°",
        t_end=8.0, win=(2.8, 8.0))
class SlopeMotion(WalkMotion):
    SLOPE_DEG = 20.0
    V_MAX = 0.3
    GAIT_T = 0.6         # 提高步频补偿拖拽损耗
    DUTY = 0.75          # 坡道保持原调定占空比 (WalkMotion 已改为 0.80)
    XA_BOOST = 1.15      # 扫腿幅值前馈 (系统性速度缺口 + 坡道拖拽)
    KP_SCALE = 2.0       # 支撑刚度加倍: 抵抗重力拖拽造成的伺服屈服
    K_FB = 0.25
    XC_LIM = 0.18
    R_ST = 0.38          # 蹲低站姿降低质心

    def __init__(self, cfg, model=None, scene=None):
        super().__init__(cfg, model=model, scene=scene)
        self.kp_st = self.kp_st * self.KP_SCALE
        self.kd_st = self.kd_st * np.sqrt(self.KP_SCALE)
        # 上坡姿态前馈: 足端支撑中心沿坡下移 r·tanθ, 等效重力线落在支撑多边形内
        # (机身相对足端前倾"迎坡", 与真机爬坡姿态一致)
        self.x_grav = -self.r_st * float(np.tan(np.radians(self.SLOPE_DEG)))


@motion("slope_full", "满载爬坡",
        "15° 上坡 + 满载 25 kg, 实测上行 0.42 m/s: 额定扭矩与热平衡的决定性持续工况 "
        "(更大坡度+满载超出时钟步态扭矩预算, 降至 15°)",
        t_end=8.0, win=(2.8, 8.0))
class SlopeFullMotion(SlopeMotion):
    PAYLOAD = 25.0
    SLOPE_DEG = 15.0
    V_MAX = 0.6


@motion("stand_load", "静站载重",
        "站立保持 + 躯干静载 100 kg (A2 站立最大负载), 结构静强度工况",
        t_end=3.0, win=(1.0, 3.0))
class StandLoadMotion(CaseSimBase):
    PAYLOAD = 100.0

    def __init__(self, cfg, model=None, scene=None):
        super().__init__(cfg, model=model, scene=scene)
        # 140 kg 整机重需高刚度保持 (常规行走增益会被自重压沉)
        self.kp_st = np.array([[400.0, 500.0, 800.0][i % 3] for i in range(12)])
        self.kd_st = np.array([[10.0, 15.0, 24.0][i % 3] for i in range(12)])

    def init_pose(self):
        q0 = np.zeros(self.model.nq)
        self.stand_pose(q0)
        self.data.qpos[:] = q0
        mujoco.mj_forward(self.model, self.data)

    def step(self):
        q_tgt = np.zeros(12)
        for li, lg in enumerate(self.cfg.legs):
            th1, th2 = self.ik2r(0.0, -self.cfg.r_stand)
            q_tgt[li * 3 + 1] = th1
            q_tgt[li * 3 + 2] = th2
        self.apply_pd(q_tgt, self.kp_st, self.kd_st)


# ============================================================ 异常/失效工况
# 可靠性设计与安全功能的输入 (设计工况统计第④类): 抗扰裕度 / 结构抗摔 / 驱动器保护。


@motion("push", "侧向推搡",
        "对角小跑中躯干侧面施加 50 N×0.25 s 冲击 (参考 58 kg 平台 Trot 抗扰试验), "
        "校核关节峰值扭矩储备与稳定裕度",
        t_end=8.0, win=(2.8, 8.0))
class PushMotion(TrotMotion):
    PUSH_T0, PUSH_T1 = 4.0, 4.25           # 冲击施加窗口 s
    PUSH_F = (0.0, 50.0, 0.0)              # 世界系侧向力 N

    def apply_pd(self, q_tgt, kp, kd):
        # xfrc_applied 跨步保持, 须每步重写 (施加/清零)
        self.data.xfrc_applied[self.trunk_bid] = 0.0
        if self.PUSH_T0 <= self.t <= self.PUSH_T1:
            self.data.xfrc_applied[self.trunk_bid][:3] = self.PUSH_F
        super().apply_pd(q_tgt, kp, kd)


@motion("fall", "侧跌倒落",
        "0.30 m 高度带 25° 初始侧倾自由跌落, 站立 PD 保持 (实机跌落时控制器仍在跑), "
        "非受控结构抗摔冲击工况",
        t_end=2.5, win=(0.0, 2.0))
class FallMotion(CaseSimBase):
    Z_LO = 0.10
    ATT_LIM = 1.4                          # 允许大角度翻滚, 充分记录冲击过程
    DROP_H = 0.30
    ROLL0 = 25.0                           # 初始侧倾角 deg (绕 x 轴)

    def init_pose(self):
        q0 = np.zeros(self.model.nq)
        self.stand_pose(q0)
        q0[2] = self.cfg.foot_r + self.cfg.r_stand + self.DROP_H
        th = np.radians(self.ROLL0)
        q0[3:7] = [np.cos(th / 2), np.sin(th / 2), 0.0, 0.0]
        self.data.qpos[:] = q0
        mujoco.mj_forward(self.model, self.data)

    def step(self):
        q_tgt = np.zeros(12)
        for li, lg in enumerate(self.cfg.legs):
            th1, th2 = self.ik2r(0.0, -self.cfg.r_stand)
            q_tgt[li * 3 + 1] = th1
            q_tgt[li * 3 + 2] = th2
        self.apply_pd(q_tgt, self.kp_st, self.kd_st)
        self.mark_stance_by_contact(self.foot_forces())


@motion("stall", "关节堵转",
        "站立中 FL 膝关节机械卡滞 (equality 焊接, 反力由约束求解承担) + 缓慢下蹲施压, "
        "执行器持续饱和: 堵转过流/驱动器 I2t 热保护工况",
        t_end=4.0, win=(1.0, 4.0))
class StallMotion(CaseSimBase):
    T_JAM = 0.8                            # 卡滞发生时刻 s
    JAM_LEG, JAM_PART = "FL", "calf"       # 卡滞关节 (踩缝卡膝场景)
    R_SQUAT = 0.30                         # 卡滞后下蹲目标腿长 m
    T_SQUAT = 2.0                          # 下蹲到位时刻 s
    # 定制场景: 焊接 FL_calf 到常量 (初始 inactive, T_JAM 时激活并锁到当前角)
    SCENE_EXTRA = ('<equality>\n'
                   '  <joint name="jam_FL_calf" joint1="FL_calf_joint" active="false"/>\n'
                   '</equality>')

    def __init__(self, cfg, model=None, scene=None):
        super().__init__(cfg, model=model, scene=scene)
        self.jammed = False
        self.jam_joint = cfg.joint_fmt.format(leg=self.JAM_LEG, part=self.JAM_PART)
        self.jam_eq = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "jam_FL_calf")
        # 堵转需高刚度施压至执行器饱和 (与跳跃版一致)
        self.kp_st = np.array([[120.0, 260.0, 600.0][i % 3] for i in range(12)])
        self.kd_st = np.array([[3.0, 6.0, 10.0][i % 3] for i in range(12)])

    def init_pose(self):
        q0 = np.zeros(self.model.nq)
        self.stand_pose(q0)
        self.data.qpos[:] = q0
        mujoco.mj_forward(self.model, self.data)

    def step(self):
        d = self.data
        if not self.jammed and self.t >= self.T_JAM:
            # 机械卡滞: 焊接约束激活, 锁定常量 = 当前关节角
            self.model.eq_data[self.jam_eq, 0] = d.qpos[self.jq[self.jam_joint]]
            d.eq_active[self.jam_eq] = 1
            self.jammed = True
        if self.t < self.T_JAM:
            r = self.cfg.r_stand
        else:
            u = smoothstep((self.t - self.T_JAM) / (self.T_SQUAT - self.T_JAM))
            r = self.cfg.r_stand + (self.R_SQUAT - self.cfg.r_stand) * u
        q_tgt = np.zeros(12)
        for li, lg in enumerate(self.cfg.legs):
            th1, th2 = self.ik2r(0.0, -r)
            q_tgt[li * 3 + 1] = th1
            q_tgt[li * 3 + 2] = th2
        self.apply_pd(q_tgt, self.kp_st, self.kd_st)
