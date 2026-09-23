"""quad_pipeline 动作库: 行走 / 奔跑 / 跳跃 / 前扑 / 跌落。

每个动作 = 一个 CaseSimBase 子类 (init_pose + step) + 元数据注册 (MOTIONS)。
参数均为 A2 实机调定值; 换机型时在子类 PARAMS 基础上派生覆盖即可。

各动作经实机仿真验证 (A2, 2026-09):
  walk  四节拍行走, 实测 0.39 m/s, 姿态峰峰值 2.3°/1.8°
  trot  对角小跑 v0.3, 实测 1.33 m/s (指令 1.5), 姿态峰峰值 8.0°/3.8°
  jump  原地跳跃, 起跳 vz≈2.0 m/s, 落地缓冲
  leap  前扑 v2, 起跳 vx=0.71/vz=1.58, 腾空 0.34 s 前进 0.24 m, 前腿落地缓冲
  drop  0.30 m 自由跌落, 触地 GRF 峰值 ≈ 4.7 kN (11.9 倍体重)
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
@motion("walk", "行走", "四节拍爬行 walk: FR->RL->FL->RR 依次迈步, 占空比0.75, 任意时刻>=3腿支撑",
        t_end=8.0, win=(2.8, 8.0))
class WalkMotion(CaseSimBase):
    GAIT_T, DUTY, STEP_H = 0.80, 0.75, 0.06
    V_MAX, K_FB = 0.6, 0.15
    T_SETTLE, T_RAMP = 0.8, 1.2
    K_PITCH, K_PITCHD = 0.10, 0.03
    K_ROLL, K_ROLLD = 0.10, 0.03

    def __init__(self, cfg, model=None, scene=None):
        super().__init__(cfg, model=model, scene=scene)
        self.PHASE0 = {"FR": 0.0, "RL": 0.25, "FL": 0.5, "RR": 0.75}

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
        self.stand_pose(q0)
        self.data.qpos[:] = q0
        mujoco.mj_forward(self.model, self.data)

    def step(self):
        d = self.data
        v_x, vcmd = d.qvel[0], self.v_cmd()
        roll, pitch = self.body_rpy()
        roll_r, pitch_r = d.qvel[3], d.qvel[4]
        T_st = self.DUTY * self.GAIT_T
        x_a = float(np.clip(0.5 * max(vcmd, 0.05) * T_st, 0.02, 0.26))
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


# ============================================================ 奔跑 (对角小跑 v0.3)
@motion("trot", "奔跑", "对角小跑 trot v0.3: 占空比0.55+扫腿幅值前馈补偿25%, 指令1.5 m/s",
        t_end=8.0, win=(2.8, 8.0))
class TrotMotion(WalkMotion):
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
