"""动作自检: 从 trace 计算步态对称性 / 打滑 / 拖地 / 腾空 / 峰值裕度。

用法:
  python check_motion.py walk              # 单个动作
  python check_motion.py walk trot drop    # 多个动作

读 results/<robot>/<key>/trace.npy, 用场景模型做足端正运动学, 得到可复核的自检结论:
  - 步态: 各腿占空比、步态周期 (确认节拍正确)
  - 对称性: 左右/前后占空比差、同名关节峰值力矩差异
  - 打滑: 触地相内足端水平漂移 (相对步长)
  - 拖地: 时钟摆动相内仍有接触力的样本比例
  - 腾空: 总反力 < 20 N 的时间占比 (四节拍行走应≈0)
  - 裕度: 力矩峰值/额定, 执行器饱和样本数

统计窗口取动作自身的 MOTIONS[key].WIN (排除站立静置/速度爬坡段);
打滑用"触地"(反力>20N) 而非时钟支撑相判定, 避免把空中摆腿算成打滑。
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np

from quad_pipeline.config import ROBOTS
from quad_pipeline.core import C_TQ, C_FZ, C_ST, C_BX, C_BZ
from quad_pipeline.motions import MOTIONS
from quad_pipeline.report import joint_meta, LEG_CN, PART_CN

AIR_THR = 20.0        # 总反力低于此值判为腾空 N
FOOT_THR = 20.0       # 单足反力高于此值判为触地 N
DRAG_THR = 20.0       # 摆动相接触力高于此值判为拖地 N
MIN_RUN = 5           # 最短有效段样本数


def foot_geom_ids(model, cfg):
    out = {}
    for lg in cfg.legs:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                cfg.foot_body_fmt.format(leg=lg))
        out[lg] = [g for g in range(model.ngeom)
                   if model.geom_bodyid[g] == bid
                   and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
    return out


def foot_paths(model, cfg, A):
    """足端球心世界坐标 (前向运动学重放)。"""
    ids = foot_geom_ids(model, cfg)
    data = mujoco.MjData(model)
    out = {lg: np.zeros((len(A), 3)) for lg in cfg.legs}
    for k in range(len(A)):
        data.qpos[:] = A[k, 1:20]
        mujoco.mj_forward(model, data)
        for lg, gids in ids.items():
            out[lg][k] = data.geom_xpos[gids[0]]
    return out


def runs_of(mask):
    """连续 True 区间的 (起, 止) 列表 (含端点)。"""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            if j - i + 1 >= MIN_RUN:
                out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def check(key, cfg, root):
    d = root / key
    A = np.load(d / "trace.npy")
    cls = MOTIONS[key]
    win = getattr(cls, "WIN", (0.0, A[-1, 0]))
    t_all = A[:, 0]
    sl = slice(int(win[0] / (t_all[1] - t_all[0])), None) if len(t_all) > 1 else slice(None)

    t = t_all[sl]
    A = A[sl]
    st = A[:, C_ST:C_ST + 4] > 0.5                       # 时钟支撑相
    fz = A[:, C_FZ:C_FZ + 4]
    ct = fz > FOOT_THR                                    # 实际触地
    tq = A[:, C_TQ:C_TQ + 12]
    fz_tot = fz.sum(axis=1)

    scene = root / f"scene_{key}.xml"
    if not scene.exists():
        scene = root / "scene.xml"
    model = mujoco.MjModel.from_xml_path(str(scene))
    mass = cfg.mass if cfg.mass > 0 else float(model.body_mass.sum())
    fps = foot_paths(model, cfg, A)

    print(f"\n===== [{key}] {cls.CN} 自检 (窗口 {win[0]:.1f}~{min(win[1], t[-1]):.1f}s) =====")
    print(f"末位置 x={A[-1, C_BX]:.2f} m, 最低躯干 {A[:, C_BZ].min():.3f} m, "
          f"平均速度 {A[:, 101].mean():.2f} m/s")

    # ---- 步态 ----
    print("\n[步态] 时钟占空比 / 触地占空比 / 步态周期")
    duty = {}
    for i, lg in enumerate(cfg.legs):
        cd = float(ct[:, i].mean())
        rr = runs_of(ct[:, i])
        onsets = [t[i0] for i0, _ in rr if t[i0] > win[0] + 0.3]
        per = np.diff(onsets) if len(onsets) > 1 else np.array([])
        duty[lg] = cd
        line = (f"  {lg}({LEG_CN[lg]}): 时钟 {st[:, i].mean():.3f} / 触地 {cd:.3f}, "
                f"触地段 {len(rr)} 次")
        if len(per):
            line += f", 步态周期 {np.median(per):.3f}s"
        print(line)
    n_sup = ct.sum(axis=1)
    print(f"  同时触地腿数: 最小 {int(n_sup.min())}, 均值 {n_sup.mean():.2f}, "
          f"≤2 腿样本 {float((n_sup <= 2).mean()) * 100:.1f}%, "
          f"≥3 腿样本 {float((n_sup >= 3).mean()) * 100:.1f}%")

    # ---- 对称性 ----
    dl = np.mean([duty["FL"], duty["RL"]])
    dr = np.mean([duty["FR"], duty["RR"]])
    df = np.mean([duty["FL"], duty["FR"]])
    db = np.mean([duty["RL"], duty["RR"]])
    print(f"\n[对称性] 触地占空比 左{dl:.3f}/右{dr:.3f} (差 {abs(dl - dr):.3f}), "
          f"前{df:.3f}/后{db:.3f} (差 {abs(df - db):.3f})")
    meta = joint_meta(cfg)
    pk = np.abs(tq).max(axis=0)
    idx = {(m[1], m[2]): i for i, m in enumerate(meta)}
    print("  同名关节峰值力矩 (左右对比):")
    for p in cfg.parts:
        for a, b, tag in ((("FL", p), ("FR", p), "FL/FR"), (("RL", p), ("RR", p), "RL/RR")):
            ia, ib = idx[a], idx[b]
            rel = abs(pk[ia] - pk[ib]) / max(pk[ia], pk[ib], 1e-9) * 100
            print(f"    {PART_CN[p]:<16}{tag}: {pk[ia]:6.1f} / {pk[ib]:6.1f} N·m  ({rel:4.1f}%)")

    # ---- 打滑 / 拖地 ----
    v_body = abs(float(A[:, 101].mean()))
    print("\n[接触质量] 触地相足端水平漂移(打滑) / 时钟摆动相接触(拖地)")
    for i, lg in enumerate(cfg.legs):
        rr = runs_of(ct[:, i])
        drifts, slips = [], []
        for i0, i1 in rr:
            seg = fps[lg][i0:i1 + 1]
            dmax = float(np.linalg.norm(seg[:, :2] - seg[0, :2], axis=1).max())
            dt = float(t[i1] - t[i0])
            drifts.append(dmax)
            if dt > 1e-6:
                slips.append(dmax / dt)
        sw = ~st[:, i]
        drag = float((fz[sw, i] > DRAG_THR).mean() * 100) if sw.any() else 0.0
        if drifts:
            # 原地动作 (跳/落/静站) 体速≈0, 打滑占体速比无意义, 只给绝对漂移
            pct = ""
            if slips and v_body > 0.1:
                pct = f", 打滑 {np.mean(slips) / v_body * 100:.0f}% 体速"
            print(f"  {lg}({LEG_CN[lg]}): 触地漂移 均值 {np.mean(drifts) * 1000:.1f} mm, "
                  f"最大 {np.max(drifts) * 1000:.1f} mm{pct}; 摆动相仍接触 {drag:.1f}%")

    air = float((fz_tot < AIR_THR).mean() * 100)
    print(f"\n[腾空] 总反力<{AIR_THR:.0f}N 时间占比 {air:.1f}%; 总反力峰值 {fz_tot.max():.0f} N "
          f"({fz_tot.max() / (mass * 9.81):.1f}×体重), 均值 {fz_tot.mean():.0f} N "
          f"({fz_tot.mean() / (mass * 9.81):.2f}×体重)")

    # ---- 航向 (直线动作的偏航漂移) ----
    qw, qx, qy, qz = A[:, 4], A[:, 5], A[:, 6], A[:, 7]
    yaw = np.degrees(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
    dyaw = float(yaw[-1] - yaw[0])
    dy = float(A[-1, 2] - A[0, 2])
    dur = float(t[-1] - t[0])
    print(f"\n[航向] 窗口内 yaw 漂移 {dyaw:+.2f}° "
          f"({dyaw / dur:+.2f} °/s), 横向漂移 y {dy:+.3f} m "
          f"(前进 x {A[-1, C_BX] - A[0, C_BX]:.2f} m)")

    # ---- 裕度 ----
    rated = np.array([m[3] for m in meta])
    ratio = pk / rated
    sat = float((np.abs(tq) >= 0.999 * rated).mean() * 100)
    print(f"\n[裕度] 峰值/额定: 最大 {ratio.max() * 100:.0f}% "
          f"({meta[int(np.argmax(ratio))][0]}), 均值 {ratio.mean() * 100:.0f}%; "
          f"执行器饱和样本 {sat:.2f}%")
    over = [meta[i][0] for i in range(12) if ratio[i] > 1.0]
    print("  超额定关节:", ", ".join(over) if over else "无")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="+", help="动作 key")
    ap.add_argument("--robot", default="a2")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = ROBOTS[args.robot]
    root = Path(args.out) if args.out else Path("results") / cfg.name
    for k in args.keys:
        check(k, cfg, root)


if __name__ == "__main__":
    main()
