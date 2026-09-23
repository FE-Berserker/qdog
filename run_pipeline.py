"""四足机器狗仿真流水线入口。

一个命令跑完某机型的全部动作: 仿真 -> 关节扭矩表/图 -> 弯矩提取 -> 视频 -> 载荷包络。

用法:
  python run_pipeline.py                     # A2 全 5 动作全流程
  python run_pipeline.py --robot a2 --motions walk trot
  python run_pipeline.py --skip-video        # 跳过视频渲染 (节省时间)
  python run_pipeline.py --list              # 列出已注册机型与动作

逐动作确认在对话层完成 (不在终端设问): 由 agent 逐动作调用本脚本
(--motions <key>), 完成后在对话框里给出指标摘要与视频, 用户直接以
自由文本提调整意见, agent 改参重跑; 确认后再跑下一个动作。

输出布局 (results/<robot>/):
  scene.xml                       场景 (绝对路径引用模型, 位置无关)
  <motion>/
    trace.csv / trace.npy         全量时序 (2 ms): 12关节 q/v/扭矩 + 重力/被动/约束/惯性分解
                                  + 四足反力 + 支撑标志 + 速度指令
    关节受力.xlsx                 扭矩时序 + 统计 + 工况指标
    torques.png / motion.png      扭矩曲线 / 运动学+地面反力曲线
    wrench.csv                    12关节 6 分量相互作用 wrench (2 ms)
    关节弯矩.xlsx                 弯矩时序 + 载荷统计 (驱动/弯矩/轴向/剪切)
    bending.png                   弯矩与绕轴驱动对比曲线
    video.avi / preview.gif       双机位跟踪视频 + GIF 预览 (有 ffmpeg 时另有 video.mp4)
    video_first/mid/last.png      抽查帧
  关节载荷包络.xlsx / .png        跨工况包络: 每关节各指标最大值+来源工况 + 工况×关节矩阵

新增动作: 在 quad_pipeline/motions.py 用 @motion 注册一个 CaseSimBase 子类。
新增机型: 在 quad_pipeline/config.py 注册 RobotConfig。
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from quad_pipeline.config import ROBOTS
from quad_pipeline.core import build_scene, run_sim, save_csv
from quad_pipeline.motions import MOTIONS
from quad_pipeline import report, wrench, video, envelope


def process_motion(key, cfg, scene, out_root, do_video=True):
    """单动作全流程: 仿真 -> 报表 -> 弯矩 -> 视频。"""
    cls = MOTIONS[key]
    mdir = out_root / key
    mdir.mkdir(parents=True, exist_ok=True)
    title = f"{cfg.name} {cls.CN}"

    print(f"\n===== [{key}] {cls.CN} 仿真 =====")
    t0 = time.time()
    sim = cls(cfg, scene=scene)
    A = run_sim(sim, cls.T_END)
    np.save(mdir / "trace.npy", A)
    save_csv(mdir / "trace.csv", A, cfg)
    stats = report.summarize(A, cfg, cls.WIN)
    report.make_plots(A, cfg, title, mdir / "torques.png", mdir / "motion.png", cls.WIN)
    report.write_torque_xlsx(A, cfg, stats, title, cls.DESC, cls.WIN, mdir / "关节受力.xlsx")
    print(f"仿真+报表完成 ({time.time() - t0:.0f}s): 窗口{cls.WIN}, "
          f"v={stats['v_mean']:.2f} m/s, GRF峰值={stats['grf_peak']:.0f} N, "
          f"力矩峰值={stats['peak_tq'].max():.1f} N·m, 跌倒={sim.fallen}")

    print(f"[{key}] 弯矩提取 (传感器模型重跑)...")
    t0 = time.time()
    Aw = wrench.run_wrench(cls, cfg, scene)
    dec = wrench.decompose(Aw, cfg)
    wrench.save_wrench_csv(mdir / "wrench.csv", Aw, cfg)
    wrench.make_bending_png(Aw, dec, cfg, title, mdir / "bending.png")
    wrench.write_bending_xlsx(Aw, dec, cfg, title, mdir / "关节弯矩.xlsx")
    print(f"弯矩完成 ({time.time() - t0:.0f}s)")

    if do_video:
        print(f"[{key}] 视频渲染...")
        t0 = time.time()
        avi, gif, _ = video.render_video(cfg, mdir / "trace.npy", scene, cls.CN,
                                         mdir / "video.avi")
        print(f"视频完成 ({time.time() - t0:.0f}s): {avi.name} ({avi.stat().st_size // 1024} KB)")
    else:
        avi = mdir / "video.avi"
    return stats, sim.fallen, avi


def main():
    ap = argparse.ArgumentParser(description="四足机器狗动作仿真与关节载荷流水线")
    ap.add_argument("--robot", default="a2", help="已注册机型 (config.py)")
    ap.add_argument("--motions", nargs="+", default=None,
                    help="动作子集 (默认全部: walk trot jump leap drop)")
    ap.add_argument("--skip-video", action="store_true", help="跳过视频渲染")
    ap.add_argument("--out", default=None, help="输出根目录 (默认 results/<robot>/)")
    ap.add_argument("--list", action="store_true", help="列出机型与动作后退出")
    args = ap.parse_args()

    if args.list:
        print("机型:", list(ROBOTS))
        print("动作:", [(k, v.CN) for k, v in MOTIONS.items()])
        return

    if args.robot not in ROBOTS:
        sys.exit(f"未知机型 {args.robot!r}, 已注册: {list(ROBOTS)}")
    cfg = ROBOTS[args.robot]

    out_root = Path(args.out) if args.out else Path("results") / cfg.name
    out_root.mkdir(parents=True, exist_ok=True)
    scene = build_scene(cfg, out_root)

    keys = args.motions or list(MOTIONS)
    for k in keys:
        if k not in MOTIONS:
            sys.exit(f"未知动作 {k!r}, 已注册: {list(MOTIONS)}")
    print(f"机型 {cfg.name} ({cfg.model_xml}), 动作: {keys}, 输出: {out_root}")

    done = []
    for k in keys:
        stats, fallen, avi = process_motion(k, cfg, scene, out_root,
                                            do_video=not args.skip_video)
        done.append(k)
        # 动作摘要行 (供调用方/agent 汇报): 一行关键指标, 跌倒显著标注
        flag = "  !!跌倒截断!!" if fallen else ""
        print(f"[{k}] 摘要: v={stats['v_mean']:.2f} m/s, 姿态pp=({stats['roll_pp']:.1f}°, "
              f"{stats['pitch_pp']:.1f}°), GRF峰值={stats['grf_peak']:.0f} N, "
              f"力矩峰值={stats['peak_tq'].max():.1f} N·m, 视频={avi}{flag}")

    # ---- 包络 (汇总本次确认完成的动作) ----
    if not done:
        print("\n无确认完成的动作, 跳过包络汇总。")
        return
    print("\n===== 载荷包络 =====")
    csvs, names, descs = {}, {}, {}
    for k in done:
        p = out_root / k / "wrench.csv"
        if p.exists():
            csvs[k] = p
            names[k] = MOTIONS[k].CN
            descs[k] = MOTIONS[k].DESC
    if csvs:
        # envelope 期望 list 顺序对齐; 转 list
        ks = list(csvs)
        ox, op, env, src = envelope.build_envelope(
            cfg,
            {k: csvs[k] for k in ks},
            [names[k] for k in ks],
            [descs[k] for k in ks],
            out_root)
        print("saved:", ox)
        print("saved:", op)
        # 终端打印包络表
        from quad_pipeline.report import joint_meta
        meta = joint_meta(cfg)
        print(f"\n{'joint':<16}{'驱动max':>9}{'弯矩max':>9}{'剪力max':>9}{'轴向max':>9}  弯矩来源")
        for i, (jn, lg, p, _) in enumerate(meta):
            print(f"{jn:<16}{env['drive'][i]:>9.1f}{env['bend'][i]:>9.1f}"
                  f"{env['shear'][i]:>9.1f}{env['axial'][i]:>9.1f}  {src['bend'][i]}")
    print("\n全部完成。")


if __name__ == "__main__":
    main()
