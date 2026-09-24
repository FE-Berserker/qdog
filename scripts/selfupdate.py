#!/usr/bin/env python3
"""qdog 技能自更新 —— 平台无关。

更新逻辑自带在技能内部: 任何 agent 平台 (ZCode / Claude Code / 其它读 SKILL.md 的
客户端) 或人类都能调它, 不依赖某个客户端的更新机制。OS 层的定时任务 (Windows 任务
计划 / cron) 也调同一个脚本, 于是"agent 驱动"与"系统驱动"两条路共用一套逻辑。

设计护栏 (自更新绝不能打断正常任务):
  - *节流*: 距上次检查不足 QDOG_UPDATE_INTERVAL (默认 24 h) 直接返回, 不联网;
  - *轻量检查*: 先用 git ls-remote 比对远端与本地 HEAD, 相同就不 fetch;
  - *工作区有未提交改动*: 只报告、不拉取 (保护用户本地调参);
  - *非 git 仓库 / 无网络 / 远端不可达*: 只报告, 不动文件;
  - *只快进*: 用 git pull --ff-only, 绝不产生合并提交或冲突;
  - *退出码恒为 0*: 调用方 (agent 或计划任务) 不会因为更新失败而中断。

用法:
  python scripts/selfupdate.py           # 节流自更新 (agent 会话开始时调这个)
  python scripts/selfupdate.py --check   # 只查有没有更新, 不拉取、不写节流文件
  python scripts/selfupdate.py --force   # 忽略节流立即检查
  python scripts/selfupdate.py --quiet   # 无输出 (供计划任务调用)

环境变量:
  QDOG_UPDATE_INTERVAL  节流间隔秒数, 默认 86400; 设 0 关闭节流
  QDOG_UPDATE_OFF       设 1 完全禁用 (离线环境)
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / ".git" / "qdog-selfupdate.json"   # 放 .git 内, 永不被跟踪


def git(*args, timeout=30):
    return subprocess.run(["git", "-C", str(ROOT), *args],
                          capture_output=True, text=True, timeout=timeout)


def load_state():
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(**kw):
    if "--check" in sys.argv:      # --check 声明为无副作用
        return
    try:
        st = load_state()
        st.update(kw)
        st["last_check"] = time.time()
        STATE.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass          # 状态写不进去不影响本次更新


def changed_files(before, after):
    r = git("diff", "--name-only", before, after)
    return [ln for ln in r.stdout.splitlines() if ln.strip()]


def main(argv):
    quiet = "--quiet" in argv
    out = (lambda m: None) if quiet else (lambda m: print(f"[qdog 自更新] {m}"))

    if os.environ.get("QDOG_UPDATE_OFF") == "1":
        out("已禁用 (QDOG_UPDATE_OFF=1)")
        return 0
    if not (ROOT / ".git").exists():
        out(f"{ROOT} 不是 git 仓库, 跳过")
        return 0

    interval = int(os.environ.get("QDOG_UPDATE_INTERVAL", "86400"))
    check_only = "--check" in argv
    force = "--force" in argv
    st = load_state()
    if not force and not check_only and interval > 0:
        need = float(st.get("retry_hint") or interval)   # 上次失败时会给一个更短的重试间隔
        idle = time.time() - float(st.get("last_check", 0))
        if idle < need:
            out(f"节流中 (距上次检查 {idle / 3600:.1f} h, 该等 {need / 3600:.1f} h)")
            return 0

    br = git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if not br or br == "HEAD":
        out("处于游离 HEAD, 跳过")
        return 0
    local = git("rev-parse", "HEAD").stdout.strip()

    try:
        r = git("ls-remote", "origin", br, timeout=25)
    except subprocess.TimeoutExpired:
        out("远端无响应 (超时), 保持本地版本")
        save_state(result="remote-timeout", head=local)
        return 0
    if r.returncode != 0:
        why = (r.stderr or r.stdout or "").strip().splitlines()
        out(f"远端不可达 (rc={r.returncode}: {why[0][:120] if why else '无输出'}), 保持本地版本")
        save_state(result="remote-unreachable", head=local, retry_hint=3600)
        return 0
    remote = r.stdout.split()[0] if r.stdout.strip() else ""

    if not remote:
        out(f"远端无 {br} 分支, 保持本地版本")
        save_state(result="no-remote-branch", head=local)
        return 0
    if remote == local:
        out(f"已是最新 ({local[:8]})")
        save_state(result="up-to-date", head=local)
        return 0

    # 远端与本地不同: 可能是本地领先(自己推过), 也可能是落后, 需要 fetch 才能分辨
    if check_only:
        out(f"远端 {remote[:8]} 与本地 {local[:8]} 不同 (--check 不拉取, 未判定领先/落后)")
        return 0
    if git("fetch", "origin", br, timeout=120).returncode != 0:
        out("fetch 失败, 保持本地版本")
        save_state(result="fetch-failed", head=local)
        return 0

    cnt = git("rev-list", "--count", f"HEAD..origin/{br}").stdout.strip()
    behind = int(cnt) if cnt.isdigit() else 0
    if behind == 0:
        out(f"本地已领先或与远端分叉, 无新提交可拉 (本地 {local[:8]})")
        save_state(result="ahead-or-diverged", head=local)
        return 0

    dirty = git("status", "--porcelain").stdout.strip()
    if dirty:
        n = len(dirty.splitlines())
        out(f"远端有 {behind} 个新提交, 但本地有 {n} 处未提交改动 → 跳过拉取 (保护你的本地修改)")
        save_state(result="skipped-dirty", head=local, behind=behind)
        return 0

    r = git("pull", "--ff-only", "origin", br, timeout=180)
    if r.returncode != 0:
        out("pull 失败 (可能需要手工处理), 保持本地版本")
        save_state(result="pull-failed", head=local, behind=behind)
        return 0

    new = git("rev-parse", "HEAD").stdout.strip()
    files = changed_files(local, new)
    out(f"已更新 {local[:8]} → {new[:8]} ({behind} 个提交, {len(files)} 个文件)")
    for f in files[:8]:
        out(f"  · {f}")
    if len(files) > 8:
        out(f"  · … 另有 {len(files) - 8} 个文件")
    if any(f.endswith(".py") for f in files):
        out("提示: 代码已变, 之前 results/ 里由旧代码跑出的结果不再与之严格对应")
    save_state(result="updated", head=new, before=local, behind=behind, files=files[:50])
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Exception as e:                      # 自更新永不打断调用方
        if "--quiet" not in sys.argv:
            print(f"[qdog 自更新] 内部错误, 已忽略: {e}")
        sys.exit(0)
