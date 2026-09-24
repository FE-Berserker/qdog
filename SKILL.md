---
name: qdog
description: 用 MuJoCo 对四足机器狗做动作仿真与关节载荷分析 —— 跑动作、提取每个关节的扭矩/弯矩/轴向力/剪切力、出动作视频与 Excel 表格、汇总跨工况载荷包络。当用户提到机器狗仿真、四足机器人仿真、关节载荷/关节弯矩分析、载荷包络、动作视频, 或要跑 walk/trot/jump/leap/drop 等动作、A2 机型实仿, 或说 "quadruped MuJoCo simulation"、"joint load analysis" 时使用。本技能强制逐动作人工确认: 一次只跑一个动作, 跑完必须交付 check (指标摘要 + 视频 + 自检结论) 并等用户确认, 禁止批跑。
---

# qdog —— 四足机器狗动作仿真与关节载荷分析

对一台四足机器狗跑 MuJoCo 动作仿真 (行走/奔跑/跳跃/前扑/跌落 + 载荷地形 + 异常失效工况),
提取每个关节的扭矩、弯矩、轴向力、剪切力, 产出动作视频、Excel 表格与跨工况载荷包络,
用于关节/轴承/结构件的设计选型校核。

## 铁律: 逐动作确认, 不是批处理

**一次只跑一个动作; 跑完必须把 check 交给用户, 等确认后再跑下一个。**

1. **一次一个 key**: `python run_pipeline.py --motions <key>`
2. **完成即交付 check**, 三件缺一不可:
   - **指标摘要**: 速度 / 姿态波动 / GRF 峰值 / 力矩峰值 / 是否跌倒
   - **视频**: `results/<robot>/<key>/video.avi` 与 `video_preview.gif` 的链接 (点开即看),
     必要时给 `video_first/mid/last.png` 抽查帧
   - **自检结论**: 步态对称性、有无打滑/拖地、腾空相位、峰值是否超额定
3. **停在这里等回复**。用户说 "确认 / 可以, 下一个" 才继续; 用户提调整意见 (自由文本,
   如 "前扑再远一点") → 改 `quad_pipeline/motions.py` 参数 → 重跑 **同一个** 动作 →
   重新交付 check
4. 用户说 "全部跑" 时: 仍按 1~3 **逐个执行、逐个交付 check**, 只是不再等待回复;
   仅当用户明确说 "直接跑完, 不用给我看" 才可合并交付
5. 全部动作确认后, 才汇总载荷包络: `python run_pipeline.py --envelope-only`

**禁止**:

- 未交付 check 就连跑下一个动作
- 用不带 `--motions` 的 `python run_pipeline.py` 一次跑全部动作
- 在动作未确认时引用其数据下结论

不带 `--motions` 的 `run_pipeline.py` 是 **全量批跑** (12 个动作约 20 分钟、约 600 MB),
仅当用户明确要求 "一次性跑完" 时才用。理由: 单动作含视频约 1.5 分钟, 第一个动作的步态
不对, 后面全部返工 —— 越早发现越省。

## 首次使用检查

0. `python scripts/selfupdate.py` —— **自更新** (任何平台、任何客户端都应先跑这一步;
   见下节)。节流默认 24 h, 因此重复调用几乎无开销
1. `pip install -r requirements.txt` (mujoco / numpy / matplotlib / openpyxl / pillow)
2. 准备机器人模型: A2 需 `unitree_robots/a2/a2.xml` (本仓库不含, 从 Unitree 官方获取)
3. `python verify_sensor.py` → 期望输出 `PASS` (弯矩测量方法的双摆数值验证)
4. `python run_pipeline.py --list` → 确认机型与动作注册
5. 之后按上面的铁律逐动作驱动

## 自更新 (平台无关)

更新逻辑**自带在技能内部** (`scripts/selfupdate.py`), 不依赖任何客户端的更新机制 ——
读 `SKILL.md` 的 agent (ZCode / Claude Code / 其它) 或人类都能调它, OS 层的定时任务
也调同一个脚本。

| 命令 | 用途 |
| --- | --- |
| `python scripts/selfupdate.py` | 节流自更新 (会话开始时调; 距上次不足 24 h 直接返回, 不联网) |
| `python scripts/selfupdate.py --check` | 只看有没有更新, 不拉取、不写状态 (无副作用) |
| `python scripts/selfupdate.py --force` | 忽略节流立即检查 |
| `python scripts/selfupdate.py --quiet` | 无输出 (供定时任务调用) |

环境变量: `QDOG_UPDATE_INTERVAL` 节流秒数 (默认 86400, 设 0 关闭)、`QDOG_UPDATE_OFF=1` 完全禁用。

**安全护栏** (自更新绝不能打断任务): 只快进 (`pull --ff-only`); 工作区有未提交改动时
只报告不拉取 (保护本地调参); 网络不可达 / 非 git 仓库时保持本地版本并**退出码仍为 0**;
瞬时失败 1 h 后重试, 而不是等满一个间隔。

**两条纪律**:

- *只在开新任务前更新, 不在跑动作途中更新* —— 代码中途变会让 `results/` 里不同工况的
  数据来自不同版本的代码, 事后无法分辨;
- 若更新报告里有 `.py` 变更, 之前 `results/` 的结果**不再与当前代码严格对应**,
  引用旧数据下结论前先重跑相关动作。

想让它在你不开 agent 时也自动更新, 挂一个 OS 级定时任务调同一个脚本即可:

```cmd
:: Windows: 每天 09:00 静默自更新
schtasks /create /tn "qdog self-update" /sc daily /st 09:00 ^
  /tr "python \"<技能目录>\scripts\selfupdate.py\" --quiet"
```

```bash
# Linux/macOS: 每小时
crontab -e   # 0 * * * * python3 <技能目录>/scripts/selfupdate.py --quiet
```

## 已注册动作 (A2)

`walk` 四节拍行走 / `trot` 对角小跑 / `jump` 原地跳跃 / `leap` 前扑跳跃 / `drop` 30 cm 跌落缓冲 /
`walk_full` 满载行走 / `slope` 20° 爬坡 / `slope_full` 15°+25 kg 满载爬坡 / `stand_load` 静站载重 /
`push` 侧向推搡 / `fall` 侧跌倒落 / `stall` 关节堵转

**实测基准** (2026-09 全 12 工况逐个跑通并人工确认, 用于回归对照):

| 工况 | 关键实测 | 说明 |
| --- | --- | --- |
| walk | **0.47 m/s**, 姿态pp 2.1°/1.0°, 力矩峰值 56 N·m (31% 额定) | 步态平滑化后 (GAIT_T 0.65 / DUTY 0.80 / SW_SCALE 2.0); 旧参数为 0.39 m/s |
| trot | **1.33 m/s**, 姿态pp 8.0°/3.8°, 力矩峰值 128 N·m (71%) | 弹跳步态, 腾空占 43%; 航向漂移 −12.2° |
| jump | 起跳 vz **2.96 m/s**, 腾空 0.536 s, 最高 0.828 m, 冲击 5.1×体重 | 蹬伸顶满额定 (大腿 120 / 小腿 180); 后腿先落地, 前腿晚 134 ms |
| leap | 起跳 vx 0.68 / vz **2.03 m/s**, 腾空 0.378 s, 前进 0.267 m, 冲击 6.0×体重 | v2 原版控制器 (深蹲蓄力->爆发蹬伸); 前腿先着地, 左右不对称 (见下) |
| drop | 0.30 m 跌落, 触地 2.44 m/s, 冲击 **11.9×体重**, 四足同时触地 | 力矩峰值 178.8 N·m (99% 额定) |
| walk_full | +25 kg, **0.40 m/s**, 总反力均值 = 满载重量 (638 N) | 航向漂移 9.7°, 为空载的 4 倍 |
| slope | 20°, 上行 **0.20 m/s**, 总反力均值 = mg·cos20° (368 N) | 航向漂移 +9.7° |
| slope_full | 15°+25 kg, 上行 **0.41 m/s**, 总反力均值 = mg·cos15° (613 N) | **航向漂移 −49.1°**, 跑偏最严重 |
| stand_load | +100 kg, 总反力 **1375 N** = 140 kg 重量 (误差 0.1%) | 后腿承 60.5% / 前腿 39.5% (按均分校核会低估后腿 1.5 倍) |
| push | trot 中 50 N×0.25 s, 姿态pp **9.0°/4.3°**, 横向速度增量 +0.28 m/s | 与理论冲量 12.5 N·s/40 kg = 0.31 m/s 吻合 |
| fall | 0.30 m+25° 侧倾, 冲击 **4.8×体重**, 右腿先触地 (FR 1027 N) | **弯矩/剪力全工况最高**: 大腿 137.8 N·m, 小腿 959 N |
| stall | FL 膝焊接 (t=0.8 s 锁 −1.3126 rad), 从 1.46 s 起**持续饱和 180 N·m 共 2.54 s** | 软约束屈服 6.5°; 载荷分配 RL 15 / RR 189 N (差 12.6 倍) |

## 关键文件

| 文件 | 作用 |
| --- | --- |
| `run_pipeline.py` | 入口: 仿真 → 报表 → 弯矩 → 视频 → 包络 |
| `quad_pipeline/motions.py` | 动作与控制器 (按用户意见调参改这里) |
| `quad_pipeline/config.py` | 机型预设 (A2 已注册; 新机型在此 `register`) |
| `quad_pipeline/core.py` | 场景构建与仿真循环 |
| `quad_pipeline/wrench.py` | 关节 6 分量 wrench 与弯矩分解 |
| `quad_pipeline/report.py` | 指标统计、曲线、受力 xlsx |
| `quad_pipeline/envelope.py` | 跨工况载荷包络 |
| `quad_pipeline/video.py` | 双机位跟踪视频 (含坡度/负载可视化; 支持 `t_win` 只渲染时间窗) |
| `check_motion.py` | 动作自检: 步态占空比/周期、左右对称性、触地相足端漂移(打滑)、摆动相接触(拖地)、腾空占比、航向漂移、峰值裕度 |
| `scripts/selfupdate.py` | 技能自更新 (平台无关): 节流、只快进、保护本地未提交改动、失败不打断调用方 |
| `verify_sensor.py` | 弯矩测量方法的数值验证 |

## 输出

每动作 `results/<robot>/<key>/`: `trace.csv`/`trace.npy` (2 ms 全量时序)、`关节受力.xlsx`、
`关节弯矩.xlsx`、`wrench.csv`、`torques.png` / `motion.png` / `bending.png`、
**check 交付物** `video.avi` + `video_preview.gif` + `video_first/mid/last.png`。

瞬态/冲击类工况 (`push` / `fall` / `stall`) 应**额外交付窗口视频**: 全程视频里事件只占极小比例
(如 push 的 0.25 s 冲击只占 3%), 用 `video.render_video(..., t_win=(t0, t1))` 单独渲染事件段,
否则用户无从核验那一下是否真的发生。`push` 另有冲击响应曲线 `push_impact.png`。

`results/<robot>/关节载荷包络.xlsx` / `.png`: 每关节各指标跨工况最大值 + 来源工况 +
工况×关节矩阵, **仅在全部动作确认后** 汇总。

## 已知限制 (2026-09 实测确认)

1. **无横向/偏航闭环**: 控制器只有腿长与俯仰/横滚配平, 直线动作会跑偏 —— `trot` −12.2°
   (横移 1.51 m)、`slope_full` −49.1° (横移 1.47 m)、`slope` +9.7°; 载重与坡道会放大
   (walk_full 漂移为空载 4 倍)
2. **`push` 不是干净的抗扰测试**: trot 自身横向漂移 −0.27 m/s 与推搡 +0.28 m/s 量级相同,
   两者几乎抵消, "恢复"里混着"本来就往外飘"; 要做干净的抗扰评估需先消掉漂移
3. **包络由冲击类工况主导, 但没有关节超出额定驱动扭矩**: 剪力包络几乎全部来自
   `drop` (12 个关节里 11 个), 弯矩来自 `drop` / `fall`, 驱动最大值分散在
   `drop` / `stand_load` / `jump`; 只有 `FL_thigh` 驱动与 `FL_hip` 弯矩来自 `trot`。
   最高占额定比 **99%** (RL/RR 小腿, 来源 `drop`), 小腿是选型瓶颈, 大腿 75~86%、髋 35~71%。
   注: 传感器测的是关节界面**总轴向力矩** (执行器限幅 + 转子惯量/约束内务项),
   冲击瞬间可能略高于执行器指令值, 结构件校核时留这一点余量即可
4. **`leap` 的左右不对称是其固有特征, 不要"优化掉"**: v2 前扑用对角步态助跑
   (FL 与 RR 同相), 起跳/落地本来就是"一前一后成对"发力的, 四足受力并不近似相等。
   曾把它改成 bound 助跑 (前后成对) 得到 0.39% 的左右对称与 0.708 m 的前扑距离,
   **但被否决并回退**: 那是"理想模型 + 确定性控制器"的产物 —— 模型左右腿完全一致、
   无装配公差、无传感器噪声、无伺服差异, 真实机器狗不会这么齐。过度对称的结果
   反而失去参考价值。详见 `README.md` 的"回退记录"
5. **视频渲染内存**: 已改逐帧 JPEG 编码 (原实现把全部帧以未压缩位图驻留内存,
   8 s@50 fps 约需 1 GB, 内存紧时会 `MemoryError`); 内存极紧时可用 `t_win` 缩短渲染区间

## 扩展

- 新增动作: 在 `quad_pipeline/motions.py` 用 `@motion` 注册一个 `CaseSimBase` 子类
- 新增载荷/地形工况: 派生已有动作并覆盖类属性 `PAYLOAD` (kg) / `SLOPE_DEG` (deg)
- 新增机型: 在 `quad_pipeline/config.py` 注册 `RobotConfig`
- 改控制器: `LeapMotion` v3 是"跑动中弹射"结构 —— 足端按躯干前进量钉地 (不刹车)、
  触发同步到后腿双支撑相位、反向下沉加长做功行程、空中用髋外展做滚转控制;
  设计取舍与各版本失败机理见该类 docstring。改步态参数时注意 `WalkMotion` 的改动会被
  `TrotMotion`/`SlopeMotion` 继承, 单独标定的动作需显式隔离 (`SW_SCALE = 1.0` / `DUTY = 0.75`)

完整说明 (功能特性、安装、输出布局、机型适配、许可证) 见 `README.md`。
