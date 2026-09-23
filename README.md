# qdog

四足机器狗 MuJoCo 动作仿真与关节载荷分析流水线。

对一台四足机器狗自动完成 **行走 / 奔跑 / 跳跃 / 前扑 / 跌落** 五个基本动作的仿真,
以及 **满载行走 / 爬坡 / 满载爬坡 / 静站载重** 载荷地形工况、
**侧向推搡 / 侧跌倒落 / 关节堵转** 异常失效工况 (按设计工况统计的工况矩阵扩展),
提取每个关节的 **扭矩、弯矩、受力(轴向/剪切)**, 生成 **动作视频** 和 **Excel 表格**,
最后跨动作汇总 **载荷包络** —— 用于关节/轴承/结构件的设计选型校核。

## 功能特性

- **五个内置基本动作**(均在 Unitree A2 上实仿调定): 四节拍行走(0.39 m/s)、对角小跑
  (1.33 m/s)、原地跳跃(起跳 2.9 m/s)、前扑跳跃(腾空 0.34 s 前进 0.24 m)、
  30 cm 跌落缓冲(冲击峰值 11.9 倍体重)
- **四个载荷/地形工况**(工况 × 负载矩阵, 对应设计工况统计中"额定扭矩与热平衡"、
  "结构静强度"两类决定性边界): 满载行走(25 kg)、20° 爬坡(实测上行 0.17 m/s)、
  满载爬坡(15°+25 kg, 实测上行 0.42 m/s)、静站载重(100 kg)。
  负载/坡度为 Motion 类属性 (`PAYLOAD`/`SLOPE_DEG`), 派生即新工况;
  坡度用重力矢量倾斜法(力学上与上坡严格等价, 视频渲染时反向旋转并放入
  斜坡障碍物, 坡角直观可见; 负载工况视频在背部挂货箱可视化)
- **三个异常/失效工况**(可靠性设计与安全功能输入): 侧向推搡(trot 中 50 N×0.25 s
  躯干冲击, 恢复)、侧跌倒落(0.30 m+25° 侧倾非受控跌落, 冲击 4.8 倍体重)、
  关节堵转(膝关节 equality 焊接卡滞+下蹲施压, 执行器持续饱和 180 N·m,
  驱动器 I²t 热保护设计输入)
- **关节载荷全口径**: 绕轴驱动扭矩(电机承担) + 弯矩(轴承/壳体承担) + 轴向力 +
  剪切力; 弯矩用 MuJoCo 原生 force/torque 传感器在关节锚点直接测量,
  经双摆数值验证(绕轴分量与执行器+被动力矩精确一致, 见 `verify_sensor.py`)
- **完整产物**: 每动作输出全量时序 CSV(2 ms, 含重力/被动/约束/惯性分解)、
  关节受力.xlsx、关节弯矩.xlsx、扭矩/运动学/弯矩曲线 PNG、双机位跟踪视频
  (纯 Python MJPEG-AVI 编码, 不依赖 ffmpeg; 有 ffmpeg 时另出 MP4)
- **载荷包络**: 跨工况取每关节各指标最大值并标注来源工况, 附工况×关节矩阵热力图、
  占额定比列与参考判据(落地冲击约 2.3×额定属正常、电机峰值 5~10×、建议 3×余量)
- **可扩展**: 新动作 = 注册一个控制器子类; 新机型 = 注册一个 RobotConfig

## 安装

```bash
pip install -r requirements.txt
```

需要自行准备机器人 MuJoCo 模型(见下文"机器人模型")。

## 快速开始

```bash
python run_pipeline.py                 # A2 全部已注册动作全流程
python run_pipeline.py --motions drop  # 只跑跌落 (快速验证, 约 1 分钟)
python run_pipeline.py --motions walk_full slope slope_full stand_load --skip-video
                                       # 只跑载荷/地形工况
python run_pipeline.py --skip-video    # 跳过视频渲染
python run_pipeline.py --list          # 查看已注册机型/动作
```

### 逐动作确认 (在 agent 对话层, 不在终端)

动作质量与控制器/动作设计强相关, 需要"看视频→提意见→调参→重跑"的循环,
而调整意见是自由文本, 终端选项装不下。因此本工具不内置终端问答, 推荐由
agent (如 ZCode) 逐动作驱动: 每跑完一个动作, 在对话中给出指标摘要与视频,
你直接回复调整意见, agent 修改 `quad_pipeline/motions.py` 参数重跑,
确认后再做下一个。全部确认后汇总载荷包络。脚本本身不暂停不提问,
同样适合批处理调用; 每个动作数据在仿真结束即刻落盘, 中断不丢结果。

## 输出布局 (`results/<robot>/`)

```
scene.xml                    场景 (绝对路径引用模型, 位置无关)
scene_stall.xml              堵转定制场景 (equality 焊接; SCENE_EXTRA 机制)
<motion>/                    walk / trot / jump / leap / drop
                             walk_full / slope / slope_full / stand_load
                             push / fall / stall
  trace.csv / trace.npy      全量时序 (2 ms): q/v/扭矩 + 重力/被动/约束/惯性分解
                             + 四足反力 + 支撑标志 + 速度指令
  关节受力.xlsx              扭矩时序 + 统计 + 工况指标
  torques.png / motion.png   扭矩曲线 / 运动学+地面反力曲线
  wrench.csv                 12 关节 6 分量相互作用 wrench
  关节弯矩.xlsx              弯矩时序 + 载荷统计 (驱动/弯矩/轴向/剪切)
  bending.png                弯矩与绕轴驱动对比曲线
  video.avi / preview.gif    双机位跟踪视频 + GIF 预览
关节载荷包络.xlsx / .png     包络: 每关节各指标最大值 + 来源工况 + 工况×关节矩阵
```

## 机器人模型

本仓库**不含**机器人模型文件。A2 预设引用 `unitree_robots/a2/a2.xml`,
请从 Unitree 官方仓库获取 (unitree_ros/robots 或 unitree_mujoco, BSD 3-Clause,
版权归杭州宇树科技), 放置于本目录 `unitree_robots/a2/` 下即可。

## 扩展

**新增动作** (`quad_pipeline/motions.py`):

```python
@motion("my_move", "我的动作", "工况说明", t_end=6.0, win=(1.0, 6.0))
class MyMove(CaseSimBase):
    def init_pose(self): ...
    def step(self): ...        # 每步算 12 关节目标 -> self.apply_pd(q_tgt, kp, kd)
```

**新增载荷/地形工况变体** (派生覆盖类属性即可):

```python
@motion("trot_full", "满载奔跑", "对角小跑 + 25 kg", t_end=8.0, win=(2.8, 8.0))
class TrotFullMotion(TrotMotion):
    PAYLOAD = 25.0             # 躯干附加负载 kg (挂于 cfg.trunk_body 质心)
    SLOPE_DEG = 0.0            # 等效坡度 deg (重力矢量倾斜法)
    # 可选: R_ST 站姿腿长 / K_FB / XC_LIM / XA_BOOST 步态参数
    # 可选: SCENE_EXTRA 场景定制片段 (如堵转的 equality 焊接), 自动生成 scene_<key>.xml
```

注意: 时钟步态 PD 控制器的扭矩预算有限 —— 30° 坡即使空载也无法持续上行
(执行器饱和), 故爬坡定为 20°、满载爬坡定为 15°(均为实仿验证的真实上行);
更陡坡度+满载需要整机姿态/力分配规划控制器。重载保持类工况(如静站载重)
需加大 PD 刚度; 更陡/更重的爬坡工况需要更强的控制器, 或按设计需要降低坡度考核值。

**新增机型** (`quad_pipeline/config.py`):

```python
register(RobotConfig(name="go2", model_xml="...", mesh_dir="...",
                     L1=..., L2=..., foot_r=..., r_max=..., r_stand=...,
                     rated=(..., ..., ...)))
```

关节命名约定 `{leg}_{part}_joint` (leg=FL/RL/FR/RR, part=hip/thigh/calf);
不一致时改 `joint_fmt`/`act_fmt`/`body_fmt` 模板即可。动作参数(步态周期/
速度/蹬伸深度等)为各 Motion 子类的类属性, 换机型时派生覆盖。

## 验证

```bash
python verify_sensor.py    # 弯矩测量方法的双摆数值验证 (应输出 PASS)
```

## 许可证

本流水线代码采用 [MIT License](LICENSE)。

第三方内容: MuJoCo (DeepMind, Apache-2.0); Unitree 机器人模型文件 (BSD 3-Clause,
版权归杭州宇树科技有限公司) 不包含在本仓库内, 需自行获取并遵守其许可证。
