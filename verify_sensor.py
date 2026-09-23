"""关节 wrench 传感器方案验证 (extract_bending.py 所用方法的依据)。

结论: MuJoCo 原生 force/torque site 传感器精确测量"父体->子体"关节相互作用 wrench:
  - 双摆两关节: 传感器绕轴分量 == 执行器+被动力矩, 误差 0.0000 (自由空间, 无接触)
  - 整狗静态站立: 力/弯矩量级与重力载荷一致, 小腿剪力 ≈ 足底反力, 左右对称
绕轴分量与"执行器力矩"的残差来自转子惯量(armature)与限位力矩等轴向内务项;
垂直分量(弯矩/剪力)不受其影响, 为纯结构载荷。
"""
import mujoco
import numpy as np

XML = """
<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <body name="l1" pos="0 0 2">
      <joint name="j1" type="hinge" axis="0 1 0" damping="0.1"/>
      <geom type="capsule" size="0.02" fromto="0 0 0  0 0 -0.3" mass="2"/>
      <site name="s1" pos="0 0 0"/>
      <body name="l2" pos="0 0 -0.3">
        <joint name="j2" type="hinge" axis="0 1 0" damping="0.1"/>
        <geom type="capsule" size="0.02" fromto="0 0 0  0 0 -0.3" mass="1"/>
        <site name="s2" pos="0 0 0"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="m1" joint="j1" gear="1" ctrlrange="-50 50"/>
    <motor name="m2" joint="j2" gear="1" ctrlrange="-50 50"/>
  </actuator>
  <sensor>
    <force name="f1" site="s1"/> <torque name="t1" site="s1"/>
    <force name="f2" site="s2"/> <torque name="t2" site="s2"/>
  </sensor>
</mujoco>"""

m = mujoco.MjModel.from_xml_string(XML)
d = mujoco.MjData(m)
d.qpos[:] = [0.4, -0.7]
d.qvel[:] = [0.5, 1.2]
d.ctrl[:] = [2.0, 0.8]
mujoco.mj_forward(m, d)

names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(m.nsensor)]
ok = True
for nm, jn, ai in (("t1", "j1", 0), ("t2", "j2", 1)):
    adr = m.sensor_adr[names.index(nm)]
    got = float(d.sensordata[adr + 1])          # 关节轴 = y
    dof = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)]
    want = float(d.actuator_force[ai] + d.qfrc_passive[dof])
    ok &= abs(got - want) < 1e-9
    print(f"{nm}: 传感器绕轴 = {got:+.6f},  act+pas = {want:+.6f}   {'OK' if abs(got-want) < 1e-9 else 'FAIL'}")
print("PASS" if ok else "FAIL")
