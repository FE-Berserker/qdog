"""quad_pipeline 视频渲染: 双机位跟踪 + 数据信息条, 纯 Python MJPEG-AVI 编码。

不依赖 ffmpeg: Pillow 出 JPEG 帧 + 手拼 RIFF 容器, WMP/VLC/浏览器可直接播放;
有 ffmpeg (imageio_ffmpeg/系统) 时顺带转 MP4。同时输出 GIF 预览与三张抽查帧。
"""

import shutil
import struct
import subprocess
from pathlib import Path

import mujoco
import numpy as np

from .core import C_FZ, C_ST, C_BX, C_BY, C_BZ, C_VX, C_VCMD
from .config import RobotConfig

PANEL_W, PANEL_H = 720, 528
STRIP_H = 80
CANVAS_W, CANVAS_H = PANEL_W * 2, PANEL_H + STRIP_H


def find_font(size):
    from PIL import ImageFont
    for p in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyh.ttf",
              r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\arial.ttf"):
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def make_cameras():
    cams = []
    for name, kw in (("侧视(跟踪)", dict(distance=2.2, elevation=4, azimuth=90)),
                     ("3/4 视角(跟踪)", dict(distance=2.6, elevation=-14, azimuth=140))):
        c = mujoco.MjvCamera()
        c.distance, c.elevation, c.azimuth = kw["distance"], kw["elevation"], kw["azimuth"]
        cams.append((name, c))
    return cams


def render_video(cfg: RobotConfig, trace_path, scene_path, label, out_avi,
                 fps=50, stride=None, checks=True, slope_deg=0.0, payload_kg=0.0):
    """从轨迹 npy 渲染视频。返回 (avi_path, gif_path, [check_pngs])。

    slope_deg>0 (爬坡工况): 物理用重力倾斜法 (平地+斜重力), 渲染时把整机位姿
    反向刚体旋转 R_y(-θ) 并在场景中放入斜坡障碍物 —— 与真实上坡视角严格等价,
    足端与坡面接触位置精确吻合, 仅用于可视化, 不影响物理。

    payload_kg>0 (负载工况): 在躯干上挂一个纯视觉货箱 (无碰撞/无质量),
    尺寸按质量立方根缩放 (以 25 kg 为基准), 满载/静载直观可读。"""
    from PIL import Image, ImageDraw

    A = np.load(trace_path)
    t_all, q_all = A[:, 0], A[:, 1:20]
    Fz_all = A[:, C_FZ:C_FZ + 4]
    st_all = A[:, C_ST:C_ST + 4]
    stride = stride or max(1, int(round(500 / fps)))

    th = np.radians(slope_deg)
    sn, cs = np.sin(th), np.cos(th)
    if slope_deg > 0.0 or payload_kg > 0.0:
        spec = mujoco.MjSpec.from_file(str(scene_path))
        if payload_kg > 0.0:
            # 背上货箱 (挂躯干随动): 25 kg 基准箱 0.24×0.20×0.18 m
            sc = (payload_kg / 25.0) ** (1.0 / 3.0)
            hx, hy, hz = 0.12 * sc, 0.10 * sc, 0.09 * sc
            spec.body(cfg.trunk_body).add_geom(
                name="payload_crate", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[hx, hy, hz], pos=[0.0, 0.0, 0.13 + hz],
                rgba=[0.72, 0.48, 0.24, 1.0], contype=0, conaffinity=0)
        if slope_deg > 0.0:
            # 斜坡障碍物: 顶面 = 旋转后的地板面 (整体上抬 h0, 下坡端埋入平地)
            h0 = 2.2 * sn / cs
            sx, sy, sz = 3.0, 1.4, 1.0           # 半长/半宽/半厚
            px, pz = 0.3, 0.3 * sn / cs + h0     # 坡顶面中心点 (机器人活动区上方)
            gx, gz = px + sz * sn, pz - sz * cs  # 盒中心 = 面点 - 半厚×法向(-sn,0,cs)
            # 姿态用四元数 (a2.xml compiler 为 radian, euler/axisangle 单位不可靠)
            qw_b, qy_b = np.cos(th / 2), -np.sin(th / 2)
            spec.worldbody.add_geom(
                name="ramp", type=mujoco.mjtGeom.mjGEOM_BOX, size=[sx, sy, sz],
                pos=[gx, 0.0, gz], quat=[qw_b, 0.0, qy_b, 0.0],
                rgba=[0.45, 0.42, 0.40, 1.0], contype=0, conaffinity=0)
        model = spec.compile()
    else:
        model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    if cfg.mass <= 0:
        # 独立渲染 (未经仿真) 时 cfg.mass 未初始化: 从模型求和 (基准自重)
        cfg.mass = float(sum(model.body_mass))
    renderer = mujoco.Renderer(model, height=PANEL_H, width=PANEL_W)
    cams = make_cameras()
    if slope_deg > 0.0:
        # 坡面视角: 侧视拉远含整段坡; 3/4 改到前下方 (默认角度会被坡体遮挡)
        cams[0][1].distance = 3.0
        cams[1][1].azimuth, cams[1][1].elevation, cams[1][1].distance = 60, -5, 3.0
    f_big, f_mid, f_small = find_font(30), find_font(22), find_font(17)
    col = (40, 130, 80)

    frames = []
    for k in range(0, len(t_all), stride):
        t, q = float(t_all[k]), q_all[k]
        fz, st = Fz_all[k], st_all[k]
        bx, by, bz = float(A[k, C_BX]), float(A[k, C_BY]), float(A[k, C_BZ])
        vx, vcmd = float(A[k, C_VX]), float(A[k, C_VCMD])
        data.qpos[:] = q
        if slope_deg > 0.0:
            # 整机刚体旋转 R_y(-θ) 到真实坡面视角 (足端精确落在坡面)
            x, z = data.qpos[0], data.qpos[2]
            data.qpos[0] = cs * x - sn * z
            data.qpos[2] = sn * x + cs * z + h0
            qw, qx, qy, qzq = data.qpos[3:7]
            ch, sh = np.cos(th / 2), np.sin(th / 2)
            data.qpos[3:7] = [ch * qw + sh * qy, ch * qx - sh * qzq,
                              ch * qy - sh * qw, ch * qzq + sh * qx]
        mujoco.mj_forward(model, data)
        canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), (24, 26, 30))
        for i, (cname, cam) in enumerate(cams):
            cam.lookat[:] = [data.qpos[0], data.qpos[1], data.qpos[2] - (bz - 0.40)]
            renderer.update_scene(data, cam)
            canvas.paste(Image.fromarray(renderer.render()), (i * PANEL_W, 0))
            d = ImageDraw.Draw(canvas)
            d.rectangle([i * PANEL_W + 8, 8, i * PANEL_W + 8 + 170, 8 + 30], fill=(0, 0, 0))
            d.text((i * PANEL_W + 14, 10), cname, font=f_small, fill=(255, 255, 255))
        d = ImageDraw.Draw(canvas)
        y0 = PANEL_H
        d.rectangle([0, y0, CANVAS_W, CANVAS_H], fill=(24, 26, 30))
        d.text((12, y0 + 8), f"t = {t:5.2f} s", font=f_big, fill=(240, 240, 240))
        d.rectangle([170, y0 + 10, 170 + 160, y0 + 42], fill=col)
        d.text((180, y0 + 12), label, font=f_mid, fill=(255, 255, 255))
        d.text((360, y0 + 8), f"vx {vx:+.2f} / 指令 {vcmd:+.1f} m/s   躯干高度 {bz:.3f} m",
               font=f_mid, fill=(200, 210, 220))
        d.text((360, y0 + 38), f"总地面反力 {fz.sum():6.0f} N  ({fz.sum()/(cfg.mass * 9.81):.1f}×体重)",
               font=f_mid, fill=(200, 210, 220))
        bx0, bw, bgap, bmax = 1000, 92, 14, 600.0
        base_y = y0 + STRIP_H - 14
        d.text((bx0 - 6, y0 + 6), "四足垂直反力 (N)", font=f_small, fill=(160, 170, 180))
        for j, lg in enumerate(cfg.legs):
            x = bx0 + j * (bw + bgap)
            h = int(38 * min(abs(fz[j]) / bmax, 1.0))
            c = (240, 120, 60) if st[j] > 0.5 else (90, 160, 240)
            d.rectangle([x, base_y - h, x + bw, base_y], fill=c)
            d.text((x + 14, base_y - h - 18), f"{abs(fz[j]):.0f}", font=f_small, fill=(220, 220, 220))
            d.text((x + 30, base_y + 3), lg, font=f_small, fill=(200, 200, 200))
        d.line([bx0 - 4, base_y, bx0 + 4 * bw + 3 * bgap + 4, base_y], fill=(120, 120, 120))
        frames.append(canvas)
    renderer.close()

    out_avi = Path(out_avi)
    write_avi_mjpeg(out_avi, frames, fps)

    # GIF 预览
    small = [f.resize((f.width // 2, f.height // 2)) for f in frames[::3]]
    gif = out_avi.with_name(out_avi.stem.replace("_video", "") + "_preview.gif")
    small[0].save(gif, save_all=True, append_images=small[1:],
                  duration=int(3000 / fps), loop=0, optimize=True)

    # 抽查帧
    checks_paths = []
    if checks:
        for tag, i in (("first", 0), ("mid", len(frames) // 2), ("last", -1)):
            p = out_avi.with_name(out_avi.stem.replace("_video", "") + f"_{tag}.png")
            frames[i].save(p)
            checks_paths.append(p)

    # 可选 MP4 转码
    exe = _ffmpeg_exe()
    if exe:
        mp4 = out_avi.with_suffix(".mp4")
        r = subprocess.run([exe, "-y", "-i", str(out_avi), "-vcodec", "libx264",
                            "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart",
                            str(mp4)], capture_output=True, text=True)
        if r.returncode != 0:
            print("ffmpeg 转 MP4 失败:", r.stderr[-200:])
    return out_avi, gif, checks_paths


# ---------------- MJPEG-AVI 编码 (纯 Python) ----------------

def _chunk(fid, tag, data):
    fid.write(tag)
    fid.write(struct.pack("<I", len(data)))
    fid.write(data)
    if len(data) % 2:
        fid.write(b"\x00")


def write_avi_mjpeg(path, frames, fps):
    import io
    n = len(frames)
    w, h = frames[0].size
    jpegs = []
    for im in frames:
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=92)
        jpegs.append(buf.getvalue())
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 0) + b"AVI ")
        avih = struct.pack("<14I", 1000000 // fps, w * h * 3 * fps, 0, 0x10,
                           n, 0, 1, w * h * 3, w, h, 0, 0, 0, 0)
        strh = struct.pack("<4s4sIHHIIIIIIIIhhhh", b"vids", b"MJPG", 0, 0, 0,
                           0, 1, fps, 0, n, w * h * 3, 0, 0, 0, 0, w, h)
        strf = struct.pack("<IiiHH4sIIIii", 40, w, h, 1, 24, b"MJPG", w * h * 3, 0, 0, 0, 0)
        strl = b"strl" + (b"strh" + struct.pack("<I", len(strh)) + strh
                          + (b"strf" + struct.pack("<I", len(strf)) + strf))
        hdrl = b"hdrl" + (b"avih" + struct.pack("<I", len(avih)) + avih
                          + (b"LIST" + struct.pack("<I", len(strl)) + strl))
        _chunk(f, b"LIST", hdrl)
        movi = b"".join(b"00dc" + struct.pack("<I", len(j)) + j + (b"\x00" if len(j) % 2 else b"")
                        for j in jpegs)
        _chunk(f, b"LIST", b"movi" + movi)
        offsets, off = [], 4
        for j in jpegs:
            offsets.append((off, len(j)))
            off += 8 + len(j) + (len(j) % 2)
        idx1 = b"idx1" + struct.pack("<I", 16 * n)
        for o, s in offsets:
            idx1 += b"00dc" + struct.pack("<II", 0x10, o) + struct.pack("<I", s)
        f.write(idx1)
        total = f.tell()
        f.seek(4)
        f.write(struct.pack("<I", total - 8))
    return path


def _ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")
