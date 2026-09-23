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
                 fps=50, stride=None, checks=True):
    """从轨迹 npy 渲染视频。返回 (avi_path, gif_path, [check_pngs])。"""
    from PIL import Image, ImageDraw

    A = np.load(trace_path)
    t_all, q_all = A[:, 0], A[:, 1:20]
    Fz_all = A[:, C_FZ:C_FZ + 4]
    st_all = A[:, C_ST:C_ST + 4]
    stride = stride or max(1, int(round(500 / fps)))

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=PANEL_H, width=PANEL_W)
    cams = make_cameras()
    f_big, f_mid, f_small = find_font(30), find_font(22), find_font(17)
    col = (40, 130, 80)

    frames = []
    for k in range(0, len(t_all), stride):
        t, q = float(t_all[k]), q_all[k]
        fz, st = Fz_all[k], st_all[k]
        bx, by, bz = float(A[k, C_BX]), float(A[k, C_BY]), float(A[k, C_BZ])
        vx, vcmd = float(A[k, C_VX]), float(A[k, C_VCMD])
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), (24, 26, 30))
        for i, (cname, cam) in enumerate(cams):
            cam.lookat[:] = [bx, by, 0.40]
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
