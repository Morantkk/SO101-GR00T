#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""官方 Isaac-GR00T 管道 —— **本机**（接机械臂那台 Windows）真机客户端。

架构
    本机(本脚本) --ZMQ/msgpack--> 127.0.0.1:5555  ==SSH隧道==>  云端 100_policy_server_jpeg.py

★ 为什么必须"和采数据/训练时一致"
    模仿学习学的是「(双相机画面 + 关节角) → 接下来 16 步关节角」。任何一项对不上，
    喂进去的都是**分布外**输入。
    ┌ 项目            采数据 / 训练时                     本脚本
    ├ 相机采集分辨率   front 640×360、wrist 1280×720（各自原生）  --front-size / --wrist-size 默认就是这两个
    ├ 喂给模型的尺寸   两路都 640×360（数据集统一降过，官方 VideoCrop 要求同分辨率）  软件 resize 到 640×360
    ├ 宽高比          **两路都是 16:9**                  会检查；若相机给成 4:3 会告警（那会纵向压扁）
    ├ 相机帧率        15 fps                              --cam-fps 15
    ├ 动作时间间隔    1/15 s = 66.7 ms                    --dt 1/15 ★ 决定机械臂运动速度，最关键
    ├ 关节单位        度                                  lerobot use_degrees=True
    ├ 任务文本        Put all the objects on the table into the box  --task 默认值
    └ action horizon  16                                  服务端按 checkpoint 里的 16

★ JPEG 上行（默认开启）
    两路原图 1.35MB/次 → 隧道往返 ~1.1s；压成 JPEG q90 后 ~85KB → **0.16s**（实测快 ~7 倍）。
    服务端 `100_policy_server_jpeg.py` 解回原尺寸再喂模型，**模型看到的东西不变**。
    关掉：`--jpeg-quality 0`。

用法
    <default python> scripts/ssh_tunnel.py --local 5555 --remote 127.0.0.1:5555   # 另开窗口保持
    <lerobot-win python> scripts/server/99_test_policy_client.py 5555             # 不需要机械臂
    <lerobot-win python> scripts/run_groot_official.py --dry-run --steps 20       # 臂不动
    <lerobot-win python> scripts/run_groot_official.py --steps 300 --max-rel 8
"""
import argparse
import io
import os
import sys
import time

import numpy as np

TASK_DEFAULT = "Put all the objects on the table into the box"   # ★ 当前任务（v3 / tune_visual 探针）；必须与训练逐字一致
TASK_PEN = "Pick up a pen and put it into the box"               # 旧任务（v1 抓笔，权重已丢失）——用 --task 显式传入
MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CAM_W, CAM_H = 640, 360            # ★ 喂给模型的尺寸（= checkpoint metadata 记录值）
CAM_FPS = 15                       # ★ 采集帧率
DT = 1.0 / 15                      # ★ 采集时相邻帧间隔
FRONT_SIZE_DEFAULT = (640, 360)    # 前置相机原生
WRIST_SIZE_DEFAULT = (1280, 720)   # 腕部相机原生

# ★★ 预设位姿**不能跨任务用** —— 它是某个数据集里真实帧的关节角，换任务/换数据集就失效。
#    当前任务 = **抓桌上所有物体**（模型 v3 = groot_arm101_all_objects，102 条演示）。
TASK_START = np.array([-5.05, -103.53, 92.79, 62.48, 6.32, 7.58])       # 102 条演示的【起始】均值
TASK_START_STD = np.array([4.71, 3.77, 2.41, 6.10, 6.58, 4.15])          # 起始位姿离散度（度）
TASK_MID = np.array([-10.25, 16.41, -8.98, 77.09, -1.45, 12.86])         # 约 35% 进度（★ 实测对首帧分布是 lift+26σ/elbow−38σ，纯分布外，别再用它判读）

# ★★★ 2026-10-15 18:4x 定案：**演示首帧那个位姿不能用**（模型会把 lift 往桌子里压）。
#   离线带符号探测（`118_sign_probe.py`）在演示首帧上得到：
#       GT 首步：lift +0.13°、elbow −7.21°   （肩不动，靠肘先把小臂抬起来）
#       PRED   ：lift −7.26°、elbow −1.95°   （★ 肩方向反了 → 往桌面压）
#   而臂平放桌面时 lift 的实际下限就是 −104.3°（= 数据集 lift 最小值），
#   真机第一步命令正是 Δlift = −10° ⇒ **物理上压不下去 ⇒ 卡死 ⇒ 闭环永久冻结**。
#   从演示**第 10 帧**起就正常了（GT lift +12.53 / elbow −8.92，PRED +8.73 / −5.97，方向都对）。
TASK_OFF10 = np.array([-4.81, -88.24, 72.69, 68.76, 7.34, 8.01])        # 演示第 10 帧均值：离桌面约 16°
TASK_OFF10_STD = np.array([5.66, 13.50, 8.42, 5.04, 7.09, 4.11])
TASK_OFF15 = np.array([-4.43, -62.84, 59.02, 67.57, 5.46, 8.04])        # 演示第 15 帧均值：离桌面约 41°
TASK_OFF15_STD = np.array([8.51, 18.09, 11.25, 6.17, 9.70, 4.10])

# ★ 2026-10-20 13:4x 按用户要求**移除 off20/off25/off30 三个"更高起点"预设**
#   （原意是避开"压桌面"的冻结，但它们会让臂一开局就被抬到 lift −43/−29/−19，
#     与演示"从桌面慢慢抬起"的起点差太远；用户要求不要这种"初始状态往上走"的设置）。
#   ⇒ 现在 `--start-pose` 不带 = **完全不摆位**，模型从臂当前所在的实际位置直接开始。
#
# ★★ 13:4x 用户反馈："从 off20 起，模型最先去找盒子" —— 这有物理原因：
#   演示里"臂抬起 + 肘伸开"多出现在**刚抓完、往盒子送**的阶段；而"低位 + 肘弯"才是**接近物体**的阶段。
#   从 off20（lift −43、elbow 49）起，模型更容易判断成"我已经抓到了 ⇒ 该去盒子"。
#   ⇒ 起点要往**更早、更低、肘更弯**的帧靠（同时离桌面别太近，免得首步下压就顶到极限）。
#   第 14 帧是最佳折中：lift −67.8（比 off10 高 20°，首步下压 16° 也只到 −84）、
#   且该帧真值动作量最大（|Δ| 中位 1.98 ≈ 全片最猛）⇒ 正是"刚快速启动"的相位。
TASK_OFF12 = np.array([-4.73, -78.26, 67.09, 68.42, 6.81, 8.01])        # 演示第 12 帧均值（lift −78°，离桌面 26°）
TASK_OFF14 = np.array([-4.52, -67.82, 61.58, 67.88, 5.92, 8.02])        # 演示第 14 帧均值（lift −68°，离桌面 36°）

# 兼容旧名：DEMO_START 直接指向**当前任务**的起始位姿（避免又拿抓笔的位姿去跑新任务）
DEMO_START = TASK_START
DEMO_START_STD = TASK_START_STD

# ★★ 演示开头那段"没有触发信号的静止期"（约占 16%）会造成闭环死锁；
#    原以为 `104_trim_static.py` 裁掉前导静止段就够，但 10-15 18:4x 的逐关节带符号探测发现更深一层：
#    **首帧那个"臂平放桌面"的位姿本身，模型第一步就会把 shoulder_lift 往桌里压 7°**（真机 −10°），
#    臂压不下去 ⇒ 状态不变 ⇒ 每个 chunk 重复同一错误 ⇒ 永久冻结。
#    ⇒ 真机**别再从 "demo" 起**，用 "off10"（演示第 10 帧位姿，方向已正确、物理有余量）。
POSE_PRESETS = {
    # ── 当前任务：抓桌上所有物体（v3 模型）──
    "off14": TASK_OFF14,  # ★★★ 推荐：第 14 帧（lift −68°，刚快速启动的相位、且离桌面 36° 有余量）
    "off12": TASK_OFF12,  # ★★ 更靠前：第 12 帧（lift −78°，更"低位待抓"，但离极限只剩 26°）
    "off10": TASK_OFF10,  # ⚠ 第 10 帧（lift −88°）—— 曾推荐过，但真机上约 1/4 概率一开局就冻结
    "off15": TASK_OFF15,  # ⚠ 第 15 帧（lift −63°）
    "demo": TASK_START,   # ⚠ 演示首帧（臂平放桌面）—— 已证伪，会卡死，仅留作对照
    "mid":  TASK_MID,     # ⚠ 分布外（lift +26σ / elbow −38σ），别再用它判读
    # ── ★「摆放引导」：3 个真实演示在**合爪那一刻**的原样姿势（k-medoids 从 102 集里挑的）──
    #    用途：让臂**自己摆到抓取姿势**停住 → 你把物体放到夹爪正下方即可，不用量距离。
    #    末端位置：guide1 r=0.211m(方位-13.6°)  guide2 r=0.221m(+33.5°)  guide3 r=0.219m(+7.5°)
    #    代表性：guide3 覆盖 57/102 条演示，guide1 33 条，guide2 12 条
    "guide1": np.array([15.12, 17.71, 17.01, 64.26, 36.00, 8.37]),    # ep004 第129帧
    "guide2": np.array([-40.53, 16.66, 14.73, 62.15, 12.44, 6.49]),   # ep045 第233帧
    "guide3": np.array([-10.81, 10.15, 25.71, 55.56, 41.71, 9.57]),   # ep095 第154帧（最典型）
    # ── ★ 子任务模型（subtask/cycle）专用：一次抓取—放置的起始位姿中位（515 段统计，lift +18° 远离极限）──
    "subtask": np.array([-7.12, 18.33, -9.98, 73.58, -2.59, 20.88]),
    "reach": np.array([-7.12, 18.33, -9.98, 73.58, -2.59, 20.88]),   # 旧名，等价
    # ── 旧任务：抓一支笔（v1 模型 groot_arm101_pick_one_pen_v1）──
    "pen_demo": np.array([-8.45, -102.29, 90.95, 71.36, 2.66, 5.08]),
    "pen_t080": np.array([-5.01, -100.00, 72.92, 68.66, 5.23, 1.20]),
    "pen_t100": np.array([-8.30, -35.50, 37.10, 70.70, 5.10, 12.00]),
    "pen_t120": np.array([-8.70, 11.00, 8.60, 75.70, 5.10, 28.60]),
}

GUIDE_ORDER = ["guide3", "guide1", "guide2"]   # 最典型的先来



def parse_size(s):
    w, h = s.lower().replace("*", "x").split("x")
    return int(w), int(h)


# ------------------------------------------------------------------ 官方协议
class ZmqPolicyClient:
    """官方 `gr00t/eval/service.py` 的最小复刻：ZeroMQ REQ + msgpack（numpy 用 np.save 打包）。"""

    def __init__(self, host="127.0.0.1", port=5555, timeout_ms=30000):
        import zmq

        self._mp = __import__("msgpack")
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect("tcp://%s:%d" % (host, port))

    @staticmethod
    def _encode(o):
        if isinstance(o, np.ndarray):
            b = io.BytesIO()
            np.save(b, o, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": b.getvalue()}
        return o

    @staticmethod
    def _decode(o):
        if "__ndarray_class__" in o:
            return np.load(io.BytesIO(o["as_npy"]), allow_pickle=False)
        return o

    def call(self, endpoint, data=None, requires_input=True):
        req = {"endpoint": endpoint}
        if requires_input:
            req["data"] = data
        payload = self._mp.packb(req, default=self._encode)
        self.sock.send(payload)
        raw = self.sock.recv()
        resp = self._mp.unpackb(raw, object_hook=self._decode)
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError("服务端报错: %s" % resp["error"])
        return resp, len(payload)

    def ping(self):
        return self.call("ping", requires_input=False)[0]

    def get_action(self, obs):
        return self.call("get_action", obs)


# ------------------------------------------------------------------ 图像
BLACK_CAMS = set()   # 由 --black-cams 填充：{'front'} / {'wrist'} / {'front','wrist'}
                     #   用途：真机 A/B —— 把某路相机整帧置黑，看策略到底用不用相机


def prep_image(img, which=None):
    """统一成 HWC RGB uint8，并 resize 到 640×360（与数据集一致；保持 16:9 不压扁）。

    which 给出相机名时，若该路在 --black-cams 里 → 整帧置黑（真机视觉消融）。
    注意：置黑发生在 resize 之后，所以喂给服务端的尺寸/类型与正常完全一致，只有像素变了。
    """
    if hasattr(img, "detach"):
        img = img.detach().cpu().numpy()
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    if img.shape[:2] != (CAM_H, CAM_W):
        import cv2
        interp = cv2.INTER_AREA if img.shape[0] >= CAM_H else cv2.INTER_LINEAR
        img = cv2.resize(img, (CAM_W, CAM_H), interpolation=interp)
    img = np.ascontiguousarray(img.astype(np.uint8))
    if which is not None and which in BLACK_CAMS:
        img = np.zeros_like(img)
    return img


def to_wire(img, quality):
    """quality>0 → JPEG dict（省带宽）；否则 → (1,H,W,3) 原图。两者服务端语义一致。"""
    if quality and quality > 0:
        import cv2
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if ok:
            return {"__jpeg__": True, "bytes": buf.tobytes()}
    return img[None, ...]


# ------------------------------------------------------------------ 机械臂
def build_robot(a, strict=True):
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    def cam(idx, size):
        kw = {"index_or_path": idx}
        if strict:                    # ★ 按**各自原生**分辨率设，别强迫两路同尺寸
            kw.update(width=size[0], height=size[1], fps=a.cam_fps)
        return OpenCVCameraConfig(**kw)

    cams = {"front": cam(a.front_idx, a.front_size), "wrist": cam(a.wrist_idx, a.wrist_size)}
    return SO101Follower(SO101FollowerConfig(port=a.robot_port, id=a.robot_id, cameras=cams))


def state_vec(obs):
    return np.array([float(obs["%s.pos" % m]) for m in MOTORS], dtype=np.float64)


def action_to_dict(vec6):
    return {"%s.pos" % m: float(v) for m, v in zip(MOTORS, vec6)}


def _no_clobber(path):
    """★ 不覆盖已有文件：同名存在时自动加时间戳。

    （踩过：第一次"能自己动"的 v3_run1.csv 被第二次运行悄悄覆盖，关键证据丢了。）
    """
    if not path or not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    new = "%s_%s%s" % (root, time.strftime("%H%M%S"), ext)
    print("⚠ %s 已存在 → 本次改写到 %s（防止覆盖上一轮的数据）" % (path, new))
    return new


def read_positions(robot):
    """★ 只读**电机位置**（不读相机，比 get_observation 便宜得多）。

    为什么需要它：CSV 里的 `state_*` 列存的是"**喂给模型的那一次观测**"，
    而一个 chunk 的 16 步共用同一次观测 → `state_*` 每 16 步才变一次，**极易误读成"臂没动"**。
    `fb_*`（feedback）才是每步读回的真实关节角。读不到就返回 None（该列留空）。
    """
    bus = getattr(robot, "bus", None)
    if bus is None:
        return None
    try:
        d = bus.sync_read("Present_Position")
        return np.array([float(d[m]) for m in MOTORS], dtype=np.float64)
    except Exception:  # noqa: BLE001
        return None


def alignment_check(robot):
    """核对每个相机的**采集分辨率 / 宽高比 / 协商帧率**。

    ★ 别用「循环 read_latest() 计数 ÷ 时间」测帧率：lerobot 的 OpenCVCamera 有后台线程持续抓帧，
      read_latest() 只返回最近一帧的**缓存**（立即返回）→ 会算出百万级天文数字。
      要看相机自己协商到的 cam.fps（来自 cv2.CAP_PROP_FPS）。
    """
    print("参数对齐自检（采集时：front 640×360 / wrist 1280×720，均 16:9，15fps → 喂模型 640×360）")
    for key in ("front", "wrist"):
        cam = robot.cameras[key]
        f1 = cam.read_latest()
        time.sleep(0.35)
        f2 = cam.read_latest()
        frozen = np.array_equal(f1, f2)
        w = int(getattr(cam, "width", None) or f2.shape[1])
        h = int(getattr(cam, "height", None) or f2.shape[0])
        ar = w / float(h)
        fps = getattr(cam, "fps", None)
        sent = prep_image(f2)
        tags = []
        if abs(ar - 16 / 9) > 0.05:
            tags.append("⚠ 宽高比 %.2f 不是 16:9 → 缩到 640×360 会**变形**（确认相机没跑 4:3）" % ar)
        if fps and abs(float(fps) - CAM_FPS) > 5:
            tags.append("⚠ 协商帧率 %.1f 与采集的 15 差得多" % float(fps))
        print("   %-6s 采集 %dx%d (ar=%.2f)  协商 %s fps  →  送出 %dx%d  %s"
              % (key, w, h, ar, ("%.1f" % float(fps)) if fps else "?", sent.shape[1], sent.shape[0],
                 "✓" if not tags else " ｜ ".join(tags)))
        if frozen:
            print("          （提示：0.35s 内两帧完全相同；场景/机械臂静止时属正常）")
    print("   动作间隔 dt = %.4fs (=1/15)" % DT)


# ------------------------------------------------------------------ 主流程
TRIAL_NOISE = 4.0   # flow matching 每次采样的固有随机量级（度）；跨试验差异要明显大于它才有意义


# ------------------------------------------------------------------ 录像（本地 mp4）
def to_bgr(img):
    """把相机帧统一成 HWC BGR uint8（兼容 CHW / float / RGB）。总是返回新数组，可安全跨线程传递。"""
    import cv2
    if hasattr(img, "detach"):
        img = img.detach().cpu().numpy()
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


class _VideoSink:
    """单个 mp4 写入器：后台线程 + 有界队列（宁可丢帧，也绝不阻塞控制环）。"""

    def __init__(self, path, fps, fourcc="mp4v", qmax=120):
        import queue as _q
        import threading as _t
        self.path, self.fps, self.fourcc = path, float(fps), fourcc
        self.q = _q.Queue(maxsize=qmax)
        self.n = 0
        self.dropped = 0
        self.err = None
        self._vw = None
        self._th = _t.Thread(target=self._run, daemon=True)
        self._th.start()

    def _run(self):
        import cv2
        try:
            while True:
                item = self.q.get()
                if item is None:
                    break
                if self._vw is None:
                    h, w = item.shape[:2]
                    self._vw = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*self.fourcc),
                                               self.fps, (w, h))
                    if not self._vw.isOpened():
                        self.err = "VideoWriter 打不开（fourcc=%s）" % self.fourcc
                        self._vw = None
                        self.dropped += 1
                        continue
                    print("   [录像] %s  %dx%d @ %.2f fps  fourcc=%s"
                          % (self.path, w, h, self.fps, self.fourcc))
                self._vw.write(item)
                self.n += 1
        except Exception as e:  # noqa: BLE001
            self.err = "%s: %s" % (type(e).__name__, e)
        finally:
            if self._vw is not None:
                try:
                    self._vw.release()
                except Exception:
                    pass

    def push(self, bgr):
        try:
            self.q.put_nowait(bgr)
        except Exception:  # noqa: BLE001
            self.dropped += 1

    def close(self):
        try:
            self.q.put(None, timeout=5)
        except Exception:  # noqa: BLE001
            pass
        self._th.join(timeout=120)
        print("   [录像] %s：写入 %d 帧%s%s"
              % (os.path.basename(self.path), self.n,
                 "" if not self.dropped else "，丢 %d 帧" % self.dropped,
                 "" if not self.err else "，错误: %s" % self.err))


class Recorder:
    """把相机画面录成本地 mp4。

    ★ 采样**独立于推理节奏**：前端相机会看到机械臂，如果只在每次推理时抓一帧（约 1.4s 一帧），
      录出来就是幻灯片。所以 chunk 执行期间**每一步**都从相机取一次最新帧。
    """

    def __init__(self, path, keys, dt, chunk, raw=False, fps=0.0, fourcc="mp4v", lat_ema=0.33):
        self.path, self.keys, self.raw, self.fourcc = path, list(keys), raw, fourcc
        if fps and fps > 0:
            self.fps = float(fps)
        else:                                   # auto：按 "dt + 推理延迟/chunk" 估步长，回放接近实时
            step = max(dt + max(lat_ema, 0.0) / max(chunk, 1), 1e-3)
            self.fps = min(max(1.0 / step, 1.0), 30.0)
        self.sinks = {}
        self.n = 0
        for k in self.keys:
            self.sinks[k] = _VideoSink(self._path_for(k), self.fps, fourcc)
        print("录像开启：%s → %s（%s，@%.2f fps）"
              % (",".join(self.keys), path, "相机原图" if raw else "送模型的 640×360 帧", self.fps))

    def _path_for(self, key):
        if len(self.keys) <= 1:
            return self.path
        root, ext = os.path.splitext(self.path)
        return "%s_%s%s" % (root, key, ext or ".mp4")

    def snap(self, robot, prepared=None):
        """取一帧并入库：优先从相机读**最新帧**（与真实动作同步）；失败则退回已准备好的帧。"""
        cams = getattr(robot, "cameras", None) or {}
        for k, snk in self.sinks.items():
            img = None
            cam = cams.get(k)
            if cam is not None:
                try:
                    img = cam.read_latest()
                except Exception:  # noqa: BLE001
                    img = None
            if img is None and prepared and k in prepared:
                img = prepared[k]
            if img is None:
                continue
            snk.push(to_bgr(img) if self.raw else to_bgr(prep_image(img)))
            self.n += 1

    def close(self):
        for snk in self.sinks.values():
            snk.close()
        print("录像结束：共 %d 帧（≈%.1f 秒 @%.2f fps）→ %s"
              % (self.n, self.n / max(self.fps, 1e-6), self.fps, self.path))


REC = None   # 全局录像器；--record-video 时创建


def _trial_summary(rows):
    if not rows:
        return
    P = np.array([r["pred"] for r in rows])
    S = np.array([r["st"] for r in rows])
    print("=" * 78)
    print("=== 汇总（%d 次试验）===" % len(rows))
    print("  %-6s %-48s %s" % ("trial", "预测首步 (pan lift elbow wflex wroll grip)", "夹爪"))
    for r in rows:
        arm = r["pred"][:-1]
        print("  %-6d [%s] %5.1f"
              % (r["k"], " ".join("%7.2f" % v for v in arm), float(r["pred"][-1])))
    rng_p = P.max(0) - P.min(0)
    rng_s = S.max(0) - S.min(0)
    print()
    print("  跨试验极差（同一关节 max-min，度）:")
    print("    预测首步  :", np.round(rng_p, 2), " -> 最大 %.2f" % rng_p.max())
    print("    当前关节角:", np.round(rng_s, 2), " -> 最大 %.2f  (%s)"
          % (rng_s.max(), "臂确实没动" if rng_s.max() < 2 else "臂动了，对照被污染"))
    print()
    print("  ★ 判读（参考噪声 ~%.0f 度）:" % TRIAL_NOISE)
    if rng_s.max() >= 2:
        print("    关节角也变了 %.1f 度，这次对照不干净（臂不该动）。" % rng_s.max())
    elif rng_p.max() > 3 * TRIAL_NOISE:
        print("    关节角几乎没变、预测极差 %.1f 度 >> 噪声 ⇒ **模型确实在看画面**。" % rng_p.max())
    else:
        print("    预测极差 %.1f 度与噪声同量级 ⇒ 这次改的场景没能改变预测，" % rng_p.max())
        print("    说明画面**几乎不影响输出**（或这个场景它本来就不在意）。")
    print("=" * 78)


def run_trials(robot, cli, a):
    """试验模式：每次按 Enter 触发**一次**单帧推理；臂默认保持不动。

    臂不动 ⇒ 关节角这个输入被冻住，唯一能变化的输入就是画面。
    于是"跨试验的预测极差"直接回答：模型到底在不在看画面。
    """
    n = a.trials
    print("=== 试验模式：%d 次单帧推理 ===" % n)
    print("  臂  : %s" % ("**会执行动作**（每次试验后请手动复位）" if a.trials_execute else "保持不动，只推理"))
    print("  流程: 摆好场景 → Enter → 看结果 → 改场景 → Enter → …  (Ctrl-C 提前结束)")
    print()
    rows = []
    csv_f = None
    if a.log_csv:
        d = os.path.dirname(os.path.abspath(a.log_csv))
        if d:
            os.makedirs(d, exist_ok=True)
        csv_f = open(a.log_csv, "w", encoding="utf-8")
        csv_f.write("trial,latency_s,payload_kb," + ",".join("state_" + m for m in MOTORS) + ","
                    + ",".join("pred_" + m for m in MOTORS)
                    + ",maxdiff_vs_state,maxdiff_vs_first,maxdiff_vs_prev\n")
    try:
        for k in range(1, n + 1):
            if k == 1:
                input(">>> 摆好场景后按 Enter 开始第 1/%d 次推理 … " % n)
            else:
                input(">>> 改变场景后按 Enter 进行第 %d/%d 次推理 … " % (k, n))
            obs = robot.get_observation()
            st = state_vec(obs)
            img_front = prep_image(obs["front"], "front")
            img_wrist = prep_image(obs["wrist"], "wrist")
            if a.dump_frames:
                import cv2 as _cv
                os.makedirs(a.dump_frames, exist_ok=True)
                _cv.imwrite(os.path.join(a.dump_frames, "t%02d_front.png" % k),
                            _cv.cvtColor(img_front, _cv.COLOR_RGB2BGR))
                _cv.imwrite(os.path.join(a.dump_frames, "t%02d_wrist.png" % k),
                            _cv.cvtColor(img_wrist, _cv.COLOR_RGB2BGR))
            obs_dict = {
                "video.front": to_wire(img_front, a.jpeg_quality),
                "video.wrist": to_wire(img_wrist, a.jpeg_quality),
                "state.single_arm": st[:5][None, :],
                "state.gripper": st[5:6][None, :],
                "annotation.human.task_description": [a.task],
            }
            t0 = time.time()
            res, nbytes = cli.get_action(obs_dict)
            lat = time.time() - t0
            sa = np.asarray(res["action.single_arm"], dtype=np.float64)
            sg = np.asarray(res["action.gripper"], dtype=np.float64)
            chunk = np.concatenate([sa.reshape(len(sa), -1), sg.reshape(len(sg), -1)], axis=1)
            first = chunk[0]
            dj = float(np.max(np.abs(first - st)))
            d1 = float(np.max(np.abs(first - rows[0]["pred"]))) if rows else float("nan")
            dp = float(np.max(np.abs(first - rows[-1]["pred"]))) if rows else float("nan")
            print("[试验 %2d/%d] 延迟 %.3fs  上行 %.0fKB" % (k, n, lat, nbytes / 1024.0))
            print("   当前关节角:", np.round(st, 2))
            print("   预测首步  :", np.round(first, 2))
            print("   首步变化  : %.2f 度(相对当前)    夹爪 %.1f -> %.1f" % (dj, st[5], first[5]))
            if rows:
                print("   与上次预测差 %.2f 度    与第 1 次差 %.2f 度" % (dp, d1))
            print("   预测前 3 步:\n%s" % np.array2string(chunk[:3], precision=2, suppress_small=True))
            print()
            rows.append(dict(k=k, st=st, pred=first, lat=lat, nb=nbytes, dj=dj, d1=d1, dp=dp))
            if REC is not None:
                REC.snap(robot, {"front": img_front, "wrist": img_wrist})
            if csv_f:
                csv_f.write("%d,%.4f,%.1f," % (k, lat, nbytes / 1024.0)
                            + ",".join("%.3f" % v for v in st) + ","
                            + ",".join("%.3f" % v for v in first) + ","
                            + "%.3f,%.3f,%.3f\n" % (dj,
                                                   0.0 if np.isnan(d1) else d1,
                                                   0.0 if np.isnan(dp) else dp))
                csv_f.flush()
            if a.trials_execute:
                m = int(min(a.chunk, len(chunk)))
                print("   -> 执行该 chunk 的 %d 步…" % m)
                for i in range(m):
                    robot.send_action(action_to_dict(chunk[i]))
                    time.sleep(a.dt)
    except KeyboardInterrupt:
        print("\n(手动中断)")
    finally:
        if csv_f:
            csv_f.close()
        if REC is not None:
            REC.close()
        try:
            robot.disconnect()
        except Exception:
            pass
        _trial_summary(rows)
    return 0


SETTLE_DEG = 1.5   # --settle-steps 判定"几乎不动"的阈值（度）

# ── ★ 客户端兜底：命令限幅 + 卡死自救（2026-10-16 加）─────────────────────────
# 为什么需要：离线实测模型的「状态纠正增益」∂T/∂s ≈ 1（0.83~1.10）⇒ 闭环里**没有任何回拉力**。
# ⇒ 一旦某个关节被物理挡住（例如 shoulder_lift 顶到行程端点、往桌子里压），
#   状态不变 ⇒ 模型输出也不变（它是无记忆映射 f(state,image)）⇒ **永久死锁**。
# 下面两件事分别对应「别让它往极限里压」和「万一压住了就自己退一步」。
# ★★★ 2026-10-20 13:2x **限幅必须"向内收"，绝不能向外放宽**（这里原来是个致命 bug）：
#   旧写法 `lo - margin, hi + margin` 把 shoulder_lift 放宽到 **−108.5°**，而物理极限只有 **−104.3°**
#   ⇒ 模型一旦给出"往下压"的命令，就被裁到 −108.5，**永远压在极限上、永远动不了**，
#     连"卡死自救"都救不回来（自救也是按这个范围判断的）。
#   真机日志实证（13:16 三次卡死 vs 13:15 一次成功）：
#     · 成功那次：act_lift −98.2 → 下一 chunk −89.0（能回升）→ 抬起到 +33°
#     · 卡死那次：act_lift −99.3 → 下一 chunk −104.8 → 之后永远 −108.5 / fb −104.0 ⇒ 冻结
#   ⇒ 现在：**边界向内收 margin**，并给 shoulder_lift 的下限单独收紧到 −101°（离物理极限还留 3.3°）。
JOINT_RANGE = {                    # 数据集 arm101_pick_all_objects（102 集）里各关节的实测行程（度）
    "shoulder_pan":  (-68.6,  49.2),
    "shoulder_lift": (-101.0, 70.5),   # ★ 数据集最小值是 −105.5，但 −104.3 就是"臂压到桌面"的物理极限
    "elbow_flex":    (-96.4,  97.1),
    "wrist_flex":    ( 34.0, 103.1),
    "wrist_roll":    (-74.6,  97.0),
    "gripper":       (  0.0,  50.9),
}
CLAMP_MARGIN = 1.0                 # ★ 向内收这么多度（安全余量），**绝不向外放宽**


def clamp_chunk(chunk, margin=CLAMP_MARGIN):
    """把命令限幅到「数据实测行程**向内收** margin」。返回 (限幅后的 chunk, 逐关节被限幅的步数)。"""
    out = np.array(chunk, dtype=np.float64, copy=True)
    hit = np.zeros(len(MOTORS), dtype=int)
    for j, m in enumerate(MOTORS):
        lo, hi = JOINT_RANGE[m]
        lo, hi = lo + margin, hi - margin      # ★ 向内收，不是向外放
        over = (out[:, j] < lo) | (out[:, j] > hi)
        hit[j] = int(over.sum())
        if hit[j]:
            out[:, j] = np.clip(out[:, j], lo, hi)
    return out, hit


def _new_recorder(a, k=0):
    """建一个录像器；k>0 时在文件名里插 _a01/_a02…（--record-split 用）。"""
    path = a.record_video
    if k:
        root, ext = os.path.splitext(path)
        path = "%s_a%02d%s" % (root, k, ext or ".mp4")
    return Recorder(path, [x.strip() for x in a.record_keys.split(",") if x.strip()],
                    a.dt, a.chunk, raw=a.record_raw, fps=a.record_fps, fourcc=a.record_fourcc)


def move_to_pose(robot, target, steps, name="", rec=None):
    """缓慢插值把机械臂摆到 target 位姿（摆位过程也会被录下来）。"""
    st0 = state_vec(robot.get_observation())
    print("缓慢移动到预设位姿%s（%d 步插值）… 当前 %s"
          % (" [%s]" % name if name else "", steps, np.round(st0, 2)))
    for i in range(1, steps + 1):
        robot.send_action(action_to_dict(st0 + (target - st0) * i / steps))
        if rec is not None:
            rec.snap(robot)
        time.sleep(0.05)
    time.sleep(0.8)
    st1 = state_vec(robot.get_observation())
    print("到位后关节角:", np.round(st1, 2), " 残余偏差最大 %.2f°" % float(np.abs(st1 - target).max()))
    return st1


def _attempt_summary(summ):
    if not summ:
        return
    print("=" * 78)
    print("=== %d 次抓取汇总 ===" % len(summ))
    print("  #    步数   用时s   夹爪min  夹爪max  lift 范围           结束位姿偏离演示起点  结束原因")
    for r in summ:
        dev = float(np.abs(r["st_end"] - DEMO_START).max())
        print("  %-3d  %5d  %6.1f  %8.1f %8.1f   %6.1f ~ %6.1f   %8.1f°            %s"
              % (r["k"], r["done"], r["secs"], r["gmin"], r["gmax"],
                 r["lmin"], r["lmax"], dev, r["reason"]))
    ok = [r for r in summ if r["gmin"] < 10]
    print()
    print("  夹爪闭合过（<10）: %d / %d   ← 粗略指标：它有没有真去夹" % (len(ok), len(summ)))
    print("  ★ 夹爪闭合 ≠ 夹住。有没有夹到，要看录像里笔是否跟着被抬起来。")
    print("=" * 78)


def run_attempts(robot, cli, a):
    """★ 实验模式：做 N 次**完整抓取**；每次结束停下，等你复位机械臂 + 摆好笔，按 Enter 再来。"""
    global REC
    n = a.attempts
    step_s = a.dt + 0.30 / max(a.chunk, 1)
    print("=== 实验模式：%d 次完整抓取 ===" % n)
    print("  每次 %d 步（约 %.0f 秒），跑完就停下" % (a.steps, a.steps * step_s))
    if a.pose_each_attempt and a.start_pose != "none":
        print("  每次开始前会自动摆到预设位姿 [%s]" % a.start_pose)
    if a.settle_steps:
        print("  连续累计 %d 步几乎不动（<%s°）就提前结束本次" % (a.settle_steps, SETTLE_DEG))
    print("  流程: [自动摆位] → 抓取 → 停下 → 你复位机械臂+摆好笔 → Enter → 下一次    (Ctrl-C 中止)")
    print()
    csv_f = None
    if a.log_csv:
        d = os.path.dirname(os.path.abspath(a.log_csv))
        if d:
            os.makedirs(d, exist_ok=True)
        csv_f = open(a.log_csv, "w", encoding="utf-8")
        csv_f.write("attempt,step,chunk,latency_s,payload_kb,"
                    + ",".join("state_" + m for m in MOTORS) + ","
                    + ",".join("act_" + m for m in MOTORS) + ","
                    + ",".join("fb_" + m for m in MOTORS) + "\n")
    summ = []
    J = (max(2, int(a.settle_steps / max(a.chunk, 1)) + 2) if a.settle_steps else 0)
    try:
        for k in range(1, n + 1):
            if k > 1:
                input(">>> 复位机械臂、摆好笔后，按 Enter 开始第 %d/%d 次抓取 … " % (k, n))
                if a.record_split and REC is not None:
                    REC.close()
                    REC = _new_recorder(a, k)
            if a.pose_each_attempt and a.start_pose != "none":
                move_to_pose(robot, POSE_PRESETS[a.start_pose], a.start_steps, a.start_pose, REC)
            done = n_infer = 0
            hist = []
            clamp_total = np.zeros(len(MOTORS), dtype=int)
            unstuck_used = 0
            reason = "跑满 %d 步" % a.steps
            t0 = time.time()
            while done < a.steps:
                obs = robot.get_observation()
                st = state_vec(obs)
                img_front = prep_image(obs["front"], "front")
                img_wrist = prep_image(obs["wrist"], "wrist")
                obs_dict = {
                    "video.front": to_wire(img_front, a.jpeg_quality),
                    "video.wrist": to_wire(img_wrist, a.jpeg_quality),
                    "state.single_arm": st[:5][None, :],
                    "state.gripper": st[5:6][None, :],
                    "annotation.human.task_description": [a.task],
                }
                t1 = time.time()
                res, nbytes = cli.get_action(obs_dict)
                lat = time.time() - t1
                n_infer += 1
                sa = np.asarray(res["action.single_arm"], dtype=np.float64)
                sg = np.asarray(res["action.gripper"], dtype=np.float64)
                chunk = np.concatenate([sa.reshape(len(sa), -1), sg.reshape(len(sg), -1)], axis=1)
                if a.joint_clamp:                      # ★ 兜底 1：别让它往行程极限里压
                    chunk, hit = clamp_chunk(chunk)
                    if hit.any():
                        clamp_total += hit
                m = int(min(a.chunk, len(chunk), a.steps - done))
                hist.append(st.copy())
                for i in range(m):
                    robot.send_action(action_to_dict(chunk[i]))
                    if REC is not None:
                        REC.snap(robot)
                    if csv_f:
                        fb = read_positions(robot)
                        csv_f.write("%d,%d,%d,%.4f,%.1f," % (k, done, n_infer, lat, nbytes / 1024.0)
                                    + ",".join("%.3f" % v for v in st) + ","
                                    + ",".join("%.3f" % v for v in chunk[i]) + ","
                                    + (",".join("%.3f" % v for v in fb) if fb is not None else
                                       ",".join([""] * len(MOTORS))) + "\n")
                        csv_f.flush()
                    time.sleep(a.dt)
                    done += 1
                if J and len(hist) >= J:
                    arr = np.array(hist[-J:])
                    if float(np.abs(arr - arr[0]).max()) < SETTLE_DEG:
                        # ★ 兜底 2：既然 ∂T/∂s≈1（状态不变⇒输出不变），干等是永远等不出来的。
                        #   先把「贴住行程极限」的关节拉开一点，让状态变一下，模型才有机会换决策。
                        if a.unstuck_deg > 0 and unstuck_used < a.unstuck_max:
                            unstuck_used += 1
                            cur = state_vec(robot.get_observation())
                            tgt = cur.copy()
                            for j, mm in enumerate(MOTORS[:5]):
                                lo, hi = JOINT_RANGE[mm]
                                if cur[j] <= lo + 6.0:          # 贴下限 ⇒ 往上抬
                                    tgt[j] = cur[j] + a.unstuck_deg
                                elif cur[j] >= hi - 6.0:        # 贴上限 ⇒ 往回收
                                    tgt[j] = cur[j] - a.unstuck_deg
                            d = tgt - cur
                            if float(np.abs(d).max()) > 1e-6:
                                print("   ⚠ 卡住了（连续 %d 次推理几乎不动）→ 自救 %d/%d："
                                      "把 %s 各拉开 %.0f° 后重试"
                                      % (J, unstuck_used, a.unstuck_max,
                                         ", ".join("%s%+.0f" % (MOTORS[j][:11], d[j])
                                                   for j in range(5) if abs(d[j]) > 1e-6),
                                         a.unstuck_deg))
                                move_to_pose(robot, tgt, 12, "unstuck", REC)
                                hist.clear()
                                continue
                        reason = "连续 %d 次推理几乎不动" % J
                        break
            el = time.time() - t0
            st_end = state_vec(robot.get_observation())
            gr = [h[5] for h in hist]
            lf = [h[1] for h in hist]
            print("[第 %2d/%d 次] 结束（%s）：执行 %d 步 / %.1f s，推理 %d 次"
                  % (k, n, reason, done, el, n_infer))
            if clamp_total.any():
                print("    ⚠ 行程限幅生效：%s（被裁的步数）——说明模型想把关节推出数据范围"
                      % ", ".join("%s %d" % (MOTORS[j][:11], int(clamp_total[j]))
                                  for j in range(len(MOTORS)) if clamp_total[j]))
            print("    起始 %s" % np.round(hist[0], 2))
            print("    结束 %s" % np.round(st_end, 2))
            print("    夹爪 %.1f ~ %.1f %s   |   shoulder_lift %.1f ~ %.1f"
                  % (min(gr), max(gr), "（闭过）" if min(gr) < 10 else "（从未闭合）", min(lf), max(lf)))
            print()
            summ.append(dict(k=k, done=done, secs=el, gmin=min(gr), gmax=max(gr),
                             lmin=min(lf), lmax=max(lf), st_end=st_end, reason=reason))
    except KeyboardInterrupt:
        print("\n(手动中止)")
    finally:
        if csv_f:
            csv_f.close()
        if REC is not None:
            REC.close()
        try:
            robot.disconnect()
        except Exception:
            pass
        _attempt_summary(summ)
    return 0


def main():
    global REC
    ap = argparse.ArgumentParser(description="官方 GR00T 真机客户端（SO-101）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--robot-port", default="COM3")
    ap.add_argument("--robot-id", default="my_awesome_follower_arm")
    ap.add_argument("--front-idx", type=int, default=1)
    ap.add_argument("--wrist-idx", type=int, default=0)
    ap.add_argument("--front-size", default="640x360", help="前置相机**采集**分辨率（原生）")
    ap.add_argument("--wrist-size", default="1280x720", help="腕部相机**采集**分辨率（原生）")
    ap.add_argument("--cam-fps", type=int, default=CAM_FPS)
    ap.add_argument("--task", default=TASK_DEFAULT)
    ap.add_argument("--jpeg-quality", type=int, default=90, help="上行 JPEG 质量，0=发原图")
    ap.add_argument("--chunk", type=int, default=16, help="一次推理执行多少个动作（训练 horizon=16）")
    ap.add_argument("--dt", type=float, default=DT, help="单个动作间隔，默认 1/15 与采集一致")
    ap.add_argument("--steps", type=int, default=300,
                    help="★ 跑多少步（1 步 = dt 秒）。**按数据集定**：本任务 trim 后每集 268~745 帧"
                         "（中位 486≈32s，p90 626≈42s，最长 745≈50s）⇒ 想跑完整任务用 --steps 800；"
                         "300 只有 20 秒，比最短的演示还短，只够看「动不动得起来」。"
                         "★ 填 0 或负数 = **不限时间**，一直跑到 Ctrl-C")
    ap.add_argument("--dry-run", action="store_true", help="★ 只推理打印，不下发动作")
    ap.add_argument("--no-calibrate", action="store_true")
    ap.add_argument("--no-check", action="store_true", help="跳过启动自检")
    ap.add_argument("--max-rel", type=float, default=15.0,
                    help="★ 每步最大关节变化（度/步），默认 15（= 225°/s @15Hz）。"
                         "演示里每帧最大关节变化：中位 1.93 / p95 5.89 / p99 7.74 / p99.9 9.67 / **最大 12.57** "
                         "⇒ 15 完全不截断演示的动作，只挡住离谱跳变。"
                         "想更保守用 8（会截断约 0.75%% 的帧）；想完全不限速传 0")
    ap.add_argument("--log-csv", default=None)
    ap.add_argument("--dump-frames", default=None,
                    help="把每次送给模型的 640×360 帧存成 PNG（排查『模型到底看到了什么』）")
    ap.add_argument("--dump-every", type=int, default=1, help="每 N 次推理存一次图")
    ap.add_argument("--trials", type=int, default=0,
                    help="★ 试验模式：做 N 次单帧推理，每次之间等你按 Enter（臂默认不动，只推理）")
    ap.add_argument("--trials-execute", action="store_true",
                    help="试验模式里真的执行 chunk（臂会动，每次试验后需手动复位）；默认不执行")
    ap.add_argument("--record-video", default=None,
                    help="★ 把画面录成本地 mp4（默认录第三视角 front）——不用再举手机拍")
    ap.add_argument("--record-keys", default="front", help="录哪几个相机，逗号分隔（默认 front）")
    ap.add_argument("--record-fps", type=float, default=0.0,
                    help="录像帧率；0=自动（按 dt+推理延迟估算，回放接近实时）")
    ap.add_argument("--record-raw", action="store_true",
                    help="录相机原图（默认录『送进模型的那一帧』640×360）")
    ap.add_argument("--record-fourcc", default="mp4v", help="cv2 编码器：mp4v（默认）/ XVID / MJPG")
    ap.add_argument("--attempts", type=int, default=0,
                    help="★ 实验模式：做 N 次【完整抓取】，每次结束停下等你复位+摆笔，按 Enter 再来一次")
    ap.add_argument("--pose-each-attempt", action="store_true",
                    help="每次实验开始前自动摆到 --start-pose 位姿（省得手动复位机械臂）")
    ap.add_argument("--settle-steps", type=int, default=0,
                    help="连续多少步几乎不动就提前结束本次（0=关闭，按 --steps 跑满）")
    ap.add_argument("--joint-clamp", action="store_true",
                    help="【默认关闭】把命令裁到「数据实测行程向内收 1°」（shoulder_lift 下限 −101°）。"
                         "★ 2026-10-20 13:3x 按用户要求默认关闭：它会在模型正常想走低位时改掉命令，"
                         "只用 --max-rel 限速 + 卡死自救就够了。需要时显式加这个开关")
    ap.add_argument("--no-joint-clamp", action="store_true", help=argparse.SUPPRESS)  # 兼容旧命令，已无效
    ap.add_argument("--unstuck-deg", type=float, default=8.0,
                    help="卡住时自动把贴住极限的关节拉开这么多度再重试（0=关闭）")
    ap.add_argument("--unstuck-max", type=int, default=2, help="每次尝试最多自救几次")
    ap.add_argument("--record-split", action="store_true",
                    help="录像按每次实验切成单独文件（xxx_a01.mp4, a02, …）")
    ap.add_argument("--start-pose", choices=["none"] + list(POSE_PRESETS.keys()), default="none",
                    help="★ 默认 none = **完全不摆位**（模型从臂当前实际位置直接开始，不预先抬升）。"
                         "只有想复现某个特定起点时才用：off10/off15=演示第 10/15 帧位姿（低位，真机可能卡）、"
                         "demo=演示首帧(已证伪) / mid=分布外(别用)；"
                         "旧任务(抓笔, v1 模型)：pen_demo / pen_t080 / pen_t100 / pen_t120。"
                         "⚠️ 预设不能跨任务用")
    ap.add_argument("--start-steps", type=int, default=30, help="摆位插值步数")
    ap.add_argument("--place-guide", action="store_true",
                    help="★ 摆放引导：把臂依次摆到 3 个真实演示的『合爪姿势』停住，"
                         "你把物体放到夹爪正下方即可（不用量距离）；摆完自动退出，不跑推理")
    ap.add_argument("--black-cams", default="",
                    help="把某路相机整帧置黑做视觉消融（真机 A/B）：front / wrist / front,wrist；默认空=正常")
    a = ap.parse_args()
    a.front_size = parse_size(a.front_size)
    a.wrist_size = parse_size(a.wrist_size)
    a.joint_clamp = bool(a.joint_clamp)      # ★ 默认 False：不加绝对边界限位（用户 13:3x 要求）

    global BLACK_CAMS
    BLACK_CAMS = {x.strip() for x in a.black_cams.split(",") if x.strip()}

    # ★ 防覆盖：同名日志/录像已存在时自动加时间戳，别把上一轮的数据悄悄冲掉
    #   （踩过：第一次"能自己动"的 run1.csv 被第二次运行覆盖，关键证据丢了）
    a.log_csv = _no_clobber(a.log_csv)
    a.record_video = _no_clobber(a.record_video)

    print("=== 官方 GR00T 真机客户端 ===")
    print("  服务    : %s:%d   JPEG 上行 q=%s" % (a.host, a.port, a.jpeg_quality or "关闭(原图)"))
    print("  机械臂  : SO101Follower port=%s id=%s" % (a.robot_port, a.robot_id))
    print("  相机    : front=idx%d@%dx%d  wrist=idx%d@%dx%d  ->  喂模型统一 %dx%d"
          % (a.front_idx, a.front_size[0], a.front_size[1],
             a.wrist_idx, a.wrist_size[0], a.wrist_size[1], CAM_W, CAM_H))
    if BLACK_CAMS:
        print("  ★ 视觉消融: 置黑 %s（其余相机正常）" % ",".join(sorted(BLACK_CAMS)))
    print("  任务    : %s" % a.task)
    print("  控制    : chunk=%d  dt=%.4fs (=1/15)  模式=%s"
          % (a.chunk, a.dt,
             ("实验模式 %d 次完整抓取（每次之间等你复位）" % a.attempts) if a.attempts > 0 else
             (("试验模式 %d 次（%s）" % (a.trials, "会执行动作" if a.trials_execute else "臂保持不动"))
              if a.trials > 0 else ("DRY-RUN（不下发动作）" if a.dry_run else "真机执行"))))
    if a.record_video:
        print("  录像    : %s ← %s（%s）" % (a.record_video, a.record_keys,
                                            "相机原图" if a.record_raw else "送模型的 640×360 帧"))
    print()

    cli = ZmqPolicyClient(a.host, a.port)
    try:
        print("ping 云端服务 …", cli.ping())
    except Exception as e:  # noqa: BLE001
        # ★ 试着自动换端口：最常见的原因不是服务挂了，而是**本机隧道开在别的端口**
        #   （本项目隧道惯例是 15555，而客户端默认 5555 → 一跑就报 Resource temporarily unavailable）
        alt = [p for p in (15555, 5555, 5556, 15556) if p != a.port]
        alive = []
        for p in alt:
            try:
                c2 = ZmqPolicyClient(a.host, p, timeout_ms=3000)
                c2.ping()
                alive.append(p)
            except Exception:  # noqa: BLE001
                pass
        if alive:
            print("!! %d 端口连不上（%s），但 **%s 端口是通的** → 自动改用 %d"
                  % (a.port, e, "/".join(str(p) for p in alive), alive[0]))
            print("   （下次直接加 --port %d 就不会再看到这条）" % alive[0])
            a.port = alive[0]
            cli = ZmqPolicyClient(a.host, a.port)
            print("ping 云端服务 …", cli.ping())
        else:
            print("!! 连不上策略服务：%s" % e)
            print("   ① 云端 screen grootpolicy 在跑？  ② 本机隧道开着？")
            print("   C:\\Users\\Fakun\\.workbuddy\\binaries\\python\\envs\\lerobot-win\\Scripts\\python.exe "
                  "scripts\\ssh_tunnel.py --local 15555 --remote 127.0.0.1:5555")
            print("   ★ 然后客户端要带 --port 15555（默认 5555 是本机直连，不是隧道口）")
            return 2

    robot = build_robot(a, strict=True)
    if a.max_rel and a.max_rel > 0:          # ★ 0/None = 不限速（别传 0 当"关"，0 会让臂完全不动）
        robot.config.max_relative_target = a.max_rel
        print("已设置 max_relative_target = %s 度/步（≈ %.0f°/s @15Hz）" % (a.max_rel, a.max_rel * 15))
    else:
        print("未设 max_relative_target（不限速）")
    try:
        robot.connect(calibrate=not a.no_calibrate)
    except RuntimeError as e:
        print("\n⚠ 相机不接受设定的采集分辨率/帧率：%s" % e)
        print("   → 退回用相机默认参数，由脚本在软件里 resize 到 %dx%d（结果一致，只是采集几何要自检确认）\n"
              % (CAM_W, CAM_H))
        robot = build_robot(a, strict=False)
        if a.max_rel and a.max_rel > 0:
            robot.config.max_relative_target = a.max_rel
        robot.connect(calibrate=not a.no_calibrate)
    print("机械臂已连接\n")

    # ---------------- ★「摆放引导」模式：让臂自己当尺子 ----------------
    if a.place_guide:
        print("=" * 78)
        print("摆放引导：机械臂会依次摆到 3 个【真实演示在合爪那一刻】的姿势并停住。")
        print("每个姿势停稳后，把一件物体放到**夹爪正下方、轻轻贴着夹爪**，然后按 Enter。")
        print("（不用量距离 —— 夹爪停在哪，物体就放哪。）")
        print("★ 摆位过程臂会动，注意别让手挡着；随时可按 Ctrl-C 中止。")
        print("=" * 78)
        steps = max(a.start_steps, 60)      # 慢一点、别甩
        for n, key in enumerate(GUIDE_ORDER, 1):
            print("\n--- 第 %d/3 个位置 [%s] ---" % (n, key))
            # ★ 先把臂抬到 off15（离开桌面）横向移动，避免在桌面高度扫过已放好的物体
            move_to_pose(robot, POSE_PRESETS["off15"], steps, "off15", None)
            move_to_pose(robot, POSE_PRESETS[key], steps, key, None)
            try:
                input("   ★ 请把物体放到夹爪正下方并贴住夹爪；放好后按 Enter 继续…")
            except EOFError:
                print("   （非交互终端，直接继续）")
        # 摆位完后抬臂待命，避免夹爪压在物体上
        print("\n--- 收尾：把臂抬到待命位姿 off15（离开桌面，不碰物体）---")
        move_to_pose(robot, POSE_PRESETS["off15"], steps, "off15", None)
        print("\n摆放引导完成 ✓  现在可以跑推理了（用 --start-pose off10）。")
        try:
            robot.disconnect()
        except Exception:  # noqa: BLE001
            pass
        return 0

    if a.record_video:
        REC = _new_recorder(a, 1 if (a.attempts > 0 and a.record_split) else 0)

    st0 = state_vec(robot.get_observation())
    dev = st0 - DEMO_START
    print("当前关节角 :", np.round(st0, 2))
    print("本任务起始位姿:", DEMO_START, " (102 条演示的【首帧】均值)")
    print("偏差       :", np.round(dev, 2), "  最大 %.2f°" % float(np.abs(dev).max()))
    if float(np.abs(dev).max()) > 25:
        print("⚠ 偏差 >25°：模型**没见过**这个起始姿势，很容易抓空。"
              "\n   建议加 `--start-pose off10` 自动摆到「演示第 10 帧」位姿（离桌面 16°，推荐）。")
    print("★ 2026-10-15 结论：停在**演示首帧**那个位姿（臂平放桌面）模型第一步会把 shoulder_lift 往桌里压 7°（真机 −10°），"
          "\n   压不下去 ⇒ 状态不变 ⇒ 永久卡死。**请用 --start-pose off10**（第 10 帧位姿，方向已正确）；"
          "\n   想留更大余量可用 --start-pose off15（第 15 帧，离桌面约 41°）。")
    print()

    if a.start_pose != "none":
        move_to_pose(robot, POSE_PRESETS[a.start_pose], a.start_steps, a.start_pose, REC)
        print()

    if not a.no_check:
        alignment_check(robot)
        print()

    csv_f = None
    if a.log_csv and a.trials <= 0 and a.attempts <= 0:
        d = os.path.dirname(os.path.abspath(a.log_csv))
        if d:
            os.makedirs(d, exist_ok=True)
        csv_f = open(a.log_csv, "w", encoding="utf-8")
        csv_f.write("step,chunk,latency_s,payload_kb," + ",".join("state_" + m for m in MOTORS) + ","
                    + ",".join("act_" + m for m in MOTORS) + ","
                    + ",".join("fb_" + m for m in MOTORS) + "\n")

    if a.attempts > 0:
        return run_attempts(robot, cli, a)
    if a.trials > 0:
        return run_trials(robot, cli, a)

    done = n_infer = 0
    lat_sum = payload_sum = 0.0
    # ★ 兜底（默认执行路径也要有 —— 之前只加在 --trials 分支里，等于没生效）
    clamp_total = np.zeros(len(MOTORS), dtype=int)
    settle_hist = []
    unstuck_used = 0
    JS = (max(2, int(a.settle_steps / max(a.chunk, 1)) + 2) if a.settle_steps else 0)
    t_start = time.time()
    if a.steps <= 0:
        print("步数限制 : 不限（跑到达成或 Ctrl-C 为止）")
    try:
        while a.steps <= 0 or done < a.steps:
            obs = robot.get_observation()
            st = state_vec(obs)
            img_front = prep_image(obs["front"], "front")
            img_wrist = prep_image(obs["wrist"], "wrist")
            if a.dump_frames and (n_infer % max(1, a.dump_every) == 0):
                import cv2 as _cv
                os.makedirs(a.dump_frames, exist_ok=True)
                pred = lambda n: os.path.join(a.dump_frames, n)
                _cv.imwrite(pred("i%03d_front.png" % n_infer), _cv.cvtColor(img_front, _cv.COLOR_RGB2BGR))
                _cv.imwrite(pred("i%03d_wrist.png" % n_infer), _cv.cvtColor(img_wrist, _cv.COLOR_RGB2BGR))
                if n_infer == 0:      # 首帧顺便存**原始采集尺寸**，用来核对取景/几何
                    for _k in ("front", "wrist"):
                        _raw = np.asarray(obs[_k])
                        if _raw.ndim == 3 and _raw.shape[0] == 3:
                            _raw = _raw.transpose(1, 2, 0)
                        _cv.imwrite(pred("i000_raw_%s.png" % _k), _cv.cvtColor(_raw.astype(np.uint8), _cv.COLOR_RGB2BGR))
                else:
                    pass
                with open(os.path.join(a.dump_frames, "states.csv"), "a", encoding="utf-8") as _f:
                    _f.write("%d," % n_infer + ",".join("%.2f" % v for v in st) + "\n")
            obs_dict = {
                "video.front": to_wire(img_front, a.jpeg_quality),
                "video.wrist": to_wire(img_wrist, a.jpeg_quality),
                "state.single_arm": st[:5][None, :],
                "state.gripper": st[5:6][None, :],
                "annotation.human.task_description": [a.task],
            }
            t0 = time.time()
            res, nbytes = cli.get_action(obs_dict)
            lat = time.time() - t0
            n_infer += 1
            lat_sum += lat
            payload_sum += nbytes

            sa = np.asarray(res["action.single_arm"], dtype=np.float64)
            sg = np.asarray(res["action.gripper"], dtype=np.float64)
            chunk = np.concatenate([sa.reshape(len(sa), -1), sg.reshape(len(sg), -1)], axis=1)
            if a.joint_clamp:                     # ★ 兜底 1：别让命令越过数据实测行程（防"往桌里压"）
                chunk, hit = clamp_chunk(chunk)
                if hit.any():
                    clamp_total += hit
            n = int(min(a.chunk, len(chunk))) if a.steps <= 0 else int(min(a.chunk, len(chunk), a.steps - done))
            jump = float(np.max(np.abs(chunk[0] - st)))
            print("[推理#%d] 延迟 %.3fs  上行 %.0fKB  chunk=%s  首步最大变化 %.2f°  夹爪 %.1f → %.1f"
                  % (n_infer, lat, nbytes / 1024.0, chunk.shape, jump, st[5], chunk[0, 5]))

            if a.dry_run:
                print("   预测前 3 步:\n%s" % np.array2string(chunk[:3], precision=2, suppress_small=True))
                if REC is not None:
                    # ★ dry-run 也必须录：否则"臂不动、只看模型想干嘛"这一段是空的。
                    #   每个 chunk 补 n 帧相同画面，回放速度才和真实执行一致（不是幻灯片快进）。
                    for _ in range(n):
                        REC.snap(robot)
                done += n
                continue

            for i in range(n):
                robot.send_action(action_to_dict(chunk[i]))
                if REC is not None:
                    REC.snap(robot)          # ★ 每步都抓，录制帧率与真实动作同步
                if csv_f:
                    fb = read_positions(robot)
                    csv_f.write("%d,%d,%.4f,%.1f," % (done, n_infer, lat, nbytes / 1024.0)
                                + ",".join("%.3f" % v for v in st) + ","
                                + ",".join("%.3f" % v for v in chunk[i]) + ","
                                + (",".join("%.3f" % v for v in fb) if fb is not None else
                                   ",".join([""] * len(MOTORS))) + "\n")
                    csv_f.flush()
                time.sleep(a.dt)
                done += 1

            # ★ 兜底 2：卡死自救。∂T/∂s≈1 ⇒ 状态不变就输出不变 ⇒ 干等永远等不出来，必须让状态先动一下
            settle_hist.append(st.copy())
            if JS and len(settle_hist) >= JS:
                arr = np.array(settle_hist[-JS:])
                if float(np.abs(arr - arr[0]).max()) < SETTLE_DEG:
                    if a.unstuck_deg > 0 and unstuck_used < a.unstuck_max:
                        unstuck_used += 1
                        cur = state_vec(robot.get_observation())
                        tgt = cur.copy()
                        for j, mm in enumerate(MOTORS[:5]):
                            lo, hi = JOINT_RANGE[mm]
                            if cur[j] <= lo + 6.0:            # 贴下限 ⇒ 往上抬
                                tgt[j] = cur[j] + a.unstuck_deg
                            elif cur[j] >= hi - 6.0:          # 贴上限 ⇒ 往回收
                                tgt[j] = cur[j] - a.unstuck_deg
                        d = tgt - cur
                        if float(np.abs(d).max()) > 1e-6:
                            print("   ⚠ 卡住了（连续 %d 个 chunk 几乎不动）→ 自救 %d/%d：把 %s 拉开 %.0f° 后重试"
                                  % (JS, unstuck_used, a.unstuck_max,
                                     ", ".join("%s%+.0f" % (MOTORS[j][:11], d[j])
                                               for j in range(5) if abs(d[j]) > 1e-6),
                                     a.unstuck_deg))
                            move_to_pose(robot, tgt, 12, "unstuck", REC)
                            settle_hist.clear()
                            continue
                    print("   ✗ 连续 %d 个 chunk 几乎不动，且自救次数已用完 → 结束本次" % JS)
                    break
    except KeyboardInterrupt:
        print("\n(手动中断)")
    finally:
        if csv_f:
            csv_f.close()
        if REC is not None:
            REC.close()
        try:
            robot.disconnect()
        except Exception:
            pass
        dt = time.time() - t_start
        print("\n结束：执行 %d 个动作、推理 %d 次，用时 %.1fs" % (done, n_infer, dt))
        if n_infer:
            print("      平均推理延迟 %.3fs、平均上行 %.0fKB"
                  % (lat_sum / n_infer, payload_sum / n_infer / 1024.0))
        if a.dry_run:
            print("      (dry-run：未下发动作，所以没有『动作/秒』可看)")
        elif done:
            print("      实际动作速率 %.2f 动作/秒（目标 15）" % (done / max(dt, 1e-9)))
        if clamp_total.any():
            print("      行程限幅生效：%s（被裁的步数）——说明模型想把关节推出数据范围"
                  % ", ".join("%s %d" % (MOTORS[j][:11], int(clamp_total[j]))
                              for j in range(len(MOTORS)) if clamp_total[j]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
