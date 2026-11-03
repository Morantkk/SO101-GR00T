#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""拍下"推理时实际喂给模型的两路相机画面"，存成 PNG，供肉眼比对。

为什么需要它：
  策略"能朝笔移动、但最后不往下探/不抓"时，第一嫌疑是**腕部(wrist)相机**这条链路
  （最后的对准/下探主要靠腕部近距离视野）。用它确认：
    ① 索引有没有装反（front / wrist 各自拍到了什么）；
    ② wrist 画面是否正常（不是全黑/过曝/糊/被遮挡/角度和训练时不同）；
    ③ 分辨率与训练一致（front 640x360、wrist 1280x720）。

用 lerobot 自己的相机驱动（与训练/推理**同一套**，所以颜色顺序、缩放都一致）。

用法（**机械臂空闲时**跑，推理进程会占用相机）：
  C:/Users/Fakun/.workbuddy/binaries/python/envs/lerobot-win/Scripts/python.exe scripts/snap_cameras.py
输出：
  outputs/cam_snaps/{front,wrist}_*.png
"""
import argparse
import os
import sys

# 与推理/训练完全一致的相机配置
CAMS = [
    ("front", dict(index_or_path=1, width=640, height=360, fps=15)),
    ("wrist", dict(index_or_path=0, width=1280, height=720, fps=15)),
]


def main():
    ap = argparse.ArgumentParser(description="拍下推理用的两路相机画面")
    ap.add_argument("--out", default=os.path.join("outputs", "cam_snaps"))
    ap.add_argument("--frames", type=int, default=3)
    o = ap.parse_args()

    import numpy as np
    from PIL import Image
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

    os.makedirs(o.out, exist_ok=True)
    print("=== 相机快照 -> %s ===" % o.out)
    print("  （注意：若推理进程正在跑，相机会被占用，这里会连不上）\n")

    rc = 0
    for name, kw in CAMS:
        cfg = OpenCVCameraConfig(**kw)
        cam = OpenCVCamera(cfg)
        try:
            cam.connect()
            for _ in range(5):            # 丢掉前几帧，等自动曝光稳定
                cam.read()
            for i in range(o.frames):
                arr = np.asarray(cam.read())
                path = os.path.join(o.out, "%s_%d.png" % (name, i))
                Image.fromarray(arr).save(path)
                print("  %-6s idx=%d -> shape=%s(H,W,C) mean=%.1f min=%d max=%d  %s"
                      % (name, kw["index_or_path"], arr.shape, arr.mean(), arr.min(), arr.max(),
                         os.path.basename(path)))
        except Exception as e:  # noqa: BLE001
            rc = 1
            print("  !! %s (idx=%d) 打不开: %s: %s" % (name, kw["index_or_path"], type(e).__name__, e))
            print("     -> 相机被别的进程占用？先用 Ctrl-C 停掉推理再试。")
        finally:
            try:
                cam.disconnect()
            except Exception:  # noqa: BLE001
                pass

    print("\n打开这些 PNG 看看：")
    print("  · 文件名 front_* 应该是**外部/俯视**视角；wrist_* 应该是**夹爪附近的近距离**视角。")
    print("  · 若两者内容对调 -> 相机索引装反（改脚本 CAMS 里的 index_or_path）。")
    print("  · 若 wrist 全黑/全白/糊 -> 腕部相机线/曝光/位置问题，这会直接导致「不往下探」。")
    print("  · 把 wrist_0.png 和训练集里的腕部画面比一比（姿态、朝向、视野范围是否一致）。")
    return rc


if __name__ == "__main__":
    sys.exit(main())
