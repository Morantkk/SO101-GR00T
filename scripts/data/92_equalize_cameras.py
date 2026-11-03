#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把转换后数据集的各相机视频统一到同一分辨率（官方 Isaac-GR00T 的硬性要求）。

为什么必须统一：
  gr00t/data/transform/video.py 的 VideoCrop.get_transform() 里
      assert len(set(self.original_resolutions.values())) == 1
  这不是一句多余的断言 —— VideoTransform.apply() 会把**所有相机视图**
      views = np.concatenate([data[k] for k in apply_to], 0)
  拼成一个张量后**统一**裁剪/缩放/变色，所以各相机必须同 H、W。
  官方 SO-100 用的是两个同型号相机，天然满足；我们 front 640x360 / wrist 1280x720。

**不丢画面 / 不降采样**的做法：统一到更高的 1280x720
  · wrist 已是 1280x720 → 完全不动（一个字节都不重编码）
  · front 640x360    → 双线性插值放大到 1280x720（内容一个不少，只是像素变多）
视野 FOV、宽高比（都是 16:9）都不变；真正会裁掉画面的只有官方增强里的
VideoCrop(scale=0.95)（那是 NVIDIA 官方配方，保留）。

只改**派生**数据集（/root/gr00t_data/...），源数据集 /root/arm101_pick_one_pen 不动。

用法:
  python 92_equalize_cameras.py [DATASET_DIR] [--target 1280x720] [--crf 20]
"""
import argparse
import glob
import json
import os
import shutil
import time
from fractions import Fraction

import av


def probe_size(path):
    with av.open(path) as c:
        s = c.streams.video[0]
        return int(s.width), int(s.height), int(c.streams.video[0].average_rate or 15)


def reencode(path, out_path, W, H, fps, crf, enc="libx264"):
    with av.open(path) as ic:
        ist = ic.streams.video[0]
        with av.open(out_path, mode="w", format="mp4") as oc:
            ost = oc.add_stream(enc, rate=fps)
            ost.width, ost.height = W, H
            ost.pix_fmt = "yuv420p"
            if enc == "libx264":
                ost.options = {"crf": str(crf), "preset": "veryfast"}
            n = 0
            for frame in ic.decode(ist):
                frame = frame.reformat(width=W, height=H, format="yuv420p")
                frame.pts = n
                frame.time_base = Fraction(1, fps)
                for pkt in ost.encode(frame):
                    oc.mux(pkt)
                n += 1
            for pkt in ost.encode():
                oc.mux(pkt)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", default="/root/gr00t_data/arm101_pick_one_pen")
    ap.add_argument("--target", default="1280x720", help="目标分辨率 WxH")
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--encoder", default="libx264")
    o = ap.parse_args()

    W, H = (int(x) for x in o.target.lower().split("x"))
    ds = o.dataset
    info_path = os.path.join(ds, "meta", "info.json")
    info = json.load(open(info_path, encoding="utf-8"))
    fps = int(float(info.get("fps", 15)) or 15)
    cams = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    print("目标 %dx%d @%dfps  crf=%d  相机=%s" % (W, H, fps, o.crf, cams))

    for cam in cams:
        files = sorted(glob.glob(os.path.join(ds, "videos", "chunk-*", cam, "*.mp4")))
        if not files:
            print("  ⚠ %s 没找到视频，跳过" % cam)
            continue
        w0, h0, _ = probe_size(files[0])
        if (w0, h0) == (W, H):
            print("  %-24s 已是 %dx%d，原样不动（%d 个文件）" % (cam, w0, h0, len(files)))
            continue
        print("  %-24s %dx%d -> %dx%d：重编码 %d 个文件" % (cam, w0, h0, W, H, len(files)))
        t0 = time.time()
        nframes = 0
        for i, f in enumerate(files, 1):
            tmp = f + ".tmp.mp4"
            nframes += reencode(f, tmp, W, H, fps, o.crf, o.encoder)
            os.replace(tmp, f)
            if i % 10 == 0 or i == len(files):
                print("     %d/%d  (%.1fs)" % (i, len(files), time.time() - t0))
        print("     共 %d 帧，用时 %.1fs" % (nframes, time.time() - t0))

        # 更新 info.json 里的 shape / info.*height|width / codec
        feat = info["features"][cam]
        names = feat.get("names", [])
        if "height" in names:
            feat["shape"][names.index("height")] = H
        if "width" in names:
            feat["shape"][names.index("width")] = W
        sub = feat.get("info")
        if isinstance(sub, dict):
            for k in list(sub):
                if k.endswith("height"):
                    sub[k] = H
                elif k.endswith("width"):
                    sub[k] = W
                elif k.endswith("codec"):
                    sub[k] = "h264"

    json.dump(info, open(info_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("info.json 已更新")

    print("\n=== 复核 ===")
    ok = True
    for cam in cams:
        f = sorted(glob.glob(os.path.join(ds, "videos", "chunk-*", cam, "*.mp4")))[0]
        w, h, fr = probe_size(f)
        print("  %-24s %dx%d @%s  shape=%s" % (cam, w, h, fr, info["features"][cam]["shape"]))
        if (w, h) != (W, H):
            ok = False
    print("统一完成" if ok else "⚠ 仍有相机分辨率不一致")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
