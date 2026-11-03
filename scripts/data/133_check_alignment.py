#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""133_check_alignment.py —— 检查「图像有没有比关节状态超前」+「开头有没有残留静止帧」。

关键：GR00T 取帧是 `argmin |视频帧时间 − parquet 的 timestamp|`。
  · 若 timestamp 已归零 ⇒ 每行取到的就是同一帧 ⇒ 超前量 0。
  · 若裁剪时忘了改 timestamp ⇒ 超前量 = 该集被裁掉的帧数（旧 trim 就是这样，+46~124 帧）。

用法：
  python 133_check_alignment.py --ds /root/gr00t_data/arm101_pick_all_objects_trim2 --sample 0
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--sample", type=int, default=0, help="0=全部；>0 只抽查这么多集")
    o = ap.parse_args()

    import av
    info = json.load(open(os.path.join(o.ds, "meta", "info.json"), encoding="utf-8"))
    CH = int(info.get("chunks_size", 1000))
    fps = float(info.get("fps", 15))
    eps = [json.loads(l) for l in open(os.path.join(o.ds, "meta", "episodes.jsonl"),
                                       encoding="utf-8") if l.strip()]
    ids = [int(r["episode_index"]) for r in eps]
    if o.sample:
        ids = ids[:: max(1, len(ids) // o.sample)]

    leads, stats, rows = [], [], []
    for e in ids:
        ck = e // CH
        df = pd.read_parquet(os.path.join(o.ds, "data", "chunk-%03d" % ck, "file-%03d.parquet" % e))
        st = np.asarray(list(df["observation.state"]), dtype=np.float32)
        v = os.path.join(o.ds, "videos", "chunk-%03d" % ck, "observation.images.front", "file-%03d.mp4" % e)
        with av.open(v) as c:
            n = sum(1 for _ in c.decode(video=0))
        ts = np.arange(n) / fps
        idx = np.abs(ts[None, :] - df["timestamp"].to_numpy()[:, None]).argmin(axis=1)
        d = idx - np.arange(len(df))
        leads.append(int(np.median(d)))
        sp = np.abs(np.diff(st[:, :5], axis=0)).max(axis=1)
        k = 0
        while k < len(sp) and sp[k] < 0.3:
            k += 1
        stats.append(k)
        rows.append(len(df))

    leads = np.array(leads)
    stats = np.array(stats)
    print("检查 %d 集" % len(ids))
    print("图像超前量（帧）: 唯一值 %s  中位 %d  max %d" % (np.unique(leads)[:5], int(np.median(leads)), leads.max()))
    print("  -> %s" % ("全部 0 ✓ 图与关节严格对齐" if np.all(leads == 0) else "✗ 仍有错位！"))
    print("开头残留静止帧: max %d ; ≥3 帧的集数 %d  -> %s"
          % (stats.max(), int((stats >= 3).sum()), "✓" if stats.max() < 3 else "✗"))
    print("总帧 %d" % sum(rows))


if __name__ == "__main__":
    main()
