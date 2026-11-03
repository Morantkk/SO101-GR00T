#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""132_retrim_from_trim.py —— 在**服务器上**把旧 trim 再往前裁几帧，并【修正 timestamp 错位】。

为什么不用上传 trim2：本机到实例的 SFTP 只有 ~200 KB/s，1.5G 要一个多小时；
而实例上已经有 `/root/gr00t_data/arm101_pick_all_objects_trim`（同一批原始数据的裁剪版）。
两者只差"头部少裁几帧"（每集 0~13 帧），在这个数据上完全等价。
★ 真正的修复是**把 timestamp / frame_index 归零**（旧 104 漏了这一步，导致图像比关节超前 k 帧）。

规则：目标长度由 `--plan` 给出（trim2 每集帧数）。
      extra = 旧长度 − 目标长度；若为负（旧裁剪起点比目标还晚 1~4 帧），取 0 —— 也就是保留旧起点，只少裁那 1~4 帧。

用法（gr00t 环境）：
  python 132_retrim_from_trim.py --src /root/gr00t_data/arm101_pick_all_objects_trim \
      --dst /root/gr00t_data/arm101_pick_all_objects_trim2 --plan /root/setup/trim2_plan.json --apply
"""
import argparse
import glob
import json
import os
import shutil
import time
from fractions import Fraction

import numpy as np
import pandas as pd

FPS = 15.0


def log(m):
    print(m, flush=True)


def cut_clip(src_mp4, dst_mp4, n_drop, crf=18, fps=15):
    import av
    with av.open(src_mp4) as cin:
        st_in = cin.streams.video[0]
        w, h = st_in.codec_context.width, st_in.codec_context.height
        out = av.open(dst_mp4, mode="w", format="mp4")
        st = out.add_stream("libx264", rate=fps)
        st.width, st.height = w, h
        st.pix_fmt = "yuv420p"
        st.options = {"crf": str(crf), "preset": "veryfast"}
        kept = 0
        for i, fr in enumerate(cin.decode(video=0)):
            if i < n_drop:
                continue
            fr = fr.reformat(format="yuv420p")
            fr.pts = kept
            fr.time_base = Fraction(1, fps)
            for pkt in st.encode(fr):
                out.mux(pkt)
            kept += 1
        for pkt in st.encode():
            out.mux(pkt)
        out.close()
    return kept


def recompute_stats(dst, files):
    p = os.path.join(dst, "meta", "stats.json")
    old = json.load(open(p, encoding="utf-8"))
    acc = {c: [] for c in ("observation.state", "action")}
    for f in files:
        d = pd.read_parquet(f)
        for c in acc:
            acc[c].append(np.asarray(list(d[c]), dtype=np.float32))
    new = {}
    for c, arrs in acc.items():
        a = np.concatenate(arrs, axis=0)
        out = {}
        for k in old[c].keys():
            if k == "min":
                out[k] = a.min(axis=0).tolist()
            elif k == "max":
                out[k] = a.max(axis=0).tolist()
            elif k == "mean":
                out[k] = a.mean(axis=0).tolist()
            elif k == "std":
                out[k] = a.std(axis=0).tolist()
            else:
                out[k] = old[c][k]
        new[c] = out
    json.dump(new, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--plan", required=True, help="JSON: {episode_index: 目标帧数}")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--apply", action="store_true")
    o = ap.parse_args()

    t0 = time.time()
    plan = {int(k): int(v) for k, v in json.load(open(o.plan, encoding="utf-8")).items()}
    pq = sorted(glob.glob(os.path.join(o.src, "data", "chunk-*", "file-*.parquet")))
    cams = sorted(d for d in os.listdir(os.path.join(o.src, "videos", "chunk-000"))
                  if os.path.isdir(os.path.join(o.src, "videos", "chunk-000", d)))
    log("源 %s：%d 集 / 相机 %s" % (o.src, len(pq), cams))

    info = {}
    rows = []
    for p in pq:
        e = int(os.path.basename(p).split("-")[1].split(".")[0])
        d = pd.read_parquet(p)
        n_old = len(d)
        n_tgt = plan.get(e, n_old)
        extra = max(0, n_old - n_tgt)          # 旧起点比目标早 ⇒ 再裁；否则不动
        rows.append((e, n_old, extra, n_old - extra))
    tot_old = sum(r[1] for r in rows)
    tot_new = sum(r[3] for r in rows)
    ex = np.array([r[2] for r in rows])
    log("旧总帧 %d → 新总帧 %d（再裁 %d 帧，%.2f%%）" % (tot_old, tot_new, ex.sum(), 100 * ex.sum() / tot_old))
    log("每集再裁: min %d 中位 %d max %d ; >0 的集数 %d"
        % (ex.min(), int(np.median(ex)), ex.max(), int((ex > 0).sum())))
    log("(负数被夹成 0 的集数: %d —— 这些集旧起点比目标晚 1~4 帧，保留旧起点)"
        % int(sum(1 for r in rows if plan.get(r[0], r[1]) > r[1])))

    if not o.apply:
        log("\n(未给 --apply，不写盘)")
        return

    if os.path.exists(o.dst):
        log("清空 %s" % o.dst)
        shutil.rmtree(o.dst)
    os.makedirs(o.dst, exist_ok=True)
    shutil.copytree(os.path.join(o.src, "meta"), os.path.join(o.dst, "meta"))

    # 1) parquet（★ timestamp / frame_index 归零）
    out_data = os.path.join(o.dst, "data", "chunk-000")
    os.makedirs(out_data, exist_ok=True)
    out_files = []
    g = 0
    for (e, n_old, extra, n_new) in rows:
        d = pd.read_parquet(os.path.join(o.src, "data", "chunk-000", "file-%03d.parquet" % e)).iloc[extra:]
        d = d.reset_index(drop=True).copy()
        d["frame_index"] = np.arange(len(d), dtype=np.int64)
        d["timestamp"] = np.arange(len(d), dtype=np.float64) / FPS
        d["index"] = np.arange(g, g + len(d), dtype=np.int64)
        q = os.path.join(out_data, "file-%03d.parquet" % e)
        d.to_parquet(q, index=False)
        out_files.append(q)
        g += len(d)
    log("parquet 写出 %d 个（timestamp/frame_index 已归零）" % len(out_files))

    # 2) 视频（extra==0 的集直接**复制**，不重编码 —— 只有 8 集需要真的再裁）
    n_copy = n_enc = 0
    for cam in cams:
        od = os.path.join(o.dst, "videos", "chunk-000", cam)
        os.makedirs(od, exist_ok=True)
        for i, (e, n_old, extra, n_new) in enumerate(rows):
            s = os.path.join(o.src, "videos", "chunk-000", cam, "file-%03d.mp4" % e)
            d = os.path.join(od, "file-%03d.mp4" % e)
            if extra == 0:
                shutil.copyfile(s, d)
                n_copy += 1
            else:
                kept = cut_clip(s, d, extra, crf=o.crf)
                n_enc += 1
                if kept != n_new:
                    log("  ⚠ ep%03d %s 帧数 %d != %d" % (e, cam, kept, n_new))
            if (i + 1) % 25 == 0 or i + 1 == len(rows):
                log("   %s %d/%d" % (cam, i + 1, len(rows)))
    log("视频完成：复制 %d 个 / 重编码 %d 个" % (n_copy, n_enc))

    # 3) meta
    info_p = os.path.join(o.dst, "meta", "info.json")
    info = json.load(open(info_p, encoding="utf-8"))
    info["total_frames"] = int(g)
    info["total_episodes"] = len(rows)
    json.dump(info, open(info_p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    ep_p = os.path.join(o.dst, "meta", "episodes.jsonl")
    eps = [json.loads(l) for l in open(ep_p, encoding="utf-8") if l.strip()]
    by = {e: n_new for (e, _a, _b, n_new) in rows}
    for r in eps:
        r["length"] = int(by.get(int(r["episode_index"]), r.get("length", 0)))
    with open(ep_p, "w", encoding="utf-8") as f:
        for r in eps:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    recompute_stats(o.dst, out_files)
    log("meta 更新完毕；total_frames=%d" % g)
    log("完成（%.1f 分钟）-> %s" % ((time.time() - t0) / 60, o.dst))


if __name__ == "__main__":
    main()
