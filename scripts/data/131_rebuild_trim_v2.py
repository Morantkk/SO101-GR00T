#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""131_rebuild_trim_v2.py —— 从**原始采集数据**一次重建「裁好头 + 双相机统一 640×360」的 GR00T 数据集。

为什么要重做（两个真问题）：
  1) 旧 104 裁完头以后**没有把 timestamp 归零**。原始 parquet 里 timestamp = frame_index/15（从 0 起），
     视频被裁掉前 n 帧后从 t=0 开始，可 parquet 里还写着 4.2s、4.267s…
     而 GR00T 取帧是 `get_frames_by_timestamps(video, timestamp[step])`（`argmin |frame_ts − ts|`）
     ⇒ **每一集的图像都比关节状态/动作晚了 n 帧**（n = 该集被裁掉的帧数，46~124 帧 = 3~8 秒，逐集不同）。
     这是 teacher-forced 离线指标看不出来、却会直接毁掉闭环的那类错误。
     本脚本把 timestamp 与 frame_index 都重置为 0 起（`timestamp = i / fps`）。
  2) 旧的静止段判据是「5 帧累计位移 > 2.0 且再往后取 5 帧」，判据偏松：个别集会残留静止帧，
     且切点落在加速段的不同位置（首帧速度 0.18~4.48°/帧，差 25 倍）。
     新判据 = 「连续 RUN 帧每帧都 ≥ TH」的第一帧，实测：
        · 裁掉 7932 帧(13.4%)，比旧判据还**少裁** 74 帧
        · 首帧 lift std 3.76（与旧的一致）、elbow std 2.41→1.87
        · 首帧速度 std 0.88→0.29（相位整齐 3 倍）
        · 头部残留静止帧 **0 集**（旧的 3 集，最多 64 帧）

一次解码一次编码完成（顺便把 wrist 1280×720 → 640×360，不用再单独跑 92）。

用法：
  python 131_rebuild_trim_v2.py --src E:/lerobot_datasets/arm101_pick_all_objects \
      --dst E:/lerobot_datasets/arm101_pick_all_objects_trim2 \
      --modality E:/lerobot_datasets/arm101_pick_all_objects_trim/meta/modality.json --apply
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


# ---------------------------------------------------------------- 起点检测
def onset_index(st5, th=0.35, run=6):
    """第一个「连续 run 帧、每帧最大关节位移都 ≥ th」的帧号。frame 0 一定已在动。"""
    v = np.abs(np.diff(st5, axis=0)).max(axis=1)
    for i in range(max(0, len(v) - run)):
        if v[i:i + run].min() >= th:
            return int(i)
    return 0


# ---------------------------------------------------------------- 读源
def load_meta(src):
    info = json.load(open(os.path.join(src, "meta", "info.json"), encoding="utf-8"))
    ep_files = sorted(glob.glob(os.path.join(src, "meta", "episodes", "**", "*.parquet"), recursive=True))
    eps = pd.concat([pd.read_parquet(p) for p in ep_files], ignore_index=True).sort_values("episode_index")
    tasks = pd.read_parquet(os.path.join(src, "meta", "tasks.parquet"))
    return info, eps, tasks


def cam_keys(info):
    return [k for k, v in info["features"].items() if k.startswith("observation.images.")]


def task_items(tasks):
    cols = list(tasks.columns)
    out = []
    for idx, row in tasks.iterrows():
        if "task_index" in cols:
            ti = int(row["task_index"])
        elif isinstance(idx, (int, np.integer)):
            ti = int(idx)
        else:
            ti = len(out)
        nm = str(row["task"]) if "task" in cols else str(idx)
        out.append((ti, nm))
    return sorted(out)


def recompute_stats(dst, files):
    p = os.path.join(dst, "meta", "stats.json")
    old = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    acc = {c: [] for c in ("observation.state", "action")}
    for f in files:
        d = pd.read_parquet(f)
        for c in acc:
            acc[c].append(np.asarray(list(d[c]), dtype=np.float32))
    new = {}
    for c, arrs in acc.items():
        a = np.concatenate(arrs, axis=0)
        src = old.get(c, {"min": 0, "max": 0, "mean": 0, "std": 0})
        out = {}
        for k in src.keys():
            if k == "min":
                out[k] = a.min(axis=0).tolist()
            elif k == "max":
                out[k] = a.max(axis=0).tolist()
            elif k == "mean":
                out[k] = a.mean(axis=0).tolist()
            elif k == "std":
                out[k] = a.std(axis=0).tolist()
            else:
                out[k] = src[k]
        new[c] = out
    json.dump(new, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--modality", required=True)
    ap.add_argument("--th", type=float, default=0.35)
    ap.add_argument("--run", type=int, default=6)
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=360)
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--limit-episodes", type=int, default=0, help="只处理前 N 集（冒烟测试用）")
    ap.add_argument("--apply", action="store_true")
    o = ap.parse_args()

    t_all = time.time()
    info, eps, tasks = load_meta(o.src)
    cams = cam_keys(info)
    log("源 %s" % o.src)
    log("  episodes=%d frames=%s fps=%s cams=%s" % (len(eps), info.get("total_frames"), info.get("fps"), cams))

    big = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(o.src, "data", "chunk-*", "*.parquet")))],
                    ignore_index=True)
    log("  源 data 行数 %d" % len(big))

    use_idx = "dataset_from_index" in eps.columns and "dataset_to_index" in eps.columns
    plan = []          # (ep, a, b, k, newlen)
    cur = 0
    for _, r in eps.iterrows():
        e = int(r["episode_index"])
        L = int(r["length"])
        if use_idx:
            a, b = int(r["dataset_from_index"]), int(r["dataset_to_index"])
        else:
            a, b = cur, cur + L
        cur = b
        st = np.asarray(list(big.iloc[a:b]["observation.state"]), dtype=np.float32)
        k = onset_index(st[:, :5], o.th, o.run)
        k = min(k, max(0, L - 30))
        plan.append((e, a, b, k, L - k))

    if o.limit_episodes:
        plan = plan[:o.limit_episodes]
        log("  (冒烟) 只处理前 %d 集" % len(plan))

    ks = np.array([p[3] for p in plan])
    tot0 = int(eps["length"].sum())
    tot1 = sum(p[4] for p in plan)
    lifts = np.array([np.asarray(list(big.iloc[a + k:b]["observation.state"]), dtype=np.float32)[0, 1]
                      for (e, a, b, k, n) in plan])
    log("\n=== 裁剪计划 ===")
    log("  裁掉 %d 帧 / %d = %.1f%%   每集 k: min %d 中位 %d max %d"
        % (ks.sum(), tot0, 100 * ks.sum() / tot0, ks.min(), int(np.median(ks)), ks.max()))
    log("  新第 0 帧 lift: std %.2f  中位 %.2f  min %.2f" % (lifts.std(), np.median(lifts), lifts.min()))
    log("  新总帧数 %d（源 %d）" % (tot1, tot0))

    if not o.apply:
        log("\n(未给 --apply，不写盘)")
        return

    if os.path.exists(o.dst):
        log("清空 %s" % o.dst)
        shutil.rmtree(o.dst)
    os.makedirs(os.path.join(o.dst, "meta"), exist_ok=True)

    # ---------- 1) parquet（★ timestamp / frame_index 归零）----------
    log("\n=== 1) 写 parquet（timestamp 归零）")
    out_data = os.path.join(o.dst, "data", "chunk-000")
    os.makedirs(out_data, exist_ok=True)
    out_files = []
    gidx = 0
    for (e, a, b, k, n) in plan:
        part = big.iloc[a + k:b].reset_index(drop=True).copy()
        part["frame_index"] = np.arange(len(part), dtype=np.int64)
        part["timestamp"] = (np.arange(len(part), dtype=np.float64) / FPS)
        part["index"] = np.arange(gidx, gidx + len(part), dtype=np.int64)
        part["episode_index"] = np.int64(e)
        q = os.path.join(out_data, "file-%03d.parquet" % e)
        part.to_parquet(q, index=False)
        out_files.append(q)
        gidx += len(part)
    log("  写出 %d 个 parquet，共 %d 帧" % (len(out_files), gidx))

    # ---------- 2) 视频（单遍解码，跳 k、缩放到目标尺寸）----------
    log("\n=== 2) 写视频（跳 k 帧 + 缩放 %dx%d）" % (o.w, o.h))
    import av
    import cv2
    enc = "libx264" if "libx264" in av.codecs_available else None
    if enc is None:
        raise RuntimeError("缺 libx264")
    order = [(e, n, k) for (e, a, b, k, n) in plan]

    for cam in cams:
        files = sorted(glob.glob(os.path.join(o.src, "videos", cam, "chunk-*", "*.mp4")))
        out_dir = os.path.join(o.dst, "videos", "chunk-000", cam)
        os.makedirs(out_dir, exist_ok=True)
        log("  %s: %d 个源文件" % (cam, len(files)))

        t0 = time.time()
        idx = 0
        ep, left, skip = order[0][0], order[0][1], order[0][2]
        out = stream = None
        n_in = 0
        written = 0
        stop = False

        def open_ep():
            nonlocal out, stream, n_in
            n_in = 0
            out = av.open(os.path.join(out_dir, "file-%03d.mp4" % ep), mode="w", format="mp4")
            stream = out.add_stream(enc, rate=int(FPS))
            stream.width, stream.height = o.w, o.h
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": str(o.crf), "preset": "veryfast"}

        def close_ep():
            for pkt in stream.encode():
                out.mux(pkt)
            out.close()

        open_ep()
        for f in files:
            if stop:
                break
            with av.open(f) as cont:
                for frame in cont.decode(video=0):
                    if left <= 0:                     # 上一集写满 → 切下一集
                        close_ep()
                        idx += 1
                        if idx >= len(order):
                            stop = True
                            break
                        ep, left, skip = order[idx]
                        open_ep()
                    if skip > 0:                      # ★ 跳过该集头部的 k 帧
                        skip -= 1
                        continue
                    if frame.width != o.w or frame.height != o.h:
                        arr = frame.to_ndarray(format="rgb24")
                        arr = cv2.resize(arr, (o.w, o.h), interpolation=cv2.INTER_AREA)
                        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
                    frame = frame.reformat(format="yuv420p")
                    frame.pts = n_in
                    frame.time_base = Fraction(1, int(FPS))
                    for pkt in stream.encode(frame):
                        out.mux(pkt)
                    n_in += 1
                    left -= 1
                    written += 1
        if not stop:
            close_ep()
        log("  %s: 写出 %d 帧（%.1fs）" % (cam, written, time.time() - t0))

    # ---------- 3) meta ----------
    log("\n=== 3) 写 meta")
    items = task_items(tasks)
    with open(os.path.join(o.dst, "meta", "tasks.jsonl"), "w", encoding="utf-8") as f:
        for ti, nm in items:
            f.write(json.dumps({"task_index": ti, "task": nm}, ensure_ascii=False) + "\n")

    with open(os.path.join(o.dst, "meta", "episodes.jsonl"), "w", encoding="utf-8") as f:
        for (e, a, b, k, n) in plan:
            f.write(json.dumps({"episode_index": e, "tasks": [items[0][1] if len(items) == 1 else ""],
                                "length": int(n)}, ensure_ascii=False) + "\n")

    shutil.copyfile(o.modality, os.path.join(o.dst, "meta", "modality.json"))

    info2 = dict(info)
    info2["data_path"] = "data/chunk-{episode_chunk:03d}/file-{episode_index:03d}.parquet"
    info2["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/file-{episode_index:03d}.mp4"
    info2["total_frames"] = int(gidx)
    info2["total_episodes"] = len(plan)
    if "observation.images.front" in info2.get("features", {}):
        for cam in cams:
            info2["features"][cam]["shape"] = [o.h, o.w, 3]
    json.dump(info2, open(os.path.join(o.dst, "meta", "info.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    # stats：源 stats 结构当模板，数值按新数据重算
    src_stats = os.path.join(o.src, "meta", "stats.json")
    if os.path.exists(src_stats):
        shutil.copyfile(src_stats, os.path.join(o.dst, "meta", "stats.json"))
    recompute_stats(o.dst, out_files)
    log("  stats.json 已按新数据重算")

    log("\n完成 -> %s（%.1f 分钟）" % (o.dst, (time.time() - t_all) / 60))
    log("下一步：python scripts/server/129b_verify_dataset.py --ds %s" % o.dst)


if __name__ == "__main__":
    main()
