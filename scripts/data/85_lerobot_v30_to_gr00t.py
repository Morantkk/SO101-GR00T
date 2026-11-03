#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 LeRobot **v3.0** 数据集转成 Isaac-GR00T 能读的布局（v2.1 风格：逐 episode 一个文件）。

为什么需要：
  Isaac-GR00T 的 `LeRobotSingleDataset` 硬性依赖：
    · meta/episodes.jsonl                    （每行 {"episode_index","tasks","length"}）
    · meta/tasks.jsonl                       （每行 {"task_index","task"}）
    · meta/modality.json                     （state/action/video 的切片与键映射）
    · info.json 的 data_path / video_path 模板，用 {episode_chunk}/{episode_index}/{video_key} 格式化
      例：data/chunk-{episode_chunk:03d}/file-{episode_index:03d}.parquet
         videos/chunk-{episode_chunk:03d}/{video_key}/file-{episode_index:03d}.mp4
  而 lerobot v3.0 数据集是"按大小分块"：50 个 episode 全在 **1 个 parquet + 每相机 1 个 mp4** 里。
  ⇒ 必须按 episode 切开，并把视频**重编码**（顺便 AV1 → H.264，避免 GR00T 侧解码器不认 AV1）。

做法（不改动源数据集，输出到新目录）：
  1) 复制 meta/（info.json / stats.json / tasks.parquet / episodes/）
  2) 用 meta/episodes 里的 dataset_from_index/dataset_to_index 把大 parquet 切成 50 个
  3) 用每集 length 累积帧数，**单遍解码**每个相机的长视频，边解边写 50 个 H.264 mp4
  4) 生成 episodes.jsonl / tasks.jsonl，拷入 modality.json，改写 info.json 的路径模板

用法（在 **lerobot** 环境跑，它自带 pyav/pandas）：
  /root/miniconda3/envs/lerobot/bin/python 85_lerobot_v30_to_gr00t.py \
      --src /root/arm101_pick_one_pen \
      --dst /root/gr00t_data/arm101_pick_one_pen \
      --modality /root/setup/so100_dualcam__modality.json
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


def log(m):
    print(m, flush=True)


# ------------------------------------------------------------------ 读源数据集
def load_meta(src):
    info = json.load(open(os.path.join(src, "meta", "info.json"), encoding="utf-8"))
    ep_files = sorted(glob.glob(os.path.join(src, "meta", "episodes", "**", "*.parquet"), recursive=True))
    eps = pd.concat([pd.read_parquet(p) for p in ep_files], ignore_index=True).sort_values("episode_index")
    tasks = pd.read_parquet(os.path.join(src, "meta", "tasks.parquet"))
    return info, eps, tasks


def cam_keys(info):
    return [k for k, v in info["features"].items() if k.startswith("observation.images.")]


# ------------------------------------------------------------------ 切 parquet
def split_parquet(src, dst, info, eps):
    data_files = sorted(glob.glob(os.path.join(src, "data", "chunk-*", "*.parquet")))
    log("  源 data 文件: %s" % [os.path.basename(p) for p in data_files])
    big = pd.concat([pd.read_parquet(p) for p in data_files], ignore_index=True)
    log("  总行数: %d（meta 里 50 集 length 之和 = %d）" % (len(big), int(eps["length"].sum())))

    out_dir = os.path.join(dst, "data", "chunk-000")
    os.makedirs(out_dir, exist_ok=True)

    # 优先用 dataset_from/to_index（v3.0 自带）；否则按 length 累积
    use_idx = "dataset_from_index" in eps.columns and "dataset_to_index" in eps.columns
    log("  切分依据: %s" % ("dataset_from_index/to_index" if use_idx else "按 length 累积"))

    cur = 0
    n_written = 0
    for _, r in eps.iterrows():
        e = int(r["episode_index"])
        L = int(r["length"])
        if use_idx:
            a, b = int(r["dataset_from_index"]), int(r["dataset_to_index"])
        else:
            a, b = cur, cur + L
        cur = b
        part = big.iloc[a:b].reset_index(drop=True)
        if len(part) != L:
            log("  ⚠ ep%03d 实取 %d 行 != length %d（from=%d to=%d）" % (e, len(part), L, a, b))
        part.to_parquet(os.path.join(out_dir, "file-%03d.parquet" % e), index=False)
        n_written += 1
    log("  写出 %d 个 episode parquet -> %s" % (n_written, out_dir))
    return eps.set_index("episode_index")["length"].astype(int).to_dict()


# ------------------------------------------------------------------ 切视频（单遍解码 + 重编码）
def pick_encoder():
    import av
    want = ["libx264", "h264_nvenc", "libopenh264", "mpeg4"]
    have = set(av.codecs_available)
    for w in want:
        if w in have:
            return w
    return None


def split_videos(src, dst, cams, lengths, info, crf=23, max_frames=None):
    import av

    enc = pick_encoder()
    log("  视频编码器: %s" % enc)
    if enc is None:
        raise RuntimeError("PyAV 里没有任何可用的视频编码器（libx264/h264_nvenc/mpeg4 都缺）")

    for cam in cams:
        files = sorted(glob.glob(os.path.join(src, "videos", cam, "chunk-*", "*.mp4")))
        if len(files) != 1:
            log("  ⚠ %s 下有 %d 个视频文件，按顺序拼接处理" % (cam, len(files)))
        if not files:
            raise FileNotFoundError("找不到 %s 的视频" % cam)
        out_dir = os.path.join(dst, "videos", "chunk-000", cam)
        os.makedirs(out_dir, exist_ok=True)
        fps = int(float(info.get("fps", 15)) or 15)

        t0 = time.time()
        order = sorted(lengths.items())        # [(ep, length), ...]
        total_expected = sum(l for _, l in order)
        k = 0
        left = order[0][1]
        out, stream = None, None
        n_total = 0
        n_in_file = 0
        stop = False

        def open_next(size):
            """★ 用解码帧的**真实尺寸**建流（info.json 的 shape 是 (H,W,C)，不要拿它当 (C,H,W)）。"""
            nonlocal out, stream, n_in_file
            n_in_file = 0
            ep = order[k][0]
            path = os.path.join(out_dir, "file-%03d.mp4" % ep)
            out = av.open(path, mode="w", format="mp4")
            stream = out.add_stream(enc, rate=fps)
            stream.width, stream.height = size
            stream.pix_fmt = "yuv420p"
            if enc == "libx264":
                stream.options = {"crf": str(crf), "preset": "veryfast"}
            elif enc == "h264_nvenc":
                stream.options = {"cq": str(crf)}
            return ep

        def close_cur():
            for pkt in stream.encode():
                out.mux(pkt)
            out.close()

        for f in files:
            if stop:
                break
            with av.open(f) as cont:
                for frame in cont.decode(video=0):
                    if left <= 0:
                        close_cur()
                        k += 1
                        if k >= len(order):
                            stop = True
                            break
                        left = order[k][1]
                        out = None
                    frame = frame.reformat(format="yuv420p")   # libx264 必须显式转像素格式
                    if out is None:                            # 延迟到拿到真实帧尺寸后再建流
                        open_next((frame.width, frame.height))
                    frame.pts = n_in_file
                    frame.time_base = Fraction(1, fps)
                    for pkt in stream.encode(frame):
                        out.mux(pkt)
                    n_in_file += 1
                    left -= 1
                    n_total += 1
        if out is not None:
            close_cur()
        log("  %s: 解出 %d 帧 -> %d 个 mp4（%.1fs）"
            % (cam, n_total, min(k + 1, len(order)), time.time() - t0))
        if n_total != total_expected:
            log("  ⚠ %s 总帧数 %d != 期望 %d（按 length 之和）" % (cam, n_total, total_expected))

        log("  %s: 解出 %d 帧 -> %d 个 mp4（%.1fs）" % (cam, n_total, len(order), time.time() - t0))
        if max_frames and n_total != max_frames:
            log("  ⚠ %s 总帧数 %d != 期望 %d" % (cam, n_total, max_frames))


# ------------------------------------------------------------------ 写 jsonl / info / modality
def task_items(tasks):
    """tasks.parquet 的索引往往是**任务文本**，task_index 在列里 —— 两种情况都兼容。"""
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


def write_meta(src, dst, info, eps, tasks, modality_path):
    meta = os.path.join(dst, "meta")
    os.makedirs(meta, exist_ok=True)

    items = task_items(tasks)
    # tasks.jsonl
    with open(os.path.join(meta, "tasks.jsonl"), "w", encoding="utf-8") as f:
        for ti, nm in items:
            f.write(json.dumps({"task_index": ti, "task": nm}, ensure_ascii=False) + "\n")
    log("  tasks.jsonl: %d 条 -> %s" % (len(items), items))

    # episodes.jsonl
    tmap = dict(items)
    with open(os.path.join(meta, "episodes.jsonl"), "w", encoding="utf-8") as f:
        for _, r in eps.iterrows():
            e = int(r["episode_index"])
            tv = r.get("tasks", None)
            if tv is None:
                tl = [tmap.get(int(r.get("task_index", 0)), "")]
            elif hasattr(tv, "tolist"):
                tl = [str(x) for x in tv.tolist()]
            elif isinstance(tv, (list, tuple)):
                tl = [str(x) for x in tv]
            else:
                tl = [str(tv)]
            f.write(json.dumps({"episode_index": e, "tasks": tl, "length": int(r["length"])},
                               ensure_ascii=False) + "\n")
    log("  episodes.jsonl: %d 条" % len(eps))

    # modality.json
    if modality_path and os.path.exists(modality_path):
        shutil.copyfile(modality_path, os.path.join(meta, "modality.json"))
    elif not os.path.exists(os.path.join(meta, "modality.json")):
        raise FileNotFoundError("缺 modality.json，请用 --modality 指定模板")
    log("  modality.json 就绪: %s" % json.dumps(json.load(open(os.path.join(meta, "modality.json"), encoding="utf-8")), ensure_ascii=False)[:200] + " ...")

    # info.json（改写路径模板；其余原样保留）
    info2 = dict(info)
    info2["data_path"] = "data/chunk-{episode_chunk:03d}/file-{episode_index:03d}.parquet"
    info2["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/file-{episode_index:03d}.mp4"
    json.dump(info2, open(os.path.join(meta, "info.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    log("  info.json: data_path/video_path 已改写")

    # stats.json 直接用源数据集的（v3.0 的聚合 stats）
    for f in ("stats.json",):
        s = os.path.join(src, "meta", f)
        if os.path.exists(s) and not os.path.exists(os.path.join(meta, f)):
            shutil.copyfile(s, os.path.join(meta, f))
            log("  已复制 %s" % f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="源数据集（lerobot v3.0）")
    ap.add_argument("--dst", required=True, help="输出目录（会被清空重建）")
    ap.add_argument("--modality", default=None, help="modality.json 模板路径")
    ap.add_argument("--skip-video", action="store_true", help="只切 parquet 与 meta（调试用）")
    ap.add_argument("--limit-episodes", type=int, default=0, help="只转前 N 集（调试用）")
    o = ap.parse_args()

    log("=== 源: %s" % o.src)
    info, eps, tasks = load_meta(o.src)
    cams = cam_keys(info)
    log("  fps=%s  episodes=%d  frames=%s  相机=%s"
        % (info.get("fps"), len(eps), info.get("total_frames"), cams))
    if o.limit_episodes:
        eps = eps[eps["episode_index"] < o.limit_episodes].copy()
        log("  (调试) 只转前 %d 集" % o.limit_episodes)

    if os.path.exists(o.dst):
        log("=== 清空输出目录 %s" % o.dst)
        shutil.rmtree(o.dst)
    os.makedirs(o.dst, exist_ok=True)

    log("=== 1) 切 parquet")
    lengths = split_parquet(o.src, o.dst, info, eps)

    if not o.skip_video:
        log("=== 2) 切视频（重编码为 H.264；单遍解码）")
        split_videos(o.src, o.dst, cams, lengths, info,
                     max_frames=int(eps["length"].sum()))
    else:
        log("=== 2) 跳过视频")

    log("=== 3) 写 meta")
    write_meta(o.src, o.dst, info, eps, tasks, o.modality)

    log("")
    log("=== 结果 ===")
    for root, _dirs, files in os.walk(o.dst):
        rel = os.path.relpath(root, o.dst)
        if rel in (".", "data/chunk-000") or rel.startswith("videos/chunk-000"):
            log("  %-46s %d 个文件" % (rel + "/", len(files)))
    log("完成 -> %s" % o.dst)
    log("下一步自检: 用 gr00t 环境跑 scripts/load_dataset.py --dataset-path %s --plot-state-action" % o.dst)


if __name__ == "__main__":
    main()
