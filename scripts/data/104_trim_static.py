#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把每条演示开头的「静止死锁区」裁掉，生成一个新数据集（不动源目录）。

为什么必须裁：
  演示开头 4~7 秒机械臂完全不动（state/action 逐帧 0.00、原始视频像素变化 0.000%），
  而"什么时候开始动"在观测里**没有任何信号** → 同样的观测对应矛盾目标（"别动" vs "开始动"）
  → 模型学成平均值"别动" → 真机闭环死锁：臂停着 → 模型说别动 → 画面不变 → 永远卡住。

做法（逐 episode **同时**裁 parquet 与视频，保证两者帧数严格一致）：
  N = 前导静止帧数（连续 5 帧累计移动 > 2.0° 的首次位置），上限 45% 集长
  1) parquet: 丢掉前 N 行
  2) mp4    : 解出来丢掉前 N 帧，libx264 crf18 重编码
  3) meta   : 重写 episodes.jsonl 的 length、info.json 的 total_frames
  4) ★ 之后必须再跑 91_fix_stats.py 重算 stats.json（归一化用的统计量变了）

用法（gr00t 环境）：
  python 104_trim_static.py --src /root/gr00t_data/arm101_pick_one_pen \
                            --dst /root/gr00t_data/arm101_pick_one_pen_trim --apply
"""
import argparse
import glob
import json
import os
import shutil
from fractions import Fraction

import numpy as np
import pandas as pd

FPS = 15.0


def log(m):
    print(m, flush=True)


def state_of(p):
    d = pd.read_parquet(p)
    return np.asarray(list(d["observation.state"]), dtype=np.float32).reshape(len(d), -1)[:, :6], d


def leading_static(st, win=5, thr=2.0, cap_frac=0.45):
    """返回"前导静止帧数"：从第几帧开始认为已经在运动。"""
    if len(st) <= win + 2:
        return 0
    d = np.abs(np.diff(st, axis=0)).max(axis=1)
    cum = np.convolve(d, np.ones(win), "valid")
    idx = np.where(cum > thr)[0]
    n = int(idx[0] + win) if len(idx) else 0
    return min(n, int(len(st) * cap_frac))


def trim_video(src_mp4, dst_mp4, n_drop, enc="libx264", crf=18, fps=15):
    import av
    with av.open(src_mp4) as cin:
        st_in = cin.streams.video[0]
        w, h = st_in.codec_context.width, st_in.codec_context.height
        out = av.open(dst_mp4, mode="w", format="mp4")
        st_out = out.add_stream(enc, rate=fps)
        st_out.width, st_out.height = w, h
        st_out.pix_fmt = "yuv420p"
        st_out.options = {"crf": str(crf), "preset": "veryfast"}
        kept = 0
        for i, fr in enumerate(cin.decode(video=0)):
            if i < n_drop:
                continue
            fr = fr.reformat(format="yuv420p")
            fr.pts = kept
            fr.time_base = Fraction(1, fps)
            for pkt in st_out.encode(fr):
                out.mux(pkt)
            kept += 1
        for pkt in st_out.encode():
            out.mux(pkt)
        out.close()
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--apply", action="store_true", help="真的写盘（否则只报告）")
    ap.add_argument("--crf", type=int, default=18)
    o = ap.parse_args()

    src, dst = o.src, o.dst
    eps_files = sorted(glob.glob(os.path.join(src, "data", "chunk-*", "file-*.parquet")))
    log("episodes: %d" % len(eps_files))

    # ---------- 1) 算每集的裁剪量 ----------
    plan = []
    for p in eps_files:
        e = int(os.path.basename(p).split("-")[1].split(".")[0])
        st, _ = state_of(p)
        n = leading_static(st)
        plan.append((e, len(st), n))
    drop = np.array([x[2] for x in plan], dtype=float)
    tot = np.array([x[1] for x in plan], dtype=float)
    log("\n=== 裁剪计划 ===")
    log("  前导静止帧: mean %.1f  min %d  max %d  (%.2f~%.2f 秒)"
        % (drop.mean(), drop.min(), drop.max(), drop.min() / FPS, drop.max() / FPS))
    log("  总帧数 %d -> %d（减少 %.1f%%）" % (tot.sum(), (tot - drop).sum(), drop.sum() / tot.sum() * 100))
    for e, t, n in plan:
        log("    ep%02d  %4d -> %4d  (裁 %3d = %.2fs)" % (e, t, t - n, n, n / FPS))

    if not o.apply:
        log("\n(--apply 未给，只报告，不写盘)")
        return

    # ---------- 2) 建目录 / 拷 meta ----------
    if os.path.exists(dst):
        log("清空 %s" % dst)
        shutil.rmtree(dst)
    os.makedirs(dst, exist_ok=True)
    shutil.copytree(os.path.join(src, "meta"), os.path.join(dst, "meta"))

    # ---------- 3) 裁 parquet ----------
    out_data = os.path.join(dst, "data", "chunk-000")
    os.makedirs(out_data, exist_ok=True)
    for (e, t, n), p in zip(plan, eps_files):
        d = pd.read_parquet(p).iloc[n:].reset_index(drop=True)
        d.to_parquet(os.path.join(out_data, "file-%03d.parquet" % e), index=False)
    log("parquet 写出 %d 个" % len(plan))

    # ---------- 4) 裁视频 ----------
    cams = sorted(d for d in os.listdir(os.path.join(src, "videos", "chunk-000"))
                  if os.path.isdir(os.path.join(src, "videos", "chunk-000", d)))
    for cam in cams:
        od = os.path.join(dst, "videos", "chunk-000", cam)
        os.makedirs(od, exist_ok=True)
        bad = []
        for e, t, n in plan:
            s = os.path.join(src, "videos", "chunk-000", cam, "file-%03d.mp4" % e)
            kept = trim_video(s, os.path.join(od, "file-%03d.mp4" % e), n, crf=o.crf)
            if kept != t - n:
                bad.append((e, kept, t - n))
        log("%s: 写出 %d 个%s" % (cam, len(plan), "" if not bad else "  ⚠ 帧数不符: %s" % bad[:5]))

    # ---------- 5) 改 meta ----------
    info_p = os.path.join(dst, "meta", "info.json")
    info = json.load(open(info_p, encoding="utf-8"))
    info["total_frames"] = int(sum(t - n for _, t, n in plan))
    json.dump(info, open(info_p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    log("info.json total_frames -> %d" % info["total_frames"])

    ep_p = os.path.join(dst, "meta", "episodes.jsonl")
    rows = [json.loads(l) for l in open(ep_p, encoding="utf-8") if l.strip()]
    by = {e: t - n for e, t, n in plan}
    for r in rows:
        r["length"] = int(by.get(int(r["episode_index"]), r.get("length", 0)))
    with open(ep_p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log("episodes.jsonl 已更新 %d 条" % len(rows))

    log("\n★ 下一步：/root/miniconda3/envs/gr00t/bin/python /root/setup/91_fix_stats.py %s" % dst)


if __name__ == "__main__":
    main()
