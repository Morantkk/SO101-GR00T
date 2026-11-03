#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""129b_verify_dataset.py —— 数据集上线前体检（只读，不改任何文件）。

查 7 项：
  A 元数据自洽：total_episodes / total_frames / 文件数
  B 逐集帧数一致：parquet 行数 == front 帧数 == wrist 帧数（★ 只用容器解复用计数，不解码像素，快）
  C 索引连续：frame_index 从 0 起、index 全局连续、episode_index 无缺号
  D 任务映射：tasks.jsonl ↔ 每帧 task_index ↔ episodes.jsonl 的 tasks 字段
  E 归一化统计：stats.json 与实际数据逐维对账（相对差）
  F 结构语义（子任务集专查）：每段恰好一次「夹爪闭合→张开」
  G 起点位姿：每集第 0 帧各关节中位 / 极值

用法：
  python 129b_verify_dataset.py --ds E:/lerobot_datasets/arm101_pick_all_objects_subtask
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd


def log(m):
    print(m, flush=True)


def n_frames(path):
    import av
    with av.open(path) as c:
        return sum(1 for p in c.demux(video=0) if p.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--skip-video", action="store_true", help="跳过逐帧计数（只查元数据）")
    o = ap.parse_args()
    ds = o.ds
    info = json.load(open(os.path.join(ds, "meta", "info.json"), encoding="utf-8"))
    eps = [json.loads(l) for l in open(os.path.join(ds, "meta", "episodes.jsonl"),
                                       encoding="utf-8") if l.strip()]
    tasks = [json.loads(l) for l in open(os.path.join(ds, "meta", "tasks.jsonl"),
                                         encoding="utf-8") if l.strip()]
    CH = info.get("chunks_size", 1000)

    log("=" * 78)
    log("数据集 %s" % ds)
    log("fps=%s  total_episodes=%s  total_frames=%s  chunks_size=%s"
        % (info.get("fps"), info.get("total_episodes"), info.get("total_frames"), CH))

    # ---------- A ----------
    pq = sorted(glob.glob(os.path.join(ds, "data", "chunk-*", "file-*.parquet")))
    cams = [d for d in sorted(os.listdir(os.path.join(ds, "videos", "chunk-000")))
            if os.path.isdir(os.path.join(ds, "videos", "chunk-000", d))]
    log("\n[A] 元数据自洽")
    log("    parquet 文件 %d 个 / episodes.jsonl %d 条 / total_episodes %s"
        % (len(pq), len(eps), info.get("total_episodes")))
    okA = len(pq) == len(eps) == int(info.get("total_episodes", -1))
    log("    -> %s" % ("OK" if okA else "✗ 数量不一致"))

    # ---------- B / C ----------
    log("\n[B] 逐集帧数（parquet == 各路视频）")
    log("    相机: %s" % cams)
    bad = []
    rowsum = 0
    idx_bad = []
    gmax = -1
    for r in eps:
        e = int(r["episode_index"])
        ck = e // CH
        q = os.path.join(ds, "data", "chunk-%03d" % ck, "file-%03d.parquet" % e)
        if not os.path.exists(q):
            bad.append((e, "缺 parquet"))
            continue
        df = pd.read_parquet(q)
        rowsum += len(df)
        if len(df) != int(r["length"]):
            bad.append((e, "parquet %d != episodes.length %d" % (len(df), r["length"])))
        fi = df["frame_index"].to_numpy()
        if fi[0] != 0 or not np.all(np.diff(fi) == 1):
            idx_bad.append((e, "frame_index"))
        ix = df["index"].to_numpy()
        if not np.all(np.diff(ix) == 1) or ix[0] <= gmax:
            idx_bad.append((e, "index"))
        gmax = int(ix[-1])
        if df["episode_index"].nunique() != 1 or int(df["episode_index"].iloc[0]) != e:
            idx_bad.append((e, "episode_index 混入"))
        if not o.skip_video:
            for cam in cams:
                v = os.path.join(ds, "videos", "chunk-%03d" % ck, cam, "file-%03d.mp4" % e)
                if not os.path.exists(v):
                    bad.append((e, "缺 %s" % cam))
                    continue
                nv = n_frames(v)
                if nv != len(df):
                    bad.append((e, "%s %d != %d" % (cam, nv, len(df))))
    log("    -> %s（不符 %d 处%s）" % ("全部一致 ✓" if not bad else "✗", len(bad), "" if not bad else "：" + str(bad[:6])))
    log("[C] 索引连续：%s" % ("OK" if not idx_bad else "✗ %s" % idx_bad[:6]))

    # ---------- D ----------
    log("\n[D] 任务映射")
    tmap = {int(t["task_index"]): t["task"] for t in tasks}
    log("    tasks.jsonl: %s" % tmap)
    d_bad = []
    for r in eps:
        e = int(r["episode_index"])
        ck = e // CH
        df = pd.read_parquet(os.path.join(ds, "data", "chunk-%03d" % ck, "file-%03d.parquet" % e))
        ti = df["task_index"].unique()
        if len(ti) != 1:
            d_bad.append((e, "task_index 不唯一 %s" % ti))
        elif tmap.get(int(ti[0])) != (r["tasks"][0] if r.get("tasks") else None):
            d_bad.append((e, "task_index %s 与 episodes.tasks %s 不符" % (ti, r.get("tasks"))))
    log("    -> %s" % ("OK" if not d_bad else "✗ %s" % d_bad[:5]))

    # ---------- E ----------
    log("\n[E] 归一化统计对账（stats.json vs 实际数据）")
    st = json.load(open(os.path.join(ds, "meta", "stats.json"), encoding="utf-8"))
    acc = {c: [] for c in ("observation.state", "action")}
    for q in pq:
        df = pd.read_parquet(q)
        for c in acc:
            acc[c].append(np.asarray(list(df[c]), dtype=np.float32))
    for c in acc:
        a = np.concatenate(acc[c], axis=0)
        for k, fn in (("min", a.min), ("max", a.max), ("mean", a.mean), ("std", a.std)):
            if k not in st.get(c, {}):
                continue
            s = np.array(st[c][k], dtype=np.float64)
            v = fn(axis=0).astype(np.float64)
            rel = np.max(np.abs(v - s) / (np.abs(s) + 1e-6))
            log("    %-18s %-5s 最大相对差 %.2e" % (c, k, rel))
    log("    （>1e-3 就要查；只有 1e-6 级才说明统计就是这份数据算的）")

    # ---------- F / G ----------
    log("\n[F] 每段「夹爪闭合→张开」次数")
    cnt = []
    starts = []
    for r in eps:
        e = int(r["episode_index"])
        ck = e // CH
        df = pd.read_parquet(os.path.join(ds, "data", "chunk-%03d" % ck, "file-%03d.parquet" % e))
        s = np.asarray(list(df["observation.state"]), dtype=np.float32)
        g = s[:, 5]
        c = (g < 12).astype(np.int8)
        edges = np.where(np.diff(np.r_[0, c, 0]) != 0)[0]
        runs = [(int(a), int(b)) for a, b in edges.reshape(-1, 2) if b - a >= 8]
        cnt.append(len(runs))
        starts.append(s[0])
    cnt = np.array(cnt)
    u, n = np.unique(cnt, return_counts=True)
    log("    每段闭合次数分布: %s" % dict(zip(u.tolist(), n.tolist())))
    S = np.array(starts)
    log("\n[G] 每段第 0 帧位姿（中位 / 最小 / 最大）")
    JN = ["pan", "lift", "elbow", "wf", "wr", "grip"]
    for i, nm in enumerate(JN[:S.shape[1]]):
        log("    %-6s %8.2f  [%8.2f, %8.2f]" % (nm, np.median(S[:, i]), S[:, i].min(), S[:, i].max()))
    log("    lift < -100 的段数: %d / %d" % (int((S[:, 1] < -100).sum()), len(S)))
    log("\n总帧数（parquet 实算）= %d ；info.total_frames = %s"
        % (rowsum, info.get("total_frames")))


if __name__ == "__main__":
    main()
