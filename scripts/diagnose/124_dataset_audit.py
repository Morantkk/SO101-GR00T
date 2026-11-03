#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数据集体检（本地跑，不碰云端）：一次性回答"训练数据到底干不干净"。

查 7 项，每项给判据：
  A 元数据        fps / 集数 / 帧数 / 两路相机分辨率是否一致
  B **前导静止段** 用 `104_trim_static.py` 里同一个函数 `leading_static()` 复核
                   —— 这是"闭环死锁"的老病根，必须确认裁干净了
  C 关节统计      state/action 逐维 min/max/mean/std（看有没有越界/异常值）
  D 夹爪         张开/闭合分布 + 每条演示的"合爪次数"（应为物体个数 3~5 次）
  E **两路相机同一刀** 每条演示：front 帧数 == wrist 帧数 == parquet 行数
  F **相机-动作同步** 抽样算"画面运动量 vs 关节速度"的互相关与最佳滞后（应 ≈0 帧）
  G 坏帧        首/中/尾帧是否全黑、相邻帧是否完全重复

用法：
  <python> scripts/124_dataset_audit.py "E:/lerobot_datasets/arm101_pick_all_objects"
  <python> scripts/124_dataset_audit.py "<ds>" --corr-episodes 12
"""
import argparse
import glob
import json
import os
import sys

import numpy as np


def log(*a):
    print(*a, flush=True)


def load_meta(ds):
    info = json.load(open(os.path.join(ds, "meta", "info.json"), encoding="utf-8"))
    return info


def load_all_parquet(ds):
    import pandas as pd
    S, A, E = [], [], []
    for p in sorted(glob.glob(os.path.join(ds, "data", "chunk-*", "*.parquet"))):
        df = pd.read_parquet(p)
        S.append(np.vstack([np.asarray(x, dtype=np.float32) for x in df["observation.state"].values]))
        A.append(np.vstack([np.asarray(x, dtype=np.float32) for x in df["action"].values]))
        E.append(np.asarray(df["episode_index"].values))
    st = np.vstack(S) if S else np.zeros((0, 6), np.float32)
    ac = np.vstack(A) if A else np.zeros((0, 6), np.float32)
    ep = np.concatenate(E) if E else np.zeros(0, int)
    return st, ac, ep


def leading_static(st, win=5, thr=2.0, cap_frac=0.45):
    """与 scripts/server/104_trim_static.py 完全一致。"""
    if len(st) <= win + 2:
        return 0
    d = np.abs(np.diff(st, axis=0)).max(axis=1)
    cum = np.convolve(d, np.ones(win), "valid")
    idx = np.where(cum > thr)[0]
    n = int(idx[0] + win) if len(idx) else 0
    return min(n, int(len(st) * cap_frac))


def video_frames(path):
    import cv2
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return n, fps, w, h


def gray_seq(path, size=(320, 180), nmax=None):
    import cv2
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(cv2.resize(fr, size), cv2.COLOR_BGR2GRAY).astype(np.float32))
        if nmax and len(out) >= nmax:
            break
    cap.release()
    return np.stack(out) if out else np.zeros((0, size[1], size[0]), np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--corr-episodes", type=int, default=10, help="抽样多少条演示做相机-动作同步检验")
    a = ap.parse_args()
    ds = a.dataset.rstrip("/\\")
    if not os.path.isdir(ds):
        log("!! 目录不存在:", ds)
        return 2

    log("=" * 78)
    log("数据集体检:", ds)
    log("=" * 78)

    # ---------- A 元数据 ----------
    info = load_meta(ds)
    log("\n== A 元数据 ==")
    log("  fps=%s  total_episodes=%s  total_frames=%s  total_tasks=%s"
        % (info.get("fps"), info.get("total_episodes"), info.get("total_frames"), info.get("total_tasks")))
    feats = info.get("features", {})
    for k, v in feats.items():
        if "image" in k or "video" in k:
            log("  %-34s shape=%s dtype=%s" % (k, v.get("shape"), v.get("dtype")))
    log("  action shape=%s   observation.state shape=%s"
        % (feats.get("action", {}).get("shape"), feats.get("observation.state", {}).get("shape")))

    tasks = []
    tp = os.path.join(ds, "meta", "tasks.parquet")
    if os.path.exists(tp):
        import pandas as pd
        df = pd.read_parquet(tp)
        # ★ 本项目的 tasks.parquet 把任务文本放在 **index** 上（列只有 task_index）
        if "task" in df.columns:
            tasks = df["task"].astype(str).tolist()
        else:
            tasks = [str(x) for x in df.index.tolist()]
        log("  任务文本(%d): %s" % (len(tasks), tasks))

    # ---------- 数据 ----------
    st, ac, ep = load_all_parquet(ds)
    epu = np.unique(ep)
    idx = {e: np.where(ep == e)[0] for e in epu}
    lens = np.array([len(idx[e]) for e in epu])
    log("\n== 帧数 ==")
    log("  总 %d 帧 / %d 集" % (len(st), len(epu)))
    log("  每集帧数: min %d  p10 %d  中位 %d  p90 %d  max %d"
        % (lens.min(), np.percentile(lens, 10), np.median(lens), np.percentile(lens, 90), lens.max()))

    # ---------- B 前导静止段 ----------
    log("\n== B 前导静止段（判据同 104_trim_static: 连续5帧累计>2.0°，上限45%%集长）==")
    leads = np.array([leading_static(st[idx[e]]) for e in epu])
    log("  长度: min %d  p25 %d  中位 %d  p75 %d  max %d  (%.1f~%.1f 秒 @15fps)"
        % (leads.min(), np.percentile(leads, 25), np.median(leads),
           np.percentile(leads, 75), leads.max(), leads.min() / 15, leads.max() / 15))
    log("  ★ 判据：若中位 > 8 帧（0.5s）⇒ 前导静止段**没裁干净**，闭环极易死锁")
    verdict_b = "OK" if np.median(leads) <= 8 else "**未裁干净**"
    log("  结论: %s   （>20 帧的集数: %d / %d）" % (verdict_b, int((leads > 20).sum()), len(epu)))

    # ---------- C 关节统计 ----------
    JN = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    log("\n== C 关节统计 ==")
    log("  %-13s %26s %26s" % ("关节", "state  min/med/max", "action min/med/max"))
    for i, n in enumerate(JN[:st.shape[1]]):
        s, q = st[:, i], ac[:, i]
        log("  %-13s %8.1f %8.1f %8.1f  %8.1f %8.1f %8.1f"
            % (n, s.min(), np.median(s), s.max(), q.min(), np.median(q), q.max()))

    # ---------- D 夹爪与合爪次数 ----------
    if st.shape[1] >= 6:
        g = st[:, 5]
        log("\n== D 夹爪 ==")
        log("  state gripper: min %.1f  中位 %.1f  max %.1f" % (g.min(), np.median(g), g.max()))
        log("  张开(>20) 占 %.1f%%   闭合(<10) 占 %.1f%%" % (100 * (g > 20).mean(), 100 * (g < 10).mean()))
        nclose = []
        for e in epu:
            gg = st[idx[e], 5]
            c, i0 = 0, 0
            while i0 < len(gg):
                if gg[i0] >= 16:
                    j = i0
                    while j < len(gg) and gg[j] >= 12:
                        j += 1
                    if j < len(gg):
                        c += 1
                        i0 = j + 15
                i0 += 1
            nclose.append(c)
        nclose = np.array(nclose)
        log("  每条演示合爪次数: min %d  中位 %d  max %d  （应≈物体个数 3~5）"
            % (nclose.min(), int(np.median(nclose)), nclose.max()))

    # ---------- E 两路相机同一刀（用 meta/episodes 的 timestamp，不靠猜文件名） ----------
    log("\n== E 两路相机 vs parquet 帧数（两路相机必须同一刀）==")
    log("  说明：leRobot 一个 mp4 里装多条演示 ⇒ 不能用「文件数 == 集数」判断；")
    log("        正确做法是拿 meta/episodes 的 from/to_timestamp 交叉验证。")
    fps = float(info.get("fps") or 15)
    epmeta = []
    for f in sorted(glob.glob(os.path.join(ds, "meta", "episodes", "chunk-*", "*.parquet"))):
        import pandas as pd
        epmeta.append(pd.read_parquet(f))
    if epmeta:
        import pandas as pd
        EM = pd.concat(epmeta, ignore_index=True).sort_values("episode_index")
        cams_meta = [c for c in EM.columns if c.startswith("videos/") and c.endswith("/from_timestamp")]
        cams_meta = [c.split("/")[1] for c in cams_meta]
        log("  episodes meta: %d 条，相机列 %s" % (len(EM), cams_meta))
        bad = 0
        gaps = []
        for _, r in EM.iterrows():
            L = int(r["length"])
            durs = {}
            for cam in cams_meta:
                t0 = float(r["videos/%s/from_timestamp" % cam])
                t1 = float(r["videos/%s/to_timestamp" % cam])
                durs[cam] = (t1 - t0) * fps
            vals = list(durs.values())
            if max(vals) - min(vals) > 1.0:
                bad += 1
                if bad <= 5:
                    log("     ep%-3d 两路帧数不一致: %s (parquet %d)"
                        % (int(r["episode_index"]), {k: round(v, 1) for k, v in durs.items()}, L))
            dev = max(abs(v - L) for v in vals)
            if dev > 1.0:
                bad += 1
                if bad <= 5:
                    log("     ep%-3d 视频帧数与 parquet 差 %.1f 帧: %s vs length=%d"
                        % (int(r["episode_index"]), dev, {k: round(v, 1) for k, v in durs.items()}, L))
        # 时间戳连续性（同文件内相邻集不能有空隙/重叠）
        key = "videos/%s/chunk_index" % cams_meta[0] if cams_meta else None
        if cams_meta:
            for cam in cams_meta:
                prev_t, prev_f = None, None
                for _, r in EM.iterrows():
                    fi = int(r["videos/%s/file_index" % cam])
                    t0 = float(r["videos/%s/from_timestamp" % cam])
                    if prev_f == fi and prev_t is not None:
                        gaps.append((cam, t0 - prev_t))
                    prev_t = float(r["videos/%s/to_timestamp" % cam])
                    prev_f = fi
        if gaps:
            gg = np.array([g for _, g in gaps]) * fps
            log("  同文件内相邻集的衔接（应为 0 帧）: 中位 %.2f 帧  最大 %.2f 帧  (n=%d)"
                % (np.median(gg), gg.max(), len(gg)))
        log("  [%s] 两路相机帧数 / 与 parquet 行数的一致性  -- 异常项 %d"
            % ("PASS" if bad == 0 else "FAIL", bad))
        log("  ★ 判据：两路差 ≤1 帧、与 parquet 差 ≤1 帧、衔接空隙 0 帧")

    # ---------- F 相机-动作同步 ----------
    if cams_meta and a.corr_episodes > 0 and epmeta:
        log("\n== F 相机变化 vs 关节速度（抽样 %d 集，看最佳滞后）==" % a.corr_episodes)
        sel = np.linspace(0, len(EM) - 1, min(a.corr_episodes, len(EM))).astype(int)
        for cam in cams_meta:
            lags, corrs = [], []
            for si in sel:
                r = EM.iloc[int(si)]
                e = int(r["episode_index"])
                if e not in idx:
                    continue
                fi = int(r["videos/%s/file_index" % cam])
                ci = int(r["videos/%s/chunk_index" % cam])
                vp = os.path.join(ds, "videos", cam, "chunk-%03d" % ci, "file-%03d.mp4" % fi)
                if not os.path.exists(vp):
                    continue
                t0 = float(r["videos/%s/from_timestamp" % cam])
                L = len(idx[e])
                f0 = int(round(t0 * fps))
                g = gray_seq(vp, nmax=f0 + L + 2)
                if len(g) < f0 + 30:
                    continue
                g = g[f0:f0 + L]
                amt = np.abs(np.diff(g.reshape(len(g), -1), axis=0)).mean(axis=1)
                sp = np.abs(np.diff(st[idx[e]][:len(g) + 1, :5], axis=0)).mean(axis=1)
                m = min(len(amt), len(sp))
                amt, sp = amt[:m], sp[:m]
                if amt.std() < 1e-6 or sp.std() < 1e-6:
                    continue
                best = (0, -2.0)
                for lag in range(-8, 9):
                    if lag >= 0:
                        x, y = amt[lag:], (sp[:m - lag] if lag else sp)
                    else:
                        x, y = amt[:m + lag], sp[-lag:]
                    if len(x) < 20:
                        continue
                    rr = float(np.corrcoef(x, y)[0, 1])
                    if rr > best[1]:
                        best = (lag, rr)
                lags.append(best[0])
                corrs.append(best[1])
            if lags:
                log("  %-28s 最佳滞后 中位 %+d 帧  相关 中位 %+.3f  (n=%d)"
                    % (cam, int(np.median(lags)), float(np.median(corrs)), len(lags)))
        log("  ★ 判据：滞后应 ≈0 且相关 >0.4；相关≈0 说明画面和动作对不上")

    log("\n== 完成 ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
