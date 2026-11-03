#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""逐帧回放审计（teacher-forced 离线开环）+ 两条基线 + 相位误差曲线。

和 96 的区别（96 只做"单点前向"且默认只抽 32 个点）：
  · 每集按归一化相位**均匀取 per_ep 个时刻**，覆盖整条演示 ⇒ 能画出"误差随相位"的曲线；
  · 同时报**两条基线**，否则"误差小"没有意义：
      B1 保持不动   pred[0] := 当前 state（= 什么都不做）
      B2 平均轨迹   用全体演示在**同一相位 bin** 的 gt[0] 均值当预测（= 只会输出平均动作）
  · 再报"命令幅度"和"方向相关性"，直接回答"模型是动太多、动太少、还是方向乱"。

它测的是"在**真值状态**上模型猜得准不准"（teacher forcing）。
★ 它**测不到闭环漂移** —— 闭环里状态是被模型自己推走的。闭环问题请另外做真机/半闭环实验。

用官方自己的封装（与训练/部署同一条管道）：
  · 数据：LeRobotSingleDataset(transforms=None) → **未归一化**的真实观测/动作（本项目单位 = 度）
  · 推理：gr00t.model.policy.Gr00tPolicy → 内部 normalize → 模型 → unnormalize

用法（在 /root/setup 下）：
  /root/miniconda3/envs/gr00t/bin/python 120_replay_audit.py \
      --ckpt /root/BACKUP_gr00t_all_objects/model \
      --dataset /root/gr00t_data/arm101_pick_all_objects_trim \
      --per-ep 25 --denoise 4 --ensemble 1 \
      --csv /root/logs/replay_audit.csv --npz /root/logs/replay_audit.npz
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np


def free_gb():
    try:
        import subprocess
        out = subprocess.run(
            "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits",
            shell=True, capture_output=True, text=True, timeout=15).stdout.strip().splitlines()
        return int(out[0]) / 1024.0
    except Exception:  # noqa: BLE001
        return None


def med(a, axis=None):
    """中位；对 B2 里因样本太少而留下的 NaN 自动忽略。"""
    if a is None:
        return float("nan")
    a = np.asarray(a, dtype=float)
    return float(np.nanmedian(a)) if axis is None else np.nanmedian(a, axis=axis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True, help="★ 必传：默认值会静默跑成别的数据集")
    ap.add_argument("--per-ep", type=int, default=25, help="每集均匀取多少个时刻（≈ L/per-ep 的间隔）")
    ap.add_argument("--episodes", type=int, default=0, help="只用前 N 集（0 = 全部）")
    ap.add_argument("--denoise", type=int, default=4)
    ap.add_argument("--ensemble", type=int, default=1, help="同一观测采样几次取平均；>1 时顺便给出噪声底")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nbins", type=int, default=20, help="归一化相位分箱数（画曲线用）")
    ap.add_argument("--csv", default="/root/logs/replay_audit.csv")
    ap.add_argument("--npz", default="/root/logs/replay_audit.npz")
    ap.add_argument("--min-free-gb", type=float, default=9.0)
    o = ap.parse_args()

    fg = free_gb()
    if fg is not None and fg < o.min_free_gb:
        print("!! 空闲显存只有 %.1fG < %.1fG —— 拒绝加载（别把训练 OOM 掉）" % (fg, o.min_free_gb))
        return 3
    print("空闲显存 %.1fG，开始逐帧回放审计 %s" % (fg if fg is not None else -1, o.ckpt), flush=True)

    import torch
    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.experiment.data_config import load_data_config
    from gr00t.model.policy import Gr00tPolicy

    torch.manual_seed(o.seed)
    np.random.seed(o.seed)

    dcfg = load_data_config("so100_dualcam")
    mc = dcfg.modality_config()
    ds = LeRobotSingleDataset(dataset_path=o.dataset, modality_configs=mc,
                              embodiment_tag="new_embodiment", video_backend="decord",
                              transforms=None)
    lens = [int(x) for x in ds.trajectory_lengths]
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(int)
    H = len(mc["action"].delta_indices)
    vk = list(mc["video"].modality_keys)
    sk = list(mc["state"].modality_keys)
    lk = list(mc["language"].modality_keys)
    ak = list(mc["action"].modality_keys)

    need = set(vk + sk + lk + ak)
    try:
        _t = [json.loads(l).get("task") for l in
              open(os.path.join(o.dataset, "meta", "tasks.jsonl"), encoding="utf-8") if l.strip()]
    except Exception:  # noqa: BLE001
        _t = None
    print("数据集 %s\n  %d 集 / %d 帧，H=%d，任务文本 %s" % (o.dataset, len(lens), len(ds), H, _t))
    print("  video=%s  state=%s  action=%s" % (vk, sk, ak))

    # ---- 采样：每集按归一化相位均匀取点（跳过尾段 H 帧，避免取到不完整的未来）
    picks = []
    eps = range(len(lens) if o.episodes <= 0 else min(o.episodes, len(lens)))
    for e in eps:
        L = lens[e]
        top = max(L - H - 1, 1)
        m = min(o.per_ep, top + 1)
        ts = np.unique(np.round(np.linspace(0, top, m)).astype(int)) if m > 1 else np.array([0])
        for t in ts:
            picks.append((e, int(t), int(top)))
    print("采样 %d 个回放点（每集约 %d 个）" % (len(picks), o.per_ep))

    t0 = time.time()
    policy = Gr00tPolicy(model_path=o.ckpt, embodiment_tag="new_embodiment",
                         modality_config=mc, modality_transform=dcfg.transform(),
                         denoising_steps=o.denoise, device="cuda")
    print("模型加载 %.1fs（denoise=%d, ensemble=%d）" % (time.time() - t0, o.denoise, o.ensemble),
          flush=True)

    def chunk_of(item, keys):
        arrs = [np.asarray(item[k]) for k in keys if k in item]
        arrs = [a.reshape(a.shape[0], -1) for a in arrs]
        return np.concatenate(arrs, axis=-1)

    rows = []
    noise = []          # ensemble>1 时的噪声（同观测多次采样的预测离散度）
    t0 = time.time()
    for i, (e, t, top) in enumerate(picks):
        item = ds[int(starts[e]) + t]
        obs = {k: np.asarray(item[k]) for k in need if k in item}
        gt = chunk_of(item, ak)        # (H, 6) 真值动作块
        st = chunk_of(item, sk)[0]     # (6,)    当前关节角
        preds = []
        for _ in range(max(1, o.ensemble)):
            p = policy.get_action({k: v.copy() for k, v in obs.items()})
            preds.append(chunk_of(p, ak))
        pred = np.mean(preds, axis=0)
        if len(preds) > 1:
            noise.append(float(np.median(np.std(np.stack(preds)[:, 0, :5], axis=0))))
        d_first = np.abs(pred[0] - gt[0])
        d_b1 = np.abs(st - gt[0])                 # 基线 B1：保持不动
        rows.append(dict(
            ep=e, t=t, phase=t / max(top, 1),
            err_first=d_first, err_all=np.abs(pred - gt).ravel(),
            b1=d_b1,
            gt_move=np.abs(gt[0] - st), pred_move=np.abs(pred[0] - st),
            dir_pred=(pred[0] - st), dir_gt=(gt[0] - st),
            p_first=pred[0].copy(), g_first=gt[0].copy()))   # 绝对动作值（画 pred vs GT 对照用）
        if (i + 1) % 25 == 0 or i + 1 == len(picks):
            el = time.time() - t0
            eta = el / (i + 1) * (len(picks) - i - 1)
            print("  %4d/%4d  已用 %.1f min  预计还要 %.1f min" % (i + 1, len(picks), el / 60, eta / 60),
                  flush=True)

    # ---------- 汇总 ----------
    M = {"first": np.array([r["err_first"] for r in rows]),          # (N,6)
         "all": np.array([r["err_all"] for r in rows]),              # (N,H*6)
         "b1": np.array([r["b1"] for r in rows]),                    # (N,6)
         "gt_move": np.array([r["gt_move"] for r in rows]),          # (N,6)
         "pred_move": np.array([r["pred_move"] for r in rows]),      # (N,6)
         "dirp": np.array([r["dir_pred"] for r in rows]),            # (N,6)
         "dirg": np.array([r["dir_gt"] for r in rows]),              # (N,6)
         "phase": np.array([r["phase"] for r in rows]),
         "ep": np.array([r["ep"] for r in rows]),
         "t": np.array([r["t"] for r in rows])}
    names = ["pan", "lift", "elbow", "wrist_flex", "wrist_roll", "gripper"]

    # 基线 B2：用同一相位 bin 的全体 gt[0] 均值当预测
    # ★ 必须【留一法】：算某样本的 B2 时要把它自己排除，否则 N 小时它给自己"投票"，
    #   会拿到假的完美分（bin 里只剩 1 个样本时误差恒为 0）。
    nb = max(2, o.nbins)
    bidx = np.clip((M["phase"] * nb).astype(int), 0, nb - 1)
    b2 = np.full_like(M["first"], np.nan)
    for b in range(nb):
        m = bidx == b
        k = int(m.sum())
        if k < 2:
            continue
        tot = M["dirg"][m].sum(axis=0)                        # 该 bin 位移之和
        loo = (tot[None, :] - M["dirg"][m]) / (k - 1)         # 留一均值
        b2[m] = np.abs(loo - M["dirg"][m])

    def medN(A):                                             # 忽略 NaN 的中位
        return float(np.nanmedian(A)) if np.isfinite(A).any() else float("nan")

    def line(tag, A):
        v = np.nanmedian(A, axis=1) if A.ndim > 1 else A
        v = v[np.isfinite(v)]
        return "  %-22s 中位 %6.2f°  p95 %6.2f°  最大 %6.2f°   (n=%d)" % (
            tag, float(np.median(v)), float(np.percentile(v, 95)), float(np.max(v)), len(v))

    print("\n================ 总览（首步，逐关节误差再取样本中位）================")
    print(line("模型", M["first"]))
    print(line("B1 保持不动", M["b1"]))
    print(line("B2 平均轨迹", b2))
    r1 = float(np.median(np.median(M["first"], axis=1)) / max(np.median(np.median(M["b1"], axis=1)), 1e-9))
    print("  ★ 模型 / B1 = %.2f      （<1 才算「比什么都不做更好」）" % r1)
    print("  全 horizon（16步×6关节）: 模型 %.2f°   B1 %.2f°"
          % (float(np.median(M["all"])), float(np.median(np.median(M["b1"], axis=1)))))
    if noise:
        print("  ★ 噪声底（同观测 ensemble 多次采样的预测离散，逐样本中位）: %.3f°" % float(np.median(noise)))

    print("\n================ 逐关节（首步误差中位）================")
    print("  %-11s %8s %8s %8s %10s %10s" % ("关节", "模型", "B1", "B2", "真值位移", "模型位移"))
    for j, n in enumerate(names):
        print("  %-11s %8.2f %8.2f %8.2f %10.2f %10.2f" % (
            n, med(M["first"][:, j]), med(M["b1"][:, j]), med(b2[:, j]),
            med(M["gt_move"][:, j]), med(M["pred_move"][:, j])))
    print("  （真值位移 = |gt首步 − 当前state| ⇒ 这一列≈0 的相位没有可读的量）")

    pr = M["dirp"] - M["dirp"].mean(0)
    gr = M["dirg"] - M["dirg"].mean(0)
    print("\n  方向相关性 corr(模型位移, 真值位移) 逐关节:")
    for j, n in enumerate(names):
        c = float((pr[:, j] * gr[:, j]).sum() /
                  (np.linalg.norm(pr[:, j]) * np.linalg.norm(gr[:, j]) + 1e-12))
        print("    %-11s %+.3f" % (n, c))
    print("  （≈0 ⇒ 模型的动作方向与真值无关；≈1 ⇒ 完全跟得上）")

    # ---- ★ 按「这一步该走多大」分层：小动作帧天然没有区分度（15Hz 下单步增量本来就小）
    mag = np.median(M["gt_move"], axis=1)          # 该样本"首步该走多少"（逐关节中位）
    print("\n================ 按「这一步该走多大」分层（★ 小动作帧没有区分度）================")
    print("  该步幅度    样本数       模型        B1        B2   方向相关(lift/elbow)")
    for a, b, tg in [(0.0, 0.5, "<0.5°"), (0.5, 2.0, "0.5~2°"),
                     (2.0, 5.0, "2~5°"), (5.0, 1e9, ">5°")]:
        m = (mag >= a) & (mag < b)
        if not m.any():
            continue
        A = M["dirp"][m] - M["dirp"][m].mean(0)
        B = M["dirg"][m] - M["dirg"][m].mean(0)
        cc = [float((A[:, j] * B[:, j]).sum() /
                    (np.linalg.norm(A[:, j]) * np.linalg.norm(B[:, j]) + 1e-12)) for j in range(6)]
        print("  %-9s %6d %10.2f %9.2f %9.2f   %+.3f / %+.3f" % (
            tg, int(m.sum()), med(M["first"][m]), med(M["b1"][m]), med(b2[m]), cc[1], cc[2]))
    print("  （★ 判读要看「2~5°」和「>5°」两行：只有在大动作帧上，模型和 B1/B2 才分得开）")

    print("\n================ 误差随相位（每 bin 首步误差中位）================")
    print("  相位bin  真值位移  模型   B1   B2")
    for b in range(nb):
        m = bidx == b
        if not m.any():
            continue
        print("  %2d(%3d%%)  %7.2f  %6.2f %6.2f %6.2f" % (
            b, int(100 * b / nb), med(M["gt_move"][m]), med(M["first"][m]),
            med(M["b1"][m]), med(b2[m])))

    os.makedirs(os.path.dirname(o.csv) or ".", exist_ok=True)
    with open(o.csv, "w", encoding="utf-8") as f:
        f.write("ep,t,phase,err_first_med,err_first_p90,err_all_med,b1_med,"
                "gt_move_med,pred_move_med,"
                + ",".join("e_" + n for n in names)
                + "," + ",".join("p_" + n for n in names)
                + "," + ",".join("g_" + n for n in names) + "\n")
        for r in rows:
            f.write("%d,%d,%.4f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%s,%s,%s\n" % (
                r["ep"], r["t"], r["phase"],
                float(np.median(r["err_first"])), float(np.percentile(r["err_first"], 90)),
                float(np.median(r["err_all"])), float(np.median(r["b1"])),
                float(np.median(r["gt_move"])), float(np.median(r["pred_move"])),
                ",".join("%.3f" % v for v in r["err_first"]),
                ",".join("%.3f" % v for v in r["p_first"]),
                ",".join("%.3f" % v for v in r["g_first"])))
    np.savez_compressed(o.npz, first=M["first"], all=M["all"], b1=M["b1"], b2=b2,
                        gt_move=M["gt_move"], pred_move=M["pred_move"],
                        dirp=M["dirp"], dirg=M["dirg"],
                        p_first=np.array([r["p_first"] for r in rows]),
                        g_first=np.array([r["g_first"] for r in rows]),
                        phase=M["phase"], ep=M["ep"], t=M["t"],
                        names=np.array(names), per_ep=o.per_ep, nbins=nb,
                        ratio_model_over_b1=r1)
    print("\n写出 %s 和 %s" % (o.csv, o.npz))
    print("REPLAY_AUDIT_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
