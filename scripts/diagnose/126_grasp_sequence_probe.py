#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""它到底有没有学会"依次抓多个物体"？—— 用**真实观测**逐次考它。

思路（零干预、输入自洽）：
  对每条演示的**每一次合爪**（gripper 谷底），取它前 15 帧（≈1s）的真实观测 `(state, image)`，
  让模型预测 16 步动作块，取**第 16 步**（≈1.07s 后，正好落在合爪点）当作"它认为该在哪儿合爪"。
  再和**该次真值合爪时的实际关节角**比。

关键判据（不是误差，而是**散布**）：
  · 真值：第 1/2/3 次合爪的 pan 跨集 std ≈ 33°（因为每次抓的笔在别的方位）
  · 若模型预测的 std **远小于** 真值 ⇒ **它没有"跟着物体换目标"**，而是每次都给一个**平均位置**
    ⇒ 也就是说：**它没学会"依次抓"这件需要看画面决定下一支在哪的事**
  · 若 std 接近真值、且逐次相关 ⇒ 它会依次抓，真机失败是闭环问题（不是没学会）

用法：
  python 126_grasp_sequence_probe.py --ckpt /root/BACKUP_gr00t_all_objects/model \
      --dataset /root/gr00t_data/arm101_pick_all_objects_trim --episodes 40 --lead 15
"""
import argparse
import glob
import os
import sys

import numpy as np


def troughs(g, hi=16, lo=12):
    out, i = [], 0
    while i < len(g):
        if g[i] >= hi:
            j = i
            while j < len(g) and g[j] >= lo:
                j += 1
            if j < len(g):
                k = j
                while k + 1 < len(g) and g[k + 1] <= g[k]:
                    k += 1
                out.append(k)
                i = k + 15
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--lead", type=int, default=15, help="在合爪前第几帧提问（15 帧 ≈ 1s ≈ 一个 chunk）")
    ap.add_argument("--denoise", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import pandas as pd
    import torch  # noqa: F401
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.model.policy import Gr00tPolicy
    from gr00t.experiment.data_config import load_data_config

    dcfg = load_data_config("so100_dualcam")
    mc = dcfg.modality_config()
    ds = LeRobotSingleDataset(dataset_path=a.dataset, modality_configs=mc,
                              embodiment_tag="new_embodiment", video_backend="decord",
                              transforms=None)
    lens = [int(x) for x in ds.trajectory_lengths]
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(int)

    policy = Gr00tPolicy(model_path=a.ckpt, embodiment_tag="new_embodiment",
                         modality_config=mc, modality_transform=dcfg.transform(),
                         denoising_steps=a.denoise, device="cuda")

    vk = list(mc["video"].modality_keys)
    sk = list(mc["state"].modality_keys)
    lk = list(mc["language"].modality_keys)
    ak = list(mc["action"].modality_keys)
    JN = ["pan", "lift", "elbow", "wf", "wr", "grip"]

    def flat(it, keys):
        out = []
        for k in keys:
            if k not in it:
                continue
            v = np.asarray(it[k])
            out.append(v.reshape(-1, v.shape[-1])[0] if v.ndim > 1 else v.reshape(-1))
        return np.concatenate(out, axis=-1) if out else np.zeros(0)

    # 读 parquet 拿每集 state（用来找合爪时刻 + 真值）
    S, E = [], []
    for p in sorted(glob.glob(os.path.join(a.dataset, "data", "chunk-*", "*.parquet"))):
        df = pd.read_parquet(p)
        S.append(np.vstack([np.asarray(x, dtype=np.float32) for x in df["observation.state"].values]))
        E.append(np.asarray(df["episode_index"].values))
    ST = np.vstack(S)
    EP = np.concatenate(E)
    epu = np.unique(EP)
    idx = {e: np.where(EP == e)[0] for e in epu}

    sel = np.linspace(0, len(epu) - 1, min(a.episodes, len(epu))).astype(int)
    rec = []      # (k, gt6, pred6, ep, t)
    for si in sel:
        e = epu[si]
        st_e = ST[idx[e]]
        tks = troughs(st_e[:, 5])
        for k, t in enumerate(tks):
            if k >= 4 or t - a.lead < 1:
                continue
            probe = t - a.lead
            it = ds[int(starts[si]) + probe]
            s = flat(it, sk)
            obs = {kk: np.asarray(it[kk]) for kk in vk + lk if kk in it}
            off = 0
            for kk in sk:
                w = np.asarray(it[kk]).shape[-1]
                obs[kk] = s[off:off + w][None, :]
                off += w
            pr = policy.get_action(obs)
            pr = np.concatenate([np.asarray(pr[x]) for x in ak if x in pr], axis=-1)
            pr = pr.reshape(-1, pr.shape[-1])
            rec.append((k, st_e[t].astype(float), pr[-1].astype(float), int(e), int(t)))

    R = rec
    if not R:
        print("!! 没有采到样本")
        return 1

    print("\n采到 %d 个「第 k 次合爪」样本（来自 %d 集）" % (len(R), len(sel)))
    print("\n================ 逐次合爪：真值 vs 模型预测（绝对关节角，度）================")
    print("%-4s %5s | %-40s | %-40s" % ("k", "n", "真值 (pan/lift/elbow)", "预测 (pan/lift/elbow)"))
    for k in range(4):
        rr = [x for x in R if x[0] == k]
        if not rr:
            continue
        G = np.array([x[1] for x in rr])
        P = np.array([x[2] for x in rr])
        print("%-4d %5d | %+7.1f %+7.1f %+7.1f  (std %4.1f) | %+7.1f %+7.1f %+7.1f  (std %4.1f)"
              % (k + 1, len(rr), G[:, 0].mean(), G[:, 1].mean(), G[:, 2].mean(), G[:, 0].std(),
                 P[:, 0].mean(), P[:, 1].mean(), P[:, 2].mean(), P[:, 0].std()))

    print("\n================ ★ 关键：散布比（预测 std ÷ 真值 std）================")
    print("%-4s %5s %12s %12s %12s" % ("k", "n", "pan", "lift", "elbow"))
    ratios = []
    for k in range(4):
        rr = [x for x in R if x[0] == k]
        if len(rr) < 5:
            continue
        G = np.array([x[1] for x in rr])
        P = np.array([x[2] for x in rr])
        rs = []
        for j in (0, 1, 2):
            rs.append(P[:, j].std() / max(G[:, j].std(), 1e-6))
        ratios.append(rs)
        print("%-4d %5d %12.2f %12.2f %12.2f" % (k + 1, len(rr), rs[0], rs[1], rs[2]))
    if ratios:
        m = np.median(np.array(ratios), axis=0)
        print("\n  中位散布比: pan %.2f / lift %.2f / elbow %.2f" % tuple(m))
        print("  ★ 判据：≈1 ⇒ 会跟着物体换目标（学会了依次抓）；<<1（如 0.3~0.6）⇒ **每次都给平均位置**")

    # 误差（绝对角，度）
    print("\n================ 逐次合爪的绝对角误差（预测 − 真值）================")
    for k in range(4):
        rr = [x for x in R if x[0] == k]
        if not rr:
            continue
        G = np.array([x[1] for x in rr])
        P = np.array([x[2] for x in rr])
        d = np.abs(P - G)
        print("  第%d次 (n=%2d): %s   最大 %5.1f°"
              % (k + 1, len(rr), "  ".join("%s %5.1f" % (n, v) for n, v in zip(JN, np.median(d, axis=0))),
                 d.max()))
    print("\nSEQUENCE_PROBE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
