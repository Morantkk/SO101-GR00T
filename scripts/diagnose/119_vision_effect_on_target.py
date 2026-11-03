#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""119_vision_effect_on_target.py —— **把"视觉对落点的贡献"换算成厘米**（离线，不接机械臂）。

要回答的问题：
  模型在"该合爪"的时刻，到底靠什么决定**往哪儿伸**？靠图像，还是靠关节角自己推？
  如果靠图像 ⇒ 换一张别人episode的图，预测落点应该明显移动；
  如果靠关节角 ⇒ 换图几乎不动，那它永远会去"平均位置"，物体一偏就夹空。

做法（关键：用**正运动学**把动作换算成末端三维坐标，单位变成米/厘米）：
  1. 在数据集里对每集找"合爪时刻"（夹爪从张开 >20 第一次掉到 <10 的那一帧）；
  2. 共 3 次前向：
       A = pred(state_i, img_i)      ← 自己的观测
       B = pred(state_i, img_j)      ← **同一个 state，换成别人 episode 的图**
       C = pred(state_j, img_j)      ← 别人自己的观测
     取动作块最后一步（16 步 ≈ 1.07s 后），用 FK 换算成末端位置 P。
  3. 三个量（都是**位置差，单位 cm**）：
       d_img   = |P(B) − P(A)|   同一状态、只换图 → **图像的影响**
       d_state = |P(B) − P(C)|   同一张图、换状态 → **关节角的影响**
       span    = |P(gt_i) − P(gt_j)|  不同 episode 真实目标之间的差 → **需要分辨的尺度**
  4. 判读：`d_img << span` ⇒ 图**不足以**分辨物体 ⇒ 落点被"平均化" ⇒ 夹空。

FK 常量直接写死（取自 refs/Lerobot-GR00TN1.5/sim_scripts/isaac_models/so101_9_28.urdf），
所以本脚本**不依赖任何 URDF 文件**，服务器上单文件就能跑。
⚠️ URDF 零位与机器人标定零位存在常量偏差（演示帧约 18% 算出 z<0）⇒
   只看"差值的量级/方向"，别当绝对坐标用。

用法：
  /root/miniconda3/envs/gr00t/bin/python 119_vision_effect_on_target.py \
      --ckpt /root/BACKUP_gr00t_all_objects/model \
      --dataset /root/gr00t_data/arm101_pick_all_objects_trim --n 24
"""
import argparse
import json
import os
import sys
import time

import numpy as np

# ---- URDF 运动学链（base → gripper_frame），顺序即串联顺序 ----
CHAIN = [
    ("shoulder_pan",  [0.0388353, -8.97657e-09, 0.0624],  [3.14159, 4.18253e-17, -3.14159]),
    ("shoulder_lift", [-0.0303992, -0.0182778, -0.0542],  [-1.5708, -1.5708, 0.0]),
    ("elbow_flex",    [-0.11257, -0.028, 1.73763e-16],    [0.0, 0.0, 1.5708]),
    ("wrist_flex",    [-0.1349, 0.0052, 3.62355e-17],     [0.0, 0.0, -1.5708]),
    ("wrist_roll",    [5.55112e-17, -0.0611, 0.0181],     [1.5708, 0.0486795, 3.14159]),
    (None,            [-0.0079, -0.000218121, -0.0981274], [0.0, 3.14159, 0.0]),   # 固定：gripper_frame
]
JN = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def _R_rpy(rpy):
    import math
    rr, pp, yy = rpy
    cr, sr, cp, sp, cy, sy = (math.cos(rr), math.sin(rr), math.cos(pp),
                              math.sin(pp), math.cos(yy), math.sin(yy))
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def fk(q5):
    """q5 = 前 5 个关节角（度）→ 夹爪末端坐标 (x,y,z)，单位米。"""
    import math
    T = np.eye(4)
    for (name, xyz, rpy) in CHAIN:
        A = np.eye(4)
        A[:3, :3] = _R_rpy(rpy)
        A[:3, 3] = xyz
        T = T @ A
        if name is not None:
            th = math.radians(float(q5[JN.index(name)]))
            K = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)  # axis 全是 (0,0,1)
            B = np.eye(4)
            B[:3, :3] = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)
            T = T @ B
    return T[:3, 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=24, help="用多少集的「合爪时刻」")
    ap.add_argument("--denoise", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    o = ap.parse_args()

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
    vk = list(mc["video"].modality_keys)
    sk = list(mc["state"].modality_keys)
    lk = list(mc["language"].modality_keys)
    ak = list(mc["action"].modality_keys)
    H = len(mc["action"].delta_indices)
    print("数据集 %d 帧 / %d 集，动作块 H=%d" % (len(ds), len(lens), H), flush=True)

    # ---- 找每集的「合爪时刻」：夹爪从 >20 第一次掉到 <10 的那一帧 ----
    # ★ 必须直接读 parquet（快），**不要**用 ds[t] 逐帧扫 —— 那会把整段视频都解码一遍（51k 帧，慢到不可用）
    import glob
    import pandas as pd
    P = []
    for fp in sorted(glob.glob(os.path.join(o.dataset, "data", "chunk-*", "*.parquet"))):
        P.append(pd.read_parquet(fp))
    pdf = pd.concat(P, ignore_index=True)
    ep_col = pdf["episode_index"].values
    st_col = np.vstack([np.asarray(x, dtype=float) for x in pdf["observation.state"].values])
    epu = np.unique(ep_col)
    print("parquet: %d 帧 / %d 集（与 ds 的集数一致: %s）"
          % (len(pdf), len(epu), len(epu) == len(lens)), flush=True)

    picks = []
    for e, ei in enumerate(epu):
        idx = np.where(ep_col == ei)[0]
        g = st_col[idx, 5]
        i = None
        for t in range(1, len(g)):
            if g[t - 1] > 20 and g[t] < 10:
                i = t
                break
        if i is None or i < 8 or i > len(g) - H - 2:
            seg0 = min(40, max(1, len(g) - 1))
            i = int(np.argmin(g[seg0:]) + seg0)          # 退而求其次：前段最闭合处
        picks.append((e, int(i)))
    picks = picks[:o.n] if len(picks) >= o.n else picks
    print("取到 %d 集的合爪时刻" % len(picks), flush=True)

    t0 = time.time()
    policy = Gr00tPolicy(model_path=o.ckpt, embodiment_tag="new_embodiment",
                         modality_config=mc, modality_transform=dcfg.transform(),
                         denoising_steps=o.denoise, device="cuda")
    print("模型加载 %.1fs" % (time.time() - t0), flush=True)

    def run(ep, t, img_from=None):
        """预测 ep 第 t 帧的动作块；img_from=(ep2,t2) 时把图像换成那一帧的。"""
        it = ds[int(starts[ep]) + t]
        obs = {k: np.asarray(it[k]) for k in vk + sk + lk if k in it}
        if img_from is not None:
            it2 = ds[int(starts[img_from[0]]) + img_from[1]]
            for k in vk:
                obs[k] = np.asarray(it2[k])
        res = policy.get_action(obs)
        return np.concatenate([np.asarray(res[x]) for x in ak if x in res], axis=-1)  # (H,6)

    rows = []
    for idx, (e, t) in enumerate(picks):
        A = run(e, t)                      # (state_i, img_i)
        j = picks[(idx + 1) % len(picks)]
        B = run(e, t, img_from=j)          # (state_i, img_j)  ← 只换图
        C = run(j[0], j[1])                # (state_j, img_j)  ← 别人自己的
        it = ds[int(starts[e]) + t]
        gt = np.asarray(it[ak[0]]).reshape(H, -1)
        g = np.concatenate([gt, np.asarray(it[ak[1]]).reshape(H, -1)], axis=-1)
        rows.append((e, t, A, B, C, g))
        print("  [%2d/%2d] ep%03d t=%d  完成" % (idx + 1, len(picks), e, t), flush=True)

    def P(a6):        # 动作(6维) → 末端坐标，用块的最后一步
        return fk(np.asarray(a6)[-1, :5])

    d_img, d_state, span, err_self = [], [], [], []
    for i in range(len(rows)):
        e, t, A, B, C, g = rows[i]
        d_img.append(P(B) - P(A))            # 只换图
        d_state.append(P(B) - P(C))          # 只换状态
        err_self.append(P(A) - P(g))
        j = (i + 1) % len(rows)
        span.append(P(g) - P(rows[j][5]))    # 真实目标之间的差
        # 下一个循环会用 (i+1) 的观测，这正好是 C 的定义，保留

    def rep(name, arr, axis_names="xyz"):
        a = np.array(arr)                # (N,3)
        d3 = np.linalg.norm(a, axis=1) * 100
        xy = np.linalg.norm(a[:, :2], axis=1) * 100
        z = np.abs(a[:, 2]) * 100
        print("  %-34s 三维 %6.2f cm ｜ 水平 %6.2f cm ｜ 竖直 %6.2f cm   (中位, n=%d)"
              % (name, np.median(d3), np.median(xy), np.median(z), len(a)))

    print("\n=== 落点差异（动作块最后一步的末端位置，中位数）===")
    rep("★ 只换图 d_img（同 state）", d_img)
    rep("  只换 state d_state（同图）", d_state)
    rep("  真实目标间差异 span", span)
    rep("  模型自预测误差 err_self", err_self)

    di = np.median(np.linalg.norm(np.array(d_img)[:, :2], axis=1) * 100)
    ds_ = np.median(np.linalg.norm(np.array(d_state)[:, :2], axis=1) * 100)
    sp = np.median(np.linalg.norm(np.array(span)[:, :2], axis=1) * 100)
    print("\n=== 判读（只看水平方向）===")
    print("  换图引起的落点移动 : %6.2f cm" % di)
    print("  换状态引起的落点移动: %6.2f cm" % ds_)
    print("  需要分辨的目标间距 : %6.2f cm" % sp)
    if sp > 1e-6:
        print("  d_img / span = %.2f" % (di / sp))
    if di < 0.4 * sp:
        print("  ★★ **图像信息不足以定位物体**：换一张别人的图，落点几乎不动 ⇒ 落点被平均化 ⇒ 物体一偏就夹空")
    elif ds_ > 1.5 * di:
        print("  ★ 落点主要由**关节角**决定（图像影响明显小于状态）⇒ 同样偏向 'state 推轨迹'")
    else:
        print("  ✓ 图像对落点有实质影响（量级与目标间距可比）⇒ 视觉在起作用，问题可能在别处")

    out = {"n": len(rows), "d_img_cm": di, "d_state_cm": ds_, "span_cm": sp,
           "err_self_cm": float(np.median(np.linalg.norm(np.array(err_self)[:, :2], axis=1) * 100))}
    with open("/root/logs/vision_target_effect.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nVISION_TARGET_DONE  -> /root/logs/vision_target_effect.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
