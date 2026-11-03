#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""官方 Isaac-GR00T checkpoint 的「学会了没学会」审计 —— 训练集回放。

原理（与上一轮 lerobot 侧同一套判据）：
  训练是 **teacher forcing**：喂数据集的**真实观测**（图像 + state），目标是人类的**动作**。
  所以"能否复现训练集自己的动作"是一票否决的**必要**判据：
     中位关节误差 <10°  ⇒ 学会了本任务（真机仍需实测）
     中位关节误差 >30°  ⇒ 欠拟合
  它不经过相机 / 带宽 / 真机，纯粹考"模型到底有没有把数据学进去"。
  ★ loss 有欺骗性：flow matching 的 loss 是随机 t 上速度场 MSE 的**平均**，
    高噪声段（x_t 里几乎直接含着 ε）极容易压低，"容易的部分"占大头；
    真正决定落点的低噪声收尾在平均里权重很小。所以必须看"积分终点 vs 人类演示"的差。

用官方自己的封装（保证与训练/部署同一条管道）：
  · 数据：`LeRobotSingleDataset(transforms=[])` → 拿**未归一化**的真实观测/动作
  · 推理：`gr00t.model.policy.Gr00tPolicy` → 内部做 normalize → 模型 → unnormalize
  · 对比：原始单位（本项目 = 度）逐关节绝对误差

用法：
  /root/miniconda3/envs/gr00t/bin/python 96_audit_gr00t_ckpt.py \
      --ckpt /root/autodl-tmp/gr00t_arm101/checkpoint-2000 \
      --n 32 --denoise 4 --ensemble 1 [--csv /root/logs/audit_gr00t.csv] [--min-free-gb 9]
"""
import argparse
import csv
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
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint 目录（内含 config.json + safetensors + experiment_cfg）")
    ap.add_argument("--dataset", default="/root/gr00t_data/arm101_pick_one_pen")
    ap.add_argument("--n", type=int, default=32, help="回放样本数")
    ap.add_argument("--denoise", type=int, default=4, help="flow matching 去噪步数")
    ap.add_argument("--ensemble", type=int, default=1, help="同一观测采样几次取平均（压 flow matching 固有随机性）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default="/root/logs/audit_gr00t.csv")
    ap.add_argument("--min-free-gb", type=float, default=9.0, help="加载前要求的最小空闲显存")
    o = ap.parse_args()

    fg = free_gb()
    if fg is not None and fg < o.min_free_gb:
        print("!! 空闲显存只有 %.1fG < %.1fG —— 拒绝加载以免把正在跑的训练 OOM 掉。"
              "（等训练结束，或先停掉训练）" % (fg, o.min_free_gb))
        return 3
    print("空闲显存 %.1fG，开始审计 %s" % (fg if fg is not None else -1, o.ckpt))

    import torch
    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.experiment.data_config import load_data_config
    from gr00t.model.policy import Gr00tPolicy

    torch.manual_seed(o.seed)
    np.random.seed(o.seed)

    # ---- 数据（未归一化、未增广）
    dcfg = load_data_config("so100_dualcam")
    modality_config = dcfg.modality_config()
    ds = LeRobotSingleDataset(
        dataset_path=o.dataset,
        modality_configs=modality_config,
        embodiment_tag="new_embodiment",     # 覆盖成新本体，与训练一致
        video_backend="decord",
        transforms=None,                      # → ComposedModalityTransform([]) 即原样输出
    )
    n_total = len(ds)

    # ★ 任务文本守卫：审计必须用与训练**同一个数据集**。
    #   本脚本的 --dataset 有默认值，换数据集后忘记传参会静默跑成"新模型 × 旧数据"，
    #   指标看着还挺好但毫无意义 → 这里把任务文本打出来，并在"路径是默认值"时显式告警。
    try:
        _t = [json.loads(l).get("task") for l in
              open(os.path.join(o.dataset, "meta", "tasks.jsonl"), encoding="utf-8") if l.strip()]
    except Exception:  # noqa: BLE001
        _t = None
    print("数据集 %d 帧；可回放（录像尾段留出 horizon）" % n_total)
    print("数据集路径: %s" % o.dataset)
    if _t:
        print("任务文本  : %s" % _t)
    if o.dataset == ap.get_default("dataset"):
        print("⚠️  注意：--dataset 用的是**默认值**(%s) —— 若你换过数据集，" % ap.get_default("dataset"))
        print("    这次审计很可能是\\\"新模型 × 旧数据\\\"，指标无效。请显式传 --dataset <本轮数据集>。")

    # 跳过每集的最后 horizon 帧（避免取到不完整的未来）
    horizon = len(modality_config["action"].delta_indices)
    lens = ds.trajectory_lengths
    ids = ds.trajectory_ids
    valid = []
    for t, L in zip(ids, lens):
        start = int(np.sum(lens[: list(ids).index(t)])) if hasattr(ids, "index") else 0
        for k in range(0, max(int(L) - horizon, 1), max(1, int(L) // 12)):
            valid.append(start + k)
    valid = np.array(sorted(set(valid)))
    if len(valid) > o.n:
        sel = np.random.choice(valid, size=o.n, replace=False)
    else:
        sel = valid
    print("采样 %d 个回放点" % len(sel))

    # ---- 模型（官方 Policy：内置 normalize + unnormalize）
    t0 = time.time()
    policy = Gr00tPolicy(
        model_path=o.ckpt,
        embodiment_tag="new_embodiment",
        modality_config=modality_config,
        modality_transform=dcfg.transform(),
        denoising_steps=o.denoise,
        device="cuda",
    )
    print("模型加载 %.1fs，denoising_steps=%d" % (time.time() - t0, o.denoise))

    obs_keys = [k for k in modality_config["video"].modality_keys] + \
               [k for k in modality_config["state"].modality_keys] + \
               [k for k in modality_config["language"].modality_keys]
    act_keys = [k for k in modality_config["action"].modality_keys]

    rows = []
    for j, idx in enumerate(sel):
        item = ds[int(idx)]
        obs = {k: np.asarray(item[k]) for k in obs_keys if k in item}
        gt = {k: np.asarray(item[k]) for k in act_keys if k in item}
        preds = []
        for _ in range(max(1, o.ensemble)):
            p = policy.get_action({k: v.copy() for k, v in obs.items()})
            preds.append(np.concatenate([np.asarray(p[k]).reshape(len(np.asarray(p[k])), -1) if np.asarray(p[k]).ndim > 2
                                         else np.asarray(p[k]) for k in act_keys], axis=-1))
        pred = np.mean(preds, axis=0)            # (horizon, total_action_dim)
        gtcat = np.concatenate([gt[k] for k in act_keys], axis=-1)   # (horizon, total_action_dim)
        d = np.abs(pred - gtcat)
        rows.append({
            "idx": int(idx),
            "err_first_med": float(np.median(d[0])),
            "err_first_max": float(np.max(d[0])),
            "err_all_med": float(np.median(d)),
            "per_joint_first": [round(float(x), 2) for x in d[0]],
            "gt_first": [round(float(x), 2) for x in gtcat[0]],
            "pred_first": [round(float(x), 2) for x in pred[0]],
        })
        if (j + 1) % 8 == 0 or j + 1 == len(sel):
            print("  %d/%d 中位 %.1f° 最大 %.1f°" % (j + 1, len(sel), rows[-1]["err_first_med"], rows[-1]["err_first_max"]), flush=True)

    fm = np.array([r["err_first_med"] for r in rows])
    fx = np.array([r["err_first_max"] for r in rows])
    am = np.array([r["err_all_med"] for r in rows])
    per_joint = np.median(np.array([r["per_joint_first"] for r in rows]), axis=0)
    names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    print("\n================= 审计结果 =================")
    print("ckpt            : %s" % o.ckpt)
    print("样本 / 去噪步数  : %d / %d (ensemble=%d)" % (len(sel), o.denoise, o.ensemble))
    print("首步关节误差 中位: %.2f°   95分位: %.2f°   最大: %.2f°" % (np.median(fm), np.percentile(fm, 95), fm.max()))
    print("全horizon平均误差: %.2f°" % np.median(am))
    print("逐关节中位误差   :")
    for nm, e in zip(names, per_joint):
        print("   %-14s %7.2f°" % (nm, e))
    print("---- 判读 ----")
    med = float(np.median(fm))
    if med < 10:
        print("中位 %.2f° < 10° ⇒ **学会了本任务**（接下来看真机）" % med)
    elif med < 30:
        print("中位 %.2f° ∈ [10,30) ⇒ 边缘：部分学会，建议继续训/多看几个 ckpt" % med)
    else:
        print("中位 %.2f° > 30° ⇒ **欠拟合**（模型没把数据学进去）" % med)
    print("（逐关节量程约 ±90°，相邻帧只差 ~1.2°；夹爪抓取时 GT ≈ 0.9，若预测一直 15~20 就是没闭合）")

    # ---- 记录 CSV，便于跨 checkpoint 比较
    if o.csv:
        new = not os.path.exists(o.csv)
        with open(o.csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["ts", "ckpt", "n", "denoise", "ensemble",
                            "first_med", "first_p95", "first_max", "all_med"] + names)
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), os.path.basename(o.ckpt.rstrip("/")),
                        len(sel), o.denoise, o.ensemble, round(float(np.median(fm)), 2),
                        round(float(np.percentile(fm, 95)), 2), round(float(fm.max()), 2),
                        round(float(np.median(am)), 2)] + [round(float(x), 2) for x in per_joint])
        print("已追加到 %s" % o.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
