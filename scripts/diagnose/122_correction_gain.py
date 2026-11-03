#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""122_correction_gain.py —— 测模型的「状态纠正增益」∂pred/∂state（离线，不接机械臂）。

★ 要回答的问题
  `120`（逐帧回放）证明：在**真值观测**上，模型单步预测的形状几乎完美（corr +0.999，误差 ~1.3°）。
  但真机上却出现：① 卡在初始位姿不动 ② 起来后定位不准、下探抓空 ③ 抓空后只在半空开合。
  为什么"开卷考得好"却"上岗不行"？—— 关键在于**闭环**里，状态不是喂进去的，而是被模型自己
  上一步动作推出来的：`s_{t+1} ≈ pred(s_t)`。于是闭环的稳定性完全由**增益 g = ∂pred/∂s** 决定：

      |g| < 1 → 收缩 ⇒ 偏差自己衰减（稳）
      g ≈ 1   → 完全"照抄当前状态" ⇒ 偏差原样保留（随机偏差按 √t 走，系统性偏差按 t 线性累积）
      g > 1   → 发散
      g ≈ 0   → 输出与当前状态无关 ⇒ 一定会去同一个绝对目标（稳，但那个不动点对不对另说）

  这个量在 `120` 那种"每步都喂真值"的测法里**永远测不到**（被重新锚定了）。

★ 怎么测（同一个观测点跑 3 种输入）
  A  基准：  (state_t,            image_t)                 → pred_A
  B  只推状态：(state_t + δ·e_j,    image_t)                 → pred_B      ⇒ J_B[j,:] = (pred_B[0]-pred_A[0])/δ
  C  自洽推状态：(state_t + δ·e_j,  image_nn(state_t+δ·e_j)) → pred_C      ⇒ J_C[j,:]
     （C 在同一集 ±W 帧窗口里找"状态最接近 s+δ"的那一帧的真实图像 ⇒ 模拟"臂真的漂到那儿、相机也拍到了"）

  ★ B 是"输入自相矛盾"（state 说偏了、画面说没偏），只用来量**敏感度**；
    C 才是**闭环的近似**，判读以 C 为准。B 与 C 的差 = "图像能把状态拉回多少"。

  同时给**噪声底**（同一输入重复采样）—— 差异不到 2σ 的一律不算。

用法（/root/setup 下）：
  /root/miniconda3/envs/gr00t/bin/python 122_correction_gain.py \
      --ckpt /root/BACKUP_gr00t_all_objects/model \
      --dataset /root/gr00t_data/arm101_pick_all_objects_trim \
      --episodes 24 --per-ep 3 --delta 5 --window 60 --denoise 4 --noise-reps 2
"""
import argparse
import json
import os
import time

import numpy as np

JN = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def free_gb():
    try:
        import subprocess
        out = subprocess.run("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits",
                             shell=True, capture_output=True, text=True, timeout=15).stdout.strip().splitlines()
        return int(out[0]) / 1024.0
    except Exception:  # noqa: BLE001
        return None


def chunk_of(item, keys):
    arrs = [np.asarray(item[k]) for k in keys if k in item]
    arrs = [a.reshape(a.shape[0], -1) for a in arrs]
    return np.concatenate(arrs, axis=-1)


def put_state(item, sk, vec):
    """把 state 向量写回 item 的状态键（返回新 dict，不动原 item）。"""
    out, off = {}, 0
    for k in sk:
        if k not in item:
            continue
        a = np.asarray(item[k])
        flat = a.reshape(a.shape[0], -1).copy()
        n = flat.shape[1]
        flat[0] = np.asarray(vec, dtype=float)[off:off + n]
        out[k] = flat.reshape(a.shape).astype(a.dtype)
        off += n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--per-ep", type=int, default=3)
    ap.add_argument("--phase-lo", type=float, default=0.15)
    ap.add_argument("--phase-hi", type=float, default=0.75)
    ap.add_argument("--delta", type=float, default=5.0, help="状态扰动幅度（度）")
    ap.add_argument("--window", type=int, default=60, help="条件 C 在同一集里找最近邻图像的±帧窗口")
    ap.add_argument("--denoise", type=int, default=4)
    ap.add_argument("--noise-reps", type=int, default=2, help="同一输入重复采样次数（测噪声底）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=9.0)
    o = ap.parse_args()
    if o.smoke:
        o.episodes, o.per_ep, o.window = 3, 2, 40

    fg = free_gb()
    if fg is not None and fg < o.min_free_gb:
        print("!! 空闲显存 %.1fG < %.1fG，拒绝加载" % (fg, o.min_free_gb))
        return 3
    print("空闲显存 %.1fG；开始纠正增益测试 %s" % (fg if fg is not None else -1, o.ckpt), flush=True)

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
    vk, sk, ak = list(mc["video"].modality_keys), list(mc["state"].modality_keys), list(mc["action"].modality_keys)
    need = set(vk + sk + ak)
    print("数据集 %s：%d 集 / %d 帧，H=%d" % (o.dataset, len(lens), len(ds), H))
    print("  video=%s  state=%s  action=%s" % (vk, sk, ak), flush=True)

    # ---- 预读全部 state（走 parquet，不解码视频，很快）→ 用于条件 C 的最近邻检索
    allst = {}
    try:
        import pandas as pd
        gs = sorted(__import__("glob").glob(os.path.join(o.dataset, "data", "chunk-*", "*.parquet")))
        cur = 0
        for p in gs:
            df = pd.read_parquet(p)
            S = np.vstack([np.asarray(x, float) for x in df["observation.state"].values])
            E = df["episode_index"].values
            for e in np.unique(E):
                m = E == e
                allst[int(e)] = S[m]
            cur += len(df)
        print("  已预读 %d 集的 state 序列（共 %d 帧）" % (len(allst), cur), flush=True)
    except Exception as ex:  # noqa: BLE001
        print("  ⚠ 预读 state 失败（%r）⇒ 条件 C 退化为只在本集内按帧号搜索" % ex, flush=True)

    # ---- 采样点：相位 [lo,hi] 内均匀取
    picks = []
    for e in range(min(o.episodes, len(lens))):
        L = lens[e]
        top = max(L - H - 1, 1)
        lo, hi = int(o.phase_lo * top), int(o.phase_hi * top)
        if hi <= lo:
            continue
        ts = np.unique(np.round(np.linspace(lo, hi, o.per_ep)).astype(int))
        for t in ts:
            picks.append((e, int(t)))
    print("采样 %d 个观测点（%d 集 × %d）" % (len(picks), min(o.episodes, len(lens)), o.per_ep), flush=True)

    policy = Gr00tPolicy(model_path=o.ckpt, embodiment_tag="new_embodiment",
                         modality_config=mc, modality_transform=dcfg.transform(),
                         denoising_steps=o.denoise, device="cuda")
    print("模型加载完成（denoise=%d）" % o.denoise, flush=True)

    def pred_of(obs):
        p = policy.get_action({k: np.asarray(v).copy() for k, v in obs.items()})
        return chunk_of(p, ak)

    NJ = 5                                    # 只对 5 个臂关节做扰动
    JB = np.full((len(picks), NJ, 6), np.nan)
    JC = np.full((len(picks), NJ, 6), np.nan)
    JC_align = np.full((len(picks), NJ), np.nan)   # 条件 C 实际达成的状态对齐误差
    noise = []
    t0 = time.time()
    for i, (e, t) in enumerate(picks):
        item = ds[int(starts[e]) + t]
        obsA = {k: np.asarray(item[k]) for k in need if k in item}
        st = chunk_of(item, sk)[0]
        pA = []
        for r in range(max(1, o.noise_reps)):
            torch.manual_seed(1000 * o.seed + i * 17 + r)
            pA.append(pred_of(obsA))
        pA = np.stack(pA)
        if len(pA) > 1:
            noise.append(float(np.median(np.std(pA[:, 0, :5], axis=0))))
        pa = pA.mean(axis=0)

        base = allst.get(e)
        for j in range(NJ):
            d = o.delta
            st2 = st.copy()
            st2[j] += d
            # B：只推 state，图不动
            obsB = dict(obsA)
            obsB.update(put_state(item, sk, st2))
            torch.manual_seed(2000 * o.seed + i * 17 + j)
            pB = pred_of(obsB)
            JB[i, j] = -(pB[0] - pa[0]) / d
            # C：推 state + 换成"状态最接近"的真实帧图像（保持自洽）
            if base is not None:
                w0, w1 = max(0, t - o.window), min(len(base), t + o.window + 1)
                if w1 > w0:
                    dd = np.abs(base[w0:w1, :5] - st2[:5]).mean(axis=1)
                    k = int(w0 + np.argmin(dd))
                    JC_align[i, j] = float(dd.min())
                    if k != t:
                        item2 = ds[int(starts[e]) + k]
                        obsC = {kk: np.asarray(item2[kk]) for kk in need if kk in item2}
                        obsC.update(put_state(item2, sk, st2))
                        torch.manual_seed(3000 * o.seed + i * 17 + j)
                        pC = pred_of(obsC)
                        JC[i, j] = -(pC[0] - pa[0]) / d
        if (i + 1) % 8 == 0 or i + 1 == len(picks):
            el = time.time() - t0
            print("  %3d/%3d  已用 %.1f min  预计还要 %.1f min"
                  % (i + 1, len(picks), el / 60, el / (i + 1) * (len(picks) - i - 1) / 60), flush=True)

    # ================== 汇总 ==================
    print("\n================ 噪声底 ================")
    if noise:
        print("  同一观测重复 %d 次，首步预测的逐关节 std 中位 = %.2f°（跨点 p90 %.2f°）"
              % (o.noise_reps, float(np.median(noise)), float(np.percentile(noise, 90))))
        print("  ⇒ 下面任何增益，其引起的首步位移要明显大于 %.2f° 才算数" % (2 * float(np.median(noise))))
    else:
        print("  （--noise-reps 1，未测）")

    def show(J, tag, note):
        print("\n================ %s ================" % tag)
        print("  %s" % note)
        print("  行=被扰动的 state 关节（+%.0f°）；列=模型首步输出关节；值 = −Δpred/δ" % o.delta)
        print("         " + "".join("%11s" % n[:10] for n in JN))
        for j in range(NJ):
            row = np.nanmedian(J[:, j, :], axis=0)
            print("  d/%-8s" % JN[j][:8] + "".join("%11.2f" % v for v in row))
        diag = np.array([np.nanmedian(J[:, j, j]) for j in range(NJ)])
        print("  ⇒ 自增益 diag: " + "  ".join("%s %+.2f" % (JN[j][:8], diag[j]) for j in range(NJ)))
        off = []
        for j in range(NJ):
            for k in range(6):
                if k != j:
                    off.append(abs(float(np.nanmedian(J[:, j, k]))))
        print("  ⇒ 自增益 中位|.| = %.2f    交叉项 中位|.| = %.2f" % (float(np.nanmedian(np.abs(diag))), float(np.median(off) if off else 0)))

    show(JB, "条件 B：只推 state（图像不动）", "平衡输入对状态的敏感度")
    show(JC, "条件 C：推 state + 换成对应真实帧图像（自洽，最接近闭环）", "★ 判读以这一张为准")
    if np.isfinite(JC_align).any():
        print("\n  条件 C 的状态对齐误差（找了最近邻帧后，state 仍差多少）：中位 %.2f°" % float(np.nanmedian(JC_align)))

    print("\n================ 怎么读 ================")
    print("  设 T(s)=模型对**当前步**输出的绝对目标，则实测 g = −ΔT/Δs = −∂T/∂s（δ=+%.0f°）。" % o.delta)
    print("  闭环里 s_{t+1} ≈ T(s_t)，所以决定稳定性/漂移的是 ∂T/∂s = −g：")
    print("    ∂T/∂s ≈ +1  （即 g ≈ −1）→ 目标完全跟着当前状态走 ⇒ **偏差原样保留、甚至放大**，")
    print("                                  闭环里只有累积没有回拉（最坏；系统性偏差按步数线性增长）")
    print("    ∂T/∂s ≈  0  （即 g ≈  0）→ 目标与当前状态无关 ⇒ 一步就把偏差吸收掉（自我纠正，最好）")
    print("    ∂T/∂s  <  0  （g > 0）    → 把状态往回推（过纠正）")
    print("  再把 g 与噪声底比较：|g|·δ 小于 2σ 的，视为“测不出依赖”。")

    np.savez_compressed("/root/logs/correction_gain.npz", JB=JB, JC=JC, align=JC_align,
                        noise=np.array(noise), picks=np.array(picks), names=np.array(JN), delta=o.delta)
    with open("/root/logs/correction_gain.json", "w", encoding="utf-8") as f:
        json.dump({"delta": o.delta, "noise_med": float(np.median(noise)) if noise else None,
                   "diag_B": [float(np.nanmedian(JB[:, j, j])) for j in range(NJ)],
                   "diag_C": [float(np.nanmedian(JC[:, j, j])) for j in range(NJ)],
                   "names": JN}, f, ensure_ascii=False, indent=1)
    print("\n写出 /root/logs/correction_gain.{npz,json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
