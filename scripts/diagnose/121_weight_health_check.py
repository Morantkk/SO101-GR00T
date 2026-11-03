#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""权重体检：判定"这个 checkpoint 本身有没有问题"（不接机械臂、不看真机效果）。

为什么需要这一套：本项目吃过两次亏 ——
  ① 早期并行下载器只比字节数，留下"大小完全正确、内容全 0"的假文件；
  ② `96` 那种"单点回放、无基线"的审计会把"什么都没做"误判成"学会了"。
所以"权重没问题"必须**逐层证**，不能只看一个分数。

  S0 文件层      分片↔index 一致、张量全可加载、无 NaN/Inf/零方差、参数量分组
  S1 身份层      与官方基座逐张量比对 ⇒ 实证"到底哪些参数变了"（实际解冻范围）
  S3 归一化对账  ckpt 的 experiment_cfg/metadata.json 统计量 vs 从训练数据集现算 ★抓系统性偏差
  S2 加载完整性  Gr00tPolicy 加载后 state_dict 与文件张量逐个相等 ★抓"某模块被随机初始化"
  S4 变换往返    真值 action → normalize → unnormalize 必须回到原值（函数级自检）
  S5 行为 sanity 噪声底 / 输入敏感性（黑图·换 state·推偏 state）/ 输出分布 vs 数据集分布

用法：
  cd /root/setup && HF_HOME=/root/hf HF_ENDPOINT=https://hf-mirror.com \
  /root/miniconda3/envs/gr00t/bin/python 121_weight_health_check.py \
      --ckpt /root/BACKUP_gr00t_all_objects/model \
      --dataset /root/gr00t_data/arm101_pick_all_objects_trim \
      [--base /root/groot_n15_base_start]
`--smoke` 只跑 S0/S3（秒级、不用 GPU、不加载模型）。
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

OK, BAD, WARN = "  [PASS]", "  [FAIL]", "  [warn]"
FAILS = []


def rep(tag, ok, msg="", hard=True):
    print("%s %s%s" % ((OK if ok else (BAD if hard else WARN)), tag, ("  -- " + msg) if msg else ""))
    if not ok and hard:
        FAILS.append(tag)
    return ok


def sec(t):
    print("\n" + "=" * 74 + "\n== " + t + "\n" + "=" * 74)


def read_shard_map(shards):
    """{key: shard_path}，每个分片只打开一次。"""
    m = {}
    for p in shards:
        with open(p, "rb") as fh:                 # 假文件检查：头 8 字节 = safetensors 头长度
            h8 = fh.read(8)
        print("    %-44s head8=%s %s" % (os.path.basename(p), h8.hex(), "ok" if h8.strip(b"\x00") else "ALL-ZERO !!"))
        from safetensors.torch import safe_open
        with safe_open(p, framework="pt", device="cpu") as f:
            for k in f.keys():
                m[k] = p
    return m


def load_ds_stats(dataset):
    """从数据集 parquet 现算 state/action 的 6 个统计量（用于与 ckpt metadata 对账）。"""
    S, A = [], []
    for p in sorted(glob.glob(os.path.join(dataset, "data", "chunk-*", "*.parquet"))):
        df = pd.read_parquet(p)
        S.append(np.vstack([np.asarray(x, float) for x in df["observation.state"].values]))
        A.append(np.vstack([np.asarray(x, float) for x in df["action"].values]))
    return np.vstack(S), np.vstack(A)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--base", default=None, help="官方基座目录（S1 用；不传则跳过 S1）")
    ap.add_argument("--n-dist", type=int, default=40)
    ap.add_argument("--ens", type=int, default=5, help="S5 同观测重复次数（测噪声底）")
    ap.add_argument("--denoise", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="只跑 S0/S3（不用 GPU、不加载模型）")
    o = ap.parse_args()

    ckpt = o.ckpt.rstrip("/")
    print("checkpoint: %s" % ckpt)
    print("dataset   : %s" % o.dataset)

    from safetensors.torch import safe_open
    import torch

    # --------------------------------------------------------------- S0 文件层
    sec("S0 文件层：分片完整性 / 张量可加载 / NaN,Inf,零方差 / 参数量")
    shards = sorted(glob.glob(os.path.join(ckpt, "*.safetensors")))
    if not rep("找到权重分片", len(shards) > 0, "%d 个" % len(shards)):
        return 2
    idx_p = os.path.join(ckpt, "model.safetensors.index.json")
    wmap = json.load(open(idx_p, encoding="utf-8"))["weight_map"] if os.path.exists(idx_p) else None
    if wmap:
        print("  index.json: %d 个张量 / %d 个分片" % (len(wmap), len(set(wmap.values()))))
    file_shard = {}
    tot, groups = 0, {}
    bad_nan, bad_inf, bad_zero, bad_flat = [], [], [], []
    cnt = 0
    print("  单趟逐张量检查（不回读、不用 get_slice、不用 strided 切片）…", flush=True)
    for s in shards:
        with open(s, "rb") as fh:                      # 假文件检查：头 8 字节 = safetensors 头长度
            h8 = fh.read(8)
        print("    %-44s head8=%s %s"
              % (os.path.basename(s), h8.hex(), "ok" if h8.strip(b"\x00") else "ALL-ZERO !!"), flush=True)
        with safe_open(s, framework="pt", device="cpu") as f:
            for k in f.keys():
                file_shard[k] = s
                cnt += 1
                if cnt % 200 == 0:
                    print("    ... %d" % cnt, flush=True)
                tt = f.get_tensor(k)
                n = tt.numel()
                if n == 0:
                    del tt
                    continue
                g = k.split(".")[0]
                groups[g] = groups.get(g, 0) + n
                tot += n
                flat = tt.detach().reshape(-1)
                if n > 2_000_000:                      # 大张量：只取连续首尾各 100 万（省内存 + 躲开 strided 坑）
                    t = torch.cat([flat[:1_000_000], flat[-1_000_000:]])
                else:
                    t = flat
                del tt, flat
                tf = t.float()
                if not torch.isfinite(tf).all():
                    (bad_nan if torch.isnan(tf).any() else bad_inf).append(k)
                else:
                    if float((tf.abs() > 0).float().mean()) < 0.01:
                        bad_zero.append(k)
                    if float(tf.std()) == 0.0:
                        bad_flat.append(k)
                del t, tf
    print("  张量数 %d，总参数 %.3f B" % (len(file_shard), tot / 1e9))
    for g, n in sorted(groups.items(), key=lambda x: -x[1]):
        print("    %-26s %12d  (%.1f%%)" % (g, n, 100.0 * n / tot))
    rep("无 NaN 张量", not bad_nan, str(bad_nan[:5]), hard=False)
    rep("无 Inf 张量", not bad_inf, str(bad_inf[:5]), hard=False)
    rep("无非零率<1% 的张量（防假文件）", not bad_zero, str(bad_zero[:5]))
    rep("无零方差张量", not bad_flat, str(bad_flat[:5]), hard=False)

    cfg_p = os.path.join(ckpt, "config.json")
    if os.path.exists(cfg_p):
        c = json.load(open(cfg_p, encoding="utf-8"))
        print("\n  config.json 关键字段（★ 加载旧模型时 tune_* 必须显式传，别信这里的值）:")
        for k in ("tune_llm", "tune_visual", "tune_projector", "tune_diffusion_model"):
            if k in c:
                print("    %-26s %-6s  <- 构建期值，不代表训练时解冻了" % (k, c[k]))
        for k in ("select_layer", "tune_visual", "project_to_dim"):
            if k in c.get("backbone_cfg", {}):
                print("    backbone_cfg.%-14s %s" % (k, c["backbone_cfg"][k]))
        for k in ("use_vlln", "backbone_embedding_dim", "input_embedding_dim", "hidden_size",
                  "num_layers", "action_horizon", "max_state_dim", "max_action_dim"):
            if k in c.get("action_head_cfg", {}):
                print("    action_head.%-17s %s" % (k, c["action_head_cfg"][k]))

    # --------------------------------------------------------------- S1 身份层
    if o.base:
        sec("S1 身份层：与官方基座逐张量比对 => 实证'到底哪些参数变了'")
        try:
            bshards = sorted(glob.glob(os.path.join(o.base.rstrip("/"), "*.safetensors")))
            rep("找到基座分片", len(bshards) > 0, "%d 个" % len(bshards))
            bmap = {}
            for p in bshards:
                with safe_open(p, framework="pt", device="cpu") as f:
                    for k in f.keys():
                        bmap[k] = p
            print("  基座张量 %d 个" % len(bmap))
            # 按 (基座分片, ckpt 分片) 成对处理，每个分片只开一次
            pairs = {}
            for k in set(bmap) & set(file_shard):
                pairs.setdefault((bmap[k], file_shard[k]), []).append(k)
            same, diff, only_ckpt = 0, [], sorted(set(file_shard) - set(bmap))
            for (bp, cp), ks in pairs.items():
                with safe_open(bp, framework="pt", device="cpu") as fb, \
                     safe_open(cp, framework="pt", device="cpu") as fa:
                    for k in ks:
                        tb, ta = fb.get_tensor(k), fa.get_tensor(k)
                        if ta.shape == tb.shape and torch.equal(ta, tb):
                            same += 1
                        else:
                            diff.append(k)
            print("  完全相同 %d；不同 %d；只在 ckpt 里 %d" % (same, len(diff), len(only_ckpt)))
            pre = {}
            for k in diff:
                key = ".".join(k.split(".")[:2])
                pre[key] = pre.get(key, 0) + 1
            for p, n in sorted(pre.items(), key=lambda x: -x[1])[:10]:
                print("    变了 %-36s %d 个" % (p, n))
            rep("存在被改动的张量（确实是微调产物，不是基座复制）", len(diff) > 0)
            rep("backbone 未被改动（只训了动作头，与 tune_llm/tune_visual=False 相符）",
                all(not k.startswith("backbone") for k in diff),
                "若 backbone 也变了 => tune_* 与预期不符", hard=False)
        except Exception as ex:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            rep("S1 基座比对", False, "%r" % ex, hard=False)

    # --------------------------------------------------------- 数据集统计（S3/S5 共用）
    st = ac = None
    try:
        st, ac = load_ds_stats(o.dataset)
        print("\n  数据集现算：%d 帧  state %s  action %s" % (len(st), st.shape, ac.shape))
    except Exception as ex:  # noqa: BLE001
        print("\n  !! 读数据集失败：%r" % ex)

    # --------------------------------------------------------------- S3 归一化对账
    sec("S3 归一化对账：ckpt 的 metadata.json 统计量 vs 从训练数据集现算  ★直指系统性偏差")
    stats = None
    meta_p = os.path.join(ckpt, "experiment_cfg", "metadata.json")
    if not rep("存在 experiment_cfg/metadata.json", os.path.exists(meta_p),
               "Gr00tPolicy._load_metadata 必须从这里读归一化统计"):
        pass
    else:
        md = json.load(open(meta_p, encoding="utf-8"))
        tag = list(md.keys())[0]
        stats = md[tag]["statistics"]
        print("  embodiment 段 %s；statistics 组 %s" % (tag, sorted(stats.keys())))

    if st is not None and stats:
        for grp, arr in (("state", st), ("action", ac)):
            for sub, sl in (("single_arm", slice(0, 5)), ("gripper", slice(5, 6))):
                got = stats.get(grp, {}).get(sub)
                if not got:
                    continue
                x = arr[:, sl]
                ref = {"min": x.min(0), "max": x.max(0), "mean": x.mean(0), "std": x.std(0),
                       "q01": np.quantile(x, 0.01, axis=0), "q99": np.quantile(x, 0.99, axis=0)}
                print("  -- %s.%s --" % (grp, sub))
                worst = 0.0
                for k in ("min", "max", "mean", "std", "q01", "q99"):
                    if k not in got:
                        continue
                    a = np.array(got[k], dtype=float)
                    b = np.asarray(ref[k], dtype=float)
                    if a.shape != b.shape:
                        print("     %-5s 形状不符 ckpt%s vs data%s" % (k, a.shape, b.shape))
                        worst = 9e9
                        continue
                    rel = float(np.abs(a - b).max()) / max(
                        float(np.abs(b).max()), float(np.abs(a).max()), 1e-6)
                    worst = max(worst, rel)
                    print("     %-5s ckpt %s | 现算 %s | 最大相对差 %.4f"
                          % (k, np.round(a, 2), np.round(b, 2), rel))
                rep("%s.%s 统计量一致（相对差<2%%）" % (grp, sub), worst < 0.02,
                    "最大相对差 %.3f => 归一化用了别的数据集/别的时刻的统计 => 会造出系统性偏差" % worst)

    if o.smoke:
        sec("结论（--smoke）")
        print("  未通过项: %s" % (FAILS if FAILS else "无"))
        print("WEIGHT_HEALTH_DONE")
        return 0 if not FAILS else 1

    # --------------------------------------------------------------- S2 加载完整性
    sec("S2 加载完整性：Gr00tPolicy 加载后 state_dict 与文件张量逐个比对  ★抓'某模块被随机初始化'")
    policy = None
    try:
        from gr00t.data.dataset import LeRobotSingleDataset
        from gr00t.experiment.data_config import load_data_config
        from gr00t.model.policy import Gr00tPolicy
        dcfg = load_data_config("so100_dualcam")
        mc = dcfg.modality_config()
        policy = Gr00tPolicy(model_path=ckpt, embodiment_tag="new_embodiment",
                             modality_config=mc, modality_transform=dcfg.transform(),
                             denoising_steps=o.denoise, device="cuda")
        mods = [(nm, v) for nm, v in vars(policy).items() if isinstance(v, torch.nn.Module)]
        print("  policy 里的 nn.Module: %s" % [m[0] for m in mods])
        best = None
        for nm, m in mods:
            inter = set(m.state_dict()) & set(file_shard)
            if best is None or len(inter) > len(best[2]):
                best = (nm, m, inter)
        if best is None:
            rep("找到可比对的 nn.Module", False, "请把 vars(policy) 的打印发我")
        else:
            nm, m, _ = best
            sd = m.state_dict()
            miss = [k for k in sd if k not in file_shard]
            unexp = [k for k in file_shard if k not in sd]
            print("  '%s': state_dict %d 项；文件 %d 项" % (nm, len(sd), len(file_shard)))
            print("    ckpt 有、文件没有（会保持随机初始化！）: %d 个 %s" % (len(miss), miss[:6]))
            print("    文件有、模型没用上: %d 个 %s" % (len(unexp), unexp[:6]))
            byshard = {}
            for k in set(sd) & set(file_shard):
                byshard.setdefault(file_shard[k], []).append(k)
            neq = []
            for s, ks in byshard.items():
                with safe_open(s, framework="pt", device="cpu") as f:
                    for k in sorted(ks):
                        a = sd[k].detach().cpu()
                        b = f.get_tensor(k)
                        if a.shape != b.shape or not torch.equal(a.to(b.dtype), b):
                            neq.append(k)
                            if len(neq) > 8:
                                break
                if len(neq) > 8:
                    break
            rep("没有'文件里缺失'的张量（不存在随机初始化模块）", not miss, str(miss[:6]))
            rep("加载后权重与文件逐个相等", not neq, str(neq[:6]))
    except Exception as ex:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        rep("S2 加载完整性", False, "%r" % ex)

    # --------------------------------------------------------------- S4 变换往返
    sec("S4 变换往返：真值 action -> normalize -> unnormalize 必须回到原值")
    if policy is not None and ac is not None:
        try:
            tr = policy._modality_transform
            x = ac[10:11]
            d = {"action.single_arm": x[:, :5], "action.gripper": x[:, 5:6],
                 "state.single_arm": st[10:11, :5], "state.gripper": st[10:11, 5:6]}
            z = tr(d)
            back = tr.unapply(z)
            r1 = float(np.abs(np.asarray(back["action.single_arm"]) - x[:, :5]).max())
            rep("action 往返残差 < 1e-3", r1 < 1e-3, "实测 %.6f" % r1)
        except Exception as ex:  # noqa: BLE001
            print("  ! 用 API 跑不通（%r），退化为按 metadata 手工验算：" % ex)
            if stats:
                g = stats["action"]["single_arm"]
                if "q01" in g:
                    lo, hi = np.array(g["q01"]), np.array(g["q99"])
                    v = np.array([-8.5, -37.0, 33.0, 73.0, 2.3])
                    rb = (np.clip((v - lo) / np.maximum(hi - lo, 1e-9), 0, 1) * 2 - 1 + 1) / 2 * (hi - lo) + lo
                    rep("q01-q99 归一化手工往返残差 ~0", float(np.abs(rb - v).max()) < 1e-6,
                        "实测 %.2e（只证明 stats 自洽，不等于管道正确）" % float(np.abs(rb - v).max()), hard=False)

    # --------------------------------------------------------------- S5 行为 sanity
    sec("S5 行为 sanity：噪声底 / 输入敏感性 / 输出分布")
    if policy is not None:
        try:
            ds = LeRobotSingleDataset(dataset_path=o.dataset, modality_configs=mc,
                                      embodiment_tag="new_embodiment", video_backend="decord",
                                      transforms=None)
            lens = [int(v) for v in ds.trajectory_lengths]
            starts = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(int)
            H = len(mc["action"].delta_indices)
            vk = list(mc["video"].modality_keys)
            sk = list(mc["state"].modality_keys)
            lk = list(mc["language"].modality_keys)
            ak = list(mc["action"].modality_keys)
            need = set(vk + sk + lk)

            def run(obs):
                r = policy.get_action(obs)
                return np.concatenate([np.asarray(r[k]).reshape(H, -1) for k in ak if k in r], axis=-1)

            eps = np.linspace(0, len(lens) - 1, min(o.n_dist, len(lens))).astype(int)
            preds, noise = [], []
            for e in eps:
                L = lens[e]
                t = int(np.random.randint(0, max(L - H - 1, 1)))
                it = ds[int(starts[e]) + t]
                obs = {k: np.asarray(it[k]) for k in need if k in it}
                reps = [run({k: v.copy() for k, v in obs.items()}) for _ in range(o.ens)]
                preds.append(reps[0])
                noise.append(float(np.median(np.std(np.stack(reps)[:, 0, :], axis=0))))
            P = np.stack(preds)
            sigma = float(np.median(noise))
            print("  噪声底 sigma（同观测重复 %d 次的预测离散）: %.3f°" % (o.ens, sigma))

            e0 = int(eps[0])
            it0 = ds[int(starts[e0]) + 5]
            b_obs = {k: np.asarray(it0[k]) for k in need if k in it0}
            b0 = run({k: v.copy() for k, v in b_obs.items()})
            it2 = ds[int(starts[int(eps[-1])])]
            cases = {}
            q = {k: v.copy() for k, v in b_obs.items()}
            for k in vk:
                if k in q:
                    q[k] = np.zeros_like(q[k])
            cases["两路相机置黑（视觉通路）"] = q
            q = {k: v.copy() for k, v in b_obs.items()}
            for k in sk:
                if k in q and k in it2:
                    q[k] = np.asarray(it2[k])
            cases["state 换成别集的（状态通路）"] = q
            q = {k: v.copy() for k, v in b_obs.items()}
            for k in sk:
                if k in q:
                    q[k] = np.asarray(q[k]) + 10.0
            cases["state 人为 +10°"] = q
            print("  对输入扰动的响应（首步最大关节变化；判据 > 2sigma = %.2f°）:" % (2 * sigma))
            for nm, ob in cases.items():
                d = float(np.abs(run(ob)[0] - b0[0]).max())
                rep("输入敏感性: %s" % nm, d > 2 * sigma,
                    "变化 %.2f°（几乎不变 => 该通路是死的）" % d, hard=False)

            if ac is not None:
                print("  输出分布 vs 数据集 action 分布（首步）:")
                print("    %-11s %10s %10s %10s %10s" % ("关节", "pred均值", "data均值", "pred_std", "data_std"))
                for j, n in enumerate(["pan", "lift", "elbow", "wrist_flex", "wrist_roll", "gripper"]):
                    print("    %-11s %10.2f %10.2f %10.2f %10.2f"
                          % (n, P[:, 0, j].mean(), ac[:, j].mean(), P[:, 0, j].std(), ac[:, j].std()))
                ratio = float(np.median([P[:, 0, j].std() / max(ac[:, j].std(), 1e-6) for j in range(5)]))
                rep("输出波动幅度与数据同量级（比值 0.3~3）", 0.3 < ratio < 3.0,
                    "中位比值 %.2f（太小=回归均值/没学，太大=发散）" % ratio, hard=False)
                lo, hi = ac[:, :5].min(0), ac[:, :5].max(0)
                outr = float(((P[:, 0, :5] < lo - 5) | (P[:, 0, :5] > hi + 5)).any(1).mean())
                rep("输出越界比例 <10%", outr < 0.1, "实测 %.1f%%" % (100 * outr), hard=False)
        except Exception as ex:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            rep("S5 行为 sanity", False, "%r" % ex)

    sec("结论")
    if FAILS:
        print("  ★ 有 %d 项未通过：" % len(FAILS))
        for f in FAILS:
            print("    - %s" % f)
        print("  => 先解决这些，再谈'模型效果好不好'。")
    else:
        print("  [PASS] 所有硬判据通过 => 这个 checkpoint 在 文件/身份/加载/归一化 四层是干净的。")
        print("  注意：S5 只是 sanity，不是能力测试。'学没学会/能不能抓'仍要看 120 的相位曲线 + 真机。")
    print("WEIGHT_HEALTH_DONE")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
