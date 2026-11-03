#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""画 arm101_pick_one_pen 两阶段微调的训练曲线（LoRA 8k / 全量 10k）。

数据来自 server 端 70_parse_train_log.py 解析的 outputs/train_curves_arm101.csv。

用法:
  <venv-python> scripts/plot_train_curves.py
输出:
  outputs/train_curves_arm101.png
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CSV = os.path.join(ROOT, "outputs", "train_curves_arm101.csv")
OUT = os.path.join(ROOT, "outputs", "train_curves_arm101.png")

# 中文字体：Windows 有 msyh 就用中文标签，否则退回英文（避免方块乱码）
CJK = None
for cand in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyh.ttf", r"C:\Windows\Fonts\simhei.ttf"):
    if os.path.exists(cand):
        CJK = cand
        break
if CJK:
    from matplotlib import font_manager
    font_manager.fontManager.addfont(CJK)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=CJK).get_name()
plt.rcParams["axes.unicode_minus"] = False
L = (lambda zh, en: zh) if CJK else (lambda zh, en: en)


def load():
    series = {}
    cur = None
    with open(CSV, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("###"):
                _, name, total, n = line.split()
                cur = name
                series[cur] = {"total": int(total), "n": int(n), "rows": []}
            elif line.startswith("step,"):
                continue
            elif cur:
                p = line.split(",")
                series[cur]["rows"].append(tuple(float(x) for x in p))
    for k, v in series.items():
        a = np.array(v["rows"], dtype=float)
        v["step"], v["loss"], v["grdn"], v["lr"], v["epoch"] = a.T
    return series


def ema(x, alpha=0.06):
    out = np.empty_like(x)
    m = x[0]
    for i, v in enumerate(x):
        m = alpha * v + (1 - alpha) * m
        out[i] = m
    return out


def main():
    S = load()
    order = [k for k in ("lora", "full") if k in S]
    style = {
        "lora": ("#2563eb", "LoRA (r=16, 8k)", "LoRA"),
        "full": ("#dc2626", "Full (SigLIP+ActionHead, 10k)", "全量"),
    }

    fig, axes = plt.subplots(2, 1, figsize=(11, 8.2), sharex=True,
                             gridspec_kw={"height_ratios": [2.6, 1]})
    ax, ax2 = axes

    for name in order:
        v = S[name]
        color, en, zh = style[name]
        ax.plot(v["step"], v["loss"], color=color, lw=0.7, alpha=0.22)
        ax.plot(v["step"], ema(v["loss"]), color=color, lw=2.0,
                label="%s" % (zh if CJK else en))
        fx, fy = v["step"][-1], v["loss"][-1]
        ax.annotate("%.3f" % fy, (fx, fy), textcoords="offset points", xytext=(6, 6),
                    color=color, fontsize=9, fontweight="bold")
        ax2.plot(v["step"], v["lr"], color=color, lw=1.6,
                 label="%s" % (zh if CJK else en))

    ax.set_ylabel(L("训练损失 (MSE)", "Loss (MSE)"), fontsize=11)
    ax.set_title(L("arm101_pick_one_pen 两阶段微调 · 训练曲线（GR00T N1.5 / lerobot 0.5.1）",
                   "arm101_pick_one_pen two-stage finetune - training curves (GR00T N1.5)"),
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, ls=":")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(bottom=0)

    ax2.set_ylabel(L("学习率", "Learning rate"), fontsize=10)
    ax2.set_xlabel(L("训练步数 (step)", "Training step"), fontsize=11)
    ax2.grid(alpha=0.3, ls=":")

    # 底部注释：关键指标
    lines = []
    for name in order:
        v = S[name]
        lines.append("%s: %d steps | final loss %.3f | epoch %.2f"
                     % (("LoRA" if name == "lora" else "Full"), v["total"], v["loss"][-1], v["epoch"][-1]))
    fig.text(0.012, 0.012, "   |   ".join(lines) + "   (细线=原始, 粗线=EMA平滑)",
             fontsize=8.5, color="#555555")

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(OUT, dpi=160)
    print("saved:", OUT)
    for name in order:
        v = S[name]
        print("  %-5s n=%d  loss first=%.3f last=%.3f  min=%.3f  epoch_end=%.2f"
              % (name, v["n"], v["loss"][0], v["loss"][-1], v["loss"].min(), v["epoch"][-1]))


if __name__ == "__main__":
    sys.exit(main())
