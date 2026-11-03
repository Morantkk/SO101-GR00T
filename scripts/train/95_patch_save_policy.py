#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""补丁：把官方 gr00t_finetune.py 的 checkpoint 策略改小，适配 41G 数据盘。

实测（冒烟）：一个 checkpoint = 16G
    model-00001-of-00002.safetensors  4.7G
    model-00002-of-00002.safetensors  2.5G   ← 模型本体 bf16，共 7.2G
    optimizer.pt                      8.5G   ← AdamW 状态（751M 可训练参数 ×2×fp32）
官方写死 `save_total_limit=5` → 训练中最多 5×16G = 80G，直接把 41G 数据盘撑爆。

改法（幂等，带标记与备份）：
  · `save_total_limit=5`  → `3`
  · 新增 `save_only_model=True` → checkpoint 不再存 optimizer.pt（我们不做断点续训）
      ⇒ 每个 checkpoint 7.2G，3 个 = 21.6G，加最后根目录 7.2G ≈ 29G，留 ~13G 余量。

用法: python 95_patch_save_policy.py [--limit N]
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

F = Path("/root/Isaac-GR00T/scripts/gr00t_finetune.py")
MARK = "PATCH(save-policy)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=3)
    o = ap.parse_args()

    s = F.read_text()
    if MARK in s:
        print("已打过补丁，跳过")
        for ln in s.splitlines():
            if "save_total_limit" in ln or "save_only_model" in ln or MARK in ln:
                print("   ", ln.strip())
        return 0

    bak = F.with_suffix(".py.orig_savepolicy")
    if not bak.exists():
        shutil.copy2(F, bak)
        print("备份 ->", bak)

    m = re.search(r"^(\s*)save_total_limit=\d+,.*$", s, flags=re.M)
    if not m:
        print("找不到 save_total_limit 行，放弃")
        return 1
    indent = m.group(1)
    new_block = (
        f"{indent}save_total_limit={o.limit},\n"
        f"{indent}# {MARK}: 每个 ckpt 16G(含 optimizer 8.5G)，官方写死 5 会撑爆数据盘；\n"
        f"{indent}# 我们不续训 → 不存 optimizer，单 ckpt 降到 7.2G。\n"
        f"{indent}save_only_model=True,\n"
    )
    s = s[: m.start()] + new_block + s[m.end():]
    F.write_text(s)
    print("已打补丁:")
    for ln in F.read_text().splitlines():
        if "save_total_limit" in ln or "save_only_model" in ln or MARK in ln:
            print("   ", ln.strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
