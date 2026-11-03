#!/usr/bin/env python
"""把数据集 meta/stats.json 规整成官方 Isaac-GR00T 的格式。

背景：官方 gr00t/data/dataset.py 的 `_get_metadata()` 会这样用 stats：
    for stat_name in le_statistics[le_modality]:          # le_modality ∈ {observation.state, action}
        indices  = np.arange(state_action_meta.start, end)
        stat     = np.array(le_statistics[le_modality][stat_name])
        ... = stat[indices].tolist()
即：它假定**每个** stat 都是长度 == 特征维度（本例 6）的数组。
而 LeRobot 自带的 stats.json 里 `count` 是长度 1 的数组 → `stat[[0..4]]` 直接
IndexError: index 1 is out of bounds for axis 0 with size 1。

官方 `calculate_dataset_statistics()` 产生的正是 canonical 格式：
    每个 key 只含 mean/std/min/max/q01/q99，且都是长度 6 的数组。
所以本脚本按官方公式重算，只保留 loader 真正读取的两张表。

用法: python 91_fix_stats.py [DATASET_DIR]
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DS = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/gr00t_data/arm101_pick_one_pen")
stats_path = DS / "meta" / "stats.json"
backup = stats_path.with_suffix(".json.orig_le")

if stats_path.exists() and not backup.exists():
    shutil.copy2(stats_path, backup)
    print("backed up ->", backup)

parquet_files = sorted(DS.glob("data/*/*.parquet"))
print("parquet files:", len(parquet_files))
assert parquet_files, "no parquet found under data/*/*.parquet"
df = pd.concat([pd.read_parquet(p) for p in parquet_files], axis=0)
print("columns:", list(df.columns))

out = {}
for key in ["observation.state", "action"]:
    arr = np.vstack([np.asarray(x, dtype=np.float32) for x in df[key]])
    assert arr.ndim == 2, f"{key} not 2-D: {arr.shape}"
    out[key] = {
        "mean": np.mean(arr, axis=0).tolist(),
        "std": np.std(arr, axis=0).tolist(),
        "min": np.min(arr, axis=0).tolist(),
        "max": np.max(arr, axis=0).tolist(),
        "q01": np.quantile(arr, 0.01, axis=0).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).tolist(),
    }
    print(f"{key}: shape={arr.shape} min={np.round(arr.min(0),1)} max={np.round(arr.max(0),1)}")

stats_path.write_text(json.dumps(out, indent=4))
print("wrote", stats_path)
