# 模型与数据集对应表

> 权重和数据都不在本仓库里（单个 7.2 GB，GitHub 放不下）。
> 这一页说清楚**每个模型是用哪份数据、什么参数训出来的**，以及怎么自己复现。

## 一、两个模型

| | **v1 · 抓单支笔** | **v3 · 桌面全物体** |
|---|---|---|
| 目录名 | `groot_arm101_pick_one_pen_v1` | `BACKUP_gr00t_all_objects_v3` |
| 任务 | `Pick up a pen and put it into the box` | `Put all the objects on the table into the box` |
| 基座 | `nvidia/GR00T-N1.5-3B` | 同左 |
| 训练框架 | Isaac-GR00T 官方 | 同左 |
| 步数 | **10,000** | **30,000** |
| batch | 16 | 16 |
| epoch | ≈9.0 | ≈9.4 |
| 数据 | 50 集 / 17,789 帧 | 102 集 / 51,209 帧 |
| 训练时长 | ≈2.6 h | 6 h 20 m |
| 最终 loss | — | 0.08 |
| 真机结果 | ✅ 能完成 | ❌ 抓空（数据时间轴错位） |

两个模型都**只训练动作头**：backbone（SigLIP2 + Qwen3-1.7B）与官方基座**逐字节相同**，
314 个动作头张量与基座**全部不同**。

> 提示：`config.json` 里可能写着 `tune_visual: true`，那是构建期写入的值，
> **不代表训练时解冻了视觉塔**。判断依据是逐张量比对（`scripts/diagnose/121`）。

## 二、数据集

| 名称 | 集数 / 帧数 | 用在哪 | 状态 |
|---|---|---|---|
| `arm101_pick_one_pen` | 50 / 17,789 | v1 | 原始录制（lerobot v3 格式，腕部 1280×720） |
| `arm101_pick_one_pen_trim` | 50 / — | v1 | 裁掉开头静止段 |
| `arm101_pick_all_objects` | 102 / 59,314 | v3 | 原始录制 |
| `arm101_pick_all_objects_trim` | 102 / 51,209 | **v3 训练实际用的** | ⚠️ `timestamp` 未归零，图像超前 3~8 s |
| `arm101_pick_all_objects_trim2` | 102 / 51,157 | 修正版 | ✅ 时间戳归零、残留静止帧 0 |

**关键**：v3 那份数据的错位不是"训练时出的问题"，而是数据本身就有。
换一份对齐的数据、**同样的超参**重训，loss 从 0.08 掉到 **0.0127** ——
这不是模型变强了，而是任务本身变简单了（不用再"看着未来解释现在"）。

详见 [03_DATA_PIPELINE.md](03_DATA_PIPELINE.md) 第二节。

## 三、怎么拿到权重

### 方案 A：自己训（推荐，能复现）

整套流程 6 小时左右，全部命令在 [04_TRAINING.md](04_TRAINING.md)。

### 方案 B：用 Git LFS 放到 Hugging Face Hub

```bash
# 训练机上
huggingface-cli login
huggingface-cli upload <你的账号>/groot-so101-pick-one-pen-v1 \
  /root/autodl-tmp/gr00t_v1 . --repo-type model
```

目录里至少要包含：

```
config.json
model-00001-of-00002.safetensors
model-00002-of-00002.safetensors
model.safetensors.index.json
experiment_cfg/metadata.json      ← 缺了它服务端起不来（归一化统计）
```

然后在 README 里加一行下载命令即可。

## 四、校验（拿到别人给的权重时必做）

```bash
# ① 结构完整
python scripts/diagnose/96_audit_gr00t_ckpt.py --dir <权重目录>

# ② 非全零 + 真训过
python scripts/diagnose/121_weight_health_check.py --ckpt <权重目录>

# ③ 手动抽样（最省事）
python -c "
import os
p='<权重目录>/model-00001-of-00002.safetensors'
sz=os.path.getsize(p)
with open(p,'rb') as f:
    for off in (0, sz//3, sz//2, sz-64):
        f.seek(off); b=f.read(64)
        print(off, 'non-zero' if any(b) else '★ 全零')
"
```

> **永远不要只比文件大小。** `truncate` 预分配 + 传输中断会留下
> "大小完全正确、内容全 0"的假文件，而且用大小做幂等判断时会被永久跳过。

## 五、权重与数据的匹配关系

服务端加载模型时，`experiment_cfg/metadata.json` 里的归一化统计必须与训练时一致。
**用 A 数据集训的模型，不能拿去跑 B 数据集的推理** —— 归一化基准不同，
输出会系统性偏移。

快速判断两份权重是不是同一次训练：

```bash
md5sum <模型A>/experiment_cfg/metadata.json <模型B>/experiment_cfg/metadata.json
```

相同 ⇒ 同一份数据统计；不同 ⇒ 不同数据集（或至少经过不同预处理）。
