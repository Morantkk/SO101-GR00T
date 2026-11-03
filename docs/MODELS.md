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

### 方案 A：从 Hugging Face 下载（最快）

```bash
hf download YOUR-HF-NAME/groot-so101-pick-one-pen-v1 --local-dir ./groot-v1
hf download YOUR-HF-NAME/groot-so101-all-objects-v3  --local-dir ./groot-v3
```

### 方案 B：自己训（能复现，6 小时）

整套流程见 [04_TRAINING.md](04_TRAINING.md)。

### 方案 C：把本地权重发布到 Hugging Face（仓库维护者用）

**为什么权重不进 GitHub**：单个 `safetensors` 5 GB，GitHub 单文件硬上限 100 MB；
Git LFS 免费额度只有 1 GB 存储 + 1 GB/月流量，两个模型 14.2 GB 需要付费。
HF Hub 对公开模型免费、支持分片与断点续传，是社区标准做法。

```bash
pip install -U huggingface_hub
hf auth login                     # token 在 https://huggingface.co/settings/tokens 生成，勾 write

# 干跑：检查文件完整性 + 生成 model card，不联网
python scripts/tools/publish_models_to_hf.py \
  --ckpt E:/trained_models/groot_arm101_pick_one_pen_v1 \
  --kind v1 --repo YOUR-HF-NAME/groot-so101-pick-one-pen-v1 --dry-run

# 真上传（14.2 GB，家宽上行 30 Mbps 约 1 小时/个）
python scripts/tools/publish_models_to_hf.py \
  --ckpt E:/trained_models/groot_arm101_pick_one_pen_v1 \
  --kind v1 --repo YOUR-HF-NAME/groot-so101-pick-one-pen-v1
```

脚本会自动做四件事：

1. **校验**：必需文件是否齐全 + 对每个 `safetensors` 抽 4 个偏移检查是否全零
   （防"大小正确、内容全 0"的假权重）
2. **生成 model card**：任务、动作空间、训练参数、可训部分、已知限制
3. **生成 `SHA256SUMS.txt`**
4. `create_repo` + `upload_folder`（自动跳过 `checkpoint-*` 与 `runs/`）

上传后把 README「模型权重」表里的 `YOUR-HF-NAME` 换成你的用户名即可。

> 国内网络：`export HF_ENDPOINT=https://hf-mirror.com` 只适合**下载**，
> 上传请直连或挂代理，镜像站一般不支持写。

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
