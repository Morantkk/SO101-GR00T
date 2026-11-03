# 04 · 微调训练

用 **NVIDIA Isaac-GR00T 官方框架**微调 `GR00T-N1.5-3B`。
注意不要用 lerobot 内置的 `--policy.type=groot`，那是另一个实现（格式与权重都不通用）。

---

## 一、一条命令

```bash
DS=/root/gr00t_data/arm101_pick_one_pen \
OUT=/root/autodl-tmp/gr00t_v1 \
LOG=/root/logs/train_v1.log \
STEPS=10000 SAVE=2000 BATCH=16 \
bash scripts/train/94_train_gr00t_official.sh
```

脚本实际展开成：

```bash
python scripts/gr00t_finetune.py \
  --dataset-path  $DS \
  --output-dir    $OUT \
  --data-config   so100_dualcam \
  --embodiment-tag new_embodiment \
  --video-backend decord \
  --num-gpus 1 \
  --batch-size 16 \
  --max-steps 10000 \
  --save-steps 2000 \
  --dataloader-num-workers 6 \
  --report-to tensorboard
```

> `--base-model-path` **不传**，即从官方 `nvidia/GR00T-N1.5-3B` 基座开始。
> 如果传了旧模型目录续训，`experiment_cfg/metadata.json` 里的归一化统计也会一起带过来，
> 换数据集时极易错位 —— 除非你确定要续训，否则别传。

---

## 二、参数表

| 参数 | v1（抓笔） | v3（全物体） | 说明 |
|---|---|---|---|
| 数据 | 50 集 / 17,789 帧 | 102 集 / 51,209 帧 | |
| `max-steps` | 10,000 | 30,000 | 步数 ≈ epoch × 总帧 / batch |
| `save-steps` | 2,000 | 2,000（配合 limit 3 → 实际每 10k） | |
| `batch-size` | 16 | 16 | 32 G 卡：占 24.4 G |
| `dataloader-num-workers` | 6 | 6 | 再高会跟视频解码抢 CPU |
| `data-config` | `so100_dualcam` | 同 | SO-101 双臂（前视+腕部） |
| `embodiment-tag` | `new_embodiment` | 同 | |
| `video-backend` | `decord` | 同 | `torchvision` 也行但更慢 |
| `tune-visual` | 未开 | 未开 | 视觉塔全程冻结 |
| epoch 数 | ≈9.0 | ≈9.4 | |
| 时长 | ≈2.6 h | 6 h 20 m | 1.3 it/s |

**batch 与显存**（实测）：

| 卡 | batch | 显存占用 | 备注 |
|---|---|---|---|
| RTX 4080S 32 G | 16 | 24.4 G | 可跑 |
| RTX 4090 24 G | 16 | 24.4 G | 会 OOM，降到 12 |
| A100 40 G | 24 | ~33 G | 未验证 |

---

## 三、训练的是什么

日志里这四行才是真开关：

```
Tune backbone vision tower: False     ← SigLIP2 视觉塔 冻结
Tune backbone LLM:          False     ← Qwen3-1.7B  冻结
Warning: No backbone trainable parameters found.
Tune action head projector: True      ← state/action 编解码 训练
Tune action head DiT:       True      ← 扩散去噪主干 训练
```

实际可训约 **0.75 B**：

| 部分 | 参数量 | 状态 |
|---|---|---|
| DiT 主干 | ~550 M | 训练 |
| `state_encoder` / `action_encoder` / `action_decoder` | ~ | 训练 |
| `vlln` + `vl_self_attention` | 201 M | **不受 `tune_*` 控制，恒可训** |
| SigLIP2 + Qwen3-1.7B | ~2.0 B | 冻结 |

> ⚠️ 日志里还会有一行 `Tune backbone visual: True`，那是加载模型时
> `from_pretrained` 的默认值打印出来的，**不代表解冻**。看上面那四行。

冻结视觉塔的直接后果：模型只能"用别人抽好的视觉特征"。而 640×360 进 224×224 之后，
一支笔只占 1 个 token —— 这就是 v1/v3 都会"合爪偏外"的根源（见 [02](02_DATA_COLLECTION.md)）。

---

## 四、checkpoint 保存（默认会撑爆磁盘）

HF Trainer 默认 `save_total_limit=None` + 保存完整 state，**每个 ckpt 16 GB**
（7.2 G 模型 + 7.2 G 优化器状态）。30k 步默认每 500 步存一次 = 60 个 = 960 GB。

`95_patch_save_policy.py` 打两个补丁：

```python
save_total_limit = 3          # 只留最近 3 个
save_only_model  = True       # 不存优化器状态 → 每个 ckpt 7.2 G
```

```bash
python scripts/train/95_patch_save_policy.py        # 每次重装 Isaac-GR00T 都要重放
```

> 这个补丁改的是 `Isaac-GR00T` 仓库里的文件，重装/重 clone 会丢，**必须重放**。

磁盘测算（30k 步、每 10k 存一次）：

```
3 × 7.2 G (ckpt) + 7.2 G (final) ≈ 29 G
```
数据盘 `/root/autodl-tmp` 至少要 35 G 空余。

---

## 五、训练中和训练后

### 看进度

```bash
tail -f /root/logs/train_v1.log          # PowerShell: gc -Wait -Tail 20 /root/logs/train_v1.log
```

或用 `scripts/tools/dashboard_gr00t.py` 起一个本地网页面板（每 10 秒拉一次远端状态：
step / loss 曲线 / ETA / GPU / 磁盘 / ckpt 列表）。

### 训练完立刻备份（血泪教训）

```bash
bash scripts/train/97_posttrain_backup.sh
```

它会等训练结束，把 `OUT` 里的最终模型 + ckpt 回拷到 `/root`（系统盘）。

> **为什么必须立刻做**：曾在 10-22 训完一个 30k 步模型，想着"回头再备份"，
> 结果实例释放后数据盘被回收，6.3 小时训练归零。系统盘 `/root` 跟着镜像走，
> 数据盘 `/root/autodl-tmp` 是临时的。

拉回本地时**只信内容哈希**：

```powershell
python scripts/tools/parallel_download.py /root/BACKUP_xxx "E:/trained_models/xxx" 12
```

它会远端 `sha256sum` + 本地逐文件比对，只有哈希相同才跳过。
**绝不要只比字节数** —— 早年因为 `truncate` 预分配 + 传输中断，
留下过"大小完全正确、内容全 0"的假模型，而远端副本已被删除 ⇒ 模型实际丢失。

---

## 六、loss 怎么读（以及不能怎么读）

v3 训练曲线（3000 个记录点）：

| 区间 | 平均 loss | 段末 |
|---|---|---|
| 0–1k | 0.1498 | 0.0843 |
| 1k–5k | 0.0621 | 0.0529 |
| 5k–10k | 0.0437 | 0.0361 |
| 10k–20k | 0.0276 | 0.0194 |
| 20k–30k | 0.0149 | 0.0104 |

学习率：线性 warmup 到峰值 **1e-4**（step ≈1510），之后衰减到 ≈3e-13。

**能看出来的**：是否发散、是否 NaN、是否已经进入平台期（v3 在 27k~30k 之间一直在 0.007~0.023 抖动）。

**看不出来的**：模型能不能闭环完成任务。

> 同一个项目里，50k 步的 loss 从 0.08 掉到 0.0127（修好数据对齐之后），
> 但"能抓不能抓"这个问题的答案，**loss 一点都没告诉我们**。
> 判断闭环性质要用 `122`（雅可比增益）和 `126`（输出散布比），
> 最终只能用真机日志验收。详见 [07_DIAGNOSTICS](07_DIAGNOSTICS.md) 与 [08_LESSONS](08_LESSONS.md)。

---

## 七、产物结构

```
/root/autodl-tmp/gr00t_v1/
├── config.json
├── model-00001-of-00002.safetensors   4.7 G
├── model-00002-of-00002.safetensors   2.5 G
├── model.safetensors.index.json
├── experiment_cfg/metadata.json       ← 归一化统计，换数据集必须匹配
├── trainer_state.json                 ← global_step / loss 全历史
├── training_args.bin
├── checkpoint-2000/  …  checkpoint-10000/
└── runs/                              ← tensorboard
```

> **最终模型直接在 `OUT` 根目录**，没有 `model/` 子目录。
> 服务端 `--model-path` 要指到这一级（`.../gr00t_v1`），不是 `.../gr00t_v1/model`。

判定"这份权重是哪个训练"的三个办法：

1. 文件 mtime
2. `trainer_state.json` 的 `global_step` / `epoch`
3. `experiment_cfg/metadata.json` 的 md5（不同数据集一定不同）
