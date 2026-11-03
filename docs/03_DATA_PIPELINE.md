# 03 · 数据流水线（含一个把 v3 带偏的致命 bug）

从 `lerobot-record` 录出来的原始数据集，到 GR00T 能直接吃的训练集，一共五步。
**其中第 4 步曾经把整个 v3 训练废掉**，这一节把它讲透。

```
raw (lerobot v3)          ① 85 转换           ② 91 stats
  meta/episodes/      ───────────────▶   GR00T 格式     ──────▶  修 stats.json
  tasks.parquet                         meta/modality.json
                                        meta/episodes.jsonl
                                        meta/tasks.jsonl
                                              │
                                              ▼  ③ 92 相机统一（wrist 720p → 360p）
                                        ④ 裁掉开头静止段   ← ★ 风险全在这一步
                                              │
                                              ▼  ⑤ 体检（129b / 133）
                                         可训练数据集
```

---

## 一、五个步骤

### ① 转成 GR00T 格式

```bash
python scripts/data/85_lerobot_v30_to_gr00t.py \
  --src  /root/arm101_pick_one_pen \
  --dst  /root/gr00t_data/arm101_pick_one_pen \
  --modality examples/so100_dualcam__modality.json
```

它做三件事：逐集切出视频、重写 parquet、生成 `modality.json / episodes.jsonl / tasks.jsonl`。
因为视频是**整集重新编码**的，`timestamp` 天然从 0 开始，这一步不会错位。

### ② 修 `stats.json`

```bash
python scripts/data/91_fix_stats.py /root/gr00t_data/arm101_pick_one_pen
```

> ⚠️ `stats.json` 里**不能有 `count` 字段**。lerobot 写出的 stats 带 count，GR00T 加载时
> 会因为多出的键报错或静默算错归一化。这个脚本就是删掉多余键并补齐 `q01/q99`。

### ③ 统一两路相机的分辨率

```bash
python scripts/data/92_equalize_cameras.py /root/gr00t_data/arm101_pick_one_pen --target 640x360 --crf 18
```

采集时腕部是 1280×720、前视是 640×360。GR00T 要求两路尺寸一致（否则同一个 batch 的
图像张量拼不起来），这一步把腕部缩到 640×360。

### ④ 裁掉开头的静止段（★ 有坑）

原始演示开头常有 3~8 秒"手放在那儿没动"，`104_trim_static.py` 用
「5 帧窗口累计位移 > 2.0° 再往后取 5 帧」定位运动起点并裁掉前面。

**它的判据是对的，社区公开的 motion-trimmed SO-101 数据集用的也是同一套。**
问题出在它裁完之后做的事（见下一节）。

### ⑤ 体检

```bash
python scripts/data/129b_verify_dataset.py --ds <数据集>
python scripts/data/133_check_alignment.py --ds <数据集>     # 专门查 timestamp 对齐
```

必须全部通过才能训练：

| 检查 | 期望 |
|---|---|
| 逐集帧数 | `parquet 行数 == front 帧数 == wrist 帧数`，102/102 |
| 索引 | `frame_index / index / episode_index` 连续 |
| 任务映射 | 逐帧 `task_index` ↔ `tasks.jsonl` ↔ `episodes.jsonl` 一致 |
| stats | 逐维对账相对差 = 0 |
| **图像超前量** | **唯一值 [0]**（← 这一条最容易被忽略） |

---

## 二、那个致命 bug：视频裁了，`timestamp` 没裁

### 现象

v3 在真机上：机械臂几乎不动，或者缓慢漂移；合爪位置系统性偏离物体 5~8 cm。
但所有离线指标都很好 —— loss 0.08、单步误差 1.4°、回放审计通过。

### 根因

原始数据里 `timestamp == frame_index / 15`，从 0 开始，图与关节一一对应。
`104_trim_static.py` 只做了：

```python
df = df.iloc[n:]        # 删掉前 n 行
# 视频也裁掉前 n 帧
# ❌ timestamp / frame_index 原样保留，还写着 4.2 秒
```

裁完之后：

| | 裁前 | 裁后 | 对不对 |
|---|---|---|---|
| 视频帧 | 0 … 602 | 0 … 539（= 原片 63…602） | ✅ |
| parquet 行（关节/动作） | 0 … 602 | 0 … 539 | ✅ |
| **`timestamp` 列** | 0, 0.067, … | **4.2, 4.267, …** | ❌ 还是原片的时间 |

而 GR00T 取图**不看行号，看这个时间标签**：

```python
# gr00t/data/dataset.py
idx = np.argmin(np.abs(video_frame_times - timestamp[step]))
```

裁剪后的新视频从 t=0 开始，所以"去找 4.2 秒"= 取第 63 帧。
**每一行的图像都超前该集裁剪量 `k` 帧：46~124 帧 = 3.1~8.3 秒，逐集不同。**

用旧数据集真实复算的结果：

| parquet 行号 | 里面写的时间 | 实际取到第几帧 | 图像超前 |
|---|---|---|---|
| 0 | 4.200 s | **63** | +63 帧 |
| 100 | 10.867 s | 163 | +63 帧 |
| 463 | 35.067 s | 526 | +63 帧 |
| 526 | 39.267 s | 539 | +13 帧（钉死在最后一帧） |

### 为什么所有离线测试都看不出来

因为"图像(t+63) + 状态(t) → 动作(t)"这个映射**在训练集内部是自洽的**。
模型学的是"看着 4 秒后的画面，说出现在的动作"，loss 可以拟合到很低。

而部署时客户端喂的是**实时画面**。等于把输入换成了模型从没见过的组合 ——
它会输出"未来画面里应该做的动作"，自然对不上现在的位置。

这也印证了一件事：**loss / 回放 / 单步误差都是 teacher-forced 指标**，
每一步都喂真值把误差锚死，对这个错误结构性失明（详见 [08_LESSONS](08_LESSONS.md)）。

### 正确做法

裁剪后**必须**重写：

```python
df["timestamp"]   = np.arange(len(df)) / fps     # 从 0 开始
df["frame_index"] = np.arange(len(df))
```

`131_rebuild_trim_v2.py` 是修正版：一次编解码完成 切集 → 裁头 → 相机统一 → **重写时间戳**，
比"先转换、再裁剪、再统一"少一代画质损失：

```bash
python scripts/data/131_rebuild_trim_v2.py \
  --src /root/arm101_pick_all_objects \
  --dst /root/gr00t_data/arm101_pick_all_objects_trim2 \
  --modality examples/so100_dualcam__modality.json --apply
```

修正后的对比：

| | 旧 `_trim` | 新 `_trim2` |
|---|---|---|
| 图像超前量 | +46~124 帧（逐集不同） | **0 帧（102/102）** |
| 开头残留静止帧 | 3 集，最多 64 帧 | **0 集** |
| 首帧 `elbow` 标准差 | 2.41° | **1.87°** |
| 首帧运动速度标准差 | 0.88 °/帧 | **0.29 °/帧** |
| 训练 loss | 0.08 | **0.0127** |

> **一句话规则**：任何对 parquet 的**行裁剪**，都必须同时重写 `timestamp` 和 `frame_index`。
> 只裁剪不重写 = 训练/部署观测错位。

---

## 三、在服务器上"补裁"几帧的正确姿势

如果只有几集需要再往前挪几帧，不必传整个数据集（1.4 GB 上传很慢），
可以只传一个"目标帧数表"，让服务器原地重切：

```bash
python scripts/data/132_retrim_from_trim.py \
  --src /root/gr00t_data/xxx_trim \
  --dst /root/gr00t_data/xxx_trim2 \
  --plan plan.json --apply
```

`plan.json` 是 `{episode_index: 目标帧数}`。脚本会**能复制就复制视频**（不重编码，零画质损失），
只有真的需要少几帧的集才重新编码。实测 102 集里 94 集直接复制、8 集重编码，0.9 分钟跑完，
而全量重编码要 40 分钟。

---

## 四、体检脚本看什么

`129b_verify_dataset.py` 输出五段：

```
A) 文件数       parquet / front / wrist 三者相等
B) 逐集帧数     视频实际解码帧数 == parquet 行数
C) 索引连续     frame_index 从 0 递增、episode_index 正确
D) 任务映射     每个 task_index 都能在 tasks.jsonl 里找到
E) stats 对账   用 parquet 重算 min/max/mean/std 与 stats.json 比，相对差应为 0
```

`133_check_alignment.py` 专查那个 bug：对每一集用真实的视频帧时间戳复算
`argmin|帧时间 − timestamp[i]|`，统计 `idx - i` 的分布。**只允许出现 [0]**。

最后一行一定是这个：

```
图像超前量: 唯一值 [0]  -> 全部 0 帧 ✓
```
