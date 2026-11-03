# 06 · 实时分块推理（RTC）

> 这一节实现的是 **Real-Time Chunking**（Black et al., *Real-Time Execution of Action Chunking Flow Policies*）
> 的常见做法：异步预取下一块 + 在去噪过程中用旧计划"软引导"接缝。

## 一、为什么需要它

策略每次推理给出 16 步（`chunk=16`），名义 15 Hz 下覆盖约 1.07 秒。
但一次推理要走 `拍照 → JPEG 编码 → 上行 → 去噪 4 步 → 下行`，实测 **0.2~0.4 秒**。

同步执行（`run_groot_official.py`）的时序：

```
[拍] [推理 0.35s 空转] [执行 16 步 1.07s] [拍] [推理 0.35s 空转] …
```

于是每 1 秒就有一次 0.35 秒的**停顿**，动作看起来一顿一顿。

RTC 的做法：**在执行第 1 步时就把"下一块"的请求发出去**，回包到达后按重叠区切进新块。
理想情况下没有空转。

```
第 k 块:  |1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16|
                                                    ↑ 发请求
第 k+1 块:            |1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16|
                      ↑ 回包，从对应的那一步切进去
                      重叠区（旧计划剩下的部分）
```

---

## 二、核心难点：重叠区里的"两个计划打架"

第 k+1 块是从"请求那一刻"往后数的 16 步，它和第 k 块剩下的 16−d 步**描述同一段时间**，
但两次推理的结果不可能逐位相同。直接切换 → 动作跳变。

RTC 的解法：在去噪时**把重叠区的速度往旧计划上拉**。

flow matching 的采样是 `x_{t+1} = x_t + dt · v_θ(x_t, t)`，
旧计划 `prefix` 对应的"精确速度"是 `v_prefix = (prefix − x_t) / (1 − t)`（直线路径）。
每个去噪步做加权：

```python
v = (1 - β) * v_model + β * v_prefix
```

`β` 沿重叠区**线性衰减**：接缝处 = `--rtc-beta`（默认 1.0，几乎贴住旧计划），
到重叠区末端衰减到 0（完全放开，模型自由发挥）。

> 这是"软"约束：模型仍有自由度，只是被引导。相比之下把 token 直接覆盖成旧计划（硬钉）
> 接缝差能做到 0.00°，但**接下来 0.5 秒的纠正也被一起冻住了**，实测手感更差。

---

## 三、两端改动

### 服务端 `101_policy_server_rtc.py`

在 `action_head.get_action` 外面包一层，拦截去噪循环：

```python
prefix = _RTC["prefix"]          # 上一次返回的 16 步
d      = _RTC["last_d"]          # 客户端上报的"已执行步数"
pin    = prefix[:, d:]           # ★ 关键：用剩下的那段，不是前 d 步
n_g    = pin.shape[1]
beta   = _RTC["beta"] * torch.linspace(1.0, 0.0, n_g).view(1, n_g, 1)

v_prefix = (pin - x0[:, :n_g]) / (1.0 - t).clamp_min(1e-3)
v = (1 - beta) * v_model
v[:, :n_g] = (1 - beta) * v_model[:, :n_g] + beta * v_prefix
```

### 客户端 `run_groot_rtc.py`

```python
# 执行到第 rtc_fire 步时异步发请求（每次新开一条 ZMQ 连接）
if ci >= a.rtc_fire and inflight is None:
    inflight = Fetcher(host, port, obs, d_at=ci)

# 回包到达：新块的 index i 对应时间 (d_at + i)
skip = ci - d_at
if skip >= len(new) - d_at:      # 太晚了，已越过重叠区 → 丢弃，重新取
    inflight = None
else:
    cur, ci = new, skip
```

---

## 四、四个踩过的坑（每个都让动作变差过）

### 坑 1：`ZMQError: Operation cannot be accomplished in current state`

REQ socket 既不是线程安全的，也不允许"请求发出还没收到就再发"。
异步线程和主线程共用一个 `ZmqPolicyClient` 时，块用尽那一刻必然撞上。

**修法**：`Fetcher` 每次自己建一条连接，用完自己 close；主线程只用自己的连接。

### 坑 2：钉子钉错了位置（这是"加了 RTC 反而更差"的原因）

最初的实现钉的是 `prefix[:, :d]` —— **旧计划里已经被执行掉的前 d 步**。
但新块的 index 0 对应"现在"，不是"过去"。于是：

- 钉住的 d 步客户端根本不会再执行 → 白扔 d 个 token；
- 真正会被执行的 `d..15` 完全自由生成 → 换块照样跳（实测 1.4~14.7°）。

**修法**：钉 `prefix[:, d:]`（旧计划**剩下**的那段），同时把客户端的 `skip` 改成 `ci - d_at`。
两边必须同时改 —— 只改一边会让接缝差从 3° 涨到 **22~49°**。

### 坑 3：忘记 `max_relative_target` → 甩臂

基线客户端一直有这行：

```python
robot.config.max_relative_target = a.max_rel    # 每条指令裁到 当前位置 ± 8°
```

RTC 版是照着基线改的，但**漏了这一句**。于是换块时 20~40° 的指令差会被**一次性执行**，
视觉上就是"机械臂突然甩一下"。

**教训**：新写执行环时，基线的安全闸门（限速 / 限幅 / 卡死自救）必须**逐条对照抄过来**。

### 坑 4：物理上限算不过来

chunk=16、延迟 0.35 s（≈5 步）。每次换块消耗约 5 步余量，两次换块就把 16 步用光
⇒ 每 ~16 步还是会有一次 0.2~0.4 秒的阻塞重取块。

**这不是 bug，是算术**。要更平滑只能降延迟（本地直连 GPU）或换更长 horizon 的模型。

---

## 五、参数

| 参数 | 默认 | 作用 |
|---|---|---|
| `--rtc-fire` | 1 | 执行到第几步发请求。1 = 重叠最多、最平滑，对延迟最敏感；抖动大就改 2 或 3 |
| `--rtc-beta` | 1.0（服务端） | 引导强度上限。1.0 最贴旧计划；0.5~0.7 更听新画面但更抖 |
| `--rtc-no-decay` | — | 关掉衰减（整个重叠区恒定 β） |
| `--max-seam` | 25.0 | 接缝差超过这个度数就**不切**，继续走旧计划 |
| `--max-rel` | 8.0 | 每步最大关节变化（度）。**必须设，否则甩臂** |
| `--no-rtc` | — | 关掉 RTC，退化成同步执行（做对照实验用） |

---

## 六、平滑：RTC 解决换块，平滑解决逐帧噪声

RTC 把接缝差从 14.7° 压到 1° 以内，但 CSV 里还剩一个问题：
**指令的逐帧反向率 40~58%** —— 相邻两步的方向有一半时间是相反的。
这是模型每步 1~2° 的输出噪声，和换块无关。

两级处理：

**① 块内对称滑动平均**（`--chunk-smooth 5`，零滞后）

模型一次给 16 步，可以在块内用"后一步"平滑"这一步"，不产生滞后（EMA 做不到）：

```python
pad = np.pad(arr, ((k, k), (0, 0)), mode="edge")
out = np.convolve(pad[:, j], np.ones(w) / w, "valid")
```

**② 下发电机一阶低通**（`--smooth 0.6`）

```python
cmd = α * cmd_raw + (1 - α) * cmd_prev        # 滞后 ≈ (1-α)/α 步 ≈ 47 ms
```

实测（同一静止画面，64 步）：

| 配置 | 指令反向率 | 每步变化 p95 | 与原始轨迹差 RMS |
|---|---|---|---|
| 全关 | 45% | 1.69° | — |
| 仅 `chunk-smooth 5` | 36% | 0.35° | 0.65° |
| 仅 `smooth 0.6` | 40% | 1.03° | 0.76° |
| **两者同开（推荐）** | **19%** | **0.31°** | 0.82° |

轨迹只改动 0.82° RMS，远小于 8° 的限幅余量，所以**不会影响抓取精度**。

---

## 七、对照实验

```powershell
# RTC 开
python scripts/deploy/run_groot_rtc.py --steps 0 --rtc-fire 1 --record-video outputs/rtc.mp4 --log-csv outputs/rtc.csv

# RTC 关（同一条代码路径，只切开关）
python scripts/deploy/run_groot_rtc.py --steps 0 --no-rtc    --record-video outputs/nortc.mp4 --log-csv outputs/nortc.csv
```

比 CSV 里的 `[切换] 接缝差` 和指令反向率。**同一段代码、同一份权重，唯一变量是开关**，
这样得出的结论才站得住。
