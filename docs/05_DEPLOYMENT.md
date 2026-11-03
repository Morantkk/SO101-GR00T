# 05 · 真机部署

推理时 GPU 在云上、机械臂在本机。**不要试图在笔记本上跑 3B 模型**。

```
┌─────────────────────────┐                       ┌──────────────────────────────┐
│ 本机 Windows             │                       │ GPU 机器                      │
│                         │   SSH 隧道 :5555       │                              │
│  run_groot_official.py  │ ◀───────────────────▶ │  100_policy_server_jpeg.py    │
│   ├ 读相机 (front/wrist) │   JPEG 上行 ~87 KB/req │   └ GR00T N1.5 (bf16)         │
│   ├ 读串口 (实测角度)     │   6×16 动作下行        │      每次推理 ~0.2-0.4 s       │
│   └ 写串口 (下发电机)     │                       │                              │
└─────────────────────────┘                       └──────────────────────────────┘
```

---

## 一、服务端

```bash
screen -dmS policy bash -lc '
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/root/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/root/miniconda3/envs/gr00t/bin/python /root/setup/100_policy_server_jpeg.py \
  --model-path /root/autodl-tmp/gr00t_v1 \
  --port 5555 > /root/logs/policy.log 2>&1'
```

成功标志：

```
[server] jpeg_decode=True model=/root/autodl-tmp/gr00t_v1 data_config=so100_dualcam denoise=4 port=5555
```

自检（在服务器上）：

```bash
python -c "import socket;s=socket.socket();s.settimeout(3);print(s.connect_ex(('127.0.0.1',5555)))"
# 0 = 已监听
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
# 约 5.6 G / 24.5 G（bf16 + denoise=4）
```

> `--model-path` 指到**最终模型所在目录**（含 `config.json` 与两个 `safetensors`），
> 不是它的上一级，也不是 `model/` 子目录。

> ⚠️ 换服务时**不要用 `pkill -f 100_policy_server`** —— 远端 `bash -c` 的命令串里
> 含这个字符串，pkill 会匹配到自己，直接把会话杀掉。用
> `screen -S policy -X quit` + `fuser -k 5555/tcp`。

---

## 二、SSH 隧道（本机）

```powershell
python scripts/deploy/ssh_tunnel.py --local 5555 --remote 127.0.0.1:5555
```

看到 `listening 127.0.0.1:5555 -> 127.0.0.1:5555` 即可。**这个窗口要一直开着。**

连通性自检：

```powershell
python scripts/deploy/rtc_up.py --local 5555
# ping -> {'status': 'ok', 'message': 'Server is running'}
```

---

## 三、客户端

### 先空跑（臂不动）

```powershell
python scripts/deploy/run_groot_official.py --dry-run --steps 32
```

dry-run 仍然会走"拍照 → 上传 → 推理 → 解析动作"的完整链路，只是不写串口。
**用它验证：相机正常、隧道通、动作维度 6、数值范围合理。**

### 真跑

```powershell
python scripts/deploy/run_groot_official.py --steps 0 --max-rel 8 --record-video outputs/run1.mp4
```

`--steps 0` = 不限步数，跑到 Ctrl-C。Ctrl-C 会自动收尾（mp4 不会坏）。

### 参数速查

| 参数 | 默认 | 说明 |
|---|---|---|
| `--robot-port` | COM3 | 从臂 |
| `--front-idx` / `--wrist-idx` | 1 / 0 | 相机索引（**换机器必须重新确认**） |
| `--host` / `--port` | 127.0.0.1 / 5555 | 走隧道 |
| `--task` | `Pick up a pen and put it into the box` | **必须与训练集 `tasks` 一字不差** |
| `--chunk` | 16 | 一次执行多少步再重新推理 |
| `--max-rel` | 8.0 | 每步最大关节变化（度）。**这是安全闸门，0 = 不限速 = 会甩臂** |
| `--steps` | 300 | 0 或负数 = 不限 |
| `--record-video` | — | 录第三视角，每执行一步抓一帧 |
| `--record-keys` | front | 要录腕部就写 `front,wrist` |
| `--dry-run` | — | 不写串口 |

### 关于 `--chunk 16`

- 观测刷新率 = `1 / (16 × dt)` ≈ **0.8 Hz**
- 有效步频 = 1 / (dt + 推理延迟/16) ≈ **12.8 Hz**（名义 15）
- 块越短 → 反应越快但越抖；块越长 → 平滑但迟钝

---

## 四、客户端内部结构（改代码前必读）

`run_groot_official.py` 里有**三条互不共用的执行路径**：

```
run_attempts()  ──┐
run_trials()    ──┼──▶  主 while 循环（默认走这条）
                  ┘
```

任何"兜底逻辑"（抓空重试、卡死自救、限速）**三处都要加**，否则会出现
"改完了但没生效"的静默失败。改完先 `--dry-run --start-pose none` 冒烟。

已有的保护：

| 机制 | 作用 |
|---|---|
| `max_relative_target = --max-rel` | lerobot 层闸门：每条指令被裁到"当前位置 ± 8°" |
| 行程限幅 | 越界指令被剪到数据集里的 min/max |
| 卡死自救 | 连续若干 chunk 几乎不动 → 手动拉开一点重试，`--unstuck-max` 次后退出 |
| 启动位姿 | `--start-pose` 可指定，或从数据集首帧取 |

---

## 五、故障排查

| 现象 | 先查什么 | 常见原因 |
|---|---|---|
| `Connection refused` | 隧道窗口还开着吗 | 云实例端口变了（每次开机都变） |
| 连不上机器人 `拒绝访问` | `--robot-port` | 上一个客户端没退出，串口被占 |
| 臂完全不动 | 看 CSV 里 `cmd` 有没有变化 | ①起始位姿压在关节极限 ②`--max-rel` 设成 0 反而没限速 ③模型输出饱和 |
| 臂突然甩一下 | 日志里 `[切换] 接缝差` | 没设 `max_relative_target`；或换块时接缝差过大 |
| 一直小幅抖 | `cmd` 的逐帧反向率 | 模型逐帧 1~2° 噪声，加 `--smooth`/`--chunk-smooth` |
| 合爪位置总是偏外 | 合爪时夹爪读数 | 视觉分辨率不足（见 02 第四节） |
| 抓空后永不重试 | 演示里有没有失败恢复样本 | 没有就得在客户端加判定 |
| 图像很暗 | 是不是第一帧 | 自动曝光未收敛，客户端有 1 s warmup |

---

## 六、真机验收清单

跑之前先想清楚"怎么算成功"，否则只剩感觉。

1. **首 30 步**：`act`（下发）与 `fb`（实测）是否同步？有没有"半步不走"？
2. **状态 z 分**：每一帧的关节角相对演示分布的标准分，> 3 说明已经跑出分布外。
3. **末端轨迹**（用 URDF 正运动学算，不要读单关节角判断"臂高低"）：
   接近阶段末端是否朝物体移动？
4. **合爪时刻**：夹爪读数 < 3 吗？合爪点与物体的实际距离多少？
5. **CSV 必留**：`--log-csv`，事后能算反向率、跟踪误差、接缝差。

```powershell
python scripts/deploy/run_groot_official.py --steps 0 --max-rel 8 `
  --record-video outputs/run1.mp4 --log-csv outputs/run1.csv
```

跑完把 `run1.mp4` 和 `run1.csv` 一起看：**视频回答"它想干什么"，CSV 回答"它实际发了什么"。**
