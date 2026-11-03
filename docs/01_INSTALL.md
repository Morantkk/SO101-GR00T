# 01 · 环境安装

两端分离：**Windows 本机**只负责相机、串口和网络（不需要 GPU），
**Linux GPU 机器**负责训练与策略推理。这样笔记本可以随时插拔机械臂，GPU 用云上的。

```
┌──────────────────────┐        SSH 隧道 :5555        ┌──────────────────────┐
│  Windows 本机         │ ───────────────────────────▶ │  Linux + NVIDIA GPU   │
│  lerobot 0.6.2       │   图像(JPEG) 上行 / 动作下行   │  GR00T N1.5 策略服务   │
│  相机 + 串口 + 控制环  │ ◀─────────────────────────── │  torch 2.5.1 + cu124  │
└──────────────────────┘                              └──────────────────────┘
```

---

## 一、Windows 客户端

### 1. 建虚拟环境

```powershell
conda create -n lerobot python=3.10 -y      # 3.11/3.12 也可，3.13 需确认轮子齐全
conda activate lerobot
pip install "lerobot==0.6.2" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 补数据集所需依赖（**版本有硬上限**）

只装 `lerobot` 本体是跑不了 `lerobot-record` 的，会依次报缺 `datasets` → `av` → `torchcodec`。
按 `lerobot` 自己的约束装，否则会踩 `av` 的 API 变更问题：

```powershell
pip install "datasets>=4.8,<5" "pyarrow>=21,<30" "av>=15,<16" "torchcodec>=0.7,<0.12" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

实测可用组合（Python 3.13 + torch 2.11）：

| 包 | 版本 |
|---|---|
| datasets | 4.8.5 |
| pyarrow | 26.0.0 |
| av | 15.1.0 |
| torchcodec | 0.11.1 |

> ⚠️ **坑**：先装成 `av 19 / torchcodec 0.17 / datasets 5.1` 时，
> `lerobot-record` 直接崩在 `AttributeError: module 'av' has no attribute 'option'`。
> 原因是新版 av 不再隐式导入子模块，而 lerobot 0.6.2 的 `pyav_utils` 依赖这个隐式行为。

验证：

```powershell
lerobot-record --help
```

能打出 usage 就说明依赖齐了。

### 3. 相机与串口

```powershell
python scripts/tools/check_ports.py --grab
```

它会列出：

- 所有串口 + 电压（SO-101 从臂接 12 V 适配器，主臂只有 USB 5 V，可据此区分）
- 相机索引 + 每台相机实际支持的分辨率（**必须看实际值**，请求不支持的分辨率会被静默回落到别的尺寸）
- 每个索引抓一张图存盘，肉眼确认哪台是前视、哪台是腕部

本项目的实测结果：

| 设备 | 索引 | 能力 |
|---|---|---|
| 前视（Web Camera） | **1** | 原生 640×360 ✅ |
| 腕部（icspring） | **0** | 只有 640×480 / 1280×720，**不支持 640×360** |
| 笔记本内置 | 2 | 排除 |

> 抓图时**必须连续读几十帧再存**。`VideoCapture()` 打开后立刻读的第一帧自动曝光还没收敛，
> 亮度只有正常值的 1/5，很容易误判成"相机坏了"。

### 4. 校准

```powershell
lerobot-calibrate --robot.type=so101_follower --robot.port=COM3 --robot.id=my_awesome_follower_arm
lerobot-calibrate --teleop.type=so101_leader --teleop.port=COM4 --teleop.id=my_awesome_leader_arm
```

校准文件落在 `C:\Users\<你>\.cache\huggingface\lerobot\calibration\`。
**之后所有命令的 `--robot.id` / `--teleop.id` 必须与这里一模一样**，换名字会要求重新标定。

---

## 二、Linux GPU 服务端

以 AutoDL 为例（`torch 2.5.1 + cu124` 基础镜像）。**环境装在系统盘 `/root`**，
数据盘 `/root/autodl-tmp` 只放训练产物 —— 数据盘在实例释放时会被回收。

### 1. 基础环境

```bash
conda create -n gr00t python=3.10 -y
conda activate gr00t
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```

### 2. Isaac-GR00T

```bash
git clone https://github.com/NVIDIA/Isaac-GR00T.git /root/Isaac-GR00T
cd /root/Isaac-GR00T
pip install -e .
```

### 3. 四个容易漏的依赖

```bash
# ① av 必须锁 12.3.0：更高版本的 VideoFrame.pts 行为变化，
#    写 mp4 时会报 "Invalid argument (Errno 22)"
pip install av==12.3.0

# ② pytorch3d：官方 wheel 在 py310+cu124 上编译困难，用预编译版
pip install pipablepytorch3d==0.7.6

# ③ diffusers 0.30.2 及以上会改 scheduler 接口
pip install diffusers==0.30.2

# ④ 策略服务用 ZeroMQ 传图像
pip install pyzmq msgpack
```

### 4. flash-attn（最耗时的一步）

要跟 torch / cuda / python 版本严格对应，否则会编译 40 分钟后失败：

```bash
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

编译期间内存要够（小于 8 G 容易 OOM）。脚本见 `scripts/train/93_build_flash_attn_gr00t.sh`。

### 5. 国内加速

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

> 这三行**必须写进启动命令里**（例如 `screen -dmS x bash -lc 'export ...; python ...'`），
> 只 export 在当前 shell 里对 `screen` 新开的会话无效。

### 6. 验证

```bash
python -c "import torch, av, gr00t; print(torch.__version__, torch.cuda.is_available(), av.__version__)"
# 期望：2.5.1 True 12.3.0
```

---

## 三、连接两端

把远端地址写进 `examples/conn.py`（从 `conn.example.py` 复制改名），或者用环境变量：

```powershell
$env:GR00T_HOST="connect.xxx.seetacloud.com"
$env:GR00T_PORT="40009"
$env:GR00T_PASSWORD="******"
python scripts/tools/remote_exec.py "nvidia-smi"
```

> **云实例每次开机端口都会变**，所以 `conn.py` 已被 `.gitignore` 排除，不要把密码提交进仓库。

建隧道并测试：

```powershell
python scripts/deploy/ssh_tunnel.py --local 5555 --remote 127.0.0.1:5555
```

隧道是**未加密的裸 TCP 转发**，只在你自己机器和实例之间使用。

---

## 四、常见报错对照

| 报错 | 原因 | 处理 |
|---|---|---|
| `module 'av' has no attribute 'option'` | av 版本过高 | 降到 15.x（客户端）/ 12.3.0（GR00T） |
| `'datasets' is required but not installed` | 缺 dataset extra | 按上面版本表装 |
| `Invalid argument (Errno 22)` | av 写 mp4 时 pts 为空 | 锁 av==12.3.0；或手动 `frame.pts = n` |
| `PermissionError: 拒绝访问 COM3` | 串口被占用 | 关掉上一个客户端进程 |
| 相机打开后画面极暗 | 自动曝光未收敛 | 连续读 30 帧以上再用 |
| `CUDA out of memory` | batch 太大 | 16 → 12（24 G 卡） |
