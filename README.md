# SO101-GR00T

**用 NVIDIA GR00T N1.5 让 SO-101 机械臂干活** —— 从数据采集、格式转换、微调训练到端云部署的完整复现记录。

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/LeRobot-0.6.2-ff9d00"/>
  <img src="https://img.shields.io/badge/GR00T-N1.5-76b900"/>
  <img src="https://img.shields.io/badge/GPU-RTX%204080S%2032G-1f6feb"/>
  <img src="https://img.shields.io/badge/license-MIT-2ea44f"/>
</p>

<p>
  <a href="docs/01_INSTALL.md">安装</a> ·
  <a href="docs/02_DATA_COLLECTION.md">采集</a> ·
  <a href="docs/04_TRAINING.md">训练</a> ·
  <a href="docs/05_DEPLOYMENT.md">部署</a> ·
  <a href="docs/08_LESSONS.md">踩坑记录</a>
</p>

<p align="center">
  <a href="https://cdn.jsdelivr.net/gh/Morantkk/SO101-GR00T@main/media/demo_80s.mp4">
    <img src="https://cdn.jsdelivr.net/gh/Morantkk/SO101-GR00T@main/media/front_camera.png" width="72%"/>
  </a><br/>
  <sub>点击画面观看 80 秒真机演示（mp4 · 6.4 MB）</sub>
</p>

> **图片加载**：README 里的图和视频走 jsDelivr CDN。如果你在中国大陆，GitHub 默认的
> `raw.githubusercontent.com` / `camo.githubusercontent.com` 经常加载不出来，那不是文件缺失。

---

## 项目效果

同一套硬件、同一套流程，做了两个任务。第二个任务把第一个任务里藏着的 bug 逼了出来。

<table align="center">
  <tr>
    <td align="center" width="50%">
      <img src="https://cdn.jsdelivr.net/gh/Morantkk/SO101-GR00T@main/media/clip1.gif" width="100%"/><br/>
      <sub>clip 1 · 0:08</sub>
    </td>
    <td align="center" width="50%">
      <img src="https://cdn.jsdelivr.net/gh/Morantkk/SO101-GR00T@main/media/clip2.gif" width="100%"/><br/>
      <sub>clip 2 · 0:26</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://cdn.jsdelivr.net/gh/Morantkk/SO101-GR00T@main/media/clip3.gif" width="100%"/><br/>
      <sub>clip 3 · 0:34</sub>
    </td>
    <td align="center">
      <img src="https://cdn.jsdelivr.net/gh/Morantkk/SO101-GR00T@main/media/clip4.gif" width="100%"/><br/>
      <sub>clip 4 · 0:50</sub>
    </td>
  </tr>
</table>

四个片段是从 80 秒录像里按画面变化量截取的，时间段标注的是原始录像的时间点。

| | **v1 · 抓单支笔** | **v3 · 桌面全物体** |
|---|---|---|
| 任务指令 | `Pick up a pen and put it into the box` | `Put all the objects on the table into the box` |
| 数据 | 50 集 / 17,789 帧 | 102 集 / 51,209 帧 |
| 训练 | 10,000 步 / batch 16 / ≈9 epoch | 30,000 步 / batch 16 / ≈9.4 epoch |
| 时长 | 约 2~3 h | 6 h 20 m |
| train_loss | — | 0.08 |
| 真机 | 跑通全链路 | 抓空（数据时间轴错位） |

> v3 的失败**不是模型能力问题**：数据预处理把视频裁了却忘了把 `timestamp` 归零，训练时每帧图像
> 都比关节状态超前数秒。修好之后同配方重训，loss 从 0.08 掉到 **0.0127**。
> 详见 [docs/03](docs/03_DATA_PIPELINE.md) 与 [docs/08](docs/08_LESSONS.md)。
>
> 上面那段演示视频跑的是**数据修正后重训的 30k 版本**，不是表格里的 v3。

两个模型都**只训练动作头**（≈0.75B），视觉塔（SigLIP2）与语言模型（Qwen3-1.7B）全程冻结。

---

## 模型权重

| 模型 | 说明 | 状态 |
|---|---|---|
| **v1** (10k) | 抓单支笔 | 本地训练完成，待发布 |
| **v3** (30k) | 桌面全物体（⚠️ 训练数据时间轴错位，反面样本） | 本地训练完成，待发布 |

两个模型合计 **14.2 GB** —— 单个 `safetensors` 就超过 GitHub 100 MB 的单文件上限，
Git LFS 免费额度（1 GB 存储 / 1 GB 月流量）也不够，所以权重不放在本仓库，
走 Hugging Face / ModelScope 托管，**链接待补**。

想自己复现的话，整套流程 6 小时，见 [docs/04](docs/04_TRAINING.md)。
发布到 HF 的脚本已经写好：`scripts/tools/publish_models_to_hf.py`。

---

## 硬件

| 部件 | 配置 | 备注 |
|---|---|---|
| 机械臂 | SO-101 follower / **COM3** / 12 V | 6 舵机，1 Mbps |
| 主臂 | SO-101 leader / **COM4** / USB 5 V | 电压特征可区分主从 |
| 前视相机 | USB index **1** | 原生 640×360 |
| 腕部相机 | USB index **0** | 只有 640×480 / 1280×720 |
| 训练 GPU | RTX 4080S 32 G（4090 24 G 需把 batch 降到 12） | batch 16 占 24.4 G |

## 软件

`LeRobot 0.6.2`（采集 / 数据格式） · `Isaac-GR00T`（训练 / 推理） ·
`torch 2.5.1+cu124` · `av 12.3.0`（**必须锁版本**） · `flash-attn 2.7.4.post1`

---

## 快速开始

```
① 采集 → ② 转换对齐 → ③ 微调 → ④ 起服务 → ⑤ 本机推理
```

**① 采集**（Windows 本机，接机械臂）

```bash
lerobot-record --robot.type=so101_follower --robot.port=COM3 --robot.id=my_awesome_follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 640, height: 360, fps: 15}, \
                     wrist: {type: opencv, index_or_path: 0, width: 1280, height: 720, fps: 15} }" \
  --teleop.type=so101_leader --teleop.port=COM4 --teleop.id=my_awesome_leader_arm \
  --dataset.single_task="Pick up a pen and put it into the box" \
  --dataset.fps=15 --dataset.num_episodes=50 --dataset.push_to_hub=false
```

或直接双击 `examples/record_pen.bat`。
`--robot.id` 必须和你的校准文件名一致（在 `~/.cache/huggingface/lerobot/calibration/` 下），换名会要求重新标定。

**② 转换 + 对齐**（本机或服务器都行）

```bash
python scripts/data/85_lerobot_v30_to_gr00t.py --src <raw> --dst <ds> \
  --modality examples/so100_dualcam__modality.json
python scripts/data/91_fix_stats.py        <ds>
python scripts/data/92_equalize_cameras.py <ds> --target 640x360 --crf 18
python scripts/data/129b_verify_dataset.py --ds <ds>     # 必须全绿
```

**③ 训练**（GPU 机器；先把 ② 的数据集传上去）

```bash
python scripts/tools/parallel_upload_checked.py E:/lerobot_datasets/arm101_pick_one_pen_gr00t \
    /root/gr00t_data/arm101_pick_one_pen 12

python scripts/train/95_patch_save_policy.py            # 首次必跑，否则每个 ckpt 存 16 GB
DS=/root/gr00t_data/arm101_pick_one_pen OUT=/root/autodl-tmp/gr00t_v1 \
STEPS=10000 SAVE=2000 BATCH=16 bash scripts/train/94_train_gr00t_official.sh
```

**④ 起策略服务**（GPU 机器）

```bash
python scripts/deploy/100_policy_server_jpeg.py --model-path /root/autodl-tmp/gr00t_v1 --port 5555
```

**⑤ 本机推理**（两个窗口）

```bash
python scripts/deploy/ssh_tunnel.py --local 5555 --remote 127.0.0.1:5555   # 窗口 A，常驻
python scripts/deploy/run_groot_official.py --steps 0 --max-rel 8          # 窗口 B
```

想要动作更顺，把 ⑤ 换成 RTC 版：`python scripts/deploy/run_groot_rtc.py --steps 0 --rtc-fire 1`
（原理与四个坑见 [docs/06](docs/06_RTC.md)）。

---

## 目录结构

```
.
├── docs/                     教程与踩坑记录（建议按序号读）
├── examples/                 连接配置、modality 模板、Windows 一键脚本
├── media/                    演示视频与动图
└── scripts/
    ├── data/                 转换 / stats / 相机统一 / 裁剪 / 体检
    ├── train/                环境安装、训练启动、save 补丁、备份
    ├── deploy/               策略服务 ×2、客户端 ×2、SSH 隧道
    ├── diagnose/             回放、增益、散布、视觉消融、权重健康
    └── tools/                上传下载、训练面板、串口与相机探测、HF 发布
```

| 文档 | 内容 |
|---|---|
| [01_INSTALL](docs/01_INSTALL.md) | 两端环境，含 flash-attn / av 版本坑 |
| [02_DATA_COLLECTION](docs/02_DATA_COLLECTION.md) | 采集协议、相机与串口、参数对齐表 |
| [03_DATA_PIPELINE](docs/03_DATA_PIPELINE.md) | 五步流水线 + **timestamp 错位这个致命 bug** |
| [04_TRAINING](docs/04_TRAINING.md) | 训练配方、显存表、ckpt 策略 |
| [05_DEPLOYMENT](docs/05_DEPLOYMENT.md) | 服务端 + 隧道 + 客户端全部参数 |
| [06_RTC](docs/06_RTC.md) | 实时分块推理（软引导实现） |
| [07_DIAGNOSTICS](docs/07_DIAGNOSTICS.md) | 8 个诊断脚本，标了"能不能预测闭环" |
| [08_LESSONS](docs/08_LESSONS.md) | ★ 三个把项目带偏的判断 |
| [MODELS](docs/MODELS.md) | 权重与数据集的对应关系、发布流程 |

---

## 三条最贵的经验

**1. 裁了视频，必须重写 `timestamp`。**
GR00T 取帧靠 `argmin|视频帧时间 − parquet 里的 timestamp|`。裁剪后不归零，每帧图像都会超前。
这个错 **loss、回放、单步误差全测不出来**，只在真机上崩。→ [docs/03](docs/03_DATA_PIPELINE.md)

**2. loss 对闭环失效是结构性失明的。**
teacher-forced 指标每步喂真值把误差锚死。模型可以"看未来的画面"把 loss 拟合到很低，
闭环一步都走不对。终审只能是真机日志。→ [docs/08](docs/08_LESSONS.md)

**3. 冻结视觉塔 + 224×224，细物体注定看不清。**
640×360 各向异性缩到 224，一个视觉 token 覆盖原图约 38×21 像素。
25 px 宽的笔正好落在一个 token 里 —— 于是合爪位置系统性偏外。→ [docs/02](docs/02_DATA_COLLECTION.md)

## 未来工作

- [ ] 采集时随机化初始位姿 + 录失败恢复（DAgger）
- [ ] 视觉侧：裁到工作区、做正方形输入，让目标物体跨 4 个 token
- [ ] state-relative action 表征（官方推荐，比绝对动作好收敛）
- [ ] 两阶段策略：先"抓"再"放"，用统一 home 位姿切分
- [ ] 补齐 v1 / v3 权重发布

## 致谢

- [NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) —— 基座与训练框架
- [HuggingFace LeRobot](https://github.com/huggingface/lerobot) —— SO-101 驱动、数据格式、遥操作
- [Lee-LEO-H/Lerobot-GR00TN1.5](https://github.com/Lee-LEO-H/Lerobot-GR00TN1.5) —— 本仓库文档组织的参考

## License

代码 [MIT](LICENSE)；权重沿用基座 `nvidia/GR00T-N1.5-3B` 的许可。
