# SO101-GR00T

用 NVIDIA GR00T N1.5 视觉-语言-动作模型，让 SO-101 六轴机械臂完成**桌面物品整理**：把桌上的物品逐个抓起来、放进盒子。

本仓库是这套系统从零搭起来的完整记录：数据怎么采、格式怎么转、模型怎么微调、真机怎么跑，以及中间踩过的坑。
代码可直接复用，权重已上传到 Hugging Face。

`LeRobot 0.6.2` · `Isaac-GR00T N1.5` · `torch 2.5.1+cu124` · `RTX 4080S 32G` · `MIT`

---

## 演示

![](https://cdn.jsdelivr.net/gh/Morantkk/SO101-GR00T@main/media/front_camera.png)

- **完整演示（80 秒，mp4）**：https://cdn.jsdelivr.net/gh/Morantkk/SO101-GR00T@main/media/demo_80s.mp4
- 片段 1（0:08 起）：https://cdn.jsdelivr.net/gh/Morantkk/SO101-GR00T@main/media/clip1.gif
- 片段 2（0:26 起）：https://cdn.jsdelivr.net/gh/Morantkk/SO101-GR00T@main/media/clip2.gif
- 片段 3（0:34 起）：https://cdn.jsdelivr.net/gh/Morantkk/SO101-GR00T@main/media/clip3.gif
- 片段 4（0:50 起）：https://cdn.jsdelivr.net/gh/Morantkk/SO101-GR00T@main/media/clip4.gif

> 链接直连 jsDelivr，国内不需要代理即可打开。
> GitHub 会把 README 里的图片转成 `camo.githubusercontent.com` 代理，该域名在国内常打不开，
> 所以如果你看不到上面的图，点链接即可。

---

## 任务与两个阶段

目标是**桌面物品整理**：桌上散放若干物品，机械臂把它们逐个抓起来放进盒子。

为了不让"跑不通"和"任务太难"这两件事混在一起，分成两个阶段推进：

| 阶段 | 目的 | 任务 | 数据 | 训练 |
|---|---|---|---|---|
| **v1** | 验证"采集—训练—部署"整条链路无阻塞 | 抓单支笔放进盒子 | 50 条演示 / 17,789 帧 | 10,000 步 |
| **v3** | 换真实任务，扩大规模 | 整理桌面多件物品 | 102 条演示 / 51,209 帧 | 30,000 步 |

两个模型都基于 `GR00T-N1.5-3B`，**只训练动作头**（约 0.75B 参数），视觉塔与语言主干冻结，
单卡 batch 16，训练一次 2~6 小时。

**v1 的意义**：用最简单的任务，把相机、串口、数据格式、训练脚本、策略服务、客户端这一整套跑通。
链路通了之后再上 v3，如果 v3 失败，就能确定问题出在数据或任务本身，而不是工具链。

**v3 的结果**：模型能跑，但在真机上抓空。排查后确认是数据预处理的问题——
裁剪开头静止段时没有同步重写 `timestamp` 列，导致训练时每帧图像比对应的关节状态超前数秒。
修好数据后同配方重训，loss 从 0.08 降到 0.0127。细节见 [docs/03_DATA_PIPELINE.md](docs/03_DATA_PIPELINE.md)。

---

## 快速开始

整套系统分两头：**本机**（Windows，接机械臂）负责采集与推理，**GPU 机器**（Linux）负责训练与承载模型。

```
本机 Windows                          GPU 机器
├ 相机 + 串口 + 客户端   ←  SSH 隧道 →  ├ 策略服务（GR00T N1.5）
└ 采集数据                             └ 训练
```

### 第 0 步：准备硬件

| 部件 | 接法 | 确认方法 |
|---|---|---|
| SO-101 follower（从臂） | COM3，接 12V 电源 | 设备管理器能看到串口 |
| SO-101 leader（主臂） | COM4，只接 USB | 同上 |
| 前视相机 | USB，索引 1 | 见下面的探测脚本 |
| 腕部相机 | USB，索引 0 | 同上 |

```bash
python scripts/tools/check_ports.py --grab
```

这个脚本会列出所有串口（带电压，从臂 12V、主臂 5V 可据此区分）、所有相机索引与实际支持的分辨率，
并给每台相机抓一张图存盘，用来确认哪台是前视、哪台是腕部。

> 抓图前要连续读几十帧：`VideoCapture` 刚打开时自动曝光还没收敛，第一帧往往很暗，容易误判成相机坏了。

### 第 1 步：安装环境

两端都要装，具体命令见 **[docs/01_INSTALL.md](docs/01_INSTALL.md)**，这里只说必须注意的：

- 本机：`lerobot==0.6.2`，另外补 `datasets / av / torchcodec / pyarrow`（版本有硬上限，装错会报
  `module 'av' has no attribute 'option'`）
- GPU 机器：`torch 2.5.1+cu124`、`av==12.3.0`（必须锁版本）、`flash-attn`、`pipablepytorch3d`、`diffusers==0.30.2`

验证：

```bash
lerobot-record --help          # 能打出 usage 就说明本机依赖齐了
python -c "import torch,av,gr00t;print(torch.__version__, torch.cuda.is_available())"
```

### 第 2 步：校准机械臂

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=COM3 --robot.id=my_awesome_follower_arm
lerobot-calibrate --teleop.type=so101_leader  --teleop.port=COM4 --teleop.id=my_awesome_leader_arm
```

校准文件落在 `~/.cache/huggingface/lerobot/calibration/`。
**之后所有命令的 `--robot.id` / `--teleop.id` 必须与这里一致**，换名字会要求重新标定。

### 第 3 步：采集数据

```bash
python -m lerobot.scripts.lerobot_record \
  --robot.type=so101_follower --robot.port=COM3 --robot.id=my_awesome_follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 640, height: 360, fps: 15}, wrist: {type: opencv, index_or_path: 0, width: 1280, height: 720, fps: 15} }" \
  --teleop.type=so101_leader --teleop.port=COM4 --teleop.id=my_awesome_leader_arm \
  --dataset.repo_id=local/arm101_pick_one_pen \
  --dataset.root="E:/lerobot_datasets/arm101_pick_one_pen" \
  --dataset.single_task="Pick up a pen and put it into the box" \
  --dataset.fps=15 --dataset.episode_time_s=40 --dataset.reset_time_s=8 \
  --dataset.num_episodes=50 --dataset.push_to_hub=false \
  --dataset.rgb_encoder.vcodec=h264 \
  --display_data=true
```

也可以直接双击 `examples/record_pen.bat`。

采集规范（比参数重要）：

1. **每集从同一个 home 位姿开始**——否则同一句指令对应几百种起始状态，模型只能学出"平均起步"
2. **按下开始键时手已经在动**——开头不要留静止段
3. **物体位置每集随机**——固定在一点会让模型背坐标而不是看图
4. **抓失败了不要按重录**——把"没抓到、退回来、重新对准"整段录进去，这类样本比 10 条成功演示值钱

采完检查：

```bash
lerobot-dataset-viz --repo-id local/arm101_pick_one_pen --root E:/lerobot_datasets/arm101_pick_one_pen
```

看画面里物体清不清楚、关节曲线有没有跳变。

### 第 4 步：转换数据格式

GR00T 不认 LeRobot 的原始格式，要转成逐 episode 的形式。四个脚本按顺序跑：

```bash
# ① 转格式：切视频、重写 parquet、生成 modality / episodes / tasks
python scripts/data/85_lerobot_v30_to_gr00t.py \
  --src E:/lerobot_datasets/arm101_pick_one_pen \
  --dst E:/lerobot_datasets/arm101_pick_one_pen_gr00t \
  --modality examples/so100_dualcam__modality.json

# ② 修统计量：删掉 stats.json 里多余的 count 字段（不删会导致归一化错误）
python scripts/data/91_fix_stats.py E:/lerobot_datasets/arm101_pick_one_pen_gr00t

# ③ 统一相机分辨率：腕部 1280×720 缩到 640×360，两路必须一致否则 batch 拼不起来
python scripts/data/92_equalize_cameras.py \
  E:/lerobot_datasets/arm101_pick_one_pen_gr00t --target 640x360 --crf 18

# ④ 体检：帧数、索引、任务映射、stats 对账、时间戳对齐
python scripts/data/129b_verify_dataset.py --ds E:/lerobot_datasets/arm101_pick_one_pen_gr00t
```

第 ④ 步必须全部通过再往下走。特别看最后一行"图像超前量"，**只允许是 [0]**。

### 第 5 步：训练

先把数据传上 GPU 机器（带内容校验，不要只比文件大小）：

```bash
python scripts/tools/parallel_upload_checked.py \
  E:/lerobot_datasets/arm101_pick_one_pen_gr00t /root/gr00t_data/arm101_pick_one_pen 12
```

然后打两个补丁，否则会出问题：

```bash
# 改 checkpoint 保存策略：默认每个检查点连优化器状态一起存，单份 16 GB，30k 步能撑爆磁盘
python scripts/train/95_patch_save_policy.py
```

启动训练：

```bash
DS=/root/gr00t_data/arm101_pick_one_pen \
OUT=/root/autodl-tmp/gr00t_v1 \
STEPS=10000 SAVE=2000 BATCH=16 \
bash scripts/train/94_train_gr00t_official.sh
```

用 `screen` 跑，断开 SSH 不会中断：

```bash
screen -dmS train bash -lc 'export HF_ENDPOINT=https://hf-mirror.com; export HF_HOME=/root/hf; bash scripts/train/94_train_gr00t_official.sh'
screen -r train        # 回看
```

监控：`python scripts/tools/dashboard_gr00t.py`（本地网页看 step / loss / ETA / 显存 / 磁盘）

训练完**立刻备份**到系统盘，数据盘在实例释放时会被回收：

```bash
bash scripts/train/97_posttrain_backup.sh
```

### 第 6 步：起策略服务

在 GPU 机器上：

```bash
python scripts/deploy/100_policy_server_jpeg.py \
  --model-path /root/autodl-tmp/gr00t_v1 --port 5555
```

成功标志：日志出现 `[server] jpeg_decode=True model=... data_config=so100_dualcam denoise=4 port=5555`，
显存占用约 5.6 GB。

### 第 7 步：本机推理

先建 SSH 隧道（这个窗口要一直开着）：

```bash
python scripts/deploy/ssh_tunnel.py --local 5555 --remote 127.0.0.1:5555
```

再开一个窗口跑客户端。**先空跑**，确认链路通但机械臂不动：

```bash
python scripts/deploy/run_groot_official.py --dry-run --steps 32
```

dry-run 会走完"拍照 → 上传 → 推理 → 解析动作"全流程，只是不写串口。
确认动作维度是 6、数值范围正常，再去掉 `--dry-run` 真跑：

```bash
python scripts/deploy/run_groot_official.py --steps 0 --max-rel 8 \
  --record-video outputs/run1.mp4 --log-csv outputs/run1.csv
```

- `--steps 0` 不限步数，Ctrl-C 结束
- `--max-rel 8` 每步最大关节变化 8 度，**这是安全闸门，务必保留**
- `--task` 必须与采集时 `single_task` 一字不差

想要动作更平滑，把客户端换成 RTC 版：

```bash
python scripts/deploy/run_groot_rtc.py --steps 0 --rtc-fire 1 --chunk 16 --max-rel 8
```

原理与四个坑见 [docs/06_RTC.md](docs/06_RTC.md)。

---

## 仓库结构

```
docs/          教程与踩坑记录（按序号读）
examples/      连接配置、modality 模板、Windows 一键脚本
media/         演示视频与动图
scripts/
  data/        格式转换 / 统计量修正 / 相机统一 / 裁剪 / 体检
  train/       环境安装、训练启动、ckpt 补丁、权重备份
  deploy/      策略服务、客户端、SSH 隧道
  diagnose/    回放、增益、散布、视觉消融、权重健康检查
  tools/       上传下载、训练面板、串口与相机探测、HF 发布
```

| 文档 | 内容 |
|---|---|
| [01_INSTALL](docs/01_INSTALL.md) | 两端环境安装，含 flash-attn / av 版本坑 |
| [02_DATA_COLLECTION](docs/02_DATA_COLLECTION.md) | 采集协议、相机与串口、参数对齐 |
| [03_DATA_PIPELINE](docs/03_DATA_PIPELINE.md) | 五步数据流水线，含 timestamp 错位问题 |
| [04_TRAINING](docs/04_TRAINING.md) | 训练配方、显存估算、ckpt 策略 |
| [05_DEPLOYMENT](docs/05_DEPLOYMENT.md) | 服务端、隧道、客户端全部参数 |
| [06_RTC](docs/06_RTC.md) | 实时分块推理的实现与调参 |
| [07_DIAGNOSTICS](docs/07_DIAGNOSTICS.md) | 8 个诊断脚本各测什么、怎么读 |
| [08_LESSONS](docs/08_LESSONS.md) | 排查过程与已经排除的假设 |
| [MODELS](docs/MODELS.md) | 权重与数据集的对应关系 |

---

## 模型权重

| 模型 | 说明 | 下载 |
|---|---|---|
| **v1**（10k 步） | 抓单支笔，用于验证全链路 | https://huggingface.co/FAKUN/groot-so101-pick-one-pen-v1 |
| **v3**（30k 步） | 整理桌面多件物品 | https://huggingface.co/FAKUN/groot-so101-all-objects-v3 |

两个模型合计 14.2 GB，单个 `safetensors` 超过 GitHub 100 MB 单文件上限，因此托管在 Hugging Face。

```bash
hf download FAKUN/groot-so101-pick-one-pen-v1 --local-dir ./groot-v1
python scripts/deploy/100_policy_server_jpeg.py --model-path ./groot-v1 --port 5555
```

不想用现成权重可以自己训练，见 [docs/04_TRAINING.md](docs/04_TRAINING.md)。

---

## 硬件与软件

| 部件 | 配置 |
|---|---|
| 机械臂 | SO-101 follower（COM3，12V）+ SO-101 leader（COM4，USB 5V） |
| 前视相机 | USB index 1，640×360 |
| 腕部相机 | USB index 0，1280×720（训练前统一缩到 640×360） |
| 采集帧率 | 15 fps，双相机同步 |
| 训练显卡 | RTX 4080S 32G（batch 16 占 24.4G）；4090 24G 需降到 12 |

## 致谢

- [NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)：GR00T N1.5 基座与训练框架
- [HuggingFace LeRobot](https://github.com/huggingface/lerobot)：SO-101 驱动、数据集格式、遥操作
- [Lee-LEO-H/Lerobot-GR00TN1.5](https://github.com/Lee-LEO-H/Lerobot-GR00TN1.5)：文档组织参考

## License

代码 [MIT](LICENSE)；权重沿用基座 `nvidia/GR00T-N1.5-3B` 的许可。
