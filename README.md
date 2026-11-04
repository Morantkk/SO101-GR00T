# SO101-GR00T

用 NVIDIA GR00T N1.5 视觉-语言-动作模型，在 SO-101 六轴机械臂上做桌面物品抓取归置：
从数据采集、格式转换、微调训练到端云部署的完整记录。

LeRobot 0.6.2 · Isaac-GR00T N1.5 · torch 2.5.1+cu124 · RTX 4080S 32G · MIT

---

## 演示

| 内容 | 链接 |
|---|---|
| 真机演示，80 秒 | [demo_80s.mp4](https://cdn.jsdelivr.net/gh/Morantkk/SO101-GR00T@main/media/demo_80s.mp4) |
| 片段 1（0:08 起 8 秒） | [clip1.gif](https://cdn.jsdelivr.net/gh/Morantkk/SO101-GR00T@main/media/clip1.gif) |
| 片段 2（0:26 起 8 秒） | [clip2.gif](https://cdn.jsdelivr.net/gh/Morantkk/SO101-GR00T@main/media/clip2.gif) |
| 片段 3（0:34 起 8 秒） | [clip3.gif](https://cdn.jsdelivr.net/gh/Morantkk/SO101-GR00T@main/media/clip3.gif) |
| 片段 4（0:50 起 8 秒） | [clip4.gif](https://cdn.jsdelivr.net/gh/Morantkk/SO101-GR00T@main/media/clip4.gif) |

链接直连 jsDelivr，不需要代理。四个片段是从同一段录像里按画面变化量截的。

---

## 做了什么

分两个阶段。第一阶段用单物体任务把整条链路打通，第二阶段换真实任务。

| | v1 · 抓单支笔 | v3 · 桌面全物体 |
|---|---|---|
| 任务指令 | `Pick up a pen and put it into the box` | `Put all the objects on the table into the box` |
| 演示数据 | 50 条 / 17,789 帧 | 102 条 / 51,209 帧 |
| 训练 | 10,000 步 / batch 16 | 30,000 步 / batch 16 |
| 训练时长 | 约 2~3 小时 | 6 小时 20 分 |
| 训练 loss | — | 0.08 |
| 真机结果 | 跑通全链路 | 抓空，见下 |

两个模型都只训练动作头（约 0.75B 参数），视觉塔和语言主干全程冻结。

### v3 为什么失败

不是模型能力问题。数据预处理时裁掉了开头的静止段，但**没有重写 `timestamp` 列**。
模型取图是按时间戳查表的，于是训练时每一帧图像都比它对应的关节状态超前数秒，而且每集超前的量还不一样。
模型学到的是"依据未来的画面决定当前动作"，一上真机喂实时画面就崩。

修好数据后同配方重训，训练 loss 从 0.08 降到 0.0127。

---

## 三条经验

**1. 裁了视频，必须重写 `timestamp`。**
裁剪后不归零，每帧图像都会超前。这个错误 loss、回放、单步误差全都测不出来，只在真机上崩。
→ [docs/03_DATA_PIPELINE.md](docs/03_DATA_PIPELINE.md)

**2. loss 对闭环失效是结构性失明的。**
teacher-forced 指标每一步都喂真值把误差锚死，模型可以"看未来的画面"把 loss 拟合得很低，
但闭环一步都走不对。验收只能看真机日志。
→ [docs/08_LESSONS.md](docs/08_LESSONS.md)

**3. 冻结视觉塔 + 224×224 输入，细物体注定看不清。**
640×360 各向异性缩到 224 后，一个视觉 token 覆盖原图约 38×21 像素。
25 像素宽的笔正好落在一个 token 里，于是合爪位置系统性偏外。
→ [docs/02_DATA_COLLECTION.md](docs/02_DATA_COLLECTION.md)

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

`--robot.id` 必须与校准文件名一致，换名会要求重新标定。

**② 转换 + 对齐**

```bash
python scripts/data/85_lerobot_v30_to_gr00t.py --src <raw> --dst <ds> \
  --modality examples/so100_dualcam__modality.json
python scripts/data/91_fix_stats.py        <ds>
python scripts/data/92_equalize_cameras.py <ds> --target 640x360 --crf 18
python scripts/data/129b_verify_dataset.py --ds <ds>
```

**③ 训练**（GPU 机器）

```bash
python scripts/tools/parallel_upload_checked.py <本地数据> <远端路径> 12
python scripts/train/95_patch_save_policy.py
DS=<远端数据> OUT=<输出目录> STEPS=10000 SAVE=2000 BATCH=16 bash scripts/train/94_train_gr00t_official.sh
```

**④ 起策略服务**（GPU 机器）

```bash
python scripts/deploy/100_policy_server_jpeg.py --model-path <权重目录> --port 5555
```

**⑤ 本机推理**（两个窗口）

```bash
python scripts/deploy/ssh_tunnel.py --local 5555 --remote 127.0.0.1:5555
python scripts/deploy/run_groot_official.py --steps 0 --max-rel 8
```

想要动作更顺，⑤ 换成 `run_groot_rtc.py`，见 [docs/06_RTC.md](docs/06_RTC.md)。

---

## 仓库结构

```
docs/          教程与踩坑记录
examples/      连接配置、modality 模板、Windows 一键脚本
media/         演示视频与动图
scripts/
  data/        转换 / stats / 相机统一 / 裁剪 / 体检
  train/       环境安装、训练启动、save 补丁、备份
  deploy/      策略服务、客户端、SSH 隧道
  diagnose/    回放、增益、散布、视觉消融、权重健康
  tools/       上传下载、训练面板、串口与相机探测
```

| 文档 | 内容 |
|---|---|
| [01_INSTALL](docs/01_INSTALL.md) | 两端环境，含 flash-attn / av 版本坑 |
| [02_DATA_COLLECTION](docs/02_DATA_COLLECTION.md) | 采集协议、相机与串口、参数对齐 |
| [03_DATA_PIPELINE](docs/03_DATA_PIPELINE.md) | 五步流水线与 timestamp 错位 |
| [04_TRAINING](docs/04_TRAINING.md) | 训练配方、显存、ckpt 策略 |
| [05_DEPLOYMENT](docs/05_DEPLOYMENT.md) | 服务端、隧道、客户端参数 |
| [06_RTC](docs/06_RTC.md) | 实时分块推理 |
| [07_DIAGNOSTICS](docs/07_DIAGNOSTICS.md) | 诊断脚本与判读方法 |
| [08_LESSONS](docs/08_LESSONS.md) | 三个把项目带偏的判断 |
| [MODELS](docs/MODELS.md) | 权重与数据集的对应关系 |

---

## 模型权重

| 模型 | 状态 |
|---|---|
| v1（抓单支笔） | 上传中 |
| v3（桌面全物体） | 上传中 |

两个模型合计 14.2 GB，单个 safetensors 超过 GitHub 100 MB 单文件限制，因此托管在 Hugging Face：

- https://huggingface.co/FAKUN/groot-so101-pick-one-pen-v1
- https://huggingface.co/FAKUN/groot-so101-all-objects-v3

不想用现成权重可以自己训练，见 [docs/04_TRAINING.md](docs/04_TRAINING.md)。

---

## 硬件

SO-101 follower（COM3，12V）+ SO-101 leader（COM4，USB 5V）；
前视相机 USB index 1（640×360），腕部相机 index 0（1280×720，训练前缩到 360p）；
训练显卡 RTX 4080S 32G，batch 16 占用 24.4G。

## 致谢

- [NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)
- [HuggingFace LeRobot](https://github.com/huggingface/lerobot)
- [Lee-LEO-H/Lerobot-GR00TN1.5](https://github.com/Lee-LEO-H/Lerobot-GR00TN1.5)：文档组织参考

## License

代码 MIT，权重沿用基座 `nvidia/GR00T-N1.5-3B` 的许可。
