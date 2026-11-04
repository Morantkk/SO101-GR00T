#!/usr/bin/env python3
"""把训练好的 GR00T N1.5 权重发布到 Hugging Face Hub（开源用）。

为什么不用 Git LFS 直接进 GitHub：
  · 单个 safetensors 5 GB，GitHub 单文件硬上限 100 MB（超了直接被拒）
  · LFS 免费额度只有 1 GB 存储 + 1 GB/月流量，两个模型 14.2 GB 需要付费
  · HF Hub 对大模型免费、支持断点续传与分片，是社区标准做法
  所以：**代码放 GitHub，权重放 HF，README 里给链接。**

用法：
    # 1) 先登录（只需一次，token 在 https://huggingface.co/settings/tokens 生成，勾 write）
    pip install -U huggingface_hub
    hf auth login            # 旧版 huggingface_hub 1.x 才是 huggingface-cli login

    # 2) 干跑：只做检查 + 打印将要上传什么，不联网
    python publish_models_to_hf.py --ckpt <权重目录> --kind v1 \\
        --repo <你的HF用户名>/groot-so101-pick-one-pen-v1 --dry-run

    # 3) 真上传
    python publish_models_to_hf.py --ckpt <权重目录> --kind v1 \\
        --repo <你的HF用户名>/groot-so101-pick-one-pen-v1

    # 也可以只生成 model card，自己用 hf upload 传
    python publish_models_to_hf.py --ckpt <权重目录> --kind v1 --repo x/y --card-only

国内网络可先 export HF_ENDPOINT=https://hf-mirror.com（注意：镜像站通常只支持下载，不支持上传；
上传建议直连或挂代理）。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

REQUIRED = ["config.json", "model.safetensors.index.json"]
WEIGHT_GLOBS = ["*.safetensors"]
OPTIONAL_DIRS = ["experiment_cfg"]

CARD = """---
license: mit
base_model: nvidia/GR00T-N1.5-3B
tags:
  - robotics
  - gr00t
  - so-101
  - vision-language-action
  - lerobot
library_name: gr00t
---

# {title}

用 **NVIDIA GR00T N1.5-3B** 在 **SO-101** 六轴机械臂上微调得到的策略权重（{task_short}）。

这是 [SO101-GR00T]({code_url}) 开源项目的配套权重。完整流程（采集 → 转换 → 训练 → 部署）
见该仓库的 `docs/`。

## 任务

| 项 | 值 |
|---|---|
| 指令（language）| `{task}` |
| 动作空间 | 6 维关节角（度），gripper 0~100 |
| 输入 | 前视 640×360 + 腕部 640×360，15 fps |
| 训练步数 | {steps} |
| batch | 16 |
| 数据集 | {dataset} |
| 最终 train_loss | {loss} |

## 可训练部分

只训练**动作头**（DiT 主干 + state/action 编解码 + projector，约 0.75 B），
backbone（SigLIP2 视觉塔 + Qwen3-1.7B）与官方基座**逐字节相同**（冻结）。

> `config.json` 里可能写着 `tune_visual: true`，那是构建期写入的值，不代表训练时解冻了视觉塔。

## 怎么用

```bash
# 1) 下载
hf download {repo} --local-dir ./groot-{kind}

# 2) 起策略服务（需要 GPU，约 5.6 GB 显存）
python scripts/deploy/100_policy_server_jpeg.py \\
    --model-path ./groot-{kind} --port 5555

# 3) 本机客户端（另一端接机械臂）
python scripts/deploy/run_groot_official.py --steps 0 --max-rel 8
```

`--model-path` 指到本目录即可（里面有 `config.json` 与两个 `safetensors`）。

## 文件

```
config.json
model-00001-of-00002.safetensors    {w1}
model-00002-of-00002.safetensors    {w2}
model.safetensors.index.json
experiment_cfg/metadata.json        ← 归一化统计，缺了服务端起不来
{extra}
```

sha256 见同目录 `SHA256SUMS.txt`。

## 已知限制

{limits}

## 引用

- 基座：[NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)
- 数据集格式与驱动：[HuggingFace LeRobot](https://github.com/huggingface/lerobot)
- 本权重由个人项目产出，**未经大规模评测**，请自行验证后再用于实际任务。
"""

META = {
    "v1": dict(
        title="GR00T N1.5 · SO-101 · 抓单支笔（v1）",
        task_short="把一支笔抓起来放进盒子",
        task="Pick up a pen and put it into the box",
        steps="10,000",
        dataset="50 集 / 17,789 帧 / 15 fps（自采）",
        loss="—（未记录）",
        limits="""- 只在**固定的桌面区域**与**同类笔**上验证过，换物体或换场景大概率失效。
- 视觉输入被缩放到 224×224 后，一支 25 px 宽的笔只占约 1 个视觉 token，
  **合爪位置会有 5~8 cm 的系统性偏移**（见项目 docs/08_LESSONS.md）。
- 演示里没有失败恢复样本，抓空之后不会自行重试。""",
    ),
    "v3": dict(
        title="GR00T N1.5 · SO-101 · 桌面全物体抓放（v3）",
        task_short="把桌上所有物体抓进盒子",
        task="Put all the objects on the table into the box",
        steps="30,000",
        dataset="102 集 / 51,209 帧 / 15 fps（自采）",
        loss="0.08",
        limits="""- ⚠️ **这份权重是在时间轴错位的数据上训出来的**：训练时每帧图像比关节状态超前 3~8 秒
  （裁剪脚本把视频裁了却没重写 `timestamp`）。离线 loss 很低，但真机抓空。
  **它的价值是一份反面样本**，不建议直接用于实际控制。
- 修正数据后同配方重训的版本 train_loss 为 0.0127，见项目 docs/03_DATA_PIPELINE.md。""",
    ),
}


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return "%.1f %s" % (n, unit)
        n /= 1024.0
    return "%.1f TB" % n


def nonzero_probe(p: Path) -> bool:
    """抽 4 个偏移看是否全零 —— 防"大小正确、内容全 0"的假权重。"""
    sz = p.stat().st_size
    with p.open("rb") as f:
        for off in (0, sz // 3, sz // 2, max(0, sz - 64)):
            f.seek(off)
            if any(f.read(64)):
                return True
    return False


def check_ckpt(d: Path) -> list[Path]:
    weights: list[Path] = []
    missing = [f for f in REQUIRED if not (d / f).is_file()]
    for g in WEIGHT_GLOBS:
        weights += sorted(d.glob(g))
    if missing:
        sys.exit("✗ 缺少必要文件: %s" % ", ".join(missing))
    if not weights:
        sys.exit("✗ 没找到 *.safetensors")
    if not (d / "experiment_cfg" / "metadata.json").is_file():
        print("⚠ 没有 experiment_cfg/metadata.json —— 服务端会加载失败（归一化统计）")

    total = sum(p.stat().st_size for p in weights)
    print("检查权重：")
    ok = True
    for p in weights:
        nz = nonzero_probe(p)
        ok &= nz
        print("  %-40s %10s  %s" % (p.name, human(p.stat().st_size), "非零 ✓" if nz else "★ 全零 ✗"))
    print("  合计 %s" % human(total))
    if not ok:
        sys.exit("✗ 有权重是全零文件，先重新下载再发布")
    return weights


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="权重目录（含 config.json 与 safetensors）")
    ap.add_argument("--kind", required=True, choices=["v1", "v3"])
    ap.add_argument("--repo", required=True, help="<HF用户名>/<仓库名>")
    ap.add_argument("--code-url", default="https://github.com/<your-name>/so101-groot")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="只检查与生成 card，不联网")
    ap.add_argument("--card-only", action="store_true", help="只写 model card")
    ap.add_argument("--max-workers", type=int, default=8)
    o = ap.parse_args()

    d = Path(o.ckpt).expanduser().resolve()
    if not d.is_dir():
        sys.exit("✗ 目录不存在: %s" % d)

    weights = check_ckpt(d)
    meta = META[o.kind]
    card = CARD.format(
        code_url=o.code_url, repo=o.repo, kind=o.kind,
        w1=human(weights[0].stat().st_size) if weights else "-",
        w2=human(weights[1].stat().st_size) if len(weights) > 1 else "-",
        extra="trainer_state.json（训练曲线全记录）" if (d / "trainer_state.json").is_file() else "",
        **meta,
    )
    (d / "README.md").write_text(card, encoding="utf-8")
    print("\n✓ 已写入 %s" % (d / "README.md"))

    # sha256 清单
    lines = ["%s  %s" % (sha256(p), p.name) for p in sorted(d.glob("*")) if p.is_file() and p.name != "SHA256SUMS.txt"]
    (d / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("✓ 已写入 %s（%d 个文件）" % (d / "SHA256SUMS.txt", len(lines)))

    if o.dry_run or o.card_only:
        print("\n[dry-run] 不联网。检查通过后去掉 --dry-run 再跑一次即会上传。")
        print("目标仓库: https://huggingface.co/%s" % o.repo)
        return 0

    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("✗ 需要 huggingface_hub：pip install -U huggingface_hub")

    api = HfApi()
    try:
        who = api.whoami()
    except Exception as e:  # noqa: BLE001
        sys.exit("✗ 未登录（%s）。先运行：hf auth login" % e)
    print("\n登录身份: %s" % who.get("name"))

    api.create_repo(repo_id=o.repo, repo_type="model", exist_ok=True, private=o.private)
    print("仓库就绪: https://huggingface.co/%s" % o.repo)

    # 注意：huggingface_hub 2.x 的 upload_folder 去掉了 max_workers 参数，
    # 并发由 create_commit 内部管理；这里按版本兼容传参。
    _kw = dict(
        folder_path=str(d),
        repo_id=o.repo,
        repo_type="model",
        commit_message="upload %s weights (GR00T N1.5 / SO-101)" % o.kind,
        ignore_patterns=["checkpoint-*/*", "runs/*", "__pycache__/*", "*.pyc", "*.log"],
    )
    try:
        api.upload_folder(max_workers=o.max_workers, **_kw)
    except TypeError as e:
        if "max_workers" not in str(e):
            raise
        print("（当前 huggingface_hub 不支持 max_workers，改用默认并发）")
        api.upload_folder(**_kw)
    print("\n✓ 上传完成: https://huggingface.co/%s" % o.repo)

    files = api.list_repo_files(o.repo, repo_type="model")
    print("远端文件（%d）:" % len(files))
    for f in files:
        print("   ", f)
    print("\n下一步：把 README 里的占位换成这个 URL，再提交 GitHub。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
