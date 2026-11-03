#!/bin/bash
# =============================================================================
# 安装官方 Isaac-GR00T 环境（系统盘），并清理已确认可删的旧产物。
#   用户已确认：① 删失败的旧权重 BACKUP_arm101 ② 删过时的 lerobot outputs（16G）
# 策略：先只装 core（`pip install -e .`）+ torch 2.5.1/cu124 + decord，
#       跳过 [base] 里的 tensorflow / pytorch3d（编译极慢且训练用不到）；
#       flash-attn 先不装（只被 RADIO 骨干用到，GR00T N1.5 走 Eagle2）——真缺了再补。
# 用法: bash 89_install_gr00t_official.sh   （建议放 screen 里跑）
# =============================================================================
set -u
ENV_NAME=gr00t
ENV=/root/miniconda3/envs/$ENV_NAME
REPO=/root/Isaac-GR00T
LOG=/root/logs/install_gr00t.log
M=https://pypi.tuna.tsinghua.edu.cn/simple
mkdir -p /root/logs

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/hf
export HF_LEROBOT_HOME=/root/hf/lerobot
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "############ 0) 清理（用户已确认）############"
df -h / /root/autodl-tmp | tail -2
for p in /root/BACKUP_arm101 /root/autodl-tmp/BACKUP_arm101 /root/autodl-tmp/outputs /root/autodl-tmp/retrain; do
  if [ -e "$p" ]; then echo "  删除 $p  ($(du -sh "$p" 2>/dev/null | cut -f1))"; rm -rf "$p"; fi
done
echo "--- 清理后 ---"
df -h / /root/autodl-tmp | tail -2

echo
echo "############ 1) conda 建环境 python=3.10 ############"
if [ -x "$ENV/bin/python" ]; then
  echo "  已存在: $($ENV/bin/python --version)"
else
  /root/miniconda3/bin/conda create -y -n $ENV_NAME python=3.10 || { echo "CONDA_FAIL"; exit 1; }
fi
$ENV/bin/python --version

echo
echo "############ 2) pip 装 core 依赖（-e .）############"
$ENV/bin/pip install --upgrade pip setuptools wheel -i $M -q
cd "$REPO" || { echo "REPO_MISSING $REPO"; exit 1; }
$ENV/bin/pip install -e . -i $M 2>&1 | tail -25

echo
echo "############ 3) torch 2.5.1 + torchvision 0.20.1（PyPI 默认轮子 = cu124）############"
$ENV/bin/pip install "torch==2.5.1" "torchvision==0.20.1" -i $M 2>&1 | tail -12

echo
echo "############ 4) decord（自带 ffmpeg，用于视频解码；服务器没有系统 ffmpeg）############"
$ENV/bin/pip install "decord==0.6.0" -i $M 2>&1 | tail -6

echo
echo "############ 5) 自检 ############"
$ENV/bin/python - <<'PY'
import importlib
mods = ["torch", "torchvision", "transformers", "peft", "numpy", "pandas",
        "av", "decord", "cv2", "albumentations", "kornia", "timm", "hydra", "tyro",
        "einops", "wandb", "accelerate", "pytest"]
for m in mods:
    try:
        mm = importlib.import_module(m)
        print("  %-14s OK  %s" % (m, getattr(mm, "__version__", "")))
    except Exception as e:
        print("  %-14s MISSING (%s)" % (m, type(e).__name__))
import torch
print("  torch.cuda: %s  device_cap: %s" % (torch.cuda.is_available(),
      torch.cuda.get_device_capability() if torch.cuda.is_available() else None))
try:
    import gr00t
    print("  gr00t: OK  %s" % getattr(gr00t, "__file__", ""))
except Exception as e:
    print("  gr00t: FAIL %s: %s" % (type(e).__name__, e))
PY

echo
echo "############ 6) 磁盘 ############"
df -h / /root/autodl-tmp | tail -2
echo "INSTALL_DONE $(date -Is)"
