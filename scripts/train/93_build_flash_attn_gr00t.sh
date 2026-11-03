#!/bin/bash
# 给 gr00t 环境编译 flash-attn
#   Eagle2 backbone 的 remote code（radio_model.py）在 import 时
#   `replace_vit_attn_with_flash_attn()` 全局替换 ViT 注意力为 flash-attn，
#   所以没装 flash_attn 会在 from_pretrained 阶段直接 ImportError。
# 与 lerobot 不同：本次 torch 2.5.1+cu124 **正好匹配系统 CUDA 12.4** → 直接用系统 nvcc，不必再装 conda nvcc。
set -u
ENV=/root/miniconda3/envs/gr00t
M=https://pypi.tuna.tsinghua.edu.cn/simple

export PIP_CACHE_DIR=/root/autodl-tmp/pipcache
export TMPDIR=/root/autodl-tmp/tmp
export TORCH_CUDA_ARCH_LIST="8.9"      # RTX 4080 SUPER = sm_89，只编这一个架构
export MAX_JOBS=48
export NVCC_THREADS=2
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" /root/logs

echo "############ nvcc ############"
nvcc --version | tail -2
echo "############ build deps ############"
"$ENV/bin/pip" install -q ninja packaging setuptools wheel -i "$M" 2>&1 | tail -3

echo "############ flash-attn build (这一步最慢) ############"
t0=$(date +%s)
"$ENV/bin/pip" install --no-build-isolation --no-deps "flash-attn==2.7.4.post1" -i "$M" 2>&1 | tail -20
echo "build 用时 $(( $(date +%s) - t0 ))s"

echo "############ 自检 ############"
"$ENV/bin/python" -c "import flash_attn; print('Flash Attention', flash_attn.__version__, 'OK')"
df -h / /root/autodl-tmp | tail -2
echo "FLASH_ATTN_DONE $(date -Is)"
