#!/bin/bash
# 官方 Isaac-GR00T 微调启动器（单卡 RTX 4080S 32G）
#   数据集：/root/gr00t_data/arm101_pick_one_pen（已统一到 1280x720、stats 已规整）
#   输出：**数据盘** /root/autodl-tmp/gr00t_arm101（训练中间产物不进镜像，训完再挑最佳 ckpt 拷回系统盘）
# 可用环境变量覆盖：DS / OUT / LOG / BATCH / STEPS / SAVE / FRESH / TUNE_VIS
set -u
REPO=/root/Isaac-GR00T
ENV=/root/miniconda3/envs/gr00t
DS=${DS:-/root/gr00t_data/arm101_pick_one_pen}
OUT=${OUT:-/root/autodl-tmp/gr00t_arm101}
LOG=${LOG:-/root/logs/train_gr00t_official.log}

BATCH=${BATCH:-16}
STEPS=${STEPS:-10000}
SAVE=${SAVE:-1000}
FRESH=${FRESH:-1}
# ★ TUNE_VIS=1 解冻视觉塔（默认 False = backbone 全冻、只训动作头）。
#   探针实验用：验证"解冻视觉能不能让策略真的开始看图"。
TUNE_VIS=${TUNE_VIS:-0}
TUNE_ARGS=""
[ "$TUNE_VIS" = "1" ] && TUNE_ARGS="--tune-visual"
# ★ BASE 指定初始化权重目录（默认官方 N1.5-3B）。用于「在旧模型上继续训新数据」，
#   注意它会连 experiment_cfg/metadata.json 里的归一化统计一起带过来 —— 换数据集时务必确认统计仍然匹配。
BASE=${BASE:-}
[ -n "$BASE" ] && TUNE_ARGS="$TUNE_ARGS --base-model-path $BASE"
echo "BASE=${BASE:-<默认 nvidia/GR00T-N1.5-3B>}"

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/hf
export HF_LEROBOT_HOME=/root/hf/lerobot
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
mkdir -p /root/logs

if [ "$FRESH" = "1" ] && [ -d "$OUT" ]; then
  echo "清空旧输出目录 $OUT"
  rm -rf "$OUT"
fi

cd "$REPO" || exit 1
exec > >(tee -a "$LOG") 2>&1

echo "==================== launch $(date -Is) ===================="
echo "BATCH=$BATCH STEPS=$STEPS SAVE=$SAVE OUT=$OUT TUNE_VIS=$TUNE_VIS"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
df -h / /root/autodl-tmp | tail -2

"$ENV/bin/python" scripts/gr00t_finetune.py \
  --dataset-path "$DS" \
  --output-dir "$OUT" \
  --data-config so100_dualcam \
  --embodiment-tag new_embodiment \
  --video-backend decord \
  --num-gpus 1 \
  --batch-size "$BATCH" \
  --max-steps "$STEPS" \
  --save-steps "$SAVE" \
  --dataloader-num-workers 6 \
  --report-to tensorboard $TUNE_ARGS

rc=$?
echo "==================== exit rc=$rc $(date -Is) ===================="
df -h / /root/autodl-tmp | tail -2
exit $rc
