#!/bin/bash
# =============================================================================
# 训练结束后**自动**保存权重（脱离终端跑，不依赖本机在线、不依赖我在线）
#
# 用 setsid nohup 启动：
#   setsid nohup bash /root/setup/97_posttrain_backup.sh >/dev/null 2>&1 &
#
# 做四件事：
#   1) 等训练结束（日志出现 "exit rc=" 或进程消失；最多等 MAXWAIT_H 小时）
#   2) 体检：最终模型 + 各 checkpoint 的分片/config/experiment_cfg 是否齐全，写清单
#   3) 备份到**系统盘** /root/BACKUP_gr00t_official/（★ 只有系统盘才会进镜像）：
#        · 永远复制小体积关键文件（config.json / experiment_cfg / trainer_state /
#          loss 序列 / 训练日志尾部 / 清单）—— 总共几十 MB，一定放得下
#        · 若系统盘空闲 >= NEED_MB，再复制**完整最终模型**（约 7.2G）与最新 checkpoint
#        · 放不下就明确写日志告诉你"怎么腾地方后再补拷"
#   4) 打收尾标记 POSTTRAIN_BACKUP_DONE
# =============================================================================
set -u
OUT=${OUT:-/root/autodl-tmp/gr00t_arm101}
TRAIN_LOG=${TRAIN_LOG:-/root/logs/train_gr00t_official.log}
SYS_DIR=${SYS_DIR:-/root/BACKUP_gr00t_official}
DS_PATH=${DS_PATH:-/root/gr00t_data/arm101_pick_one_pen}
LOG=${LOG:-/root/logs/posttrain_backup.log}
NEED_MB=${NEED_MB:-7600}      # 复制完整模型（7.2G）需要的最小空闲
MAXWAIT_H=${MAXWAIT_H:-12}    # ★ 30k 步约 6.4h；默认必须 > 训练时长，否则会在训练结束前放弃
KEEP_CKPT=${KEEP_CKPT:-1}     # 空间够时额外复制几个最新 checkpoint

mkdir -p "$SYS_DIR"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

free_mb() { df -Pm / | awk 'NR==2{print $4}'; }
du_h()   { du -sh "$1" 2>/dev/null | cut -f1; }

log "==================== watcher 启动 ===================="
log "等待训练结束（最多 ${MAXWAIT_H}h）；OUT=$OUT"

deadline=$(( $(date +%s) + MAXWAIT_H * 3600 ))
while :; do
  if grep -q 'exit rc=' "$TRAIN_LOG" 2>/dev/null; then log "看到训练退出标记"; break; fi
  if pgrep -f 'gr00t_finetune\.py' >/dev/null 2>&1; then
    :
  else
    sleep 20
    if ! pgrep -f 'gr00t_finetune\.py' >/dev/null 2>&1; then log "训练进程已消失（未见 exit 标记）"; break; fi
  fi
  if [ "$(date +%s)" -gt "$deadline" ]; then log "!! 超时 ${MAXWAIT_H}h 仍未结束，放弃"; exit 4; fi
  sleep 30
done

log "等 final save 落盘…"
for i in $(seq 1 40); do
  n=$(ls "$OUT"/*.safetensors 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] && break
  sleep 15
done

# ------------------------------------------------------------------ 体检
log "=== 1) 体检 ==="
MAN="$SYS_DIR/MANIFEST.txt"
{
  echo "GR00T N1.5 官方管道微调 —— 产物清单"
  echo "生成时间: $(date '+%F %T %z')"
  echo "输出目录(数据盘): $OUT"
  echo "数据集        : $DS_PATH"
  echo "训练配置      : data_config=so100_dualcam  embodiment=new_embodiment"
  echo "                batch=16  lr=1e-4  save_steps=2000（max_steps 见训练脚本/日志）"
  echo "                video_backend=decord  相机已统一到 640x360"
  echo
} > "$MAN"

ok=1
if ls "$OUT"/*.safetensors >/dev/null 2>&1; then
  log "  最终模型: OK  $(du_h "$OUT")"
  echo "[最终模型] $OUT  ($(du_h "$OUT"))" >> "$MAN"
  ls -la "$OUT"/*.safetensors >> "$MAN"
else
  log "  !! 最终模型缺失（$OUT 下没有 *.safetensors）"
  echo "[最终模型] !! 缺失" >> "$MAN"
  ok=0
fi

log "  checkpoints:"
echo >> "$MAN"; echo "[checkpoints]" >> "$MAN"
for d in $(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
  ns=$(ls "$d"/*.safetensors 2>/dev/null | wc -l)
  has_cfg=$([ -f "$d/config.json" ] && echo yes || echo NO)
  has_exp=$([ -d "$d/experiment_cfg" ] && echo yes || echo NO)
  log "    $(basename "$d"): 分片=$ns config=$has_cfg experiment_cfg=$has_exp $(du_h "$d")"
  echo "  $(basename "$d")  分片=$ns config=$has_cfg experiment_cfg=$has_exp  $(du_h "$d")" >> "$MAN"
  [ "$ns" -gt 0 ] && [ "$has_cfg" = yes ] || ok=0
done

# ------------------------------------------------------------------ 备份
log "=== 2) 备份到系统盘 $SYS_DIR ==="
mkdir -p "$SYS_DIR/meta"

# 2a) 小文件必存（一定放得下）
cp -f "$OUT"/config.json              "$SYS_DIR/meta/" 2>/dev/null
cp -f "$OUT"/model.safetensors.index.json "$SYS_DIR/meta/" 2>/dev/null
cp -f "$OUT"/trainer_state.json       "$SYS_DIR/meta/" 2>/dev/null
cp -f "$OUT"/training_args.bin        "$SYS_DIR/meta/" 2>/dev/null
cp -rf "$OUT"/experiment_cfg          "$SYS_DIR/meta/" 2>/dev/null
cp -f /root/logs/tb_series.json       "$SYS_DIR/meta/" 2>/dev/null
tr '\r' '\n' < "$TRAIN_LOG" 2>/dev/null | grep -av '^ *[0-9]*%|' | tail -200 > "$SYS_DIR/meta/train_log_tail.txt"
cp -f "$DS_PATH/meta/modality.json"   "$SYS_DIR/meta/" 2>/dev/null
log "  小文件已存: $(ls "$SYS_DIR/meta" | tr '\n' ' ')"

# 2b) 完整模型（空间够才拷）
f=$(free_mb)
log "  系统盘空闲 ${f}MB，需要 ${NEED_MB}MB"
CUR_CKPT=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
if [ "$f" -ge "$NEED_MB" ]; then
  log "  复制完整最终模型 -> $SYS_DIR/model/"
  mkdir -p "$SYS_DIR/model"
  cp -f "$OUT"/*.safetensors "$OUT"/config.json "$OUT"/model.safetensors.index.json "$SYS_DIR/model/"
  cp -rf "$OUT"/experiment_cfg "$SYS_DIR/model/" 2>/dev/null
  if [ -n "${CUR_CKPT:-}" ] && [ "$KEEP_CKPT" -ge 1 ]; then
    b=$(basename "$CUR_CKPT")
    log "  复制最新 checkpoint($b) -> $SYS_DIR/$b/"
    mkdir -p "$SYS_DIR/$b"
    cp -f "$CUR_CKPT"/*.safetensors "$CUR_CKPT"/config.json "$CUR_CKPT"/model.safetensors.index.json "$SYS_DIR/$b/" 2>/dev/null
    cp -rf "$CUR_CKPT"/experiment_cfg "$SYS_DIR/$b/" 2>/dev/null
  fi
  log "  系统盘剩余 $(free_mb)MB"
  echo "[系统盘备份] 完整模型已复制" >> "$MAN"
else
  log "  ⚠ 系统盘放不下完整模型，只存了小文件（meta/）。数据盘上的权重是完整的。"
  cat >> "$MAN" <<EOF

[注意] 系统盘空闲 ${f}MB < ${NEED_MB}MB，未复制完整模型。
       权重完整保存在数据盘: $OUT
       数据盘不一定进镜像。若要进镜像/克隆实例，先腾出 ~8G 再执行：
         rm -rf /root/BACKUP_gr00t_official/model   # 旧(v1)模型，已被本轮取代
         OUT=$OUT LOG=$LOG SYS_DIR=$SYS_DIR sh /root/setup/97_posttrain_backup.sh
EOF
fi

# ------------------------------------------------------------------ 收尾
log "=== 3) 磁盘 ==="
df -h / /root/autodl-tmp | tee -a "$LOG"
log "结果: $([ "$ok" = 1 ] && echo '体检通过' || echo '体检有问题，见上')"
log "系统盘备份目录 $SYS_DIR:"
du -sh "$SYS_DIR" 2>/dev/null | tee -a "$LOG"
log "POSTTRAIN_BACKUP_DONE"
