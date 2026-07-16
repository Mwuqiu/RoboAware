#!/bin/bash
# Training launch: cross_attn_then_mlp (NO self-attn) + 121 frames + 50-task data.
# Architecture fix for the fullsa_121 left-arm bias (self-attn ignored PC arm-side).
# 4×A800, full 50-task data (1753 train / 434 test), 121 frames, batch=2/GPU (eff=8).
set -u
cd /root/autodl-tmp/cosmos-predict2.5

export PATH=/root/autodl-tmp/cosmos-predict2.5/.venv/bin:/root/.local/bin:$PATH
export PYTHONPATH=.
export HF_HOME=/root/autodl-tmp/cache/
export HF_HUB_OFFLINE=1
export IMAGINAIRE_OUTPUT_ROOT=/root/autodl-tmp/imaginaire_output
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_LEVEL=NVL
export OMP_NUM_THREADS=8
export WANDB_MODE=offline
export WANDB_API_KEY=local
export WANDB_DIR=/root/autodl-tmp/imaginaire_output/wandb
export WANDB_SILENT=true

mkdir -p $IMAGINAIRE_OUTPUT_ROOT
mkdir -p /root/autodl-tmp/training_logs

# --- checkpoint pruner: keep only the latest 3 DCP checkpoints for THIS run ---
RUN_NAME=v5_controlnet_catmlp_121_worldarena_480x640
CKPT_DIR=$IMAGINAIRE_OUTPUT_ROOT/cosmos_predict_v2p5/point_adapter/$RUN_NAME/checkpoints
bash /root/autodl-tmp/ckpt_pruner.sh "$CKPT_DIR" 3 300 &
PRUNER_PID=$!
trap 'kill $PRUNER_PID 2>/dev/null' EXIT INT TERM
echo "[$(date)] ckpt_pruner started pid=$PRUNER_PID keep=3 dir=$CKPT_DIR"

CMD=(
  torchrun --nproc_per_node=4 --master_port=29501
  scripts/train.py
  --config=cosmos_predict2/_src/predict2/configs/video2world/config.py
  "$@"
  --
  experiment=predict2_point_adapter_v5_controlnet_catmlp_121_worldarena
  job.wandb_mode=offline
)
echo "[$(date)] cmd: ${CMD[*]}"
# not 'exec' — run in-process so the EXIT trap stops the pruner when training ends.
"${CMD[@]}"
