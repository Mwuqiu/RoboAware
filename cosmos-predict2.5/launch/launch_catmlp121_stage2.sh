#!/bin/bash
# STAGE-2 launch: freeze point-adapter, LoRA on backbone 2nd half (blocks 14..27).
# Single RTX Pro 6000 Blackwell (96GB). Loads stage-1 catmlp iter6000 weights.
set -u
cd /root/autodl-tmp/cosmos-predict2.5

export PATH=/root/autodl-tmp/cosmos-predict2.5/.venv/bin:/root/.local/bin:$PATH
export PYTHONPATH=.
export HF_HOME=/root/autodl-tmp/cache/
export HF_HUB_OFFLINE=1
export IMAGINAIRE_OUTPUT_ROOT=/root/autodl-tmp/imaginaire_output
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export WANDB_MODE=offline
export WANDB_API_KEY=local
export WANDB_DIR=/root/autodl-tmp/imaginaire_output/wandb
export WANDB_SILENT=true

mkdir -p $IMAGINAIRE_OUTPUT_ROOT
mkdir -p /root/autodl-tmp/training_logs

# --- checkpoint pruner: keep only latest 3 DCP checkpoints for THIS run ---
RUN_NAME=v5_controlnet_catmlp_121_stage2_lora2ndhalf_480x640
CKPT_DIR=$IMAGINAIRE_OUTPUT_ROOT/cosmos_predict_v2p5/point_adapter/$RUN_NAME/checkpoints
bash /root/autodl-tmp/ckpt_pruner.sh "$CKPT_DIR" 3 300 &
PRUNER_PID=$!
trap 'kill $PRUNER_PID 2>/dev/null' EXIT INT TERM
echo "[$(date)] ckpt_pruner started pid=$PRUNER_PID keep=3 dir=$CKPT_DIR"

CMD=(
  torchrun --nproc_per_node=1 --master_port=29502
  scripts/train.py
  --config=cosmos_predict2/_src/predict2/configs/video2world/config.py
  "$@"
  --
  experiment=predict2_point_adapter_v5_controlnet_catmlp_121_stage2_lora
  job.wandb_mode=offline
)
echo "[$(date)] cmd: ${CMD[*]}"
"${CMD[@]}"
