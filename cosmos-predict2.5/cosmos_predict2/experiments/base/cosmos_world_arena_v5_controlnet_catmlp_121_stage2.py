# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# STAGE-2 of the catmlp (cross_attn_then_mlp) point-adapter model.
#
# Curriculum:
#   Stage-1 (cosmos_world_arena_v5_controlnet_catmlp_121.py): backbone FROZEN,
#           train ONLY the point-adapter (cross_attn_then_mlp + ControlNet copy).
#   Stage-2 (THIS file): load the stage-1 weights, FREEZE the point-adapter,
#           add LoRA to the BACKBONE second-half blocks (14..27) and train ONLY LoRA.
#
# Why: stage-1's frozen backbone never adapts to the RoboTwin domain -> slightly
# grainy frames vs the action model (which carries backbone LoRA). Stage-2 lets the
# backbone's later blocks (appearance/texture) adapt a little WITHOUT touching the
# conditioning path (adapter frozen; early inject blocks 5/11 stay frozen).
#
# Guard-rails baked in (small rank, low LR, short run) so LoRA polishes IMAGE
# QUALITY without diluting the trajectory advantage. Success = MUSIQ/Image-Quality
# up AND Trajectory Accuracy flat (gate on PC-swap + Trajectory before/after).
#
# Requires the use_lora patch to text2world_model_rectified_flow.py that gates the
# set_up_model freezing on config.use_lora and forwards lora_layers_to_transform /
# lora_layers_pattern into LoraConfig.
#
# Diffs vs stage-1 catmlp_121:
#   - checkpoint.load_path -> stage-1 catmlp iter6000 EMA weights (not pretrained default)
#   - model.config.use_lora=True, rank=16, alpha=16
#   - lora_layers_to_transform=[14..27] (2nd half), lora_layers_pattern="blocks"
#     -> LoRA ONLY on backbone blocks 14..27; point_adapter (self.point_adapter,
#        not under .blocks) is EXCLUDED automatically.
#   - single-GPU RTX Pro 6000 96GB: batch_size=2, lower LR, max_iter=2000

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.utils.checkpoint_db import get_checkpoint_path
from cosmos_predict2._src.predict2.callbacks.validation_draw_sample import ValidationDrawSample
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video import (
    VideoDataset,
    get_generic_dataloader,
    get_sampler,
)
from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey


DATASET_ROOT = "/root/autodl-tmp/cosmos_training_data_world_arena"
TRAIN_DATASET_DIR = f"{DATASET_ROOT}/training"
VAL_DATASET_DIR = f"{DATASET_ROOT}/test"

DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[ModelKey(post_trained=False)]

# Stage-1 catmlp iter6000 EMA weights (consolidated .pt, keys "net.*"), pulled onto
# this machine at /root/autodl-tmp/ckpt_stage1/. The checkpointer loads with
# strict=False and (under use_lora) strips "base_layer." / "base_model.model." so
# the frozen backbone + trained adapter load into the LoRA-injected model; LoRA A/B
# stay at init (B=0 -> zero initial contribution).
STAGE1_CKPT = "/root/autodl-tmp/ckpt_stage1/catmlp_50t_iter6000_model_ema_bf16.pt"

checkpoint_conf = dict(
    load_path=STAGE1_CKPT,
    load_from_object_store=dict(enabled=False),
    save_to_object_store=dict(enabled=False),
    save_iter=500,
)

defaults = [
    f"/experiment/{DEFAULT_CHECKPOINT.experiment}",
    {"override /conditioner": "pc_video_prediction_conditioner"},
    {"override /data_train": "mock"},
    {"override /data_val": "mock"},
    "_self_",
]

pc_encoder_config = dict(
    DATASET="robotwin",
    CONFIG="semseg-pt-v3m1-0-base",
    EXP_NAME="semseg-pt-v3m1-0-base-cosmos-pcenc",
    WEIGHT_NAME="model_last",
    v5_layer="dec_0",
    v5_feat_input="coord",
)

train_pc_conditioning_mode_probs = dict(full=1.0)

dataset_train = L(VideoDataset)(
    dataset_dir=TRAIN_DATASET_DIR,
    num_frames=121,
    video_size=(480, 640),
    pc_latent_source="precomputed",
    pc_latent_k=104,
    pc_latent_amp=True,
    pc_encoder_config=pc_encoder_config,
    pc_conditioning_mode_probs=train_pc_conditioning_mode_probs,
    pc_conditioning_prefix_frames=[1, 2],
)

dataset_val = L(VideoDataset)(
    dataset_dir=VAL_DATASET_DIR,
    num_frames=121,
    video_size=(480, 640),
    pc_latent_source="precomputed",
    pc_latent_k=104,
    pc_latent_amp=True,
    pc_encoder_config=pc_encoder_config,
    pc_conditioning_mode_probs=dict(full=1.0),
    pc_conditioning_prefix_frames=[1, 2],
)

dataloader_train = L(get_generic_dataloader)(
    dataset=dataset_train,
    sampler=L(get_sampler)(dataset=dataset_train),
    batch_size=2,    # single Pro6000 96GB
    drop_last=True,
    num_workers=8,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
)

dataloader_val = L(get_generic_dataloader)(
    dataset=dataset_val,
    sampler=L(get_sampler)(dataset=dataset_val),
    batch_size=2,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
)

# LoRA-appropriate LR (few params, frozen backbone). Kept conservative so LoRA
# polishes appearance without over-writing the conditioning path. Main knob to tune.
optimizer_conf = dict(
    lr=1.0e-4,
    weight_decay=0.0,
)

scheduler_conf = dict(
    f_max=[0.5],
    f_min=[0.2],
    warm_up_steps=[200],
    cycle_lengths=[100000],
)

trainer_conf = dict(
    run_validation=False,
    validation_iter=2000,
    logging_iter=25,
    max_iter=2000,    # short stage-2 polish; gate on Trajectory/PC-swap at 500/1000
    callbacks=dict(
        heart_beat=dict(save_s3=False),
        iter_speed=dict(hit_thres=200, save_s3=False),
        device_monitor=dict(save_s3=False),
        every_n_sample_reg=dict(every_n=1000, save_s3=False),
        every_n_sample_ema=dict(every_n=1000000, save_s3=False),
        wandb=dict(save_s3=False),
        wandb_10x=dict(save_s3=False),
        dataloader_speed=dict(save_s3=False),
        validation_draw_sample_reg=L(ValidationDrawSample)(
            n_samples=2,
            is_ema=False,
            save_s3=False,
            do_x0_prediction=True,
        ),
        validation_draw_sample_ema=L(ValidationDrawSample)(
            n_samples=2,
            is_ema=True,
            save_s3=False,
            do_x0_prediction=True,
        ),
    ),
)

model_conf = dict(
    config=dict(
        # --- stage-2 LoRA on backbone 2nd half ---
        use_lora=True,
        lora_rank=16,
        lora_alpha=16,
        lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
        # LoRA is injected into ALL backbone blocks (peft), but only blocks >= 14 are
        # left trainable (set_up_model). First-half LoRA stays at zero-init (B=0) -> no
        # effect. This is the robust 2nd-half restriction (peft layers_to_transform does
        # not match this model's nested blocks.N.self_attn.q_proj naming).
        lora_trainable_block_min=14,                     # backbone blocks 14..27 (2nd half of 28)
        # --- conditioning / adapter: identical to stage-1 (adapter is FROZEN in stage-2) ---
        min_num_conditional_frames=0,
        max_num_conditional_frames=2,
        conditional_frames_probs={0: 0.333, 1: 0.333, 2: 0.334},
        conditional_frame_timestep=-1.0,
        conditioning_strategy="frame_replace",
        denoise_replace_gt_frames=True,
        net=dict(
            point_adapter_d_a=None,
            point_adapter_num_adapter_blocks=4,
            point_adapter_block_depth=1,
            point_adapter_num_heads=None,
            point_adapter_inject_block_ids=[5, 11, 17, 23],
            point_adapter_inject_every_k=6,
            point_adapter_mlp_ratio=None,
            point_adapter_dropout=0.0,
            point_adapter_d_pc=512,
            point_adapter_use_layernorm=True,
            point_adapter_adapter_mode="cross_attn_then_mlp",
            point_adapter_controlnet_copy=True,
        ),
    ),
)

model_parallel_conf = dict(
    context_parallel_size=1,
)

EXPERIMENT_NAME = "predict2_point_adapter_v5_controlnet_catmlp_121_stage2_lora"

experiment_config = dict(
    defaults=defaults,
    job=dict(
        project="cosmos_predict_v2p5",
        group="point_adapter",
        name="v5_controlnet_catmlp_121_stage2_lora2ndhalf_480x640",
    ),
    dataloader_train=dataloader_train,
    dataloader_val=dataloader_val,
    checkpoint=checkpoint_conf,
    optimizer=optimizer_conf,
    scheduler=scheduler_conf,
    trainer=trainer_conf,
    model=model_conf,
    model_parallel=model_parallel_conf,
)

cs = ConfigStore.instance()
cs.store(
    group="experiment",
    package="_global_",
    name=EXPERIMENT_NAME,
    node=experiment_config,
)
