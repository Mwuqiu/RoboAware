# SPDX-License-Identifier: Apache-2.0
#
# Action-conditioned training on WorldArena RoboTwin 10-task subset.
# Parallels the video-only LoRA baseline (same data/frames/res) but injects the
# 14-dim dual-arm relative-joint action via the action-conditioned net (AdaLN).
# v1: full FSDP fine-tune (action embedders are new params), null t5 text.
from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_action_video import ActionVideoDataset
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video import (
    get_generic_dataloader,
    get_sampler,
)
from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey

DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[ModelKey(post_trained=False)]

DATASET_ROOT = "/root/autodl-tmp/cosmos_training_data_world_arena_10tasks"
NUM_FRAMES = 121
NUM_ACTION_PER_CHUNK = NUM_FRAMES - 1  # 120; net flattens action[T-1,14] -> global AdaLN emb
ACTION_DIM = 14

dataset_train = L(ActionVideoDataset)(
    dataset_dir=f"{DATASET_ROOT}/training", num_frames=NUM_FRAMES, video_size=(480, 640),
    caption_format="text", action_dim=ACTION_DIM,
)
dataloader_train = L(get_generic_dataloader)(
    dataset=dataset_train, sampler=L(get_sampler)(dataset=dataset_train),
    batch_size=1, drop_last=True, num_workers=8, pin_memory=True,
)
dataset_val = L(ActionVideoDataset)(
    dataset_dir=f"{DATASET_ROOT}/test", num_frames=NUM_FRAMES, video_size=(480, 640),
    caption_format="text", action_dim=ACTION_DIM,
)
dataloader_val = L(get_generic_dataloader)(
    dataset=dataset_val, sampler=L(get_sampler)(dataset=dataset_val),
    batch_size=1, drop_last=True, num_workers=4, pin_memory=True,
)

# Local .pt base (NOT s3.uri DCP dir -> would silently load nothing -> loss ~3.0)
BASE_CKPT = (
    "/root/autodl-tmp/cache/hub/models--nvidia--Cosmos-Predict2.5-2B/snapshots/"
    "15a82a2ec231bc318692aa0456a36537c806e7d4/base/pre-trained/"
    "d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt"
)

EXPERIMENT_NAME = "predict2_action_2b_world_arena_10tasks"

experiment_config = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT.experiment}",
        {"override /model": "action_conditioned_video2world_fsdp_rectified_flow"},
        {"override /net": "cosmos_v1_2B_action_chunk_conditioned"},
        {"override /conditioner": "action_conditioned_video_conditioner"},
        {"override /data_train": "mock"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    job=dict(project="cosmos_predict_v2p5", group="action", name="action_10tasks_480x640"),
    dataloader_train=dataloader_train,
    dataloader_val=dataloader_val,
    checkpoint=dict(
        load_path=BASE_CKPT, load_training_state=False, strict_resume=False,
        load_ema_to_reg=False,
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
        save_iter=500,
    ),
    optimizer=dict(lr=2 ** (-14.5), weight_decay=0.1),
    scheduler=dict(f_max=[0.5], f_min=[0.2], warm_up_steps=[500], cycle_lengths=[100000]),
    trainer=dict(
        run_validation=False, logging_iter=50, max_iter=10000,
        callbacks=dict(
            heart_beat=dict(save_s3=False), iter_speed=dict(save_s3=False),
            device_monitor=dict(save_s3=False), wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
            every_n_sample_reg=dict(every_n=1000, save_s3=False),
            every_n_sample_ema=dict(every_n=1000000, save_s3=False),
        ),
    ),
    model=dict(config=dict(
        min_num_conditional_frames=1, max_num_conditional_frames=1,
        conditional_frames_probs=None,
        state_t=1 + NUM_ACTION_PER_CHUNK // 4,
        net=dict(action_dim=ACTION_DIM, num_action_per_chunk=NUM_ACTION_PER_CHUNK),
    )),
    model_parallel=dict(context_parallel_size=1),
)

cs = ConfigStore.instance()
cs.store(group="experiment", package="_global_", name=EXPERIMENT_NAME, node=experiment_config)
