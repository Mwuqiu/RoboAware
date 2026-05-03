# GRPO Training Configuration for Cosmos Predict 2 with Point Adapter
#
# This config extends the SO100 Point Adapter experiment config with GRPO-specific
# hyperparameters. The base model (2B Cosmos) is fully frozen; only the Point
# Adapter (~7M params) is trained via RL.

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# GRPO-specific hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class CosmosGRPOConfig:
    # --- Group sampling ---
    group_size: int = 4
    """Number of video samples per conditioning (G). Group-relative advantage is
    computed within each group of G rollouts."""

    # --- Update loop ---
    num_update_epochs: int = 1
    """How many gradient update passes to run per rollout buffer."""

    max_iter: int = 5_000
    """Total number of RL iterations (each iteration = 1 rollout + num_update_epochs updates)."""

    # --- Advantage ---
    adv_clip: float = 10.0
    """Symmetric clip value for group-normalised advantages."""

    # --- KL regularisation (disabled by default; wire in later) ---
    kl_coef: float = 0.0
    """KL penalty coefficient against reference policy. 0.0 = disabled."""

    # --- Optimiser ---
    lr: float = 2 ** (-14.5)
    """Learning rate for Point Adapter parameters (matches SFT setting)."""

    weight_decay: float = 0.001

    grad_clip: float = 1.0
    """Gradient norm clip for Point Adapter update."""

    # --- Diffusion sampling (rollout) ---
    num_diffusion_steps: int = 35
    """Number of denoising steps when generating rollout videos."""

    guidance: float = 1.5
    """Classifier-free guidance scale during rollout."""

    shift: float = 5.0
    """Shift parameter for the rectified-flow scheduler."""

    # --- Checkpoint & logging ---
    save_every: int = 100
    """Save Point Adapter checkpoint every N iterations."""

    log_every: int = 10
    """Print training metrics every N iterations."""

    output_dir: str = os.path.join(os.path.dirname(__file__), "..", "outputs")
    """Directory to write checkpoints and logs."""

    # --- Model loading (set these paths before training) ---
    base_model_checkpoint: Optional[str] = (
        "/root/autodl-tmp/cosmos-output/cosmos_predict_v2p5/point_adapter/2b_cosmos_so100_point_adapter/checkpoints/iter_000006000/model_ema_bf16.pt"
    )
    """Path to the consolidated Cosmos 2B base model checkpoint (.pt).
    Example: 'checkpoints/Cosmos-Predict2-2B-Video2World/model.pt'
    Leave None to use the path resolved from the experiment config."""

    point_adapter_checkpoint: Optional[str] = None
    """Path to a previously-saved Point Adapter state dict (.pt) to resume from.
    Leave None to start from the adapter weights inside base_model_checkpoint."""

    # --- Dataloader (mirrors cosmos_so100_point_adapter settings) ---
    dataset_dir: str = "datasets/cosmos_so100_point"
    num_frames: int = 93
    video_size: tuple = field(default_factory=lambda: (480, 832))
    batch_size: int = 1           # keep small for G=4 on a single 80 G A100
    num_workers: int = 4

    # --- Experiment / Hydra config references ---------------------------------
    hydra_config_module: str = (
        "cosmos_predict2._src.predict2.configs.video2world.config"
    )
    hydra_experiment_override: str = (
        "predict2_point_adapter_training_2b_cosmos_so100_point"
    )

    # --- Online reward pipeline (SAM + Depth + ICP) ---------------------------
    sam2_checkpoint: str = "/root/autodl-tmp/sam2/checkpoints/sam2.1_hiera_large.pt"
    sam2_model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam_obj_id: int = 1

    depth_model_id: str = "depth-anything/DA3NESTED-GIANT-LARGE"
    depth_conf_percentile: float = 20.0

    online_max_frames: int = 16
    online_frame_stride: int = 4
    max_points_per_frame: int = 120000

    frame_pair_stride: int = 1
    frame_pair_k: float = 1.0
    frame_pair_offset: float = 0.0
    max_icp_pairs: int = 16

    icp_method: str = "p2p"
    icp_multiscale: bool = True
    icp_voxel_coarse: float = 0.02
    icp_voxel_medium: float = 0.015
    icp_voxel_fine: float = 0.01
    icp_corr_coarse: float = 0.20
    icp_corr_medium: float = 0.15
    icp_corr_fine: float = 0.10
    icp_iters_coarse: int = 60
    icp_iters_medium: int = 40
    icp_iters_fine: int = 30
    icp_voxel: float = 0.01
    icp_max_corr: float = 0.10
    icp_iters: int = 50
    icp_fitness_thresh: float = 0.3
    icp_rmse_thresh: float = 0.2

    icp_global_alpha: float = 10.0
    icp_global_top_k: Optional[int] = None
    icp_global_refine: bool = True
    icp_global_refine_max_frames: int = 30

    # --- Coarse SE(3) search (secondary initialization before per-frame ICP) ----
    icp_coarse_search: bool = False
    """Enable SE(3) grid search before each frame’s ICP to escape bad T_init."""
    icp_coarse_trans_range: float = 0.10
    """Translation search range in meters (±range around current T)."""
    icp_coarse_trans_step: float = 0.05
    """Translation step; 5 values/axis = 125 translation candidates."""
    icp_coarse_rot_range_deg: float = 30.0
    """Rotation search range in degrees."""
    icp_coarse_rot_step_deg: float = 15.0
    """Rotation step in degrees; 5 values = 5 rotation candidates for z-only."""
    icp_coarse_rot_axes: str = "z"
    """Axes to search: 'none' | 'z' | 'xyz'."""
    icp_coarse_top_k_icp: int = 10
    """Number of top NN-scored candidates to refine with quick ICP."""
    icp_coarse_fast_voxel: float = 0.03
    icp_coarse_fast_corr: float = 0.25
    icp_coarse_fast_iters: int = 10

    reward_weight_fitness: float = 1.0
    reward_weight_rmse: float = 1.0
    reward_bias: float = 0.0
    reward_weight_alignment: float = 1.0
    reward_weight_local_icp: float = 0.0
    frame_agg: str = "mean"
    outlier_thresh: float = 0.05
    failure_fallback_reward: float = 0.0

    # --- Debug artifacts ------------------------------------------------------
    debug_save_intermediates: bool = False
    """If True, save rollout/reward intermediate artifacts for debugging."""

    debug_save_every: int = 1
    """Save debug artifacts every N reward calls (usually every training iter)."""

    debug_max_videos_per_iter: int = 1
    """Max rollout videos to dump per iter to control storage usage."""

    debug_dir: Optional[str] = None
    """Debug output directory; defaults to <output_dir>/debug when None."""

    # --- Memory offloading ----------------------------------------------------
    offload_diffusion_model: bool = False
    """Offload diffusion net (model.net) to CPU between encode and decode steps to save VRAM."""

    offload_text_encoder: bool = False
    """Offload text encoder to CPU after computing embeddings."""

    offload_tokenizer: bool = False
    """Offload tokenizer encoder/decoder to CPU when not in use."""

    offload_model_for_reward: bool = False
    """Move the entire Cosmos model to CPU before running SAM+Depth+ICP reward computation,
    then restore it to GPU for the next rollout. Maximises available VRAM for SAM/Depth models."""

    # --- Weights & Biases logging ---------------------------------------------
    wandb_enabled: bool = True
    """Enable Weights & Biases experiment tracking."""

    wandb_project: str = "cosmos-grpo"
    """W&B project name."""

    wandb_run_name: str = ""
    """W&B run name. Empty string = auto-generated by W&B."""

    wandb_tags: List[str] = field(default_factory=list)
    """List of tags to attach to the W&B run."""

    wandb_notes: str = ""
    """Free-form notes attached to the W&B run."""

    wandb_log_every: int = 1
    """Log metrics to W&B every N training iterations."""

    wandb_watch_model: bool = False
    """Call wandb.watch() on Point Adapter params to log gradient/weight histograms.
    Expensive – disable for production runs."""

    wandb_log_media_every: int = 0
    """Log debug alignment images to W&B every N iterations (0 = disabled)."""
