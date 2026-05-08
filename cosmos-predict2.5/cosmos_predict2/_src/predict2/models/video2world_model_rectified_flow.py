# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum
from typing import Callable, Dict, Optional, Tuple

import attrs
import torch
import torch.nn.functional as F
from einops import rearrange
from megatron.core import parallel_state
from torch import Tensor

from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.configs.video2world.defaults.conditioner import Video2WorldCondition
from cosmos_predict2._src.predict2.models.denoise_prediction import DenoisePrediction
from cosmos_predict2._src.predict2.models.text2world_model_rectified_flow import (
    Text2WorldCondition,
    Text2WorldModelRectifiedFlow,
    Text2WorldModelRectifiedFlowConfig,
)

NUM_CONDITIONAL_FRAMES_KEY: str = "num_conditional_frames"


class ConditioningStrategy(str, Enum):
    FRAME_REPLACE = "frame_replace"  # First few frames of the video are replaced with the conditional frames

    def __str__(self) -> str:
        return self.value


@attrs.define(slots=False)
class Video2WorldModelRectifiedFlowConfig(Text2WorldModelRectifiedFlowConfig):
    min_num_conditional_frames: int = 1  # Minimum number of latent conditional frames
    max_num_conditional_frames: int = 2  # Maximum number of latent conditional frames
    conditional_frame_timestep: float = (
        -1.0
    )  # Noise level used for conditional frames; default is -1 which will not take effective
    conditioning_strategy: str = str(ConditioningStrategy.FRAME_REPLACE)  # What strategy to use for conditioning
    denoise_replace_gt_frames: bool = True  # Whether to denoise the ground truth frames
    conditional_frames_probs: Optional[Dict[int, float]] = None  # Probability distribution for conditional frames
    point_diffusion_loss_weight: float = 1.0
    point_condition_frames: int = 2
    point_latent_scale: float = 1.0

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        assert self.conditioning_strategy in [
            str(ConditioningStrategy.FRAME_REPLACE),
        ]


class Video2WorldModelRectifiedFlow(Text2WorldModelRectifiedFlow):
    def get_data_and_condition(
        self, data_batch: dict[str, torch.Tensor]
    ) -> Tuple[Tensor, Tensor, Video2WorldCondition]:
        # generate random number of conditional frames for training
        raw_state, latent_state, condition = super().get_data_and_condition(data_batch)
        condition = condition.set_video_condition(
            gt_frames=latent_state.to(**self.tensor_kwargs),
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=data_batch.get(NUM_CONDITIONAL_FRAMES_KEY, None),
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        return raw_state, latent_state, condition

    def _replace_condition_fields(self, condition: Text2WorldCondition, **updates) -> Text2WorldCondition:
        kwargs = condition.to_dict(skip_underscore=False)
        kwargs.update({key: value for key, value in updates.items() if key in kwargs})
        return type(condition)(**kwargs)

    def _point_latent_scale(self) -> float:
        return max(float(getattr(self.config, "point_latent_scale", 1.0)), 1.0e-6)

    def _align_point_temporal(self, point: torch.Tensor, target_t: int) -> torch.Tensor:
        if point is None or point.shape[1] == target_t:
            return point

        if point.ndim == 4:
            B, T, K, D = point.shape
            point_BK_D_T = rearrange(point, "b t k d -> (b k) d t")
            if T > target_t:
                point_BK_D_T = F.adaptive_avg_pool1d(point_BK_D_T, target_t)
            else:
                point_BK_D_T = F.interpolate(point_BK_D_T, size=target_t, mode="linear", align_corners=False)
            return rearrange(point_BK_D_T, "(b k) d t -> b t k d", b=B, k=K)

        if point.ndim == 3:
            B, T, K = point.shape
            is_bool = point.dtype == torch.bool
            point_BK_1_T = rearrange(point.float(), "b t k -> (b k) 1 t")
            point_BK_1_T = F.interpolate(point_BK_1_T, size=target_t, mode="nearest")
            out = rearrange(point_BK_1_T, "(b k) 1 t -> b t k", b=B, k=K)
            return out.bool() if is_bool else out

        raise ValueError(f"Unsupported point temporal tensor shape: {point.shape}")

    def _prepare_point_diffusion_condition(
        self,
        condition: Text2WorldCondition,
        sigmas_B_1: torch.Tensor,
        ref: torch.Tensor,
    ) -> tuple[Text2WorldCondition, Optional[dict[str, torch.Tensor]]]:
        pc_x0 = getattr(condition, "pc_latent_x0", None)
        if pc_x0 is None:
            return condition, None

        pc_x0 = pc_x0.to(device=ref.device, dtype=ref.dtype) / self._point_latent_scale()
        pc_noise = torch.randn_like(pc_x0)
        pc_sigmas_B_1 = sigmas_B_1.to(device=pc_x0.device, dtype=pc_x0.dtype)
        pc_xt, pc_vt = self.rectified_flow.get_interpolation(pc_noise, pc_x0, pc_sigmas_B_1)

        B, T_pc, K, _ = pc_x0.shape
        n_cond = min(max(int(self.config.point_condition_frames), 0), T_pc)
        pc_condition_mask = torch.zeros((B, T_pc, K), device=pc_x0.device, dtype=torch.bool)
        if n_cond > 0:
            pc_condition_mask[:, :n_cond, :] = True
            pc_xt = torch.where(pc_condition_mask[..., None], pc_x0, pc_xt)

        point_info = {
            "pc_vt": pc_vt,
            "pc_valid_mask": getattr(condition, "pc_latent_mask", None),
            "pc_condition_mask": pc_condition_mask,
        }
        condition = self._replace_condition_fields(
            condition,
            pc_latent_x0=None,
            pc_latent_xt=pc_xt,
            pc_condition_mask=pc_condition_mask,
        )
        return condition, point_info

    def _point_diffusion_loss(
        self,
        point_info: dict[str, torch.Tensor],
        pc_velocity_pred: torch.Tensor,
    ) -> torch.Tensor:
        target = self._align_point_temporal(point_info["pc_vt"], pc_velocity_pred.shape[1]).to(pc_velocity_pred)
        pc_loss = (target - pc_velocity_pred) ** 2

        valid_mask = point_info.get("pc_valid_mask")
        if valid_mask is not None:
            valid_mask = self._align_point_temporal(valid_mask.to(device=pc_velocity_pred.device).bool(), pc_velocity_pred.shape[1])
        else:
            valid_mask = torch.ones(pc_velocity_pred.shape[:3], device=pc_velocity_pred.device, dtype=torch.bool)

        condition_mask = point_info.get("pc_condition_mask")
        if condition_mask is not None:
            condition_mask = self._align_point_temporal(condition_mask.to(device=pc_velocity_pred.device).bool(), pc_velocity_pred.shape[1])
            valid_mask = valid_mask & (~condition_mask)

        valid = valid_mask[..., None].to(pc_loss.dtype)
        denom = valid.sum().clamp(min=1.0) * pc_loss.shape[-1]
        return (pc_loss * valid).sum() / denom

    def _point_sampling_condition_tensors(
        self,
        point_state: dict[str, torch.Tensor],
        target_t: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        pc_xt = point_state["pc_xt"]
        B, _, K, D = pc_xt.shape
        n_cond = min(int(point_state["n_cond"]), target_t)
        pc_condition_mask = torch.zeros((B, target_t, K), device=device, dtype=torch.bool)
        pc_condition_x0 = torch.zeros((B, target_t, K, D), device=device, dtype=dtype)

        pc_condition_prefix = point_state.get("pc_condition_prefix")
        if n_cond > 0 and pc_condition_prefix is not None:
            prefix = pc_condition_prefix.to(device=device, dtype=dtype)
            n_prefix = min(n_cond, prefix.shape[1])
            pc_condition_mask[:, :n_prefix, :] = True
            pc_condition_x0[:, :n_prefix, :, :] = prefix[:, :n_prefix, :, :]

        pc_valid_mask = point_state.get("pc_valid_mask_source")
        if pc_valid_mask is not None:
            pc_valid_mask = self._align_point_temporal(pc_valid_mask.to(device=device).bool(), target_t)

        return pc_condition_mask, pc_condition_x0, pc_valid_mask

    def _condition_from_point_sampling_state(
        self,
        condition: Text2WorldCondition,
        point_state: dict[str, torch.Tensor],
    ) -> Text2WorldCondition:
        pc_xt = point_state["pc_xt"]
        pc_condition_mask, pc_condition_x0, pc_valid_mask = self._point_sampling_condition_tensors(
            point_state,
            pc_xt.shape[1],
            pc_xt.device,
            pc_xt.dtype,
        )
        pc_xt = torch.where(pc_condition_mask[..., None], pc_condition_x0, pc_xt)
        point_state["pc_xt"] = pc_xt
        point_state["pc_condition_mask"] = pc_condition_mask
        point_state["pc_valid_mask"] = pc_valid_mask

        return self._replace_condition_fields(
            condition,
            pc_latent_x0=None,
            pc_latent_xt=pc_xt,
            pc_latent_mask=pc_valid_mask,
            pc_condition_mask=pc_condition_mask,
        )

    def _prepare_point_sampling_condition(
        self,
        condition: Text2WorldCondition,
        ref: torch.Tensor,
    ) -> tuple[Text2WorldCondition, Optional[dict[str, torch.Tensor]]]:
        pc_x0 = getattr(condition, "pc_latent_x0", None)
        if pc_x0 is None:
            condition = self._replace_condition_fields(
                condition,
                pc_latent_x0=None,
                pc_latent_xt=None,
                pc_condition_mask=None,
            )
            return condition, None

        pc_x0 = pc_x0.to(device=ref.device, dtype=ref.dtype) / self._point_latent_scale()
        B, T_pc, K, _ = pc_x0.shape
        n_cond = min(max(int(self.config.point_condition_frames), 0), T_pc)

        pc_xt = torch.randn_like(pc_x0)
        pc_condition_prefix = None
        if n_cond > 0:
            pc_condition_prefix = pc_x0[:, :n_cond].clone()
            pc_xt[:, :n_cond] = pc_condition_prefix

        pc_valid_mask = getattr(condition, "pc_latent_mask", None)
        if pc_valid_mask is not None:
            pc_valid_mask = pc_valid_mask.to(device=ref.device).bool()

        point_state = {
            "pc_xt": pc_xt,
            "n_cond": n_cond,
            "pc_valid_mask_source": pc_valid_mask,
            "pc_condition_prefix": pc_condition_prefix,
        }
        condition = self._condition_from_point_sampling_state(condition, point_state)
        return condition, point_state

    def _pc_x0_from_velocity(
        self,
        pc_xt: torch.Tensor,
        pc_velocity: torch.Tensor,
        timesteps_B_T: torch.Tensor,
    ) -> torch.Tensor:
        pc_xt = self._align_point_temporal(pc_xt.to(device=pc_velocity.device, dtype=pc_velocity.dtype), pc_velocity.shape[1])
        sigmas = (timesteps_B_T.float() / self.rectified_flow.num_train_timesteps).to(
            device=pc_velocity.device, dtype=pc_velocity.dtype
        )
        sigmas = sigmas.mean(dim=1, keepdim=True)
        if sigmas.shape[0] == 1 and pc_velocity.shape[0] > 1:
            sigmas = sigmas.expand(pc_velocity.shape[0], -1)
        sigmas = sigmas[: pc_velocity.shape[0]]
        return pc_xt - rearrange(sigmas, "b t -> b t 1 1") * pc_velocity

    def _update_point_sampling_state(
        self,
        point_state: Optional[dict[str, torch.Tensor]],
        pc_velocity_pred: Optional[torch.Tensor],
        timesteps_B_T: torch.Tensor,
    ) -> None:
        if point_state is None or pc_velocity_pred is None:
            return

        pc_x0_pred = self._pc_x0_from_velocity(point_state["pc_xt"], pc_velocity_pred.detach(), timesteps_B_T)
        point_state["pc_xt"] = pc_x0_pred
        pc_condition_mask, pc_condition_x0, pc_valid_mask = self._point_sampling_condition_tensors(
            point_state,
            pc_x0_pred.shape[1],
            pc_x0_pred.device,
            pc_x0_pred.dtype,
        )
        point_state["pc_xt"] = torch.where(pc_condition_mask[..., None], pc_condition_x0, pc_x0_pred)
        point_state["pc_condition_mask"] = pc_condition_mask
        point_state["pc_valid_mask"] = pc_valid_mask

    def denoise(
        self,
        noise: torch.Tensor,
        xt_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        condition: Text2WorldCondition,
        return_point_pred: bool = False,
    ) -> torch.Tensor | DenoisePrediction:
        """
        Args:
            xt (torch.Tensor): The input noise data.
            sigma (torch.Tensor): The noise level.
            condition (Text2WorldCondition): conditional information, generated from self.conditioner

        Returns:
            velocity prediction
        """
        if condition.is_video:
            condition_state_in_B_C_T_H_W = condition.gt_frames.type_as(xt_B_C_T_H_W)
            if not condition.use_video_condition:
                # When using random dropout, we zero out the ground truth frames
                condition_state_in_B_C_T_H_W = condition_state_in_B_C_T_H_W * 0

            _, C, _, _, _ = xt_B_C_T_H_W.shape
            condition_video_mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(1, C, 1, 1, 1).type_as(
                xt_B_C_T_H_W
            )

            # Make the first few frames of x_t be the ground truth frames
            xt_B_C_T_H_W = condition_state_in_B_C_T_H_W * condition_video_mask + xt_B_C_T_H_W * (
                1 - condition_video_mask
            )

            if self.config.conditional_frame_timestep >= 0:
                condition_video_mask_B_1_T_1_1 = condition_video_mask.mean(dim=[1, 3, 4], keepdim=True)
                timestep_cond_B_1_T_1_1 = (
                    torch.ones_like(condition_video_mask_B_1_T_1_1) * self.config.conditional_frame_timestep
                )

                timesteps_B_1_T_1_1 = timestep_cond_B_1_T_1_1 * condition_video_mask_B_1_T_1_1 + timesteps_B_T * (
                    1 - condition_video_mask_B_1_T_1_1
                )

                timesteps_B_T = timesteps_B_1_T_1_1.squeeze()
                timesteps_B_T = (
                    timesteps_B_T.unsqueeze(0) if timesteps_B_T.ndim == 1 else timesteps_B_T
                )  # add dimension for batch

        # forward pass through the network
        net_output = self.net(
            x_B_C_T_H_W=xt_B_C_T_H_W.to(**self.tensor_kwargs),  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            timesteps_B_T=timesteps_B_T,  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            **condition.to_dict(),
        )
        pc_velocity_pred = None
        if isinstance(net_output, tuple):
            net_output_B_C_T_H_W, pc_velocity_pred = net_output
        else:
            net_output_B_C_T_H_W = net_output
        net_output_B_C_T_H_W = net_output_B_C_T_H_W.float()
        if pc_velocity_pred is not None:
            pc_velocity_pred = pc_velocity_pred.float()

        if condition.is_video and self.config.denoise_replace_gt_frames:
            gt_frames_x0 = condition.gt_frames.type_as(net_output_B_C_T_H_W)
            gt_frames_velocity = noise - gt_frames_x0
            net_output_B_C_T_H_W = gt_frames_velocity * condition_video_mask + net_output_B_C_T_H_W * (
                1 - condition_video_mask
            )

        if return_point_pred:
            pc_x0_pred = None
            pc_xt = getattr(condition, "pc_latent_xt", None)
            if pc_velocity_pred is not None and pc_xt is not None:
                pc_x0_pred = self._pc_x0_from_velocity(pc_xt, pc_velocity_pred, timesteps_B_T)
            return DenoisePrediction(velocity=net_output_B_C_T_H_W, pc_x0=pc_x0_pred, pc_velocity=pc_velocity_pred)

        return net_output_B_C_T_H_W

    def forward(self, data_batch: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        # Obtain text embeddings online
        if self.config.text_encoder_config is not None and self.config.text_encoder_config.compute_online:
            text_embeddings = self.text_encoder.compute_text_embeddings_online(data_batch, self.input_caption_key)
            data_batch["t5_text_embeddings"] = text_embeddings
            data_batch["t5_text_mask"] = torch.ones(text_embeddings.shape[0], text_embeddings.shape[1], device="cuda")

        _, x0_B_C_T_H_W, condition = self.get_data_and_condition(data_batch)

        epsilon_B_C_T_H_W = torch.randn(x0_B_C_T_H_W.size(), **self.tensor_kwargs_fp32)
        batch_size = x0_B_C_T_H_W.size()[0]
        t_B = self.rectified_flow.sample_train_time(batch_size).to(**self.tensor_kwargs_fp32)
        t_B = rearrange(t_B, "b -> b 1")

        x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, t_B = self.broadcast_split_for_model_parallelsim(
            x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, t_B
        )
        timesteps = self.rectified_flow.get_discrete_timestamp(t_B, self.tensor_kwargs_fp32)

        if self.config.use_high_sigma_strategy:
            raise NotImplementedError("High sigma strategy is buggy when using CP")

        sigmas = self.rectified_flow.get_sigmas(
            timesteps,
            self.tensor_kwargs_fp32,
        )

        timesteps = rearrange(timesteps, "b -> b 1")
        sigmas = rearrange(sigmas, "b -> b 1")
        xt_B_C_T_H_W, vt_B_C_T_H_W = self.rectified_flow.get_interpolation(
            epsilon_B_C_T_H_W, x0_B_C_T_H_W, sigmas
        )

        condition, point_info = self._prepare_point_diffusion_condition(condition, sigmas, x0_B_C_T_H_W)
        denoise_pred = self.denoise(
            noise=epsilon_B_C_T_H_W,
            xt_B_C_T_H_W=xt_B_C_T_H_W.to(**self.tensor_kwargs),
            timesteps_B_T=timesteps,
            condition=condition,
            return_point_pred=True,
        )
        vt_pred_B_C_T_H_W = denoise_pred.velocity

        time_weights_B = self.rectified_flow.train_time_weight(timesteps, self.tensor_kwargs_fp32)
        per_instance_loss = torch.mean(
            (vt_pred_B_C_T_H_W - vt_B_C_T_H_W) ** 2, dim=list(range(1, vt_pred_B_C_T_H_W.dim()))
        )

        video_loss = torch.mean(time_weights_B * per_instance_loss)
        loss = video_loss
        output_batch = {
            "x0": x0_B_C_T_H_W,
            "xt": xt_B_C_T_H_W,
            "sigma": sigmas,
            "condition": condition,
            "model_pred": vt_pred_B_C_T_H_W,
            "edm_loss": video_loss,
            "video_loss_unscaled": video_loss,
            "video_loss_scaled": video_loss,
        }

        if point_info is not None and denoise_pred.pc_velocity is not None:
            pc_loss = self._point_diffusion_loss(point_info, denoise_pred.pc_velocity)
            pc_weighted_loss = pc_loss * float(self.config.point_diffusion_loss_weight)
            output_batch["pc_diffusion_loss"] = pc_loss
            output_batch["pc_loss_weighted"] = pc_weighted_loss
            output_batch["pc_to_video_loss_ratio"] = pc_weighted_loss.detach() / video_loss.detach().clamp(min=1.0e-8)
            output_batch["aux_loss"] = pc_weighted_loss
            loss = loss + pc_weighted_loss

        return output_batch, loss

    def get_velocity_fn_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
    ) -> Callable:
        """
        Generates a callable function `x0_fn` based on the provided data batch and guidance factor.

        This function first processes the input data batch through a conditioning workflow (`conditioner`) to obtain conditioned and unconditioned states. It then defines a nested function `x0_fn` which applies a denoising operation on an input `noise_x` at a given noise level `sigma` using both the conditioned and unconditioned states.

        Args:
        - data_batch (Dict): A batch of data used for conditioning. The format and content of this dictionary should align with the expectations of the `self.conditioner`
        - guidance (float, optional): A scalar value that modulates the influence of the conditioned state relative to the unconditioned state in the output. Defaults to 1.5.
        - is_negative_prompt (bool): use negative prompt t5 in uncondition if true

        Returns:
        - Callable: A function `x0_fn(noise_x, sigma)` that takes two arguments, `noise_x` and `sigma`, and return velocity predictoin

        The returned function is suitable for use in scenarios where a denoised state is required based on both conditioned and unconditioned inputs, with an adjustable level of guidance influence.
        """

        if NUM_CONDITIONAL_FRAMES_KEY in data_batch:
            num_conditional_frames = data_batch[NUM_CONDITIONAL_FRAMES_KEY]
        else:
            num_conditional_frames = 1

        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        is_image_batch = self.is_image_batch(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        uncondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        _, x0, _ = self.get_data_and_condition(data_batch)
        # override condition with inference mode; num_conditional_frames used Here!
        condition = condition.set_video_condition(
            gt_frames=x0,
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        uncondition = uncondition.set_video_condition(
            gt_frames=x0,
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        condition = condition.edit_for_inference(is_cfg_conditional=True, num_conditional_frames=num_conditional_frames)
        uncondition = uncondition.edit_for_inference(
            is_cfg_conditional=False, num_conditional_frames=num_conditional_frames
        )

        _, condition, _, _ = self.broadcast_split_for_model_parallelsim(x0, condition, None, None)
        _, uncondition, _, _ = self.broadcast_split_for_model_parallelsim(x0, uncondition, None, None)

        condition, point_state = self._prepare_point_sampling_condition(condition, x0)
        uncondition = self._replace_condition_fields(
            uncondition,
            pc_latent_x0=None,
            pc_latent_xt=None,
            pc_latent_mask=None,
            pc_condition_mask=None,
        )

        if parallel_state.is_initialized():
            pass
        else:
            assert not self.net.is_context_parallel_enabled, (
                "parallel_state is not initialized, context parallel should be turned off."
            )

        def velocity_fn(noise: torch.Tensor, noise_x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            nonlocal point_state
            condition_for_step = (
                self._condition_from_point_sampling_state(condition, point_state)
                if point_state is not None else condition
            )
            cond_pred = self.denoise(noise, noise_x, timestep, condition_for_step, return_point_pred=True)
            uncond_v = self.denoise(noise, noise_x, timestep, uncondition)
            self._update_point_sampling_state(point_state, cond_pred.pc_velocity, timestep)
            cond_v = cond_pred.velocity
            velocity_pred = cond_v + guidance * (cond_v - uncond_v)
            return velocity_pred

        return velocity_fn
