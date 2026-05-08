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

import math
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple

import attrs
import torch
import torch.nn.functional as F
from einops import rearrange
from megatron.core import parallel_state
from torch import Tensor

from cosmos_predict2._src.imaginaire.utils.high_sigma_strategy import HighSigmaStrategy as HighSigmaStrategy
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.configs.video2world.defaults.conditioner import Video2WorldCondition
from cosmos_predict2._src.predict2.models.text2world_model import (
    DenoisePrediction,
    Text2WorldCondition,
    Text2WorldModelConfig,
)
from cosmos_predict2._src.predict2.models.text2world_model import DiffusionModel as Text2WorldModel

NUM_CONDITIONAL_FRAMES_KEY: str = "num_conditional_frames"


class ConditioningStrategy(str, Enum):
    FRAME_REPLACE = "frame_replace"  # First few frames of the video are replaced with the conditional frames

    def __str__(self) -> str:
        return self.value


@attrs.define(slots=False)
class Video2WorldConfig(Text2WorldModelConfig):
    min_num_conditional_frames: int = 1  # Minimum number of latent conditional frames
    max_num_conditional_frames: int = 2  # Maximum number of latent conditional frames
    sigma_conditional: float = 0.0001  # Noise level used for conditional frames
    conditioning_strategy: str = str(ConditioningStrategy.FRAME_REPLACE)  # What strategy to use for conditioning
    denoise_replace_gt_frames: bool = True  # Whether to denoise the ground truth frames
    high_sigma_strategy: str = str(HighSigmaStrategy.UNIFORM80_2000)  # What strategy to use for high sigma
    high_sigma_ratio: float = 0.05  # Ratio of high sigma frames
    low_sigma_ratio: float = 0.05  # Ratio of low sigma frames
    conditional_frames_probs: Optional[Dict[int, float]] = None  # Probability distribution for conditional frames
    point_diffusion_loss_weight: float = 1.0
    point_condition_frames: int = 2

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        assert self.conditioning_strategy in [
            str(ConditioningStrategy.FRAME_REPLACE),
        ]
        assert self.high_sigma_strategy in [
            str(HighSigmaStrategy.NONE),
            str(HighSigmaStrategy.UNIFORM80_2000),
            str(HighSigmaStrategy.LOGUNIFORM200_100000),
            str(HighSigmaStrategy.BALANCED_TWO_HEADS_V1),
            str(HighSigmaStrategy.SHIFT24),
            str(HighSigmaStrategy.HARDCODED_20steps),
        ]


LOG_200 = math.log(200)
LOG_100000 = math.log(100000)


class Video2WorldModel(Text2WorldModel):
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

    def draw_training_sigma_and_epsilon(self, x0_size: int, condition: Any) -> torch.Tensor:
        sigma_B_1, epsilon = super().draw_training_sigma_and_epsilon(x0_size, condition)
        is_video_batch = condition.data_type == DataType.VIDEO
        # if is_video_batch, with 5% ratio, we regenerate sigma_B_1 with uniformally from 80 to 2000
        # with remaining 95% ratio, we keep the original sigma_B_1
        if is_video_batch:
            if self.config.high_sigma_strategy == str(HighSigmaStrategy.UNIFORM80_2000):
                mask = torch.rand(sigma_B_1.shape, device=sigma_B_1.device) < self.config.high_sigma_ratio
                new_sigma = torch.rand(sigma_B_1.shape, device=sigma_B_1.device).type_as(sigma_B_1) * 1920 + 80
                sigma_B_1 = torch.where(mask, new_sigma, sigma_B_1)
            elif self.config.high_sigma_strategy == str(HighSigmaStrategy.LOGUNIFORM200_100000):
                mask = torch.rand(sigma_B_1.shape, device=sigma_B_1.device) < self.config.high_sigma_ratio
                log_new_sigma = (
                    torch.rand(sigma_B_1.shape, device=sigma_B_1.device).type_as(sigma_B_1) * (LOG_100000 - LOG_200)
                    + LOG_200
                )
                sigma_B_1 = torch.where(mask, log_new_sigma.exp(), sigma_B_1)
            elif self.config.high_sigma_strategy == str(HighSigmaStrategy.SHIFT24):
                # sample t from uniform distribution between 0 and 1, with same shape as sigma_B_1
                _t = torch.rand(sigma_B_1.shape, device=sigma_B_1.device).double()
                _t = 24 * _t / (24 * _t + 1 - _t)
                sigma_B_1 = (_t / (1.0 - _t)).float()

                mask = torch.rand(sigma_B_1.shape, device=sigma_B_1.device) < self.config.high_sigma_ratio
                new_sigma = torch.rand(sigma_B_1.shape, device=sigma_B_1.device).type_as(sigma_B_1) * 1920 + 80
                sigma_B_1 = torch.where(mask, new_sigma, sigma_B_1)
            elif self.config.high_sigma_strategy == str(HighSigmaStrategy.BALANCED_TWO_HEADS_V1):
                # replace high sigma parts
                mask = torch.rand(sigma_B_1.shape, device=sigma_B_1.device) < self.config.high_sigma_ratio
                log_new_sigma = (
                    torch.rand(sigma_B_1.shape, device=sigma_B_1.device).type_as(sigma_B_1) * (LOG_100000 - LOG_200)
                    + LOG_200
                )
                sigma_B_1 = torch.where(mask, log_new_sigma.exp(), sigma_B_1)
                # replace low sigma parts
                mask = torch.rand(sigma_B_1.shape, device=sigma_B_1.device) < self.config.low_sigma_ratio
                low_sigma_B_1 = torch.rand(sigma_B_1.shape, device=sigma_B_1.device).type_as(sigma_B_1) * 2.0 + 0.00001
                sigma_B_1 = torch.where(mask, low_sigma_B_1, sigma_B_1)
            elif self.config.high_sigma_strategy == str(HighSigmaStrategy.HARDCODED_20steps):
                if not hasattr(self, "hardcoded_20steps_sigma"):
                    from cosmos_predict2._src.imaginaire.modules.res_sampler import get_rev_ts

                    hardcoded_20steps_sigma = get_rev_ts(
                        t_min=self.sde.sigma_min, t_max=self.sde.sigma_max, num_steps=20, ts_order=7.0
                    )
                    # add extra 100000 to the beginning
                    self.hardcoded_20steps_sigma = torch.cat(
                        [torch.tensor([100000.0], device=hardcoded_20steps_sigma.device), hardcoded_20steps_sigma],
                        dim=0,
                    )
                sigma_B_1 = self.hardcoded_20steps_sigma[
                    torch.randint(0, len(self.hardcoded_20steps_sigma), sigma_B_1.shape)
                ].type_as(sigma_B_1)
            elif self.config.high_sigma_strategy == str(HighSigmaStrategy.NONE):
                pass
            else:
                raise ValueError(f"High sigma strategy {self.config.high_sigma_strategy} is not supported")
        return sigma_B_1, epsilon

    def _denoise_prediction_from_flow_t(
        self, noise_x_in_t_space: torch.Tensor, t_B_T: torch.Tensor, condition: Text2WorldCondition
    ) -> DenoisePrediction:
        """
        This function is used when self.config.use_flowunipc_scheduler is set.
        """
        if t_B_T.ndim == 1:
            t_B_T = rearrange(t_B_T, "b -> b 1")
        elif t_B_T.ndim == 2:
            t_B_T = t_B_T
        else:
            raise ValueError(f"t_B_T shape {t_B_T.shape} is not supported")
        # our model expects input of sigma and x_sigma, so convert t -> sigma, x_t to x_sigma
        sigma_B_T = t_B_T / (1.0 - t_B_T)
        x_B_C_T_H_W_in_sigma_space = noise_x_in_t_space * (1.0 + rearrange(sigma_B_T, "b t -> b 1 t 1 1"))
        return self.denoise(x_B_C_T_H_W_in_sigma_space, sigma_B_T, condition)

    def denoise_with_velocity(
        self, noise_x_in_t_space: torch.Tensor, t_B_T: torch.Tensor, condition: Text2WorldCondition
    ) -> torch.Tensor:
        denoise_output = self._denoise_prediction_from_flow_t(noise_x_in_t_space, t_B_T, condition)
        return denoise_output.eps - denoise_output.x0


    def _replace_condition_fields(self, condition: Text2WorldCondition, **updates) -> Text2WorldCondition:
        kwargs = condition.to_dict(skip_underscore=False)
        kwargs.update({key: value for key, value in updates.items() if key in kwargs})
        return type(condition)(**kwargs)

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
        sigma_B_T: torch.Tensor,
        ref: torch.Tensor,
    ) -> tuple[Text2WorldCondition, Optional[dict[str, torch.Tensor]]]:
        pc_x0 = getattr(condition, "pc_latent_x0", None)
        if pc_x0 is None:
            return condition, None

        pc_x0 = pc_x0.to(device=ref.device, dtype=ref.dtype)
        pc_sigma_B_1 = sigma_B_T.mean(dim=1, keepdim=True).to(device=pc_x0.device, dtype=pc_x0.dtype)
        pc_epsilon = torch.randn_like(pc_x0)
        pc_xt = pc_x0 + pc_epsilon * rearrange(pc_sigma_B_1, "b t -> b t 1 1")

        B, T_pc, K, _ = pc_x0.shape
        n_cond = min(max(int(self.config.point_condition_frames), 0), T_pc)
        pc_condition_mask = torch.zeros((B, T_pc, K), device=pc_x0.device, dtype=torch.bool)
        if n_cond > 0:
            pc_condition_mask[:, :n_cond, :] = True
            pc_xt = torch.where(pc_condition_mask[..., None], pc_x0, pc_xt)

        point_info = {
            "pc_x0": pc_x0,
            "pc_epsilon": pc_epsilon,
            "pc_sigma_B_1": pc_sigma_B_1,
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

        pc_x0 = pc_x0.to(device=ref.device, dtype=ref.dtype)
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

    def _update_point_sampling_state(
        self,
        point_state: Optional[dict[str, torch.Tensor]],
        pc_x0_pred: Optional[torch.Tensor],
    ) -> None:
        if point_state is None or pc_x0_pred is None:
            return

        pc_xt = pc_x0_pred.detach()
        point_state["pc_xt"] = pc_xt
        pc_condition_mask, pc_condition_x0, pc_valid_mask = self._point_sampling_condition_tensors(
            point_state,
            pc_xt.shape[1],
            pc_xt.device,
            pc_xt.dtype,
        )
        point_state["pc_xt"] = torch.where(pc_condition_mask[..., None], pc_condition_x0, pc_xt)
        point_state["pc_condition_mask"] = pc_condition_mask
        point_state["pc_valid_mask"] = pc_valid_mask

    def _point_diffusion_loss(
        self,
        point_info: dict[str, torch.Tensor],
        pc_x0_pred: torch.Tensor,
    ) -> torch.Tensor:
        target = self._align_point_temporal(point_info["pc_x0"], pc_x0_pred.shape[1]).to(pc_x0_pred)
        pc_loss = (target - pc_x0_pred) ** 2

        valid_mask = point_info.get("pc_valid_mask")
        if valid_mask is not None:
            valid_mask = self._align_point_temporal(valid_mask.to(device=pc_x0_pred.device).bool(), pc_x0_pred.shape[1])
        else:
            valid_mask = torch.ones(pc_x0_pred.shape[:3], device=pc_x0_pred.device, dtype=torch.bool)

        condition_mask = point_info.get("pc_condition_mask")
        if condition_mask is not None:
            condition_mask = self._align_point_temporal(condition_mask.to(device=pc_x0_pred.device).bool(), pc_x0_pred.shape[1])
            valid_mask = valid_mask & (~condition_mask)

        valid = valid_mask[..., None].to(pc_loss.dtype)
        denom = valid.sum().clamp(min=1.0) * pc_loss.shape[-1]
        return (pc_loss * valid).sum() / denom

    def denoise(
        self, xt_B_C_T_H_W: torch.Tensor, sigma: torch.Tensor, condition: Text2WorldCondition
    ) -> DenoisePrediction:
        """
        Performs denoising on the input noise data, noise level, and condition

        Args:
            xt (torch.Tensor): The input noise data.
            sigma (torch.Tensor): The noise level.
            condition (Text2WorldCondition): conditional information, generated from self.conditioner

        Returns:
            DenoisePrediction: The denoised prediction, it includes clean data predicton (x0), \
                noise prediction (eps_pred).
        """

        if sigma.ndim == 1:
            sigma_B_T = rearrange(sigma, "b -> b 1")
        elif sigma.ndim == 2:
            sigma_B_T = sigma
        else:
            raise ValueError(f"sigma shape {sigma.shape} is not supported")

        sigma_B_1_T_1_1 = rearrange(sigma_B_T, "b t -> b 1 t 1 1")
        # get precondition for the network
        c_skip_B_1_T_1_1, c_out_B_1_T_1_1, c_in_B_1_T_1_1, c_noise_B_1_T_1_1 = self.scaling(sigma=sigma_B_1_T_1_1)

        net_state_in_B_C_T_H_W = xt_B_C_T_H_W * c_in_B_1_T_1_1

        if condition.is_video:
            condition_state_in_B_C_T_H_W = condition.gt_frames.type_as(net_state_in_B_C_T_H_W) / self.config.sigma_data
            if not condition.use_video_condition:
                # When using random dropout, we zero out the ground truth frames
                condition_state_in_B_C_T_H_W = condition_state_in_B_C_T_H_W * 0

            _, C, _, _, _ = xt_B_C_T_H_W.shape
            condition_video_mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(1, C, 1, 1, 1).type_as(
                net_state_in_B_C_T_H_W
            )

            # Replace the first few frames of the video with the conditional frames
            # Update the c_noise as the conditional frames are clean and have very low noise

            # Make the first few frames of x_t be the ground truth frames
            net_state_in_B_C_T_H_W = condition_state_in_B_C_T_H_W * condition_video_mask + net_state_in_B_C_T_H_W * (
                1 - condition_video_mask
            )
            # Adjust c_noise for the conditional frames
            sigma_cond_B_1_T_1_1 = torch.ones_like(sigma_B_1_T_1_1) * self.config.sigma_conditional
            _, _, _, c_noise_cond_B_1_T_1_1 = self.scaling(sigma=sigma_cond_B_1_T_1_1)
            condition_video_mask_B_1_T_1_1 = condition_video_mask.mean(dim=[1, 3, 4], keepdim=True)
            c_noise_B_1_T_1_1 = c_noise_cond_B_1_T_1_1 * condition_video_mask_B_1_T_1_1 + c_noise_B_1_T_1_1 * (
                1 - condition_video_mask_B_1_T_1_1
            )

        # forward pass through the network
        net_output = self.net(
            x_B_C_T_H_W=net_state_in_B_C_T_H_W.to(
                **self.tensor_kwargs
            ),  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            timesteps_B_T=c_noise_B_1_T_1_1.squeeze(dim=[1, 3, 4]).to(
                **{
                    **self.tensor_kwargs,
                    "dtype": torch.float32 if self.config.use_wan_fp32_strategy else self.tensor_kwargs["dtype"],
                },
            ),  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            **condition.to_dict(),
        )
        pc_net_output = None
        if isinstance(net_output, tuple):
            net_output_B_C_T_H_W, pc_net_output = net_output
        else:
            net_output_B_C_T_H_W = net_output
        net_output_B_C_T_H_W = net_output_B_C_T_H_W.float()

        x0_pred_B_C_T_H_W = c_skip_B_1_T_1_1 * xt_B_C_T_H_W + c_out_B_1_T_1_1 * net_output_B_C_T_H_W
        if condition.is_video and self.config.denoise_replace_gt_frames:
            # Set the first few frames to the ground truth frames. This will ensure that the loss is not computed for the first few frames.
            x0_pred_B_C_T_H_W = condition.gt_frames.type_as(
                x0_pred_B_C_T_H_W
            ) * condition_video_mask + x0_pred_B_C_T_H_W * (1 - condition_video_mask)

        # get noise prediction based on sde
        eps_pred_B_C_T_H_W = (xt_B_C_T_H_W - x0_pred_B_C_T_H_W) / sigma_B_1_T_1_1

        pc_x0_pred = None
        pc_eps_pred = None
        pc_xt = getattr(condition, "pc_latent_xt", None)
        if pc_net_output is not None and pc_xt is not None:
            pc_net_output = pc_net_output.float()
            pc_xt = pc_xt.to(device=pc_net_output.device, dtype=pc_net_output.dtype)
            pc_xt_aligned = self._align_point_temporal(pc_xt, pc_net_output.shape[1])
            sigma_pc_B_T = sigma_B_T.mean(dim=1, keepdim=True).expand(-1, pc_net_output.shape[1]).to(
                device=pc_net_output.device, dtype=pc_net_output.dtype
            )
            c_skip_pc, c_out_pc, _, _ = self.scaling(sigma=rearrange(sigma_pc_B_T, "b t -> b t 1 1"))
            pc_x0_pred = c_skip_pc * pc_xt_aligned + c_out_pc * pc_net_output
            pc_eps_pred = (pc_xt_aligned - pc_x0_pred) / rearrange(sigma_pc_B_T, "b t -> b t 1 1")

        return DenoisePrediction(
            x0_pred_B_C_T_H_W,
            eps_pred_B_C_T_H_W,
            None,
            pc_x0=pc_x0_pred,
            pc_eps=pc_eps_pred,
        )


    def compute_loss_with_epsilon_and_sigma(
        self,
        x0_B_C_T_H_W: torch.Tensor,
        condition: Text2WorldCondition,
        epsilon_B_C_T_H_W: torch.Tensor,
        sigma_B_T: torch.Tensor,
    ):
        mean_B_C_T_H_W, std_B_T = self.sde.marginal_prob(x0_B_C_T_H_W, sigma_B_T)
        xt_B_C_T_H_W = mean_B_C_T_H_W + epsilon_B_C_T_H_W * rearrange(std_B_T, "b t -> b 1 t 1 1")

        condition, point_info = self._prepare_point_diffusion_condition(condition, sigma_B_T, x0_B_C_T_H_W)
        model_pred = self.denoise(xt_B_C_T_H_W, sigma_B_T, condition)

        weights_per_sigma_B_T = self.get_per_sigma_loss_weights(sigma=sigma_B_T)
        pred_mse_B_C_T_H_W = (x0_B_C_T_H_W - model_pred.x0) ** 2
        edm_loss_B_C_T_H_W = pred_mse_B_C_T_H_W * rearrange(weights_per_sigma_B_T, "b t -> b 1 t 1 1")

        kendall_loss = edm_loss_B_C_T_H_W
        video_edm_loss = edm_loss_B_C_T_H_W.mean()
        output_batch = {
            "x0": x0_B_C_T_H_W,
            "xt": xt_B_C_T_H_W,
            "sigma": sigma_B_T,
            "weights_per_sigma": weights_per_sigma_B_T,
            "condition": condition,
            "model_pred": model_pred,
            "mse_loss": pred_mse_B_C_T_H_W.mean(),
            "edm_loss": video_edm_loss,
            "video_loss_unscaled": video_edm_loss,
            "video_loss_scaled": video_edm_loss * float(self.loss_scale),
            "edm_loss_per_frame": torch.mean(edm_loss_B_C_T_H_W, dim=[1, 3, 4]),
        }

        if point_info is not None and model_pred.pc_x0 is not None:
            pc_loss = self._point_diffusion_loss(point_info, model_pred.pc_x0)
            pc_weighted_loss = pc_loss * float(self.config.point_diffusion_loss_weight)
            video_loss_scaled = output_batch["video_loss_scaled"].detach().clamp(min=1.0e-8)
            output_batch["pc_diffusion_loss"] = pc_loss
            output_batch["pc_loss_weighted"] = pc_weighted_loss
            output_batch["pc_to_video_loss_ratio"] = pc_weighted_loss.detach() / video_loss_scaled
            output_batch["aux_loss"] = pc_weighted_loss

        return output_batch, kendall_loss, pred_mse_B_C_T_H_W, edm_loss_B_C_T_H_W

    def get_x0_fn_from_batch(
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
        - Callable: A function `x0_fn(noise_x, sigma)` that takes two arguments, `noise_x` and `sigma`, and return x0 predictoin

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

        def x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            nonlocal point_state
            condition_for_step = (
                self._condition_from_point_sampling_state(condition, point_state)
                if point_state is not None else condition
            )

            if self.config.use_flowunipc_scheduler:
                cond_pred = self._denoise_prediction_from_flow_t(noise_x, sigma, condition_for_step)
                uncond_pred = self._denoise_prediction_from_flow_t(noise_x, sigma, uncondition)
                self._update_point_sampling_state(point_state, cond_pred.pc_x0)
                cond_velocity = cond_pred.eps - cond_pred.x0
                uncond_velocity = uncond_pred.eps - uncond_pred.x0
                velocity = uncond_velocity + guidance * (cond_velocity - uncond_velocity)
                return velocity

            cond_pred = self.denoise(noise_x, sigma, condition_for_step)
            uncond_pred = self.denoise(noise_x, sigma, uncondition)
            self._update_point_sampling_state(point_state, cond_pred.pc_x0)
            cond_x0 = cond_pred.x0
            uncond_x0 = uncond_pred.x0
            raw_x0 = cond_x0 + guidance * (cond_x0 - uncond_x0)
            if "guided_image" in data_batch:
                # replacement trick that enables inpainting with base model
                assert "guided_mask" in data_batch, "guided_mask should be in data_batch if guided_image is present"
                guide_image = data_batch["guided_image"]
                guide_mask = data_batch["guided_mask"]
                raw_x0 = guide_mask * guide_image + (1 - guide_mask) * raw_x0
            return raw_x0

        return x0_fn
