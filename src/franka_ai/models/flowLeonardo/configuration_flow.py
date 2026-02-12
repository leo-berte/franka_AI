#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode


@PreTrainedConfig.register_subclass("flow_leonardo")
@dataclass
class FlowConfig(PreTrainedConfig):

    # Input / output structure
    N_history: int = 1
    N_chunk: int = 100
    n_action_steps: int = 100

    # normalization
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Vision architecture params
    vision_backbone_name : str | None = "vit_base_patch16_224.mae" # "resnet18d.ra2_in1k" "vit_base_patch16_224.mae" "vit_base_patch14_dinov2.lvd142m" "vit_base_patch14_reg4_dinov2.lvd142m" 
    freeze_vision_backbone: bool = True

    # Resampler hparams
    use_perceiver_resampler: bool = True
    resampler_params: dict = field(
        default_factory=lambda: {
            "depth": 4,
            "dim_head": 64,
            "heads": 8,
            "num_latents": 64,
        }
    )

    # Transformer architecture params
    dim_model: int = 512
    dim_feedforward_enc: int = 3200
    nhead_enc: int = 8
    num_layers_enc: int = 4
    
    # FM type
    head_type: str = "flow_matching_transformer" # "mlp" "flow_matching_mlp" "flow_matching_unet" "flow_matching_transformer"

    # Common
    use_film: bool = True
    timestep_embed_dim: int = 256
    
    # Unet
    down_dims: tuple[int, ...] = (256, 512)
    kernel_size: int = 5
    n_groups: int = 8

    # Transformer decoder
    dim_feedforward_dec: int = 2048
    nhead_dec: int = 8
    num_layers_dec: int = 4

    # Flow inference
    denoising_steps: int = 20
    ode_type: str = "euler" # "euler" "runge_kutta" 
    use_fixed_src_dist: bool = False # decide whether to start always from same source distribution during inference

    def __post_init__(self):

        super().__post_init__()

        if self.n_action_steps > self.N_chunk:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.N_chunk} for `N_chunk`."
            )
    
    # Mandatory methods due to inheritance from 'PreTrainedConfig'

    def get_optimizer_preset(self):
        pass

    def get_scheduler_preset(self):
        pass

    def validate_features(self):
        pass

    @property
    def observation_delta_indices(self):
        pass

    @property
    def action_delta_indices(self):
        pass

    @property
    def reward_delta_indices(self):
        pass