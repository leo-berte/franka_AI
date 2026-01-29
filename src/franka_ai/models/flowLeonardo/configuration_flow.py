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

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Vision architecture params
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: int = False
    # Transformer architecture params
    dim_model: int = 512
    dim_feedforward_tf: int = 3200
    nhead: int = 8
    num_layers: int = 4
    # Flow Head
    dim_feedforward_flow: int = 512*4
    # Flow inference
    denoising_steps: int = 100
    use_fixed_src_dist = True # decide whether to start always from same source distribution during inference


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