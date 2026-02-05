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
    
    # Head type
    head_type: str = "flow_matching_mlp" # "mlp" "flow_matching_mlp" "flow_matching_unet" "difusion"

    # Flow Head
    use_film: bool = True
    timestep_embed_dim: int = 256
    
    # Flow inference
    denoising_steps: int = 50
    use_fixed_src_dist: bool = False # decide whether to start always from same source distribution during inference
    
    # Unet
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True


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




# Total number of trainable parameters: 51,584,906
#   - vae_encoder: 17,332,736
#   - vae_encoder_cls_embed: 512
#   - vae_encoder_robot_state_input_proj: 8,192
#   - vae_encoder_action_input_proj: 5,632
#   - vae_encoder_latent_output_proj: 32,832
#   - backbone: 11,166,912
#   - encoder: 17,332,736
#   - decoder: 5,385,856
#   - encoder_robot_state_input_proj: 8,192
#   - encoder_latent_input_proj: 16,896
#   - encoder_img_feat_input_proj: 262,656
#   - encoder_1d_feature_pos_embed: 1,024
#   - encoder_cam_feat_pos_embed: 0
#   - decoder_pos_embed: 25,600
#   - action_head: 5,130


# Total number of trainable parameters: 270,278,442
#   - rgb_encoder: 11,197,088
#   - unet: 259,081,354


# flow MLP
# Total number of trainable parameters: 46,229,912
#   - obs_projector: 5,120
#   - vision_projector: 11,429,568
#   - encoder: 17,334,272
#   - flow_head: 17,460,952


# mlp puro
# Total number of trainable parameters: 31,868,584
#   - obs_projector: 5,120
#   - vision_projector: 11,429,568
#   - encoder: 17,334,272
#   - mlp_head: 3,099,624

