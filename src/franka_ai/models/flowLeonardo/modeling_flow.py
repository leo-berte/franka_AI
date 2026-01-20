#!/usr/bin/env python

from collections import deque

import einops
import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from franka_ai.models.template.configuration_template import TemplateConfig

from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pretrained import PreTrainedPolicy


class TemplatePolicy(PreTrainedPolicy):

    config_class = TemplateConfig
    name = "template"

    def __init__(
        self,
        config: TemplateConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """

        super().__init__(config)
        config.validate_features()
        self.config = config

        # Normalization settings
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(config.output_features, config.normalization_mapping, dataset_stats)
        self.unnormalize_outputs = Unnormalize(config.output_features, config.normalization_mapping, dataset_stats)

        # Define policy model
        self.model = Template(config)

        self.reset()

    def get_optim_params(self):

        """Mandatory method due to inheritance from 'PreTrainedPolicy'."""

        pass

    def reset(self):

        """This should be called whenever the environment is reset."""

        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def prepare_image_features(self, batch):

        if not self.config.image_features:
            return batch
        
        batch = dict(batch) # shallow copy so that adding a key doesn't modify the original
        batch["observation.images"] = [batch[k] for k in self.config.image_features] # list containing each img tensor

        return batch

    def remove_time_dimension(self, batch):

        batch = dict(batch) # shallow copy so that adding a key doesn't modify the original

        # PATCH: remove time dimension from state taking only last step
        batch["observation.state"] = batch["observation.state"][:, -1, :] # (B, N_h, D) --> (B, D)

        # PATCH: remove time dimension from images taking only last step
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and v.dim() == 5:  # (B, N_h, C, H, W)
                batch[k] = v[:, -1, ...] # (B, C, H, W)

        return batch

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        
        """
        Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.

        `batch` should have the following structure:

        {
            [robot_state_feature]: (B, N_history, state_dim) batch of robot states

            [image_features]: (B, N_history, C, H, W) batch of images
        }

        Returns:
            (B, N_chunk, action_dim) batch of action sequences
        """
        
        self.eval()

        # PATCH
        batch = self.remove_time_dimension(batch)

        # Normalize inputs
        batch = self.normalize_inputs(batch)

        # Processing
        batch = self.prepare_image_features(batch)

        # Action queue logic for n_action_steps > 1
        if len(self._action_queue) == 0:
            
            # When the action_queue is depleted, populate it by querying the policy
            actions = self.model(batch)[:, : self.config.n_action_steps]

            # unnormalize outputs
            actions = self.unnormalize_outputs({"action": actions})["action"]

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:

        """
        Run the batch through the model and compute the loss for training or validation.
        
        `batch` should have the following structure:

        {
            [robot_state_feature]: (B, N_history, state_dim) batch of robot states

            [image_features]: (B, N_history, C, H, W) batch of images
        }

        Returns:
            (B, N_chunk, action_dim) batch of action sequences
        """

        # PATCH
        batch = self.remove_time_dimension(batch)

        # Normalize inputs
        batch = self.normalize_inputs(batch)

        # Processing
        batch = self.prepare_image_features(batch)

        # Normalize targets
        batch = self.normalize_targets(batch)

        # Inference
        actions_hat = self.model(batch)

        # Compute loss
        l1_loss = (F.l1_loss(batch["action"], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)).mean()

        loss_dict = {"l1_loss": l1_loss.item()}

        return l1_loss, loss_dict





class Template(nn.Module):

    """
    Template policy model.
    """

    def __init__(self, config: TemplateConfig):

        super().__init__()
        self.config = config

        # Backbone for image feature extraction.
        if self.config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            # Note: The assumption here is that we are using a ResNet model (and hence layer4 is the final
            # feature map). The forward method of this returns a dict: {"feature_map": output}.
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        # The tokens will be structured like: [robot_state, image_feature_map_pixels]

        if self.config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                self.config.robot_state_feature.shape[0], config.dim_model
            )

        if self.config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )

        self.encoder_layers = nn.TransformerEncoderLayer(d_model=config.dim_model, nhead=config.nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, num_layers=config.num_layers)

        # Final action head
        self.action_head = nn.Linear(config.dim_model, config.N_chunk * self.config.action_feature.shape[0])

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:

        """
        A forward pass through the Template policy.

        `batch` should have the following structure:

        {
            [robot_state_feature]: (B, N_history, state_dim) batch of robot states

            [image_features]: (B, N_history, C, H, W) batch of images
        }

        Returns:
            (B, N_chunk, action_dim) batch of action sequences
        """

        # Prepare transformer encoder inputs
        encoder_in_tokens = []

        # Robot state token
        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch["observation.state"])) # (B, d_model)

        # Camera observation features
        if self.config.image_features:
            
            all_cam_features = []

            # Create a list of cameras
            for img in batch["observation.images"]:

                cam_features = self.backbone(img)["feature_map"]
                cam_features = self.encoder_img_feat_input_proj(cam_features)

                # Rearrange features to (sequence, batch, dim)
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")

                all_cam_features.append(cam_features) # (n_features, B, d_model)

            encoder_in_tokens.extend(torch.cat(all_cam_features, axis=0)) 

        # Stack all tokens along the sequence dimension
        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0) # (1+n_features, B, d_model)

        # Pass through transformer
        encoder_out_tokens = self.transformer_encoder(encoder_in_tokens) # (1+n_features, B, d_model)
        encoder_out_tokens = encoder_out_tokens.mean(dim=0)  # (B, d_model) --> sequence average (or just take last sequence element)
        
        # Pass through action head
        actions = self.action_head(encoder_out_tokens) # (B, N_chunk*action_dim)
        actions = actions.view(-1, self.config.N_chunk, self.config.action_feature.shape[0]) # (B, N_chunk, action_dim)

        return actions