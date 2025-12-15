"""This scripts demonstrates how to train Diffusion Policy on the Aloha environment.

Once you have trained a model with this script, you can try to evaluate it on
examples/evaluate_pretrained_policy_custom.py
"""

from pathlib import Path
import numpy as np
import time

import torch

from lerobot.common.policies.act.modeling_act import PreTrainedPolicy, F, deque, ACTTemporalEnsembler, Normalize, Unnormalize, ACTEncoder, ACTSinusoidalPositionEmbedding2d, ACTDecoder, einops, torchvision, nn, IntermediateLayerGetter, FrozenBatchNorm2d, create_sinusoidal_pos_embedding, chain

#from lerobot.common.policies.flow.modeling_flow import FlowPolicy
#from lerobot.common.policies.flow.configuration_flow import FlowConfig
from franka_ai.models.actPatch.configuration_actpatch import ACTConfigPatch

from torchvision.models._utils import IntermediateLayerGetter
    
class ACTPolicyPatch(PreTrainedPolicy):
    """
    Action Chunking Transformer Policy as per Learning Fine-Grained Bimanual Manipulation with Low-Cost
    Hardware (paper: https://arxiv.org/abs/2304.13705, code: https://github.com/tonyzhaozh/act)
    """

    config_class = ACTConfigPatch
    name = "act_patch"

    def __init__(
        self,
        config: ACTConfigPatch,
        dataset_stats = None,
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

        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        self.model = ACTPatch(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def get_optim_params(self) -> dict:
        # TODO(aliberts, rcadene): As of now, lr_backbone == lr
        # Should we remove this and just `return self.parameters()`?
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        """This should be called whenever the environment is reset."""
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad
    def select_action(self, batch):
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()

        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = [torch.flatten(batch[key], start_dim=1, end_dim=2) for key in self.config.image_features]

        # If we are doing temporal ensembling, do online updates where we keep track of the number of actions
        # we are ensembling over.
        if self.config.temporal_ensemble_coeff is not None:
            actions = self.model(batch)[0]  # (batch_size, chunk_size, action_dim)
            actions = self.unnormalize_outputs({"action": actions})["action"]
            action = self.temporal_ensembler.update(actions)
            return action

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            actions = self.model(batch)[0][:, : self.config.n_action_steps]

            # TODO(rcadene): make _forward return output dictionary?
            actions = self.unnormalize_outputs({"action": actions})["action"]

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()
        
    @torch.no_grad
    def predict_actions(self, queue):
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()
        
        batch = {k: torch.stack(list(queue[k]), dim=1).to(self.device) for k in queue}
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = [torch.flatten(batch[key], start_dim=1, end_dim=2) for key in self.config.image_features]

        # If we are doing temporal ensembling, do online updates where we keep track of the number of actions
        # we are ensembling over.
        if self.config.temporal_ensemble_coeff is not None:
            actions = self.model(batch)[0]  # (batch_size, chunk_size, action_dim)
            actions = self.unnormalize_outputs({"action": actions})["action"]
            action = self.temporal_ensembler.update(actions)
            return action

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        actions = self.model(batch)[0][:, : self.config.n_action_steps]

        # TODO(rcadene): make _forward return output dictionary?
        actions = self.unnormalize_outputs({"action": actions})["action"]
            
        return actions

    def forward(self, batch):
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = [torch.flatten(batch[key], start_dim=1, end_dim=2) for key in self.config.image_features]

        batch = self.normalize_targets(batch)
        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        l1_loss = (
            F.l1_loss(batch["action"], actions_hat, reduction="none") * ~batch["action_is_pad"][:,self.config.n_obs_steps:].unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.item()}
        if self.config.use_vae:
            # Calculate Dₖₗ(latent_pdf || standard_normal). Note: After computing the KL-divergence for
            # each dimension independently, we sum over the latent dimension to get the total
            # KL-divergence per batch element, then take the mean over the batch.
            # (See App. B of https://arxiv.org/abs/1312.6114 for more details).
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss

        return loss, loss_dict

class ACTPatch(nn.Module):
    def __init__(self, config):

        # BERT style VAE encoder with input tokens [cls, robot_state, *action_sequence].
        # The cls token forms parameters of the latent's distribution (like this [*means, *log_variances]).
        super().__init__()
        self.config = config

        if self.config.use_vae:
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            # Projection layer for joint-space configuration to hidden dimension.
            if self.config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    self.config.robot_state_feature.shape[0], config.dim_model
                )
            # Projection layer for action (joint-space target) to hidden dimension.
            self.vae_encoder_action_input_proj = nn.Linear(
                self.config.action_feature.shape[0],
                config.dim_model,
            )
            # Projection layer from the VAE encoder's output to the latent distribution's parameter space.
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            # Fixed sinusoidal positional embedding for the input to the VAE encoder. Unsqueeze for batch
            # dimension.
            num_input_token_encoder = 1 + config.chunk_size
            if self.config.robot_state_feature:
                num_input_token_encoder += 1
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
            )

        # Backbone for image feature extraction.
        if self.config.image_features:
            self.rgb_encoder = ACTRgbEncoder(config)
            
            #backbone_model = getattr(torchvision.models, config.vision_backbone)(
            #    weights=config.pretrained_backbone_weights,
            #)
            # Note: The assumption here is that we are using a ResNet model (and hence layer4 is the final
            # feature map).
            # Note: The forward method of this returns a dict: {"feature_map": output}.
            #self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        # Transformer (acts as VAE decoder when training with the variational objective).
        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)

        # Transformer encoder input projections. The tokens will be structured like
        # [latent, (robot_state), (env_state), (image_feature_map_pixels)].
        if self.config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                self.config.robot_state_feature.shape[0], config.dim_model
            )
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                self.config.env_state_feature.shape[0], config.dim_model
            )
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        if self.config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                self.rgb_encoder.in_features, config.dim_model, kernel_size=1
            )
        # Transformer encoder positional embeddings.
        n_1d_tokens = 1  # for the latent
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if self.config.image_features:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        # Transformer decoder.
        # Learnable positional embedding for the transformer's decoder (in the style of DETR object queries).
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        # Final action regression head on the output of the transformer's decoder.
        self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier-uniform initialization of the transformer parameters as in the original code."""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch):
        """A forward pass through the Action Chunking Transformer (with optional VAE encoder).

        `batch` should have the following structure:
        {
            [robot_state_feature] (optional): (B, state_dim) batch of robot states.

            [image_features]: (B, n_cameras, C, H, W) batch of images.
                AND/OR
            [env_state_feature]: (B, env_dim) batch of environment states.

            [action_feature] (optional, only if training with VAE): (B, chunk_size, action dim) batch of actions.
        }

        Returns:
            (B, chunk_size, action_dim) batch of action sequences
            Tuple containing the latent PDF's parameters (mean, log(σ²)) both as (B, L) tensors where L is the
            latent dimension.
        """
        if self.config.use_vae and self.training:
            assert "action" in batch, (
                "actions must be provided when using the variational objective in training mode."
            )

        if "observation.images" in batch:
            batch_size = batch["observation.images"][0].shape[0]
        else:
            batch_size = batch["observation.environment_state"].shape[0]

        # Prepare the latent for input to the transformer encoder.
        if self.config.use_vae and "action" in batch:
            # Prepare the input to the VAE encoder: [cls, *joint_space_configuration, *action_sequence].
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )  # (B, 1, D)
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch["observation.state"])
            action_embed = self.vae_encoder_action_input_proj(batch["action"])  # (B, S, D)

            #print(cls_embed.shape, robot_state_embed.shape, action_embed.shape)
            if self.config.robot_state_feature:
                vae_encoder_input = [cls_embed, robot_state_embed, action_embed]  # (B, S+2, D)
            else:
                vae_encoder_input = [cls_embed, action_embed]
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

            # Prepare fixed positional embedding.
            # Note: detach() shouldn't be necessary but leaving it the same as the original code just in case.
            pos_embed = self.vae_encoder_pos_enc.clone().detach()  # (1, S+2, D)

            # Prepare key padding mask for the transformer encoder. We have 1 or 2 extra tokens at the start of the
            # sequence depending whether we use the input states or not (cls and robot state)
            # False means not a padding token.
            cls_joint_is_pad = torch.full(
                (batch_size, 1),
                False,
                device=batch["observation.state"].device,
            )
            key_padding_mask = torch.cat(
                [cls_joint_is_pad, batch["action_is_pad"]], axis=1
            )  # (bs, seq+1 or 2)

            #print(vae_encoder_input.permute(1, 0, 2).shape, key_padding_mask.shape, batch["action_is_pad"], batch["frame_index"], batch["episode_index"])

            # Forward pass through VAE encoder to get the latent PDF parameters.
            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]  # select the class token, with shape (B, D)
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            # This is 2log(sigma). Done this way to match the original implementation.
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]

            # Sample the latent with the reparameterization trick.
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            # When not using the VAE encoder, we set the latent to be all zeros.
            mu = log_sigma_x2 = None
            # TODO(rcadene, alexander-soare): remove call to `.to` to speedup forward ; precompute and use buffer
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
                batch["observation.state"].device
            )

        # Prepare transformer encoder inputs.
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        # Robot state token.
        if self.config.robot_state_feature:
            #TODO:BIG MODIF
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch["observation.state"].squeeze(1)))
        # Environment state token.
        if self.config.env_state_feature:
            encoder_in_tokens.append(
                self.encoder_env_state_input_proj(batch["observation.environment_state"])
            )

        # Camera observation features and positional embeddings.
        if self.config.image_features:
            all_cam_features = []
            all_cam_pos_embeds = []

            # For a list of images, the H and W may vary but H*W is constant.
            for img in batch["observation.images"]:
                img = self.rgb_encoder(img)
                
                #cam_features = self.backbone(img)["feature_map"]
                cam_features = img
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)

                # Rearrange features to (sequence, batch, dim).
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                all_cam_features.append(cam_features)
                all_cam_pos_embeds.append(cam_pos_embed)

            encoder_in_tokens.extend(torch.cat(all_cam_features, axis=0))
            encoder_in_pos_embed.extend(torch.cat(all_cam_pos_embeds, axis=0))

        # Stack all tokens along the sequence dimension.
        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        # Forward pass through the transformer modules.
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        # TODO(rcadene, alexander-soare): remove call to `device` ; precompute and use buffer
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        # Move back to (B, S, C).
        decoder_out = decoder_out.transpose(0, 1)

        actions = self.action_head(decoder_out)

        return actions, (mu, log_sigma_x2)
    
class ACTRgbEncoder(nn.Module):
    """Encodes an RGB image into a 1D feature vector.

    Includes the ability to normalize and crop the image first.
    """

    def __init__(self, config):
        super().__init__()

        if config.crop_pos is not None:
            self.do_crop_pos = True
            self.crop_pos = config.crop_pos
            self.crop_shape = config.crop_shape
        else:
            self.do_crop_pos = False
        
        if config.random_crop_shape is not None:
            self.do_crop = True
            # Always use center crop for eval
            self.center_crop = torchvision.transforms.CenterCrop(config.random_crop_shape)
            self.maybe_random_crop = torchvision.transforms.RandomCrop(config.random_crop_shape)
        else:
            self.do_crop = False

        if(config.resize_shape is not None):
            self.do_resize = True
            self.resizer = torchvision.transforms.Resize(config.resize_shape)
        else:
            self.do_resize = False

        #TODO: verify if this addition is correct
        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        # Note: This assumes that the layer4 feature map is children()[-3]
        # TODO(alexander-soare): Use a safer alternative.
        #self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))

        self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        #self.in_features = backbone_model.heads.head.in_features # ViT
        self.in_features = backbone_model.fc.in_features
            
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) image tensor with pixel values in [0, 1].
        Returns:
            (B, C, H, W) preprocessed image tensor
        """
        # Preprocess: maybe crop (if it was set up in the __init__).
        if self.do_crop_pos:
            x = torchvision.transforms.functional.crop(x, self.crop_pos[1], self.crop_pos[0], self.crop_shape[1], self.crop_shape[0])
        if self.do_resize:
            x = self.resizer(x)
        
        if self.do_crop:
            if self.training:  # noqa: SIM108
                x = self.maybe_random_crop(x)
            else:
                # Always use center crop for eval.
                x = self.center_crop(x)
        
        return self.backbone(x)["feature_map"] #Resnet
        return self.backbone(x) #ViT
    
