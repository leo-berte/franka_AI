#!/usr/bin/env python

from collections import deque
import math

import einops
import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from franka_ai.models.flowLeonardo.configuration_flow import FlowConfig

from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pretrained import PreTrainedPolicy


# Pipeline: Transformer → embedding → Flow network
# Transformer acts as a state encoder: e = encoderTransf(image, text, proprio)
# I take the [CLS] token to get a single embedding as sumup
# A smaller network (often MLP) predicts the flow: flow = MLP(concat(x_t, t, e))


# TODO:

# 1) Studia con esempi codice: cat/stack, view/reshape/flatten, einops/repeat, ..



class FlowPolicy(PreTrainedPolicy):

    config_class = FlowConfig
    name = "flow_leonardo"

    def __init__(
        self,
        config: FlowConfig,
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
        self.model = Flow(config)

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

        # Normalize inputs
        batch = self.normalize_inputs(batch)

        # Processing
        batch = self.prepare_image_features(batch)

        # Action queue logic for n_action_steps > 1
        if len(self._action_queue) == 0:
            
            # When the action_queue is depleted, populate it by querying the policy
            actions = self.model.generate_action(batch)[:, :self.config.n_action_steps]   ??????

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

        # Normalize inputs
        batch = self.normalize_inputs(batch)

        # Processing
        batch = self.prepare_image_features(batch)

        # Normalize targets
        batch = self.normalize_targets(batch)

        # Compute loss
        loss = self.model.compute_loss(batch)
        
        # no output_dict so returning None
        return loss, None


class Flow(nn.Module):

    """
    Flow policy model.
    """

    def __init__(self, config: FlowConfig):

        super().__init__()
        self.config = config

        if self.config.robot_state_feature:
            self.obs_projector = ObsProjector(input_dim=self.config.robot_state_feature.shape[0], output_dim=self.config.dim_model)
            # self.binary_ee_projector = nn.Embedding(2, self.config.embed_dim) # 2 states: close/open

        if self.config.image_features:
            self.vision_projector = VisionProjector(output_dim=self.config.dim_model)

        self.encoder = TransformerEncoder(dim_model=self.config.dim_model, num_layers=self.config.num_layers, nhead=self.config.nhead)
        
        # Final flow head
        self.flow = FlowHead(action_dim=self.config.action_feature.shape[0], dim_model=self.config.dim_model)
        # self.action_head = nn.Linear(config.dim_model, config.N_chunk * self.config.action_feature.shape[0])

    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:

        """
        A forward pass through the Flow policy.

        `batch` should have the following structure:

        {
            [robot_state_feature]: (B, N_history, state_dim) batch of robot states

            [image_features]: (B, N_history, C, H, W) batch of images
        }

        Returns:
            (B, N_chunk, action_dim) batch of action sequences
        """

        # --> x_t, t, obs, img, ee_status

        # obs projector
        if self.config.robot_state_feature:
            obs_features = self.obs_projector(batch["observation.state"]) # [B, N_HISTORY, dim_model]
            # ee_embeds = self.binary_ee_projector(ee_status) # [B, N_HISTORY, dim_model] --> I look up integers in ee_status and return corresponding embeddings

        # vision projector
        if self.config.image_features:
            
            all_cam_features = []

            # Create a list of cameras
            for images in batch["observation.images"]: # [B, N_HIST, C, H, W]

                B, N_HIST, C, H, W = images.shape
                images = images.view(B * N_HIST, C, H, W)

                self.vision_projector(images) # [B*N_HIST, dim_model, H', W']

                _, dim_model, H_prime, W_prime = cam_features.shape
                cam_features = cam_features.view(B, N_HIST, dim_model, H_prime, W_prime)

                all_cam_features.append(cam_features) # [B, N_HIST, dim_model, H', W']

        # transformer embedding
        # e = self.encoder(obs_embeds, img_embeds, ee_embeds) # [B, dim_model]
        e = self.encoder(obs_features, all_cam_features) # [B, dim_model]

        # predict flow
        v_pred = self.flow(x_t, t, e)

        return v_pred
    
    def generate_action(self, obs, img, ee, steps=20):
    
        # init nets
        model = PolicyModel(OBS_DIM, ACT_DIM, EMBED_DIM)

        # eval
        model.eval()

        with torch.no_grad():

            # add batch dimension
            obs = obs.unsqueeze(0) # [1, N_HISTORY, OBS_DIM]
            img = img.unsqueeze(0) # [1, N_HISTORY, 3, H, W]
            ee  = ee.unsqueeze(0)  # [1, N_HISTORY]                   

            # start from noise (gaussian or uniform)
            x = torch.rand(1, ACT_HORIZON, ACT_DIM)  

            # integrate flow
            dt = 1.0 / steps
            for i in range(steps):
                t = torch.ones(1, 1) * (i/steps)  # shape: [1,1]
                v = model(x, t, obs, img, ee) # [B, ACT_HORIZON, ACT_DIM]
                x = x + dt * v  # follow flow towards clean action

            return x  # denoised action (trajectory chunk)
    

class ObsProjector(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim), 
        )
        
    def forward(self, obs):
        return self.mlp(obs) 


class VisionProjector(nn.Module):
    
    def __init__(self, output_dim):
        
        super().__init__()

        # Backbone for image feature extraction
        backbone_model = getattr(torchvision.models, self.config.vision_backbone)(
            replace_stride_with_dilation=[False, False, self.config.replace_final_stride_with_dilation],
            weights=self.config.pretrained_backbone_weights,
            norm_layer=FrozenBatchNorm2d,
        )
        
        # Note: The assumption here is that we are using a ResNet model (and hence layer4 is the final
        # feature map). The forward method of this returns a dict: {"feature_map": output}.
        self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        self.img_projector = nn.Conv2d(
            backbone_model.fc.in_features, output_dim, kernel_size=1
        )

    def forward(self, obs): # [B, N_HIST, C, H, W]
        
        obs = self.backbone(obs)["feature_map"]
        obs = self.img_projector(obs) 
        return obs # [B*N_HIST, dim_model, H', W']
        
    
class TransformerEncoder(nn.Module):   

    def __init__(self, dim_model, num_layers, nhead):
        
        super(TransformerEncoder, self).__init__()
        
        self.pos_enc1D = PosEmbedding1D()
        self.pos_enc2D = PosEmbedding2D()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_model)) 
        encoder_layers = nn.TransformerEncoderLayer(d_model=dim_model, nhead=nhead, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

    def forward(self, obs_features, all_cam_features): # , ee_embeds):

        # get dimensions
        B, n_hist, dim_model = obs_features.shape

        # add 1D positional encoding to observations
        enc1D = self.pos_enc1D.compute(B, n_hist, dim_model)
        obs_features = obs_features + enc1D
        # ee_embeds = ee_embeds + enc1D

        # get dimensions
        B, n_hist, dim_model, H_prime, W_prime = all_cam_features[0].shape

        processed_all_cam_features = []

        for cam_features in all_cam_features: # loop over each camera

            # add spatial 2D positional encoding to images
            enc2D = self.pos_enc2D.compute(B, dim_model, H_prime, W_prime)
            cam_features = cam_features + enc2D.unsqueeze(1)  # [B, N_HIST, dim_model, H', W'] --> broadcast along N_HIST dimension

            # build token representation (B, seq_len, d_model)
            cam_features = cam_features.flatten(3).permute(0, 1, 3, 2)  # [B, N_HIST, H'*W', dim_model]
            cam_features = cam_features.reshape(B, n_hist * H_prime * W_prime, dim_model)  # [B, N_HIST*H'*W', dim_model]

            # add temporal 1D positional encoding to images
            enc1D = self.pos_enc1D.compute(B, n_hist * H_prime * W_prime, dim_model)
            cam_features = cam_features + enc1D # [B, N_HIST*H'*W', dim_model]

            processed_all_cam_features.append(cam_features)

        img_features = torch.cat(processed_all_cam_features, dim=1) # [B, N_cam*N_HIST*H'*W', dim_model]

        # preperare data for transformer
        cls_tokens = self.cls_token.repeat(B, 1, 1) # duplicate learnable CLS token for all samples in the batch --> [B, 1, dim_model]

        # x = torch.cat([cls_tokens, obs_features, img_features, ee_embeds], dim=1) # [B, seq_length=1+n_hist+N_cam*N_HIST*H'*W', dim_model]
        x = torch.cat([cls_tokens, obs_features, img_features], dim=1) # [B, seq_length=1+n_hist+N_cam*N_HIST*H'*W', dim_model]

        # pass through transformer
        x = self.transformer_encoder(x) # [B, seq_length, dim_model]
        x = x[:,0,:]  # take only CLS token as summary --> [B, dim_model]
        
        return x


class FlowHead(nn.Module):     
    
    def __init__(self, action_dim, dim_model):
        
        super().__init__()
        
        in_dim = action_dim + 1 + dim_model  # per timestep: action + time + context
        out_dim = action_dim  # per timestep flow

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, x_t, t, e):

        B, H, A = x_t.shape

        # duplicate t and e across horizon (each timestep in the action chunk sees the same time t and same context embedding e)
        t_expand = t.unsqueeze(1).expand(B, H, 1)   # [B,H,1]
        e_expand = e.unsqueeze(1).expand(B, H, self.embed_dim)  # [B,H,E]

        # concat along feature dim
        inp = torch.cat([x_t, t_expand, e_expand], dim=-1)  # [B,H,A+1+E]

        # flatten horizon so MLP sees each timestep independently
        inp = inp.reshape(B*H, -1)  # [B*H, A+1+E]

        # predict flows
        out = self.mlp(inp)  # [B*H, A]

        # reshape back to chunk
        out = out.view(B, H, A)  # [B,H,A]
        
        return out


class PosEmbedding1D():

    """
    1D sinusoidal positional embeddings logic
    """

    def __init__(self):

        # params
        self.temperature = 10000 # ratio for the geometric progression in sinusoid frequencies

    def compute(self, batch, seq_length, d_model):

        # compute positions and frequencies
        pos = torch.arange(seq_length).unsqueeze(1)             # [seq_length, 1]
        i = torch.arange(d_model).unsqueeze(0)                  # [1, d_model]
        frequencies = 1.0 / (self.temperature ** (2 * (i//2) / d_model))   # [1, d_model] --> [f0,f0,f1,f1,...]

        # apply formula: even index -> sin, odd index -> cos
        angle_rads = pos * frequencies # [seq_length, d_model] --> [pos*f0,pos*f0,pos*f1,pos*f1,...]
        encodings = torch.zeros_like(angle_rads) # [seq_length, d_model]
        encodings[:, 0::2] = torch.sin(angle_rads[:, 0::2]) # even (slicing syntax -> start:stop:step)
        encodings[:, 1::2] = torch.cos(angle_rads[:, 1::2]) # odd  (slicing syntax -> start:stop:step))
        
        return encodings.unsqueeze(0).expand(batch, -1, -1) # [batch, seq_length, d_model]


class PosEmbedding2D():

    """
    2D sinusoidal positional embeddings logic
    """

    def __init__(self):

        # params
        self.two_pi = 2 * math.pi
        self.temperature = 10000 # ratio for the geometric progression in sinusoid frequencies
    
    def compute(self, B, C, H, W):

        # Create 2D positions
        y_pos = torch.arange(H, dtype=torch.float32).unsqueeze(1)  # (H,1)
        x_pos = torch.arange(W, dtype=torch.float32).unsqueeze(0)  # (1,W)

        # Normalize positions to [0, 2pi]
        y_pos = y_pos / H * self.two_pi  # (H,1)
        x_pos = x_pos / W * self.two_pi  # (1,W)

        # Compute inverse frequencies for half of the channels each
        d_model = C // 2
        assert d_model % 2 == 0, "d_model must be even for sine/cosine pairing in 2D positional encoding"
        indexes = torch.arange(0, d_model, 2, dtype=torch.float32) # (d_model/2)
        inv_freq = 1.0 / (self.temperature ** (indexes / d_model)) # (d_model/2)

        # Compute angles (pos*freq) + Expand to H/W dimensions
        y_angle = y_pos.unsqueeze(2) / inv_freq  # (H,1,d_model/2)
        x_angle = x_pos.unsqueeze(2) / inv_freq  # (1,W,d_model/2)

        # Apply sin/cos alternately
        y_embed = torch.zeros(H,1,d_model) # (H,1,d_model)
        y_embed[:, :, 0::2] = y_angle.sin()
        y_embed[:, :, 1::2] = y_angle.cos()

        x_embed = torch.zeros(1,W,d_model) # (1,W,d_model)
        x_embed[:, :, 0::2] = x_angle.sin()
        x_embed[:, :, 1::2] = x_angle.cos()

        # Broadcast to (H,W,d_model)
        y_embed = y_embed.expand(H,W,d_model)
        x_embed = x_embed.expand(H,W,d_model)

        # Concatenate along channel dim
        # --> i.e with C=8: cell (i,j) has features: [sin(i*f0) cos(i*f0) sin(i*f1) cos(i*f1) | sin(j*f0) cos(j*f0) sin(j*f1) cos(j*f1)]
        pos_embed = torch.cat([y_embed, x_embed], dim=-1)  # (H,W,C)

        # Add batch dim and permute to (B,C,H,W)
        pos_embed = pos_embed.permute(2,0,1).unsqueeze(0).expand(B,-1,-1,-1)  # (B,C,H,W)

        return pos_embed