#!/usr/bin/env python

from collections import deque

import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from franka_ai.models.flowLeonardo.configuration_flow import FlowConfig

from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pretrained import PreTrainedPolicy


# scrivi nelle note procedimento:

# tracking ma rumeroso: adjust lr e weight decays alto (xke ho a che fare con rumore, e per evitare che overfitti rumore con pesi enormi uso weifght decay)
# layer norm dentro flow quando metto insieme diversi segnali
# come condizionare sul tempo: singolo value vs dim_model values per MLP
# sampling lontano dai bordi (tanto rumore pure, inutile) +  weight on loss based on time t 
# --> (xke ai bordi ho rumore pure e loss alta, rischio che mi dia contributo enorme in update pesi anche se quella loss è "cieca")
# change numer denois steps + RK 



# TODO:

# usa mlp pure dopo trasformer invece di fm
# metti 2 tricks sul tempo insieme + remove MLP e tiene solo pos encoding per t + film modulation for time


# try to plot offline evaluation for cubes_with_grasps
# training senza augmentation??
# training con codice mathis

# Use CLIP-pretrained ViT-B/32 encoder [29]. Images are resized to 224×224 pixels --> See force UMI gripper
# update comments strings



class FlowPolicy(PreTrainedPolicy):

    """
    Pipeline: Transformer → embedding → Flow network
    Transformer acts as a state encoder: e = encoderTransf(image, text, proprio)
    I take the [CLS] token to get a single embedding as sumup
    A smaller network (often MLP) predicts the flow: flow = MLP(concat(x_t, t, e))

    Args:
        config: Policy configuration class instance or None, in which case the default instantiation of
                the configuration class is used.
        dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
            that they will be passed with a call to `load_state_dict` before the policy is used.
    """
    
    config_class = FlowConfig
    name = "flow_leonardo"

    def __init__(
        self,
        config: FlowConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):

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

        Return one action at a time for execution in the environment. It works by managing the actions 
        in a queue and only calling `select_actions` when the queue is empty.

        `batch` should have the following structure:

        {
            [robot_state_feature]: (B, N_history, state_dim) batch of robot states

            [image_features]: (B, N_history, C, H, W) batch of images
        }

        Returns:
            (B, action_dim) batch of action sequences
        """
        
        self.eval()

        # Normalize inputs
        batch = self.normalize_inputs(batch)

        # Processing
        batch = self.prepare_image_features(batch)

        # Action queue logic for n_action_steps > 1
        if len(self._action_queue) == 0:
            
            # When the action_queue is depleted, populate it by querying the policy
            actions = self.model.generate_action(batch) # [1, N_chunk, act_dim]
            actions = actions[:, :self.config.n_action_steps, :] # [1, n_action_steps, act_dim]

            # unnormalize outputs
            actions = self.unnormalize_outputs({"action": actions})["action"]

            # The queue has shape (n_action_steps, batch_size, *), hence the transpose
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
            self.obs_projector = ObsProjector(input_dim=self.config.robot_state_feature.shape[0], 
                                              output_dim=self.config.dim_model)
            # self.binary_ee_projector = nn.Embedding(2, self.config.embed_dim) # 2 states: close/open

        if self.config.image_features:
            self.vision_projector = VisionProjector(replace_final_stride_with_dilation=self.config.replace_final_stride_with_dilation, 
                                                    vision_backbone=self.config.vision_backbone, 
                                                    pretrained_backbone_weights=self.config.pretrained_backbone_weights, 
                                                    output_dim=self.config.dim_model)

        self.encoder = TransformerEncoder(dim_model=self.config.dim_model, 
                                          dim_feedforward_tf=self.config.dim_feedforward_tf,
                                          nhead=self.config.nhead,
                                          num_layers=self.config.num_layers)
        
        # Final flow head
        self.flow_head = FlowHead(N_chunk=self.config.N_chunk, 
                                  action_dim=self.config.action_feature.shape[0], 
                                  dim_model=self.config.dim_model,
                                  dim_feedforward_flow=self.config.dim_feedforward_flow)

    def predict_flow(self, x_t : Tensor, t : Tensor, batch_obs : Tensor, batch_images : list[Tensor]) -> Tensor | None:

        """
        Predicts the flow given the source action, current time and batch.
        """

        # obs projector
        if self.config.robot_state_feature:
            obs_features = self.obs_projector(batch_obs) # [B, N_HISTORY, dim_model]
            # ee_embeds = self.binary_ee_projector(ee_status) # [B, N_HISTORY, dim_model] --> I look up integers in ee_status and return corresponding embeddings

        # vision projector
        if self.config.image_features:
            
            all_cam_features = []

            # Create a list of cameras
            for images in batch_images: # [B, N_HIST, C, H, W]

                B, N_HIST, C, H, W = images.shape
                images = images.view(B * N_HIST, C, H, W)

                cam_features = self.vision_projector(images) # [B*N_HIST, dim_model, H', W']

                _, dim_model, H_prime, W_prime = cam_features.shape
                cam_features = cam_features.view(B, N_HIST, dim_model, H_prime, W_prime) # [B, N_HIST, dim_model, H', W']

                all_cam_features.append(cam_features)

        # transformer embedding
        # e = self.encoder(obs_embeds, img_embeds, ee_embeds) # [B, dim_model]
        e = self.encoder(obs_features, all_cam_features) # [B, dim_model] (since I take only CLS token)

        # predict flow
        v_pred = self.flow_head(x_t, t, e) # [B, N_chunk, act_dim]

        return v_pred
    
    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor | None:

        """
        A forward pass through the Flow policy.

        `batch` should have the following structure:

        {
            [robot_state_feature]: (B, N_history, state_dim) batch of robot states

            [image_features]: (B, N_history, C, H, W) batch of images
        }
        """

        # get device type
        device = batch["action"].device

        # sample random noise for source distribution (uniform or gaussian)
        B, N_chunk, act_dim = batch["action"].shape
        # action_chunk_src = torch.rand(B, N_chunk, act_dim, device=device) # uniform
        action_chunk_src = torch.randn(B, N_chunk, act_dim, device=device) # gaussian

        # target flow (ground truth vector field)
        v_target = batch["action"] - action_chunk_src # [B, N_chunk, act_dim] 

        # sample random time t ~ U(0,1) --> BASELINE
        t = torch.rand(B, 1, 1, device=device)  # each sample has its own time

        # # sample random time t (trick: avoid sampling frequently near borders 0.0 & 1.0)
        # m = torch.randn(B, 1, 1, device=device) # gaussian distribution
        # sigma = 1.0 # the lower this param, the more the values will be around 0.5
        # t = torch.sigmoid(m * sigma) # result will be again in [0,1]

        # interpolation toward noise
        x_t = action_chunk_src + t * v_target # [B, N_chunk, act_dim] 

        # predict flow
        v_pred = self.predict_flow(x_t, 
                                   t, 
                                   batch["observation.state"], 
                                   batch["observation.images"]) # [B, N_chunk, act_dim]
        
        # Flow Matching loss
        loss = F.mse_loss(v_pred, v_target, reduction="none") # [B, N_chunk, act_dim]

        # select valid timestamps only
        in_episode_bound = ~batch["action_is_pad"] # [B, N_chunk]
        loss = (loss * in_episode_bound.unsqueeze(-1))

        # give less weight for small values of time t, since prediction is less relevant there
        weight = t / (t + 0.1)
        loss = (loss * weight)

        return loss.mean()
    
    def generate_action(self, batch: dict[str, Tensor]) -> Tensor | None:

        # get device type
        device = batch["observation.state"].device

        # get dimensions
        B = batch["observation.state"].shape[0]
        N_chunk = self.config.N_chunk
        act_dim = self.config.action_feature.shape[0]

        # start from noise (gaussian or uniform)
        # x = torch.rand(B, N_chunk, act_dim, device=device) # uniform
        x = torch.randn(B, N_chunk, act_dim, device=device) # gaussian

        # integrate flow
        dt = 1.0 / self.config.denoising_steps

        for i in range(self.config.denoising_steps):

            t = torch.ones((B, 1, 1), device=device) * (i + 1) / self.config.denoising_steps # shape: [B,1]
            
            # predict flow
            v = self.predict_flow(x, 
                                  t, 
                                  batch["observation.state"], 
                                  batch["observation.images"])
    
            x = x + dt * v 

            # # compute Runge Kutta terms
            # k1 = self.predict_flow(x, t, batch["observation.state"], batch["observation.images"])
            # k2 = self.predict_flow(x + 0.5 * dt * k1, t + 0.5 * dt, batch["observation.state"], batch["observation.images"])
            # k3 = self.predict_flow(x + 0.5 * dt * k2, t + 0.5 * dt, batch["observation.state"], batch["observation.images"])
            # k4 = self.predict_flow(x + dt * k3, t + dt, batch["observation.state"], batch["observation.images"])
            
            # # weighted average
            # x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4) 

        return x # [B, N_chunk, act_dim]
    

class ObsProjector(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim), 
        )
        
    def forward(self, obs):
        return self.mlp(obs) 


class VisionProjector(nn.Module):
    
    def __init__(self, replace_final_stride_with_dilation, vision_backbone, pretrained_backbone_weights, output_dim):
        
        super().__init__()

        # Backbone for image feature extraction
        backbone_model = getattr(torchvision.models, vision_backbone)(
            replace_stride_with_dilation=[False, False, replace_final_stride_with_dilation],
            weights=pretrained_backbone_weights,
            norm_layer=FrozenBatchNorm2d,
        )

        # Note: The assumption here is that we are using a ResNet model (and hence layer4 is the final
        # feature map). The forward method of this returns a dict: {"feature_map": output}.
        self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        self.img_projector = nn.Conv2d(
            backbone_model.fc.in_features, output_dim, kernel_size=1
        )

    def forward(self, obs): # [B*N_HIST, C, H, W]
        
        obs = self.backbone(obs)["feature_map"]
        obs = self.img_projector(obs)

        return obs # [B*N_HIST, dim_model, H', W']
        
    
class TransformerEncoder(nn.Module):   

    def __init__(self, dim_model, dim_feedforward_tf, nhead, num_layers):
        
        super(TransformerEncoder, self).__init__()
        
        self.pos_enc1D = PosEmbedding1D(dim_model)
        self.pos_enc2D = PosEmbedding2D()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_model)) 
        encoder_layers = nn.TransformerEncoderLayer(d_model=dim_model, nhead=nhead, dim_feedforward=dim_feedforward_tf, dropout=0.1, batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(dim_model)

    def forward(self, obs_features, all_cam_features): # , ee_embeds):

        # get dimensions
        B, n_hist, dim_model = obs_features.shape
        device = obs_features.device

        # add same time to all the position features of the same image
        time_indices = torch.arange(n_hist, dtype=torch.float32, device=device)
        time_enc = self.pos_enc1D.compute(time_indices) # [n_hist, d_model]
        
        # add positional encoding to observations to take into account history
        obs_features = obs_features + time_enc.unsqueeze(0)
        # ee_embeds = ee_embeds + enc1D

        # get dimensions
        B, n_hist, dim_model, H_prime, W_prime = all_cam_features[0].shape

        processed_all_cam_features = []

        for cam_features in all_cam_features: # loop over each camera

            # add spatial 2D positional encoding to images
            enc2D = self.pos_enc2D.compute(B, dim_model, H_prime, W_prime, device=device)
            cam_features = cam_features + enc2D.unsqueeze(1)  # [B, N_HIST, dim_model, H', W'] --> broadcast along N_HIST dimension

            # reshape tensor
            cam_features = cam_features.flatten(3).permute(0, 1, 3, 2)  # [B, N_HIST, H'*W', dim_model]

            # add positional encoding to each cam to take into account history (expand to match: [B, N_HIST, H'*W', dim_model])
            cam_features = cam_features + time_enc.view(1, n_hist, 1, dim_model)

            # final representation of the tensor
            cam_features = cam_features.reshape(B, n_hist * H_prime * W_prime, dim_model)  # [B, N_HIST*H'*W', dim_model]

            processed_all_cam_features.append(cam_features)

        img_features = torch.cat(processed_all_cam_features, dim=1) # [B, N_cam*N_HIST*H'*W', dim_model]

        # duplicate learnable CLS token for all samples in the batch (each sample in batch has its own CLS, and they all share same weights)
        cls_tokens = self.cls_token.repeat(B, 1, 1) # [B, 1, dim_model]

        # x = torch.cat([cls_tokens, obs_features, img_features, ee_embeds], dim=1) # [B, seq_length=1+n_hist+N_cam*N_HIST*H'*W', dim_model]
        x = torch.cat([cls_tokens, obs_features, img_features], dim=1) # [B, seq_length=1+n_hist+N_cam*N_HIST*H'*W', dim_model]

        # pass through transformer
        x = self.transformer_encoder(x) # [B, seq_length, dim_model]
        x = x[:,0,:]  # take only CLS token as summary --> [B, dim_model]

        # Layer normalization for stability
        out = self.final_norm(x)
        
        return out

# # FILM
# class FlowHead(nn.Module):     
    
#     def __init__(self, N_chunk, action_dim, dim_model, dim_feedforward_flow):
        
#         super().__init__()
        
#         in_dim = N_chunk*action_dim + dim_model  
#         out_dim = N_chunk*action_dim 

#         # FiLM modulation to handle time information
#         self.film_gen = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(dim_model, 2 * dim_model) # outputs both gamma & beta
#         )

#         self.mlp_flow = nn.Sequential(
#             nn.Linear(in_dim, dim_feedforward_flow),
#             nn.LayerNorm(dim_feedforward_flow), # since I concateneted different sources, better to re-normalize to N(0, I)
#             nn.SiLU(), # ReLU
#             nn.Linear(dim_feedforward_flow, dim_feedforward_flow),
#             nn.SiLU(), # ReLU
#             nn.Linear(dim_feedforward_flow, out_dim)
#         )

#         self.pos_enc1D = PosEmbedding1D(dim_model)

#     def forward(self, x_t, t, e):

#         B, N_chunk, act_dim = x_t.shape
        
#         # flatten
#         x_flat = x_t.reshape(B, -1)  # [B, N_chunk*act_dim]

#         # embed time with sinusoidal embedding + MLP
#         t = t.squeeze(1) # [B, 1]
#         t_embedded = self.pos_enc1D.compute(t) # [B, dim_model]

#         # generate gamma and beta
#         film_params = self.film_gen(t_embedded) # [B, 2 * dim_model]
#         gamma, beta = torch.chunk(film_params, 2, dim=-1) # each one is: [B, dim_model] 
        
#         # use t informations to scale 'e' tensor
#         e_modulated = e * (1 + gamma) + beta # [B, dim_model]

#         # concat along feature dim
#         inp = torch.cat([x_flat, e_modulated], dim=-1)  # [B, N_chunk*act_dim + dim_model]

#         # predict flows
#         out = self.mlp_flow(inp)  # [B, N_chunk*act_dim]

#         # reshape back to chunk
#         out = out.view(B, N_chunk, act_dim)  # [B, N_chunk, act_dim]

#         return out
    
# NORMAL
class FlowHead(nn.Module):     
    
    def __init__(self, N_chunk, action_dim, dim_model, dim_feedforward_flow):
        
        super().__init__()
        
        in_dim = N_chunk*action_dim + dim_model + dim_model  
        out_dim = N_chunk*action_dim 

        self.mlp_time = nn.Sequential(
            nn.Linear(dim_model, dim_model),
            nn.SiLU(),
            nn.Linear(dim_model, dim_model)
        )

        self.mlp_flow = nn.Sequential(
            nn.Linear(in_dim, dim_feedforward_flow),
            nn.LayerNorm(dim_feedforward_flow), # since I concateneted different sources, better to re-normalize to N(0, I)
            nn.SiLU(), # ReLU
            nn.Linear(dim_feedforward_flow, dim_feedforward_flow),
            nn.SiLU(), # ReLU
            nn.Linear(dim_feedforward_flow, out_dim)
        )

        self.pos_enc1D = PosEmbedding1D(dim_model)

    def forward(self, x_t, t, e):

        B, N_chunk, act_dim = x_t.shape
        
        # flatten
        x_t = x_t.reshape(B, -1)  # [B, N_chunk*act_dim]

        # embed time with sinusoidal embedding + MLP
        t = t.squeeze(1) # [B, 1]
        t_embedded = self.pos_enc1D.compute(t)
        t_embedded = self.mlp_time(t_embedded) # [B, dim_model]

        # concat along feature dim
        inp = torch.cat([x_t, t_embedded, e], dim=-1)  # [B, N_chunk*act_dim + dim_model + dim_model]

        # predict flows
        out = self.mlp_flow(inp)  # [B, N_chunk*act_dim]

        # reshape back to chunk
        out = out.view(B, N_chunk, act_dim)  # [B, N_chunk, act_dim]

        return out
    

class PosEmbedding1D():

    def __init__(self, d_model, temperature=10000):

        self.d_model = d_model
        self.temperature = temperature

    def compute(self, x):

        # x: [B, 1] (used in flow matching) o [seq_len] (used in transformers tokens)
        device = x.device
        half_dim = self.d_model // 2
        
        # exponent formula: 2 * i / d_model  with i in [0, half_dim]
        exponent = 2 * torch.arange(half_dim, dtype=torch.float32, device=device) / self.d_model
        inv_freq = 1.0 / (self.temperature ** exponent)
        
        if x.ndim == 1:
            x = x.unsqueeze(1) # [seq_len, 1] or [B, 1]
            
        # [seq_len, 1] * [1, half_dim] -> [seq_len, half_dim] or
        # [B, 1] * [1, half_dim] -> [B, half_dim]
        angle_rads = x * inv_freq.unsqueeze(0)
        
        # final embedding: [sin, sin, cos, cos, ..]
        return torch.cat([torch.sin(angle_rads), torch.cos(angle_rads)], dim=-1) # [seq_len or B, d_model]
    
class PosEmbedding2D():

    """
    2D sinusoidal positional embeddings logic
    """

    def __init__(self):

        # params
        self.temperature = 10000 # ratio for the geometric progression in sinusoid frequencies
    
    def compute(self, B, C, H, W, device=None):

        # Set device
        device = device or torch.device('cpu')

        # Params
        d_model = C // 2
        half_dim = d_model // 2
        
        # Compute inverse frequencies
        exponent = 2 * torch.arange(half_dim, dtype=torch.float32, device=device) / d_model
        inv_freq = 1.0 / (self.temperature ** exponent)

        # Create 2D positions
        y_pos = torch.arange(H, dtype=torch.float32, device=device)
        x_pos = torch.arange(W, dtype=torch.float32, device=device)

        # Compute angles
        y_angle = y_pos.unsqueeze(1) * inv_freq.unsqueeze(0) # (H, 1) * (1, half_dim) -> (H, half_dim)
        x_angle = x_pos.unsqueeze(1) * inv_freq.unsqueeze(0) # (W, 1) * (1, half_dim) -> (W, half_dim)

        # Generate embeddings with format [sin, sin, cos, cos])
        y_emb = torch.cat([y_angle.sin(), y_angle.cos()], dim=-1) # (H, d_model)
        x_emb = torch.cat([x_angle.sin(), x_angle.cos()], dim=-1) # (W, d_model)

        # Expand to H/W dimensions
        y_emb2d = y_emb.view(H, 1, d_model).expand(H, W, d_model) # (H, W, d_model)
        x_emb2d = x_emb.view(1, W, d_model).expand(H, W, d_model) # (H, W, d_model)

        # Concatenate along channel 
        pos_embed = torch.cat([y_emb2d, x_emb2d], dim=-1) # (H, W, C)

        # Add batch dim and permute 
        pos_embed = pos_embed.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)
        
        return pos_embed # (B,C,H,W)