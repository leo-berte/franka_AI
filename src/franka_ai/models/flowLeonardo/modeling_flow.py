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


# TODO:

# Use ADALN in EACH decoder layer instead of unique global FiLM
# capire se vision backbone ha già nel risultato spatial pos embeddings + change backbone


# =============================================================================
# POLICY CLASS DEFINITION
# =============================================================================

class FlowPolicy(PreTrainedPolicy):

    """
    Policy class implementing Flow-based imitation learning for robotic control.

    This is an high level class handling state/action normalization, loss computation
    during training and action generation during inference.

    Args:
        config (FlowConfig): Configuration instance containing model hyperparameters 
            and feature definitions.
        dataset_stats (dict, optional): Dataset statistics used for normalization. 
            If not provided, they must be loaded via `load_state_dict` before inference.
    """
    
    config_class = FlowConfig
    name = "flow_leonardo"

    def __init__(self,
                 config: FlowConfig,
                 dataset_stats: dict[str, dict[str, Tensor]] | None = None):

        super().__init__(config)
        config.validate_features()
        self.config = config

        # Normalization settings
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(config.output_features, config.normalization_mapping, dataset_stats)
        self.unnormalize_outputs = Unnormalize(config.output_features, config.normalization_mapping, dataset_stats)

        # Define policy model
        self.model = Flow(config)

        self.count_parameters()

        self.reset()

    def get_optim_params(self):

        """Mandatory method due to inheritance from 'PreTrainedPolicy'."""

        pass

    def count_parameters(self):
        
        """Print number of trainable parameters in the policy."""

        # Count total number of trainable parameters
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total number of trainable parameters: {params:,}")
    
        # Count number of trainable parameters in each sub-module
        for name, module in self.model.named_children():
            sub_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"  - {name}: {sub_params:,}")

    def reset(self):

        """This should be called whenever the environment is reset."""

        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def prepare_image_features(self, batch):

        """Prepare images batch data as expected by policy."""

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

            if self.config.head_type == "mlp":
                actions = self.model.generate_action_mlp(batch) # [1, N_chunk, act_dim]
            elif self.config.head_type in ["flow_matching_mlp", "flow_matching_unet", "flow_matching_transformer"]:
                actions = self.model.generate_action(batch) # [1, N_chunk, act_dim]
            else:
                raise NotImplementedError(f"Head type with name {self.config.head_type} is not implemented.")

            # slice actions
            actions = actions[:, :self.config.n_action_steps, :] # [1, n_action_steps, act_dim]

            # unnormalize outputs
            actions = self.unnormalize_outputs({"action": actions})["action"]

            # The queue has shape (n_action_steps, batch_size, *), hence the transpose
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:

        """
        Run the batch through the model and compute the loss.
        
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
        if self.config.head_type == "mlp":
            loss = self.model.compute_loss_mlp(batch)
        elif self.config.head_type in ["flow_matching_mlp", "flow_matching_unet", "flow_matching_transformer"]:
            loss = self.model.compute_loss(batch)
        else:
            raise NotImplementedError(f"Head type with name {self.config.head_type} is not implemented.")
        
        # no output_dict so returning None
        return loss, None


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class Flow(nn.Module):

    """
    Core architecture for the Flow-based policy.

    This class serves as the model container, integrating multi-modal feature extraction and a conditional Flow Matching head.
    """

    def __init__(self, config: FlowConfig):

        super().__init__()
        self.config = config
        self.use_cls_token = self.config.head_type != "flow_matching_transformer"

        if self.config.robot_state_feature:
            self.obs_projector = ObsProjector(input_dim=self.config.robot_state_feature.shape[0], 
                                              output_dim=self.config.dim_model)

        if self.config.image_features:
            self.vision_projector = VisionProjector(replace_final_stride_with_dilation=self.config.replace_final_stride_with_dilation, 
                                                    vision_backbone=self.config.vision_backbone, 
                                                    pretrained_backbone_weights=self.config.pretrained_backbone_weights, 
                                                    output_dim=self.config.dim_model)

        self.encoder = TransformerEncoder(dim_model=self.config.dim_model, 
                                          dim_feedforward=self.config.dim_feedforward_enc,
                                          nhead=self.config.nhead_enc,
                                          num_layers=self.config.num_layers_enc,
                                          use_cls_token=self.use_cls_token)
        
        if self.config.head_type == "mlp":

            self.mlp_head = MlpHead(N_chunk=self.config.N_chunk, 
                                    action_dim=self.config.action_feature.shape[0], 
                                    dim_model=self.config.dim_model)
            
        elif self.config.head_type == "flow_matching_mlp":

            self.flow_head = FlowHeadMlp(action_dim=self.config.action_feature.shape[0], 
                                         dim_model=self.config.dim_model,
                                         timestep_embed_dim=self.config.timestep_embed_dim,
                                         use_film=self.config.use_film)
            
        elif self.config.head_type == "flow_matching_unet":

            self.flow_head = FlowHeadUNet(action_dim=self.config.action_feature.shape[0], 
                                          dim_model=self.config.dim_model, 
                                          timestep_embed_dim=self.config.timestep_embed_dim, 
                                          use_film=self.config.use_film,
                                          down_dims=self.config.down_dims,
                                          kernel_size=self.config.kernel_size,
                                          n_groups=self.config.n_groups,
                                          )
            
        elif self.config.head_type == "flow_matching_transformer":

            self.flow_head = FlowHeadTransformerDecoder(action_dim=self.config.action_feature.shape[0],
                                                        dim_model=self.config.dim_model, 
                                                        use_film=self.config.use_film,
                                                        timestep_embed_dim=self.config.timestep_embed_dim,
                                                        dim_feedforward=self.config.dim_feedforward_dec, 
                                                        nhead=self.config.nhead_dec, 
                                                        num_layers=self.config.num_layers_dec)
                                                    
        
    def global_conditioning(self,  batch: dict[str, Tensor]) -> Tensor | None:

        """
        Processes and aggregates multi-modal observations into a global conditioning latent.

        This method encodes robot states and multiple camera streams using their respective 
        projectors. It then fuses these features through a Transformer encoder. 
        Depending on the head type, it returns either the full sequence of embeddings or 
        a compressed summary token.

        Args:
            batch (dict): Dictionary containing 'observation.state' and 'observation.images'.
        Returns:
            Tensor: Global conditioning feature of shape (B, seq_length, dim_model) 
                    or (B, dim_model).
        """

        all_cam_features = []

        # obs projector
        if self.config.robot_state_feature:
            batch_obs = batch["observation.state"]
            obs_features = self.obs_projector(batch_obs) # [B, N_HISTORY, dim_model]

        # vision projector
        if self.config.image_features:

            batch_images = batch["observation.images"]
            
            # Create a list of cameras
            for images in batch_images: # [B, N_HIST, C, H, W]

                B, N_HIST, C, H, W = images.shape
                images = images.view(B * N_HIST, C, H, W)

                cam_features = self.vision_projector(images) # [B*N_HIST, dim_model, H', W']

                _, dim_model, H_prime, W_prime = cam_features.shape
                cam_features = cam_features.view(B, N_HIST, dim_model, H_prime, W_prime) # [B, N_HIST, dim_model, H', W']

                all_cam_features.append(cam_features) # each element: [B, N_HIST, dim_model, H', W']

        # transformer embedding
        glob_cond = self.encoder(obs_features, all_cam_features) # [B, seq_length, dim_model]

        # Eventually reduce dims to: [B, dim_model]
        if (self.use_cls_token == True):
            glob_cond = glob_cond[:,-1,:]  # take only CLS token as summary
            # glob_cond = torch.mean(glob_cond, dim=1) # mean instead of CLS
        
        return glob_cond # [B, N_HIST, dim_model] or [B, dim_model]
    
    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor | None:

        """
        Computes the Flow Matching loss for training.

        The method implements the Flow Matching objective by:
        1. Sampling Gaussian noise as the source distribution.
        2. Defining a linear probability path (Optimal Transport) toward the ground truth actions.
        3. Sampling time 't' using a Logit-Normal distribution to focus training on non-terminal regions.
        4. Predicting the velocity vector field 'v_pred' conditioned on the noisy actions, 
        time, and global observation context.
        
        The loss is calculated as the Mean Squared Error between the predicted and target 
        velocities, masked to exclude padded steps.

        Args:
            batch (dict): Dictionary containing 'action', 'action_is_pad', and observations.
        Returns:
            Tensor: MSE loss for the velocity field.
        """

        # get device type
        device = batch["action"].device

        # sample random noise for source distribution (uniform or gaussian)
        B, N_chunk, act_dim = batch["action"].shape
        # action_chunk_src = torch.rand(B, N_chunk, act_dim, device=device) # uniform
        action_chunk_src = torch.randn(B, N_chunk, act_dim, device=device) # gaussian

        # target flow (ground truth vector field)
        v_target = batch["action"] - action_chunk_src # [B, N_chunk, act_dim] 

        # # sample random time t ~ U(0,1) --> BASELINE
        # t = torch.rand(B, 1, 1, device=device)  # each sample has its own time

        # Logit-Normal Sampling: sample t around the middle, to avoid sampling frequently near borders 0.0 & 1.0
        m = torch.randn(B, 1, 1, device=device) # gaussian distribution
        sigma = 1.0 # the lower this param, the more the values will be around 0.5
        t = torch.sigmoid(m * sigma) # result will be again in [0,1] and shape: [B, 1, 1]

        # # Beta Sampling
        # alpha, beta, s = 1.0, 1.5, 0.999
        # beta_dist = torch.distributions.Beta(torch.tensor([alpha], device=device), torch.tensor([beta], device=device))
        # u = beta_dist.sample([B]).view(B, 1, 1)
        # t = s * u # [B, 1, 1]

        # interpolation toward noise
        x_t = action_chunk_src + t * v_target # [B, N_chunk, act_dim]

        # get global conditioning
        glob_cond = self.global_conditioning(batch) # [B, dim_model]
        
        # predict flow
        v_pred = self.flow_head(x_t, t, glob_cond) # [B, N_chunk, act_dim]
        
        # Flow Matching loss
        loss = F.mse_loss(v_pred, v_target, reduction="none") # [B, N_chunk, act_dim]

        # select valid timestamps only
        in_episode_bound = ~batch["action_is_pad"] # [B, N_chunk]
        loss = (loss * in_episode_bound.unsqueeze(-1))

        # # give less weight for small values of time t, since prediction is less relevant there
        # weight = t / (t + 0.1)
        # loss = (loss * weight)

        return loss.mean()
    
    def compute_loss_mlp(self, batch: dict[str, Tensor]) -> Tensor | None:

        """
        Computes the standard Mean Squared Error loss for the MLP baseline.

        This method bypasses the iterative Flow Matching process and directly predicts 
        the entire action chunk from the global conditioning latent. It is used for 
        baseline comparisons only.

        Args:
            batch (dict): Dictionary containing 'action', 'action_is_pad', and observations.
        Returns:
            Tensor: MSE loss between predicted and ground truth actions.
        """

        # get global conditioning
        glob_cond = self.global_conditioning(batch) # [B, dim_model]
        
        # real actions
        actions_pred = self.mlp_head(glob_cond) # [B, N_chunk, act_dim]

        # target actions
        actions_target = batch["action"] # [B, N_chunk, act_dim] 
        
        # Flow Matching loss
        loss = F.mse_loss(actions_pred, actions_target, reduction="none") # [B, N_chunk, act_dim]

        # select valid timestamps only
        in_episode_bound = ~batch["action_is_pad"] # [B, N_chunk]
        loss = (loss * in_episode_bound.unsqueeze(-1))

        return loss.mean()

    def generate_action(self, batch: dict[str, Tensor]) -> Tensor | None:

        """
        Generates an action sequence by integrating the predicted velocity field.

        This method performs inference by:
        1. Sampling initial noise from the source distribution (Gaussian or Uniform).
        2. Encoding current observations into a global conditioning latent.
        3. Solving the Probability Flow ODE using the specified numerical integrator

        Args:
            batch (dict): Dictionary of current environment observations.
        Returns:
            Tensor: Generated action chunk of shape (B, N_chunk, act_dim).
        """
        
        # get device type
        device = batch["observation.state"].device

        # get dimensions
        B = batch["observation.state"].shape[0]
        N_chunk = self.config.N_chunk
        act_dim = self.config.action_feature.shape[0]

        # decide whether to start always from same source distribution during inference
        seed = 42 if self.config.use_fixed_src_dist else None
        gen = torch.Generator(device=device).manual_seed(seed) if seed else None

        # start from noise (gaussian or uniform)
        # x = torch.rand(B, N_chunk, act_dim, device=device, generator=gen) # uniform
        x = torch.randn(B, N_chunk, act_dim, device=device, generator=gen) # gaussian

        # get global conditioning
        glob_cond = self.global_conditioning(batch)

        # integrate flow
        dt = 1.0 / self.config.denoising_steps

        for i in range(self.config.denoising_steps):

            t = torch.ones((B, 1, 1), device=device) * (i) / self.config.denoising_steps # shape: [B,1]
            
            if (self.config.ode_type == "euler"):
                # predict flow
                v = self.flow_head(x, t, glob_cond) # [B, N_chunk, act_dim] 
                # integrate
                x = x + dt * v 
            elif (self.config.ode_type == "runge_kutta"):
                # compute Runge Kutta terms
                k1 = self.flow_head(x, t, glob_cond)
                k2 = self.flow_head(x + 0.5 * dt * k1, t + 0.5 * dt, glob_cond)
                k3 = self.flow_head(x + 0.5 * dt * k2, t + 0.5 * dt, glob_cond)
                k4 = self.flow_head(x + dt * k3, t + dt, glob_cond)
                # weighted average
                x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4) 
            else:
                raise NotImplementedError(f"ODE type with name {self.config.ode_type} is not implemented.")

        return x # [B, N_chunk, act_dim]
    
    def generate_action_mlp(self, batch: dict[str, Tensor]) -> Tensor | None:

        """
        Generates an action sequence in a single forward pass using the MLP head.

        This method provides a non-iterative alternative to Flow Matching. 
        It is used for baseline comparisons only.

        Args:
            batch (dict): Dictionary of current environment observations.
        Returns:
            Tensor: Predicted action chunk of shape (B, N_chunk, act_dim).
        """

        # get global conditioning
        glob_cond = self.global_conditioning(batch)

        return self.mlp_head(glob_cond) # [B, N_chunk, act_dim] 


# =============================================================================
# SUBMODULES
# =============================================================================

class ObsProjector(nn.Module):

    """Projector for robot proprioceptive states."""
    
    def __init__(self, 
                 input_dim, 
                 output_dim):
        
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim), 
        )
        
    def forward(self, obs):
        return self.mlp(obs) 


class VisionProjector(nn.Module):

    """Image feature extractor and projection layer for visual observations."""
    
    """
    Nella tua TransformerEncoder, tu concateni tutto: [CLS, obs, cam1_hist, cam2_hist].
    Se hai N_HIST = 5, 2 camere e mappe 7x7, la tua sequenza è:
    $1 + 5 + (2 \times 5 \times 49) = 496$ token.

    A. Cambia il modo in cui tratti le immagini (Bottleneck)
    Invece di mandare TUTTI i patch al Transformer, dovresti comprimerli PRIMA.
    Opzione 1 (LeRobot style): Usa lo Spatial Softmax dopo la VisionProjector. 
    Invece di 49 patch per immagine, avresti solo 32-64 numeri (coordinate keypoints). 
    La sequenza passerebbe da 496 token a circa 15-20 token. Questo eliminerebbe il jitter istantaneamente.
    Opzione 2 (Pooling): Fai un Global Average Pooling sulle mappe 7x7 prima di darle al Transformer.

    # Use CLIP-pretrained ViT-B/32 encoder [29]. Images are resized to 224×224 pixels --> See force UMI gripper for augmentations params values also
    """

    def __init__(self, 
                 replace_final_stride_with_dilation, 
                 vision_backbone, 
                 pretrained_backbone_weights, 
                 output_dim):
        
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

    def forward(self, obs):

        """
        Extracts and projects spatial features from a batch of images.

        Args:
            obs (Tensor): Batch of images with shape (B * N_HISTORY, C, H, W).
        Returns:
            Tensor: Spatial feature map of shape (B * N_HISTORY, output_dim, H', W').
        """
        
        obs = self.backbone(obs)["feature_map"]
        obs = self.img_projector(obs)

        return obs # [B*N_HIST, dim_model, H', W']
        
    
class TransformerEncoder(nn.Module):   

    """
    Transformer Encoder for multi-modal sensor fusion.

    This module integrates proprioceptive state features and multi-camera visual features 
    into a unified latent representation. It employs a hybrid positional encoding strategy:
    1. 1D Positional Embeddings to encode temporal history across observations.
    2. 2D Positional Embeddings to preserve spatial structure within image feature maps.
    
    The encoder processes these inputs as a sequence of tokens, including eventually a learnable 
    [CLS] token that serves as a global summary of the multi-modal context. The 
    resulting embeddings are used to condition the downstream action generation heads.

    Args:
        dim_model (int): The internal dimension of the transformer embeddings.
        dim_feedforward (int): Dimension of the hidden layer in the MLP network.
        nhead (int): Number of attention heads.
        num_layers (int): Number of transformer encoder layers.
        use_cls_token (bool): Flag to use or not CLS token.
    """

    def __init__(self, 
                 dim_model, 
                 dim_feedforward, 
                 nhead, 
                 num_layers,
                 use_cls_token):
        
        super().__init__()
        
        # params
        self.use_cls_token = use_cls_token

        self.pos_enc1D = PosEmbedding1D(dim_model)
        self.pos_enc2D = PosEmbedding2D()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_model)) if (self.use_cls_token == True) else None
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=0.1, batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(dim_model)

    def forward(self, obs_features, all_cam_features):

        """
        Fuses state and vision features into a sequence of latent tokens.

        Args:
            obs_features (Tensor): Proprioceptive features of shape (B, N_HIST, D).
            all_cam_features (list[Tensor]): List of camera feature maps, 
                each of shape (B, N_HIST, D, H', W').

        Returns:
            Tensor: Refined token sequence of shape (B, seq_length, D).
        """

        # get dimensions
        B, n_hist, dim_model = obs_features.shape
        device = obs_features.device

        inputs_list = []

        # add same time to all the position features of the same image
        time_indices = torch.arange(n_hist, dtype=torch.float32, device=device)
        time_enc = self.pos_enc1D.compute(time_indices) # [n_hist, d_model]
        
        # add positional encoding to observations to take into account history
        obs_features = obs_features + time_enc.unsqueeze(0) # [B, N_h, D]
        inputs_list.append(obs_features)

        # take into account vision
        if all_cam_features:

            # get dimensions
            B, n_hist, dim_model, H_prime, W_prime = all_cam_features[0].shape

            processed_all_cam_features = []

            for cam_features in all_cam_features: # loop over each camera

                # print("cam_features: ", cam_features.shape)

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

            # print("img_features: ", img_features.shape)

            inputs_list.append(img_features)

        # # DEBUG: REMOVE TRANSFORMER
        # return torch.cat(inputs_list, dim=1).flatten(1) # [B, D]

        # add a learnable CLS token as sumup
        if (self.use_cls_token == True):
            cls_token = self.cls_token.repeat(B, 1, 1) # [B, 1, dim_model]
            inputs_list.append(cls_token)

        # compose all the tokens as input for transformer
        x = torch.cat(inputs_list, dim=1) # [B, seq_length=1+n_hist+N_cam*N_HIST*H'*W', dim_model]

        # --> x:  torch.Size([B, 22, 512]) con img [120, 160] e N_h = 1 e cam=1
        # --> x:  torch.Size([B, 130, 512]) con img [300, 500] e N_h = 1 e cam=1

        # pass through transformer
        x = self.transformer_encoder(x) # [B, seq_length, dim_model]

        # layer normalization for stability
        out = self.final_norm(x)
        
        return out # [B, seq_length, dim_model]


# =============================================================================
# BASIC MLP HEAD
# =============================================================================

class MlpHead(nn.Module):     

    """Simple MLP head."""
    
    def __init__(self, 
                 N_chunk, 
                 action_dim, 
                 dim_model):
        
        super().__init__()

        self.N_chunk = N_chunk
        self.action_dim = action_dim
        
        in_dim = dim_model 
        out_dim = N_chunk*action_dim 

        # Encoder for the timestep
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, dim_model * 4),
            nn.Mish(),
            # nn.Linear(in_dim * 4, in_dim * 4),
            # nn.Mish(), # ReLU SiLU
            nn.Linear(dim_model * 4, out_dim)
        )

    def forward(self, e):

        # predict flows
        out = self.mlp(e) # [B, N_chunk*act_dim]

        # reshape back to chunk
        out = out.view(out.shape[0], self.N_chunk, self.action_dim)  # [B, N_chunk, act_dim]

        return out


# =============================================================================
# FLOW MLP HEAD
# =============================================================================
    
class FlowHeadMlp(nn.Module):     

    """
    MLP-based head for Conditional Flow Matching.

    This module predicts the velocity vector field 'v' using a deep Multi-Layer Perceptron. 
    It processes action chunks by lifting them into a latent space and adding temporal 
    positional embeddings to the sequence. Then it injects global context (observations 
    and diffusion time 't'). 
    
    The conditioning can be injected either through simple additive bias or via 
    FiLM (Feature-wise Linear Modulation) for more expressive control over the 
    latent features.

    Args:
        action_dim (int): Dimension of the action space.
        dim_model (int): Internal latent dimension for action and conditioning features.
        timestep_embed_dim (int): Dimension for the sinusoidal time embedding.
        use_film (bool): If True, applies FiLM (scale and bias) modulation; 
            otherwise, uses additive bias.
    """
    
    def __init__(self, 
                 action_dim, 
                 dim_model, 
                 timestep_embed_dim, 
                 use_film):
        
        super().__init__()

        # params
        self.use_film = use_film
        self.cond_dim_tot = timestep_embed_dim + dim_model
        self.dim_model = dim_model
        self.in_dim = dim_model
        self.out_dim = action_dim 

        # used to add sinusoidal embedding to time
        self.pos_enc1D_actions = PosEmbedding1D(dim_model) 

        self.mlp_actions = nn.Sequential(
            nn.Linear(action_dim, dim_model),
            nn.Mish(),
            nn.Linear(dim_model, dim_model)
        )

        # used to add sinusoidal embedding to time
        self.pos_enc1D_time = PosEmbedding1D(timestep_embed_dim) 

        self.mlp_time = nn.Sequential(
            nn.Linear(timestep_embed_dim, timestep_embed_dim * 4),
            nn.Mish(),
            nn.Linear(timestep_embed_dim * 4, timestep_embed_dim)
        )

        # I may use FiLM to combine conditioning with 'x' tensor
        cond_channels = dim_model * 2 if use_film else dim_model
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(self.cond_dim_tot, cond_channels)
        )

        self.mlp_flow = nn.Sequential(
            nn.LayerNorm(self.in_dim), # since I concateneted different sources, better to re-normalize to N(0, I)
            nn.Linear(self.in_dim, self.in_dim * 2), 
            nn.Mish(), # ReLU              
            nn.Linear(self.in_dim * 2, self.in_dim * 4),
            nn.Mish(), # ReLU 
            nn.Linear(self.in_dim * 4, self.in_dim * 2),
            nn.Mish(), # ReLU              
            nn.Linear(self.in_dim * 2, self.out_dim)
        )

    def forward(self, x, t, e):

        """
        Predicts the flow velocity given actions, time and global conditioning.

        Args:
            x (Tensor): Current noisy action chunk of shape (B, N_chunk, action_dim).
            t (Tensor): Diffusion time steps of shape (B, 1, 1).
            e (Tensor): Global observation features of shape (B, dim_model).

        Returns:
            Tensor: Predicted velocity field 'v' of shape (B, N_chunk, action_dim).
        """

        B, N_chunk, act_dim = x.shape  
        device = x.device

        # MLP + embed N_chunk with sinusoidal embedding
        x = self.mlp_actions(x) # [B, N_chunk, dim_model]
        chunk_indices = torch.arange(N_chunk, dtype=torch.float32, device=device)
        chunk_embedded = self.pos_enc1D_actions.compute(chunk_indices) # [N_c, d_model]
        x = x + chunk_embedded.unsqueeze(0) # [B, N_c, d_model]
        
        # embed time with sinusoidal embedding + MLP
        t = t.squeeze(-1) # [B, 1]
        t_embedded = self.pos_enc1D_time.compute(t) # [B, timestep_embed_dim]
        t_embedded = self.mlp_time(t_embedded) # [B, timestep_embed_dim]
            
        cond = torch.cat([t_embedded, e], dim=-1)  # [B, cond_dim_tot]

        # wisely combine conditioning and actions
        cond_embed = self.cond_encoder(cond).unsqueeze(1) # [B, 1, dim_model] or [B, 1, 2*dim_model]

        if self.use_film:
            # treat the embedding as a list of scales and biases
            scale = cond_embed[:, :, :self.dim_model]
            bias = cond_embed[:, :, self.dim_model:]
            x = scale * x + bias  # [B, N_chunk, dim_model]
        else:
            # treat the embedding as biases
            x = x + cond_embed  # [B, N_chunk, dim_model]
        
        # predict flow
        out = self.mlp_flow(x)  # [B, N_chunk, act_dim]

        return out
    

# =============================================================================
# FLOW U-NET 1D HEAD
# =============================================================================

class Conv1dBlock(nn.Module):
    
    """
    Utility block: Conv1d --> GroupNorm --> Mish.
    """
    
    def __init__(self, 
                 inp_channels, 
                 out_channels, 
                 kernel_size=3, 
                 n_groups=8):

        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)
    

class ConditionalResidualBlock1d(nn.Module):
    
    """
    Residual block with conditioning injection for 1D sequences.

    It processes action chunks and modulates its features
    based on a global conditioning vector (combined time and observation embeddings). 
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        cond_dim (int): Dimension of the global conditioning vector.
        kernel_size (int): Size of the convolutional kernel. Default: 3.
        n_groups (int): Number of groups for Group Normalization. Default: 8.
        use_film (bool): If True, applies FiLM modulation; otherwise, applies additive bias.
    """

    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 cond_dim, 
                 kernel_size=3, 
                 n_groups=8, 
                 use_film=False):
        
        super().__init__()

        # params
        self.use_film = use_film
        self.out_channels = out_channels

        self.conv1 = Conv1dBlock(in_channels, out_channels, kernel_size, n_groups)

        # I may use FiLM to combine conditioning with 'x' tensor
        cond_channels = out_channels * 2 if use_film else out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels)
        )

        self.conv2 = Conv1dBlock(out_channels, out_channels, kernel_size, n_groups)
        
        # shortcut in case dimensions doesn't match
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):

        """
        Forward pass with conditional feature modulation.

        Args:
            x (Tensor): Input sequence tensor of shape (B, in_channels, T).
            cond (Tensor): Global conditioning tensor of shape (B, cond_dim).
        Returns:
            Tensor: Modulated and residual-summed output of shape (B, out_channels, T).
        """

        out = self.conv1(x) # [B, out_channels, N_chunk]
        
        # wisely combine conditioning and actions
        cond_embed = self.cond_encoder(cond).unsqueeze(-1) # [B, out_channels, 1] or [B, 2*out_channels, 1]

        if self.use_film:
            # treat the embedding as a list of scales and biases
            scale = cond_embed[:, : self.out_channels]
            bias = cond_embed[:, self.out_channels :]
            out = scale * out + bias  # [B, out_channels, N_chunk]
        else:
            # treat the embedding as biases
            out = out + cond_embed  # [B, out_channels, N_chunk]

        out = self.conv2(out) # [B, out_channels, N_chunk]
        
        # residual connection
        return out + self.residual_conv(x) # [B, out_channels, N_chunk]


class FlowHeadUNet(nn.Module):
    
    """
        1D Convolutional UNet with global conditioning for trajectory generation.

        This architecture follows a classic U-shaped structure (encoder-bottleneck-decoder) 
        designed to process 1D temporal sequences (action chunks). It utilizes 
        Conditional Residual Blocks to inject a combined context of time 
        and multi-modal observations at every level.

        Args:
            action_dim (int): Dimension of the action vector.
            dim_model (int): Dimension of the observation conditioning vector.
            timestep_embed_dim (int): Dimension for sinusoidal time embedding.
            use_film (bool): Whether to use FiLM modulation in residual blocks.
            down_dims (list[int]): List of feature dimensions for each UNet level.
            kernel_size (int): Convolutional kernel size.
            n_groups (int): Group count for Group Normalization layers.
        """

    def __init__(self, 
                 action_dim, 
                 dim_model, 
                 timestep_embed_dim, 
                 use_film,
                 down_dims,
                 kernel_size,
                 n_groups):
        
        super().__init__()

        # add sinusoidal embedding to time
        self.pos_enc1D = PosEmbedding1D(timestep_embed_dim) 

        # Encoder for the timestep.
        self.mlp_time = nn.Sequential(
            nn.Linear(timestep_embed_dim, timestep_embed_dim * 4),
            nn.Mish(),
            nn.Linear(timestep_embed_dim * 4, timestep_embed_dim),
        )

        # The FiLM conditioning dimension.
        cond_dim = timestep_embed_dim + dim_model

        # In channels / out channels for each downsampling block in the Unet's encoder. 
        # For the decoder, we just reverse these. 
        in_out = [(action_dim, down_dims[0])] + list(
            zip(down_dims[:-1], down_dims[1:], strict=True)
        ) # [(act_dim, down_dims[0]),(down_dims[0], down_dims[1]),(down_dims[1], down_dims[2])]

        # Unet encoder
        common_res_block_kwargs = {
            "cond_dim": cond_dim,
            "kernel_size": kernel_size,
            "n_groups": n_groups,
            "use_film": use_film
        }

        # For each level in the Unet, I perform the package of operations contained in ModuleList

        self.down_modules = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(in_out): # --> [(7, 256), (256, 512), (512, 1024)]
            
            is_last = ind >= (len(in_out) - 1)

            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1d(dim_in, dim_out, **common_res_block_kwargs), # augment number of features, same channels (time)
                        ConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs), # same number of features, same channels (time)
                        # Downsample as long as it is not the last block
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(), # same number of features, half channels (time)
                    ]
                )
            )

        # Processing in the middle of the auto-encoder

        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1d(
                    down_dims[-1], down_dims[-1], **common_res_block_kwargs
                ),
                ConditionalResidualBlock1d(
                    down_dims[-1], down_dims[-1], **common_res_block_kwargs
                ),
            ]
        )

        # Unet decoder

        self.up_modules = nn.ModuleList([])

        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])): # --> [(512, 1024), (256, 512)]

            is_last = ind >= (len(in_out) - 1)

            self.up_modules.append(
                nn.ModuleList(
                    [
                        # dim_in * 2, because it takes the encoder's skip connection as well
                        ConditionalResidualBlock1d(dim_in * 2, dim_out, **common_res_block_kwargs),
                        ConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        # Upsample as long as it is not the last block.
                        nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )
 
        self.final_conv = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size=kernel_size),
            nn.Conv1d(down_dims[0], action_dim, 1),
        )

    def forward(self, x, timestep, global_cond):
        
        """
        Predicts the flow velocity field through the UNet architecture.

        Args:
            x (Tensor): Input action chunk of shape (B, N_chunk, action_dim).
            timestep (Tensor): Current flow time steps of shape (B, 1, 1).
            global_cond (Tensor): Multi-modal context of shape (B, dim_model).

        Returns:
            Tensor: Predicted velocity field 'v' of shape (B, N_chunk, action_dim).
        """

        # for 1D convolutions we'll need feature dimension first.
        x = x.permute(0,2,1)  # [B, act_dim, N_c]

        # add pos encoding to time + MLP
        timestep = timestep.squeeze(-1) # [B, 1]
        t_embedded = self.pos_enc1D.compute(timestep) # [B, dim_model]
        timesteps_embed = self.mlp_time(t_embedded)

        # concatenate global conditioning to the timestep embedding
        global_feature = torch.cat([timesteps_embed, global_cond], axis=-1)

        # run encoder, keeping track of skip features to pass to the decoder
        encoder_skip_features: list[Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            encoder_skip_features.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # run decoder, using the skip features from the encoder
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, encoder_skip_features.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x) # [B, act_dim, N_c]

        return x.permute(0,2,1)  # [B, N_c, act_dim]


# =============================================================================
# FLOW TRANSFORMER DECODER HEAD
# =============================================================================

class FlowHeadTransformerDecoder(nn.Module):   

    """
    Transformer Decoder-based head for Conditional Flow Matching.

    This module treats action generation as a sequence modeling task. It uses a 
    Transformer Decoder where the noisy action chunk acts as the 'target' (tgt) 
    and the global observation context acts as the 'memory'. 
    
    The architecture integrates:
    1. Sinusoidal embeddings for both action chunk indices and diffusion time 't'.
    2. FiLM or additive modulation to inject time-dependency into the action latent.
    3. Cross-attention mechanism to allow the action sequence to dynamically 
       query relevant information from the multi-modal observation tokens.

    Args:
        action_dim (int): Dimension of the action vector.
        dim_model (int): Hidden dimension of the transformer.
        use_film (bool): Whether to use FiLM modulation for time injection.
        timestep_embed_dim (int): Dimension for time embedding.
        dim_feedforward (int): Hidden dimension of the decoder's MLP sub-layers.
        nhead (int): Number of cross-attention heads.
        num_layers (int): Number of transformer decoder layers.
    """

    def __init__(self, 
                 action_dim,
                 dim_model, 
                 use_film,
                 timestep_embed_dim,
                 dim_feedforward, 
                 nhead, 
                 num_layers):
        
        super().__init__()

        # params
        self.use_film = use_film
        self.dim_model = dim_model

        # used to add sinusoidal embedding to time
        self.pos_enc1D_actions = PosEmbedding1D(dim_model) 

        self.mlp_actions = nn.Sequential(
            nn.Linear(action_dim, dim_model),
            nn.Mish(),
            nn.Linear(dim_model, dim_model)
        )

        # used to add sinusoidal embedding to time
        self.pos_enc1D_time = PosEmbedding1D(timestep_embed_dim) 

        self.mlp_time = nn.Sequential(
            nn.Linear(timestep_embed_dim, timestep_embed_dim * 4),
            nn.Mish(),
            nn.Linear(timestep_embed_dim * 4, timestep_embed_dim)
        )

        # I may use FiLM to combine conditioning with 'x' tensor
        cond_channels = dim_model * 2 if use_film else dim_model
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(timestep_embed_dim, cond_channels)
        )

        # decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=0.1, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.mlp_final = nn.Sequential(    
            nn.Linear(dim_model, action_dim)
        )

    def forward(self, x, t, e):

        """
        Predicts the flow velocity field using cross-attention over observations.

        Args:
            x (Tensor): Noisy action chunk of shape (B, N_chunk, action_dim).
            t (Tensor): Diffusion time steps of shape (B, 1, 1).
            e (Tensor): Observation tokens/memory of shape (B, seq_len, dim_model).

        Returns:
            Tensor: Predicted velocity field 'v' of shape (B, N_chunk, action_dim).
        """

        B, N_chunk, act_dim = x.shape  
        device = x.device

        # MLP + embed N_chunk with sinusoidal embedding
        x = self.mlp_actions(x) # [B, N_chunk, dim_model]
        chunk_indices = torch.arange(N_chunk, dtype=torch.float32, device=device)
        chunk_embedded = self.pos_enc1D_actions.compute(chunk_indices) # [N_c, d_model]
        x = x + chunk_embedded.unsqueeze(0) # [B, N_c, d_model]
        
        # embed time with sinusoidal embedding + MLP
        t = t.squeeze(-1) # [B, 1]
        t_embedded = self.pos_enc1D_time.compute(t) # [B, timestep_embed_dim]
        t_embedded = self.mlp_time(t_embedded) # [B, timestep_embed_dim]
            
        # wisely combine conditioning and actions
        cond_embed = self.cond_encoder(t_embedded).unsqueeze(1) # [B, 1, dim_model] or [B, 1, 2*dim_model]

        if self.use_film:
            # treat the embedding as a list of scales and biases
            scale = cond_embed[:, :, :self.dim_model]
            bias = cond_embed[:, :, self.dim_model:]
            x = scale * x + bias  # [B, N_chunk, dim_model]
        else:
            # treat the embedding as biases
            x = x + cond_embed  # [B, N_chunk, dim_model]

        # predict flow
        out = self.transformer_decoder(tgt=x, memory=e)  # [B, N_chunk, dim_model]

        # final layer
        out = self.mlp_final(out)  # [B, N_chunk, act_dim]

        return out


# =============================================================================
# POSITIONAL ENCODINGS
# =============================================================================

class PosEmbedding1D():

    """
    1D Sinusoidal Positional Embedding.

    This class implements the fixed-frequency sinusoidal encoding used to provide 
    positional information to transformers or to embed continuous scalar values.
    
    By mapping a single scalar to a set of sine and cosine functions of varying 
    frequencies, it allows the network to learn relative relationships between 
    different positions or timesteps in a high-dimensional embedding space.

    Args:
        d_model (int): The dimension of the output embedding. Must be an even number.
        temperature (int): The base of the geometric progression for frequencies. 
    """

    def __init__(self, d_model, temperature=10000):

        self.d_model = d_model
        self.temperature = temperature

    def compute(self, x):

        """
        Computes the positional encoding for a given input tensor. 
        The method handles both discrete sequence indices and continuous scalar values. 

        Args:
            x (Tensor): Input tensor of shape (N,) or (B, 1), representing 
                sequence indices or continuous timesteps.
        Returns:
            Tensor: Positional embeddings of shape (N, d_model) or (B, d_model).
        """

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
        return torch.cat([torch.sin(angle_rads), torch.cos(angle_rads)], dim=-1) # [seq_len, d_model] or [B, d_model]
    
class PosEmbedding2D():

    """
    2D Sinusoidal Positional Embeddings for spatial feature maps.

    This module provides a fixed spatial coordinate system for visual features. 
    It decomposes the 2D grid into independent vertical (Y) and horizontal (X) 
    sinusoidal components, which are then concatenated to form a unique 
    embedding for every pixel location. 

    Attributes:
        temperature (float): Controls the frequency scaling of the sinusoids. 
    """

    def __init__(self):

        # params
        self.temperature = 10000 # ratio for the geometric progression in sinusoid frequencies
    
    def compute(self, B, C, H, W, device=None):

        """
        Generates a 2D positional encoding tensor.

        Args:
            B (int): Batch size.
            C (int): Total embedding channels (must be divisible by 4).
            H (int): Height of the feature map.
            W (int): Width of the feature map.
            device (torch.device, optional): Target device for the tensor.

        Returns:
            Tensor: Spatial embeddings of shape (B, C, H, W).
        """

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