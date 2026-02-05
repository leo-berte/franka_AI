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
# change numer denois steps + RK + random init delle azioni con rumore usando generatore fisso
# still spikes, so back to papers + github implementations
# try to embed time with FiLM or with pos embedding + MLP
# provato MLP head e funziona, FLOW head no --> problema isolato nel FLOW head
# confronto numero params ACT (), MLP head () e FM head () 
# regularization su delta_output

# TODO:

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
            elif self.config.head_type in ["flow_matching_mlp", "flow_matching_unet"]:
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
        if self.config.head_type == "mlp":
            loss = self.model.compute_loss_mlp(batch)
        elif self.config.head_type in ["flow_matching_mlp", "flow_matching_unet"]:
            loss = self.model.compute_loss(batch)
        else:
            raise NotImplementedError(f"Head type with name {self.config.head_type} is not implemented.")
        
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

        if self.config.image_features:
            self.vision_projector = VisionProjector(replace_final_stride_with_dilation=self.config.replace_final_stride_with_dilation, 
                                                    vision_backbone=self.config.vision_backbone, 
                                                    pretrained_backbone_weights=self.config.pretrained_backbone_weights, 
                                                    output_dim=self.config.dim_model)

        self.encoder = TransformerEncoder(dim_model=self.config.dim_model, 
                                          dim_feedforward_tf=self.config.dim_feedforward_tf,
                                          nhead=self.config.nhead,
                                          num_layers=self.config.num_layers)
        
        if self.config.head_type == "mlp":

            self.mlp_head = MlpHead(N_chunk=self.config.N_chunk, 
                                    action_dim=self.config.action_feature.shape[0], 
                                    dim_model=self.config.dim_model)
            
        elif self.config.head_type == "flow_matching_mlp":

            self.flow_head = FlowHeadMlp(N_chunk=self.config.N_chunk, 
                                         action_dim=self.config.action_feature.shape[0], 
                                         dim_model=self.config.dim_model,
                                         timestep_embed_dim=self.config.timestep_embed_dim,
                                         use_film=self.config.use_film)
            
        elif self.config.head_type == "flow_matching_unet":

            # self.flow_head = FlowHeadUNet(N_chunk=self.config.N_chunk, 
            #                               action_dim=self.config.action_feature.shape[0], 
            #                               dim_model=self.config.dim_model)
            
            self.flow_head = DiffusionConditionalUnet1d(config, 
                                                        global_cond_dim=self.config.dim_model) #  * config.n_obs_steps)
        
    def global_conditioning(self,  batch: dict[str, Tensor]) -> Tensor | None:
        
        """
        Compute global conditioning given observations.
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
        glob_cond = self.encoder(obs_features, all_cam_features) # [B, dim_model] (since I take only CLS token)

        return glob_cond
    
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
        A forward pass through the Flow policy.

        `batch` should have the following structure:

        {
            [robot_state_feature]: (B, N_history, state_dim) batch of robot states

            [image_features]: (B, N_history, C, H, W) batch of images
        }
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

            # t = torch.ones((B, 1, 1), device=device) * (i + 1) / self.config.denoising_steps # shape: [B,1]
            t = torch.ones((B, 1, 1), device=device) * (i) / self.config.denoising_steps # shape: [B,1]
            
            # predict flow
            v = self.flow_head(x, t, glob_cond) # [B, N_chunk, act_dim] 
    
            x = x + dt * v 

            # # compute Runge Kutta terms
            # k1 = self.flow_head(x, t, glob_cond)
            # k2 = self.flow_head(x + 0.5 * dt * k1, t + 0.5 * dt, glob_cond)
            # k3 = self.flow_head(x + 0.5 * dt * k2, t + 0.5 * dt, glob_cond)
            # k4 = self.flow_head(x + dt * k3, t + dt, glob_cond)
            
            # # weighted average
            # x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4) 

        return x # [B, N_chunk, act_dim]
    
    def generate_action_mlp(self, batch: dict[str, Tensor]) -> Tensor | None:

        # get global conditioning
        glob_cond = self.global_conditioning(batch)

        return self.mlp_head(glob_cond) # [B, N_chunk, act_dim] 


class ObsProjector(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim), 
        )
        
    def forward(self, obs):
        return self.mlp(obs) 


class VisionProjector(nn.Module):

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

    """
    B. Cross-Attention invece di Concatenazione
    Invece di un unico Transformer "polpettone", potresti:
    Usare un encoder per le immagini.
    Usare le obs_features (stato robot) come Query in una Cross-Attention verso le feature delle immagini. 
    Questo costringe il modello a guardare le immagini solo in funzione di dove si trova il robot ora.
    """

    def __init__(self, dim_model, dim_feedforward_tf, nhead, num_layers):
        
        super(TransformerEncoder, self).__init__()
        
        self.pos_enc1D = PosEmbedding1D(dim_model)
        self.pos_enc2D = PosEmbedding2D()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_model)) 
        encoder_layers = nn.TransformerEncoderLayer(d_model=dim_model, nhead=nhead, dim_feedforward=dim_feedforward_tf, dropout=0.1, batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(dim_model)

    def forward(self, obs_features, all_cam_features):

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

        # duplicate learnable CLS token for all samples in the batch (each sample in batch has its own CLS, and they all share same weights)
        cls_token = self.cls_token.repeat(B, 1, 1) # [B, 1, dim_model]
        inputs_list.append(cls_token)

        # compose all the tokens as input for transformer
        x = torch.cat(inputs_list, dim=1) # [B, seq_length=1+n_hist+N_cam*N_HIST*H'*W', dim_model]

        # --> x:  torch.Size([B, 22, 512]) con img [120, 160] e N_h = 1 e cam=1
        # --> x:  torch.Size([B, 130, 512]) con img [300, 500] e N_h = 1 e cam=1

        # pass through transformer
        x = self.transformer_encoder(x) # [B, seq_length, dim_model]
        x = x[:,-1,:]  # take only CLS token as summary --> [B, dim_model]
        # x = torch.mean(x, dim=1) # TOGLI CLS PERÒ

        # TODO: prova torch.mean invece del CLS token come sumup

        # Layer normalization for stability
        out = self.final_norm(x)
        
        return out


class MlpHead(nn.Module):     

    """Simple MLP head."""
    
    def __init__(self, N_chunk, action_dim, dim_model):
        
        super().__init__()

        self.N_chunk = N_chunk
        self.action_dim = action_dim
        
        in_dim = dim_model 
        out_dim = N_chunk*action_dim 

        # Encoder for the diffusion timestep.
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
    
    
class FlowHeadMlp(nn.Module):     

    """
    Flow Matching made with MLP.
    Possible to use MLP or FiLM modulation for time embedding.
    """
    
    def __init__(self, N_chunk, action_dim, dim_model, timestep_embed_dim, use_film):
        
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

        # x: [B, N_chunk, action_dim]
        # t: [B, 1, 1]
        # e: [B, dim_model]

        B, N_chunk, act_dim = x.shape  
        device = x.device

        # MLP + embed N_chunk with sinusoidal embedding
        x = self.mlp_actions(x) # [B, dim_model]
        chunk_indices = torch.arange(N_chunk, dtype=torch.float32, device=device)
        chunk_embedded = self.pos_enc1D_actions.compute(chunk_indices) # [N_c, d_model]
        x = x + chunk_embedded.unsqueeze(0) # [B, N_c, d_model]
        
        # embed time with sinusoidal embedding + MLP
        t = t.squeeze(-1) # [B, 1]
        t_embedded = self.pos_enc1D_time.compute(t) # [B, timestep_embed_dim]
        t_embedded = self.mlp_time(t_embedded) # [B, timestep_embed_dim]
            
        cond = torch.cat([t_embedded, e], dim=-1)  # [B, cond_dim_tot]

        # Iniezione condizionamento come bias per ogni step temporale N. Unsqueeze for broadcasting to `out`
        cond_embed = self.cond_encoder(cond).unsqueeze(1) # [B, 1, dim_model] or [B, 1, 2*dim_model]

        if self.use_film:
            # Treat the embedding as a list of scales and biases
            scale = cond_embed[:, :, :self.dim_model]
            bias = cond_embed[:, :, self.dim_model:]
            x = scale * x + bias  # [B, N_chunk, dim_model]
        else:
            # Treat the embedding as biases
            x = x + cond_embed  # [B, N_chunk, dim_model]
        
        # predict flow
        out = self.mlp_flow(x)  # [B, N_chunk, act_dim]

        return out
    
        






class Conv1dBlock(nn.Module):
    
    """
    Utility block: Conv1d --> GroupNorm --> Mish.
    Riproduce la struttura esatta di Diffusion Policy.
    """
    
    def __init__(self, inp_channels, out_channels, kernel_size=3, n_groups=8):

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
    Blocco residuo che inietta il condizionamento (tempo + obs) 
    tramite una proiezione lineare (stile FiLM semplificato).
    """

    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8, use_film=False):
        
        super().__init__()

        # params
        self.use_film = use_film

        self.conv1 = Conv1dBlock(in_channels, out_channels, kernel_size, n_groups)

        # I may use FiLM to combine conditioning with 'x' tensor
        cond_channels = out_channels * 2 if use_film else out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels)
        )

        self.conv2 = Conv1dBlock(out_channels, out_channels, kernel_size, n_groups)
        
        # Shortcut se le dimensioni non matchano
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):

        """
        Blocco residuo che inietta il condizionamento (tempo + obs) 
        tramite una proiezione lineare (stile FiLM semplificato).

        Args:
            x: (B, in_channels, T)
            cond: (B, cond_dim)
        Returns:
            (B, out_channels, T)
        """

        # x: [B, action_dim, N_chunk], cond: [B, cond_dim]
        out = self.conv1(x) # [B, out_channels, N_chunk]
        
        # Iniezione condizionamento come bias per ogni step temporale N. Unsqueeze for broadcasting to `out`
        cond_embed = self.cond_encoder(cond).unsqueeze(-1) # [B, out_channels, 1] or [B, 2*out_channels, 1]

        if self.use_film:
            # Treat the embedding as a list of scales and biases
            scale = cond_embed[:, : self.out_channels]
            bias = cond_embed[:, self.out_channels :]
            out = scale * out + bias  # [B, out_channels, N_chunk]
        else:
            # Treat the embedding as biases
            out = out + cond_embed  # [B, out_channels, N_chunk]

        out = self.conv2(out) # [B, out_channels, N_chunk]
        
        # Connessione residua
        return out + self.residual_conv(x) # [B, out_channels, N_chunk]


# class FlowHeadUNet(nn.Module):     

#     """
#     Flow Matching UNet Lite.
#     Eredita la logica di embedding temporale della versione MLP ma usa 
#     convoluzioni 1D per garantire fluidità temporale senza esplodere nei parametri.
#     """

#     def __init__(self, N_chunk, action_dim, dim_model):

#         super().__init__()
        
#         # 1. Gestione Tempo (come tua MLP)
#         self.pos_enc1D = PosEmbedding1D(dim_model) 
#         self.mlp_time = nn.Sequential(
#             nn.Linear(dim_model, dim_model * 2),
#             nn.Mish(),
#             nn.Linear(dim_model * 2, dim_model)
#         )

#         # 2. Proiezione del condizionamento globale
#         # Uniamo t_embedded [dim_model] + e [dim_model] = 2 * dim_model
#         cond_dim = dim_model * 2

#         # 3. Struttura UNet (Canali ridotti per risparmiare parametri)
#         self.down1 = ConditionalResidualBlock1d(action_dim, 256, cond_dim)
#         self.down2 = ConditionalResidualBlock1d(256, 512, cond_dim)
        
#         # Decoder con skip connection
#         self.up1 = ConditionalResidualBlock1d(256+512, 256, cond_dim)

#         # Output finale per tornare ad action_dim
#         self.final_conv = nn.Sequential(
#             Conv1dBlock(256, 128, kernel_size=3),
#             nn.Conv1d(128, action_dim, kernel_size=1)
#         )

#     def forward(self, x_t, t, e):
        
#         # x_t: [B, N_chunk, act_dim]
#         B, N, D = x_t.shape       
        
#         # Trasponiamo per Conv1D: [B, act_dim, N_chunk]
#         x = x_t.transpose(1, 2)

#         # Embedding del tempo (Sincronizzato con tua logica MLP)
#         t = t.squeeze(1) # [B, 1]
#         t_embedded = self.pos_enc1D.compute(t)
#         t_embedded = self.mlp_time(t_embedded) # [B, dim_model]

#         # Creazione del vettore di condizionamento unico [B, dim_model*2]
#         cond = torch.cat([t_embedded, e], dim=-1)

#         # UNet encoder
#         skip1 = self.down1(x, cond) # [B, 256, N_chunk]     
#         x = self.down2(skip1, cond) # [B, 512, N_chunk]     

#         # Unet Decoder + Skip Connection
#         x = torch.cat([x, skip1], dim=1) # [B, 256+512, N_chunk]   
#         x = self.up1(x, cond) # [B, 256, N_chunk]        

#         # Output
#         out = self.final_conv(x) # [B, action_dim, N_chunk]
        
#         return out.transpose(1, 2) # [B, N_chunk, act_dim]




class PosEmbedding1D():

    def __init__(self, d_model, temperature=10000):

        self.d_model = d_model
        self.temperature = temperature

    def compute(self, x):

        # x: [B, 1] (used in flow matching) o x: [seq_len, ] (used in transformers tokens)
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
    



# import math
import einops


class DiffusionConditionalUnet1d(nn.Module):
    
    """A 1D convolutional UNet with FiLM modulation for conditioning.

    Note: this removes local conditioning as compared to the original diffusion policy code.
    """

    def __init__(self, config: FlowConfig, global_cond_dim: int):
        
        super().__init__()

        self.config = config

        # add sinusoidal embedding to time
        # ADDED
        self.pos_enc1D = PosEmbedding1D(config.diffusion_step_embed_dim) 

        # Encoder for the diffusion timestep.
        self.diffusion_step_encoder = nn.Sequential(
            # DiffusionSinusoidalPosEmb(config.diffusion_step_embed_dim),
            nn.Linear(config.diffusion_step_embed_dim, config.diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(config.diffusion_step_embed_dim * 4, config.diffusion_step_embed_dim),
        )

        # The FiLM conditioning dimension.
        cond_dim = config.diffusion_step_embed_dim + global_cond_dim

        # In channels / out channels for each downsampling block in the Unet's encoder. For the decoder, we
        # just reverse these.
        in_out = [(config.action_feature.shape[0], config.down_dims[0])] + list(
            zip(config.down_dims[:-1], config.down_dims[1:], strict=True)
        )

        # Unet encoder.
        common_res_block_kwargs = {
            "cond_dim": cond_dim,
            "kernel_size": config.kernel_size,
            "n_groups": config.n_groups,
            # "use_film_scale_modulation": config.use_film_scale_modulation,
        }
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1d(dim_in, dim_out, **common_res_block_kwargs),
                        ConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        # Downsample as long as it is not the last block.
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        # Processing in the middle of the auto-encoder.
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
                ConditionalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
            ]
        )

        # Unet decoder.
        self.up_modules = nn.ModuleList([])
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
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
            Conv1dBlock(config.down_dims[0], config.down_dims[0], kernel_size=config.kernel_size),
            nn.Conv1d(config.down_dims[0], config.action_feature.shape[0], 1),
        )

    def forward(self, x: Tensor, timestep: Tensor | int, global_cond=None) -> Tensor:
        
        """
        Args:
            x: (B, T, input_dim) tensor for input to the Unet.
            timestep: (B,) tensor of (timestep_we_are_denoising_from - 1).
            global_cond: (B, global_cond_dim)
        Returns:
            (B, T, input_dim) diffusion model prediction.
        """

        # For 1D convolutions we'll need feature dimension first.
        x = einops.rearrange(x, "b t d -> b d t")

        # ADDED
        timestep = timestep.squeeze(-1) # [B, 1]
        t_embedded = self.pos_enc1D.compute(timestep) # [B, dim_model]
        timesteps_embed = self.diffusion_step_encoder(t_embedded)

        # timesteps_embed = self.diffusion_step_encoder(timestep)

        # If there is a global conditioning feature, concatenate it to the timestep embedding.
        if global_cond is not None:
            global_feature = torch.cat([timesteps_embed, global_cond], axis=-1)
        else:
            global_feature = timesteps_embed

        # Run encoder, keeping track of skip features to pass to the decoder.
        encoder_skip_features: list[Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            encoder_skip_features.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # Run decoder, using the skip features from the encoder.
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, encoder_skip_features.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, "b d t -> b t d")
        return x