import torch
import timm
from torch import nn

from flamingo_pytorch import PerceiverResampler


"""
Run the code: python src/franka_ai/models/flowLeonardo/examples/test_vision_modules.py 
"""


class VisionProjector(nn.Module):

    """Image feature extractor and projection layer for visual observations."""
    
    """
    Nella tua TransformerEncoder, tu concateni tutto: [CLS, obs, cam1_hist, cam2_hist].
    Se hai N_HIST = 5, 2 camere e mappe 7x7, la tua sequenza è:
    $1 + 5 + (2 \times 5 \times 49) = 496$ token.

    Invece di mandare TUTTI i patch al Transformer, dovresti comprimerli PRIMA.
    Opzione 1 (LeRobot style): Usa lo Spatial Softmax dopo la VisionProjector. 
    Invece di 49 patch per immagine, avresti solo 32-64 numeri (coordinate keypoints). 
    La sequenza passerebbe da 496 token a circa 15-20 token. Questo eliminerebbe il jitter istantaneamente.
    Opzione 2 (Pooling): Fai un Global Average Pooling sulle mappe 7x7 prima di darle al Transformer.

    numero parametri che hanno?
    """

    def __init__(self, 
                 vision_backbone_name, 
                 freeze_backbone=True,
                 output_dim=512,
                 pretrained=True):

        super().__init__()
        
        # load model from pretrained weights
        self.backbone = timm.create_model(
            vision_backbone_name, 
            pretrained=pretrained, 
            num_classes=0, # to avoid classification head
            global_pool='', # to maintain token dimensions
            dynamic_img_size=True if "vit" in vision_backbone_name else None
        )

        # params
        self.feature_dim = self.backbone.num_features # get output dim
        self.freeze_backbone = freeze_backbone

        # freeze backbone parameters
        if (freeze_backbone == True):
            for param in self.backbone.parameters():
                param.requires_grad = False

        # project to 'output_dim'
        if ("resnet" in vision_backbone_name):
            self.img_projector2D = nn.Conv2d(self.feature_dim, output_dim, kernel_size=1) # 1x1 conv2D
        elif ("vit" in vision_backbone_name):
            self.img_projector1D = nn.Conv1d(self.feature_dim, output_dim, kernel_size=1) # 1x1 conv1D
            self.n_prefix = self.backbone.num_prefix_tokens # number of CLS token (ViT)
        else:
            raise ValueError(f"Vision backbone name is wrong.")
        
        
        # gemini dice che vuole comunque spatial pos embeddings in input il perceiver
        # sembra che codice gr1 non lo fa, potrei iniziare cosi o aggiungere semplicemente un 1d encoding


        # Resampler hparams
        resampler_params = dict()
        resampler_params['patch_feat_dim'] = ...
        resampler_params['depth'] = ...
        resampler_params['dim_head'] = ...
        resampler_params['heads'] = ...
        resampler_params['num_latents'] = ...
        resampler_params['num_media_embeds'] = ...

        # Perciever resampler
        self.n_patch_latents = resampler_params['num_latents']
        self.perceiver_resampler = PerceiverResampler(
            dim=resampler_params['patch_feat_dim'],
            depth=resampler_params['depth'],
            dim_head=resampler_params['dim_head'],
            heads=resampler_params['heads'],
            num_latents=self.n_patch_latents,
            num_media_embeds=resampler_params['num_media_embeds'])


    def train(self, mode=True):
            
        """
        Used to override command .train() and avoid batch normalization.
        This is required in resnet architecture when using pretrained weights.
        """

        super().train(mode)
        
        if (mode==True):
            # force backbone in eval
            self.backbone.eval()
            # to be fully safe, block also batch norm
            for m in self.backbone.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
                    m.weight.requires_grad = False 
                    m.bias.requires_grad = False

    def forward(self, obs):

        """
        Extracts and projects spatial features from a batch of images.

        Args:
            obs (Tensor): Batch of images with shape (B * N_HISTORY, C, H, W).
        Returns:
            Tensor: Spatial feature map of shape (B * N_HISTORY, seq_len, output_dim).
        """

        # feature extraction
        if (self.freeze_backbone == True):
            with torch.no_grad(): # avoid to accumulte gradients
                x = self.backbone(obs) # [B*N, C_feat, H', W'] for CNN or [B*N, 1+seq_len, C_feat] for ViT
        else:
            x = self.backbone(obs)
            
        # project to transformer expected input dimension

        if (x.dim() == 4): # resnet
            x = self.img_projector2D(x) # [B*N, output_dim, H', W']
            x = x.flatten(2).permute(0, 2, 1) # [B*N, H'*W', output_dim]
        elif (x.dim() == 3): # ViT
            x = x[:, self.n_prefix:, :] # remove special tokens --> [B*N, seq_len, output_dim]
            x = x.permute(0, 2, 1).contiguous() # since I did slicing, use "contiguous"
            x = self.img_projector1D(x) # [B*N, seq_len, output_dim]
            x = x.permute(0, 2, 1)
        
        print("x pre: ", x)
        x = self.perceiver_resampler(x)
        print("x post: ", x)

        return x 
    


# test inference

vision_backbone_name = "vit_base_patch16_224.mae" # "resnet18d.ra2_in1k" "vit_base_patch16_224.mae" "vit_base_patch14_dinov2.lvd142m" "vit_base_patch14_reg4_dinov2.lvd142m" 

vision_projector = VisionProjector(vision_backbone_name=vision_backbone_name) 

# Input image
example_input = torch.randn(9, 3, 208, 272) # mae
# example_input = torch.randn(9, 3, 210, 280) # dino - resnet

print("Ready for inference...")

# Forward pass
with torch.no_grad():
    output = vision_projector(example_input)

print(f"Nome modello: {vision_backbone_name}")
print(f"Shape dell'input:  {example_input.shape}")  # [B, C, H, W]
print(f"Shape del risultato: {output.shape}")         # [B, L, D]

# # print available models in timm
# avail_pretrained_models = timm.list_models(pretrained=True)
# for model in avail_pretrained_models:
#     if 'resnet' in model.lower():
#         print(model)