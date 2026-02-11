import torch
import timm
from torch import nn


"""
Run the code: python src/franka_ai/models/flowLeonardo/examples/test_vision_modules.py 
"""


class VisionProjector(nn.Module):

    def __init__(self, vision_backbone, pretrained=True, output_dim=512):

        super().__init__()
        
        # load model from pretrained weights
        self.backbone = timm.create_model(
            vision_backbone, 
            pretrained=pretrained, 
            num_classes=0, # to avoid classification head
            global_pool='' # to maintain token dimensions
        )
        
        # get output dim
        self.feature_dim = self.backbone.num_features
        
        # 1x1 conv to project to 'output_dim'
        self.img_projector = nn.Conv2d(self.feature_dim, output_dim, kernel_size=1)

    def forward(self, obs):

        # 1. Feature extraction
        x = self.backbone(obs) # [B*N, C_feat, H, W] per CNN o [B*N, L, D] per ViT
        
        # 2. Se è un ViT (3D tensor), dobbiamo trasformarlo in griglia 2D
        if x.dim() == 3:
            # Rimuoviamo i token speciali (prefix)
            n_prefix = self.backbone.num_prefix_tokens
            x = x[:, n_prefix:, :] # [B*N, L_spaziale, D]
            
            # Reshape a griglia quadrata (es. 196 -> 14x14)
            hw = int(x.shape[1]**0.5)
            x = x.transpose(1, 2).reshape(x.shape[0], -1, hw, hw) # [B*N, D, H, W]
            
        # 3. Proiezione alla dimensione del Transformer della Policy
        obs = self.img_projector(x)
        return obs # [B*N_HIST, output_dim, H', W']
    


# test inference

vision_backbone_name = 'resnet18d.ra2_in1k' #  resnet18d.ra2_in1k        vit_base_patch14_dinov2.lvd142m      vit_base_patch14_reg4_dinov2.lvd142m   vit_base_patch16_224.mae

vision_projector = VisionProjector(vision_backbone=vision_backbone_name, 
                                   output_dim=512) 

# Creiamo un tensore immagine di esempio
example_input = torch.randn(1, 3, 208, 272)
example_input = torch.randn(1, 3, 224, 224)

print("Ready for inference...")

# Eseguiamo il forward pass
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



