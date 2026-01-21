import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math


"""
Run the code: 

python src/franka_ai/models/flowLeonardo/examples/flow_matching.py 

"""


# --------------------------
# 0. Dataloader
# --------------------------  

class ToyDataset(Dataset):
    
    def __init__(self, num_samples, n_hist, obs_dim, act_dim, horizon):
        self.obs = torch.randn(num_samples, n_hist, obs_dim) 
        self.img = torch.randn(num_samples, n_hist, 3, 480, 640) 
        self.ee = torch.randint(0, 2, (num_samples, n_hist))
        self.actions = torch.randn(num_samples, horizon, act_dim)  

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.img[idx], self.ee[idx], self.actions[idx]


# --------------------------
# 1. Networks
# --------------------------  

class ObsProjector(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256), 
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        
    def forward(self, obs):
        return self.mlp(obs) 
    

class ImgProjector(nn.Module):
    
    def __init__(self, output_dim):
        
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # [B, 3, H, W] -> [B, 32, H/2, W/2]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> [B, 64, H/4, W/4]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # -> [B, 128, H/8, W/8]
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # -> [B, 256, H/16, W/16]
            nn.ReLU(),
            nn.Conv2d(256, output_dim, kernel_size=3, stride=2, padding=1), # -> [B, output_dim, H/32, W/32]
            nn.ReLU(),

            # Note: ACT use 1x1 conv to project channel from [B, dim_model, H’, W’] to [B, dim_model, H’, W’]
            # where: H’=15, W’=20
        )
   
    def forward(self, img):

        B, N_HIST, C, H, W = img.shape

        # Apply CNN to each step in N_HIST
        img_embeds = [self.cnn(img[:, t]) for t in range(N_HIST)]  # List: [B, output_dim, H', W']
        
        # Stack embeddings along temporal dimension
        img_embeds = torch.stack(img_embeds, dim=1)  # [B, N_HIST, output_dim, H', W']

        return img_embeds
    
    
class TransformerEncoder(nn.Module):   

    def __init__(self, embed_size, hidden_size, num_layers, nhead, output_size):
        
        super(TransformerEncoder, self).__init__()
        
        self.pos_enc1D = PosEmbedding1D()
        self.pos_enc2D = PosEmbedding2D()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size)) 
        self.embedding = nn.Linear(embed_size, hidden_size) # embedding layer
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size) # output layer

    def forward(self, obs_embeds, img_embeds, ee_embeds):

        # get dimensions
        B, n_hist, d_embed = obs_embeds.shape
        B, n_hist, d_embed, H, W = img_embeds.shape

        # add 1D positional encoding to observations
        enc1D = self.pos_enc1D.compute(B, n_hist, d_embed)
        obs_embeds = obs_embeds + enc1D
        ee_embeds = ee_embeds + enc1D

        # add spatial 2D positional encoding to images
        enc2D = self.pos_enc2D.compute(B, d_embed, H, W)
        img_embeds = img_embeds + enc2D.unsqueeze(1)  # [B, N_HIST, d_embed, H', W'] --> broadcast along N_HIST dimension
        img_embeds = img_embeds.flatten(3).permute(0, 1, 3, 2)  # [B, N_HIST, H'*W', d_embed]
        img_embeds = img_embeds.reshape(B, n_hist * H * W, d_embed)  # [B, N_HIST*H'*W', d_embed]

        # add temporal 1D positional encoding to images
        enc1D = self.pos_enc1D.compute(B, n_hist * H * W, d_embed)
        img_embeds = img_embeds + enc1D

        # preperare data for transformer
        cls_tokens = self.cls_token.repeat(B, 1, 1) # duplicate learnable CLS token for all samples in the batch --> [B, 1, embed_size]

        x = torch.cat([cls_tokens, obs_embeds, img_embeds, ee_embeds], dim=1) # [B, seq_length=1+n_hist+n_hist+1200, embed_size]
        x = self.embedding(x) # [B, seq_length, hidden_size]

        # pass through transformer
        x = self.transformer_encoder(x) # [B, seq_length, hidden_size]
        x = x[:,0,:]  # take only CLS token as summary --> [B, hidden_size]

        # final layer
        x = self.fc(x) # [B, output_size]
        
        return x


class FlowHead(nn.Module):     
    
    def __init__(self, action_dim, embed_dim):
        
        super().__init__()
        
        self.action_dim = action_dim
        self.embed_dim = embed_dim

        in_dim = action_dim + 1 + embed_dim  # per timestep: action + time + context
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


class PolicyModel(nn.Module):

    def __init__(self, obs_dim, act_dim, embed_dim):

        super().__init__()

        self.obs_projector = ObsProjector(input_dim=obs_dim, output_dim=embed_dim)
        self.img_projector = ImgProjector(output_dim=embed_dim)
        self.binary_ee_projector = nn.Embedding(2, embed_dim) # 2 states: close/open --> better reducing embed_dim and then add FC to bring output to embed_dim      
        self.encoder = TransformerEncoder(embed_size=embed_dim, hidden_size=512, num_layers=6, nhead=8, output_size=embed_dim)
        self.flow = FlowHead(action_dim=act_dim, embed_dim=embed_dim)
    
    def forward(self, x_t, t, obs, img, ee_status):

        # obs, img and gripper projector
        obs_embeds = self.obs_projector(obs) # [B, N_HISTORY, EMBED_DIM]
        img_embeds = self.img_projector(img) # [B, N_HISTORY, EMBED_DIM, H, W]
        ee_embeds = self.binary_ee_projector(ee_status) # [B, N_HISTORY, EMBED_DIM] --> I look up integers in ee_status and return corresponding embeddings

        # transformer embedding
        e = self.encoder(obs_embeds, img_embeds, ee_embeds) # [B, EMBED_DIM]

        # predict flow
        v_pred = self.flow(x_t, t, e)

        return v_pred


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
    

# --------------------------
# 2. Training Loop
# --------------------------

def train():

    # get dataset
    dataset = ToyDataset(num_samples=5, n_hist=N_HISTORY, obs_dim=OBS_DIM, act_dim=ACT_DIM, horizon=ACT_HORIZON)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # init nets
    model = PolicyModel(OBS_DIM, ACT_DIM, EMBED_DIM)

    # train
    model.train()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    # loop over dataset
    for obs, img, ee, action_chunk_target in dataloader:  
        
        # obs: [B, N_HISTORY, OBS_DIM]
        # img: [B, N_HISTORY, 3, H, W]
        # ee: [B, N_HISTORY]
        # action_chunk: [B, ACT_HORIZON, ACT_DIM] 

        # sample random noise for source distribution (uniform or gaussian)
        B, H, A = action_chunk_target.shape
        action_chunk_src = torch.rand(B, H, A) # uniform
        # action_chunk_src = torch.randn(B, H, A) # gaussian
        
        # target flow (ground truth vector field)
        v_target = action_chunk_target - action_chunk_src # [B, ACT_HORIZON, ACT_DIM] 
        
        # sample random time t ~ U(0,1)
        t = torch.rand(B, 1)  # each sample has its own time
        
        # interpolation toward noise
        x_t = action_chunk_src + t * v_target # [B, ACT_HORIZON, ACT_DIM] 

        # predict flow
        v_pred = model(x_t, t, obs, img, ee) # [B, ACT_HORIZON, ACT_DIM] 

        # Flow Matching loss
        loss = loss_fn(v_pred, v_target) # investigate loss normalization flow matching
        print(f"Loss: : {loss.item()}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# --------------------------
# 3. Inference
# --------------------------

def generate_action(obs, img, ee, steps=20):
    
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


# --------------------------
# 4. Main
# --------------------------

# params
OBS_DIM = 14
EMBED_DIM = 256
ACT_DIM = 7
ACT_HORIZON = 10
N_HISTORY = 2

# training
train()


# run inference
obs = torch.randn(N_HISTORY, OBS_DIM)
img = torch.randn(N_HISTORY, 3, 480, 640) 
ee = torch.randint(0, 2, (N_HISTORY,))

action_chunk_hat = generate_action(obs, img, ee)
print("action_chunk_hat: \n", action_chunk_hat)
        
        
        