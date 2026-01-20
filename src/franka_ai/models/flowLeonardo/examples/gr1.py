# Copyright (2024) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GR-1 model."""
import torch
import torch.nn as nn

import transformers
from flamingo_pytorch import PerceiverResampler

from models.trajectory_gpt2 import GPT2Model
from models.vision_transformer import Block
from models.transformer_utils import get_2d_sincos_pos_embed



# 1) compara gestione history col mio
# 2) compara gestione pos/temporal encoding per la history + spatial encoding images (sinusoid Vs learned) --> transformer utils.py
# 3) capire numero param modello e quali sono scelte che lo renderanno + lento in inference
# 4) aggiungo flow match heads
# 6) aggiungo gestione N camere --> vedi foto
# 7) capire come gestire correttamente casual mask con history




class GR1(nn.Module):
    def __init__(
            self,
            model_clip,
            model_mae,
            state_dim,
            act_dim,
            hidden_size,
            sequence_length,
            training_target,
            img_feat_dim,
            patch_feat_dim,
            lang_feat_dim,
            resampler_params,
            without_norm_pixel_loss=False,
            use_hand_rgb=True,
            **kwargs
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.sequence_length = sequence_length

        # GPT
        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,
            n_embd=hidden_size,
            **kwargs
        )
        self.transformer = GPT2Model(config)

        # Perciever resampler
        self.n_patch_latents = resampler_params['num_latents']
        self.perceiver_resampler = PerceiverResampler(
            dim=patch_feat_dim,
            depth=resampler_params['depth'],
            dim_head=resampler_params['dim_head'],
            heads=resampler_params['heads'],
            num_latents=self.n_patch_latents,
            num_media_embeds=resampler_params['num_media_embeds'])        

        # CLIP for language encoding
        self.model_clip = model_clip
        for _, param in self.model_clip.named_parameters():
            param.requires_grad = False

        # MAE for image encoding
        self.model_mae = model_mae
        for _, param in self.model_mae.named_parameters():
            param.requires_grad = False
        
        self.n_patches = 49
        self.patch_size = 16
        self.image_size = 224
        self.img_feat_dim = img_feat_dim
        self.lang_feat_dim = lang_feat_dim
        self.patch_feat_dim = patch_feat_dim
        self.use_hand_rgb = use_hand_rgb

        self.act_pred = False
        self.fwd_pred = False
        self.fwd_pred_hand = False
        if 'act_pred' in training_target:
            self.act_pred = True
        if 'fwd_pred' in training_target:
            self.fwd_pred = True
        if 'fwd_pred_hand' in training_target:
            self.fwd_pred_hand = True
        
        self.without_norm_pixel_loss = without_norm_pixel_loss

        # Embedding functions for states
        self.embed_arm_state = torch.nn.Linear(self.state_dim - 1, hidden_size)
        self.embed_gripper_state = torch.nn.Linear(2, hidden_size) # one-hot gripper state
        self.embed_state = torch.nn.Linear(2*hidden_size, hidden_size)

        # Relative timestep embedding (this is essentially a positional encoding, but learnable instead of sinusoidal)
        # I need it when I use history so model knows: "this token corresponds to timestep t in the sequence")
        self.embed_timestep = nn.Embedding(self.sequence_length, hidden_size)

        # Embedding function for languages
        self.embed_lang = torch.nn.Linear(self.lang_feat_dim, hidden_size)

        # sequence_length = [1 (language) + 1 (state) + 1 (full_image) + n_patches ] * history_len
        # --> I sum self.embed_timestep to get temporal encoding for history (seq_len is the history_len here)

        # Embedding functions for images
        self.embed_hand_img = torch.nn.Linear(self.img_feat_dim, hidden_size)
        self.embed_img = torch.nn.Linear(self.img_feat_dim, hidden_size)
        self.embed_hand_patch = torch.nn.Linear(self.patch_feat_dim, hidden_size) 
        self.embed_patch = torch.nn.Linear(self.patch_feat_dim, hidden_size)

        # Layer norm
        self.embed_ln = nn.LayerNorm(hidden_size)

        # Action query token
        self.action_queries = nn.Embedding(1, hidden_size) # arm + gripper

        # Observation query token
        self.obs_queries = nn.Embedding(self.n_patch_latents + 1, self.hidden_size)
        self.obs_hand_queries = nn.Embedding(self.n_patch_latents + 1, self.hidden_size)

        # action_queries → “What action should the robot take next?”
        # obs_queries → “What will the scene look like in the next timestep?”
        # They are like CLS token in BERT, and they attend to all previous tokens (language, state, image latents) using the GPT’s causal attention.

        # Action prediction
        self.pred_act_mlps = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size//2),
            nn.Linear(hidden_size//2, hidden_size//2)])
        self.pred_arm_act = nn.Linear(hidden_size//2, self.act_dim-1) # arm action
        self.pred_gripper_act = nn.Linear(hidden_size//2, 1) # gripper action (binary)
        
        # Forward prediction
        self.decoder_embed = nn.Linear(hidden_size, hidden_size, bias=True) # for each OBS token, I pass through linear layer
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, hidden_size)) 
        decoder_depth = 2
        self.decoder_blocks = nn.ModuleList([ # each OBS token goes thorugh a small decoder transformer
            Block(hidden_size, 16, 4, qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm)
            for i in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(hidden_size)
        self.decoder_pred = nn.Linear(hidden_size, self.patch_size**2 * 3, bias=True) # output from decoder transformer becomes an image patch
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, (self.image_size//self.patch_size)**2, # add 2D pos encoding for patches (non trainable)
            hidden_size), requires_grad=False)  # (1, n_patch, h)
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], (self.image_size//self.patch_size)) 
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)) # I add this to have 2D pos encoding in the model,
        # otherwise tensor is not included neither in the device (CPU/GPU) nor in state_dict()

    def forward(self, 
                rgb, 
                hand_rgb, 
                state, 
                language, 
                attention_mask
    ):
        obs_preds = None
        obs_hand_preds = None
        obs_targets = None
        obs_hand_targets = None
        arm_action_preds = None
        gripper_action_preds = None

        # in forward I already pass history of data (sequence_length) -> below: sequence_length == l
        batch_size, sequence_length, c, h, w = rgb.shape
        
        # Embed state
        arm_state = state['arm']
        gripper_state = state['gripper']
        arm_state_embeddings = self.embed_arm_state(arm_state.view(batch_size, sequence_length, self.state_dim-1))
        gripper_state_embeddings = self.embed_gripper_state(gripper_state)
        state_embeddings = torch.cat((arm_state_embeddings, gripper_state_embeddings), dim=2)
        state_embeddings = self.embed_state(state_embeddings)  # (b, l, h)

        # Embed language
        lang_embeddings = self.model_clip.encode_text(language)
        lang_embeddings = lang_embeddings / (lang_embeddings.norm(dim=1, keepdim=True) + 1e-6) # normalization L2 row by row (each sentence of the batch)
        lang_embeddings = self.embed_lang(lang_embeddings.float())  # (b, h)
    
        # Get obs and patch feature from MAE
        obs_embeddings, patch_embeddings = self.model_mae(
            rgb.view(batch_size*sequence_length, c, h, w))  # (b * l, img_feat_dim), (b * l, n_patches, patch_feat_dim)
        obs_embeddings = obs_embeddings.view(batch_size, sequence_length, -1)  # (b, l, img_feat_dim)
        if self.use_hand_rgb:
            hand_obs_embeddings, hand_patch_embeddings = self.model_mae(
                hand_rgb.view(batch_size*sequence_length, c, h, w))
            hand_obs_embeddings = hand_obs_embeddings.view(batch_size, sequence_length, -1)  # (b, l, img_feat_dim)
        
        # Prepare the image labels for future MSE pixel loss
        if self.fwd_pred:
            p = self.patch_size
            h_p = h // p
            w_p = w // p
            rgb = rgb.reshape(shape=(batch_size, sequence_length, 3, h_p, p, w_p, p)) 
            obs_targets = rgb.permute(0, 1, 3, 5, 4, 6, 2)
            obs_targets = obs_targets.reshape(shape=(batch_size, sequence_length, h_p * w_p, (p**2) * 3))  # (b, l, n_patches, p*p*3)
            if not self.without_norm_pixel_loss:
                # norm the target 
                obs_targets = (obs_targets - obs_targets.mean(dim=-1, keepdim=True)
                    ) / (obs_targets.var(dim=-1, unbiased=True, keepdim=True).sqrt() + 1e-6)
            if self.fwd_pred_hand:
                hand_rgb = hand_rgb.reshape(shape=(batch_size, sequence_length, 3, h_p, p, w_p, p))
                obs_hand_targets = hand_rgb.permute(0, 1, 3, 5, 4, 6, 2)
                obs_hand_targets = obs_hand_targets.reshape(shape=(batch_size, sequence_length, h_p * w_p, (p**2)*3))  # (b, l, n_patches, p*p*3)
                if not self.without_norm_pixel_loss:
                    # norm the target 
                    obs_hand_targets = (obs_hand_targets - obs_hand_targets.mean(dim=-1, keepdim=True)
                        ) / (obs_hand_targets.var(dim=-1, unbiased=True, keepdim=True).sqrt() + 1e-6)            

        # Use resampler to process patch embeddings
        patch_embeddings = patch_embeddings.unsqueeze(1)  # (b * l, 1, n_patches, patch_feat_dim)
        patch_embeddings = self.perceiver_resampler(patch_embeddings)  # (b * l, 1, n_patch_latents, patch_feat_dim)
        patch_embeddings = patch_embeddings.squeeze(1)  # (b * l, n_patch_latents, patch_feat_dim)
        patch_embeddings = patch_embeddings.view(batch_size, sequence_length, self.n_patch_latents, self.patch_feat_dim)  # (b, l, n_patch_latents, patch_feat_dim)
        if self.use_hand_rgb:
            hand_patch_embeddings = hand_patch_embeddings.unsqueeze(1)
            hand_patch_embeddings = self.perceiver_resampler(hand_patch_embeddings)
            hand_patch_embeddings = hand_patch_embeddings.squeeze(1)
            hand_patch_embeddings = hand_patch_embeddings.view(batch_size, sequence_length, self.n_patch_latents, self.patch_feat_dim)  # (b, l, n_patch_latents, patch_feat_dim)
        
        # Embed images and patches
        obs_embeddings = self.embed_img(obs_embeddings.float())  # (b, l, h)
        patch_embeddings = self.embed_patch(patch_embeddings.float())  # (b, l, n_patch_latents, h)
        if self.use_hand_rgb:
            hand_obs_embeddings = self.embed_hand_img(hand_obs_embeddings.float())  # (b, l, h)
            hand_patch_embeddings = self.embed_hand_patch(hand_patch_embeddings.float())  # (b, l, n_patch_latents, h)
        
        # Add timestep embeddings
        time_embeddings = self.embed_timestep.weight  # (l, h)
        lang_embeddings = lang_embeddings.view(batch_size, 1, -1) + time_embeddings  # (b, 1, h) + (l, h) --> I aso repeat language input along seq_len and add temporal encoding
        state_embeddings = state_embeddings + time_embeddings # (b, l, h) + (l, h) 
        patch_embeddings = patch_embeddings + time_embeddings.view(sequence_length, 1, self.hidden_size) # (b, l, n_patch_latents, patch_feat_dim) + (l, 1, h)
        obs_embeddings = obs_embeddings + time_embeddings # (b, l, h) + (l, h) 
        if self.use_hand_rgb:
            hand_obs_embeddings = hand_obs_embeddings + time_embeddings
            hand_patch_embeddings = hand_patch_embeddings + time_embeddings.view(sequence_length, 1, self.hidden_size)

        # # Broadcasting NOTE
        # PyTorch allinea i tensori da destra verso sinistra (come NumPy):
        # Se due dimensioni sono uguali, ok
        # Se una delle due è 1, viene “replicata” sull’altra
        # Se nessuna delle due è 1 e sono diverse → errore
        # # Example
        # patch_embeddings: (b, l, n_patch_latents, h)
        # time_embeddings: (l, 1, h)
        # --> PyTorch automatically expands time_embeddings to (b, l, n_patch_latents, h)

        # Format sequence for single timestamp: (lang, state, patch, obs, hand_patch, hand_obs, [ACT], [OBS], [OBS_HAND])_i_th
        lang_embeddings = lang_embeddings.view(batch_size, sequence_length, 1, self.hidden_size)
        state_embeddings = state_embeddings.view(batch_size, sequence_length, 1, self.hidden_size)
        obs_embeddings = obs_embeddings.view(batch_size, sequence_length, 1, self.hidden_size)
        stacked_inputs = torch.cat(
                (lang_embeddings, 
                 state_embeddings, 
                 patch_embeddings, 
                 obs_embeddings), dim=2)  # (b, l, n_tokens, h)
        if self.use_hand_rgb:
            hand_obs_embeddings = hand_obs_embeddings.view(batch_size, sequence_length, 1, self.hidden_size)
            stacked_inputs = torch.cat(
                (stacked_inputs,
                 hand_patch_embeddings, 
                 hand_obs_embeddings), dim=2)  # (b, l, n_tokens, h)
        if self.act_pred: # add token for action prediction (repeated for each timestamp)
            action_queries = self.action_queries.weight  # (1, h)
            # .repeat() --> it expands the first two dimensions by copying the same content
            action_queries = action_queries.view(1, 1, 1, self.hidden_size).repeat(batch_size, sequence_length, 1, 1)  # (b, l, 1, h)
            stacked_inputs = torch.cat((stacked_inputs, action_queries), dim=2)  # (b, l, n_tokens, h)
        if self.fwd_pred: # add token for OBS predictions (repeated for each timestamp)
            obs_queries = self.obs_queries.weight  # (n_patch_latents + 1, h)
            # .repeat() --> it expands the first two dimensions by copying the same content
            obs_queries = obs_queries.view(1, 1, self.n_patch_latents + 1, self.hidden_size).repeat(batch_size, sequence_length, 1, 1)  # (b, l, n_patch_latents + 1, h)
            stacked_inputs = torch.cat((stacked_inputs, obs_queries), dim=2)
            if self.fwd_pred_hand:
                obs_hand_queries = self.obs_hand_queries.weight # 10, h
                obs_hand_queries = obs_hand_queries.view(1, 1, self.n_patch_latents+1, self.hidden_size).repeat(batch_size, sequence_length, 1, 1)
                stacked_inputs = torch.cat((stacked_inputs, obs_hand_queries), dim=2)
        
        # Number of tokens
        n_lang_tokens = 1
        n_state_tokens = 1
        n_patch_tokens = self.n_patch_latents
        n_obs_tokens = 1
        n_hand_patch_tokens = self.n_patch_latents
        n_hand_obs_tokens = 1
        n_tokens = n_lang_tokens + n_state_tokens + n_patch_tokens + n_obs_tokens
        if self.use_hand_rgb:
            n_tokens += n_hand_obs_tokens
            n_tokens += n_hand_patch_tokens
        n_act_pred_tokens = 1
        if self.act_pred:
            act_query_token_start_i = n_tokens
            n_tokens += 1
        if self.fwd_pred:
            obs_query_token_start_i = n_tokens
            n_tokens += (n_patch_tokens + n_obs_tokens)
            if self.fwd_pred_hand:
                obs_hand_query_token_start_i = n_tokens
                n_tokens += (n_patch_tokens + n_obs_tokens) 

        # Layer norm: normalize only the features within each token
        # Each token (whether it comes from language, state, or image) is “put on the same internal scale”
        # This way, all tokens have comparable feature magnitudes before being processed together by the transformer.
        stacked_inputs = stacked_inputs.reshape(batch_size, n_tokens * sequence_length, self.hidden_size) # merge history_len and n_tokens
        stacked_inputs = self.embed_ln(stacked_inputs)

        # Attention mask needed for:
        # 1) Ignore empty timestamps since buffer is not full yet (padding)
        # 2) Ignore special tokens which does not contain real data (OBS, ACT tokens)
        
        stacked_attention_mask = attention_mask.view(batch_size, sequence_length, 1) # b, l --> b, l, 1

        # Attention mask to handle 1)
        # Each timestep t doesn’t have just one token, it has many.
        # .repeat(...) copies the same validity mask for every token type at that timestep.
        
        if self.use_hand_rgb:
            stacked_attention_mask = stacked_attention_mask.repeat( 
                1, 1, n_lang_tokens + n_state_tokens + n_hand_patch_tokens + n_hand_obs_tokens + n_patch_tokens + n_obs_tokens) # b, l, n_tokens
        else:
            stacked_attention_mask = stacked_attention_mask.repeat(
                1, 1, n_lang_tokens + n_state_tokens + n_patch_tokens + n_obs_tokens) # b, l, n_tokens

        # example with mask = (1, 5, 5):
        # mask = [
        #    [[1,1,1,1,1],
        #     [1,1,1,1,1],
        #     [1,1,1,1,1],
        #     [0,0,0,0,0],
        #     [0,0,0,0,0]]
        # ]
        # --> only timesteps t₁..t₃ have all 5 token types valid

        # Attention mask to handle 2)
        # [ACT] / [OBS] can see history to make predictions, but themselves don’t contribute as real input to future timesteps
        # So I set them to zero in the mask all the tokens related to ACT, OBS

        if self.act_pred:
            act_query_attention_mask = torch.zeros((batch_size, sequence_length, n_act_pred_tokens), dtype=torch.long).cuda() # b, l, 1
            stacked_attention_mask = torch.cat((stacked_attention_mask, act_query_attention_mask), dim=2) # stack along token space
        if self.fwd_pred:
            obs_query_attention_mask = torch.zeros((batch_size, sequence_length, n_patch_tokens + n_obs_tokens), dtype=torch.long).cuda() # b, l, n_OBS_tokens
            stacked_attention_mask = torch.cat((stacked_attention_mask, obs_query_attention_mask), dim=2) # stack along token space
            if self.fwd_pred_hand:
                obs_hand_query_attention_mask = torch.zeros((batch_size, sequence_length, n_patch_tokens + n_obs_tokens), dtype=torch.long).cuda()
                stacked_attention_mask = torch.cat((stacked_attention_mask, obs_hand_query_attention_mask), dim=2)

        # example with mask (1, 5, 9) by adding ACT, OBS tokens: 
        # mask = [
        #    [[1,1,1,1,1,0,0,0,0],
        #     [1,1,1,1,1,0,0,0,0],
        #     [1,1,1,1,1,0,0,0,0],
        #     [0,0,0,0,0,0,0,0,0],
        #     [0,0,0,0,0,0,0,0,0]]
        # ]

        # flatten to pass it to GPT like this: [1,1,1,1,1,0,0,0,0,  1,1,1,1,1,0,0,0,0,  1,1,1,1,1,0,0,0,0,  0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0]
        stacked_attention_mask = stacked_attention_mask.reshape(batch_size, n_tokens * sequence_length)

        # inside, the GPT will transform that HF mask in the real one used in attention formula (dim: SEQ_LEN x SEQ_LEN)

        # the casual triangular mask built internally, seems not to enable tokens (i.e. PATCH_11) from same timestamp to consider the weight from
        # other tokens belonging to same timestamp (i.e. PATCH_15)

        # GPT forward pass
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs, # b, n_tokens * l, hidden_size
            attention_mask=stacked_attention_mask, # b, n_tokens * l
        )
        x = transformer_outputs['last_hidden_state']
        x = x.reshape(batch_size, sequence_length, n_tokens, self.hidden_size) # b, l, n_tokens, hidden_size

        # Action prediction
        if self.act_pred:
            # extract ACT token and forecast action
            action_embedding = x[:, :, act_query_token_start_i] # (b, l, hidden_size) 
            for pred_act_mlp in self.pred_act_mlps:
                action_embedding = pred_act_mlp(action_embedding)
            arm_action_preds = self.pred_arm_act(action_embedding)  # (b, l, act_dim - 1)
            gripper_action_preds = self.pred_gripper_act(action_embedding)  # (b, l, 1)
            
        # Forward prediction
        if self.fwd_pred:
            # The model copies this same learned vector for every patch position that needs to be reconstructed (for each timestamp)
            mask_token = self.mask_token  # (1, 1, 1, h)
            mask_tokens = mask_token.repeat(batch_size, sequence_length, (self.image_size//self.patch_size)**2, 1)  # (b, l, n_patches, h)
            mask_tokens = mask_tokens + self.decoder_pos_embed.unsqueeze(0).repeat(batch_size, sequence_length, 1, 1)  # (b, l, n_patches, h) + (b, l, n_patches, h)
            # extract OBS tokens and forecast future OBS
            obs_pred = self.decoder_embed(x[:, :, obs_query_token_start_i:(obs_query_token_start_i + self.n_patch_latents + n_obs_tokens)])  # (b, l, n_patch_latents + 1, h)
            obs_pred_ = torch.cat([obs_pred, mask_tokens], dim=2)  # (b, l, n_patch_latens + 1 + n_patches, h) --> [OBS_QUERY_1, OBS_QUERY_2, ..., MASK_PATCH_1, MASK_PATCH_2, ...]
            obs_pred_ = obs_pred_.reshape(-1, obs_pred_.shape[-2], obs_pred_.shape[-1])  # (b * l, n_patch_latens + 1 + n_patches, h)
            for blk in self.decoder_blocks:
                obs_pred_ = blk(obs_pred_)
            obs_pred_ = self.decoder_norm(obs_pred_)
            obs_preds = self.decoder_pred(obs_pred_)  # (b * l, n_patch_latens + 1 + n_patches, patch_size**2 * 3)   
            obs_preds = obs_preds.reshape(batch_size, sequence_length, -1, obs_preds.shape[-1])  # (b, l, n_patch_latens + 1 + n_patches, patch_size**2 * 3)
            obs_preds = obs_preds[:, :, (self.n_patch_latents+n_obs_tokens):]  # (b, l, n_patches, patch_size**2 * 3)

            if self.fwd_pred_hand:
                obs_pred_hand = self.decoder_embed(x[:, :, obs_hand_query_token_start_i:(obs_hand_query_token_start_i + self.n_patch_latents + n_obs_tokens)])
                obs_pred_hand_ = torch.cat([obs_pred_hand, mask_tokens], dim=2)
                obs_pred_hand_ = obs_pred_hand_.reshape(-1, obs_pred_hand_.shape[-2], obs_pred_hand_.shape[-1])
                for blk in self.decoder_blocks:
                    obs_pred_hand_ = blk(obs_pred_hand_)
                obs_pred_hand_ = self.decoder_norm(obs_pred_hand_)
                obs_hand_preds = self.decoder_pred(obs_pred_hand_)
                obs_hand_preds = obs_hand_preds.reshape(batch_size, sequence_length, -1, obs_hand_preds.shape[-1])
                obs_hand_preds = obs_hand_preds[:, :, (self.n_patch_latents+n_obs_tokens):]
        
        prediction = {
            'obs_preds': obs_preds,
            'obs_targets': obs_targets,
            'obs_hand_preds': obs_hand_preds,
            'obs_hand_targets': obs_hand_targets,
            'arm_action_preds': arm_action_preds,
            'gripper_action_preds': gripper_action_preds,
        }
        return prediction
    
