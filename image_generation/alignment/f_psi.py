from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers.models.embeddings import TimestepEmbedding, Timesteps, get_timestep_embedding
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle, GlobalResponseNorm, RMSNorm
from diffusers.models.embeddings import CombinedTimestepGuidanceTextProjEmbeddings, CombinedTimestepTextProjEmbeddings,TimestepEmbedding, get_timestep_embedding #,FluxPosEmbed
from diffusers.models.activations import get_activation

from meissonic.transformer import UVit2DConvEmbed, FluxPosEmbed, SingleTransformerBlock

class FpsiModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        pooled_projection_dim = 1024
        micro_cond_embed_dim = 1280
        inner_dim = 1024
        use_bias = False

        text_time_guidance_cls = CombinedTimestepTextProjEmbeddings
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=inner_dim, pooled_projection_dim=pooled_projection_dim
        )
        self.cond_embed = TimestepEmbedding( 
            micro_cond_embed_dim + pooled_projection_dim, inner_dim, sample_proj_bias=use_bias
        )

        joint_attention_dim = 1024
        self.context_embedder = nn.Linear(joint_attention_dim, inner_dim)

        num_attention_heads = 8
        attention_head_dim = 128
        self.transformer_blocks = nn.ModuleList(
            [
                SingleTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for i in range(3)
            ]
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(inner_dim, 1)

    def forward(self, timesteps, micro_conds, pooled_projections, encoder_hidden_states):
        dtype = next(self.cond_embed.parameters()).dtype # float32
        micro_cond_encode_dim = 256

        micro_cond_embeds = get_timestep_embedding(
            micro_conds.flatten(), micro_cond_encode_dim, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        micro_cond_embeds = micro_cond_embeds.reshape((timesteps.shape[0], -1)) 
        pooled_projections = torch.cat([pooled_projections, micro_cond_embeds], dim=1) # cat([bf16, float32]) -> float32
        pooled_projections = self.cond_embed(pooled_projections)

        hidden_states = self.context_embedder(encoder_hidden_states.to(dtype=dtype)) # bf16 -> float32

        timesteps = timesteps * 1000
        temb = self.time_text_embed(timesteps, pooled_projections)
        
        for index_block, block in enumerate(self.transformer_blocks):
            hidden_states = block(
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=None,
            )

        hidden_states = self.pool(hidden_states.permute(0, 2, 1)).squeeze(-1)
        out = self.fc(hidden_states).squeeze()

        return out
