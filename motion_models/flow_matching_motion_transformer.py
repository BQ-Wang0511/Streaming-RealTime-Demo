import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from .motion_transformer import (
    DecoderLayerStack,
    FiLMTransformerDecoderLayer,
    PositionalEncoding,
    RotaryEmbedding,
    SinusoidalPosEmb,
    TransformerEncoderLayer,
    prob_mask_like,
)


class FlowMatchingMotionTransformer(nn.Module):
    def __init__(
        self,
        nfeats,
        person_num,
        seq_len=25,
        latent_dim=256,
        ff_size=1024,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
        cond_feature_dim=512,
        activation=F.gelu,
        use_rotary=True,
    ):
        super().__init__()
        if person_num <= 0:
            raise ValueError("person_num must be positive for habit-conditioned generation")

        self.seq_len = seq_len
        self.person_num = person_num
        self.rotary = RotaryEmbedding(dim=latent_dim) if use_rotary else None
        self.abs_pos_encoding = nn.Identity()
        if not use_rotary:
            self.abs_pos_encoding = PositionalEncoding(
                latent_dim,
                dropout=dropout,
                batch_first=True,
            )

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(latent_dim),
            nn.Linear(latent_dim, latent_dim * 4),
            nn.Mish(),
        )
        self.to_time_cond = nn.Sequential(nn.Linear(latent_dim * 4, latent_dim))
        self.to_time_tokens = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim * 2),
            Rearrange("b (r d) -> b r d", r=2),
        )

        self.null_cond_embed = nn.Parameter(torch.randn(1, seq_len, latent_dim))
        self.null_cond_hidden = nn.Parameter(torch.randn(1, latent_dim))
        self.null_habit_tokens = nn.Parameter(torch.randn(1, 2, latent_dim))
        self.null_habit_hidden = nn.Parameter(torch.randn(1, latent_dim))
        self.norm_cond = nn.LayerNorm(latent_dim)

        self.input_projection = nn.Linear(nfeats * 2, latent_dim)
        self.cond_projection = nn.Linear(cond_feature_dim, latent_dim)
        self.cond_encoder = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
                for _ in range(2)
            ]
        )
        self.non_attn_cond_projection = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        self.habit_projection = nn.Linear(person_num, latent_dim, bias=False)
        self.habit_to_hidden = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.habit_to_tokens = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim * 2),
            Rearrange("b (r d) -> b r d", r=2),
        )
        self.seq_decoder = DecoderLayerStack(
            [
                FiLMTransformerDecoderLayer(
                    latent_dim,
                    num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer = nn.Linear(latent_dim, nfeats)

    def guided_forward(
        self,
        x,
        cond_frame,
        cond_embed,
        habit_one_hot,
        times,
        guidance_weight,
    ):
        uncond = self.forward(
            x,
            cond_frame,
            cond_embed,
            habit_one_hot,
            times,
            cond_drop_prob=1.0,
        )
        conditioned = self.forward(
            x,
            cond_frame,
            cond_embed,
            habit_one_hot,
            times,
            cond_drop_prob=0.0,
        )
        return uncond + (conditioned - uncond) * guidance_weight

    def forward(
        self,
        x,
        cond_frame,
        cond_embed,
        habit_one_hot,
        times,
        cond_drop_prob=0.0,
    ):
        batch_size, device = x.shape[0], x.device
        if habit_one_hot is None:
            habit_one_hot = torch.zeros(batch_size, self.person_num, device=device, dtype=x.dtype)
        else:
            habit_one_hot = habit_one_hot.to(device=device, dtype=x.dtype)

        cond_frame = cond_frame.unsqueeze(1).repeat(1, x.shape[1], 1)
        x = torch.cat([x, cond_frame], dim=-1)
        x = self.input_projection(x)
        x = self.abs_pos_encoding(x)

        keep_mask = prob_mask_like((batch_size,), 1 - cond_drop_prob, device=device)
        keep_mask_embed = rearrange(keep_mask, "b -> b 1 1")
        keep_mask_hidden = rearrange(keep_mask, "b -> b 1")

        cond_tokens = self.cond_projection(cond_embed)
        cond_tokens = self.abs_pos_encoding(cond_tokens)
        for layer in self.cond_encoder:
            cond_tokens = layer(cond_tokens)

        null_cond_embed = self.null_cond_embed[:, : cond_tokens.shape[1]].to(cond_tokens.dtype)
        cond_tokens = torch.where(keep_mask_embed, cond_tokens, null_cond_embed)
        mean_pooled_cond_tokens = cond_tokens.mean(dim=-2)
        cond_hidden = self.non_attn_cond_projection(mean_pooled_cond_tokens)

        habit_embed = self.habit_projection(habit_one_hot)
        habit_hidden = self.habit_to_hidden(habit_embed)
        habit_tokens = self.habit_to_tokens(habit_embed)

        null_habit_tokens = self.null_habit_tokens.to(habit_tokens.dtype)
        null_habit_hidden = self.null_habit_hidden.to(habit_hidden.dtype)
        habit_tokens = torch.where(keep_mask_embed, habit_tokens, null_habit_tokens)
        habit_hidden = torch.where(keep_mask_hidden, habit_hidden, null_habit_hidden)

        t_hidden = self.time_mlp(times)
        time_cond = self.to_time_cond(t_hidden)
        time_tokens = self.to_time_tokens(t_hidden)

        null_cond_hidden = self.null_cond_hidden.to(time_cond.dtype)
        cond_hidden = torch.where(keep_mask_hidden, cond_hidden, null_cond_hidden)
        time_cond = time_cond + cond_hidden + habit_hidden

        cond_tokens = self.norm_cond(torch.cat((cond_tokens, habit_tokens, time_tokens), dim=-2))
        output = self.seq_decoder(x, cond_tokens, time_cond)
        return self.final_layer(output)
