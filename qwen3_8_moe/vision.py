"""Qwen3.5/3.6/3.8 vision tower (shape-compatible with Qwen3.6-35B-A3B)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .configuration import Qwen38MoeVisionConfig
from .layers import rotate_half


def gelu_pytorch_tanh(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        return torch.outer(seq, self.inv_freq)


def apply_rotary_pos_emb_vision(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q, orig_k = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_q), k_embed.to(orig_k)


class VisionMLP(nn.Module):
    def __init__(self, config: Qwen38MoeVisionConfig) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(gelu_pytorch_tanh(self.linear_fc1(hidden_state)))


class VisionPatchEmbed(nn.Module):
    def __init__(self, config: Qwen38MoeVisionConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        kernel = (self.temporal_patch_size, self.patch_size, self.patch_size)
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel, stride=kernel, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        return self.proj(hidden_states.to(self.proj.weight.dtype)).view(-1, self.embed_dim)


class VisionPatchMerger(nn.Module):
    def __init__(self, config: Qwen38MoeVisionConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size ** 2)
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x).view(-1, self.hidden_size)
        return self.linear_fc2(F.gelu(self.linear_fc1(x)))


class VisionAttention(nn.Module):
    def __init__(self, config: Qwen38MoeVisionConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=True)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.scaling = self.head_dim ** -0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, self.head_dim).permute(1, 0, 2, 3).unbind(0)
        q, k = apply_rotary_pos_emb_vision(q, k, *position_embeddings)
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        outputs = []
        offset = 0
        for length in lengths:
            qi = q[:, offset : offset + length]
            ki = k[:, offset : offset + length]
            vi = v[:, offset : offset + length]
            attn = torch.softmax((qi @ ki.transpose(-1, -2)) * self.scaling, dim=-1)
            outputs.append((attn @ vi).transpose(0, 1))
            offset += length
        attn_output = torch.cat(outputs, dim=0).reshape(seq_length, -1)
        return self.proj(attn_output)


class VisionBlock(nn.Module):
    def __init__(self, config: Qwen38MoeVisionConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = VisionAttention(config)
        self.mlp = VisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cu_seqlens, position_embeddings)
        return hidden_states + self.mlp(self.norm2(hidden_states))


class Qwen38MoeVisionModel(nn.Module):
    def __init__(self, config: Qwen38MoeVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_embed = VisionPatchEmbed(config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings ** 0.5)
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList([VisionBlock(config) for _ in range(config.depth)])
        self.merger = VisionPatchMerger(config)

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size
        max_hw = int(max(max(h, w) for _, h, w in grid_thw.tolist()))
        freq_table = self.rotary_pos_emb(max_hw)
        embeddings = []
        for num_frames, height, width in grid_thw.tolist():
            merged_h, merged_w = height // merge_size, width // merge_size
            block_rows = torch.arange(merged_h, device=freq_table.device)
            block_cols = torch.arange(merged_w, device=freq_table.device)
            intra = torch.arange(merge_size, device=freq_table.device)
            row_idx = (block_rows[:, None, None, None] * merge_size + intra[None, None, :, None]).expand(
                merged_h, merged_w, merge_size, merge_size
            ).reshape(-1)
            col_idx = (block_cols[None, :, None, None] * merge_size + intra[None, None, None, :]).expand(
                merged_h, merged_w, merge_size, merge_size
            ).reshape(-1)
            coords = torch.stack((row_idx, col_idx), dim=-1)
            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)
            embeddings.append(freq_table[coords].flatten(1))
        return torch.cat(embeddings, dim=0)

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        device = self.pos_embed.weight.device
        patches = []
        for t, h, w in grid_thw.tolist():
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h, device=device)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w, device=device)
            hf, wf = h_idxs.int(), w_idxs.int()
            hc = (hf + 1).clamp(max=self.num_grid_per_side - 1)
            wc = (wf + 1).clamp(max=self.num_grid_per_side - 1)
            dh, dw = h_idxs - hf, w_idxs - wf
            idxs = [
                (hf[:, None] * self.num_grid_per_side + wf[None, :]).reshape(-1),
                (hf[:, None] * self.num_grid_per_side + wc[None, :]).reshape(-1),
                (hc[:, None] * self.num_grid_per_side + wf[None, :]).reshape(-1),
                (hc[:, None] * self.num_grid_per_side + wc[None, :]).reshape(-1),
            ]
            weights = [
                ((1 - dh)[:, None] * (1 - dw)[None, :]).reshape(-1),
                ((1 - dh)[:, None] * dw[None, :]).reshape(-1),
                (dh[:, None] * (1 - dw)[None, :]).reshape(-1),
                (dh[:, None] * dw[None, :]).reshape(-1),
            ]
            pos = sum(self.pos_embed(idx.long()) * weight[:, None] for idx, weight in zip(idxs, weights))
            pos = pos.repeat(t, 1)
            merge = self.spatial_merge_size
            pos = (
                pos.view(t, h // merge, merge, w // merge, merge, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patches.append(pos)
        return torch.cat(patches, dim=0)

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states) + self.fast_pos_embed_interpolate(grid_thw)
        rotary = self.rot_pos_emb(grid_thw)
        emb = torch.cat((rotary, rotary), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens, position_embeddings)
        return self.merger(hidden_states)
