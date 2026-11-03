"""
Radar-SwinIR: Lightweight Swin Transformer for Range-Doppler Map Denoising
and Micro-Doppler Harmonic Super-Resolution.
Adapted for 2-channel Radar feature tensors [Batch, 2, H, W].
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Partitions input tensor [B, H, W, C] into non-overlapping windows [num_windows*B, window_size, window_size, C].
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    Reverses window_partition back to [B, H, W, C].
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """
    Shifted Window Multi-head Self Attention (W-MSA / SW-MSA) tailored for 2D Doppler features.
    """
    def __init__(self, dim: int, window_size: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table for Doppler spatial-frequency geometry
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Coordinate grid for relative indexing
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # Relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    Single Swin Transformer Block with alternating regular and shifted window self-attention.
    """
    def __init__(self, dim: int, num_heads: int, window_size: int = 8, shift_size: int = 0, mlp_ratio: float = 2.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim=dim, window_size=window_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, x: torch.Tensor, x_size: Tuple[int, int]) -> torch.Tensor:
        H, W = x_size
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Window partition
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + x

        # FFN
        x = x + self.mlp(self.norm2(x))
        return x


class ResidualSwinBlock(nn.Module):
    """
    Residual Swin Transformer Block (RSTB) grouping multiple Swin layers with residual convolution.
    """
    def __init__(self, dim: int, num_heads: int = 4, window_size: int = 8, depth: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2
            )
            for i in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, x_size: Tuple[int, int]) -> torch.Tensor:
        res = x
        for blk in self.blocks:
            x = blk(x, x_size)

        B, L, C = x.shape
        H, W = x_size
        x_conv = x.transpose(1, 2).view(B, C, H, W)
        x_conv = self.conv(x_conv)
        x = x_conv.flatten(2).transpose(1, 2)
        return res + x


class RadarSwinIR(nn.Module):
    """
    Complete Lightweight Radar-SwinIR Denoising & Super-Resolution Model.
    Inputs: [Batch, 2, 64, 64] (Noisy, Low-SNR Range-Doppler + Micro-Doppler Maps)
    Outputs: [Batch, 2, 64, 64] (Restored, Crisp, High-SNR Feature Maps)
    """
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        embed_dim: int = 36,
        num_blocks: int = 2,
        num_heads: int = 4,
        window_size: int = 8,
        depth: int = 2
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.window_size = window_size

        # 1. Shallow Feature Extraction
        self.conv_first = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)

        # 2. Deep Feature Extraction (Residual Swin Blocks)
        self.rstb_blocks = nn.ModuleList([
            ResidualSwinBlock(dim=embed_dim, num_heads=num_heads, window_size=window_size, depth=depth)
            for _ in range(num_blocks)
        ])
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)

        # 3. High-Quality Reconstruction Head with Residual Connection
        self.conv_last = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(embed_dim, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual shortcut from input
        residual = x

        # 1. Shallow features
        feat = self.conv_first(x)
        H, W = feat.shape[2], feat.shape[3]
        x_flat = feat.flatten(2).transpose(1, 2)  # [B, H*W, C]

        # 2. Deep Swin features
        for rstb in self.rstb_blocks:
            x_flat = rstb(x_flat, (H, W))

        feat_deep = x_flat.transpose(1, 2).view(feat.shape[0], self.embed_dim, H, W)
        feat_body = self.conv_after_body(feat_deep) + feat

        # 3. Reconstruction
        out = self.conv_last(feat_body)

        # Add residual skip to learn additive correction / noise removal
        restored = torch.clamp(residual + out, 0.0, 1.0)
        return restored


def get_radar_swinir(
    device: str = "cpu",
    embed_dim: int = 36,
    num_blocks: int = 2,
    num_heads: int = 4,
    window_size: int = 8,
    depth: int = 2
) -> RadarSwinIR:
    model = RadarSwinIR(
        in_channels=2,
        out_channels=2,
        embed_dim=embed_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        window_size=window_size,
        depth=depth
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Radar-SwinIR] Initialized Model | Trainable Params: {total_params:,} (~{total_params * 4 / 1e6:.2f} MB)")
    return model


if __name__ == "__main__":
    dummy = torch.randn(2, 2, 64, 64)
    net = get_radar_swinir()
    res = net(dummy)
    print("Forward pass successful. Output shape:", res.shape)
