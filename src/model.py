"""
AeroRadarNet: Physics-Aware Doppler-CNN Architecture for 77 GHz Drone Surveillance.
Combines Doppler-Harmonic Separable Convolutions with Channel Attention.
Ultra-lightweight (<250k params) and optimized for real-time edge execution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

try:
    from src import config
except ImportError:
    import config


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Attention Block.
    Recalibrates channel-wise feature responses to emphasize distinctive Doppler frequencies.
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.fc(x).view(b, c, 1, 1)
        return x * w


class DopplerSeparableBlock(nn.Module):
    """
    Physics-Aware Doppler Convolutions:
    - Vertical (7x1) kernel: Captures high-frequency rotor blade Doppler harmonics
    - Horizontal (1x5) kernel: Captures dwell-time cadence dynamics
    - Pointwise (1x1) conv: Channel projection
    - SE Attention + Residual Connection
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.stride = stride

        # Vertical kernel: Doppler velocity harmonics
        self.freq_conv = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=(7, 1),
            stride=(stride, 1),
            padding=(3, 0),
            groups=in_channels,
            bias=False
        )
        # Horizontal kernel: Temporal cadence
        self.time_conv = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=(1, 5),
            stride=(1, stride),
            padding=(0, 2),
            groups=in_channels,
            bias=False
        )
        # Pointwise projection
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.act = nn.SiLU()

        # Shortcut for residual
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.freq_conv(x)
        out = self.time_conv(out)
        out = self.pointwise(out)
        out = self.bn(out)
        out = self.se(out)
        return self.act(out + res)


class AeroRadarNet(nn.Module):
    """
    Complete Lightweight Doppler-CNN for Airspace Surveillance.
    Inputs: [Batch, 2, 64, 64]
      - Channel 0: Range-Doppler Map
      - Channel 1: Micro-Doppler Time-Frequency Spectrogram
    Outputs: [Batch, num_classes] logits
    """
    def __init__(self, in_channels: int = 2, num_classes: int = 3, dropout_rate: float = 0.3):
        super().__init__()
        self.num_classes = num_classes

        # Stem: Initial feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )

        # Stage 1: Spatial 64x64 -> 32x32
        self.stage1 = nn.Sequential(
            DopplerSeparableBlock(32, 48, stride=2),
            DopplerSeparableBlock(48, 48, stride=1)
        )

        # Stage 2: Spatial 32x32 -> 16x16
        self.stage2 = nn.Sequential(
            DopplerSeparableBlock(48, 96, stride=2),
            DopplerSeparableBlock(96, 96, stride=1)
        )

        # Stage 3: Spatial 16x16 -> 8x8
        self.stage3 = nn.Sequential(
            DopplerSeparableBlock(96, 160, stride=2),
            DopplerSeparableBlock(160, 160, stride=1)
        )

        # Global Pooling & Head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(160, 64),
            nn.SiLU(),
            self.dropout,
            nn.Linear(64, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        feat = self.stage1(feat)
        feat = self.stage2(feat)
        feat = self.stage3(feat)
        feat = self.global_pool(feat)
        return torch.flatten(feat, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.extract_features(x)
        logits = self.classifier(embeddings)
        return logits


def get_model(num_classes: int = 3, device: str = "cpu") -> AeroRadarNet:
    model = AeroRadarNet(in_channels=config.INPUT_CHANNELS, num_classes=num_classes)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Initialized AeroRadarNet (Classes={num_classes}) | Trainable Params: {total_params:,} (~{total_params * 4 / 1e6:.2f} MB)")
    return model


if __name__ == "__main__":
    dummy_input = torch.randn(2, 2, 64, 64)
    net = get_model(num_classes=3)
    out = net(dummy_input)
    print("Forward pass successful. Output shape:", out.shape)
