import torch
import torch.nn as nn


class GAM(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                channels,
                channels // reduction,
                1
            ),
            nn.ReLU(),
            nn.Conv2d(
                channels // reduction,
                channels,
                1
            ),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                7,
                padding=3,
                groups=channels
            ),
            nn.Sigmoid()
        )


    def forward(self, x):

        ca = self.channel_attention(x)
        x = x * ca

        sa = self.spatial_attention(x)
        x = x * sa

        return x