import torch
from torch import nn

class TemporalConv(nn.Module):
    def __init__(self, channels, temp_kernel_size, temp_stride=1, residual=True):
        super().__init__()

        self.residual = residual

        self.temp_conv = nn.Sequential(
            # nn.BatchNorm2d(channels),
            # nn.ReLU(),
            nn.Conv2d(
                in_channels=channels,   # the # features do not change
                out_channels=channels,
                kernel_size=(temp_kernel_size, 1),
                stride=(temp_stride, 1),
                padding=((temp_kernel_size - 1) // 2, 0)
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

    def forward(self, x):
        result = self.temp_conv(x)
        if self.residual and result.size() == x.size():
            result += x
        return result