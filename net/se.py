from torch import nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # X.shape: (N, C, T, V), to be pooled as (N, C, 1, 1)
        N, C, _, _ = x.size()
        y = self.avg_pool(x).view(N, C)
        y = self.fc(y).view(N, C, 1, 1)
        return x * y.expand_as(x)