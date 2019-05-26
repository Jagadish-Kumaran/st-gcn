import torch
import torch.nn as nn

from net.se import SELayer


class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_graphs, residual=True):
        super().__init__()

        self.residual = residual
        # Here, `num_graphs` = # graphs (depending on config, e.g. spatial gives 3)
        self.num_graphs = num_graphs
        self.expand_channel_conv = nn.Conv2d(
            in_channels,
            out_channels * num_graphs,
            kernel_size=1,  # Use 1x1 conv since we are just making more channels
            padding=0,
            stride=1,
            dilation=1,
            bias=True
        )

        # self.se = SELayer(
        #     channel=out_channels,
        #     reduction=8)

        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x, A):
        # x.shape = (# batches, # channels, # frames, # nodes)
        # A.shape = (# graphs (e.g. due to spatial config), # nodes, # nodes (Adjacency))

        x_original = x

        assert A.size(0) == self.num_graphs, '# graphs do not match'

        x = self.expand_channel_conv(x)

        # (# batchs, # channels, # frames, # nodes)
        N, C, T, V = x.size()

        x = x.view(N, self.num_graphs, C // self.num_graphs, T, V)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        x = x.contiguous()

        # # Apply Squeeze & Excitation!
        # x = self.se(x)

        if self.residual:
            x += x_original

        x = self.relu(x)
        x = self.batchnorm(x)

        return x, A