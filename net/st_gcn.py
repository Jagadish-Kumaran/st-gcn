# import sys
# import os
# # Add parent directory
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

import torch
import torch.nn.functional as F

from torch import nn

from net.se import SELayer
from net.gcn import GraphConv
from net.tcn import TemporalConv
from net.utils.graph import Graph


class Model(nn.Module):
    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()

        use_edge_importance = edge_importance_weighting

        # load graph
        self.graph = Graph(**graph_args)
        # Adjacency matrix
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # BN layer for input batch
        num_graphs, num_nodes, _ = A.size()
        self.data_bn = nn.BatchNorm1d(in_channels * num_nodes)

        # Size of node feature vector
        self.node_feats = 32

        # self.input_gcn_layer = GraphConv(in_channels, self.node_feats, num_graphs, residual=False)

        self.num_gcn_layers = 8
        self.gcn_layers = nn.ModuleList([
            GraphConv(self.node_feats, self.node_feats, num_graphs)
            for i in range(self.num_gcn_layers)
        ])

        # Change the dimensions of the first GCN layer (which increases C)
        self.gcn_layers[0] = GraphConv(in_channels, self.node_feats, num_graphs, residual=False)

        # Edge importance on Graph
        if use_edge_importance:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in range(self.num_gcn_layers)
            ])
        else:
            self.edge_importance = [1] * self.num_gcn_layers


        # SE layer for stacked GCN features
        self.out_node_feats = self.num_gcn_layers * self.node_feats
        self.se_layer = SELayer(
            channel=(self.out_node_feats),
            reduction=8)

        # Temporal conv layers to squash sequence
        temp_kernel_size = 3
        self.temporal_conv_layers = nn.ModuleList([
            TemporalConv(self.out_node_feats, temp_kernel_size),
            TemporalConv(self.out_node_feats, temp_kernel_size),
            TemporalConv(self.out_node_feats, temp_kernel_size, temp_stride=2),
            TemporalConv(self.out_node_feats, temp_kernel_size),
            TemporalConv(self.out_node_feats, temp_kernel_size),
            TemporalConv(self.out_node_feats, temp_kernel_size, temp_stride=2),
            TemporalConv(self.out_node_feats, temp_kernel_size),
        ])

        # Last layer
        self.fc = nn.Conv2d(self.out_node_feats, num_class, kernel_size=1)


    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)

        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)

        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # Forward through GCN layers
        gcn_outputs = []
        for gcn, importance in zip(self.gcn_layers, self.edge_importance):
            x, _ = gcn(x, self.A * importance)
            gcn_outputs.append(x)

        # !!! NOTE: Stack along the node feature dimension
        x = torch.cat(gcn_outputs, dim=1)

        # Apply Sqeeuze and Excitation
        x = self.se_layer(x)

        # Apply temporal convs
        for tconv in self.temporal_conv_layers:
            x = tconv(x)

        # Global average over each frame & each node
        x = F.avg_pool2d(x, x.size()[2:])

        # Average over the number of persons
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # Prediction with fully connected
        x = self.fc(x).view(x.size(0), -1)

        return x


# if __name__ == "__main__":
#     in_channels = 3
#     out_channels = 2
#     num_class = 30
#     # self, in_channels, num_class, graph_args, edge_importance_weighting, **kwargs
#     model = Model(in_channels, num_class, dict(strategy='spatial'), True)

#     # print('Model details:')
#     # print(model)

#     # x = x.view(N * M, V * C, T)
#     N = 5
#     C = 3
#     T = 300  # The sequence dimension will shrink (bc conv with stride 2)
#     V = 18
#     M = 2

#     print(f'N={N}, C={C}, T={T}, V={V}, M={M}')
#     x = torch.randn(N, C, T, V, M)

#     from torchsummary import summary
#     ins = tuple(x.size()[1:])
#     print(ins)
#     summary(model, input_size=ins)

#     # model.forward(x)