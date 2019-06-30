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


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)  # Default momentum: 0.9 (pytorch uses 1-momentum)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

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

        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.node_feats, kernel_size=1),
            nn.BatchNorm2d(self.node_feats),
            nn.ReLU(inplace=True)
        )

        self.num_gcn_layers = 8
        self.gcn_layers = nn.ModuleList([
            GraphConv(self.node_feats, self.node_feats, num_graphs, residual=False)
            for i in range(self.num_gcn_layers)
        ])

        total_feat_size = self.node_feats * num_nodes

        # One LSTM for each GCN layer output
        print('total_feat_size', total_feat_size)
        self.lstm_hidden_size = 50
        self.lstms = nn.ModuleList([
            nn.LSTM(input_size=total_feat_size, hidden_size=self.lstm_hidden_size, bidirectional=True, batch_first=True)
            for i in range(self.num_gcn_layers + 1) # 1 extra LSTM for original input sequence
        ])

        # Edge importance on Graph
        if use_edge_importance:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in range(self.num_gcn_layers)
            ])
        else:
            self.edge_importance = [1] * self.num_gcn_layers

        # Output channels of concatenation
        self.out_node_feats = (self.num_gcn_layers + 1) * self.lstm_hidden_size

        # Last layer
        self.fc = nn.Linear(self.out_node_feats, num_class)
        # self.fc = nn.Conv2d(self.out_node_feats, num_class, kernel_size=1)


    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)

        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)

        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)      # Batch, Channels, Time, Nodes

        cur_input = self.input_conv(x)
        spatial_stages = [cur_input]
        for gcn, importance in zip(self.gcn_layers, self.edge_importance):
            cur_input, _ = gcn(cur_input, self.A * importance)
            spatial_stages.append(cur_input)

        lstm_outs = []
        for stage, layer_lstm in zip(spatial_stages, self.lstms):
            # Merge the node features into a single vector: (Time, Batch, Node features)
            stage_transposed = stage.permute(0, 2, 1, 3).contiguous()
            stage_transposed = stage_transposed.view(-1, T, self.node_feats * V)
            layer_lstm.flatten_parameters()
            lstm_out, (h_state, c_state) = layer_lstm(stage_transposed)
            lstm_outs.append(h_state[0])

        x = torch.cat(lstm_outs, dim=1)
        x = x.view(N, M, -1).mean(dim=1)
        x = self.fc(x)

        # TODO:
        # Problems with this approach:
        # - wasted spatial information between nodes, since LSTM take concatenated vector
        # - final FC may bring too many parameters; LSTM too has many parameters
        # - different person should contribute to different weighting in the outputs; currently a simple average is done
        # FIXME:
        # Solutions:
        # - Graph Embedding (e.g. DeepWalk) to compress the graph into a vector for the LSTM
        return x

'''

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
        # self.node_feats = 16

        # self.input_gcn_layer = GraphConv(in_channels, self.node_feats, num_graphs, residual=False)
        # self.input_conv = BasicConv2d(in_channels, self.node_feats, kernel_size=1, padding=0)
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.node_feats, kernel_size=1),
            nn.BatchNorm2d(self.node_feats),
            nn.ReLU(inplace=True)
        )

        self.num_gcn_layers = 8
        self.gcn_layers = nn.ModuleList([
            # GraphConv(self.node_feats, self.node_feats, num_graphs)
            GraphConv(self.node_feats * (i+1), self.node_feats, num_graphs, residual=False)
            for i in range(self.num_gcn_layers)
        ])

        # Change the dimensions of the first GCN layer (which increases C)
        # self.gcn_layers[0] = GraphConv(in_channels, self.node_feats, num_graphs, residual=False)

        # Edge importance on Graph
        if use_edge_importance:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in range(self.num_gcn_layers)
            ])
        else:
            self.edge_importance = [1] * self.num_gcn_layers

        # Output channels of concatenation
        # self.out_node_feats = self.num_gcn_layers * self.node_feats
        self.out_node_feats = (self.num_gcn_layers + 1) * self.node_feats

        # # SE layer for stacked GCN features
        # self.se_layer = SELayer(
        #     channel=(self.out_node_feats),
        #     reduction=8)

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

        cur_input = self.input_conv(x)

        # Forward through GCN layers
        # gcn_outputs = []
        for gcn, importance in zip(self.gcn_layers, self.edge_importance):
            # x, _ = gcn(x, self.A * importance)
            out, _ = gcn(cur_input, self.A * importance)
            cur_input = torch.cat((cur_input, out), 1)
            # gcn_outputs.append(x)
            # gcn_outputs.append(out)

        # !!! NOTE: Stack along the node feature dimension
        # x = torch.cat(gcn_outputs, dim=1)
        x = cur_input

        # # Apply Sqeeuze and Excitation
        # x = self.se_layer(x)

        # Apply temporal convs
        for tconv in self.temporal_conv_layers:
            x = tconv(x)

        # Global average over each frame & each node
        # Average pool window size = (Batch, Node features)
        # i.e. left with (Batch, Node channels, 1, 1)
        x = F.avg_pool2d(x, x.size()[2:])
        # Average over the number of persons
        x = x.view(N, M, -1, 1, 1).mean(dim=1)
        # Prediction with fully connected
        x = self.fc(x).view(x.size(0), -1)
        return x
'''


if __name__ == "__main__":
    in_channels = 3
    out_channels = 2
    num_class = 30
    # self, in_channels, num_class, graph_args, edge_importance_weighting, **kwargs
    model = Model(in_channels, num_class, dict(strategy='spatial'), True)

    # print('Model details:')
    # print(model)

    # x = x.view(N * M, V * C, T)
    N = 5       # Batch size
    C = 3       # channels
    T = 300  # The sequence dimension will shrink (bc conv with stride 2)
    V = 18      # number of nodes
    M = 2       # number of people in the frame

    print(f'N={N}, C={C}, T={T}, V={V}, M={M}')
    x = torch.randn(N, C, T, V, M)

    # from torchsummary import summary
    # ins = tuple(x.size()[1:])
    # print(ins)
    # summary(model, input_size=ins)

    model.forward(x)

    num_params = 0
    for params in model.parameters():
        num_params += params.numel() if params.requires_grad else 0
    print(f'Model total number of params: {num_params}')