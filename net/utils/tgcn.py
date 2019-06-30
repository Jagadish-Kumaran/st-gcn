# The based unit of graph convolutional networks.

import torch
import torch.nn as nn

class ConvTemporalGraphical(nn.Module):

    r"""The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        # !!!! NOTE: this is the actual graph convolution

        # X.shape is (# batch, # channels, # frames, # nodes) = (N, C, T, V)
        print(f'In ConvTempGraphical: x.shape: {x.shape}')
        print(f'In ConvTempGraphical: A.shape: {A.shape}')  # 1, 18, 18
        print(f'In ConvTempGraphical: kernel_size: {self.kernel_size}') # 1

        assert A.size(0) == self.kernel_size

        # in_channels -> out_channels, so Cin -> Cout
        # Width, Height = (T, V) --> conv with (t_kernel_size, 1), so shrink the Temporal dimension
        # Since `t_kernel_size` = 1 and pad = 0 and stride = 1, all this conv does is just
        # altering the # channels to `in_channels` * `# graphs` (i.e. 1x1 conv)
        # This is so that each graph (e.g. 3 of them in Spatial config)
        # has enough channels for graph conv (since now # channels *= # graphs)
        x = self.conv(x)

        # TODO: understand this graph conv, and the "temporal conv"
        # TODO: and then change stuff if needed
        # TODO: Think about how to do concatenation given different feature map sizes

        # X now has dim (batch_size, channels, smaller sequence, same number of nodes)
        # T = number of frames, V = number of nodes
        n, c, t, v = x.size()

        # X is now (# Batch, A.size(0) == # graphs == kernelsize, channels / kernel_size, T, V)
        # Here, A.size(0) == # graphs == kernelsize is not necessarily 1, because
        # there can be multiple graphs depending on uniform/distance/spatial configuration
        # So this view separates the dimensions
        x = x.view(n, self.kernel_size, c//self.kernel_size, t, v)

        # X is now (# batch, # channels, # sequence, # nodes)
        # Here, each dimension of the graph (e.g. in Spatial config) is multiplied
        # with the corresponding dimension
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        print(f'in `tgcn.py`, x final shape: {x.shape}')

        return x.contiguous(), A
