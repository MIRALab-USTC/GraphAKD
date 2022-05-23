import torch as th
from torch import nn
from torch.nn import init
import dgl.function as fn
from dgl.ops import edge_softmax


class GraphConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 bias=True,
                 activation=None):
        super(GraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats

        self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        r"""
        Description
        -----------
        Reinitialize learnable parameters.
        Note
        ----
        The model parameters are initialized as in the
        `original implementation <https://github.com/tkipf/gcn/blob/master/gcn/layers.py>`__
        where the weight :math:`W^{(l)}` is initialized using Glorot uniform initialization
        and the bias is initialized to be zero.
        """
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, graph, feat):
        r"""
        Description
        -----------
        Compute graph convolution.
        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
        Returns
        -------
        torch.Tensor
            The output feature
        Note
        ----
        * Input shape: :math:`(N, *, \text{in_feats})` where * means any number of additional
          dimensions, :math:`N` is the number of nodes.
        * Output shape: :math:`(N, *, \text{out_feats})` where all but the last dimension are
          the same shape as the input.
        * Weight shape: :math:`(\text{in_feats}, \text{out_feats})`.
        """
        with graph.local_scope():
            degs = graph.out_degrees().float().clamp(min=1)
            norm = th.pow(degs, -0.5).unsqueeze(-1)
            feat = feat * norm
            weight = self.weight

            if self._in_feats > self._out_feats:
                # mult W first to reduce the feature size for aggregation.
                feat = th.matmul(feat, weight)
                graph.srcdata['h'] = feat
                graph.update_all(fn.copy_u('h', 'm'), fn.sum(msg='m', out='h'))
                # if 'z' not in graph.edata.keys():
                #     graph.update_all(fn.copy_u('h', 'm'), fn.sum(msg='m', out='h'))
                # else:
                #     # graph.edata['z'] = edge_softmax(graph, graph.edata['z'])
                #     graph.update_all(fn.u_mul_e('h', 'z', 'm'), fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
            else:
                # aggregate first then mult W
                graph.srcdata['h'] = feat
                graph.update_all(fn.copy_u('h', 'm'), fn.sum(msg='m', out='h'))
                # if 'z' not in graph.edata.keys():
                #     graph.update_all(fn.copy_u('h', 'm'), fn.sum(msg='m', out='h'))
                # else:
                #     graph.update_all(fn.u_mul_e('h', 'z', 'm'), fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
                rst = th.matmul(rst, weight)

            degs = graph.in_degrees().float().clamp(min=1)
            norm = th.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = th.reshape(norm, shp)
            rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst