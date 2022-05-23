"""GCN using DGL nn package

References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
# from gcnconv import GraphConv


class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        # self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)
        self.emb = None

        self.linear = nn.Linear(in_feats, n_hidden)

    def forward(self, features):
        h = features
        h = self.linear(h) # for yelp
        for i, layer in enumerate(self.layers):
            if i == len(self.layers)-1:
                # import ipdb; ipdb.set_trace()
                self.emb = h
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h
