# models, gnn architectures and readout functions taken from dgl-lifesci on 06/08/2020
# https://github.com/awslabs/dgl-lifesci

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import edge_softmax
import torch.nn as nn
from dgl.nn.pytorch.glob import GlobalAttentionPooling, SumPooling, AvgPooling, MaxPooling
from torch.distributions import Normal
from dgl.nn import GraphConv
from dgl.nn.pytorch import Set2Set
from dgl.nn.pytorch import NNConv,GATConv
from modifiedGAP import GlobalAttentionPoolingPMG

class Classifier_gen(nn.Module):
    def __init__(self, in_dim, hidden_dim_graph,hidden_dim1,n_classes,dropout,num_layers,pooling):
        super(Classifier_gen, self).__init__()
        if num_layers ==1:
            self.conv1 = GraphConv(in_dim, hidden_dim1)
        if num_layers==2:
            self.conv1 = GraphConv(in_dim, hidden_dim_graph)
            self.conv2 = GraphConv(hidden_dim_graph, hidden_dim1)
        if pooling == 'att':
            pooling_gate_nn = nn.Linear(hidden_dim1, 1)
            self.pooling = GlobalAttentionPoolingPMG(pooling_gate_nn)
        self.classify = nn.Sequential(nn.Linear(hidden_dim1,hidden_dim1),nn.Dropout(dropout))
        self.classify2 = nn.Sequential(nn.Linear(hidden_dim1, n_classes),nn.Dropout(dropout))
        self.out_act = nn.Sigmoid()
    def forward(self, g,num_layers,pooling):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
        # Perform graph convolution and activation function.
        # Calculate graph representation by averaging all the node representations.
        h = g.ndata['h_n'].float()
        # Perform graph convolution and activation function.
        h = F.relu(self.conv1(g, h))
        if num_layers=='2':
            h = F.relu(self.conv2(g, h))
            
        g.ndata['h'] = h
        # Calculate graph representation by averaging all the node representations.
        #hg = dgl.mean_nodes(g, 'h')
        if pooling == "max":
            hg = dgl.max_nodes(g, 'h')
        elif pooling=="mean":
            hg = dgl.mean_nodes(g, 'h')
        elif pooling == "sum":
            hg = dgl.sum_nodes(g, 'h') 
        elif pooling =='att':  
            # Calculate graph representation by averaging all the node representations.
            [hg,g2] = self.pooling(g,h) 
        
        g2 = hg
        a2=self.classify(hg)
        a3=self.classify2(a2)
        return self.out_act(a3),g2,hg
