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

class Classifier_oneGCN_sumP(nn.Module):
    def __init__(self, in_dim, hidden_dim_graph,hidden_dim1,n_classes,dropout,num_heads):
        super(Classifier_oneGCN_sumP, self).__init__()
        # self.conv1 = GraphConv(in_dim, hidden_dim_graph)
        # self.conv2 = GraphConv(hidden_dim_graph, hidden_dim1)
        self.conv1 = GraphConv(in_dim, hidden_dim1)
        pooling_gate_nn = nn.Linear(hidden_dim1, 1)
        self.pooling = SumPooling()
        self.classify = nn.Sequential(nn.Linear(hidden_dim1,hidden_dim1),nn.Dropout(dropout))
        self.classify2 = nn.Sequential(nn.Linear(hidden_dim1, n_classes),nn.Dropout(dropout))
        self.out_act = nn.Sigmoid()
    def forward(self, g):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
        h = g.ndata['h_n'].float()
        # Perform graph convolution and activation function.
        h = F.relu(self.conv1(g, h))
        g.ndata['h'] = h
        # Calculate graph representation by averaging all the node representations.
        hg = self.pooling(g,h)
        g2 = hg
        a2=self.classify(hg)
        a3=self.classify2(a2)
        return self.out_act(a3),g2,hg




class Classifier_twoGCN_avgP(nn.Module):
    def __init__(self, in_dim, hidden_dim_graph,hidden_dim1,n_classes,dropout,num_heads):
        super(Classifier_oneGCN_sumP, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim_graph)
        self.conv2 = GraphConv(hidden_dim_graph, hidden_dim1)
        #self.conv1 = GraphConv(in_dim, hidden_dim1)
        pooling_gate_nn = nn.Linear(hidden_dim1, 1)
        self.pooling = SumPooling()
        self.classify = nn.Sequential(nn.Linear(hidden_dim1,hidden_dim1),nn.Dropout(dropout))
        self.classify2 = nn.Sequential(nn.Linear(hidden_dim1, n_classes),nn.Dropout(dropout))
        self.out_act = nn.Sigmoid()
    def forward(self, g):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
        h = g.ndata['h_n'].float()
        # Perform graph convolution and activation function.
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        g.ndata['h'] = h
        # Calculate graph representation by averaging all the node representations.
        hg = self.pooling(g,h)
        g2 = hg
        a2=self.classify(hg)
        a3=self.classify2(a2)
        return self.out_act(a3),g2,hg
class Classifier_oneGCN_sumP(nn.Module):
    def __init__(self, in_dim, hidden_dim_graph,hidden_dim1,n_classes,dropout,num_heads):
        super(Classifier_oneGCN_sumP, self).__init__()
        # self.conv1 = GraphConv(in_dim, hidden_dim_graph)
        # self.conv2 = GraphConv(hidden_dim_graph, hidden_dim1)
        self.conv1 = GraphConv(in_dim, hidden_dim1)
        pooling_gate_nn = nn.Linear(hidden_dim1, 1)
        self.pooling = SumPooling()
        self.classify = nn.Sequential(nn.Linear(hidden_dim1,hidden_dim1),nn.Dropout(dropout))
        self.classify2 = nn.Sequential(nn.Linear(hidden_dim1, n_classes),nn.Dropout(dropout))
        self.out_act = nn.Sigmoid()
    def forward(self, g):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
        h = g.ndata['h_n'].float()
        # Perform graph convolution and activation function.
        h = F.relu(self.conv1(g, h))
        g.ndata['h'] = h
        # Calculate graph representation by averaging all the node representations.
        hg = self.pooling(g,h)
        g2 = hg
        a2=self.classify(hg)
        a3=self.classify2(a2)
        return self.out_act(a3),g2,hg
class Classifier_oneGCN_avgP(nn.Module):
    def __init__(self, in_dim, hidden_dim_graph,hidden_dim1,n_classes,dropout,num_heads):
        super(Classifier_oneGCN_avgP, self).__init__()
        # self.conv1 = GraphConv(in_dim, hidden_dim_graph)
        # self.conv2 = GraphConv(hidden_dim_graph, hidden_dim1)
        self.conv1 = GraphConv(in_dim, hidden_dim1)
        pooling_gate_nn = nn.Linear(hidden_dim1, 1)
        self.pooling = AvgPooling()
        self.classify = nn.Sequential(nn.Linear(hidden_dim1,hidden_dim1),nn.Dropout(dropout))
        self.classify2 = nn.Sequential(nn.Linear(hidden_dim1, n_classes),nn.Dropout(dropout))
        self.out_act = nn.Sigmoid()
    def forward(self, g):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
        h = g.ndata['h_n'].float()
        # Perform graph convolution and activation function.
        h = F.relu(self.conv1(g, h))
        g.ndata['h'] = h
        # Calculate graph representation by averaging all the node representations.
        hg = self.pooling(g,h)
        g2 = hg
        a2=self.classify(hg)
        a3=self.classify2(a2)
        return self.out_act(a3),g2,hg

class Classifier_GCN_GAP(nn.Module):
    def __init__(self, in_dim, hidden_dim_graph,hidden_dim1,n_classes,dropout,num_heads):
        super(Classifier_GCN_GAP, self).__init__()
        # self.conv1 = GraphConv(in_dim, hidden_dim_graph)
        # self.conv2 = GraphConv(hidden_dim_graph, hidden_dim1)
        self.conv1 = GraphConv(in_dim, hidden_dim_graph)
        self.conv2 = GraphConv(hidden_dim_graph, hidden_dim1)  
        pooling_gate_nn = nn.Linear(hidden_dim1, 1)
        self.pooling = GlobalAttentionPoolingPMG(pooling_gate_nn)
        self.classify = nn.Sequential(nn.Linear(hidden_dim1,hidden_dim1),nn.Dropout(dropout))
        self.classify2 = nn.Sequential(nn.Linear(hidden_dim1, n_classes),nn.Dropout(dropout))
        self.out_act = nn.Sigmoid()
    def forward(self, g):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
        h = g.ndata['h_n'].float()
        # Perform graph convolution and activation function.
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        g.ndata['h'] = h
        # Calculate graph representation by averaging all the node representations.
        [hg,g2] = self.pooling(g,h)       
        a2=self.classify(hg)
        a3=self.classify2(a2)
        return self.out_act(a3),g2,hg
class Classifier_oneGCN_GAP(nn.Module):
    def __init__(self, in_dim, hidden_dim_graph,hidden_dim1,n_classes,dropout,num_heads):
        super(Classifier_oneGCN_GAP, self).__init__()
        # self.conv1 = GraphConv(in_dim, hidden_dim_graph)
        # self.conv2 = GraphConv(hidden_dim_graph, hidden_dim1)
        self.conv1 = GraphConv(in_dim, hidden_dim1)
        pooling_gate_nn = nn.Linear(hidden_dim1, 1)
        self.pooling = GlobalAttentionPoolingPMG(pooling_gate_nn)
        self.classify = nn.Sequential(nn.Linear(hidden_dim1,hidden_dim1),nn.Dropout(dropout))
        self.classify2 = nn.Sequential(nn.Linear(hidden_dim1, n_classes),nn.Dropout(dropout))
        self.out_act = nn.Sigmoid()
    def forward(self, g):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
        h = g.ndata['h_n'].float()
        # Perform graph convolution and activation function.
        h = F.relu(self.conv1(g, h))
        g.ndata['h'] = h
        # Calculate graph representation by averaging all the node representations.
        [hg,g2] = self.pooling(g,h)
        a2=self.classify(hg)
        a3=self.classify2(a2)
        return self.out_act(a3),g2,hg

class Classifier_GAT_GAP(nn.Module):

    def __init__(self, in_dim, hidden_dim_graph,hidden_dim1,n_classes,dropout,num_heads):
        super(Classifier_GAT_GAP, self).__init__()
        # self.conv1 = GraphConv(in_dim, hidden_dim_graph)
        # self.conv2 = GraphConv(hidden_dim_graph, hidden_dim1)
        self.conv1 = GATConv(in_dim, hidden_dim_graph,num_heads,dropout)
        self.conv2 = GATConv(hidden_dim_graph, hidden_dim1,num_heads,dropout)  
        pooling_gate_nn = nn.Linear(hidden_dim1, 1)
        self.pooling = GlobalAttentionPoolingPMG(pooling_gate_nn)
        self.classify = nn.Sequential(nn.Linear(hidden_dim1,hidden_dim1),nn.Dropout(dropout))
        self.classify2 = nn.Sequential(nn.Linear(hidden_dim1, n_classes),nn.Dropout(dropout))
        self.out_act = nn.Sigmoid()
    def forward(self, g):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
        h = g.ndata['h_n'].float()
        # Perform graph convolution and activation function.
        h = F.relu(self.conv1(g, h))
        h = h.mean(dim=1)
        h = F.relu(self.conv2(g, h))
        h = h.mean(dim=1)
        g.ndata['h'] = h
        # Calculate graph representation by averaging all the node representations.
        [hg,g2] = self.pooling(g,h)       
        a2=self.classify(hg)
        a3=self.classify2(a2)
        return self.out_act(a3),g2,hg


# MPNN:
# Neural Message Passing for Quantum Chemistry https://arxiv.org/abs/1704.01212

# pylint: disable=W0221
class MPNNGNN(nn.Module):
    """MPNN.
    MPNN is introduced in `Neural Message Passing for Quantum Chemistry
    <https://arxiv.org/abs/1704.01212>`__.
    This class performs message passing in MPNN and returns the updated node representations.
    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    node_out_feats : int
        Size for the output node representations. Default to 64.
    edge_in_feats : int
        Size for the input edge features. Default to 128.
    edge_hidden_feats : int
        Size for the hidden edge representations.
    num_step_message_passing : int
        Number of message passing steps. Default to 6.
    """
    def __init__(self, node_in_feats, edge_in_feats, node_out_feats=64,
                 edge_hidden_feats=128, num_step_message_passing=6):
        super(MPNNGNN, self).__init__()

        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, node_out_feats),
            nn.ReLU()
        )
        self.num_step_message_passing = num_step_message_passing
        edge_network = nn.Sequential(
            nn.Linear(edge_in_feats, edge_hidden_feats),
            nn.ReLU(),
            nn.Linear(edge_hidden_feats, node_out_feats * node_out_feats)
        )
        self.gnn_layer = NNConv(
            in_feats=node_out_feats,
            out_feats=node_out_feats,
            edge_func=edge_network,
            aggregator_type='sum'
        )
        self.gru = nn.GRU(node_out_feats, node_out_feats)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.project_node_feats[0].reset_parameters()
        self.gnn_layer.reset_parameters()
        for layer in self.gnn_layer.edge_nn:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        self.gru.reset_parameters()

    def forward(self, g, node_feats, edge_feats):
        """Performs message passing and updates node representations.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features. V for the number of nodes in the batch of graphs.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features. E for the number of edges in the batch of graphs.
        Returns
        -------
        node_feats : float32 tensor of shape (V, node_out_feats)
            Output node representations.
        """
        node_feats = self.project_node_feats(node_feats) # (V, node_out_feats)
        hidden_feats = node_feats.unsqueeze(0)           # (1, V, node_out_feats)

        for _ in range(self.num_step_message_passing):
            node_feats = F.relu(self.gnn_layer(g, node_feats, edge_feats))
            node_feats, hidden_feats = self.gru(node_feats.unsqueeze(0), hidden_feats)
            node_feats = node_feats.squeeze(0)

        return node_feats

# pylint: disable=W0221
class MPNNPredictor(nn.Module):
    """MPNN for regression and classification on graphs.
    MPNN is introduced in `Neural Message Passing for Quantum Chemistry
    <https://arxiv.org/abs/1704.01212>`__.
    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    edge_in_feats : int
        Size for the input edge features.
    node_out_feats : int
        Size for the output node representations. Default to 64.
    edge_hidden_feats : int
        Size for the hidden edge representations. Default to 128.
    n_tasks : int
        Number of tasks, which is also the output size. Default to 1.
    num_step_message_passing : int
        Number of message passing steps. Default to 6.
    num_step_set2set : int
        Number of set2set steps. Default to 6.
    num_layer_set2set : int
        Number of set2set layers. Default to 3.
    """
    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 node_out_feats=64,
                 edge_hidden_feats=128,
                 n_tasks=1,
                 num_step_message_passing=6,
                 num_step_set2set=6,
                 num_layer_set2set=3):
        super(MPNNPredictor, self).__init__()

        self.gnn = MPNNGNN(node_in_feats=node_in_feats,
                           node_out_feats=node_out_feats,
                           edge_in_feats=edge_in_feats,
                           edge_hidden_feats=edge_hidden_feats,
                           num_step_message_passing=num_step_message_passing)
        self.readout = Set2Set(input_dim=node_out_feats,
                               n_iters=num_step_set2set,
                               n_layers=num_layer_set2set)
        self.predict = nn.Sequential(
            nn.Linear(2 * node_out_feats, node_out_feats),
            nn.ReLU(),
            nn.Linear(node_out_feats, n_tasks)
        )

    def forward(self, g, node_feats, edge_feats):
        """Graph-level regression/soft classification.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features.
        Returns
        -------
        float32 tensor of shape (G, n_tasks)
            Prediction for the graphs in the batch. G for the number of graphs.
        """
        node_feats = self.gnn(g, node_feats, edge_feats)
        graph_feats = self.readout(g, node_feats)
        return self.predict(graph_feats)


# Weave 
# Molecular Graph Convolutions: Moving Beyond Fingerprints https://arxiv.org/abs/1603.00856

# pylint: disable=W0221, E1101, E1102
class WeaveGather(nn.Module):
    r"""Readout in Weave
    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    gaussian_expand : bool
        Whether to expand each dimension of node features by gaussian histogram.
        Default to True.
    gaussian_memberships : list of 2-tuples
        For each tuple, the first and second element separately specifies the mean
        and std for constructing a normal distribution. This argument comes into
        effect only when ``gaussian_expand==True``. By default, we set this to be
        ``[(-1.645, 0.283), (-1.080, 0.170), (-0.739, 0.134), (-0.468, 0.118),
        (-0.228, 0.114), (0., 0.114), (0.228, 0.114), (0.468, 0.118),
        (0.739, 0.134), (1.080, 0.170), (1.645, 0.283)]``.
    activation : callable
        Activation function to apply. Default to tanh.
    """
    def __init__(self,
                 node_in_feats,
                 gaussian_expand=True,
                 gaussian_memberships=None,
                 activation=nn.Tanh()):
        super(WeaveGather, self).__init__()

        self.gaussian_expand = gaussian_expand
        if gaussian_expand:
            if gaussian_memberships is None:
                gaussian_memberships = [
                    (-1.645, 0.283), (-1.080, 0.170), (-0.739, 0.134), (-0.468, 0.118),
                    (-0.228, 0.114), (0., 0.114), (0.228, 0.114), (0.468, 0.118),
                    (0.739, 0.134), (1.080, 0.170), (1.645, 0.283)]
            means, stds = map(list, zip(*gaussian_memberships))
            self.means = nn.ParameterList([
                nn.Parameter(torch.tensor(value), requires_grad=False)
                for value in means
            ])
            self.stds = nn.ParameterList([
                nn.Parameter(torch.tensor(value), requires_grad=False)
                for value in stds
            ])
            self.to_out = nn.Linear(node_in_feats * len(self.means), node_in_feats)
            self.activation = activation

    def gaussian_histogram(self, node_feats):
        r"""Constructs a gaussian histogram to capture the distribution of features
        Parameters
        ----------
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features. V for the number of nodes in the batch of graphs.
        Returns
        -------
        float32 tensor of shape (V, node_in_feats * len(self.means))
            Updated node representations
        """
        gaussian_dists = [Normal(self.means[i], self.stds[i])
                          for i in range(len(self.means))]
        max_log_probs = [gaussian_dists[i].log_prob(self.means[i])
                         for i in range(len(self.means))]
        # Normalize the probabilities by the maximum point-wise probabilities,
        # whose results will be in range [0, 1]. Note that division of probabilities
        # is equivalent to subtraction of log probabilities and the latter one is cheaper.
        log_probs = [gaussian_dists[i].log_prob(node_feats) - max_log_probs[i]
                     for i in range(len(self.means))]
        probs = torch.stack(log_probs, dim=2).exp() # (V, node_in_feats, len(self.means))
        # Add a bias to avoid numerical issues in division
        probs = probs + 1e-7
        # Normalize the probabilities across all Gaussian distributions
        probs = probs / probs.sum(2, keepdim=True)

        return probs.reshape(node_feats.shape[0],
                             node_feats.shape[1] * len(self.means))

    def forward(self, g, node_feats):
        r"""Computes graph representations out of node representations.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features. V for the number of nodes in the batch of graphs.
        Returns
        -------
        g_feats : float32 tensor of shape (G, node_in_feats)
            Output graph representations. G for the number of graphs in the batch.
        """
        if self.gaussian_expand:
            node_feats = self.gaussian_histogram(node_feats)

        with g.local_scope():
            g.ndata['h'] = node_feats
            g_feats = dgl.sum_nodes(g, 'h')

        if self.gaussian_expand:
            g_feats = self.to_out(g_feats)
            if self.activation is not None:
                g_feats = self.activation(g_feats)

        return g_feats

# pylint: disable=W0221, E1101
class WeaveLayer(nn.Module):
    r"""Single Weave layer from `Molecular Graph Convolutions: Moving Beyond Fingerprints
    <https://arxiv.org/abs/1603.00856>`__
    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    edge_in_feats : int
        Size for the input edge features.
    node_node_hidden_feats : int
        Size for the hidden node representations in updating node representations.
        Default to 50.
    edge_node_hidden_feats : int
        Size for the hidden edge representations in updating node representations.
        Default to 50.
    node_out_feats : int
        Size for the output node representations. Default to 50.
    node_edge_hidden_feats : int
        Size for the hidden node representations in updating edge representations.
        Default to 50.
    edge_edge_hidden_feats : int
        Size for the hidden edge representations in updating edge representations.
        Default to 50.
    edge_out_feats : int
        Size for the output edge representations. Default to 50.
    activation : callable
        Activation function to apply. Default to ReLU.
    """
    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 node_node_hidden_feats=50,
                 edge_node_hidden_feats=50,
                 node_out_feats=50,
                 node_edge_hidden_feats=50,
                 edge_edge_hidden_feats=50,
                 edge_out_feats=50,
                 activation=F.relu):
        super(WeaveLayer, self).__init__()

        self.activation = activation

        # Layers for updating node representations
        self.node_to_node = nn.Linear(node_in_feats, node_node_hidden_feats)
        self.edge_to_node = nn.Linear(edge_in_feats, edge_node_hidden_feats)
        self.update_node = nn.Linear(
            node_node_hidden_feats + edge_node_hidden_feats, node_out_feats)

        # Layers for updating edge representations
        self.left_node_to_edge = nn.Linear(node_in_feats, node_edge_hidden_feats)
        self.right_node_to_edge = nn.Linear(node_in_feats, node_edge_hidden_feats)
        self.edge_to_edge = nn.Linear(edge_in_feats, edge_edge_hidden_feats)
        self.update_edge = nn.Linear(
            2 * node_edge_hidden_feats + edge_edge_hidden_feats, edge_out_feats)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.node_to_node.reset_parameters()
        self.edge_to_node.reset_parameters()
        self.update_node.reset_parameters()
        self.left_node_to_edge.reset_parameters()
        self.right_node_to_edge.reset_parameters()
        self.edge_to_edge.reset_parameters()
        self.update_edge.reset_parameters()

    def forward(self, g, node_feats, edge_feats, node_only=False):
        r"""Update node and edge representations.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features. V for the number of nodes in the batch of graphs.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features. E for the number of edges in the batch of graphs.
        node_only : bool
            Whether to update node representations only. If False, edge representations
            will be updated as well. Default to False.
        Returns
        -------
        new_node_feats : float32 tensor of shape (V, node_out_feats)
            Updated node representations.
        new_edge_feats : float32 tensor of shape (E, edge_out_feats)
            Updated edge representations.
        """
        g = g.local_var()

        # Update node features
        node_node_feats = self.activation(self.node_to_node(node_feats))
        g.edata['e2n'] = self.activation(self.edge_to_node(edge_feats))
        g.update_all(fn.copy_edge('e2n', 'm'), fn.sum('m', 'e2n'))
        edge_node_feats = g.ndata.pop('e2n')
        new_node_feats = self.activation(self.update_node(
            torch.cat([node_node_feats, edge_node_feats], dim=1)))

        if node_only:
            return new_node_feats

        # Update edge features
        g.ndata['left_hv'] = self.left_node_to_edge(node_feats)
        g.ndata['right_hv'] = self.right_node_to_edge(node_feats)
        g.apply_edges(fn.u_add_v('left_hv', 'right_hv', 'first'))
        g.apply_edges(fn.u_add_v('right_hv', 'left_hv', 'second'))
        first_edge_feats = self.activation(g.edata.pop('first'))
        second_edge_feats = self.activation(g.edata.pop('second'))
        third_edge_feats = self.activation(self.edge_to_edge(edge_feats))
        new_edge_feats = self.activation(self.update_edge(
            torch.cat([first_edge_feats, second_edge_feats, third_edge_feats], dim=1)))

        return new_node_feats, new_edge_feats

class WeaveGNN(nn.Module):
    r"""The component of Weave for updating node and edge representations.
    Weave is introduced in `Molecular Graph Convolutions: Moving Beyond Fingerprints
    <https://arxiv.org/abs/1603.00856>`__.
    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    edge_in_feats : int
        Size for the input edge features.
    num_layers : int
        Number of Weave layers to use, which is equivalent to the times of message passing.
        Default to 2.
    hidden_feats : int
        Size for the hidden node and edge representations. Default to 50.
    activation : callable
        Activation function to be used. It cannot be None. Default to ReLU.
    """
    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 num_layers=2,
                 hidden_feats=50,
                 activation=F.relu):
        super(WeaveGNN, self).__init__()

        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.gnn_layers.append(WeaveLayer(node_in_feats=node_in_feats,
                                                  edge_in_feats=edge_in_feats,
                                                  node_node_hidden_feats=hidden_feats,
                                                  edge_node_hidden_feats=hidden_feats,
                                                  node_out_feats=hidden_feats,
                                                  node_edge_hidden_feats=hidden_feats,
                                                  edge_edge_hidden_feats=hidden_feats,
                                                  edge_out_feats=hidden_feats,
                                                  activation=activation))
            else:
                self.gnn_layers.append(WeaveLayer(node_in_feats=hidden_feats,
                                                  edge_in_feats=hidden_feats,
                                                  node_node_hidden_feats=hidden_feats,
                                                  edge_node_hidden_feats=hidden_feats,
                                                  node_out_feats=hidden_feats,
                                                  node_edge_hidden_feats=hidden_feats,
                                                  edge_edge_hidden_feats=hidden_feats,
                                                  edge_out_feats=hidden_feats,
                                                  activation=activation))

    def reset_parameters(self):
        """Reinitialize model parameters."""
        for layer in self.gnn_layers:
            layer.reset_parameters()

    def forward(self, g, node_feats, edge_feats, node_only=True):
        """Updates node representations (and edge representations).
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features. V for the number of nodes in the batch of graphs.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features. E for the number of edges in the batch of graphs.
        node_only : bool
            Whether to return updated node representations only or to return both
            node and edge representations. Default to True.
        Returns
        -------
        float32 tensor of shape (V, gnn_hidden_feats)
            Updated node representations.
        float32 tensor of shape (E, gnn_hidden_feats), optional
            This is returned only when ``node_only==False``. Updated edge representations.
        """
        for i in range(len(self.gnn_layers) - 1):
            node_feats, edge_feats = self.gnn_layers[i](g, node_feats, edge_feats)
        return self.gnn_layers[-1](g, node_feats, edge_feats, node_only)

class WeavePredictor(nn.Module):
    r"""Weave for regression and classification on graphs.
    Weave is introduced in `Molecular Graph Convolutions: Moving Beyond Fingerprints
    <https://arxiv.org/abs/1603.00856>`__
    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    edge_in_feats : int
        Size for the input edge features.
    num_gnn_layers : int
        Number of GNN (Weave) layers to use. Default to 2.
    gnn_hidden_feats : int
        Size for the hidden node and edge representations.
        Default to 50.
    gnn_activation : callable
        Activation function to be used in GNN (Weave) layers.
        Default to ReLU.
    graph_feats : int
        Size for the hidden graph representations. Default to 50.
    gaussian_expand : bool
        Whether to expand each dimension of node features by
        gaussian histogram in computing graph representations.
        Default to True.
    gaussian_memberships : list of 2-tuples
        For each tuple, the first and second element separately
        specifies the mean and std for constructing a normal
        distribution. This argument comes into effect only when
        ``gaussian_expand==True``. By default, we set this to be
        ``[(-1.645, 0.283), (-1.080, 0.170), (-0.739, 0.134),
        (-0.468, 0.118), (-0.228, 0.114), (0., 0.114),
        (0.228, 0.114), (0.468, 0.118), (0.739, 0.134),
        (1.080, 0.170), (1.645, 0.283)]``.
    readout_activation : callable
        Activation function to be used in computing graph
        representations out of node representations. Default to Tanh.
    n_tasks : int
        Number of tasks, which is also the output size. Default to 1.
    """
    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 num_gnn_layers=2,
                 gnn_hidden_feats=50,
                 gnn_activation=F.relu,
                 graph_feats=128,
                 gaussian_expand=True,
                 gaussian_memberships=None,
                 readout_activation=nn.Tanh(),
                 n_tasks=1):
        super(WeavePredictor, self).__init__()

        self.gnn = WeaveGNN(node_in_feats=node_in_feats,
                            edge_in_feats=edge_in_feats,
                            num_layers=num_gnn_layers,
                            hidden_feats=gnn_hidden_feats,
                            activation=gnn_activation)
        self.node_to_graph = nn.Sequential(
            nn.Linear(gnn_hidden_feats, graph_feats),
            readout_activation,
            nn.BatchNorm1d(graph_feats)
        )
        self.readout = WeaveGather(node_in_feats=graph_feats,
                                   gaussian_expand=gaussian_expand,
                                   gaussian_memberships=gaussian_memberships,
                                   activation=readout_activation)
        self.predict = nn.Linear(graph_feats, n_tasks)

    def forward(self, g, node_feats, edge_feats):
        """Graph-level regression/soft classification.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features. V for the number of nodes.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features. E for the number of edges.
        Returns
        -------
        float32 tensor of shape (G, n_tasks)
            Prediction for the graphs in the batch. G for the number of graphs.
        """
        node_feats = self.gnn(g, node_feats, edge_feats, node_only=True)
        node_feats = self.node_to_graph(node_feats)
        g_feats = self.readout(g, node_feats)

        return self.predict(g_feats)

#GIN based model:
# How Powerful are Graph Neural Networks? https://arxiv.org/abs/1810.00826
# Strategies for Pre-training Graph Neural Networks https://arxiv.org/abs/1905.12265

# pylint: disable=W0221, C0103
class GINLayer(nn.Module):
    r"""Single Layer GIN from `Strategies for
    Pre-training Graph Neural Networks <https://arxiv.org/abs/1905.12265>`__
    Parameters
    ----------
    num_edge_emb_list : list of int
        num_edge_emb_list[i] gives the number of items to embed for the
        i-th categorical edge feature variables. E.g. num_edge_emb_list[0] can be
        the number of bond types and num_edge_emb_list[1] can be the number of
        bond direction types.
    emb_dim : int
        The size of each embedding vector.
    batch_norm : bool
        Whether to apply batch normalization to the output of message passing.
        Default to True.
    activation : None or callable
        Activation function to apply to the output node representations.
        Default to None.
    """
    def __init__(self, num_edge_emb_list, emb_dim, batch_norm=True, activation=None):
        super(GINLayer, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim)
        )
        self.edge_embeddings = nn.ModuleList()
        for num_emb in num_edge_emb_list:
            emb_module = nn.Embedding(num_emb, emb_dim)
            self.edge_embeddings.append(emb_module)

        if batch_norm:
            self.bn = nn.BatchNorm1d(emb_dim)
        else:
            self.bn = None
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize model parameters."""
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

        for emb_module in self.edge_embeddings:
            nn.init.xavier_uniform_(emb_module.weight.data)

        if self.bn is not None:
            self.bn.reset_parameters()

    def forward(self, g, node_feats, categorical_edge_feats):
        """Update node representations.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        node_feats : FloatTensor of shape (N, emb_dim)
            * Input node features
            * N is the total number of nodes in the batch of graphs
            * emb_dim is the input node feature size, which must match emb_dim in initialization
        categorical_edge_feats : list of LongTensor of shape (E)
            * Input categorical edge features
            * len(categorical_edge_feats) should be the same as len(self.edge_embeddings)
            * E is the total number of edges in the batch of graphs
        Returns
        -------
        node_feats : float32 tensor of shape (N, emb_dim)
            Output node representations
        """
        edge_embeds = []
        for i, feats in enumerate(categorical_edge_feats):
            edge_embeds.append(self.edge_embeddings[i](feats))
        edge_embeds = torch.stack(edge_embeds, dim=0).sum(0)
        g = g.local_var()
        g.ndata['feat'] = node_feats
        g.edata['feat'] = edge_embeds
        g.update_all(fn.u_add_e('feat', 'feat', 'm'), fn.sum('m', 'feat'))

        node_feats = self.mlp(g.ndata.pop('feat'))
        if self.bn is not None:
            node_feats = self.bn(node_feats)
        if self.activation is not None:
            node_feats = self.activation(node_feats)

        return node_feats

class GIN(nn.Module):
    r"""Graph Isomorphism Network from `Strategies for
    Pre-training Graph Neural Networks <https://arxiv.org/abs/1905.12265>`__
    This module is for updating node representations only.
    Parameters
    ----------
    num_node_emb_list : list of int
        num_node_emb_list[i] gives the number of items to embed for the
        i-th categorical node feature variables. E.g. num_node_emb_list[0] can be
        the number of atom types and num_node_emb_list[1] can be the number of
        atom chirality types.
    num_edge_emb_list : list of int
        num_edge_emb_list[i] gives the number of items to embed for the
        i-th categorical edge feature variables. E.g. num_edge_emb_list[0] can be
        the number of bond types and num_edge_emb_list[1] can be the number of
        bond direction types.
    num_layers : int
        Number of GIN layers to use. Default to 5.
    emb_dim : int
        The size of each embedding vector. Default to 300.
    JK : str
        JK for jumping knowledge as in `Representation Learning on Graphs with
        Jumping Knowledge Networks <https://arxiv.org/abs/1806.03536>`__. It decides
        how we are going to combine the all-layer node representations for the final output.
        There can be four options for this argument, ``concat``, ``last``, ``max`` and ``sum``.
        Default to 'last'.
        * ``'concat'``: concatenate the output node representations from all GIN layers
        * ``'last'``: use the node representations from the last GIN layer
        * ``'max'``: apply max pooling to the node representations across all GIN layers
        * ``'sum'``: sum the output node representations from all GIN layers
    dropout : float
        Dropout to apply to the output of each GIN layer. Default to 0.5
    """
    def __init__(self, num_node_emb_list, num_edge_emb_list,
                 num_layers=5, emb_dim=300, JK='last', dropout=0.5):
        super(GIN, self).__init__()

        self.num_layers = num_layers
        self.JK = JK
        self.dropout = nn.Dropout(dropout)

        if num_layers < 2:
            raise ValueError('Number of GNN layers must be greater '
                             'than 1, got {:d}'.format(num_layers))

        self.node_embeddings = nn.ModuleList()
        for num_emb in num_node_emb_list:
            emb_module = nn.Embedding(num_emb, emb_dim)
            self.node_embeddings.append(emb_module)

        self.gnn_layers = nn.ModuleList()
        for layer in range(num_layers):
            if layer == num_layers - 1:
                self.gnn_layers.append(GINLayer(num_edge_emb_list, emb_dim))
            else:
                self.gnn_layers.append(GINLayer(num_edge_emb_list, emb_dim, activation=F.relu))

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize model parameters."""
        for emb_module in self.node_embeddings:
            nn.init.xavier_uniform_(emb_module.weight.data)

        for layer in self.gnn_layers:
            layer.reset_parameters()

    def forward(self, g, categorical_node_feats, categorical_edge_feats):
        """Update node representations
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        categorical_node_feats : list of LongTensor of shape (N)
            * Input categorical node features
            * len(categorical_node_feats) should be the same as len(self.node_embeddings)
            * N is the total number of nodes in the batch of graphs
        categorical_edge_feats : list of LongTensor of shape (E)
            * Input categorical edge features
            * len(categorical_edge_feats) should be the same as
              len(num_edge_emb_list) in the arguments
            * E is the total number of edges in the batch of graphs
        Returns
        -------
        final_node_feats : float32 tensor of shape (N, M)
            Output node representations, N for the number of nodes and
            M for output size. In particular, M will be emb_dim * (num_layers + 1)
            if self.JK == 'concat' and emb_dim otherwise.
        """
        node_embeds = []
        for i, feats in enumerate(categorical_node_feats):
            node_embeds.append(self.node_embeddings[i](feats))
        node_embeds = torch.stack(node_embeds, dim=0).sum(0)

        all_layer_node_feats = [node_embeds]
        for layer in range(self.num_layers):
            node_feats = self.gnn_layers[layer](g, all_layer_node_feats[layer],
                                                categorical_edge_feats)
            node_feats = self.dropout(node_feats)
            all_layer_node_feats.append(node_feats)

        if self.JK == 'concat':
            final_node_feats = torch.cat(all_layer_node_feats, dim=1)
        elif self.JK == 'last':
            final_node_feats = all_layer_node_feats[-1]
        elif self.JK == 'max':
            all_layer_node_feats = [h.unsqueeze(0) for h in all_layer_node_feats]
            final_node_feats = torch.max(torch.cat(all_layer_node_feats, dim=0), dim=0)[0]
        elif self.JK == 'sum':
            all_layer_node_feats = [h.unsqueeze(0) for h in all_layer_node_feats]
            final_node_feats = torch.sum(torch.cat(all_layer_node_feats, dim=0), dim=0)
        else:
            return ValueError("Expect self.JK to be 'concat', 'last', "
                              "'max' or 'sum', got {}".format(self.JK))

        return final_node_feats


# pylint: disable=W0221
class GINPredictor(nn.Module):
    """GIN-based model for regression and classification on graphs.
    GIN was first introduced in `How Powerful Are Graph Neural Networks
    <https://arxiv.org/abs/1810.00826>`__ for general graph property
    prediction problems. It was further extended in `Strategies for
    Pre-training Graph Neural Networks <https://arxiv.org/abs/1905.12265>`__
    for pre-training and semi-supervised learning on large-scale datasets.
    For classification tasks, the output will be logits, i.e. values before
    sigmoid or softmax.
    Parameters
    ----------
    num_node_emb_list : list of int
        num_node_emb_list[i] gives the number of items to embed for the
        i-th categorical node feature variables. E.g. num_node_emb_list[0] can be
        the number of atom types and num_node_emb_list[1] can be the number of
        atom chirality types.
    num_edge_emb_list : list of int
        num_edge_emb_list[i] gives the number of items to embed for the
        i-th categorical edge feature variables. E.g. num_edge_emb_list[0] can be
        the number of bond types and num_edge_emb_list[1] can be the number of
        bond direction types.
    num_layers : int
        Number of GIN layers to use. Default to 5.
    emb_dim : int
        The size of each embedding vector. Default to 300.
    JK : str
        JK for jumping knowledge as in `Representation Learning on Graphs with
        Jumping Knowledge Networks <https://arxiv.org/abs/1806.03536>`__. It decides
        how we are going to combine the all-layer node representations for the final output.
        There can be four options for this argument, ``'concat'``, ``'last'``, ``'max'`` and
        ``'sum'``. Default to 'last'.
        * ``'concat'``: concatenate the output node representations from all GIN layers
        * ``'last'``: use the node representations from the last GIN layer
        * ``'max'``: apply max pooling to the node representations across all GIN layers
        * ``'sum'``: sum the output node representations from all GIN layers
    dropout : float
        Dropout to apply to the output of each GIN layer. Default to 0.5.
    readout : str
        Readout for computing graph representations out of node representations, which
        can be ``'sum'``, ``'mean'``, ``'max'``, or ``'attention'``. Default to 'mean'.
    n_tasks : int
        Number of tasks, which is also the output size. Default to 1.
    """
    def __init__(self, num_node_emb_list, num_edge_emb_list, num_layers=5,
                 emb_dim=300, JK='last', dropout=0.5, readout='mean', n_tasks=1):
        super(GINPredictor, self).__init__()

        if num_layers < 2:
            raise ValueError('Number of GNN layers must be greater '
                             'than 1, got {:d}'.format(num_layers))

        self.gnn = GIN(num_node_emb_list=num_node_emb_list,
                       num_edge_emb_list=num_edge_emb_list,
                       num_layers=num_layers,
                       emb_dim=emb_dim,
                       JK=JK,
                       dropout=dropout)

        if readout == 'sum':
            self.readout = SumPooling()
        elif readout == 'mean':
            self.readout = AvgPooling()
        elif readout == 'max':
            self.readout = MaxPooling()
        elif readout == 'attention':
            if JK == 'concat':
                self.readout = GlobalAttentionPooling(
                    gate_nn=nn.Linear((num_layers + 1) * emb_dim, 1))
            else:
                self.readout = GlobalAttentionPooling(
                    gate_nn=nn.Linear(emb_dim, 1))
        else:
            raise ValueError("Expect readout to be 'sum', 'mean', "
                             "'max' or 'attention', got {}".format(readout))

        if JK == 'concat':
            self.predict = nn.Linear((num_layers + 1) * emb_dim, n_tasks)
        else:
            self.predict = nn.Linear(emb_dim, n_tasks)

    def forward(self, g, categorical_node_feats, categorical_edge_feats):
        """Graph-level regression/soft classification.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        categorical_node_feats : list of LongTensor of shape (N)
            * Input categorical node features
            * len(categorical_node_feats) should be the same as len(num_node_emb_list)
            * N is the total number of nodes in the batch of graphs
        categorical_edge_feats : list of LongTensor of shape (E)
            * Input categorical edge features
            * len(categorical_edge_feats) should be the same as
              len(num_edge_emb_list) in the arguments
            * E is the total number of edges in the batch of graphs
        Returns
        -------
        FloatTensor of shape (B, n_tasks)
            * Predictions on graphs
            * B for the number of graphs in the batch
        """
        node_feats = self.gnn(g, categorical_node_feats, categorical_edge_feats)
        graph_feats = self.readout(g, node_feats)
        return self.predict(graph_feats)
# AttentiveFP:
# “Pushing the Boundaries of Molecular Representation for Drug Discovery with the
# Graph Attention Mechanism” https://pubmed.ncbi.nlm.nih.gov/31408336/

# pylint: disable=W0221
class GlobalPool(nn.Module):
    """One-step readout in AttentiveFP
    Parameters
    ----------
    feat_size : int
        Size for the input node features, graph features and output graph
        representations.
    dropout : float
        The probability for performing dropout.
    """
    def __init__(self, feat_size, dropout):
        super(GlobalPool, self).__init__()

        self.compute_logits = nn.Sequential(
            nn.Linear(2 * feat_size, 1),
            nn.LeakyReLU()
        )
        self.project_nodes = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_size, feat_size)
        )
        self.gru = nn.GRUCell(feat_size, feat_size)

    def forward(self, g, node_feats, g_feats, get_node_weight=False):
        """Perform one-step readout
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        g_feats : float32 tensor of shape (G, graph_feat_size)
            Input graph features. G for the number of graphs.
        get_node_weight : bool
            Whether to get the weights of atoms during readout.
        Returns
        -------
        float32 tensor of shape (G, graph_feat_size)
            Updated graph features.
        float32 tensor of shape (V, 1)
            The weights of nodes in readout.
        """
        with g.local_scope():
            g.ndata['z'] = self.compute_logits(
                torch.cat([dgl.broadcast_nodes(g, F.relu(g_feats)), node_feats], dim=1))
            g.ndata['a'] = dgl.softmax_nodes(g, 'z')
            g.ndata['hv'] = self.project_nodes(node_feats)

            g_repr = dgl.sum_nodes(g, 'hv', 'a')
            context = F.elu(g_repr)

            if get_node_weight:
                return self.gru(context, g_feats), g.ndata['a']
            else:
                return self.gru(context, g_feats)

class AttentiveFPReadout(nn.Module):
    """Readout in AttentiveFP
    AttentiveFP is introduced in `Pushing the Boundaries of Molecular Representation for
    Drug Discovery with the Graph Attention Mechanism
    <https://www.ncbi.nlm.nih.gov/pubmed/31408336>`__
    This class computes graph representations out of node features.
    Parameters
    ----------
    feat_size : int
        Size for the input node features, graph features and output graph
        representations.
    num_timesteps : int
        Times of updating the graph representations with GRU. Default to 2.
    dropout : float
        The probability for performing dropout. Default to 0.
    """
    def __init__(self, feat_size, num_timesteps=2, dropout=0.):
        super(AttentiveFPReadout, self).__init__()

        self.readouts = nn.ModuleList()
        for _ in range(num_timesteps):
            self.readouts.append(GlobalPool(feat_size, dropout))

    def forward(self, g, node_feats, get_node_weight=False):
        """Computes graph representations out of node features.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        get_node_weight : bool
            Whether to get the weights of nodes in readout. Default to False.
        Returns
        -------
        g_feats : float32 tensor of shape (G, graph_feat_size)
            Graph representations computed. G for the number of graphs.
        node_weights : list of float32 tensor of shape (V, 1), optional
            This is returned when ``get_node_weight`` is ``True``.
            The list has a length ``num_timesteps`` and ``node_weights[i]``
            gives the node weights in the i-th update.
        """
        with g.local_scope():
            g.ndata['hv'] = node_feats
            g_feats = dgl.sum_nodes(g, 'hv')

        if get_node_weight:
            node_weights = []

        for readout in self.readouts:
            if get_node_weight:
                g_feats, node_weights_t = readout(g, node_feats, g_feats, get_node_weight)
                node_weights.append(node_weights_t)
            else:
                g_feats = readout(g, node_feats, g_feats)

        if get_node_weight:
            return g_feats, node_weights
        else:
            return g_feats
# pylint: disable=W0221, C0103, E1101
class AttentiveGRU1(nn.Module):
    """Update node features with attention and GRU.
    This will be used for incorporating the information of edge features
    into node features for message passing.
    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_feat_size : int
        Size for the input edge (bond) features.
    edge_hidden_size : int
        Size for the intermediate edge (bond) representations.
    dropout : float
        The probability for performing dropout.
    """
    def __init__(self, node_feat_size, edge_feat_size, edge_hidden_size, dropout):
        super(AttentiveGRU1, self).__init__()

        self.edge_transform = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(edge_feat_size, edge_hidden_size)
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.edge_transform[1].reset_parameters()
        self.gru.reset_parameters()

    def forward(self, g, edge_logits, edge_feats, node_feats):
        """Update node representations.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        edge_logits : float32 tensor of shape (E, 1)
            The edge logits based on which softmax will be performed for weighting
            edges within 1-hop neighborhoods. E represents the number of edges.
        edge_feats : float32 tensor of shape (E, edge_feat_size)
            Previous edge features.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Previous node features. V represents the number of nodes.
        Returns
        -------
        float32 tensor of shape (V, node_feat_size)
            Updated node features.
        """
        g = g.local_var()
        g.edata['e'] = edge_softmax(g, edge_logits) * self.edge_transform(edge_feats)
        g.update_all(fn.copy_edge('e', 'm'), fn.sum('m', 'c'))
        context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))

class AttentiveGRU2(nn.Module):
    """Update node features with attention and GRU.
    This will be used in GNN layers for updating node representations.
    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_hidden_size : int
        Size for the intermediate edge (bond) representations.
    dropout : float
        The probability for performing dropout.
    """
    def __init__(self, node_feat_size, edge_hidden_size, dropout):
        super(AttentiveGRU2, self).__init__()

        self.project_node = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(node_feat_size, edge_hidden_size)
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.project_node[1].reset_parameters()
        self.gru.reset_parameters()

    def forward(self, g, edge_logits, node_feats):
        """Update node representations.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        edge_logits : float32 tensor of shape (E, 1)
            The edge logits based on which softmax will be performed for weighting
            edges within 1-hop neighborhoods. E represents the number of edges.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Previous node features. V represents the number of nodes.
        Returns
        -------
        float32 tensor of shape (V, node_feat_size)
            Updated node features.
        """
        g = g.local_var()
        g.edata['a'] = edge_softmax(g, edge_logits)
        g.ndata['hv'] = self.project_node(node_feats)

        g.update_all(fn.src_mul_edge('hv', 'a', 'm'), fn.sum('m', 'c'))
        context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))

class GetContext(nn.Module):
    """Generate context for each node by message passing at the beginning.
    This layer incorporates the information of edge features into node
    representations so that message passing needs to be only performed over
    node representations.
    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_feat_size : int
        Size for the input edge (bond) features.
    graph_feat_size : int
        Size of the learned graph representation (molecular fingerprint).
    dropout : float
        The probability for performing dropout.
    """
    def __init__(self, node_feat_size, edge_feat_size, graph_feat_size, dropout):
        super(GetContext, self).__init__()

        self.project_node = nn.Sequential(
            nn.Linear(node_feat_size, graph_feat_size),
            nn.LeakyReLU()
        )
        self.project_edge1 = nn.Sequential(
            nn.Linear(node_feat_size + edge_feat_size, graph_feat_size),
            nn.LeakyReLU()
        )
        self.project_edge2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * graph_feat_size, 1),
            nn.LeakyReLU()
        )
        self.attentive_gru = AttentiveGRU1(graph_feat_size, graph_feat_size,
                                           graph_feat_size, dropout)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.project_node[0].reset_parameters()
        self.project_edge1[0].reset_parameters()
        self.project_edge2[1].reset_parameters()
        self.attentive_gru.reset_parameters()

    def apply_edges1(self, edges):
        """Edge feature update.
        Parameters
        ----------
        edges : EdgeBatch
            Container for a batch of edges
        Returns
        -------
        dict
            Mapping ``'he1'`` to updated edge features.
        """
        return {'he1': torch.cat([edges.src['hv'], edges.data['he']], dim=1)}

    def apply_edges2(self, edges):
        """Edge feature update.
        Parameters
        ----------
        edges : EdgeBatch
            Container for a batch of edges
        Returns
        -------
        dict
            Mapping ``'he2'`` to updated edge features.
        """
        return {'he2': torch.cat([edges.dst['hv_new'], edges.data['he1']], dim=1)}

    def forward(self, g, node_feats, edge_feats):
        """Incorporate edge features and update node representations.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        edge_feats : float32 tensor of shape (E, edge_feat_size)
            Input edge features. E for the number of edges.
        Returns
        -------
        float32 tensor of shape (V, graph_feat_size)
            Updated node features.
        """
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.ndata['hv_new'] = self.project_node(node_feats)
        g.edata['he'] = edge_feats

        g.apply_edges(self.apply_edges1)
        g.edata['he1'] = self.project_edge1(g.edata['he1'])
        g.apply_edges(self.apply_edges2)
        logits = self.project_edge2(g.edata['he2'])

        return self.attentive_gru(g, logits, g.edata['he1'], g.ndata['hv_new'])

class GNNLayer(nn.Module):
    """GNNLayer for updating node features.
    This layer performs message passing over node representations and update them.
    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    graph_feat_size : int
        Size for the graph representations to be computed.
    dropout : float
        The probability for performing dropout.
    """
    def __init__(self, node_feat_size, graph_feat_size, dropout):
        super(GNNLayer, self).__init__()

        self.project_edge = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * node_feat_size, 1),
            nn.LeakyReLU()
        )
        self.attentive_gru = AttentiveGRU2(node_feat_size, graph_feat_size, dropout)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.project_edge[1].reset_parameters()
        self.attentive_gru.reset_parameters()

    def apply_edges(self, edges):
        """Edge feature generation.
        Generate edge features by concatenating the features of the destination
        and source nodes.
        Parameters
        ----------
        edges : EdgeBatch
            Container for a batch of edges.
        Returns
        -------
        dict
            Mapping ``'he'`` to the generated edge features.
        """
        return {'he': torch.cat([edges.dst['hv'], edges.src['hv']], dim=1)}

    def forward(self, g, node_feats):
        """Perform message passing and update node representations.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        Returns
        -------
        float32 tensor of shape (V, graph_feat_size)
            Updated node features.
        """
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.apply_edges(self.apply_edges)
        logits = self.project_edge(g.edata['he'])

        return self.attentive_gru(g, logits, node_feats)

class AttentiveFPGNN(nn.Module):
    """`Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph
    Attention Mechanism <https://www.ncbi.nlm.nih.gov/pubmed/31408336>`__
    This class performs message passing in AttentiveFP and returns the updated node representations.
    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_feat_size : int
        Size for the input edge features.
    num_layers : int
        Number of GNN layers. Default to 2.
    graph_feat_size : int
        Size for the graph representations to be computed. Default to 200.
    dropout : float
        The probability for performing dropout. Default to 0.
    """
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 graph_feat_size=200,
                 dropout=0.):
        super(AttentiveFPGNN, self).__init__()

        self.init_context = GetContext(node_feat_size, edge_feat_size, graph_feat_size, dropout)
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.gnn_layers.append(GNNLayer(graph_feat_size, graph_feat_size, dropout))

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.init_context.reset_parameters()
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    def forward(self, g, node_feats, edge_feats):
        """Performs message passing and updates node representations.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        edge_feats : float32 tensor of shape (E, edge_feat_size)
            Input edge features. E for the number of edges.
        Returns
        -------
        node_feats : float32 tensor of shape (V, graph_feat_size)
            Updated node representations.
        """
        node_feats = self.init_context(g, node_feats, edge_feats)
        for gnn in self.gnn_layers:
            node_feats = gnn(g, node_feats)
        return node_feats




# pylint: disable=W0221
class AttentiveFPPredictor(nn.Module):
    """AttentiveFP for regression and classification on graphs.
    AttentiveFP is introduced in
    `Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph
    Attention Mechanism. <https://www.ncbi.nlm.nih.gov/pubmed/31408336>`__
    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_feat_size : int
        Size for the input edge features.
    num_layers : int
        Number of GNN layers. Default to 2.
    num_timesteps : int
        Times of updating the graph representations with GRU. Default to 2.
    graph_feat_size : int
        Size for the learned graph representations. Default to 200.
    n_tasks : int
        Number of tasks, which is also the output size. Default to 1.
    dropout : float
        Probability for performing the dropout. Default to 0.
    """
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 num_timesteps=2,
                 graph_feat_size=200,
                 n_tasks=1,
                 dropout=0.):
        super(AttentiveFPPredictor, self).__init__()

        self.gnn = AttentiveFPGNN(node_feat_size=node_feat_size,
                                  edge_feat_size=edge_feat_size,
                                  num_layers=num_layers,
                                  graph_feat_size=graph_feat_size,
                                  dropout=dropout)
        self.readout = AttentiveFPReadout(feat_size=graph_feat_size,
                                          num_timesteps=num_timesteps,
                                          dropout=dropout)
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(graph_feat_size, n_tasks)
        )

    def forward(self, g, node_feats, edge_feats, get_node_weight=False):
        """Graph-level regression/soft classification.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        edge_feats : float32 tensor of shape (E, edge_feat_size)
            Input edge features. E for the number of edges.
        get_node_weight : bool
            Whether to get the weights of atoms during readout. Default to False.
        Returns
        -------
        float32 tensor of shape (G, n_tasks)
            Prediction for the graphs in the batch. G for the number of graphs.
        node_weights : list of float32 tensor of shape (V, 1), optional
            This is returned when ``get_node_weight`` is ``True``.
            The list has a length ``num_timesteps`` and ``node_weights[i]``
            gives the node weights in the i-th update.
        """
        node_feats = self.gnn(g, node_feats, edge_feats)
        if get_node_weight:
            g_feats, node_weights = self.readout(g, node_feats, get_node_weight)
            return self.predict(g_feats), node_weights
        else:
            g_feats = self.readout(g, node_feats, get_node_weight)
            
            return self.predict(g_feats)
