
import torch as th
import torch.nn as nn
import numpy as np

from dgl.backend import pytorch as F
from dgl.readout import sum_nodes, mean_nodes, max_nodes, broadcast_nodes,\
    softmax_nodes, topk_nodes

class GlobalAttentionPoolingPMG(nn.Module):
    r"""

    Description
    -----------
    Apply Global Attention Pooling (`Gated Graph Sequence Neural Networks
    <https://arxiv.org/abs/1511.05493.pdf>`__) over the nodes in a graph.

    .. math::
        r^{(i)} = \sum_{k=1}^{N_i}\mathrm{softmax}\left(f_{gate}
        \left(x^{(i)}_k\right)\right) f_{feat}\left(x^{(i)}_k\right)

    Parameters
    ----------
    gate_nn : torch.nn.Module
        A neural network that computes attention scores for each feature.
    feat_nn : torch.nn.Module, optional
        A neural network applied to each feature before combining them with attention
        scores.
    """
    def __init__(self, gate_nn, feat_nn=None):
        super(GlobalAttentionPoolingPMG, self).__init__()
        self.gate_nn = gate_nn
        self.feat_nn = feat_nn

    def forward(self, graph, feat):
        r"""

        Compute global attention pooling.

        Parameters
        ----------
        graph : DGLGraph
            A DGLGraph or a batch of DGLGraphs.
        feat : torch.Tensor
            The input feature with shape :math:`(N, D)` where :math:`N` is the
            number of nodes in the graph, and :math:`D` means the size of features.

        Returns
        -------
        torch.Tensor
            The output feature with shape :math:`(B, D)`, where :math:`B` refers
            to the batch size.
        """
        with graph.local_scope():
            gate = self.gate_nn(feat)
            assert gate.shape[-1] == 1, "The output of gate_nn should have size 1 at the last axis."
            feat = self.feat_nn(feat) if self.feat_nn else feat

            graph.ndata['gate'] = gate
            gate = softmax_nodes(graph, 'gate')
            graph.ndata.pop('gate')

            graph.ndata['r'] = feat * gate
            readout = sum_nodes(graph, 'r')
            graph.ndata.pop('r')

            return readout,gate
