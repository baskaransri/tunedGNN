#!/usr/bin/env python3

import torch as th
import numpy as np

# import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import to_edge_index

from math import sqrt


def csbm(graph_size, feat_dim, num_labels, c_in, c_out, mu):
    class1 = th.randperm(graph_size)[:num_labels]

    labels = th.zeros(graph_size)
    labels[class1] = 1

    probs = np.equal.outer(labels, labels)
    probs = th.from_numpy(probs).float()
    # We now have 0s on the c_outs and 1s on the c_ins
    probs = (probs * (c_in - c_out)) + c_out
    # We calculate only the upper triangular, and then make the graph undirected
    # We remove all self-loops
    probs = th.triu(probs)
    A = th.bernoulli(probs)
    A = A + A.T
    A.fill_diagonal_(0)
    edge_index = to_edge_index(A.to_sparse())[0]

    # Make Labels +/- 1 Rademacher rather than 0/1:
    labels = (labels * 2) - 1

    # latent vector:
    u = th.randn(feat_dim) / sqrt(feat_dim)
    if feat_dim == 1:
        # This is cheating so that sign works as a simple classifier
        # else we would need to be smarter
        u = u.abs()
    B_unnoised = sqrt(mu / graph_size) * th.outer(labels, u)
    B = B_unnoised + th.randn_like(B_unnoised) / sqrt(feat_dim)

    return Data(
        edge_index=edge_index,
        x=B,
        x_clean=B_unnoised,
        y=labels.long(),
        num_nodes=graph_size,
    )
