#!/usr/bin/env python3

from math import pi, sqrt, acos, atan
import scipy.stats
from scipy.special import owens_t
from functools import reduce

from networkx import barabasi_albert_graph
import torch as th
import graph_construct as G
import simplecs as s
from torch import vmap


import cvxopt as cvx
import numpy as np
from cvxopt.solvers import coneqp

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from csbm import csbm
from torch_geometric.data import Data
import torch_geometric.transforms as T

from torch.nn import ReLU, Softmax, Sigmoid, Parameter
from torch_geometric.nn import Sequential, GCNConv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from tqdm import tqdm, trange
import tqdm.contrib.itertools as tqdm_itertools
import itertools
from joblib import Parallel, delayed, parallel_config
import time

from rich.rule import Rule
from rich.console import Console

import gc
from typing import Optional, Tuple, Callable, Dict

console = Console()
path = Path("camsap_output").resolve()
if not path.is_dir():
    path.mkdir(parents=True)


"""
We now move onto the experiments in sampling.
Here we always use the normalized laplacian.

For each graph type:
For each noise level:
We have the following steps:
1) Construct task: (Graph, Features, Labels, Regression Outputs)
    This will be stored in a pytorch geometric Data object
    Data(edge_index, x, y_label, y_cts, noise_level).
    x.shape = (num_nodes, num_tasks).
    'num_tasks' acts as feat_dim for SGC and GCN
2) Construct classification method (e.g. train GNN) -- currently constructed in the above step
3) Construct regression method (e.g. train GNN) -- currently not doing this
4) Construct Sample for each sample type
5) a) Reconstruct Features
   b) Use methods to construct predicted labels & regression
   c) Return errors
6) (optional)) Return analytic errors
7) Plot

"""


# num_tasks = num_signals
# we use normalized laplacian
def mk_tasks_bl(
    graph_type="BA",
    num_nodes=300,
    num_tasks=30,
    noise_level=0,
    signal_type="bl",
    raw_signal_dist="normal",
    # raw_signal_dist="sqrt",
):
    sig_shape = (num_nodes, num_tasks)
    # make graph
    if graph_type == "ER":
        g, n = G.connected_erdos_renyi_graph(num_nodes, 0.8)
    elif graph_type == "SBM":
        g, n = G.sbm_constructor(th.tensor(num_nodes // 10).repeat(10), 0.1, 0.7)
    elif graph_type == "BA":
        g, n = G.clean_graph(G.barabasi_albert_graph(num_nodes, 3))
    else:
        raise ValueError("graph_type not ER/SB/BA")

    U = s.calc_eigenbasis(g, n, double=False, normalization="sym")
    L = s.calc_laplacian(
        g,
        n,
        normalization="sym",
    )
    # The smallest eigenvalue of L is 0, but sometimes computational
    # errors calculate it as negative. We manually override this.
    sqrteigs = th.linalg.eigvalsh(L).sqrt().float()
    sqrteigs[0] = 0

    rsqrteigs = th.linalg.eigvalsh(L).rsqrt().float()
    rsqrteigs[0] = 0

    # make features & tasks
    bandwidth = n // 10
    U_k = s.restrict_eigenbasis(U, bandwidth)
    if signal_type == "bl":
        proj = s.calc_proj(U, bandwidth)
    # elif signal_type == "lap":
    #     proj = U @ sqrteigs.diag() @ U.T
    elif signal_type == "lap_pinv":
        proj = U @ rsqrteigs.diag() @ U.T
    else:
        raise ValueError("signal_type isn't bl or lap_pinv")

    if raw_signal_dist in ["normal", "gaussian"]:
        raw_signals = th.randn(sig_shape)
    elif raw_signal_dist == "uniform":
        raw_signals = th.empty(sig_shape).uniform_(-1, 1)
    elif raw_signal_dist == "sqrt":
        raw_signals = th.empty(sig_shape).uniform_(-1, 1)
        raw_signals = raw_signals.abs().sqrt() * raw_signals.sign()

    y_cts = proj @ raw_signals
    # y_cts = y_cts.refine_names("noise_levels", "tasks", "nodes")
    y_label = th.sign(y_cts)

    # noise is a tensor with shape (total_num_tasks, n)
    noise = noise_level * th.randn(sig_shape)
    x = y_cts + noise
    return Data(
        edge_index=g,
        x=x,
        y_label=y_label,
        y_cts=y_cts,
        noise_level=noise_level,
        U_k=U_k,
        num_nodes=n,
        class_fn=th.sign,
        signal_type=signal_type,
    )


# We assume 1 noise level
def mk_tasks_gcn(
    feat_dim=64,
    hidden_dim=32,
    **kwargs,
):
    kwargs["num_tasks"] = feat_dim
    data = mk_tasks_bl(**kwargs)
    model = Sequential(
        "x, edge_index",
        [
            (GCNConv(feat_dim, hidden_dim), "x, edge_index -> x"),
            ReLU(inplace=True),
            (GCNConv(hidden_dim, hidden_dim), "x, edge_index -> x"),
            ReLU(inplace=True),
            Linear(hidden_dim, 1, weight_initializer="glorot"),
        ],
    )
    intercept = model(data.y_cts, data.edge_index).median().item()
    model.module_4.bias = Parameter(model.module_4.bias - intercept)
    data.y_label = model(data.y_cts, data.edge_index).squeeze().sign()

    return data, model


def weightless_conv(model, x, edge_index):
    local_conv = model.module_0
    normalised_edge_index, normalised_edge_weight = gcn_norm(
        edge_index,
        None,
        x.size(local_conv.node_dim),
        local_conv.improved,
        local_conv.add_self_loops,
        local_conv.flow,
        x.dtype,
    )
    return local_conv.propagate(
        normalised_edge_index, x=x, edge_weight=normalised_edge_weight
    )


def mk_tasks_sgc(
    feat_dim=64,
    hidden_dim=32,
    signal_type="lap_pinv",
    num_conv_layers=2,
    **kwargs,
):
    kwargs["num_tasks"] = feat_dim
    data = mk_tasks_bl(signal_type=signal_type, **kwargs)
    if num_conv_layers < 1:
        raise ValueError(
            "num_layers of sgc needs to be at least 1 (i.e. at least 1 convolutional layer)"
        )
    else:

        def gconv(d1, d2):
            return (GCNConv(d1, d2), "x, edge_index -> x")

        layers = (
            [gconv(feat_dim, hidden_dim)]
            + [gconv(hidden_dim, hidden_dim) for _ in range(num_conv_layers - 1)]
            + [Linear(hidden_dim, 1, bias=False, weight_initializer="glorot")]
        )
    model = Sequential("x, edge_index", layers)
    data.y_label = model(data.y_cts, data.edge_index).sign().squeeze()

    G = weightless_conv(model, th.eye(data.num_nodes), data.edge_index)
    # G = weightless_conv(model, G, data.edge_index)
    w_prod = reduce(
        th.matmul,
        reversed(
            [
                w
                for name, w in model.named_parameters()
                if name.split(".")[-1] == "weight"
            ]
        ),
    ).squeeze()
    model.G = G.float()
    model.w_prod = w_prod
    model.num_conv_layers = num_conv_layers
    return data, model


# Can't really construct uncorrelated tasks
def mk_tasks_csbm(
    num_nodes=300,
    feat_dim=1,
    noise_level=0,
):

    data = csbm(
        num_nodes,
        feat_dim,
        num_nodes // 2,
        0.7,
        0.1,
        100000 if noise_level == 0 else 1 / noise_level,
    )

    U = s.calc_eigenbasis(
        data.edge_index, data.num_nodes, double=False, normalization="sym"
    )
    bandwidth = num_nodes // 10
    U_k = s.restrict_eigenbasis(U, bandwidth)
    # data.x_clean = data.x_clean.T.unsqueeze(0)
    data.x = data.x_clean if noise_level == 0 else data.x
    data.y_label = data.y
    data.y_cts = data.x_clean
    data.U_k = U_k
    data.num_nodes = num_nodes
    data.signal_type = "csbm"
    return data
