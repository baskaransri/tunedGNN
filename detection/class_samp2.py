#!/usr/bin/env python3

from math import pi, sqrt, acos, atan
import scipy.stats
from scipy.special import owens_t
from functools import reduce

from networkx import barabasi_albert_graph
import torch as th
import graph_construct as G
import simplecs as s
import tspcode as ts
from torch import vmap

import datasets

import cvxopt as cvx
import numpy as np
from cvxopt.solvers import coneqp

from scipy.sparse.linalg import LinearOperator
from torch_geometric.utils import spmm

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

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

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


def mk_tasks_real(name="weather", bandlimit_signals=True):
    match name:
        case "weather":
            data = datasets.weather()
            k = 8
        case "fmri":
            data = datasets.fmri_subsample()
            k = 36

    g = data.edge_index
    n = data.num_nodes
    U = s.calc_eigenbasis(g, n, double=False, normalization="sym")
    L = s.calc_laplacian(
        g,
        n,
        normalization="sym",
    )
    U_k = s.restrict_eigenbasis(U, k)

    # We center the values for each 'task'
    if name == "weather":
        data.x -= data.x.median(dim=0).values
    else:
        data.x -= data.x.mean()
    # Note that removing a constant doesn't keep signals
    # bandlimited under L_sym!
    if bandlimit_signals:
        data.x = U_k @ (U_k.T @ data.x)
    data.y_cts = data.x
    data.y_label = data.x.sign()

    data.U_k = U_k
    data.noise_level = 0

    # Yes both cases are bl, but you might want to change the latter
    data.signal_type = "bl" if bandlimit_signals else "bl"
    return data


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


def weightless_conv_adjt(model, x, adj_t):
    local_conv = model.module_0
    normalised_edge_index = gcn_norm(  # yapf: disable
        adj_t,
        None,
        x.size(local_conv.node_dim),
        local_conv.improved,
        local_conv.add_self_loops,
        local_conv.flow,
        x.dtype,
    )

    return local_conv.propagate(normalised_edge_index, x=x, edge_weight=None)


def normalize_adjt_like_conv(model, x, adj_t):
    local_conv = model.module_0
    return gcn_norm(  # yapf: disable
        adj_t,
        None,
        x.size(local_conv.node_dim),
        local_conv.improved,
        local_conv.add_self_loops,
        local_conv.flow,
        x.dtype,
    )


# # We assume M is symmetric
# def sparseTensorToLinear(M):
#     return LinearOperator(
#         shape=tuple(M.sizes()),
#         matvec=lambda v: spmm(M, v.),
#         rmatvec=lambda v: spmm(M, v),
#         matmat=lambda X: spmm(M, X),
#         rmatmat=lambda X: spmm(M, X),
#         dtype=M.dtype(),
#     )


def mk_tasks_cora(
    feat_dim=64,
    hidden_dim=32,
    num_conv_layers=1,
):
    assert num_conv_layers > 0

    def gconv(d1, d2):
        return (GCNConv(d1, d2), "x, edge_index -> x")

    layers = (
        [gconv(feat_dim, hidden_dim)]
        + [gconv(hidden_dim, hidden_dim) for _ in range(num_conv_layers - 1)]
        + [Linear(hidden_dim, 1, bias=False, weight_initializer="glorot")]
    )
    model = Sequential("x, edge_index", layers)
    data = Planetoid(root="/tmp/Citeseer", name="Cora")[0]
    data = T.LargestConnectedComponents(1)(data)

    # Train model
    console.log("Training SGC...")


def mk_tasks_sgc(
    feat_dim=64,
    hidden_dim=32,
    signal_type="lap_pinv",
    num_conv_layers=1,
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


# y_obs.shape = sample_size x batch_size
def rec_LS_mult_signals(sample_set, U_k, y_obs):
    # Intermediate dimensions:
    # A = sample_size x bandwidth
    # y_obs = sample size x batch
    # bs = batch x sample_size
    # batch_size = ys.shape[-1]
    A = U_k[sample_set, :]
    # do the calcs as per usual
    # bandwidth x batch:
    xs = th.linalg.lstsq(A, y_obs, driver="gelsd").solution
    zs = U_k @ xs  # nodes x batch
    return zs


# Assumes x_obs.shape is either (num_nodes,)
# or (num_nodes, num_tasks)
def rec_dirichlet(sample_set, L, x_obs):
    num_nodes = L.shape[0]
    M = s.construct_sample_matrix(sample_set, num_nodes).double().to_dense().numpy()

    # M = cvx.spmatrix(1.0, np.arange(num_samples), np.array(sample_set), size=(num_samples, num_nodes))
    def inner(x_obs):
        args = {
            "P": cvx.matrix(L.numpy()),
            "q": cvx.matrix(np.zeros(num_nodes)),
            "A": cvx.matrix(M),
            "b": cvx.matrix(x_obs.numpy().astype(np.double)),
        }
        # Note that the solution, x, to this has the property that
        # Lx is all zeros except at the sample, where it has the value
        # -1 * y, which is the negation of the slack variable.
        x_all = th.from_numpy(np.array(coneqp(**args)["x"])).squeeze(-1).float()
        # if len(sample_set) == 10:
        #     breakpoint()
        return x_all

    if len(x_obs.shape) == 1:
        return inner(x_obs)
    else:
        res = [inner(x) for x in x_obs.T]
        return th.stack(res).T


def rec_dirichlet_direct(sample_set, L, x_obs):
    num_nodes = L.shape[0]
    sample_mask_c = th.ones(num_nodes, dtype=bool)
    sample_mask_c[sample_set] = False

    C = L[sample_mask_c][:, sample_mask_c]
    BT = L[sample_mask_c][:, sample_set]
    x_u = -1 * th.linalg.lstsq(C, BT @ x_obs, driver="gelsd").solution

    result_shape = list(x_obs.shape)
    result_shape[0] = num_nodes
    result = th.zeros(result_shape)
    result[sample_set] = x_obs
    result[sample_mask_c] = x_u
    return result


# generic greedy sampling function
# have to turn typechecking off for parallelism
# @typechecked
def greedy_sampling_gc(
    graph: th.Tensor,
    num_nodes: int,
    sample_fn: Callable,
    calc_once_a_loop: Optional[Callable] = None,
    bandwidth: int = 1,
    max_samples: Optional[int] = None,
    U: Optional[th.Tensor] = None,
    normalization: Optional[str] = "sym",
    internal_dtype=th.float32,
):
    if max_samples is None:
        max_samples = bandwidth
    if U is None:
        U = s.calc_eigenbasis(graph, num_nodes, normalization=normalization)
    L = s.calc_laplacian(graph, num_nodes, normalization=normalization).type(
        internal_dtype
    )
    U_k = s.restrict_eigenbasis(U, bandwidth).type(internal_dtype).contiguous()
    sampling_set, overall_aggs = [], []
    for _ in tqdm(range(max_samples)):
        # The once-per-outer-loop calculations
        if calc_once_a_loop is None:
            big_calc_output = None
        else:
            big_calc_output = calc_once_a_loop(L, U_k, sampling_set)
        # the inner loop
        possible_samples = list(set(range(num_nodes)) - set(sampling_set))
        # this version is slower:
        # possible_sampling_sets = th.tensor(
        #     [sampling_set + [s] for s in possible_samples]
        # )
        possible_sampling_sets = th.hstack(
            [
                th.tensor(sampling_set, dtype=th.long).repeat(len(possible_samples), 1),
                th.tensor(possible_samples, dtype=th.long).unsqueeze(1),
            ]
        )
        ## vmap halves speed
        sfn = lambda sset: sample_fn(big_calc_output, L, U_k, sset)
        losses, aggs = zip(*[sfn(sset) for sset in possible_sampling_sets])
        # find optimal greedy choice
        min_loss_index = np.argmin(losses)
        best_s = possible_samples[min_loss_index]
        corresponding_agg = aggs[min_loss_index]
        sampling_set.append(best_s)
        overall_aggs.append(corresponding_agg)
        gc.collect()
    return sampling_set, overall_aggs


# Note: need to use th.float64 for feat_prop so it doesnt explode
def greedy_sampling_data(
    data,
    sample_fn,
    calc_once_a_loop=None,
    max_samples=None,
    U=None,
    normalization="sym",
    internal_dtype=th.float64,
):
    return greedy_sampling_gc(
        graph=data.edge_index,
        num_nodes=data.num_nodes,
        sample_fn=sample_fn,
        calc_once_a_loop=calc_once_a_loop,
        bandwidth=data.U_k.shape[1],
        max_samples=max_samples,
        U=U,
        normalization=normalization,
        internal_dtype=internal_dtype,
    )[0]


# In the case of signal_type == "bl" and no noise, we get a lot of
# numeric instability issues from sigma being low rank; it's much better to
# directly work with U_k
def mk_anal_class_err_fn_stable(data, rec_type, noise_level):
    L = s.calc_laplacian(
        data.edge_index,
        data.num_nodes,
        normalization="sym",
    )
    L_eigs = th.linalg.eigvalsh(L)
    L_eigs[0] = 0
    if (noise_level == 0) & (data.signal_type == "bl"):
        # sig2 = (data.U_k @ data.U_k.T).diag()
        sig2 = data.U_k.square().sum(dim=1)

        if rec_type == "feat_prop":
            # Unlike the other version, only half the time is spent in solve()
            # why??
            def class_err(_, L, U_k, sample_set):
                N = L.shape[0]
                k = U_k.shape[1]
                sample_mask_c = th.ones(N, dtype=bool)
                sample_mask_c[sample_set] = False
                # About 12% of the time is spent casting L down to float
                C = L[sample_mask_c][:, sample_mask_c].float()
                try:
                    delta = th.linalg.solve(C, U_k[sample_mask_c]).float()
                except:
                    try:
                        delta = th.linalg.lstsq(
                            C, U_k[sample_mask_c], driver="gelsd"
                        ).solution.float()
                    except:
                        return np.infty, ()
                RSU_k = U_k.clone()
                RSU_k[sample_mask_c] -= delta * L_eigs[:k]
                # c = (RSU_k @ U_k.T).diag()
                c = th.linalg.vecdot(RSU_k, U_k)
                nu2 = RSU_k.square().sum(dim=1)
                class_err = (c / th.sqrt(nu2 * sig2)).clamp(-1, 1).arccos().sum(
                    dim=-1
                ) / pi
                return class_err, ()

            return class_err
        elif rec_type == "LS":
            # nu2 = c
            def class_err(_, L, U_k, sample_set):
                # calculate P = U_k[sample_set].pinverse() @ U_k[sample_set]
                _, _, VV = th.linalg.svd(U_k[sample_set], full_matrices=False)
                # annoyingly VV.T @ VV is close but not exactly I
                # and this causes some noticable instability in our arccos calc
                P = VV.T @ VV
                # RS = U_k @ th.linalg.pinv(U_k[sample_set])
                RSU_Sk = U_k @ P
                # c = th.diag(RS @ signal_cov[sample_set])
                c = RSU_Sk.square().sum(dim=1)
                class_err = th.sqrt(c / sig2).clamp(-1, 1).arccos().sum(dim=-1) / pi
                return class_err, ()

            return class_err

        else:
            return mk_anal_class_err_fn(data, rec_type, noise_level)
    else:
        return mk_anal_class_err_fn(data, rec_type, noise_level)


# In the case of signal_type == "bl" and no noise, we get a lot of
# numeric instability issues from sigma being low rank; it's much better to
# directly work with U_k
def mk_anal_class_err_fn_sgc_stable(data, model, rec_type, noise_level):
    L = s.calc_laplacian(
        data.edge_index,
        data.num_nodes,
        normalization="sym",
    )
    L_eigs = th.linalg.eigvalsh(L)
    L_eigs[0] = 0
    G = model.G
    if (noise_level == 0) & (data.signal_type == "bl"):
        # sig2 = (data.U_k @ data.U_k.T).diag()
        sig2 = (G @ data.U_k).square().sum(dim=1)

        if rec_type == "feat_prop":
            # Unlike the other version, only half the time is spent in solve()
            # why??
            def class_err(_, L, U_k, sample_set):
                N = L.shape[0]
                k = U_k.shape[1]
                sample_mask_c = th.ones(N, dtype=bool)
                sample_mask_c[sample_set] = False
                # About 12% of the time is spent casting L down to float
                C = L[sample_mask_c][:, sample_mask_c].float()
                U_k = U_k.float()
                try:
                    delta = th.linalg.solve(C, U_k[sample_mask_c]).float()
                except:
                    try:
                        delta = th.linalg.lstsq(
                            C, U_k[sample_mask_c], driver="gelsd"
                        ).solution.float()
                    except:
                        return np.infty, ()
                RSU_k = U_k.clone()
                RSU_k[sample_mask_c] -= delta * L_eigs[:k]
                GRSU_k = G @ RSU_k
                # c = (RSU_k @ U_k.T).diag()
                c = th.linalg.vecdot(GRSU_k, G @ U_k)
                nu2 = GRSU_k.square().sum(dim=1)
                class_err = (c / th.sqrt(nu2 * sig2)).clamp(-1, 1).arccos().sum(
                    dim=-1
                ) / pi
                return class_err, ()

            return class_err
        elif rec_type == "LS":
            # nu2 = c
            GUk = G @ data.U_k

            def class_err(_, L, U_k, sample_set):
                # calculate P = U_k[sample_set] @ U_k[sample_set].pinverse()
                _, _, VV = th.linalg.svd(U_k[sample_set], full_matrices=False)
                # P = VV.T @ VV
                # RS = U_k @ th.linalg.pinv(U_k[sample_set])
                # RSU_Sk = U_k @ P
                # GRSU_Sk = G @ U_k @ P
                # this following has the right 2-norm
                # and is much cheaper to compute than G @ U_k @ P
                GRSU_Sk = GUk @ VV.T
                # c = th.diag(RS @ signal_cov[sample_set])
                c = GRSU_Sk.square().sum(dim=1)
                class_err = th.sqrt(c / sig2).clamp(-1, 1).arccos().sum(dim=-1) / pi
                return class_err, ()

            return class_err

        else:
            return mk_anal_class_err_fn_sgc(data, model, rec_type, noise_level)
    else:
        return mk_anal_class_err_fn_sgc(data, model, rec_type, noise_level)


def mk_anal_class_err_fn(data, rec_type, noise_level):
    L = s.calc_laplacian(
        data.edge_index,
        data.num_nodes,
        normalization="sym",
    )
    match data.signal_type:
        case "bl":
            signal_cov = data.U_k @ data.U_k.T
        case "lap":
            signal_cov = L.float()
        case "lap_pinv":
            signal_cov = s.connected_laplacian_pinv(L, normalization="sym").float()

    sig2 = signal_cov.diag()
    noisy_signal_cov = signal_cov + ((noise_level**2) * th.eye(signal_cov.shape[0]))
    if rec_type == "LS":

        def class_err(_, L, U_k, sample_set):
            RS = U_k @ th.linalg.pinv(U_k[sample_set])
            c = th.diag(RS @ signal_cov[sample_set])
            nu2 = ((RS @ noisy_signal_cov[sample_set][:, sample_set]) @ RS.T).diag()
            class_err = (c / th.sqrt(nu2 * sig2)).clamp(-1, 1).arccos().sum(dim=-1) / pi
            return class_err, ()

    elif rec_type == "feat_prop":

        def class_err(_, L, U_k, sample_set):
            N = data.num_nodes
            m = sample_set.shape[0]
            RS = th.zeros(N, m)
            sample_mask_c = th.ones(N, dtype=bool)
            sample_mask_c[sample_set] = False
            RS[sample_set] = th.eye(m)
            C = L[sample_mask_c][:, sample_mask_c]
            BT = L[sample_mask_c][:, sample_set]
            try:
                RS[sample_mask_c] = -1 * th.linalg.solve(C, BT).float()
            except:
                try:
                    RS[sample_mask_c] = (
                        -1 * th.linalg.lstsq(C, BT, driver="gelsd").solution.float()
                    )
                except:
                    return np.infty, ()
            c = th.diag(RS @ signal_cov[sample_set])
            # nu2 = ((RS @ noisy_signal_cov[sample_set][:, sample_set]) @ RS.T).diag()
            nu2 = th.linalg.vecdot(RS @ noisy_signal_cov[sample_set][:, sample_set], RS)
            class_err = (c / th.sqrt(nu2 * sig2)).clamp(-1, 1).arccos().sum(dim=-1) / pi
            return class_err, ()

    return class_err


def mk_anal_class_err_fn_sgc(data, model, rec_type, noise_level):
    L = s.calc_laplacian(
        data.edge_index,
        data.num_nodes,
        normalization="sym",
    )
    G = model.G
    match data.signal_type:
        case "bl":
            signal_cov = data.U_k @ data.U_k.T
        case "lap":
            signal_cov = L.float()
        case "lap_pinv":
            signal_cov = s.connected_laplacian_pinv(L, normalization="sym").float()

    sig2 = (G @ signal_cov @ G).diag()
    noisy_signal_cov = signal_cov + ((noise_level**2) * th.eye(signal_cov.shape[0]))
    if rec_type == "LS":

        def class_err(_, L, U_k, sample_set):
            RS = U_k @ th.linalg.pinv(U_k[sample_set])
            c = th.diag(G @ RS @ signal_cov[sample_set] @ G)
            nu2 = (
                (G @ RS @ noisy_signal_cov[sample_set][:, sample_set]) @ RS.T @ G.T
            ).diag()
            class_err = (c / th.sqrt(nu2 * sig2)).clamp(-1, 1).arccos().sum(dim=-1) / pi
            return class_err, ()

    elif rec_type == "feat_prop":

        def class_err(_, L, U_k, sample_set):
            N = data.num_nodes
            m = sample_set.shape[0]
            RS = th.zeros(N, m)
            sample_mask_c = th.ones(N, dtype=bool)
            sample_mask_c[sample_set] = False
            RS[sample_set] = th.eye(m)
            C = L[sample_mask_c][:, sample_mask_c]
            BT = L[sample_mask_c][:, sample_set]
            # Have to use lstsq as solve is not stable enough
            try:
                # RS[sample_mask_c] = (
                #     -1 * th.linalg.lstsq(C, BT, driver="gelsd").solution.float()
                # )
                RS[sample_mask_c] = -1 * th.linalg.solve(C, BT).float()
            except:
                try:
                    RS[sample_mask_c] = (
                        -1 * th.linalg.lstsq(C, BT, driver="gelsd").solution.float()
                    )
                except:
                    return np.infty, ()
            GRS = G @ RS
            c = th.diag(GRS @ (signal_cov[sample_set] @ G))
            nu2 = ((GRS @ noisy_signal_cov[sample_set][:, sample_set]) @ GRS.T).diag()
            # c = th.diag(G @ RS @ signal_cov[sample_set] @ G)
            # nu2 = (
            #     (G @ RS @ noisy_signal_cov[sample_set][:, sample_set]) @ RS.T @ G.T
            # ).diag()
            class_err = (c / th.sqrt(nu2 * sig2)).clamp(-1, 1).arccos().sum(dim=-1) / pi
            return class_err, ()

    return class_err


def mk_anal_rec_err_fn(data, rec_type, noise_level):
    L = s.calc_laplacian(
        data.edge_index,
        data.num_nodes,
        normalization="sym",
    )
    match data.signal_type:
        case "bl":
            signal_cov = data.U_k @ data.U_k.T
        case "lap":
            signal_cov = L.float()
        case "lap_pinv":
            signal_cov = s.connected_laplacian_pinv(L, normalization="sym").float()

    sig2 = signal_cov.diag()
    noisy_signal_cov = signal_cov + ((noise_level**2) * th.eye(signal_cov.shape[0]))
    if rec_type == "LS":

        def rec_err_fn(_, L, U_k, sample_set):
            RS = U_k @ th.linalg.pinv(U_k[sample_set])
            c = th.diag(RS @ signal_cov[sample_set])
            nu2 = ((RS @ noisy_signal_cov[sample_set][:, sample_set]) @ RS.T).diag()
            rec_err = (sig2 + nu2 - (2 * c)).sum()
            return rec_err, ()

    elif rec_type == "feat_prop":

        def rec_err_fn(_, L, U_k, sample_set):
            N = data.num_nodes
            m = sample_set.shape[0]
            RS = th.zeros(N, m)
            sample_mask_c = th.ones(N, dtype=bool)
            sample_mask_c[sample_set] = False
            RS[sample_set] = th.eye(m)
            C = L[sample_mask_c][:, sample_mask_c]
            BT = L[sample_mask_c][:, sample_set]
            try:
                RS[sample_mask_c] = -1 * th.linalg.solve(C, BT).solution.float()
            except:
                try:
                    RS[sample_mask_c] = (
                        -1 * th.linalg.lstsq(C, BT, driver="gelsd").solution.float()
                    )
                except:
                    return np.infty, ()
            c = th.diag(RS @ signal_cov[sample_set])
            nu2 = ((RS @ noisy_signal_cov[sample_set][:, sample_set]) @ RS.T).diag()
            rec_err = (sig2 + nu2 - (2 * c)).sum()
            return rec_err, ()

    return rec_err_fn


def analytic_error_simple_given_sets(data, rec_type, noise_level, sample_sets):
    L = s.calc_laplacian(
        data.edge_index,
        data.num_nodes,
        normalization="sym",
    )
    match data.signal_type:
        case "bl":
            signal_cov = data.U_k @ data.U_k.T
        case "lap":
            signal_cov = L
        case "lap_pinv":
            signal_cov = s.connected_laplacian_pinv(L, normalization="sym")

    if rec_type == "LS":
        RSs = vmap(lambda s: data.U_k @ th.linalg.pinv(data.U_k[s]))(sample_sets)
    elif rec_type == "feat_prop":

        def genRS(sample_set):
            N = data.num_nodes
            m = sample_set.shape[0]
            R_S = th.zeros(N, m)
            sample_mask_c = th.ones(N, dtype=bool)
            sample_mask_c[sample_set] = False
            R_S[sample_set] = th.eye(m)
            C = L[sample_mask_c][:, sample_mask_c]
            BT = L[sample_mask_c][:, sample_set]
            R_S[sample_mask_c] = (
                -1 * th.linalg.lstsq(C, BT, driver="gelsd").solution.float()
            )
            return R_S

        RSs = th.stack([genRS(ss) for ss in sample_sets])

    else:
        raise ValueError("RS not defined for {rec_type} in analytic error")

    signal_cov = signal_cov.float()
    RSs = RSs.float()

    c = vmap(th.diag)(th.bmm(RSs, signal_cov[sample_sets]))
    # c = th.einsum("bij,bji->bi", RSs, signal_cov[sample_sets])
    sig2 = signal_cov.diag()
    noisy_signal_cov = signal_cov + ((noise_level**2) * th.eye(signal_cov.shape[0]))
    nu2 = vmap(lambda RS, S: ((RS @ noisy_signal_cov[S][:, S]) @ RS.T).diag())(
        RSs, sample_sets
    )
    rec_err = vmap(lambda s2, v2, c: th.sum(s2 + v2 - (2 * c)), in_dims=(None, 0, 0))(
        sig2, nu2, c
    )
    class_err = (c / th.sqrt(nu2 * sig2)).clamp(-1, 1).arccos().sum(dim=1) / pi
    # true_rec_err = data.U_k.shape[1] - th.linalg.matrix_rank(RSs)
    return rec_err, class_err
    # return sig2, nu2, c


def analytic_error_sgc_given_sets(
    data, model, signal_type, rec_type, noise_level, sample_sets
):
    L = s.calc_laplacian(
        data.edge_index,
        data.num_nodes,
        normalization="sym",
    )
    match signal_type:
        case "bl":
            signal_cov = data.U_k @ data.U_k.T
        case "lap":
            signal_cov = L
        case "lap_pinv":
            signal_cov = s.connected_laplacian_pinv(L, normalization="sym")

    if rec_type == "LS":
        RSs = vmap(lambda s: data.U_k @ th.linalg.pinv(data.U_k[s]))(sample_sets)
    elif rec_type == "feat_prop":

        def genRS(sample_set):
            N = data.num_nodes
            m = sample_set.shape[0]
            R_S = th.zeros(N, m)
            sample_mask_c = th.ones(N, dtype=bool)
            sample_mask_c[sample_set] = False
            R_S[sample_set] = th.eye(m)
            C = L[sample_mask_c][:, sample_mask_c]
            BT = L[sample_mask_c][:, sample_set]
            R_S[sample_mask_c] = (
                -1 * th.linalg.lstsq(C, BT, driver="gelsd").solution.float()
            )
            return R_S

        RSs = th.stack([genRS(ss) for ss in sample_sets])

    else:
        raise ValueError("RS not defined for {rec_type} in analytic error")

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
    signal_cov = signal_cov.float()
    RSs = RSs.float()

    c = vmap(th.diag)(th.bmm(RSs, signal_cov[sample_sets]))
    # c = th.einsum("bij,bji->bi", RSs, signal_cov[sample_sets])
    sig2 = signal_cov.diag()
    noisy_signal_cov = signal_cov + ((noise_level**2) * th.eye(signal_cov.shape[0]))
    nu2 = vmap(lambda RS, S: ((RS @ noisy_signal_cov[S][:, S]) @ RS.T).diag())(
        RSs, sample_sets
    )
    rec_err = vmap(lambda s2, v2, c: th.sum(s2 + v2 - (2 * c)), in_dims=(None, 0, 0))(
        sig2, nu2, c
    )
    class_err = (c / th.sqrt(nu2 * sig2)).clamp(-1, 1).arccos().sum(dim=1) / pi
    # true_rec_err = data.U_k.shape[1] - th.linalg.matrix_rank(RSs)
    return rec_err, class_err
    # return sig2, nu2, c


def mk_anal_df(
    class_fn, rec_fn, L, U_k, all_samps, sample_sizes, sample_types, feat_dim=1
):
    anal_rec_errs = []
    anal_class_errs = []
    for samp_size in tqdm(sample_sizes):
        samp_sets = all_samps[:, :samp_size]
        anal_rec_errs.append(th.stack([rec_fn((), L, U_k, s)[0] for s in samp_sets]))
        anal_class_errs.append(
            th.stack([class_fn((), L, U_k, s)[0] for s in samp_sets])
        )
    # we assume rec_fn calculates per feature dimension
    anal_rec_errs = feat_dim * th.stack(anal_rec_errs).transpose(0, 1).numpy()
    anal_class_errs = th.stack(anal_class_errs).transpose(0, 1).numpy()
    idx = pd.MultiIndex.from_product(
        [
            sample_types,
            sample_sizes,
        ],
        # [np.arange(length) for length in class_error.shape[1:]],
        names=("Sampling Type", "Sample Size"),
    )

    error_df = pd.DataFrame(
        {
            "Analytic Reconstruction Error": anal_rec_errs.flatten(),
            "Analytic Classification Error": anal_class_errs.flatten(),
        },
        index=idx,
    )
    error_df = error_df.reset_index(level=[0, 1])
    return error_df


def analytic_error_simple(
    data,
    signal_type,
    rec_type,
    noise_levels,
    sample_sizes,
    num_sample_sets,
):
    def inner(noise_level, sample_size):
        sample_sets = th.stack(
            [th.randperm(data.num_nodes)[:sample_size] for _ in range(num_sample_sets)]
        )
        rec_errs, class_errs = analytic_error_simple_given_sets(
            data, signal_type, rec_type, noise_level, sample_sets
        )
        return {
            "Sample Size": sample_size,
            "Noise Level": noise_level,
            "Analytic Reconstruction Error": rec_errs.mean().item(),
            "Analytic Classification Error": class_errs.mean().item(),
        }

    res = Parallel(n_jobs=-1, verbose=5)(
        delayed(inner)(*tup)
        for tup in itertools.product(noise_levels, sample_sizes.numpy())
    )
    return pd.DataFrame(res)


def eval_errs(data, x_rec, sample_sizes, sample_types):
    y_label_pred = th.sign(x_rec)
    y_cts_pred = x_rec
    class_error = th.abs(y_label_pred - data.y_label) / 2
    rec_error = (y_cts_pred - data.y_cts).square()

    # Aggregate errs:
    # sum over nodes
    class_error = class_error.sum(dim=-2)  # .mean(dim=3)
    rec_error = rec_error.sum(dim=-2)  # .norm(dim=-1).square()
    # These each have dims (num sampling types, |sample sizes|, num_tasks)

    # We now have
    idx = pd.MultiIndex.from_product(
        [
            sample_types,
            sample_sizes,
            np.arange(class_error.shape[2]),
        ],
        # [np.arange(length) for length in class_error.shape[1:]],
        names=("Sampling Type", "Sample Size", "Signal idx"),
    )

    error_df = pd.DataFrame(
        {"class_error": class_error.flatten(), "rec_error": rec_error.flatten()},
        index=idx,
    )
    error_df = error_df.reset_index(level=[0, 1, 2])
    return error_df


def eval_errs_gcn(gcn, data, x_rec, sample_sizes, sample_types):
    data = T.ToSparseTensor()(data)
    # fff = lambda x: gcn(x, data.edge_index).squeeze().sign()
    fff = lambda x: gcn(x, data.adj_t).squeeze().sign()
    print("Running GCN on reconstructed data...")
    # tick = time.perf_counter()
    # y_label_pred = th.stack([vmap(vmap(fff))(x) for x in tqdm(x_rec)])
    y_label_pred = fff(x_rec)
    # tock = time.perf_counter()
    # print(f"took {tock-tick}s")
    # [fff(x) for x in x_rec[0, 0]]
    # y_label_pred = data.model(data.y_cts[0].T, data.edge_index).sign()
    y_cts_pred = x_rec
    # class_error = (y_label_pred - data.y_label.T.squeeze()).abs() * 0.5
    # # sum over hidden dim
    # rec_error = (y_cts_pred - data.y_cts).sum(dim=-2)
    class_error = th.abs(y_label_pred - data.y_label) / 2
    rec_error = (y_cts_pred - data.y_cts).square()
    # Aggregate errs:
    # sum over nodes
    class_error = class_error.sum(dim=-1).detach()  # .mean(dim=3)
    rec_error = rec_error.sum(dim=[-1, -2]).detach()  # .norm(dim=-1).square()
    # We now have
    idx = pd.MultiIndex.from_product(
        [
            sample_types,
            sample_sizes,
        ],
        # [np.arange(length) for length in class_error.shape[1:]],
        names=("Sampling Type", "Sample Size"),
    )

    error_df = pd.DataFrame(
        {"class_error": class_error.flatten(), "rec_error": rec_error.flatten()},
        index=idx,
    )
    error_df = error_df.reset_index(level=[0, 1])
    return error_df


def plot_errors_(error_df):
    plt.figure(figsize=(24, 6))
    sns.scatterplot(
        data=error_df,
        x="rec_error",
        y="class_error",
        hue="Sample Size",
        palette="coolwarm",
    )
    sns.lineplot(
        data=error_df.groupby("Sample Size").mean(),
        x="rec_error",
        y="class_error",
        color="black",
        linewidth=2,
        label="Mean Err (empirical)",
    )

    plt.legend()
    plt.show()


def plot_errors_no_anal(
    error_df, title="", filename="", bandlimit=None, indep_var="Sample Size"
):
    fig, axes = plt.subplots(1, 3, figsize=(24, 6), constrained_layout=True)

    # Plot 1: rec_error vs class_error
    sns.scatterplot(
        data=error_df.drop(columns=["Sampling Type"]),
        x="rec_error",
        y="class_error",
        hue=indep_var,
        palette="coolwarm",
        ax=axes[0],
    )
    sns.lineplot(
        data=error_df.drop(columns=["Sampling Type"]).groupby(indep_var).mean(),
        x="rec_error",
        y="class_error",
        color="black",
        linewidth=2,
        label="Mean Err (empirical)",
        ax=axes[0],
    )
    axes[0].set_title("Class Error vs. Reconstruction Error")
    axes[0].legend()
    # Plot 2: lineplot with CI for rec_error vs sample size
    sns.lineplot(
        data=error_df,
        x="Sample Size",
        y="rec_error",
        hue="Sampling Type",
        estimator="mean",
        errorbar=("ci", 95),
        linewidth=2,
        # color="steelblue",
        ax=axes[1],
    )
    # sns.lineplot(
    #     data=error_df.groupby(["Sampling Type", "Sample Size"]).mean(),
    #     x="Sample Size",
    #     y="rec_error",
    #     color="red",
    #     linewidth=2,
    #     label="Mean Err (empirical)",
    #     ax=axes[1],
    # )
    axes[1].set_title(f"Reconstruction Error vs. {indep_var}")

    # Plot 3: lineplot with CI for class_error vs sample size
    sns.lineplot(
        data=error_df,
        x="Sample Size",
        y="class_error",
        hue="Sampling Type",
        estimator="mean",
        errorbar=("ci", 95),
        linewidth=2,
        # color="indianred",
        ax=axes[2],
    )

    axes[2].set_title(f"Classification Error vs. {indep_var}")

    if bandlimit is not None:
        axes[1].axvline(x=bandlimit, color="gray", linestyle="--")
        axes[2].axvline(x=bandlimit, color="gray", linestyle="--")

    for ax in axes:
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    fig.suptitle(title)
    if filename:
        fig.savefig(filename.with_suffix(".png"), bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()


def plot_errors(
    error_df, title="", filename="", bandlimit=None, indep_var="Sample Size"
):
    fig, axes = plt.subplots(1, 3, figsize=(24, 6), constrained_layout=True)

    # Plot 1: rec_error vs class_error
    sns.scatterplot(
        data=error_df.drop(columns=["Sampling Type"]),
        x="rec_error",
        y="class_error",
        hue=indep_var,
        palette="coolwarm",
        ax=axes[0],
    )
    sns.lineplot(
        data=error_df.drop(columns=["Sampling Type"]).groupby(indep_var).mean(),
        x="rec_error",
        y="class_error",
        color="black",
        linewidth=2,
        label="Mean Err (empirical)",
        ax=axes[0],
    )
    axes[0].set_title("Classification Error vs. Reconstruction Error")
    axes[0].legend()
    # Plot 2: lineplot with CI for rec_error vs sample size
    # sns.lineplot(
    #     data=error_df,
    #     x=indep_var,
    #     y="rec_error",
    #     estimator="mean",
    #     errorbar=("ci", 95),
    #     linewidth=2,
    #     color="steelblue",
    #     ax=axes[1],
    # )
    sns.scatterplot(
        data=error_df,
        x=indep_var,
        y="rec_error",
        palette="coolwarm",
        label="Error (empirical)",
        ax=axes[1],
    )
    sns.lineplot(
        data=error_df.groupby(["Sampling Type", indep_var]).mean(),
        x=indep_var,
        y="Analytic Reconstruction Error",
        hue="Sampling Type",
        # color="black",
        linewidth=2,
        # label="Mean Err (Analytic)",
        ax=axes[1],
    )
    # sns.lineplot(
    #     data=error_df.groupby(indep_var).mean(),
    #     x=indep_var,
    #     y="rec_error",
    #     color="red",
    #     linewidth=2,
    #     label="Mean Err (empirical)",
    #     ax=axes[1],
    # )
    axes[1].set_title(f"Reconstruction Error vs. {indep_var}")

    # Plot 3: lineplot with CI for class_error vs sample size
    # sns.lineplot(
    #     data=error_df,
    #     x=indep_var,
    #     y="class_error",
    #     estimator="mean",
    #     errorbar=("ci", 95),
    #     linewidth=2,
    #     color="indianred",
    #     ax=axes[2],
    # )
    #
    # sns.scatterplot(
    #     data=error_df,
    #     x=indep_var,
    #     y="class_error",
    #     palette="coolwarm",
    #     label="Error (empirical)",
    #     ax=axes[2],
    # )
    sns.lineplot(
        data=error_df.groupby(["Sampling Type", indep_var]).mean(),
        x=indep_var,
        y="Analytic Classification Error",
        # color="black",
        errorbar=("ci", 95),
        hue="Sampling Type",
        linewidth=2,
        # label="Mean Err (Analytic)",
        ax=axes[2],
    )
    axes[2].set_title(f"Classification Error vs. {indep_var}")

    if bandlimit is not None:
        axes[1].axvline(x=bandlimit, color="gray", linestyle="--")
        axes[2].axvline(x=bandlimit, color="gray", linestyle="--")

    for ax in axes:
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    fig.suptitle(title)
    if filename:
        print(f"Saving to filename {filename}")
        fig.savefig(filename.with_suffix(".png"), bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()


def plot_errors_camsap(
    error_df, title="", filename=Path("."), bandlimit=None, indep_var="Sample Size"
):
    fig, ax = plt.subplots(
        figsize=(3.5, 2.5), constrained_layout=True
    )  # width x height in inches
    # fig, ax = plt.subplots(figsize=(7, 5))  # width x height in inches
    sns.set_palette("deep")
    # fig, axes = plt.subplots(1, 3, figsize=(24, 6), constrained_layout=True)

    # Plot 1: rec_error vs class_error
    sns.scatterplot(
        data=error_df.drop(columns=["Sampling Type"]),
        x="rec_error",
        y="class_error",
        hue=indep_var,
        palette="coolwarm",
        ax=ax,
    )
    sns.lineplot(
        data=error_df.drop(columns=["Sampling Type"]).groupby(indep_var).mean(),
        x="rec_error",
        y="class_error",
        color="black",
        linewidth=2,
        label="Mean Err (empirical)",
        ax=ax,
    )
    # ax.set_title("Classification Error vs. Reconstruction Error")
    ax.legend()

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    comp_filename = filename.with_name(filename.stem + "_class_vs_rec")
    print(f"Saving to filename {comp_filename}")
    fig.savefig(comp_filename.with_suffix(".png"), bbox_inches="tight", dpi=150)
    fig.savefig(comp_filename.with_suffix(".svg"), bbox_inches="tight", dpi=150)
    # Plot 2: lineplot with CI for rec_error vs sample size
    # sns.lineplot(
    #     data=error_df,
    #     x=indep_var,
    #     y="rec_error",
    #     estimator="mean",
    #     errorbar=("ci", 95),
    #     linewidth=2,
    #     color="steelblue",
    #     ax=axes[1],
    # )
    fig, ax = plt.subplots(
        figsize=(3.5, 2.5), constrained_layout=True
    )  # width x height in inches
    # fig, ax = plt.subplots(figsize=(7, 5))  # width x height in inches
    sns.set_palette("deep")
    sns.scatterplot(
        data=error_df,
        x=indep_var,
        y="rec_error",
        palette="coolwarm",
        label="Error (empirical)",
        ax=ax,
    )
    sns.lineplot(
        data=error_df.groupby(["Sampling Type", indep_var]).mean(),
        x=indep_var,
        y="Analytic Reconstruction Error",
        hue="Sampling Type",
        # color="black",
        linewidth=2,
        # label="Mean Err (Analytic)",
        ax=ax,
    )
    # sns.lineplot(
    #     data=error_df.groupby(indep_var).mean(),
    #     x=indep_var,
    #     y="rec_error",
    #     color="red",
    #     linewidth=2,
    #     label="Mean Err (empirical)",
    #     ax=axes[1],
    # )
    # axes[1].set_title(f"Reconstruction Error vs. {indep_var}")

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    if bandlimit is not None:
        axes[1].axvline(x=bandlimit, color="gray", linestyle="--")
    rec_filename = filename.with_name(filename.stem + "_recon")
    print(f"Saving to filename {rec_filename}")
    fig.savefig(rec_filename.with_suffix(".png"), bbox_inches="tight", dpi=150)
    fig.savefig(rec_filename.with_suffix(".svg"), bbox_inches="tight", dpi=150)

    # Plot 3: lineplot with CI for class_error vs sample size
    # sns.lineplot(
    #     data=error_df,
    #     x=indep_var,
    #     y="class_error",
    #     estimator="mean",
    #     errorbar=("ci", 95),
    #     linewidth=2,
    #     color="indianred",
    #     ax=axes[2],
    # )
    #
    # sns.scatterplot(
    #     data=error_df,
    #     x=indep_var,
    #     y="class_error",
    #     palette="coolwarm",
    #     label="Error (empirical)",
    #     ax=axes[2],
    # )
    #

    fig, ax = plt.subplots(
        figsize=(3.5, 2.5), constrained_layout=True
    )  # width x height in inches
    # fig, ax = plt.subplots(figsize=(7, 5))  # width x height in inches
    sns.set_palette("deep")
    sns.lineplot(
        data=error_df.groupby(["Sampling Type", indep_var]).mean(),
        x=indep_var,
        y="Analytic Classification Error",
        # color="black",
        errorbar=("ci", 95),
        hue="Sampling Type",
        linewidth=2,
        # label="Mean Err (Analytic)",
        ax=ax,
    )
    # axes[2].set_title(f"Classification Error vs. {indep_var}")
    #
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    if bandlimit is not None:
        ax.axvline(x=bandlimit, color="gray", linestyle="--")
    class_filename = filename.with_name(filename.stem + "_classif")
    print(f"Saving to filename {class_filename}")
    fig.savefig(class_filename.with_suffix(".png"), bbox_inches="tight", dpi=150)
    fig.savefig(class_filename.with_suffix(".svg"), bbox_inches="tight", dpi=150)

    if bandlimit is not None:
        axes[1].axvline(x=bandlimit, color="gray", linestyle="--")
        axes[2].axvline(x=bandlimit, color="gray", linestyle="--")

    # fig.suptitle(title)
    # print(f"Saving to filename {filename}")
    # fig.savefig(filename.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close()


def simple_pipeline(
    num_graphs=16,
    graph_type="BA",
    signal_type="bl",
    rec_type="LS",
    num_nodes=300,
    num_sigs=300,
    max_sample_size=None,
    bandlimit=None,
    # sample_type="Random",
    noise_level=0,
    raw_signal_dist="gaussian",
):
    if max_sample_size is None:
        max_sample_size = num_nodes

    def per_graph(i):
        # make graph
        tqdm.write("Making Graph...")
        # if graph_type == "CSBM":
        #     data = mk_tasks_csbm(
        #         num_nodes=num_nodes,
        #         feat_dim=1,
        #         num_tasks_per_noise_level=None,
        #         noise_levels=noise_levels,
        #     )
        # else:
        data = mk_tasks_bl(
            graph_type=graph_type,
            num_nodes=num_nodes,
            num_tasks=num_sigs,
            noise_level=noise_level,
            signal_type=signal_type,
            raw_signal_dist="gaussian",
        )
        # Make sample sets
        tqdm.write("Sampling...")
        class_fn = mk_anal_class_err_fn_stable(
            data, rec_type=rec_type, noise_level=noise_level
        )
        rec_fn = mk_anal_rec_err_fn(data, rec_type=rec_type, noise_level=noise_level)
        rec_fn_extra_noise = mk_anal_rec_err_fn(
            data, rec_type=rec_type, noise_level=0.01
        )
        rand_samp = th.randperm(data.num_nodes)[:max_sample_size]
        class_optimal_samp = greedy_sampling_data(
            data,
            class_fn,
            max_samples=max_sample_size,
            internal_dtype=th.float64 if rec_type == "feat_prop" else th.float32,
        )
        if (rec_type == "LS") & (signal_type == "bl") & (noise_level == 0):
            # rec_optimal_samp = greedy_sampling_data(
            #     data,
            #     rec_fn_extra_noise,
            #     max_samples=max_sample_size,
            #     internal_dtype=th.float64 if rec_type == "feat_prop" else th.float32,
            # )

            rec_optimal_samp = ts.greedy_a_samples_only_fast(
                graph=data.edge_index,
                num_nodes=data.num_nodes,
                bandwidth=data.U_k.shape[1],
                max_samples=max_sample_size,
                normalization="sym",
            )

        else:
            rec_optimal_samp = greedy_sampling_data(
                data,
                rec_fn,
                max_samples=max_sample_size,
                internal_dtype=th.float64 if rec_type == "feat_prop" else th.float32,
            )
        all_samps = th.tensor(
            [rand_samp.tolist(), class_optimal_samp, rec_optimal_samp]
        )
        # breakpoint()
        tqdm.write("Reconstructing...")
        L = s.calc_laplacian(
            data.edge_index,
            data.num_nodes,
            normalization="sym",
        ).float()
        x_recs = []
        sample_sizes = np.arange(1, max_sample_size)
        for samp_size in tqdm(sample_sizes):
            samp_sets = all_samps[:, :samp_size]
            if rec_type == "feat_prop":
                x_rec = th.stack(
                    [rec_dirichlet_direct(s, L, data.x[s]) for s in samp_sets]
                )
            elif rec_type == "LS":
                x_rec = vmap(lambda s: rec_LS_mult_signals(s, data.U_k, data.x[s]))(
                    samp_sets
                )
            else:
                raise ValueError("only allowed rec_types are feat_prop and LS")
            x_recs.append(x_rec)
        x_rec = th.stack(x_recs).transpose(0, 1)

        tqdm.write("Evaluating Errs Analytically...")
        sample_types = ["Random", "Classification-optimal", "Reconstruction-optimal"]
        anal_err_df = mk_anal_df(
            class_fn, rec_fn, L, data.U_k, all_samps, sample_sizes, sample_types
        )

        # x_rec now has shape (num_sampling_types, |sample_sizes|, num_nodes, num_tasks)

        # calculate errors
        # tqdm.write("Evaluating Errs Analytically...")
        # anal_err_df = analytic_error_simple(
        #     data,
        #     signal_type,
        #     rec_type,
        #     noise_levels,
        #     sample_sizes,
        #     num_sample_sets,
        # )
        tqdm.write("Evaluating Errs Empirically...")
        err_df = eval_errs(data, x_rec, sample_sizes, sample_types)
        err_df = err_df.join(
            anal_err_df.set_index(["Sampling Type", "Sample Size"]),
            how="left",
            on=["Sampling Type", "Sample Size"],
        )
        err_df["graph_id"] = i
        return err_df

    # total_df = pd.concat([per_graph(i) for i in trange(num_graphs)])
    total_df = pd.concat(
        Parallel(n_jobs=-1, verbose=5)(delayed(per_graph)(i) for i in range(num_graphs))
    )
    sig_dist_path = path / raw_signal_dist
    if not sig_dist_path.is_dir():
        sig_dist_path.mkdir(parents=True)
    fname = f"{graph_type}_{signal_type}_{rec_type}_{noise_level}_noise_{num_nodes}_nodes.csv"
    # plot_errors(
    #     total_df,
    #     title=f"{signal_type} signals + {rec_type} reconstruction + noise level {noise_level}",
    #     filename=path / raw_signal_dist / fname,
    #     bandlimit=bandlimit,
    # )
    total_df.to_csv(path / raw_signal_dist / fname)


def real_pipeline(
    dataset_name="weather",
    bandlimit_signals=False,
    rec_type="LS",
    max_sample_size=None,
    # sample_type="Random",
    noise_level=0,
    raw_signal_dist="gaussian",
):
    assert noise_level == 0, "noise not implemented for real datasets yet"

    tqdm.write("Making Graph...")
    data = mk_tasks_real(dataset_name, bandlimit_signals=bandlimit_signals)
    signal_type = data.signal_type
    if max_sample_size is None:
        max_sample_size = data.num_nodes

        # Make sample sets
    tqdm.write("Sampling...")
    noise_level = 10 ** (-1.5)
    class_fn = mk_anal_class_err_fn_stable(
        data, rec_type=rec_type, noise_level=noise_level
    )
    rec_fn = mk_anal_rec_err_fn(data, rec_type=rec_type, noise_level=noise_level)
    rec_fn_extra_noise = mk_anal_rec_err_fn(data, rec_type=rec_type, noise_level=0.01)
    rand_samp = th.randperm(data.num_nodes)[:max_sample_size]
    class_optimal_samp = greedy_sampling_data(
        data,
        class_fn,
        max_samples=max_sample_size,
        internal_dtype=th.float64 if rec_type == "feat_prop" else th.float32,
    )
    if (rec_type == "LS") & (signal_type == "bl") & (noise_level == 0):
        # rec_optimal_samp = greedy_sampling_data(
        #     data,
        #     rec_fn_extra_noise,
        #     max_samples=max_sample_size,
        #     internal_dtype=th.float64 if rec_type == "feat_prop" else th.float32,
        # )

        rec_optimal_samp = ts.greedy_a_samples_only_fast(
            graph=data.edge_index,
            num_nodes=data.num_nodes,
            bandwidth=data.U_k.shape[1],
            max_samples=max_sample_size,
            normalization="sym",
        )

    else:
        rec_optimal_samp = greedy_sampling_data(
            data,
            rec_fn,
            max_samples=max_sample_size,
            internal_dtype=th.float64 if rec_type == "feat_prop" else th.float32,
        )
    all_samps = th.tensor([rand_samp.tolist(), class_optimal_samp, rec_optimal_samp])
    # breakpoint()
    tqdm.write("Reconstructing...")
    L = s.calc_laplacian(
        data.edge_index,
        data.num_nodes,
        normalization="sym",
    ).float()
    x_recs = []
    sample_sizes = np.arange(1, max_sample_size)
    for samp_size in tqdm(sample_sizes):
        samp_sets = all_samps[:, :samp_size]
        if rec_type == "feat_prop":
            x_rec = th.stack([rec_dirichlet_direct(s, L, data.x[s]) for s in samp_sets])
        elif rec_type == "LS":
            x_rec = vmap(lambda s: rec_LS_mult_signals(s, data.U_k, data.x[s]))(
                samp_sets
            )
        else:
            raise ValueError("only allowed rec_types are feat_prop and LS")
        x_recs.append(x_rec)
    x_rec = th.stack(x_recs).transpose(0, 1)

    tqdm.write("Evaluating Errs Analytically...")
    sample_types = ["Random", "Classification-optimal", "Reconstruction-optimal"]
    anal_err_df = mk_anal_df(
        class_fn, rec_fn, L, data.U_k, all_samps, sample_sizes, sample_types
    )

    # x_rec now has shape (num_sampling_types, |sample_sizes|, num_nodes, num_tasks)

    # calculate errors
    # tqdm.write("Evaluating Errs Analytically...")
    # anal_err_df = analytic_error_simple(
    #     data,
    #     signal_type,
    #     rec_type,
    #     noise_levels,
    #     sample_sizes,
    #     num_sample_sets,
    # )
    tqdm.write("Evaluating Errs Empirically...")
    err_df = eval_errs(data, x_rec, sample_sizes, sample_types)
    err_df = err_df.join(
        anal_err_df.set_index(["Sampling Type", "Sample Size"]),
        how="left",
        on=["Sampling Type", "Sample Size"],
    )
    total_df = err_df
    # total_df = pd.concat([per_graph(i) for i in trange(num_graphs)])
    #
    real_dist_path = path / "real"
    if not real_dist_path.is_dir():
        real_dist_path.mkdir(parents=True)
    signal_type = "bl" if bandlimit_signals else "fb"
    num_nodes = data.num_nodes
    fname = f"real_{dataset_name}_{signal_type}_{rec_type}_{noise_level}_noise_{num_nodes}_nodes.csv"
    plot_errors(
        total_df,
        title=f"{signal_type} signals + {rec_type} reconstruction + noise level {noise_level}",
        filename=real_dist_path / fname,
        bandlimit=data.U_k.shape[1],
    )
    total_df.to_csv(real_dist_path / fname)


def sgc_pipeline_untrained(
    num_graphs=8,
    graph_type="BA",
    signal_type="lap_pinv",
    rec_type="feat_prop",
    num_nodes=200,
    num_sigs=30,
    num_sample_sets=200,
    max_sample_size=None,
    bandlimit=None,
    raw_signal_dist="gaussian",
    noise_level=0,
):
    if max_sample_size is None:
        max_sample_size = num_nodes

    num_conv_layers = 2 if signal_type == "lap_pinv" else 1

    feat_dim = 64

    def per_graph(i):
        # make graph
        tqdm.write("Making Graph...")
        data, gcn = mk_tasks_sgc(
            feat_dim=feat_dim,
            hidden_dim=feat_dim // 2,
            graph_type=graph_type,
            num_nodes=num_nodes,
            noise_level=noise_level,
            signal_type=signal_type,
            raw_signal_dist=raw_signal_dist,
            num_conv_layers=num_conv_layers,
        )
        # Sample and reconstruct signal

        tqdm.write("Sampling...")
        class_fn = mk_anal_class_err_fn_sgc_stable(
            data, gcn, rec_type=rec_type, noise_level=noise_level
        )
        rec_fn = mk_anal_rec_err_fn(data, rec_type=rec_type, noise_level=noise_level)
        rec_fn_extra_noise = mk_anal_rec_err_fn(
            data, rec_type=rec_type, noise_level=0.01
        )
        rand_samp = th.randperm(data.num_nodes)[:max_sample_size]
        class_optimal_samp = greedy_sampling_data(
            data,
            class_fn,
            max_samples=max_sample_size,
            internal_dtype=th.float64 if rec_type == "feat_prop" else th.float32,
        )
        if (rec_type == "LS") & (signal_type == "bl") & (noise_level == 0):
            rec_optimal_samp = greedy_sampling_data(
                data,
                rec_fn_extra_noise,
                max_samples=max_sample_size,
                internal_dtype=th.float64 if rec_type == "feat_prop" else th.float32,
            )
        else:
            rec_optimal_samp = greedy_sampling_data(
                data,
                rec_fn,
                max_samples=max_sample_size,
                internal_dtype=th.float64 if rec_type == "feat_prop" else th.float32,
            )
        all_samps = th.tensor(
            [rand_samp.tolist(), class_optimal_samp, rec_optimal_samp]
        )
        # breakpoint()
        tqdm.write("Reconstructing...")
        L = s.calc_laplacian(
            data.edge_index,
            data.num_nodes,
            normalization="sym",
        ).float()
        x_recs = []
        sample_sizes = np.arange(1, max_sample_size)
        for samp_size in tqdm(sample_sizes):
            samp_sets = all_samps[:, :samp_size]
            # empirical reconstruction errs
            if rec_type == "feat_prop":
                x_rec = th.stack(
                    [rec_dirichlet_direct(s, L, data.x[s]) for s in samp_sets]
                )
            elif rec_type == "LS":
                x_rec = vmap(lambda s: rec_LS_mult_signals(s, data.U_k, data.x[s]))(
                    samp_sets
                )
            else:
                raise ValueError("only allowed rec_types are feat_prop and LS")
            x_recs.append(x_rec)
            # analytic errs

        x_rec = th.stack(x_recs).transpose(0, 1)
        # x_rec now has shape (num_sampling_types, |sample_sizes|, num_nodes, num_tasks)
        tqdm.write("Evaluating Errs Analytically...")
        sample_types = ["Random", "Classification-optimal", "Reconstruction-optimal"]
        # for some reason need to redefine this else rec_fn_extra_noise gets used
        rec_fn = mk_anal_rec_err_fn(data, rec_type=rec_type, noise_level=noise_level)
        anal_err_df = mk_anal_df(
            class_fn,
            rec_fn,
            L,
            data.U_k,
            all_samps,
            sample_sizes,
            sample_types,
            feat_dim=feat_dim,
        )

        # calculate errors
        tqdm.write("Evaluating Errs...")
        err_df = eval_errs_gcn(gcn, data, x_rec, sample_sizes, sample_types)
        err_df = err_df.join(
            anal_err_df.set_index(["Sampling Type", "Sample Size"]),
            how="left",
            on=["Sampling Type", "Sample Size"],
        )
        err_df["graph_id"] = i
        err_df["num_conv_layers"] = gcn.num_conv_layers
        return err_df

    # total_df = pd.concat([per_graph(i) for i in trange(num_graphs)])

    total_df = pd.concat(
        Parallel(n_jobs=-1, verbose=5)(delayed(per_graph)(i) for i in range(num_graphs))
    )
    pic_name = f"{graph_type}_{signal_type}_{rec_type}_{noise_level}_noise_{num_nodes}_nodes_{num_conv_layers}_layers.csv"
    sgc_path = path / "sgc"
    if not sgc_path.is_dir():
        sgc_path.mkdir(parents=True)
    total_df.to_csv(sgc_path / pic_name)
    # plot_errors(
    #     total_df,
    #     title=f"{signal_type} signals + {num_conv_layers} SGC layers + {rec_type} reconstruction + {noise_level} noise",
    #     filename=sgc_path / pic_name,
    #     bandlimit=bandlimit,
    # )


def gcn_pipeline_untrained(
    num_graphs=4,
    graph_type="BA",
    signal_type="bl",
    rec_type="feat_prop",
    num_nodes=300,
    num_sigs=30,
    num_sample_sets=200,
    max_sample_size=None,
    bandlimit=None,
    raw_signal_dist="gaussian",
):
    if max_sample_size is None:
        max_sample_size = num_nodes

    def per_graph(i):
        # make graph
        tqdm.write("Making Graph...")
        noise_levels = [0]
        data, gcn = mk_tasks_gcn(
            feat_dim=128,
            hidden_dim=64,
            graph_type=graph_type,
            num_nodes=num_nodes,
            num_tasks_per_noise_level=num_sigs,
            noise_levels=noise_levels,
            signal_type=signal_type,
            raw_signal_dist=raw_signal_dist,
        )
        # Sample and reconstruct signal
        sample_sizes = th.arange(1, max_sample_size)
        tqdm.write("Sampling & Reconstructing...")
        if rec_type == "feat_prop":
            x_rec = sample_and_reconstruct_dirichlet_vmap(
                data, sample_sizes, num_sample_sets
            )
        elif rec_type == "LS":
            x_rec = sample_and_reconstruct_LS_vmap(data, sample_sizes, num_sample_sets)
        else:
            raise ValueError("only allowed rec_types are feat_prop and LS")

        # calculate errors
        tqdm.write("Evaluating Errs...")
        err_df = eval_errs_gcn(gcn, data, x_rec, sample_sizes.numpy(), noise_levels)
        err_df["graph_id"] = i
        return err_df

    total_df = pd.concat([per_graph(i) for i in trange(num_graphs)])

    pic_name = (f"{graph_type}_{signal_type}_{rec_type}_0_noise_{num_nodes}_nodes.csv",)

    plot_errors(
        total_df,
        title=f"{signal_type} signals + {rec_type} reconstruction + no noise",
        filename=path / pic_name,
        bandlimit=bandlimit,
    )


def bandwidth_vs_laplacian(num_nodes, graph_type="BA"):
    if graph_type == "CSBM":
        data = mk_tasks_csbm(
            num_nodes=num_nodes,
            feat_dim=1,
            num_tasks_per_noise_level=None,
            noise_levels=[0],
        )
    else:
        data = mk_tasks_bl(
            graph_type=graph_type,
            num_nodes=num_nodes,
            num_tasks_per_noise_level=33,
            noise_levels=[0],
            signal_type="lap",
        )
    U = s.calc_eigenbasis(
        data.edge_index, data.num_nodes, double=False, normalization="sym"
    )

    def inner(k):
        return sample_and_reconstruct_LS_vmap_single(
            data.x, s.restrict_eigenbasis(U, k), num_nodes, 128
        )

    # x_rec = [inner(k) for k in trange(1, num_nodes + 1)]
    bandwidths = np.arange(1, num_nodes + 1)
    x_rec = Parallel(n_jobs=-1, verbose=5)(delayed(inner)(k) for k in bandwidths)
    x_rec = th.stack(x_rec)

    total_df = eval_errs(data, x_rec, bandwidths, [0])
    total_df = total_df.rename(columns={"Sample Size": "Bandwidth"})
    # return total_df
    #

    b_path = (path / "bandwidth").resolve()
    if not b_path.is_dir():
        b_path.mkdir(parents=True)
    total_df.to_csv(b_path / "blah.csv")
    plot_errors(
        total_df,
        title=f"lap signals + LS reconstruction (varying k) + no noise + full observation",
        filename=b_path / f"{graph_type}_lap_LS_0_noise_varying_bandwidth.png",
        bandlimit=None,
        indep_var="Bandwidth",
    )


def analytic_lap(L):
    N = L.shape[0]

    def rec_err(sample_set):
        m = len(sample_set)
        sample_mask_c = th.ones(N, dtype=bool)
        sample_mask_c[sample_set] = False

        A = L[sample_set][:, sample_set]
        C = L[sample_mask_c][:, sample_mask_c]
        BT = L[sample_mask_c][:, sample_set]
        Cinv = C.inverse()
        # The interesting shape comes from (Cinv @ BT @ A @ BT.T @ Cinv).trace()
        return (
            N
            - m
            + (Cinv @ BT @ A @ BT.T @ Cinv).trace()
            - (2 * (BT.T @ Cinv @ BT).trace())
        )

    def sample_rec_err(sample_size, num_reps):
        return th.stack(
            [rec_err(th.randperm(N)[:sample_size]) for _ in range(num_reps)]
        ).mean()

    return th.stack([sample_rec_err(m, 10) for m in trange(1, N)])


def analytic_lap_samp(data, sample_sets):
    L = s.calc_laplacian(
        data.edge_index,
        data.num_nodes,
        normalization="sym",
    )
    N = L.shape[0]

    def rec_err(sample_set):
        m = len(sample_set)
        sample_mask_c = th.ones(N, dtype=bool)
        sample_mask_c[sample_set] = False

        A = L[sample_set][:, sample_set]
        C = L[sample_mask_c][:, sample_mask_c]
        BT = L[sample_mask_c][:, sample_set]
        Cinv = C.inverse()
        # The interesting shape comes from (Cinv @ BT @ A @ BT.T @ Cinv).trace()
        return (
            N
            - m
            + (Cinv @ BT @ A @ BT.T @ Cinv).trace()
            - (2 * (BT.T @ Cinv @ BT).trace())
        )

    return th.stack([rec_err(s) for s in sample_sets])


def analytic_lap_pinv(L):
    Lpinv = s.connected_laplacian_pinv(L, normalization="sym")
    v = th.linalg.eigh(L).eigenvectors[:, 0]
    N = L.shape[0]

    def rec_err(sample_set):
        m = len(sample_set)
        sample_mask_c = th.ones(N, dtype=bool)
        sample_mask_c[sample_set] = False
        sample_set_c = np.where(sample_mask_c)[0]

        X = Lpinv[sample_set][:, sample_set]
        Z = Lpinv[sample_mask_c][:, sample_mask_c]
        C = L[sample_mask_c][:, sample_mask_c]
        BT = L[sample_mask_c][:, sample_set]
        YT = Lpinv[sample_mask_c][:, sample_set]
        Cinv = C.inverse()
        # The interesting shape comes from (Cinv @ BT @ A @ BT.T @ Cinv).trace()
        # return (
        #     4 * Z.trace()
        #     - 3 * Cinv.trace()
        #     + th.dot(v[sample_set_c], Cinv @ v[sample_set_c])
        # )
        # return (Z - 2 * YT @ BT.T @ Cinv + Cinv @ BT @ X @ BT.T @ Cinv).trace()
        true_err = (
            4 * Z.trace()
            # - 3 * Cinv.trace()
            - 3 * Cinv.trace()
            + 4 * v[sample_set_c].dot(Cinv @ v[sample_set_c])
        )
        lower_bound = (
            4 * Z.trace()
            # - 3 * (N - m)
            - 3 * Cinv.trace()
            # + 4 * v[sample_set_c].dot(Cinv @ v[sample_set_c])
            + 4 * N / m
            # + 4
            # * v[sample_set_c].square().sum()
            # / v[sample_set_c].dot(C @ v[sample_set_c])
            # + 4 * v[sample_set_c].dot(Cinv @ v[sample_set_c])
        )
        upper_bound = (
            4 * Z.trace()
            # - 3 * (N - m)
            # - 3 * Cinv.trace()
            # + 4 * v[sample_set_c].dot(Cinv @ v[sample_set_c])
            # + 4 * N / m
            # + 4
            # * v[sample_set_c].square().sum()
            # / v[sample_set_c].dot(C @ v[sample_set_c])
            + v[sample_set_c].dot(Cinv @ v[sample_set_c])
            # - 3 * ((N - m) - 1)
        )
        return th.tensor((true_err, lower_bound, upper_bound))

    def sample_rec_err(sample_size, num_reps):
        return th.vstack(
            [rec_err(th.randperm(N)[:sample_size]) for _ in range(num_reps)]
        ).mean(dim=0)

    res = th.stack([sample_rec_err(m, 10) for m in trange(1, N)])
    df = pd.DataFrame(
        {
            "Sample Size": np.arange(1, N),
            "exp": res[:, 0].numpy(),
            "lower_bound": res[:, 1].numpy(),
            "upper_bound": res[:, 2].numpy(),
        }
    )
    return df.melt(id_vars="Sample Size")


def main_fn():
    num_nodes = 500
    # for gtype in ["ER", "SBM", "BA"]:
    # for gtype in ["BA", "SBM"]:
    for gtype in ["BA"]:
        # , "SBM", "ER"]:
        # for gtype in ["CSBM"]:
        # bandwidth_vs_laplacian(num_nodes, graph_type=gtype)
        # stypes = ["default"] if gtype == "CSBM" else ["lap_pinv", "bl", "lap"]
        # stypes = ["bl", "lap_pinv", "lap"]
        # stypes = ["bl", "lap_pinv"]
        stypes = ["lap_pinv"]
        # stypes = ["bl", "lap_pinv"]
        # stypes = ["bl"]
        for stype in stypes:
            for rtype in ["LS", "feat_prop"]:
                # for rtype in ["LS"]:
                # rtypes = ["feat_prop"] if gtype == "BA" else ["LS"]
                # for rtype in rtypes:
                # for rtype in ["LS", "feat_prop"]:
                # for rtype in ["LS"]:
                noise_levels = [0, (10 ** (-1.5))] if "LS" else [0]
                # noise_levels = [10 ** (-1.5)]
                for noise_level in noise_levels:
                    console.rule(
                        f" {gtype}, {stype}, {rtype}, Noise level {noise_level} ",
                        characters="=",
                    )
                    # print(
                    #     f"================== {gtype}, {stype}, {rtype}, Noise level {noise_level} ================== "
                    # )
                    bandlimit = (
                        None
                        if (stype, rtype) == ("lap", "feat_prop")
                        else num_nodes // 10
                    )
                    if stype == "bl" and rtype == "LS":
                        if noise_level == 0:
                            max_sample_size = num_nodes // 8
                        else:
                            max_sample_size = num_nodes // 4
                    else:
                        max_sample_size = num_nodes

                    # Need 32 examples for errors to average properly
                    # console.rule("SGC", characters="+")
                    console.log("SGC")
                    try:
                        sgc_pipeline_untrained(
                            num_graphs=32,
                            signal_type=stype,
                            rec_type=rtype,
                            graph_type=gtype,
                            num_nodes=num_nodes,
                            bandlimit=bandlimit,
                            max_sample_size=max_sample_size,
                            raw_signal_dist="gaussian",
                            noise_level=noise_level,
                        )
                    except:
                        pass

                    # console.log("Simple")
                    # # console.rule("Simple", characters="~")
                    # try:
                    #     simple_pipeline(
                    #         num_graphs=32,
                    #         signal_type=stype,
                    #         rec_type=rtype,
                    #         graph_type=gtype,
                    #         num_nodes=num_nodes,
                    #         bandlimit=bandlimit,
                    #         max_sample_size=max_sample_size,
                    #         noise_level=noise_level,
                    #         # max_sample_size=18,
                    #         num_sigs=200,
                    #     )
                    # except:
                    #     pass


if __name__ == "__main__":
    main_fn()
