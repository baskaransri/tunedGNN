#!/usr/bin/env python3
#import incremental as Incr
from networkx.convert import to_networkx_graph
from networkx.generators.small import bull_graph
import torch as th
from torch import linalg, vmap
from torch.linalg import eigh, eigvals, lstsq, matrix_rank
from torch.nn.functional import normalize, relu, one_hot
import torch.sparse

import datasets

import torch_geometric as pyg
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import (
    get_laplacian,
    to_dense_adj,
    erdos_renyi_graph,
    stochastic_blockmodel_graph,
    barabasi_albert_graph,
    to_undirected,
    remove_isolated_nodes,
    remove_self_loops,
    to_networkx,
    degree,
    from_networkx,
)
from torch_geometric.nn.conv.message_passing import MessagePassing
from math import floor, ceil, sqrt, log, prod
import math
import time
from pprint import pprint
from tqdm import tqdm
import tqdm.contrib.itertools as tqdm_iter
import tqdm.auto as tqdma

from pathlib import Path

import matplotlib.pyplot as plt

# from matplotlib.animation import funcanimation
import seaborn as sns

import numpy as np
import pandas as pd
import scipy.sparse.linalg
import scipy.optimize
from scipy.stats import spearmanr

from numba import njit

import joblib
from joblib import Parallel, delayed, parallel_config
from joblib.externals.loky import set_loky_pickler

from einops import rearrange
import einops

# from opt_einsum import contract

from functools import lru_cache, reduce
import itertools
import operator

# 20240925: everything is broken, get working in environment later if we need curvature stuff
# from GraphRicciCurvature.OllivierRicci import OllivierRicci
import matplotlib.pyplot as plt
import networkx as nx
import pydot
from networkx.drawing.nx_pydot import graphviz_layout

from hypothesis import (
    given,
    assume,
    settings,
    Verbosity,
    strategies as hstrat,
)
import hypothesis.extra.numpy as hen
from datetime import timedelta

from typing import Optional, Tuple, Callable, Dict
import torchtyping
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked


torchtyping.patch_typeguard()

transfer_output_path = Path("transfer_output").resolve()


@hstrat.composite
def gen_undirected_graph(draw, max_nodes=1000):
    num_nodes = draw(hstrat.integers(2, max_nodes))
    num_edges = draw(hstrat.integers(1, num_nodes * (num_nodes - 1)))
    # graph_edges = draw(
    #     hen.arrays(int, (2, num_edges), elements=hstrat.integers(0, num_nodes - 1))
    # )
    edge_pairs = hstrat.tuples(
        hstrat.integers(1, num_nodes), hstrat.integers(1, num_nodes)
    )
    graph_edge_list = draw(
        hstrat.lists(edge_pairs, min_size=1, max_size=num_edges, unique=True)
    )
    graph_edges = th.tensor(graph_edge_list).T
    return to_undirected(graph_edges), num_nodes


@hstrat.composite
def gen_undirected_graph_and_sample(draw, max_nodes=10):
    num_nodes = draw(hstrat.integers(2, max_nodes))
    bandwidth = draw(hstrat.integers(1, num_nodes))
    num_edges = draw(hstrat.integers(1, num_nodes * (num_nodes - 1)))
    # graph_edges = draw(
    #     hen.arrays(int, (2, num_edges), elements=hstrat.integers(0, num_nodes - 1))
    # )
    edge_pairs = hstrat.tuples(
        hstrat.integers(1, num_nodes), hstrat.integers(1, num_nodes)
    )
    graph_edge_list = draw(
        hstrat.lists(edge_pairs, min_size=1, max_size=num_edges, unique=True)
    )
    graph_edges = th.tensor(graph_edge_list).T
    graph_edges = remove_isolated_nodes(graph_edges)[0]
    assume(graph_edges.numel() > 0)
    actual_num_nodes = int((graph_edges.max() + 1).item())
    bandwidth = draw(hstrat.integers(1, actual_num_nodes))
    # draw a list of indices to sample at
    unsorted_omega = draw(
        hstrat.lists(
            hstrat.integers(0, actual_num_nodes - 1),
            min_size=1,
            max_size=num_nodes,
            unique=True,
        )
    )
    omega = th.tensor(sorted(unsorted_omega))
    return to_undirected(graph_edges), actual_num_nodes, bandwidth, omega


##Utils


# reads a file whose rows look like
def read_dot_edges(dot_edges_file):
    row, col, weight, graph_num = np.genfromtxt(
        dot_edges_file, dtype=["int", "int", "float", "int"], unpack=True
    )
    graphs = []
    for n in np.unique(graph_num):
        mask = graph_num == n
        th_weight = th.from_numpy(weight[mask])
        edges = [row[mask], col[mask]]
        th_edges = th.from_numpy(np.vstack(edges))
        graphs.append(Data(edge_index=th_edges, edge_attr=th_weight))
    return graphs


# aves_graphs = read_dot_edges("data/aves-wildbird-network.edges")
# aves = aves_graphs[0]


# th.diag for 1d vectors, but returns a sparse tensor
def sparse_diag(vec, size=None):
    n = vec.shape[0]
    # diag_sparse = th.sparse_coo_tensor(
    #     th.arange(n).repeat(2, 1), vec, (n, n), dtype=th.float, device=th.device("cpu")
    # )
    diag_sparse = th.sparse_csr_tensor(
        crow_indices=th.arange(n + 1, dtype=th.int32),
        col_indices=th.arange(n, dtype=th.int32),
        values=vec,
        size=(n, n),
    )
    return diag_sparse


def fast_approx_rank(
    A,  #: TensorType["b", "m", "n"],
    max_rank: int,
) -> float:
    # idea 1: bound eigvals below, use scipy eigsh on AA^T.
    # idea 2:
    #  a) bound eigvals below and above
    #  b) scale AA^T so evals in [v,1],
    #  c) construct poly that takes values below v to 0 and above v to 1
    #  d) use frob norm
    # cov = A @ A.T
    # idea 3: dumb poly filter - in practice, this is useless.
    # idea 4: make A square (X = A @ A.T or A.T @ A) and invert X + eps * I, take the trace
    # for a tree graph with 1000 nodes, you have the smallest evalue being o(10^-5), so this isn't the greatest
    _, m, n = A.shape
    if m > n:
        X = th.bmm(th.transpose(A, 2, 1), A)
    else:
        X = th.bmm(A, th.transpose(A, 2, 1))
    eps = 1e-6
    traces = vmap(th.trace)(th.inverse(X + eps * th.eye(X.shape[-1])))
    return X.shape[-1] - (traces * eps).round().int()


## Graph constructors
@typechecked
def clean_graph(graph: TensorType[2, "m"]) -> Tuple[TensorType[2, "n"], int]:
    graph, _, mask = remove_isolated_nodes(graph)
    num_nodes = int(mask.sum().item())
    return graph, num_nodes


def sbm_constructor(block_sizes, p, q):
    n = len(block_sizes)
    graph = stochastic_blockmodel_graph(
        block_sizes, th.tensor(p).repeat(n, n).fill_diagonal_(q)
    )
    return clean_graph(graph)


def binary_tree(num_layers):
    num_nodes = (2**num_layers) - 1
    directed_tree = th.stack(
        (
            th.arange(num_nodes).repeat_interleave(2)[: num_nodes - 1],
            th.arange(1, num_nodes),
        )
    )
    return to_undirected(directed_tree)


def grid(n):
    assert n > 0
    grid_points = th.arange(n * n).reshape(n, n)

    def edges_downwards(points):
        return rearrange([points[:-1], points[1:]], "two a b -> two (a b)", two=2)
        # zipped_points = th.stack([points[:-1], points[1:]])
        # return zipped_points.T.flatten(end_dim=1).T

    edges_down = edges_downwards(grid_points)
    edges_side = edges_downwards(grid_points.T)
    all_edges = th.cat([edges_down, edges_side], axis=1)
    return to_undirected(all_edges)


def circle(n):
    nodes = th.arange(n)
    next_nodes = nodes.roll(1, 0)
    edges_one_way = th.vstack([nodes, next_nodes])
    return to_undirected(edges_one_way)


def path(n):
    return to_undirected(th.vstack([th.arange(n - 1), 1 + th.arange(n - 1)]))


def random_k_regular_graph(num_nodes, k):
    G = nx.random_regular_graph(k, num_nodes)
    return to_undirected(from_networkx(G).edge_index)


def connected_erdos_renyi_graph(num_nodes, p):
    g, n = clean_graph(erdos_renyi_graph(num_nodes, p))
    while not pyg_is_connected(g, n):
        g, n = clean_graph(erdos_renyi_graph(num_nodes, p))
    return (g, n)


# attaches the last named node of graph1 to node 0 of graph 2
def add_bridge(graph1, graph2, num_nodes_graph1=None):
    last_named_node_graph1 = graph1.max()
    if num_nodes_graph1 is None:
        # TODO: should the plus live here
        num_nodes_graph1 = last_named_node_graph1
    new_graph2 = graph2 + num_nodes_graph1 + 1
    first_named_node_graph2 = new_graph2.min()
    bridge = th.tensor(
        [
            [last_named_node_graph1, first_named_node_graph2],
            [first_named_node_graph2, last_named_node_graph1],
        ]
    )
    total_graph = th.cat((graph1, bridge, new_graph2), dim=1)
    return total_graph


def bridge_constructors(graph_constructor_1, graph_constructor_2):
    g1, n1 = graph_constructor_1()
    g2, n2 = graph_constructor_2()
    graph = add_bridge(g1, g2, n1)
    return (graph, n1 + n2 + 1)


## Signal construction
def noise_signal(num_nodes, num_feats):
    return th.randn(num_nodes, num_feats)


# Generate bandlimited signal with noise
# If normalise=False, SNR will be in expectation
# If normalise=True, it will normalise each signal and noise so SNR is exact
# keeping the norm of the clean signal at 1
def bandlimited_signals_with_noise(
    U_k: th.Tensor,
    SNR: float,
    num_signals: Optional[int] = None,
    normalise: bool = True,
    bandlimit_noise: bool = False,
):
    num_nodes, bandwidth = U_k.shape
    if num_signals is None:
        raw_bandlimited_signals = U_k @ th.randn(bandwidth)
        if bandlimit_noise:
            raw_noise_signals = U_k @ th.randn(bandwidth)
        else:
            raw_noise_signals = th.randn(num_nodes)
    else:
        raw_bandlimited_signals = U_k @ noise_signal(bandwidth, num_signals)
        if bandlimit_noise:
            raw_noise_signals = U_k @ noise_signal(bandwidth, num_signals)
        else:
            raw_noise_signals = noise_signal(num_nodes, num_signals)
    if normalise:
        SNR = th.tensor(SNR)
        # This version will make sure the total signal has norm 1:
        # bandlimited_signals = th.sqrt(SNR / (SNR + 1)) * normalize(
        #     raw_bandlimited_signals, dim=0
        # )
        # noise_signals = th.rsqrt(SNR + 1) * normalize(raw_noise_signals, dim=0)

        # This version keeps the clean signal at norm 1, to make MSEs comparable
        # at different SNRs
        bandlimited_signals = normalize(raw_bandlimited_signals, dim=0)
        noise_signals = th.rsqrt(SNR) * normalize(raw_noise_signals, dim=0)
        corrupted_signals = bandlimited_signals + noise_signals
        return {
            "clean_signals": bandlimited_signals,
            "corrupted_signals": corrupted_signals,
        }
    else:
        # else, we operate in expectation:
        if bandlimit_noise:
            noise_err = th.sqrt(th.tensor(1 / SNR))
            noise = noise_err * U_k @ noise_signal(bandwidth, num_signals)
        else:
            noise_err = th.sqrt(th.tensor(bandwidth / (num_nodes * SNR)))
            noise = noise_err * noise_signal(num_nodes, num_signals)
        return {
            "clean_signals": raw_bandlimited_signals,
            "corrupted_signals": raw_bandlimited_signals + noise,
        }


def bandlimited_signals_with_noise_tuple(
    U_k: th.Tensor,
    SNR: float,
    num_signals: Optional[int] = None,
    normalise: bool = True,
    bandlimit_noise: bool = False,
):
    num_nodes, bandwidth = U_k.shape
    if num_signals is None:
        raw_bandlimited_signals = U_k @ th.randn(bandwidth)
        if bandlimit_noise:
            raw_noise_signals = U_k @ th.randn(bandwidth)
        else:
            raw_noise_signals = th.randn(num_nodes)
    else:
        raw_bandlimited_signals = U_k @ noise_signal(bandwidth, num_signals)
        if bandlimit_noise:
            raw_noise_signals = U_k @ noise_signal(bandwidth, num_signals)
        else:
            raw_noise_signals = noise_signal(num_nodes, num_signals)
    if normalise:
        SNR = th.tensor(SNR)
        # This version will make sure the total signal has norm 1:
        # bandlimited_signals = th.sqrt(SNR / (SNR + 1)) * normalize(
        #     raw_bandlimited_signals, dim=0
        # )
        # noise_signals = th.rsqrt(SNR + 1) * normalize(raw_noise_signals, dim=0)

        # This version keeps the clean signal at norm 1, to make MSEs comparable
        # at different SNRs
        bandlimited_signals = normalize(raw_bandlimited_signals, dim=0)
        noise_signals = th.rsqrt(SNR) * normalize(raw_noise_signals, dim=0)
        corrupted_signals = bandlimited_signals + noise_signals
        clean_signals = bandlimited_signals
        corrupted_signals = corrupted_signals

    else:
        # else, we operate in expectation:
        if bandlimit_noise:
            noise_err = th.sqrt(th.tensor(1 / SNR))
            noise = noise_err * U_k @ noise_signal(bandwidth, num_signals)
        else:
            noise_err = th.sqrt(th.tensor(bandwidth / (num_nodes * SNR)))
            noise = noise_err * noise_signal(num_nodes, num_signals)
        clean_signals = raw_bandlimited_signals
        corrupted_signals = raw_bandlimited_signals + noise

    return (clean_signals, corrupted_signals)


# Generate bandlimited signal with noise
# If normalise=False, SNR will be in expectation
# If normalise=True, it will normalise each signal and noise so SNR is exact
# keeping the norm of the clean signal at 1
def bandlimited_signals_with_noise_real(
    U_k: th.Tensor,
    SNR: float,
    provided_signals: th.Tensor,
    num_signals: Optional[int] = None,
    normalise: bool = True,
    bandlimit_noise: bool = False,
):
    num_nodes, bandwidth = U_k.shape
    # We normalize the provided signals:

    if num_signals is None:
        num_signals = provided_signals.shape[1]
    num_signal_multiplier = num_signals // provided_signals.shape[1]

    raw_signals = provided_signals / provided_signals.norm(dim=0)
    raw_signals = raw_signals.repeat(1, num_signal_multiplier)
    num_signals = raw_signals.shape[1]

    # raw_bandlimited_signals = U_k @ noise_signal(bandwidth, num_signals)
    if bandlimit_noise:
        raw_noise_signals = U_k @ noise_signal(bandwidth, num_signals)
    else:
        raw_noise_signals = noise_signal(num_nodes, num_signals)
    if normalise:
        raise NotImplementedError
    else:
        # else, we operate in expectation:
        if bandlimit_noise:
            noise_err = th.sqrt(th.tensor(1 / SNR))
            noise = noise_err * U_k @ noise_signal(bandwidth, num_signals)
        else:
            noise_err = th.sqrt(th.tensor(bandwidth / (num_nodes * SNR)))
            noise = noise_err * noise_signal(num_nodes, num_signals)
        return {
            "clean_signals": raw_signals,
            "corrupted_signals": raw_signals + noise,
        }


## Calculate the laplacian eigenbasis
# Take n=bandwidth columns
def restrict_eigenbasis(eigenbasis, bandwidth):
    return eigenbasis[:, :bandwidth]


def zero_restrict_eigenbasis(eigenbasis, bandwidth):
    return eigenbasis * (th.arange(eigenbasis.shape[1]) < bandwidth)


def calc_laplacian(graph_edge_index, num_nodes=None, normalization="sym"):
    laplacian_edges, laplacian_edge_weights = get_laplacian(
        graph_edge_index, normalization=normalization, num_nodes=num_nodes  # "sym",
    )
    laplacian = (
        to_dense_adj(
            laplacian_edges, edge_attr=laplacian_edge_weights, max_num_nodes=num_nodes
        )
        .squeeze()
        .double()
    )
    return laplacian


def calc_eigenbasis(
    graph_edge_index,
    num_nodes=None,
    normalization="sym",
    eps=None,
    double=False,
):
    laplacian = calc_laplacian(
        graph_edge_index, num_nodes=num_nodes, normalization=normalization
    )
    # construct U_k, matrix of k eigenvectors
    # corresponding to smallest eigenvalues. eigh seems to be faster than lobpcg.
    _, U = eigh(laplacian)
    if eps is not None:
        U[U.abs() < eps] = 0
    if not double:
        U = U.float()
    return U


def optimal_sampling_distn(
    proj: TensorType["num_nodes", "num_nodes"]
) -> TensorType["num_nodes"]:
    diag = th.diag(proj)
    k = th.sum(diag)
    return diag / k


def uniform_sampling_distn(U_k):
    n = U_k.shape[0]
    return th.ones(n) / n


# m is number of nodes to be sampled
def sample_nodes_with_distn(p_star, m):
    # coalescing the sparse tensors doesn't seem to be a win
    omega = th.multinomial(p_star, num_samples=m, replacement=False)
    P_Omega_inv_sqrt = sparse_diag(th.rsqrt(p_star[omega]))
    # M_sparse = th.sparse_coo_tensor(
    #     th.stack((th.arange(m), omega)),
    #     th.ones(m),  # values are ones, otherwise 0
    #     (m, p_star.shape[0]),  # shape
    #     dtype=th.float,
    #     device=th.device("cpu"),
    # )
    M = th.sparse_csr_tensor(
        crow_indices=th.arange(m + 1, dtype=th.int32),  # we have m non-zero elements
        col_indices=omega.int(),
        values=th.ones(m),
        size=(m, p_star.shape[0]),
    )
    return {
        "omega": omega,
        "M": M,
        "P_Omega_inv_sqrt": P_Omega_inv_sqrt,
    }


def batch_M_uniform(U_k, sample_size, num_repeats):
    p = uniform_sampling_distn(U_k)
    return batch_sample_nodes_with_distn(p, sample_size, num_repeats)["M"]


def deficient_nodes(U_k, sample_size, num_repeats=1000):
    Ms = batch_M_uniform(U_k, sample_size, num_repeats)
    MU_ks = th.bmm(Ms, U_k.unsqueeze(0).expand(num_repeats, -1, -1))
    ranks = th.linalg.matrix_rank(MU_ks)
    max_rank = ranks.max()
    deficient_indices = th.stack(
        [m.coalesce().indices()[1] for m, rk in zip(Ms, ranks) if rk < max_rank]
    )
    return th.stack(th.unique(deficient_indices, return_counts=True)).T


# m is number of nodes to be sampled
def batch_sample_nodes_with_distn(
    p_star, sample_size: int, num_repeats: int, num_repeats_batch: Optional[int] = None
):
    # if m < U_k.shape[1]:
    #    print("m < k; you'll need more measurements than that!")
    # sampled nodes
    m = sample_size
    num_nodes = p_star.shape[0]
    total_repeats = num_repeats
    size = th.Size((num_repeats, m, num_nodes))
    indices = th.cartesian_prod(th.arange(num_repeats), th.arange(m)).T
    if num_repeats_batch is not None:
        total_repeats = num_repeats_batch * num_repeats
        size = th.Size((num_repeats_batch, num_repeats, m, num_nodes))
        indices = th.cartesian_prod(
            th.arange(num_repeats_batch), th.arange(num_repeats), th.arange(m)
        ).T
    # omega is num_repeats x sample_size
    omega = th.multinomial(
        p_star.unsqueeze(0).expand(total_repeats, -1),
        num_samples=m,
        replacement=False,
    )
    # dividing by k first in optimal_sampling_distn and then inverting kinda sucks
    # P_Omega_inv_sqrt = th.diag(th.rsqrt(p_star[omega]))
    M = th.sparse_coo_tensor(
        indices=th.vstack(
            (
                indices,
                omega.flatten(),
            )
        ),
        values=th.ones(total_repeats * m),  # values are ones, otherwise 0
        size=size,
        dtype=th.float32,
        device=th.device("cpu"),
    )
    # P_Omega_inv_sqrt = th.diag_embed(th.rsqrt(p_star[omega]))
    # P_Omega_inv_sqrt = sparse_diag_embed(th.rsqrt(p_star[omega]))
    return {
        "omega": omega,
        "M": M,
        # "P_Omega_inv_sqrt": P_Omega_inv_sqrt,
    }


# the inputs we usually give are VERY ill conditioned, use gelsd
@typechecked
def standard_decoder(
    P_Omega_inv_sqrt: TensorType["sample_size", "sample_size"],
    M: TensorType["sample_size", "num_nodes"],
    U_k: TensorType["num_nodes", "bandwidth"],
    y: TensorType["sample_size"],
) -> TensorType["num_nodes"]:
    # minimize || P^{-1/2}_\Omega ( Mz-y ) ||_2 in z \in Span(U_k)
    # z = U_k x
    # bracketing (M@U_k) is important for perf reasons.
    # Multiplying by sparse M is faster than indexing by omega !!
    A = P_Omega_inv_sqrt @ (M @ U_k)  # TensorType["sample_size","bandwidth"]
    b = P_Omega_inv_sqrt @ y  # TensorType["sample_size"]
    output = lstsq(A, b, driver="gelsd")
    x = output.solution
    z = U_k @ x
    return z


def standard_decoder_first_half(P_Omega_inv_sqrt, M, U_k, y):
    # minimize || P^{-1/2}_\Omega ( Mz-y ) ||_2 in z \in Span(U_k)
    # z = U_k x
    # bracketing (M@U_k) is important for perf reasons.
    # Multiplying by sparse M is faster than indexing by omega !!
    A = P_Omega_inv_sqrt @ (M @ U_k)  # TensorType["sample_size","bandwidth"]
    b = P_Omega_inv_sqrt @ y  # TensorType["sample_size"]
    return (A, b)


def standard_decoder_second_half(A, b, U_k):
    output = lstsq(A, b)
    x = output.solution
    z = U_k @ x
    return z


@typechecked
def standard_decoder_multiple_signals(
    P_Omega_inv_sqrt: TensorType["sample_size", "sample_size"],
    M: TensorType["sample_size", "num_nodes"],
    U_k: TensorType["num_nodes", "bandwidth"],
    ys: TensorType["sample_size", "batch"],
) -> TensorType["num_nodes", "batch"]:
    # Intermediate dimensions:
    # A = sample_size x bandwidth
    # As = batch x sample_size x bandwidth
    # bs = batch x sample_size
    batch_size = ys.shape[-1]
    A = P_Omega_inv_sqrt @ (M @ U_k)  #
    As = A.unsqueeze(0).expand(batch_size, -1, -1)
    bs = (P_Omega_inv_sqrt @ ys).T.contiguous()
    # do the calcs as per usual
    output = lstsq(As, bs, driver="gelsd")
    xs = output.solution  # batch x bandwidth
    zs = U_k @ xs.T  # nodes x batch
    return zs


@typechecked
def standard_decoder_multiple_signals_no_M(
    P_Omega_inv_sqrt: TensorType["sample_size", "sample_size"],
    sample_set: TensorType["sample_size"],
    U_k: TensorType["num_nodes", "bandwidth"],
    ys: TensorType["sample_size", "batch"],
) -> TensorType["num_nodes", "batch"]:
    # Intermediate dimensions:
    # A = sample_size x bandwidth
    # As = batch x sample_size x bandwidth
    # bs = batch x sample_size
    batch_size = ys.shape[-1]
    A = P_Omega_inv_sqrt @ (U_k[sample_set, :])  #
    As = A.unsqueeze(0).expand(batch_size, -1, -1)
    bs = (P_Omega_inv_sqrt @ ys).T.contiguous()
    # do the calcs as per usual
    output = lstsq(As, bs, driver="gelsd")
    xs = output.solution  # batch x bandwidth
    zs = U_k @ xs.T  # nodes x batch
    return zs


# turn off typechecking because it behaves badly with joblib
def graph_laplacian_decoder_multiple_signals(
    laplacian: TensorType["num_nodes", "num_nodes"],
    omega: TensorType["sample_size"],
    mu: float,
    ys: TensorType["sample_size", "batch_size"],
) -> TensorType["num_nodes", "batch_size"]:
    # some sizes
    num_nodes = laplacian.shape[0]
    batch_size = ys.shape[1]
    # construct A in Ax = b
    rec = (laplacian * mu).clone()  # mu * L
    rec[omega, omega] += 1.0  # (M^T M + mu * L)
    As = rec.unsqueeze(0).expand(batch_size, -1, -1)
    # Calculate b = M^T y
    ys_big = th.zeros(num_nodes, batch_size)
    ys_big[omega] = ys
    # solve Ax = b, remembering to use high precision
    # as we care about badly conditioned matrices
    # output = lstsq(As, ys_big.T, driver="gelsd")
    output = lstsq(As, ys_big.T, driver="gels")
    return output.solution.T


def graph_laplacian_decoder_multiple_signals_no_check(
    laplacian,
    omega,
    mu,
    ys,
):
    # some sizes
    num_nodes = laplacian.shape[0]
    batch_size = ys.shape[1]
    # construct A in Ax = b
    rec = (laplacian * mu).clone()  # mu * L
    rec[omega, omega] += 1.0  # (M^T M + mu * L)
    As = rec.unsqueeze(0).expand(batch_size, -1, -1)
    # Calculate b = M^T y
    ys_big = th.zeros(num_nodes, batch_size)
    ys_big[omega] = ys
    # solve Ax = b, remembering to use high precision
    # as we care about badly conditioned matrices
    # output = lstsq(As, ys_big.T, driver="gelsd")
    output = lstsq(As, ys_big.T, driver="gels")
    return output.solution.T


# doesn't seem to always converge, hmm.
@typechecked
def iterative_decoder(
    indices: TensorType["sample_size"],
    proj: TensorType["num_nodes", "num_nodes"],
    ys: TensorType["sample_size", "batch"],
    eps: float = 1e-7,
):
    f_prev = th.zeros((proj.shape[0], ys.shape[1]))
    f_next = f_prev.clone()
    # print(f"indices: {indices}, shpes (ind,proj,ys): {indices.shape}, {proj.shape}, {ys.shape}, {f_next.shape}")
    f_next[indices] = ys
    while th.norm(f_next - f_prev) > eps:
        f_prev = f_next.clone()
        f_next = proj @ f_next
        f_next[indices] = ys
    return f_next


@lru_cache(maxsize=2)
def calc_proj(U, k):
    U_k = restrict_eigenbasis(U, k)
    return U_k @ U_k.T


## U :: num_nodes x num_nodes
## ks :: batch_bandwidth
## returns batch_bandwidth x num_nodes x num_nodes
def batch_calc_proj(
    U: TensorType["num_nodes", "num_nodes"],
    ks: TensorType["batch_bandwidth"],
) -> TensorType["batch_bandwidth", "num_nodes", "num_nodes"]:
    num_nodes = U.shape[0]
    masks = th.ones(num_nodes + 1, num_nodes).tril(diagonal=-1)
    return th.einsum("ij,bj,kj->bik", U, masks[ks], U)


def construct_sample_matrix(sample_set, num_nodes):
    sample_set = th.tensor(sample_set).int()
    sample_size = sample_set.shape[0]
    return th.sparse_csr_tensor(
        crow_indices=th.arange(
            sample_size + 1, dtype=th.int32
        ),  # we have m non-zero elements
        col_indices=sample_set,
        values=th.ones(sample_size),
        size=(sample_size, num_nodes),
    )


# returns average squared error
def sampled_reconstruction_error_unregularised_noisy(
    U_k,
    sample_set,
    SNR=1.0,
    num_signals=1000,
    normalise=True,
    bandlimit_noise=False,
):
    num_nodes, bandwidth = U_k.shape
    signal_dict = bandlimited_signals_with_noise(
        U_k,
        SNR=SNR,
        num_signals=num_signals,
        normalise=normalise,
        bandlimit_noise=bandlimit_noise,
    )
    M = construct_sample_matrix(sample_set, num_nodes)
    sample_signals = signal_dict["corrupted_signals"][sample_set]
    pois = th.eye(len(sample_set))
    reconstructed_signals = standard_decoder_multiple_signals(
        pois, M, U_k, sample_signals
    )
    err = reconstructed_signals - signal_dict["clean_signals"]
    return err.square().sum(dim=0).mean()


def sampled_reconstruction_error_unregularised_noisy_real(
    U_k,
    sample_set,
    provided_signals,
    SNR=1.0,
    num_signals=1000,
    normalise=True,
    bandlimit_noise=False,
):
    num_nodes, bandwidth = U_k.shape
    signal_dict = bandlimited_signals_with_noise_real(
        U_k,
        SNR=SNR,
        provided_signals=provided_signals,
        num_signals=num_signals,
        normalise=normalise,
        bandlimit_noise=bandlimit_noise,
    )
    M = construct_sample_matrix(sample_set, num_nodes)
    sample_signals = signal_dict["corrupted_signals"][sample_set]
    pois = th.eye(len(sample_set)).to(U_k.dtype)
    reconstructed_signals = standard_decoder_multiple_signals(
        pois, M, U_k, sample_signals
    )
    err = reconstructed_signals - signal_dict["clean_signals"]
    return err.square().sum(dim=0).mean()


def sampled_reconstruction_error_glr_noisy(
    L,
    U_k,
    sample_set,
    SNR=1.0,
    num_signals=1000,
    normalise=True,
    mu=0.01,
    bandlimited_noise=False,
):
    num_nodes, bandwidth = U_k.shape
    # signal_dict = bandlimited_signals_with_noise(
    #     U_k,
    #     SNR=SNR,
    #     num_signals=num_signals,
    #     normalise=normalise,
    #     bandlimit_noise=bandlimited_noise,
    # )
    clean_signals, corrupted_signals = bandlimited_signals_with_noise_tuple(
        U_k,
        SNR=SNR,
        num_signals=num_signals,
        normalise=normalise,
        bandlimit_noise=bandlimited_noise,
    )
    # M = construct_sample_matrix(sample_set, num_nodes)
    sample_signals = corrupted_signals[sample_set]
    # pois = th.eye(len(sample_set))
    # no typechecking at all, so can be parallelised
    reconstructed_signals = graph_laplacian_decoder_multiple_signals_no_check(
        L, sample_set, mu=mu, ys=sample_signals
    )
    err = reconstructed_signals - clean_signals
    return err.square().sum(dim=0).mean()


def sampled_reconstruction_error_glr_noisy_real(
    L,
    U_k,
    sample_set,
    provided_signals,
    SNR=1.0,
    num_signals=1000,
    normalise=True,
    mu=0.01,
    bandlimited_noise=False,
):
    num_nodes, bandwidth = U_k.shape
    signal_dict = bandlimited_signals_with_noise_real(
        U_k,
        SNR=SNR,
        provided_signals=provided_signals,
        num_signals=num_signals,
        normalise=normalise,
        bandlimit_noise=bandlimited_noise,
    )
    # M = construct_sample_matrix(sample_set, num_nodes)
    sample_signals = signal_dict["corrupted_signals"][sample_set]
    # pois = th.eye(len(sample_set))
    reconstructed_signals = graph_laplacian_decoder_multiple_signals(
        L, sample_set, mu=mu, ys=sample_signals
    )
    err = reconstructed_signals - signal_dict["clean_signals"]
    return err.square().sum(dim=0).mean()


def reconstruction_error_with_eigenbasis(eigenbasis, signal, bandwidth, sample_size):
    U_k = restrict_eigenbasis(eigenbasis, bandwidth)
    proj = calc_proj(eigenbasis, bandwidth)
    # normalise this!
    raw_bandlimited_signal = proj @ signal
    bandlimited_signal = normalize(raw_bandlimited_signal, dim=0)
    p_star = optimal_sampling_distn(proj)
    sample_matrices = sample_nodes_with_distn(p_star, sample_size)
    sample_signal = sample_matrices["M"] @ bandlimited_signal
    reconstructed_signal = standard_decoder(
        sample_matrices["P_Omega_inv_sqrt"], sample_matrices["M"], U_k, sample_signal
    )
    return th.norm(bandlimited_signal - reconstructed_signal, p=2)


def nodewise_reconstruction_error_with_eigenbasis(
    eigenbasis, signal, bandwidth, sample_size
):
    U_k = restrict_eigenbasis(eigenbasis, bandwidth)
    proj = calc_proj(eigenbasis, bandwidth)
    raw_bandlimited_signal = proj @ signal
    bandlimited_signal = normalize(raw_bandlimited_signal, dim=0)
    p_star = optimal_sampling_distn(proj)
    sample_matrices = sample_nodes_with_distn(p_star, sample_size)
    sample_signal = sample_matrices["M"] @ bandlimited_signal
    reconstructed_signal = standard_decoder(
        sample_matrices["P_Omega_inv_sqrt"], sample_matrices["M"], U_k, sample_signal
    )
    return bandlimited_signal - reconstructed_signal


# to account for:
# normalisation!
# P Omega inverse square root (it doesn't work properly)
def analytic_squared_errors(U_k, M, pois=None):
    U_k = U_k.double()
    M = th.sparse_csr_tensor(
        crow_indices=M.crow_indices(),
        col_indices=M.col_indices(),
        values=th.ones_like(M.col_indices(), dtype=th.float64),
        size=tuple(M.shape),
    )
    MU_k = M @ U_k
    if pois is not None:
        MU_k = pois @ MU_k
    # for all A, A.pinverse() @ A is a projection
    P = th.linalg.pinv(MU_k) @ MU_k
    I = th.eye(P.shape[0])
    # given you project a gaussian signal
    # then sample the projected signal,
    # reconstruct and compare it to the projected signal
    # these errors are a linear transform of a gaussian
    # and have distribution N(0, U_k (I - P) U_k.T)
    # so the node errors have mean 0 and variance the diagonal
    # the total mse is the trace of this, and is n - rank(MU_k)
    cov = U_k @ (I - P) @ U_k.T
    err = th.diag(cov)
    err[err.abs() < 1e-10] = 0.0
    return err


# @torch.jit.script
# it's much faster to multiply the matrix by its transpose
# and use hermitian = True
def fast_exact_matrix_rank(A):  #: TensorType["batch", "m", "k"]):
    # I'm surprised that matrix rank doesn't already do this, honestly.
    _, m, k = A.shape
    # if m > 1.1 * k:
    if m > k:
        X = th.bmm(th.transpose(A, 2, 1), A)
    # elif k > 1.1 * m:
    else:
        X = th.bmm(A, th.transpose(A, 2, 1))
    if X.shape[-1] == 1:
        return (X.squeeze() != 0).int()  # this gives a surprising amount of savings
    else:
        return th.linalg.matrix_rank(X, hermitian=True)
    # else:
    #     X = A
    #     return th.linalg.matrix_rank(X)
    # return th.linalg.matrix_rank(X, hermitian=True)


def all_sample_total_analytic_error(U, num_nodes, num_repeats: int = 100):
    num_nodes = U.shape[0]
    bandwidths = th.arange(1, num_nodes)
    sample_sizes = th.arange(1, num_nodes)
    distribution = uniform_sampling_distn(U)

    # reorder so you can stack all matrices of the same size up and matrix_rank
    # turning 40k calls of matrix rank into 200
    result = []
    for sample_size in tqdm(sample_sizes):
        if math.comb(num_nodes, sample_size) < num_repeats:
            M_indices = index_combinations(
                num_nodes, sample_size, with_replacement=False
            )
            square_small_sample_size = lambda bm: th.bmm(bm, th.transpose(bm, 2, 1))
            MU_ks = th.stack(
                [
                    square_small_sample_size(restrict_eigenbasis(U, k)[M_indices])
                    for k in bandwidths
                ]
            )

            result.append(
                bandwidths - (th.linalg.matrix_rank(MU_ks).float().mean(dim=-1))
            )
        # else:
        all_Ms = batch_sample_nodes_with_distn(
            distribution, sample_size, num_repeats, num_repeats_batch=len(bandwidths)
        )["M"]
        square_it = lambda bm: th.bmm(bm, th.transpose(bm, 2, 1))
        MU_ks = th.stack(
            [
                square_it(
                    th.bmm(
                        Ms,
                        restrict_eigenbasis(U, k)
                        .unsqueeze(0)
                        .expand(num_repeats, -1, -1),
                    )
                )
                for Ms, k in zip(all_Ms, bandwidths)
            ]
        )
        result.append(
            th.cat(
                [
                    bandwidths - th.linalg.matrix_rank(MU_ks).float().mean(dim=1),
                ]
            )
        )

    return th.stack(result)


def index_combinations(n, sample_size: int, with_replacement: bool = False):
    # th.combinations for th.arange(11), r=11, with_replacement=False hangs the computer
    # th.combinations(
    #     th.arange(num_nodes), r=sample_size, with_replacement=False
    # )
    #
    if with_replacement:
        raise NotImplementedError
    else:
        return th.tensor(
            list(itertools.combinations(range(n), sample_size)), dtype=th.long
        )


def all_MUks(U, bandwidth, sample_size):
    num_nodes = U.shape[0]
    U_k = restrict_eigenbasis(U, bandwidth)
    M_indices = index_combinations(num_nodes, sample_size, with_replacement=False)
    MU_ks = U_k[M_indices]
    return MU_ks


# @th.compile()
def sampled_MUks(U, bandwidth, sample_size, num_repeats: int = 10):
    U_k = restrict_eigenbasis(U, bandwidth)
    distribution = uniform_sampling_distn(U_k)
    # Ms = batch_sample_nodes_with_distn(distribution, sample_size, num_repeats)["M"]
    omegas = batch_sample_nodes_with_distn(distribution, sample_size, num_repeats)[
        "omega"
    ]
    # MU_ks = th.bmm(Ms, U_k.unsqueeze(0).expand(num_repeats, -1, -1))
    MU_ks = U_k[omegas]
    return MU_ks


# we use the fact that if a bunch of rows are independent in
# U_k, they are independent in U_(k+1)
# The extra complexity in enabling this gives a 20% speedup
# Also if
def all_exact_total_analytic_error(U, exact_rank_calc=False, parallel=True):
    num_nodes = U.shape[0]

    all_coranks = []
    # A dependence system is a set (bandwidth k, subset of rows of U_k called S), where any strict
    # subset of S is independent. if (k,S) is a dependence system, so is (k-1,S).
    # dependence_systems = []
    # prev_rank_cache, prev_indices = None, None
    for sample_size in tqdm(range(1, num_nodes)):
        M_indices = index_combinations(num_nodes, sample_size, with_replacement=False)
        # cache of size bandwidths x num_combs
        rank_cache = th.full(
            (num_nodes - 1, M_indices.shape[0]), fill_value=-1, dtype=th.long
        )
        for bandwidth in range(1, num_nodes):
            # print(f"bandwidth: {bandwidth}, sample_size: {sample_size}")
            # print(all_coranks)
            # if len(all_coranks) and all_coranks[-1][bandwidth - 1] == 0:
            # if all m-sized samples are error free, then all m+1 sized samples will be error free
            # rank_cache[bandwidth - 1] = bandwidth
            U_k = restrict_eigenbasis(U, bandwidth=bandwidth)
            interesting_combs = rank_cache[bandwidth - 1] == -1
            if not interesting_combs.any():
                break
            MU_ks = U_k[M_indices[interesting_combs]]
            rank_cache[bandwidth - 1, interesting_combs] = (
                fast_exact_matrix_rank(MU_ks)
                if exact_rank_calc
                else fast_approx_rank(MU_ks, max_rank=bandwidth).long()
            )
            rank_cache[bandwidth - 1 :, rank_cache[bandwidth - 1] == sample_size] = (
                sample_size
            )
            # print(rank_cache)
        # prev_rank_cache, prev_indices = rank_cache, M_indices
        coranks = th.arange(1, num_nodes) - rank_cache.float().mean(dim=1)
        all_coranks.append(coranks)
    print(num_nodes)
    print(list(range(1, num_nodes)))
    print(th.stack(all_coranks))
    return th.stack(all_coranks).T


def sample_total_analytic_error(
    U, bandwidth, sample_size, num_repeats: int = 10, exact_rank_calc=False
):
    num_nodes = U.shape[0]
    # if it's easier to sample exactly, do so
    num_combs = math.comb(num_nodes, sample_size)
    if num_combs < num_repeats:
        MU_ks = all_MUks(U, bandwidth, sample_size)
    else:
        MU_ks = sampled_MUks(U, bandwidth, sample_size, num_repeats)
    # analytically, the error is trace(U_k^T U_k (I-P)) = k - tr(P) = k - rank(MU_k)
    ranks = (
        fast_exact_matrix_rank(MU_ks)
        if exact_rank_calc
        else fast_approx_rank(MU_ks, max_rank=bandwidth)
    )
    avg_err = (bandwidth - ranks).float().mean()
    return avg_err


def sample_total_analytic_error_extra_noise(
    U,
    bandwidth,
    sample_size,
    noise_var: float = 0.001,
    num_repeats: int = 10,
    exact_rank_calc=False,
):
    num_nodes = U.shape[0]
    # if it's easier to sample exactly, do so
    num_combs = math.comb(num_nodes, sample_size)
    if num_combs < num_repeats:
        all_omegas = index_combinations(num_nodes, sample_size, with_replacement=False)
        MU_ks = all_MUks(U, bandwidth, sample_size)
    else:
        MU_ks = sampled_MUks(U, bandwidth, sample_size, num_repeats)
    # analytically, the error is trace(U_k^T U_k (I-P)) = k - tr(P) = k - rank(MU_k)
    ranks = (
        fast_exact_matrix_rank(MU_ks)
        if exact_rank_calc
        else fast_approx_rank(MU_ks, max_rank=bandwidth)
    )
    avg_err = (bandwidth - ranks).float().mean()
    return avg_err


def get_mean_corank(U, bandwidth, sample_size, num_repeats=4000000):
    num_nodes = U.shape[0]
    U_k = U[:, :bandwidth]
    U_k_prime = U[:, bandwidth:]
    if math.comb(num_nodes, sample_size) < num_repeats:
        M_indices = index_combinations(num_nodes, sample_size, with_replacement=False)
    else:
        distribution = uniform_sampling_distn(U_k)
        M_indices = batch_sample_nodes_with_distn(
            distribution, sample_size, num_repeats
        )["omega"]
    MU_ks = U_k[M_indices]  # batch x sample_size x k
    coranks = bandwidth - fast_exact_matrix_rank(MU_ks).float()
    return th.mean(coranks)


def high_freq_noise_decoded(U, bandwidth, sample_size, num_repeats=4000, agg=th.mean):
    num_nodes = U.shape[0]
    U_k = U[:, :bandwidth]
    U_k_prime = U[:, bandwidth:]
    if math.comb(num_nodes, sample_size) < num_repeats:
        M_indices = index_combinations(num_nodes, sample_size, with_replacement=False)
    else:
        distribution = uniform_sampling_distn(U_k)
        M_indices = batch_sample_nodes_with_distn(
            distribution, sample_size, num_repeats
        )["omega"]
    MU_ks = U_k[M_indices]  # batch x sample_size x k
    MU_k_primes = U_k_prime[M_indices]  # batch x sample_size x n-k
    p_inv_MU_ks = th.linalg.pinv(MU_ks)  # batch x k x sample_size
    not_proj = th.bmm(p_inv_MU_ks, MU_k_primes)
    # trace(X^T X) is the sum of the square of the entries of all of X
    # mse = tr(var(XZ)) = tr(XX^T) = tr(X^T X), Z ~ N(0,I), X a matrix
    mse = agg(th.sum(not_proj**2, dim=(1, 2)))
    # min_mse = th.min(th.sum(not_proj**2, dim=(1, 2)))
    return mse


def high_freq_noise_decoded_analytic(
    U, bandwidth, sample_size, num_repeats=4000, agg=th.mean
):
    num_nodes = U.shape[0]
    U_k = U[:, :bandwidth]
    if math.comb(num_nodes, sample_size) < num_repeats:
        M_indices = index_combinations(num_nodes, sample_size, with_replacement=False)
    else:
        distribution = uniform_sampling_distn(U_k)
        M_indices = batch_sample_nodes_with_distn(
            distribution, sample_size, num_repeats
        )["omega"]
    MU_ks = U_k[M_indices]  # batch x sample_size x k
    p_inv_MU_ks = th.linalg.pinv(MU_ks)  # batch x k x sample_size
    square_frob = th.sum(p_inv_MU_ks**2, dim=(1, 2))
    ranks = fast_exact_matrix_rank(MU_ks)
    mse = square_frob - ranks
    return agg(mse)


# approximates the function 1{x=0} with a quadratic over [0,1]
# under a probability distribution which is uniform on [0,1], and takes 0 with probability p
def find_best_quadratic_approximation(p):
    assert p < 1
    gamma = p / (1 - p)

    def cost(arr):
        a, b, c = arr
        return (
            gamma * ((1 - c) ** 2)
            + ((a**2) / 5)
            + (a * (3 * b + 4 * c) / 6)
            + (b**2) / 3
            + b * c
            + (c**2)
        )

    return scipy.optimize.minimize(cost, [1, 0, 0])


def rank_submatrices(U, bandwidth, sample_size, num_repeats, eps=1e-7):
    num_nodes = U.shape[0]
    U_k = U[:, :bandwidth]
    if math.comb(num_nodes, sample_size) < num_repeats:
        M_indices = index_combinations(num_nodes, sample_size, with_replacement=False)
    else:
        distribution = uniform_sampling_distn(U_k)
        M_indices = batch_sample_nodes_with_distn(
            distribution, sample_size, num_repeats
        )["omega"]
    MU_ks = U_k[M_indices]  # batch x sample_size x k
    # eigenvalues of projectors:
    sq_singular_vals = th.linalg.svdvals(MU_ks) ** 2
    sq_singular_vals[sq_singular_vals.abs() < eps] = 0
    MProj = th.einsum("bij,bkj -> bik", MU_ks, MU_ks)
    traces = vmap(th.trace)(MProj)
    sq_frobs = th.sum(MProj**2, dim=(1, 2))
    sigma_sq = 0.0000000
    # calculate actual loss
    loss = sq_singular_vals.clone()
    if sigma_sq == 0:
        loss[loss != 0] = 0.0
    else:
        loss[loss != 0] = sigma_sq / loss[loss != 0]
    loss[sq_singular_vals == 0] = 1.0
    loss = loss.sum(dim=1)
    proxy_loss = sq_frobs - (2 * traces) + sample_size
    linear_loss = -1 * traces
    # return scaled proxy_loss, loss
    proxy_loss -= th.min(proxy_loss)
    linear_loss -= th.min(linear_loss)
    proxy_loss /= th.max(proxy_loss)
    linear_loss /= th.max(linear_loss)

    fig, ax = plt.subplots(1, 2)
    fig.suptitle(
        f"My vs Puy loss: Binary Tree,  extra noise var = {sigma_sq}, bandwidth = {bandwidth}, sample size = {sample_size}"
    )
    ax = ax.ravel()
    sns.scatterplot(x=proxy_loss, y=th.log(loss), ax=ax[0])
    ax[0].set_xlabel("proxy quadratic loss")
    ax[0].set_ylabel("log loss")
    ax[0].text(
        0,
        1,
        f"spearman_corr: {spearmanr(proxy_loss, loss).correlation}",
        transform=ax[0].transAxes,
    )
    sns.scatterplot(x=linear_loss, y=th.log(loss), ax=ax[1])
    ax[1].text(
        0,
        1,
        f"spearman_corr: {spearmanr(linear_loss, loss).correlation}",
        transform=ax[1].transAxes,
    )
    ax[1].set_xlabel("proxy linear loss (Puy)")
    ax[1].set_ylabel("log loss")
    plt.show()


def find_bad_combos(U, bandwidth, sample_size):
    num_nodes = U.shape[0]
    # if it's easier to sample exactly, do so
    num_combs = math.comb(num_nodes, sample_size)
    if num_combs > 10000:
        print("too many combinations!")
        return ()
    else:
        U_k = restrict_eigenbasis(U, bandwidth)
        M_indices = index_combinations(num_nodes, sample_size, with_replacement=False)
        MU_ks = U_k[M_indices]
        coranks = min(bandwidth, sample_size) - th.linalg.matrix_rank(MU_ks)
        return [
            (indices, muk, corank)
            for (corank, indices, muk) in zip(coranks, M_indices, MU_ks)
            if corank > 0
        ]


def find_worst_conditioned_pair(U, eps=1e-8):
    num_nodes = U.shape[0]
    if math.comb(num_nodes, 2) > 1000000:
        print("too many combs!")
    else:
        U_k = restrict_eigenbasis(U, 2).double()
        M_indices = index_combinations(num_nodes, 2)
        MU_ks = U_k[M_indices]
        svds = th.linalg.svdvals(MU_ks)
        svds[svds.abs() < eps] = th.inf
        noise_vars = svds.reciprocal().square().sum(dim=1)
        max_noise_std, index_of_max = noise_vars.sqrt().max(dim=0)
        return max_noise_std, U_k[M_indices[index_of_max]]


def find_worst_conditioned_pair_sympy(graph, num_nodes):
    if num_nodes > 20:
        print("too many nodes!")
    import sympy

    # using unnormalized laplacian - it's integral, so can get exact values
    laplacian_edges, laplacian_edge_weights = get_laplacian(
        graph, normalization=None, num_nodes=num_nodes  # "sym",
    )
    laplacian = (
        to_dense_adj(
            laplacian_edges, edge_attr=laplacian_edge_weights, max_num_nodes=num_nodes
        )
        .squeeze()
        .int()
    )
    L = sympy.Matrix(laplacian)
    evecs = L.eigenvects()
    evec_dict = {evalue: evectors for (evalue, _, evectors) in evecs}
    evalues = list(evec_dict.keys())
    evalues_float = [sympy.N(sympy.re(evalue), chop=True) for evalue in evalues]
    sorted_evalues = np.array(evalues)[np.argsort(np.array(evalues_float))]
    first_evec = sympy.re(evec_dict[sorted_evalues[0]][0])
    second_evec = sympy.re(evec_dict[sorted_evalues[1]][0])
    # print(second_evec)
    # print(f"evalue:{sympy.re(sorted_evalues[1]).evalf()}")
    U_2 = first_evec.row_join(second_evec)

    M_indices = index_combinations(num_nodes, 2)
    errs = []
    for m in tqdm(M_indices):
        s = U_2[list(m), :]
        # sts = s * s.transpose()
        # det_s = sympy.re(sympy.det(s))
        # if det_s != 0:
        #     det_sts = det_s**2
        #     tr = sympy.trace(sts)
        #     # evals = sts.eigenvals()
        a = s[0, 0]
        b = s[0, 1]
        c = s[1, 0]
        d = s[1, 1]
        det_s = (a * d) - (b * c)
        if det_s != 0:
            det_sts = det_s**2
            sts = s * s.transpose()
            tr_sts = sympy.trace(sts)
            err = sympy.N(sympy.re(tr_sts / det_sts), maxn=10000)
            errs.append((err, m, sts))
    # return evec_dict, sorted_evalues, U_2, errs
    return errs


@typechecked
def principal_min_eigenvalues(M: TensorType["N", "N"], k: int):
    # assumes M square, real, symmetric
    # for all principal submatrices of size k,
    # find min eigenvalues
    N = M.shape[0]
    # return th.hstack([smallest_abs(th.linalg.eigvalsh(m)) for m in princ_submatrices])
    res = []
    max_sub = []
    for comb in tqdm(index_combinations(N, k)):
        princ = M[comb, :][:, comb]
        eigs = th.linalg.eigvalsh(princ)
        smallest = eigs[th.argmin(eigs.abs())]
        res.append(smallest)
        sub_eigs = []
        for comb2 in index_combinations(k, k - 1):
            subprinc = princ[comb2, :][:, comb2]
            eigs = th.linalg.eigvalsh(subprinc)
            smallest = eigs[th.argmin(eigs.abs())]
            sub_eigs.append(smallest)
        max_sub.append(th.max(th.hstack(sub_eigs)))

    return th.hstack(res), th.hstack(max_sub)


def compound_k(M, k):
    # M is n x m
    # return type is (nCk) x (mCk)
    rows = index_combinations(M.shape[0], k)
    cols = index_combinations(M.shape[1], k)
    # if M.dtype in [th.int32, th.long]:
    #     det_fn = lambda x: th.tensor(int_det(x))
    # else:
    det_fn = th.linalg.det
    return th.vstack([th.hstack([det_fn(M[js][:, ks]) for js in rows]) for ks in cols])


def some_daggers(laplacian, U, sample_size, bandwidth=2, mu=0.01, num_repeats=10):
    num_nodes = laplacian.shape[0]
    assert sample_size <= num_nodes
    U_k = restrict_eigenbasis(U, bandwidth)
    for _ in range(num_repeats):
        omega = th.randperm(num_nodes)[:sample_size]
        M = th.zeros(sample_size, num_nodes)
        M[range(sample_size), omega] = 1.0
        mdaggermu = ((M.T @ M) + mu * laplacian.float()).inverse() @ M.T
        print(th.linalg.svdvals(mdaggermu))
        bias = U_k - (mdaggermu @ U_k[omega])
        bias_no_uk = th.eye(num_nodes) - (mdaggermu @ M)
        print(
            f"noise coeff:{(mdaggermu**2).sum().item()}, signal_coeff:{(bias**2).sum().item()}, signal coeff without uk:{(bias_no_uk ** 2).sum().item()}"
        )


# for a square matrix, calculate the adjugate
def calc_adjugate(M):
    n = M.shape[0]
    comp = compound_k(M, n - 1)
    for i in range(n):
        for j in range(n):
            if (i + j) % 2:
                comp[i, j] *= -1.0
    return comp.T


def test_glr_err(num_signals=1000):
    err_diffs = []
    for _ in range(5):
        graph, num_nodes = clean_graph(erdos_renyi_graph(50, 0.8))
        L = calc_laplacian(graph, num_nodes).float()
        U = calc_eigenbasis(graph, num_nodes).float()
        for _ in range(10):
            bandwidth = 20
            omega = th.randperm(num_nodes)[:bandwidth]
            snr = 1
            mu = 0.01
            sigma_sq = bandwidth / float(num_nodes * snr)
            sample_err = sample_glr_err(
                L, U, bandwidth, num_signals, omega, mu=mu, snr=snr
            )
            U_k = restrict_eigenbasis(U, bandwidth)
            err_sig, err_noise = analytic_glr_err(L, U_k, mu=mu, sample_set=omega)
            analytic_err = err_sig + sigma_sq * err_noise
            err_diff = (analytic_err - sample_err).abs()
            err_diffs.append(err_diff)
            print(
                f"sample: {sample_err}, predicted: {analytic_err}, diff = {(sample_err - analytic_err).abs()}"
            )
    print(f"avg diff: {th.tensor(err_diffs).mean()}")


def sample_glr_err(
    laplacian, eigenbasis, bandwidth, num_signals, omega, mu=0.01, snr=5.0
):
    num_nodes = laplacian.shape[0]
    U_k = restrict_eigenbasis(eigenbasis, bandwidth)
    # proj = calc_proj(eigenbasis, bandwidth)
    # equivalent in distribution to proj @ noise signals
    raw_bandlimited_signals = U_k @ noise_signal(bandwidth, num_signals)
    # bandlimited_signals = raw_bandlimited_signals
    # bandlimited_signals = normalize(raw_bandlimited_signals, dim=0)
    bandlimited_signals = raw_bandlimited_signals
    # note that if noise_err = \sigma, noise has variance multiplied by
    # \sigma^2
    noise_err = th.sqrt(th.tensor(bandwidth / (num_nodes * snr)))
    noise = noise_err * noise_signal(num_nodes, num_signals)
    noisy_bandlimited_signals = bandlimited_signals + noise
    sample_signals = noisy_bandlimited_signals[omega]
    reconstructed_signals = graph_laplacian_decoder_multiple_signals(
        laplacian, omega, mu=mu, ys=sample_signals
    )
    return ((bandlimited_signals - reconstructed_signals) ** 2).sum(dim=0).mean()


# this needs to include the bias term, and I don't know if it does rn!
# returns (err_sig, err_noise) s.t. total MSE = err_sig + sigma^2 err_noise
def analytic_glr_err(laplacian, U_k, mu, sample_set, bandlimited_noise=False):
    rec = (mu * laplacian).clone()  # mu * L
    rec[sample_set, sample_set] += 1.0  # (M^T M + mu * L)
    # rec = rec.inverse()[:, sample_set]  # (M^TM + mu * L)^-1 M^T
    # Using solve is 2x faster for sample_size << num_nodes:
    #
    # checking if it's a tensor stops a warning
    sample_set_th = (
        sample_set if type(sample_set) == th.Tensor else th.tensor(sample_set)
    )
    MT = one_hot(sample_set_th, laplacian.shape[0]).T.to(laplacian.dtype)
    rec = th.linalg.solve(rec, MT)  # (M^TM + mu * L)^-1 M^T

    # err_sig_sqrt = U_k - (rec @ U_k[sample_set])
    if bandlimited_noise:
        R_SM_SU_k = rec @ U_k[sample_set]
        err_sig_sqrt = U_k - R_SM_SU_k
        err_sig = (err_sig_sqrt**2).sum()
        err_noise = (R_SM_SU_k**2).sum()
        return (err_sig, err_noise)
    else:
        err_sig_sqrt = th.addmm(U_k, rec, U_k[sample_set], alpha=-1.0)
        err_sig = (err_sig_sqrt**2).sum()
        err_noise = (rec**2).sum()
        return (err_sig, err_noise)


@njit(nopython=True)
def analytic_glr_err_numpy(laplacian, U_k, mu, sample_set):
    N, _ = laplacian.shape
    rec = np.copy(mu * laplacian)  # mu * L
    if len(sample_set) > 0:
        for s in sample_set:
            rec[s, s] += 1.0  # (M^T M + mu * L)
    # Using solve is 2x faster for sample_size << num_nodes:
    MT = np.eye(N, dtype=rec.dtype)[sample_set].T
    rec = np.linalg.solve(rec, MT)  # (M^TM + mu * L)^-1 M^T
    rec = rec.astype(laplacian.dtype)  # otherwise it upcasts to float64

    err_noise = (rec**2).sum()
    # err_sig_sqrt = U_k - (rec @ U_k[sample_set])
    err_sig_sqrt = U_k - rec @ U_k[sample_set]
    err_sig = (err_sig_sqrt**2).sum()
    return (err_sig, err_noise)


# TODO : make this actual pytest test
def test_glr_mult(N=1000, k=100, m=20, mu=0.01, num_old_samples=1):
    g1, n1 = clean_graph(erdos_renyi_graph(N, 0.8))
    L = calc_laplacian(g1, n1)
    U = calc_eigenbasis(g1, n1, double=True)
    U_k = restrict_eigenbasis(U, k)
    samples = th.randperm(N)[:m]
    old_samples = samples[:num_old_samples]
    new_samples = samples[num_old_samples:]

    old_inv = (mu * L).clone()
    old_inv[old_samples, old_samples] += 1.0
    old_inv = th.linalg.inv(old_inv)

    test_err_sig, test_err_noise = analytic_glr_err_update_multiple(
        old_inv,
        old_inv_sq=old_inv @ old_inv,
        old_inv_T_U_k=old_inv @ U_k,
        U_k=U_k,
        U_k_U_k_T=U_k @ U_k.T,
        new_samples=new_samples,
        all_samples=samples,
    )

    new_inv = (mu * L).clone()
    new_inv[samples, samples] += 1.0
    new_inv = th.linalg.inv(new_inv)

    # err_noise = new_inv[samples].square().sum()
    err_sig, err_noise = analytic_glr_err(L, U_k, mu, samples)

    print(f"sig err: {err_sig}, deviation: {(test_err_sig - err_sig).abs()}")
    print(f"noise err: {err_noise}, deviation: {(test_err_noise - err_noise).abs()}")


@typechecked
# Assumes old_inv is symmetric
# requires everything to be Double to stop error from increasing too much
def analytic_glr_err_update_multiple(
    old_inv: TensorType["N", "N"],
    old_inv_sq: TensorType["N", "N"],
    old_inv_T_U_k: TensorType["N", "k"],
    U_k: TensorType["N", "k"],
    U_k_U_k_T: TensorType["N", "N"],
    new_samples: TensorType["n"],
    all_samples: TensorType["m"],
) -> Tuple[TensorType[()], TensorType[()]]:
    new_samples_set = set(new_samples.tolist())
    all_samples_set = set(all_samples.tolist())
    assert new_samples_set.issubset(all_samples_set)
    N, k = U_k.shape
    m, n = all_samples.shape[0], new_samples.shape[0]  # m > n

    # 1/6 of the time is spent in this function:
    def submatrixh(X, rows, cols):
        """for a symmetric matrix X, efficiently calculates X[rows][:,cols]"""
        return (X[cols].T)[rows]

    # Use woodbury algorithm
    # Can expand err_noise and err_sig_sqrt calculations into a bunch of mxm multiplications
    # and avoid any NxN multiplications. Only operations on dimensions of size N are indexing.
    # This gives ~5x speedup vs calculating the full inverse.
    # It is, however, less numerically stable for some reason (more operations?)
    #
    # note that err_sig = ||U_k||^2_2 + `tr(RTR MUUTMT) - 2 * tr(U_k.T @ rec @ M @ U_k)`
    #
    # We need to materialise U_k.T @ rec and rec.T @ rec
    #
    # F = (I + M_Delta A^-1 M_Delta^T) :: n x n - it's tiny!
    F = th.eye(n) + submatrixh(old_inv, new_samples, new_samples)

    # precompute F^-1 M_Delta A^-1 M_S
    F_inv_old_inv_S = th.linalg.solve(F, submatrixh(old_inv, new_samples, all_samples))
    # Z : m x m and Z.T are the cross terms in calculating
    # ||R_S||^2_2 via woodbury.
    Z = submatrixh(old_inv_sq, all_samples, new_samples) @ F_inv_old_inv_S
    RecTRec = (
        submatrixh(old_inv_sq, all_samples, all_samples)
        + (
            F_inv_old_inv_S.T
            @ submatrixh(old_inv_sq, new_samples, new_samples)
            @ F_inv_old_inv_S
        )
        - Z
        - Z.T
    )

    err_noise = th.trace(RecTRec)
    # ||U_k||^2_2 = k
    # tensordot(A,B) = trace(A.T @ B)
    # tensordot may be slower if, but seems more numerically stable
    Rec_T_U_k = old_inv_T_U_k[all_samples] - (
        F_inv_old_inv_S.T @ old_inv_T_U_k[new_samples]
    )
    err_sig = (
        k
        + th.tensordot(RecTRec, submatrixh(U_k_U_k_T, all_samples, all_samples))
        - (2 * th.tensordot(Rec_T_U_k, U_k[all_samples]))
    )
    return (err_sig, err_noise)
    # new_inv = old_inv - (old_inv[:, new_samples] @ F_inv @ old_inv[new_samples])
    # return new_inv


def sqsum(m):
    return (m**2).sum()


def analytic_glr_err_update(old_inv, U_k, sample_set):
    # assumes the new sample is at the end of 'sample_set'
    N = U_k.shape[0]
    # u = th.zeros(N)
    # u[sample_set[-1]] = 1.0
    u = one_hot(th.tensor(sample_set[-1]), N).type(old_inv.dtype)
    # u = th.eye(N)[sample_set[-1]]
    # this updates (M^T M + mu * L)^-1
    new_inv = update_inv_sym_uut(old_inv, u)
    # new_inv = update_inv_index(old_inv, sample_set[-1])
    rec = new_inv[:, sample_set]  # this is (M^T M + mu * L)^-1 M^T
    # recT = new_inv[sample_set] # b
    # ecause
    # err_noise = (rec**2).sum()
    err_noise = sqsum(rec)
    # err_sig_sqrt = U_k - (rec @ U_k[sample_set])
    err_sig_sqrt = th.addmm(U_k, rec, U_k[sample_set], alpha=-1.0)
    # err_sig = (err_sig_sqrt**2).sum()
    err_sig = sqsum(err_sig_sqrt)
    return (err_sig, err_noise)


def analytic_glr_err_update_fast(old_inv, U_k, sample_set):
    # this updates (M^T M + mu * L)^-1 via the sherman-morrison formula.
    # only computes the needed rows.
    i = sample_set[-1]
    recT = th.addr(
        old_inv[sample_set],
        old_inv[sample_set, i],
        old_inv[i],
        alpha=-1.0 / (1 + old_inv[i, i]),
    )
    # err_noise = (rec**2).sum()
    err_noise = sqsum(recT)
    # err_sig_sqrt = U_k - (rec @ U_k[sample_set])
    err_sig_sqrt = th.addmm(U_k, recT.T, U_k[sample_set], alpha=-1.0)
    # err_sig = (err_sig_sqrt**2).sum()
    err_sig = sqsum(err_sig_sqrt)
    return (err_sig, err_noise)


@njit(nopython=True)
def analytic_glr_err_update_fast_numpy(old_inv, U_k, sample_set):
    # this updates (M^T M + mu * L)^-1 via the sherman-morrison formula.
    # only computes the needed rows.
    i = sample_set[-1]
    recT = old_inv[sample_set] - (
        np.outer(old_inv[sample_set, i], old_inv[i]) / (1 + old_inv[i, i])
    )
    err_noise = (recT**2).sum()
    rec = recT.T
    rec = rec.astype(U_k.dtype)
    err_sig_sqrt = U_k - (rec @ U_k[sample_set])
    err_sig = (err_sig_sqrt**2).sum()
    return (err_sig, err_noise)


def greedy_glr(laplacian, U, bandwidth, mu=0.01, max_samples=None):
    N = laplacian.shape[0]
    sig_errs = []
    errs = []
    sampling_set = []
    if max_samples is None:
        max_samples = bandwidth
    U_k = restrict_eigenbasis(U, bandwidth).contiguous()
    with tqdm(total=max_samples * N) as pbar:
        for _ in range(max_samples):
            inner_errs = []
            inner_sig_errs = []
            for s in range(N):
                if s in sampling_set:
                    inner_sig_errs.append(np.inf)
                    inner_errs.append(np.inf)
                else:
                    err = analytic_glr_err(
                        laplacian, U_k, mu, sample_set=sampling_set + [s]
                    )
                    inner_sig_errs.append(err[0].item())
                    inner_errs.append(err[1].item())
                pbar.update(1)
            new_addition = np.argmin(inner_errs)
            errs.append(inner_errs[new_addition])
            sig_errs.append(inner_sig_errs[new_addition])
            sampling_set.append(new_addition)
    SNRs = np.diff(errs) / (-1 * np.diff(sig_errs)) * float(bandwidth) / float(N)
    snr_dbs = 10 * np.log10(np.clip(SNRs, 1e-1, np.inf))
    return {
        "sampling_set": sampling_set,
        "errs": errs,
        "sig_errs": sig_errs,
        "SNRs": SNRs,
        "snr_dbs": snr_dbs,
    }


# attempts at using vmap slow this down;
def greedy_glr_fast(
    laplacian, U, bandwidth, mu=0.01, max_samples=None, internal_dtype=th.float64
):
    N = laplacian.shape[0]
    laplacian = laplacian.type(internal_dtype)
    U = U.type(internal_dtype)
    U_k = restrict_eigenbasis(U, bandwidth).contiguous()
    sig_errs = []
    errs = []
    sampling_set = []
    if max_samples is None:
        max_samples = bandwidth
    with tqdm(total=max_samples * N) as pbar:
        for i in range(max_samples):
            inner_errs = []
            inner_sig_errs = []
            if i == 0:
                for s in range(N):
                    err = analytic_glr_err(
                        laplacian, U_k, mu, sample_set=sampling_set + [s]
                    )
                    inner_sig_errs.append(err[0].item())
                    inner_errs.append(err[1].item())
                    pbar.update(1)
            else:
                # calculate (M^T M + mu * L)^-1 via sherman-morrison
                # can't do this if sampling_set is empty!
                adjustedL = (mu * laplacian).clone()
                adjustedL[sampling_set, sampling_set] += 1.0
                current_inv = adjustedL.inverse().contiguous()
                for s in range(N):
                    if s in sampling_set:
                        inner_sig_errs.append(np.inf)
                        inner_errs.append(np.inf)
                    else:
                        err = analytic_glr_err_update_fast(
                            current_inv, U_k, sampling_set + [s]
                        )
                        inner_sig_errs.append(err[0].item())
                        inner_errs.append(err[1].item())
                    pbar.update(1)
            new_addition = np.argmin(inner_errs)
            errs.append(inner_errs[new_addition])
            sig_errs.append(inner_sig_errs[new_addition])
            sampling_set.append(new_addition)
    SNRs = np.diff(errs) / (-1 * np.diff(sig_errs)) * float(bandwidth) / float(N)
    snr_dbs = 10 * np.log10(np.clip(SNRs, 1e-1, np.inf))
    return {
        "sampling_set": sampling_set,
        "errs": errs,
        "sig_errs": sig_errs,
        "SNRs": SNRs,
        "snr_dbs": snr_dbs,
    }


def plot_glr_greedy_mse(
    graph,
    num_nodes,
    bandwidth,
    mus=[1, 0.01, 0.0001, 1e-7],
    SNRs=[1.0],
    graph_name=None,
    U=None,
    max_samples=None,
):
    laplacian = calc_laplacian(graph, num_nodes)
    U = calc_eigenbasis(graph, num_nodes)
    dfs = []
    for mu in mus:
        err_dict = greedy_glr_fast(
            laplacian,
            U,
            bandwidth,
            mu=mu,
            max_samples=max_samples,
            internal_dtype=th.float64,
        )
        for snr in SNRs:
            snrdb = 10 * np.log10(snr)
            total_mse = np.array(err_dict["sig_errs"]) + (
                np.array(err_dict["errs"])
                * (float(bandwidth) / (float(num_nodes) * snr))
            )
            df = pd.DataFrame(
                {
                    "mu": str(mu),
                    "Sample Size": list(range(1, 1 + len(err_dict["errs"]))),
                    "SNR (dB)": "{:.1f}".format(snrdb),
                    "log MSE": np.log(total_mse),
                }
            )
            dfs.append(df)
    total_df = pd.concat(dfs).reset_index()
    total_df["Mu | SNR (db)"] = total_df["mu"] + " | " + total_df["SNR (dB)"]
    fig, ax = plt.subplots()
    # sns.heatmap(log_hm, ax=ax)
    # sns.lineplot(data=data, ax=ax)
    sns.lineplot(
        data=total_df,
        x="Sample Size",
        y="log MSE",
        hue="Mu | SNR (db)",
        # ci="sd",
    )
    ax.set_title(
        f"GLR Signal Reconstruction of noisy signals on a Barabasi-Albert Graph under greedy sampling (bandwidth = {bandwidth})"
    )
    # ax.invert_yaxis()
    plt.show()


def plot_glr_greedy_thresholds(
    graph, num_nodes, bandwidth, mu=0.01, graph_name=None, U=None
):
    pass


def random_glr(graph, num_nodes, mu=0.01, bandwidth=1, SNR=1.0, U=None):
    L = calc_laplacian(graph, num_nodes).float()
    if U is None:
        U = calc_eigenbasis(graph, num_nodes)
    U_k = restrict_eigenbasis(U, bandwidth).contiguous()
    sample_order = th.randperm(num_nodes)
    sigma_sq = bandwidth / float(num_nodes * SNR)

    def calc_err(a):
        return a[0] + sigma_sq * a[1]

    return [
        calc_err(analytic_glr_err(L, U_k, mu, sample_order[:i]))
        for i in tqdm(range(1, bandwidth + 1))
    ]


# returns a tensor of size |bandwidths| x |SNRs|
def random_glr_heatmap(
    graph, num_nodes, mu=0.01, bandwidths=[2], SNRs=None, num_repeats=100, U=None
):
    L = calc_laplacian(graph, num_nodes)
    if U is None:
        U = calc_eigenbasis(graph, num_nodes)
    U_k = restrict_eigenbasis(U, bandwidth).contiguous()
    if type(SNRs) is not th.tensor:
        SNRs = th.tensor(SNRs)
    prob_errs = []
    for k in bandwidths:
        all_err_sigs = []
        all_err_noises = []
        for _ in range(num_repeats):
            sample_order = th.randperm(num_nodes)
            err_sigs, err_noises = zip(
                *[
                    analytic_glr_err(L, U_k, mu, sample_order[:i])
                    for i in range(1, k + 1)
                ]
            )
            all_err_sigs.append(th.tensor(err_sigs))
            all_err_noises.append(th.tensor(err_noises))
        all_err_sigs = th.vstack(all_err_sigs)
        all_err_noises = th.vstack(all_err_noises)
        sigma_sqs = k / (SNRs * float(num_nodes))

        def calc_err(e_sigs, e_noises):
            print(f"esig:{e_sigs}")
            print(f"e_noises:{e_noises}")
            print(f"sig sq:{sigma_sqs}")
            print(f"prod:{th.outer(sigma_sqs, e_noises)}")
            errs = e_sigs + th.outer(sigma_sqs, e_noises)
            print(errs)
            print("====")
            diffs = th.diff(errs) > 0  # if error is increasing with sample size!
            return diffs.float()

        tot_errs = vmap(calc_err)(all_err_sigs, all_err_noises)
        prob_err = tot_errs.mean(dim=0)
        # prob_err = einops.reduce(tot_errs, "repeats snrs samples -> snrs", "mean")
        prob_errs.append(prob_err)
        print(sigma_sqs)
    # return th.vstack(prob_errs)
    return prob_errs


# generic greedy sampling function
# have to turn typechecking off for parallelism
# @typechecked
def greedy_sampling(
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
        U = calc_eigenbasis(graph, num_nodes, normalization=normalization)
    L = calc_laplacian(graph, num_nodes, normalization=normalization).type(
        internal_dtype
    )
    U_k = restrict_eigenbasis(U, bandwidth).type(internal_dtype).contiguous()
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
    return sampling_set, overall_aggs


# generic greedy sampling function. Inner functions manipulate numpy arrays.
# TODO: make entirely numpy
@typechecked
def greedy_sampling_numpy(
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
        U = calc_eigenbasis(graph, num_nodes, normalization=normalization)
    L = calc_laplacian(graph, num_nodes, normalization=normalization).type(
        internal_dtype
    )
    U_k = restrict_eigenbasis(U, bandwidth).type(internal_dtype).contiguous()
    return greedy_sampling_numpy_inner(
        L.numpy(), U_k.numpy(), num_nodes, max_samples, sample_fn, calc_once_a_loop
    )


@njit
def greedy_sampling_numpy_inner(
    L,
    U_k,
    num_nodes,
    max_samples,
    sample_fn,
    calc_once_a_loop=None,
):
    sampling_set, overall_aggs = np.empty((2, 0), dtype=np.intc)
    for _ in range(max_samples):
        # The once-per-outer-loop calculations
        if calc_once_a_loop is None:
            big_calc_output = None
        else:
            big_calc_output = calc_once_a_loop(L, U_k, sampling_set)
        # the inner loop
        possible_samples = np.delete(np.arange(num_nodes, dtype=np.intc), sampling_set)

        # this version is slower:
        # possible_sampling_sets = np.array(
        #     [sampling_set + [s] for s in possible_samples], dtype=np.intc
        # )
        possible_sampling_sets = np.zeros(
            (num_nodes - sampling_set.shape[0], 1 + sampling_set.shape[0])
        )
        possible_sampling_sets += np.append(sampling_set, 0)
        possible_sampling_sets[:, -1] = possible_samples
        ## vmap halves speed
        sfn = lambda sset: sample_fn(big_calc_output, L, U_k, sset)
        # losses_and_aggs = [sfn(sset) for sset in possible_sampling_sets]
        # losses = [x for x, _ in losses_and_aggs]
        # aggs = [y for _, y in losses_and_aggs]
        losses = [sfn(sset)[0] for sset in possible_sampling_sets]
        # find optimal greedy choice
        min_loss_index = np.argmin(losses)
        best_s = possible_samples[min_loss_index]
        # corresponding_agg = aggs[min_loss_index]
        sampling_set.append(best_s)
        # overall_aggs.append(corresponding_agg)
    return sampling_set, None


def greedy_a_samples_only(graph, num_nodes, bandwidth, max_samples=None, **kwargs):
    def sample_fn(_, L, U_k, sampling_set):
        MU_k = U_k[sampling_set]
        svds = th.linalg.svdvals(MU_k)
        loss = svds.reciprocal().square().sum().item()
        return loss, svds

    sampling_set, _svds = greedy_sampling(
        graph,
        num_nodes,
        sample_fn=sample_fn,
        max_samples=max_samples,
        bandwidth=bandwidth,
        **kwargs,
    )
    return sampling_set


# directly takes trace of inverse
# caching MU_k once a loop is important
def greedy_a_samples_only_fast(graph, num_nodes, bandwidth, max_samples=None, **kwargs):
    def calc_once_a_loop(L, U_k, old_sampling_set):
        MU_k = U_k[old_sampling_set]
        # think harder about why we switch
        if len(old_sampling_set) < bandwidth:
            muu = MU_k @ MU_k.T
        else:
            muu = MU_k.T @ MU_k
        return (muu.inverse().contiguous(), MU_k)

    def sample_fn(prev_inv_and_muk, L, U_k, sampling_set):
        # MU_k = U_k[sampling_set]
        prev_inv, prev_MU_k = prev_inv_and_muk
        # prev_MU_k = MU_k[:-1]
        new_row = U_k[sampling_set[-1]]
        # note that the changeover point is different to calc_old_inverse
        if sampling_set.shape[0] <= bandwidth:
            loss = update_xxt_inv_trace_diff(prev_inv, prev_MU_k, new_row)
        else:
            loss = update_xtx_inv_trace_diff(prev_inv, new_row)
        return loss, ()

    sampling_set, _ = greedy_sampling(
        graph,
        num_nodes,
        sample_fn=sample_fn,
        calc_once_a_loop=calc_once_a_loop,
        max_samples=max_samples,
        bandwidth=bandwidth,
        internal_dtype=th.float64,
        **kwargs,
    )
    return sampling_set


def greedy_puy_samples_only(
    graph, num_nodes, bandwidth, max_samples=None, normalization="sym", **kwargs
):
    if max_samples is None:
        max_samples = bandwidth
    U = calc_eigenbasis(graph, num_nodes, normalization=normalization, double=True)
    U_k = restrict_eigenbasis(U, bandwidth)
    weights = th.diag(U_k @ U_k.T)
    weights = weights / th.sum(weights)
    return th.multinomial(weights, num_samples=max_samples, replacement=False)


def greedy_e_samples_only(graph, num_nodes, bandwidth, max_samples=None, **kwargs):
    def sample_fn(_, L, U_k, sampling_set):
        MU_k = U_k[sampling_set]
        svds = th.linalg.svdvals(MU_k)
        loss = -1 * svds.min().item()
        return loss, svds

    sampling_set, _svds = greedy_sampling(
        graph,
        num_nodes,
        sample_fn=sample_fn,
        max_samples=max_samples,
        bandwidth=bandwidth,
        **kwargs,
    )
    return sampling_set


# internally uses numpy,
def greedy_e_samples_only_fast_incorrect(
    graph, num_nodes, bandwidth, max_samples=None, **kwargs
):
    if max_samples is None:
        max_samples = bandwidth

    def calc_once_a_loop(_, U_k, prev_sampling_set):
        # the mu is captured from outer scope
        old_MUk = U_k[prev_sampling_set]
        princ = old_MUk @ old_MUk.T
        D, V = np.linalg.eigh(princ)
        return (D, V)

    # figure out how to update
    def sample_fn(prev_eigensystem, L, _, sampling_set):
        N, _ = L.shape
        D, V = prev_eigensystem
        new_sample = sampling_set[-1]
        # we spew out update_smallest_eigenvalueh to avoid
        # doing v.T @ u/constructing the onehot
        # This is a 2x speedup
        VTu = V[new_sample]
        new_smallest_eval, INFO = Incr.dlaed4_ex(D, VTu, rho=1.0, idx=1)
        if INFO > 0 or np.isnan(new_smallest_eval):
            u = one_hot(new_sample, N).numpy().astype("double")
            new_smallest_eval = Incr.slow_update_smallest_eigenvalueh(V, D, u, rho=1.0)
        loss = -1 * new_smallest_eval
        return loss, ()

    sampling_set, _ = greedy_sampling(
        graph,
        num_nodes,
        sample_fn=sample_fn,
        max_samples=max_samples,
        bandwidth=bandwidth,
        calc_once_a_loop=calc_once_a_loop,
        internal_dtype=th.float64,
        **kwargs,
    )
    return sampling_set


# this is more correct than greedy_d_optimal_sampling(U_k)
# consider using the matrix determinant lemma
def greedy_d_samples_only(graph, num_nodes, bandwidth, max_samples=None, **kwargs):
    if max_samples is None:
        max_samples = bandwidth

    def sample_fn(_, L, U_k, sampling_set):
        MU_k = U_k[sampling_set]
        if len(sampling_set) <= bandwidth:
            muu = MU_k @ MU_k.T
        else:
            muu = MU_k.T @ MU_k
        logdet = th.slogdet(muu).logabsdet.item()
        loss = -1 * logdet
        return loss, logdet

    sampling_set, logdets = greedy_sampling(
        graph,
        num_nodes,
        sample_fn=sample_fn,
        max_samples=max_samples,
        bandwidth=bandwidth,
        **kwargs,
    )
    return sampling_set


def greedy_d_samples_only_fast_incorrect(graph, num_nodes, bandwidth, max_samples=None):
    if max_samples is None:
        max_samples = bandwidth

    def calc_once_a_loop(L, U_k, old_sampling_set):
        MU_k = U_k[old_sampling_set]
        # think harder about why we switch
        if len(old_sampling_set) < bandwidth:
            # muu = MU_k @ MU_k.T
            muu = th.eye(1)
        else:
            muu = MU_k.T @ MU_k
        return muu.inverse().contiguous()

    def sample_fn(prev_inv, L, U_k, sampling_set):
        MU_k = U_k[sampling_set].contiguous()
        # does det(muu) make sense for |S| > k?
        # I don't think so. :/
        if len(sampling_set) <= bandwidth:
            muu = MU_k @ MU_k.T
            logdet = th.slogdet(muu).logabsdet.item()
            loss = -1 * logdet
        else:
            # muu = MU_k.T @ MU_k
            new_row = U_k[sampling_set[-1]]
            loss = -1 * (prev_inv @ new_row).dot(new_row)
        return loss, ()

    sampling_set, _ = greedy_sampling(
        graph,
        num_nodes,
        sample_fn=sample_fn,
        calc_once_a_loop=calc_once_a_loop,
        max_samples=max_samples,
        bandwidth=bandwidth,
        internal_dtype=th.float64,
    )
    return sampling_set


def greedy_glr_samples_only(
    graph, num_nodes, bandwidth, max_samples=None, mu=0.01, **kwargs
):
    if max_samples is None:
        max_samples = bandwidth

    def sample_fn(_, L, U_k, sampling_set):
        err_sig, err_noise = analytic_glr_err(L, U_k, mu, sample_set=sampling_set)
        loss = err_noise
        return loss, (err_sig, err_noise)

    sampling_set, _ = greedy_sampling(
        graph,
        num_nodes,
        sample_fn=sample_fn,
        max_samples=max_samples,
        bandwidth=bandwidth,
        **kwargs,
    )
    return sampling_set


# actually MMSE unlike greedy_glr_samples_only
def greedy_glr_samples_only_MMSE(
    graph,
    num_nodes,
    bandwidth,
    max_samples=None,
    mu=0.01,
    SNR=10.0,
    bandlimited_noise=False,
    **kwargs,
):
    if bandlimited_noise:
        sigma = 1 / SNR
    else:
        sigma = float(bandwidth) / (float(num_nodes) * SNR)
    if max_samples is None:
        max_samples = bandwidth

    def sample_fn(_, L, U_k, new_sampling_set):
        err_sig, err_noise = analytic_glr_err(
            L, U_k, mu, sample_set=new_sampling_set, bandlimited_noise=True
        )
        loss = err_sig + sigma * err_noise
        return loss, (err_sig, err_noise)

    sampling_set, _ = greedy_sampling(
        graph,
        num_nodes,
        sample_fn=sample_fn,
        max_samples=max_samples,
        bandwidth=bandwidth,
        **kwargs,
    )
    return sampling_set


# Also MMSE!
def greedy_glr_samples_only_fast(
    graph, num_nodes, bandwidth, max_samples=None, mu=0.01, SNR=10.0, **kwargs
):
    sigma = float(bandwidth) / (float(num_nodes) * SNR)
    if max_samples is None:
        max_samples = bandwidth

    def calc_once_a_loop(l, _, prev_sampling_set):
        # the mu is captured from outer scope
        adjustedl = (mu * l).clone()
        adjustedl[prev_sampling_set, prev_sampling_set] += 1.0
        current_inv = adjustedl.inverse().contiguous()
        return current_inv

    def sample_fn(prev_inv, L, U_k, new_sampling_set):
        if new_sampling_set.shape[0] == 1:
            err_sig, err_noise = analytic_glr_err(
                L, U_k, mu, sample_set=new_sampling_set
            )
        else:
            err_sig, err_noise = analytic_glr_err_update_fast(
                prev_inv, U_k, new_sampling_set
            )
        loss = err_sig + sigma * err_noise
        return loss, (err_sig, err_noise)

    sampling_set, _ = greedy_sampling(
        graph,
        num_nodes,
        sample_fn=sample_fn,
        calc_once_a_loop=calc_once_a_loop,
        max_samples=max_samples,
        bandwidth=bandwidth,
        **kwargs,
    )
    return sampling_set


def greedy_glr_samples_only_fast_numpy(
    graph, num_nodes, bandwidth, max_samples=None, mu=0.01, SNR=10.0, **kwargs
):
    sigma = float(bandwidth) / (float(num_nodes) * SNR)
    if max_samples is None:
        max_samples = bandwidth

    @njit
    def calc_once_a_loop(l, _, prev_sampling_set):
        # the mu is captured from outer scope
        if len(prev_sampling_set):
            adjustedl = np.copy(mu * l)
            for s in prev_sampling_set:
                adjustedl[s, s] += 1.0
            current_inv = np.linalg.inv(adjustedl)
            return current_inv.astype(l.dtype)
        else:
            return np.asfortranarray(np.zeros_like(l))

    @njit
    def sample_fn(prev_inv, L, U_k, new_sampling_set):
        if new_sampling_set.shape[0] == 1:
            err_sig, err_noise = analytic_glr_err_numpy(
                L, U_k, mu, sample_set=new_sampling_set
            )
        else:
            err_sig, err_noise = analytic_glr_err_update_fast_numpy(
                prev_inv, U_k, new_sampling_set
            )
        loss = err_sig + sigma * err_noise
        return loss, (err_sig, err_noise)

    sampling_set, _ = greedy_sampling_numpy(
        graph,
        num_nodes,
        sample_fn=sample_fn,
        calc_once_a_loop=calc_once_a_loop,
        max_samples=max_samples,
        bandwidth=bandwidth,
        **kwargs,
    )
    return sampling_set


# is somewhat inaccurate without using float64
def greedy_glr_e_samples_only(
    graph, num_nodes, bandwidth, mu=0.01, max_samples=None, **kwargs
):
    if max_samples is None:
        max_samples = bandwidth

    def sample_fn(_, L, U_k, sampling_set):
        adjustedL = (mu * L).clone()
        adjustedL[sampling_set, sampling_set] += 1.0
        eig_vals = th.linalg.eigvalsh(adjustedL)
        loss = -1 * eig_vals.min().item()
        return loss, ()

    sampling_set, _ = greedy_sampling(
        graph,
        num_nodes,
        sample_fn=sample_fn,
        max_samples=max_samples,
        bandwidth=bandwidth,
        # internal_dtype=th.float64,
        **kwargs,
    )
    return sampling_set


# internally uses numpy,
def greedy_glr_e_samples_only_fast(
    graph, num_nodes, bandwidth, mu=0.01, max_samples=None, **kwargs
):
    if max_samples is None:
        max_samples = bandwidth

    def calc_once_a_loop(l, _, prev_sampling_set):
        # the mu is captured from outer scope
        adjustedl = (mu * l).clone()
        adjustedl[prev_sampling_set, prev_sampling_set] += 1.0
        adjustedl = adjustedl.numpy()
        D, V = np.linalg.eigh(adjustedl)
        return (D, V)

    def sample_fn(prev_eigensystem, L, _, sampling_set):
        N, _ = L.shape
        D, V = prev_eigensystem
        new_sample = sampling_set[-1]
        # we spew out update_smallest_eigenvalueh to avoid
        # doing v.T @ u/constructing the onehot
        # This is a 2x speedup
        VTu = V[new_sample]
        new_smallest_eval, INFO = Incr.dlaed4_ex(D, VTu, rho=1.0, idx=1)
        if INFO > 0 or np.isnan(new_smallest_eval):
            u = one_hot(new_sample, N).numpy().astype("double")
            new_smallest_eval = Incr.slow_update_smallest_eigenvalueh(V, D, u, rho=1.0)
        loss = -1 * new_smallest_eval
        return loss, ()

    sampling_set, _ = greedy_sampling(
        graph,
        num_nodes,
        sample_fn=sample_fn,
        max_samples=max_samples,
        bandwidth=bandwidth,
        calc_once_a_loop=calc_once_a_loop,
        internal_dtype=th.float64,
        **kwargs,
    )
    return sampling_set


def greedy_a_optimal_sampling(U_k, max_samples=None):
    N = U_k.shape[0]
    k = U_k.shape[1]
    errs = []
    sampling_set = []
    if max_samples is None:
        max_samples = k
    for _ in tqdm(range(max_samples)):
        # new_sampling_sets = th.tensor([sampling_set + [x] for x in range(N)])
        # MU_ks = U_k[new_sampling_sets]
        # svds = th.linalg.svdvals(MU_ks)
        # inner_errs = svds.reciprocal().square().sum(dim=1).numpy()
        inner_errs = []
        for s in range(N):
            if s in sampling_set:
                inner_errs.append(np.inf)
            else:
                MU_k = U_k[th.tensor(sampling_set + [s])]
                svds = th.linalg.svdvals(MU_k)
                inner_errs.append(svds.reciprocal().square().sum().item())
        new_addition = np.argmin(inner_errs)
        errs.append(inner_errs[new_addition])
        sampling_set.append(new_addition)
    SNRs = np.diff(errs) * float(k) / float(N)
    snr_dbs = 10 * np.log10(SNRs)
    return {
        "sampling_set": sampling_set,
        "errs": errs,
        "snr_dbs": snr_dbs,
    }


# @th.compile(dynamic=True)
def greedy_calc_errs_LS(U_k, sampling_set, decibels=True):
    """For a sampling set of size |S|, provides |S| thresholds.
    If the SNR is below the i^{th} threshold, one should on average
    not observe the i^{th} sample."""
    print("Calculating Errors...")
    N, k = U_k.shape
    errs = [0.0]  # ξ_2({}) = 0.0
    # for i in tqdm(range(len(sampling_set))):
    for i in range(len(sampling_set)):
        ss = sampling_set[: i + 1]
        MU_k = U_k[ss]
        svds = th.linalg.svdvals(MU_k)
        svds = svds[svds.abs() > 1e-10]
        errs.append(svds.reciprocal().square().sum().item())
    SNRs = np.diff(errs) * float(k) / float(N)
    snr_dbs = 10 * np.log10(SNRs)
    return snr_dbs if decibels else SNRs
    # return {
    #     "sampling_set": sampling_set,
    #     "errs": errs,
    #     "snr_dbs": snr_dbs,
    # }


def greedy_calc_errs_GLR(L, U_k, mu, sampling_set):
    print("Calculating Errors...")
    N, k = U_k.shape
    err_sigs = []  # delta_1s
    err_noises = []  # delta_2s
    for i in tqdm(range(len(sampling_set))):
        ss = sampling_set[: i + 1]
        err_sig, err_noise = analytic_glr_err(L, U_k, mu, ss)
        err_sigs.append(err_sig)
        err_noises.append(err_noise)
    delta1s = np.diff(err_sigs)
    delta2s = np.diff(err_noises)
    SNRs = (delta2s / delta1s) * float(k) / float(N)
    snr_dbs = 10 * np.log10(SNRs)
    return snr_dbs


# requires arguments be of type double
def greedy_calc_errs_GLR_fast(L, U_k, mu, sampling_set):
    print("Calculating Errors...")
    # regardless of input, we need double precision for the intermediate stages
    # otherwise errors balloon.
    L = L.double()
    U_k = U_k.double()
    N, k = U_k.shape
    err_sigs = []  # delta_1s
    err_noises = []  # delta_2s

    # we can't use update for the first step as L is singular
    first_sample = sampling_set[0]
    old_inv = (mu * L).clone()
    old_inv[first_sample, first_sample] += 1.0  # (M^T M + mu * L)
    old_inv = old_inv.inverse()

    rec0 = old_inv[:, [first_sample]]
    err_noises.append(rec0.square().sum())
    # err_sig_sqrt = U_k - (rec @ U_k[sample_set])
    err_sigs.append(
        th.addmm(U_k, rec0, U_k[[first_sample], :], alpha=-1.0).square().sum()
    )

    # now, some precomputations: around 15% of time is here
    err_args = {
        "old_inv": old_inv,
        "U_k": U_k,
        "old_inv_sq": old_inv @ old_inv,
        "old_inv_T_U_k": old_inv @ U_k,
        "U_k_U_k_T": U_k @ U_k.T,
    }

    for i in tqdm(range(1, len(sampling_set))):
        all_samples = sampling_set[: i + 1]
        new_samples = sampling_set[1 : i + 1]
        err_sig, err_noise = analytic_glr_err_update_multiple(
            **err_args,
            new_samples=new_samples,
            all_samples=all_samples,
        )
        err_sigs.append(err_sig)
        err_noises.append(err_noise)
    delta1s = np.diff(err_sigs)
    delta2s = np.diff(err_noises)
    SNRs = (delta2s / delta1s) * float(k) / float(N)
    snr_dbs = 10 * np.log10(SNRs)
    return snr_dbs


def update_inv(A_inv, u, v):
    # https://timvieira.github.io/blog/post/2021/03/25/fast-rank-one-updates-to-matrix-inverse/
    Bu = A_inv @ u
    vTB = v @ A_inv
    # return A_inv - (th.outer(Bu, vTB) / (1 + v.dot(Bu)))
    return A_inv.addr(Bu, vTB, alpha=-1.0 / (1 + v.dot(Bu)))


# specialising this to use indexing instead of one hot vectors slows it down massively
def update_inv_sym_uut(A_inv, u):
    # assumes A_inv is symmetric, and calculates
    # (a + uu^T)^-1
    Bu = A_inv @ u
    return th.addr(A_inv, Bu, Bu, alpha=-1.0 / (1 + u.dot(Bu)))


def update_inv_sym_uut_vmappable(A_inv, u):
    # assumes A_inv is symmetric, and calculates
    # (a + uu^T)^-1
    Bu = A_inv @ u
    Bu_scaled = Bu / (1 + u.dot(Bu))
    return th.addr(A_inv, Bu, Bu_scaled, alpha=-1.0)  # this version works with vmap


# given (X^T X)^-1 and a new row to add to X,
# compute the change in tr((X^T X)^-1)
def update_xtx_inv_trace_diff(xtx_inv, new_row):
    Br = xtx_inv @ new_row
    return -1 * Br.dot(Br) / (1 + Br.dot(new_row))


# given (XX^T)^-1 and a new row to add to X
# compute the new (XX^T)^-1 which is larger.
# we assume invertibility.
def update_xxt_inv(xxt_inv, old_x, new_row):
    n = xxt_inv.shape[0]
    # embedded old inverse is the result if new_row is
    # [0,0,...,0,r] for some r
    embedded_old_inv = th.zeros(n + 1, n + 1, dtype=xxt_inv.dtype)
    embedded_old_inv[:n, :n] = xxt_inv
    embedded_old_inv[-1, -1] = new_row.dot(new_row).reciprocal()
    # we now do two sherman-morrison rank one updates to get from
    # this inverse to the actual result. the updates are
    # (A + st^T + uv^T)
    # s,t = r0, new_basis_vec and u,v = new_basis_vec, r0
    r0 = th.zeros(n + 1, dtype=xxt_inv.dtype)
    r0[:-1] = old_x @ new_row
    new_basis_vec = one_hot(th.tensor(n), n + 1).type(xxt_inv.dtype)
    return update_inv(
        update_inv(embedded_old_inv, r0, new_basis_vec), new_basis_vec, r0
    )


# given (XX^T)^-1 and a new row to add to X
# compute the new (XX^T)^-1 which is larger.
# we assume invertibility.
def update_xxt_inv2(xxt_inv, old_x, new_row):
    n = xxt_inv.shape[0]
    # embedded old inverse is the result if new_row is
    # [0,0,...,0,r] for some r
    row_scale = new_row.dot(new_row).reciprocal()
    embedded_old_inv = th.zeros(n + 1, n + 1, dtype=xxt_inv.dtype)
    embedded_old_inv[:n, :n] = xxt_inv
    embedded_old_inv[-1, -1] = row_scale
    # we now do two sherman-morrison rank one updates to get from
    # this inverse to the actual result. the updates are
    # (A + st^T + uv^T)
    # s,t = r0, new_basis_vec and u,v = new_basis_vec, r0
    r = old_x @ new_row
    r0 = th.cat([r, th.zeros(1)])
    new_basis_vec = one_hot(th.tensor(n), n + 1).type(xxt_inv.dtype)
    # the following line is the same as update_inv(embedded_old_inv, r0, new_basis_vec)
    embedded_old_inv[:n, n] = -1 * row_scale * (xxt_inv @ r)
    w = xxt_inv @ r
    B_prime = embedded_old_inv
    denom = 1 + (r0.dot(B_prime @ new_basis_vec))
    print(B_prime)
    print(r0)
    print(B_prime @ new_basis_vec)
    print(-1 * row_scale * (xxt_inv @ r))
    my_denom = 1 - (row_scale * r.dot(xxt_inv @ r))
    print(denom)
    print(denom - my_denom)
    print("===")
    print(B_prime @ new_basis_vec)
    print(r0 @ B_prime)
    print((r0 @ B_prime) - r @ B_prime[:-1])
    print("------")
    print(B_prime[:, -1])
    print(th.hstack([-1 * w, th.ones(1)]) * row_scale)
    print("------")
    print(th.trace(B_prime) - (xxt_inv.trace() + row_scale))
    myres = B_prime - th.outer(B_prime[:, -1], r @ B_prime[:-1]) / my_denom
    res = update_inv(B_prime, new_basis_vec, r0)
    print((myres - res).abs().max())
    return res


def update_xxt_inv_trace_diff(xxt_inv, old_x, new_row):
    v = old_x @ new_row
    row_scale = new_row.dot(new_row).reciprocal()
    A_inv_v = xxt_inv @ v
    vMv_scaled = row_scale * v.dot(A_inv_v)

    res = row_scale + row_scale * (A_inv_v.dot(A_inv_v) + vMv_scaled) / (1 - vMv_scaled)
    if res.isnan():
        return np.inf
    else:
        return res


# Calculates pinv(XX^T + YY^T) for real matrices X : n x a,Y: n x b
# Used to
def woodbury_pinv(X, Y):
    pinv = th.linalg.pinv
    n, a = X.shape
    _, b = Y.shape
    Xpinv = pinv(X)  # a x n
    Xpinv_Y = pinv(X) @ Y  # a x b
    Z = (th.eye(n) - (X @ Xpinv)) @ Y  # n x b
    Zpinv = pinv(Z)  # b x n
    IminusZpinv_Z = th.eye(b) - (Zpinv @ Z)  # b x b
    F = th.eye(b) + (
        (IminusZpinv_Z @ Y.T) @ pinv(X @ X.T) @ (Y @ IminusZpinv_Z)
    )  # b x b
    E = th.eye(a) - Xpinv_Y @ IminusZpinv_Z @ F.inverse() @ Xpinv_Y.T  # a x a
    IminusY_Zpinv = th.eye(n) - (Y @ Zpinv)  # n x n
    result = pinv(Z @ Z.T) + IminusY_Zpinv.T @ Xpinv.T @ E @ Xpinv @ IminusY_Zpinv
    return result


# For u : R^n, construct M:  2 x N
# s.t. M M^T = ue_n^T + e_nu^T
# Note: M is complex!
def construct_2_decomp(u):
    u_norm = th.linalg.norm(u)
    uv = u[-1]
    # if uv < u_norm:
    #     raise Exception("invalid u, last entry too small")
    alpha = 1
    signs = th.tensor([1, -1])
    betas = alpha * u_norm * signs
    evals = uv + (u_norm * signs)
    res = th.stack([u, u]) * alpha
    res[:, -1] += betas
    res = (res.T / th.norm(res, dim=1)).T
    return th.sqrt(evals.to(th.cfloat)).unsqueeze(-1) * res.to(th.cfloat)


# uses sherman-morrison.
# Is wrong when compared to svd :(
def greedy_a_optimal_sampling_fast(U_k, max_samples=None):
    U_k = U_k.double().contiguous()
    N = U_k.shape[0]
    k = U_k.shape[1]
    errs = []
    sampling_set = []
    # U_k = U_k.double()
    if max_samples is None:
        max_samples = k
    for i in tqdm(range(max_samples)):
        if i <= k:
            new_sampling_sets = th.tensor([sampling_set + [x] for x in range(N)])
            MU_ks = U_k[new_sampling_sets]
            princs = th.bmm(MU_ks, MU_ks.transpose(2, 1))
            invs, infos = vmap(th.linalg.inv_ex)(princs)
            inner_errs = vmap(th.trace)(invs)
            inner_errs[infos.bool()] = np.inf
            inner_errs[sampling_set] = np.inf
            inner_errs[inner_errs < 0] = np.inf
        else:
            # princs = th.bmm(MU_ks.transpose(2, 1), MU_ks)
            # sherman-morrison updates are so fast!
            poss_samples = list(set(range(N)) - set(sampling_set))
            MU_k_orig = U_k[sampling_set]
            orig_inv = (MU_k_orig.T @ MU_k_orig).inverse()
            calced_errs = th.trace(orig_inv) + vmap(
                lambda r: update_xtx_inv_trace_diff(orig_inv, r)
            )(U_k[poss_samples])
            inner_errs = th.tensor(np.inf).double().repeat(N)
            inner_errs[poss_samples] = calced_errs
        inner_errs = inner_errs.numpy()
        new_addition = np.argmin(inner_errs)
        errs.append(inner_errs[new_addition])
        sampling_set.append(new_addition)
    SNRs = np.diff(errs) * float(k) / float(N)
    snr_dbs = 10 * np.log10(SNRs)
    return {
        "sampling_set": sampling_set,
        "errs": errs,
        "snr_dbs": snr_dbs,
    }


def greedy_e_optimal_sampling(U_k, max_samples=None):
    N = U_k.shape[0]
    k = U_k.shape[1]
    errs = []
    sampling_set = []
    if max_samples is None:
        max_samples = k
    for _ in tqdm(range(max_samples)):
        inner_errs = []
        inner_smallests = []
        for s in range(N):
            if s in sampling_set:
                inner_errs.append(np.inf)
                inner_smallests.append(-np.inf)
            else:
                MU_k = U_k[th.tensor(sampling_set + [s])]
                svds = th.linalg.svdvals(MU_k)
                inner_smallests.append(svds.min().item())
                inner_errs.append(svds.reciprocal().square().sum().item())
        new_addition = np.argmax(inner_smallests)
        errs.append(inner_errs[new_addition])
        sampling_set.append(new_addition)
    SNRs = np.diff(errs) * float(k) / float(N)
    snr_dbs = 10 * np.log10(SNRs)
    ret_dict = {
        "sampling_set": sampling_set,
        "errs": errs,
        "snr_dbs": snr_dbs,
    }
    return ret_dict


def greedy_random_sampling(U_k, max_samples=None):
    N = U_k.shape[0]
    k = U_k.shape[1]
    errs = []
    sampling_set = []
    if max_samples is None:
        max_samples = k
    for _ in tqdm(range(max_samples)):
        inner_errs = []
        inner_smallests = []
        for s in range(N):
            if s in sampling_set:
                inner_errs.append(np.inf)
                inner_smallests.append(-np.inf)
            else:
                MU_k = U_k[th.tensor(sampling_set + [s])]
                svds = th.linalg.svdvals(MU_k)
                inner_smallests.append(th.randn(1).item())
                inner_errs.append(svds.reciprocal().square().sum().item())
        new_addition = np.argmax(inner_smallests)
        errs.append(inner_errs[new_addition])
        sampling_set.append(new_addition)
    SNRs = np.diff(errs) * float(k) / float(N)
    snr_dbs = 10 * np.log10(SNRs)
    return {
        "sampling_set": sampling_set,
        "errs": errs,
        "snr_dbs": snr_dbs,
    }


def greedy_d_optimal_sampling(U_k, max_samples=None):
    N = U_k.shape[0]
    k = U_k.shape[1]
    errs = []
    dets = []
    sampling_set = []
    soovoodoo = []
    if max_samples is None:
        max_samples = k
    for _ in tqdm(range(max_samples)):
        # new_sampling_sets = th.tensor([sampling_set + [x] for x in range(N)])
        # MU_ks = U_k[new_sampling_sets]
        # svds = th.linalg.svdvals(MU_ks)
        # inner_errs = svds.reciprocal().square().sum(dim=1).numpy()
        inner_errs = []
        inner_dets = []
        for s in range(N):
            MU_k = U_k[th.tensor(sampling_set + [s])]
            muu = MU_k @ MU_k.T
            logdet = th.slogdet(muu).logabsdet.item()
            inner_dets.append(logdet)
        new_addition = np.argmax(inner_dets)
        sampling_set.append(new_addition)
        dets.append(inner_dets[new_addition])
        svds = th.linalg.svdvals(U_k[th.tensor(sampling_set)])
        # print((2 * svds.log().sum() - inner_dets[new_addition]).abs())
        soovoodoo.append(svds)
        new_err = svds.reciprocal().square().sum()
        errs.append(new_err)
    SNRs = np.diff(errs) * float(k) / float(N)
    snr_dbs = 10 * np.log10(SNRs)
    return {
        "sampling_set": sampling_set,
        "errs": errs,
        "snr_dbs": snr_dbs,
        "dets": dets,
        "svds": soovoodoo,
    }


# this version analytically chooses the sampling set
# but uses sampling to verify the MSE


def plot_greedy_thresholds(
    graph, num_nodes, k, graph_name=None, U=None, rand_incl=True
):
    if U is None:
        U = calc_eigenbasis(graph, num_nodes)
    U_k = restrict_eigenbasis(U, k)
    # thresholds
    data = {
        "A-optimal": greedy_a_optimal_sampling(U_k)["snr_dbs"],
        "E-optimal": greedy_e_optimal_sampling(U_k)["snr_dbs"],
        "D-optimal": greedy_d_optimal_sampling(U_k)["snr_dbs"],
    }
    if rand_incl:
        rand_snr_dbs = greedy_random_sampling(U_k)["snr_dbs"]
        data["Random"] = rand_snr_dbs
    fig, ax = plt.subplots()
    sns.lineplot(data=data, ax=ax)
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("SNR Threshold (dB)")
    if graph_name is None:
        graph_name = ""
    else:
        graph_name = graph_name + " Graph"
    ax.set_title(f"{graph_name}, {num_nodes} nodes, bandwidth = {k}")
    plt.show()


def plot_greedy_thresholds2(graph, num_nodes, k, graph_name=None, normalization="sym"):
    U = calc_eigenbasis(graph, num_nodes, normalization=normalization)
    U_k = restrict_eigenbasis(U, k)
    a_samples = greedy_a_samples_only_fast(
        graph, num_nodes, k, normalization=normalization
    )
    d_samples = greedy_d_samples_only(graph, num_nodes, k, normalization=normalization)
    e_samples = greedy_e_samples_only(graph, num_nodes, k, normalization=normalization)
    # thresholds
    data = {
        "A-optimal": greedy_calc_errs_LS(U_k, a_samples),
        "D-optimal": greedy_calc_errs_LS(U_k, d_samples),
        "E-optimal": greedy_calc_errs_LS(U_k, e_samples),
    }
    fig, ax = plt.subplots()
    sns.lineplot(data=data, ax=ax)
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("SNR Threshold (dB)")
    if graph_name is None:
        graph_name = ""
    else:
        graph_name = graph_name + " Graph"
    ax.set_title(f"{graph_name}, {num_nodes} nodes, bandwidth = {k}")
    plt.show()


def plot_greedy_thresholds_LS_multiple_graphs(
    graph_constructor,
    bandwidth,
    num_graphs=4,
    graph_name=None,
    normalization="sym",
    output_folder=Path(),
    max_samples=None,
    decibels=True,
    verbosity=5,
):
    dfs = []
    k = bandwidth
    if max_samples is None:
        max_samples = k

    threshold_str = "Threshold (dB)" if decibels else "τ"

    def inner_fn(i):
        graph, num_nodes = graph_constructor()
        U = calc_eigenbasis(graph, num_nodes, normalization=normalization, double=True)
        U_k = restrict_eigenbasis(U, k)
        sampargs = {
            "graph": graph,
            "num_nodes": num_nodes,
            "bandwidth": bandwidth,
            "normalization": normalization,
            "max_samples": max_samples,
        }
        samples_dict = {
            "MMSE": greedy_a_samples_only_fast(**sampargs),
            "Confidence Ellipsoid": greedy_d_samples_only(**sampargs),
            "WMSE": greedy_e_samples_only(**sampargs),
            "Weighted Random": greedy_puy_samples_only(**sampargs),
        }

        def mk_df(criterion, samples):
            return pd.DataFrame(
                data={
                    "Sampling Criterion": criterion,
                    "Sample Size": np.arange(1, max_samples + 1),
                    threshold_str: greedy_calc_errs_LS(U_k, samples, decibels=decibels),
                }
            )

        df = pd.concat(
            [mk_df(criterion, samples) for criterion, samples in samples_dict.items()]
        )
        df["graph_id"] = i
        return df

    # dfs = [inner_fn(i) for i in range(num_graphs)]
    dfs = Parallel(n_jobs=-1, verbose=verbosity)(
        delayed(inner_fn)(i) for i in range(num_graphs)
    )
    total_df = pd.concat(dfs).reset_index()

    total_df = total_df.replace("Confidence Ellipsoid", "Conf. Ellips.")
    total_df = total_df.replace("Weighted Random", "W. Random")
    plt.rcParams["font.weight"] = "normal"
    plt.rcParams["axes.labelweight"] = "normal"
    # plt.rcParams["font.weight"] = "bold"
    # plt.rcParams["axes.labelweight"] = "bold"
    fig, ax = plt.subplots(figsize=(3.5, 2.5))  # width x height in inches
    # fig, ax = plt.subplots(figsize=(7, 5))  # width x height in inches
    sns.set_palette("deep")
    g = sns.lineplot(
        data=total_df[total_df["Sample Size"] <= bandwidth],
        x="Sample Size",
        y=threshold_str,
        # hue="SNR (dB)",
        hue="Sampling Criterion",
        markers=True,
        errorbar=("pi", 90),
        ax=ax,
    )
    g = sns.lineplot(
        data=total_df[total_df["Sample Size"] > bandwidth],
        x="Sample Size",
        y=threshold_str,
        # hue="SNR (dB)",
        hue="Sampling Criterion",
        markers=True,
        errorbar=("pi", 90),
        legend=False,
        ax=ax,
    )
    if not decibels:
        ax.set_yscale("symlog")
        plt.locator_params(axis="y", numticks=6)
        ax.axhline(y=0, color="k")
        # ax.tick_params(axis="both", labelsize=8.0)

    g.legend_.set_title(None)
    plt.legend(fontsize=(8.0))
    # plt.rc("axes", labelsize=8.0)
    num_nodes = graph_constructor()[1]
    # ax.set_title(
    # f"Threshold for {num_nodes} node {graph_name} Graph under greedy sampling (bandwidth = {bandwidth})"
    # )
    output_folder = output_folder.resolve()
    filename_csv = (
        output_folder
        / f"{graph_name}_{num_nodes}_bandwidth_{bandwidth}_thresholds_LS.csv"
    )

    print(str(filename_csv))
    total_df.to_csv(
        filename_csv,
        index=False,
    )
    fig.savefig(filename_csv.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close()


def plot_greedy_thresholds_LS_multiple_graphs_real(
    dataset,
    bandwidth,
    graph_name=None,
    normalization="sym",
    output_folder=Path(),
    max_samples=None,
    decibels=True,
    verbosity=5,
):
    dfs = []
    k = bandwidth
    if max_samples is None:
        max_samples = k

    threshold_str = "Threshold (dB)" if decibels else "τ"
    num_graphs = 1

    def inner_fn(i):
        graph = dataset.edge_index
        num_nodes = dataset.num_nodes
        U = calc_eigenbasis(graph, num_nodes, normalization=normalization, double=True)
        U_k = restrict_eigenbasis(U, k)
        sampargs = {
            "graph": graph,
            "num_nodes": num_nodes,
            "bandwidth": bandwidth,
            "normalization": normalization,
            "max_samples": max_samples,
        }
        samples_dict = {
            "MMSE": greedy_a_samples_only(**sampargs),
            "Confidence Ellipsoid": greedy_d_samples_only(**sampargs),
            "WMSE": greedy_e_samples_only(**sampargs),
            "Weighted Random": greedy_puy_samples_only(**sampargs),
        }

        def mk_df(criterion, samples):
            return pd.DataFrame(
                data={
                    "Sampling Criterion": criterion,
                    "Sample Size": np.arange(1, max_samples + 1),
                    threshold_str: greedy_calc_errs_LS(U_k, samples, decibels=decibels),
                }
            )

        df = pd.concat(
            [mk_df(criterion, samples) for criterion, samples in samples_dict.items()]
        )
        df["graph_id"] = i
        return df

    dfs = [inner_fn(0)]
    # dfs = Parallel(n_jobs=-1, verbose=verbosity)(
    #     delayed(inner_fn)(i) for i in range(num_graphs)
    # )
    total_df = pd.concat(dfs).reset_index()

    total_df = total_df.replace("Confidence Ellipsoid", "Conf. Ellips.")
    total_df = total_df.replace("Weighted Random", "W. Random")
    plt.rcParams["font.weight"] = "normal"
    plt.rcParams["axes.labelweight"] = "normal"
    # plt.rcParams["font.weight"] = "bold"
    # plt.rcParams["axes.labelweight"] = "bold"
    fig, ax = plt.subplots(figsize=(3.5, 2.5))  # width x height in inches
    # fig, ax = plt.subplots(figsize=(7, 5))  # width x height in inches
    sns.set_palette("deep")
    g = sns.lineplot(
        data=total_df[total_df["Sample Size"] <= bandwidth],
        x="Sample Size",
        y=threshold_str,
        # hue="SNR (dB)",
        hue="Sampling Criterion",
        markers=True,
        errorbar=("pi", 90),
        ax=ax,
    )
    g = sns.lineplot(
        data=total_df[total_df["Sample Size"] > bandwidth],
        x="Sample Size",
        y=threshold_str,
        # hue="SNR (dB)",
        hue="Sampling Criterion",
        markers=True,
        errorbar=("pi", 90),
        legend=False,
        ax=ax,
    )
    if not decibels:
        ax.set_yscale("symlog")
        plt.locator_params(axis="y", numticks=6)
        ax.axhline(y=0, color="k")
        # ax.tick_params(axis="both", labelsize=8.0)

    g.legend_.set_title(None)
    plt.legend(fontsize=(8.0))
    # plt.rc("axes", labelsize=8.0)
    num_nodes = dataset.num_nodes
    # ax.set_title(
    # f"Threshold for {num_nodes} node {graph_name} Graph under greedy sampling (bandwidth = {bandwidth})"
    # )
    output_folder = output_folder.resolve()
    filename_csv = (
        output_folder
        / f"{graph_name}_{num_nodes}_bandwidth_{bandwidth}_thresholds_LS.csv"
    )

    print(str(filename_csv))
    total_df.to_csv(
        filename_csv,
        index=False,
    )
    fig.savefig(filename_csv.with_suffix(".png"), bbox_inches="tight", dpi=150)
    fig.savefig(filename_csv.with_suffix(".svg"), bbox_inches="tight")
    plt.close()


def plot_greedy_thresholds_GLR_multiple_graphs(
    graph_constructor,
    bandwidth,
    mus=[0.01],
    num_graphs=4,
    graph_name=None,
    normalization="sym",
    output_folder=Path(),
    decibels=False,  # Currently always false
    verbosity=5,
    parallel=True,
    bl_noise=False,
):
    dfs = []
    k = bandwidth
    threshold_str = "Threshold (dB)" if decibels else "τ_GLR"

    mus = th.tensor(mus)

    def inner_fn(mus, i):
        graph, num_nodes = graph_constructor()
        inner_dtype = th.float32
        L = calc_laplacian(graph, num_nodes, normalization=normalization).type(
            inner_dtype
        )
        # Lpinv = connected_laplacian_pinv(L)
        L_eigs = th.linalg.eigvalsh(L)
        Bmopt = calc_GLR_Bmopt(L)
        thresholds = vmap(
            lambda mu: calc_GLR_threshold(L, L_eigs, Bmopt, k, mu, bl_noise=bl_noise)
        )(mus)
        df = pd.DataFrame(data={"μ": mus.numpy(), threshold_str: thresholds.numpy()})
        df["graph_id"] = i
        return df

    # dfs = [inner_fn(i) for i in range(num_graphs)]
    dfs = Parallel(n_jobs=-1, verbose=verbosity)(
        delayed(inner_fn)(mus, i) for i in range(num_graphs)
    )
    total_df = pd.concat(dfs).reset_index()

    plt.rcParams["font.weight"] = "normal"
    plt.rcParams["axes.labelweight"] = "normal"
    # plt.rcParams["font.weight"] = "bold"
    # plt.rcParams["axes.labelweight"] = "bold"
    fig, ax = plt.subplots(figsize=(3.5, 2.5))  # width x height in inches
    # fig, ax = plt.subplots(figsize=(7, 5))  # width x height in inches
    sns.set_palette("deep")
    g = sns.lineplot(
        data=total_df,
        x="μ",
        y=threshold_str,
        # hue="SNR (dB)",
        # hue="Sampling Criterion",
        markers=False,
        errorbar=("pi", 90),
        ax=ax,
        legend=False,
    )
    if not decibels:
        ax.set_xscale("log")
        # ax.set_yscale("symlog")
        # plt.locator_params(axis="y", numticks=6)
        plt.locator_params(axis="x", numticks=6)
        # ax.tick_params(axis="both", labelsize=8.0)
        ax.axhline(y=0, color="k")

    # g.legend_.set_title(None)
    plt.legend(fontsize=(8.0))
    # plt.rc("axes", labelsize=8.0)
    num_nodes = graph_constructor()[1]
    # ax.set_title(
    # f"Threshold for {num_nodes} node {graph_name} Graph under greedy sampling (bandwidth = {bandwidth})"
    # )
    output_folder = output_folder.resolve()
    noise_name = "bl_noise" if bl_noise else "full_band"
    filename_csv = (
        output_folder
        / f"{graph_name}_{num_nodes}_bandwidth_{bandwidth}_thresholds_GLR_{noise_name}.csv"
    )

    print(str(filename_csv))
    total_df.to_csv(
        filename_csv,
        index=False,
    )
    fig.savefig(filename_csv.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close()


def plot_greedy_thresholds_GLR_multiple_graphs_multiple_sizes(
    graph_constructor,
    graph_sizes,
    bandwidth_divisor,
    mus=[0.01],
    num_graphs=4,
    graph_name=None,
    normalization="sym",
    output_folder=Path(),
    decibels=False,  # Currently always false
    verbosity=5,
    parallel=True,
    bl_noise=False,
):
    dfs = []
    # k = bandwidth

    threshold_str = (
        "Threshold (dB)" if decibels else ("τ_GLR_bl" if bl_noise else "τ_GLR")
    )

    mus = th.tensor(mus)

    def wrapped_graph_constructor(n):
        num_nodes = 0
        while num_nodes != n:
            graph, num_nodes = graph_constructor(n)
        return graph, num_nodes

    def inner_fn(mus, graph_id, graph, num_nodes):
        # graph, num_nodes = graph_constructor(n)
        n = num_nodes
        k = n // bandwidth_divisor
        inner_dtype = th.float32
        L = calc_laplacian(graph, num_nodes, normalization=normalization).type(
            inner_dtype
        )
        # Lpinv = connected_laplacian_pinv(L)
        L_eigs = th.linalg.eigvalsh(L)
        Bmopt, opt_samp_size = calc_GLR_Bmopt_and_sample_size(L)
        r_bl = calc_GLR_r_bl(L, k)
        Bmopt_bl = r_bl * ((n / opt_samp_size) + opt_samp_size - 1)
        Bmopt_bl = min(Bmopt, Bmopt_bl)
        thresholds = vmap(
            lambda mu: calc_GLR_threshold_journal(
                L, L_eigs, Bmopt, Bmopt_bl, k, mu, bl_noise=bl_noise
            )
        )(mus)
        df = pd.DataFrame(
            data={"# Vertices": n, "μ": mus.numpy(), threshold_str: thresholds.numpy()}
        )
        df["graph_id"] = graph_id
        return df

    graphs = [
        (i, *wrapped_graph_constructor(n))
        for i, n in tqdm_iter.product(
            range(num_graphs), graph_sizes, desc="Constructing Graphs"
        )
    ]
    print("Calculating Thresholds...")
    dfs = Parallel(n_jobs=-1, verbose=verbosity)(
        delayed(inner_fn)(mus, graph_id, graph, num_nodes)
        for graph_id, graph, num_nodes in graphs
    )
    # breakpoint()
    total_df = pd.concat(dfs).reset_index()

    plt.rcParams["font.weight"] = "normal"
    plt.rcParams["axes.labelweight"] = "normal"
    # plt.rcParams["font.weight"] = "bold"
    # plt.rcParams["axes.labelweight"] = "bold"
    fig, ax = plt.subplots(figsize=(3.5, 2.5))  # width x height in inches
    # fig, ax = plt.subplots(figsize=(7, 5))  # width x height in inches
    sns.set_palette("deep")
    g = sns.lineplot(
        data=total_df,
        x="μ",
        y=threshold_str,
        hue="# Vertices",
        # hue="SNR (dB)",
        # hue="Sampling Criterion",
        # palette=["Blue", "Orange", "Green", "Red"],
        palette=sns.color_palette("deep")[:4],
        markers=False,
        errorbar=("pi", 90),
        ax=ax,
        # legend=False,
    )
    if not decibels:
        ax.set_xscale("log")
        # ax.set_yscale("symlog")
        # plt.locator_params(axis="y", numticks=6)
        plt.locator_params(axis="x", numticks=6)
        # ax.tick_params(axis="both", labelsize=8.0)
        ax.axhline(y=0, color="k")

    # g.legend_.set_title(None)
    plt.legend(fontsize=(8.0))
    # plt.rc("axes", labelsize=8.0)
    # num_nodes = graph_constructor()[1]
    # ax.set_title(
    # f"Threshold for {num_nodes} node {graph_name} Graph under greedy sampling (bandwidth = {bandwidth})"
    # )
    output_folder = output_folder.resolve()
    noise_name = "bl_noise" if bl_noise else "full_band"
    num_nodes_str = "_".join([str(n) for n in graph_sizes])
    filename_csv = (
        output_folder
        / f"{graph_name}_{num_nodes_str}_bandwidth_div_{bandwidth_divisor}_thresholds_GLR_{noise_name}_num_nodes_{num_nodes_str}.csv"
    )

    print(str(filename_csv))
    total_df.to_csv(
        filename_csv,
        index=False,
    )
    fig.savefig(filename_csv.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close()


def plot_greedy_thresholds_GLR_multiple_graphs_single_plot(
    constructors_and_names,
    bandwidth,
    mus=[0.01],
    num_graphs=4,
    graph_name=None,
    normalization="sym",
    output_folder=Path(),
    decibels=False,  # Currently always false
    verbosity=5,
):
    dfs = []
    k = bandwidth
    threshold_str = "Threshold (dB)" if decibels else "τ_GLR"

    mus = th.tensor(mus)
    dfs = []
    for d in constructors_and_names:

        def inner_fn(mus, i):
            graph, num_nodes = d["con"]()
            inner_dtype = th.float32
            L = calc_laplacian(graph, num_nodes, normalization=normalization).type(
                inner_dtype
            )
            L_eigs = th.linalg.eigvalsh(L)
            thresholds = vmap(lambda mu: calc_GLR_threshold_poopy(L_eigs, k, mu))(mus)
            df = pd.DataFrame(
                data={"μ": mus.numpy(), threshold_str: thresholds.numpy()}
            )
            df["graph_id"] = i
            df["graph_name"] = d["name"]
            return df

        # dfs = [inner_fn(i) for i in range(num_graphs)]
        dfs += Parallel(n_jobs=-1, verbose=verbosity)(
            delayed(inner_fn)(mus, i) for i in range(num_graphs)
        )
    total_df = pd.concat(dfs).reset_index()

    plt.rcParams["font.weight"] = "normal"
    plt.rcParams["axes.labelweight"] = "normal"
    # plt.rcParams["font.weight"] = "bold"
    # plt.rcParams["axes.labelweight"] = "bold"
    fig, ax = plt.subplots(figsize=(3.5, 2.5))  # width x height in inches
    # fig, ax = plt.subplots(figsize=(7, 5))  # width x height in inches
    sns.set_palette("deep")
    g = sns.lineplot(
        data=total_df,
        x="μ",
        y=threshold_str,
        # hue="SNR (dB)",
        # hue="Sampling Criterion",
        hue="graph_name",
        markers=False,
        errorbar=("pi", 90),
        ax=ax,
    )
    if not decibels:
        ax.set_xscale("log")
        # ax.set_yscale("symlog")
        # plt.locator_params(axis="y", numticks=6)
        plt.locator_params(axis="x", numticks=6)
        # ax.tick_params(axis="both", labelsize=8.0)
        ax.axhline(y=0, color="k")

    # g.legend_.set_title(None)
    plt.legend(fontsize=(8.0))
    # plt.rc("axes", labelsize=8.0)
    num_nodes = constructors_and_names[0]["con"]()[1]
    # ax.set_title(
    # f"Threshold for {num_nodes} node {graph_name} Graph under greedy sampling (bandwidth = {bandwidth})"
    # )
    output_folder = output_folder.resolve()
    filename_csv = (
        output_folder
        / f"{graph_name}_{num_nodes}_bandwidth_{bandwidth}_multiple_thresholds_GLR.csv"
    )

    print(str(filename_csv))
    total_df.to_csv(
        filename_csv,
        index=False,
    )
    fig.savefig(filename_csv.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close()


def heatmap_greedy_unregularisedH(
    graph, num_nodes, bandwidth, SNRs, sampling_type="a", U=None, max_samples=None
):
    if U is None:
        U = calc_eigenbasis(graph, num_nodes)
    U_k = restrict_eigenbasis(U, bandwidth)
    sample_dict = {}
    if sampling_type == "a":
        sample_dict = greedy_a_optimal_sampling(U_k, max_samples=max_samples)
    if sampling_type == "e":
        sample_dict = greedy_e_optimal_sampling(U_k, max_samples=max_samples)
    if sampling_type == "d":
        sample_dict = greedy_d_optimal_sampling(U_k, max_samples=max_samples)
    sampling_set = th.tensor(sample_dict["sampling_set"])
    # sum of 1/svd^2,
    err_noise = th.tensor(sample_dict["errs"])
    # these all select non-degenerate options
    # so total err is (bandwidth - sample_size) + (bandwidth/num_nodes) * err_noises / SNR
    SNRs = th.tensor(SNRs)
    sigma_sqs = bandwidth / (num_nodes * SNRs)
    # MSE_noises = th.outer(sigma_sqs, err_noise)
    MSE_sigs = relu((bandwidth - 1) - th.arange(sampling_set.shape[-1]))
    return vmap(lambda s: MSE_sigs + s * err_noise)(sigma_sqs)


# @th.no_grad()
def heatmap_greedy_unregularised_experimentH(
    graph,
    num_nodes,
    bandwidth,
    SNRs,
    sampling_type="a",
    U=None,
    max_samples=None,
    num_signals=1000,
    normalization="sym",
    bandlimit_noise=False,
):
    if max_samples is None:
        max_samples = bandwidth

    if U is None:
        U = calc_eigenbasis(graph, num_nodes, normalization=normalization)
    U_k = restrict_eigenbasis(U, bandwidth).contiguous()

    if 0 in SNRs:
        raise ValueError("SNRs must be nonzero!")
    print("Picking sample set...")
    sampling_args = {
        "graph": graph,
        "num_nodes": num_nodes,
        "bandwidth": bandwidth,
        "max_samples": max_samples,
        "normalization": normalization,
    }
    if sampling_type == "a":
        sample_set = greedy_a_samples_only_fast(**sampling_args)
    elif sampling_type == "e":
        sample_set = greedy_e_samples_only(**sampling_args)
    elif sampling_type == "d":
        sample_set = greedy_d_samples_only(**sampling_args)
    elif sampling_type == "r":
        sample_set = th.randperm(num_nodes)[:max_samples]
    elif sampling_type == "puy":
        sample_set = greedy_puy_samples_only(**sampling_args)
    # sum of 1/svd^2,
    print("Sampling errors...")
    # this is just a nested list comprehension
    with tqdm(total=len(SNRs) * len(sample_set)) as pbar:
        errs = []
        for SNR in tqdm(SNRs):
            inner_errs = []
            for i in range(max_samples):
                inner_errs.append(
                    sampled_reconstruction_error_unregularised_noisy(
                        U_k,
                        sample_set[: i + 1],
                        SNR=SNR,
                        num_signals=num_signals,
                        normalise=False,  # different to 'normalization' above which refers to the laplacian. This refers to signal normalization. We disable it briefly in the journal version to match GLR.
                        bandlimit_noise=bandlimit_noise,
                    )
                )
                pbar.update(1)
            errs.append(th.hstack(inner_errs))
    return th.vstack(errs)


def heatmap_greedy_unregularised_experimentH_real(
    dataset,
    bandwidth,
    SNRs,
    sampling_type="a",
    U=None,
    max_samples=None,
    num_signals=1000,
    normalization="sym",
    bandlimit_noise=False,
    bandlimit_signal=True,
):
    graph = dataset.edge_index
    num_nodes = dataset.num_nodes

    if max_samples is None:
        max_samples = bandwidth
    if U is None:
        U = calc_eigenbasis(graph, num_nodes, normalization=normalization)
    U_k = restrict_eigenbasis(U, bandwidth).contiguous()

    if 0 in SNRs:
        raise ValueError("SNRs must be nonzero!")
    print("Picking sample set...")
    sampling_args = {
        "graph": graph,
        "num_nodes": num_nodes,
        "bandwidth": bandwidth,
        "max_samples": max_samples,
        "normalization": normalization,
    }
    if sampling_type == "a":
        # Instead of samples_only_fast, as it breaks for
        # some reason on weather
        sample_set = greedy_a_samples_only(**sampling_args)
    elif sampling_type == "e":
        sample_set = greedy_e_samples_only(**sampling_args)
    elif sampling_type == "d":
        sample_set = greedy_d_samples_only(**sampling_args)
    elif sampling_type == "r":
        sample_set = th.randperm(num_nodes)[:max_samples]
    elif sampling_type == "puy":
        sample_set = greedy_puy_samples_only(**sampling_args)
    # sum of 1/svd^2,
    if bandlimit_signal:
        print("Bandlimiting real-world signals...")
        provided_signals = U_k @ (U_k.T @ dataset.x)
    else:
        provided_signals = dataset.x
    print("Sampling errors...")
    # this is just a nested list comprehension
    with tqdm(total=len(SNRs) * len(sample_set)) as pbar:
        errs = []
        for SNR in tqdm(SNRs):
            inner_errs = []
            for i in range(max_samples):
                inner_errs.append(
                    sampled_reconstruction_error_unregularised_noisy_real(
                        U_k,
                        sample_set[: i + 1],
                        provided_signals=provided_signals,
                        SNR=SNR,
                        num_signals=num_signals,
                        normalise=False,  # different to 'normalization' above which refers to the laplacian. This refers to signal normalization. We disable it briefly in the journal version to match GLR.
                        bandlimit_noise=bandlimit_noise,
                    )
                )
                pbar.update(1)
            errs.append(th.hstack(inner_errs))
    return th.vstack(errs)


@th.no_grad()
def heatmap_greedy_GLR_experimentH(
    graph,
    num_nodes,
    bandwidth,
    SNRs,
    mu,
    sampling_type="a",
    U=None,
    max_samples=None,
    num_signals=200,
    normalization="sym",
):
    if max_samples is None:
        max_samples = bandwidth

    L = calc_laplacian(graph, num_nodes, normalization=normalization).float()
    if U is None:
        U = calc_eigenbasis(graph, num_nodes, normalization=normalization).float()
    U_k = restrict_eigenbasis(U, bandwidth).contiguous()

    if 0 in SNRs:
        raise ValueError("SNRs must be nonzero!")
    print("Picking sample set...")
    sampargs = {
        "graph": graph,
        "num_nodes": num_nodes,
        "bandwidth": bandwidth,
        "mu": mu,
        "max_samples": max_samples,
        "normalization": normalization,
    }
    if sampling_type == "a":
        sample_set = greedy_glr_samples_only_fast(**sampargs)
    elif sampling_type == "e":
        sample_set = greedy_glr_e_samples_only_fast(**sampargs)
    elif sampling_type == "r":
        sample_set = th.randperm(num_nodes)[:max_samples]

    sample_set = th.tensor(sample_set)
    # sum of 1/svd^2,
    print("Sampling errors...")
    # this is just a nested list comprehension
    with tqdm(total=len(SNRs) * len(sample_set)) as pbar:
        errs = []
        for SNR in tqdm(SNRs):
            inner_errs = []
            for i in range(max_samples):
                inner_errs.append(
                    sampled_reconstruction_error_glr_noisy(
                        L,
                        U_k,
                        sample_set[: i + 1],
                        SNR=SNR,
                        num_signals=num_signals,
                        normalise=True,
                        mu=mu,
                    )
                )
                pbar.update(1)
            errs.append(th.hstack(inner_errs))
    return th.vstack(errs)


@th.no_grad()
def heatmap_greedy_GLR_experimentH_single_SNR(
    graph,
    num_nodes,
    bandwidth,
    SNR,
    mu,
    sampling_type="a",
    U=None,
    max_samples=None,
    num_signals=200,
    normalization=None,
    bandlimited_noise=False,
):
    if max_samples is None:
        max_samples = bandwidth

    L = calc_laplacian(graph, num_nodes, normalization=normalization).float()
    if U is None:
        U = calc_eigenbasis(graph, num_nodes, normalization=normalization).float()
    U_k = restrict_eigenbasis(U, bandwidth).contiguous()

    if SNR == 0:
        raise ValueError("SNR must be nonzero!")
    print("Picking sample set...")
    # joblib can no longer handle dicts of tensors, so we manually pass in 'graph'
    sampargs = {
        # "graph": graph,
        "num_nodes": num_nodes,
        "bandwidth": bandwidth,
        "mu": mu,
        "max_samples": max_samples,
        "normalization": normalization,
    }
    if sampling_type == "a":
        if bandlimited_noise:
            sample_set = greedy_glr_samples_only_MMSE(
                graph=graph, **sampargs, bandlimited_noise=True, SNR=SNR
            )
        else:
            sample_set = greedy_glr_samples_only_fast(graph=graph, **sampargs, SNR=SNR)
    elif sampling_type == "e":
        sample_set = greedy_glr_e_samples_only_fast(graph=graph, **sampargs)
    elif sampling_type == "r":
        sample_set = th.randperm(num_nodes)[:max_samples]
    else:
        raise ValueError("sampling_type should be a,e or r for GLR")

    if not type(sample_set) == th.Tensor:
        sample_set = th.tensor(sample_set)
    # sum of 1/svd^2,
    print("Sampling errors...")
    # this is just a nested list comprehension
    errs = [
        sampled_reconstruction_error_glr_noisy(
            L,
            U_k,
            sample_set[: i + 1],
            SNR=SNR,
            num_signals=num_signals,
            normalise=False,
            mu=mu,
            bandlimited_noise=bandlimited_noise,
        )
        for i in tqdm(range(max_samples))
    ]
    return th.hstack(errs)


@th.no_grad()
def heatmap_greedy_GLR_experimentH_single_SNR_real(
    dataset,
    bandwidth,
    SNR,
    mu,
    sampling_type="a",
    U=None,
    max_samples=None,
    num_signals=200,
    normalization=None,
    bandlimited_noise=False,
    bandlimited_signal=True,
):
    graph = dataset.edge_index
    num_nodes = dataset.num_nodes
    if max_samples is None:
        max_samples = bandwidth

    L = calc_laplacian(graph, num_nodes, normalization=normalization).float()
    if U is None:
        U = calc_eigenbasis(graph, num_nodes, normalization=normalization).float()
    U_k = restrict_eigenbasis(U, bandwidth).contiguous()

    if SNR == 0:
        raise ValueError("SNR must be nonzero!")
    print("Picking sample set...")
    sampargs = {
        "graph": graph,
        "num_nodes": num_nodes,
        "bandwidth": bandwidth,
        "mu": mu,
        "max_samples": max_samples,
        "normalization": normalization,
    }
    if sampling_type == "a":
        if bandlimited_noise:
            sample_set = greedy_glr_samples_only_MMSE(
                **sampargs, bandlimited_noise=True, SNR=SNR
            )
        else:
            sample_set = greedy_glr_samples_only_fast(**sampargs, SNR=SNR)
    elif sampling_type == "e":
        sample_set = greedy_glr_e_samples_only_fast(**sampargs)
    elif sampling_type == "r":
        sample_set = th.randperm(num_nodes)[:max_samples]
    else:
        raise ValueError("sampling_type should be a,e or r for GLR")

    sample_set = th.tensor(sample_set)
    # sum of 1/svd^2,
    if bandlimited_signal:
        print("Bandlimiting Real-world signals...")
        provided_signals = U_k @ (U_k.T @ dataset.x)
    else:
        provided_signals = dataset.x
    print("Sampling errors...")
    # this is just a nested list comprehension
    errs = [
        sampled_reconstruction_error_glr_noisy_real(
            L,
            U_k,
            sample_set[: i + 1],
            provided_signals=provided_signals,
            SNR=SNR,
            num_signals=num_signals,
            normalise=False,
            mu=mu,
            bandlimited_noise=bandlimited_noise,
        )
        for i in tqdm(range(max_samples))
    ]
    return th.hstack(errs)


def plot_heatmap_greedy_unregularised(
    graph, num_nodes, bandwidth, SNRs, sampling_type="a", U=None, max_samples=None
):
    hm = heatmap_greedy_unregularised_experimentH(
        graph,
        num_nodes,
        bandwidth,
        SNRs,
        sampling_type=sampling_type,
        U=U,
        max_samples=max_samples,
        num_signals=200,
    )
    snr_dbs = [f"SNR: {str(10 * np.log10(x))}dB" for x in SNRs]
    data = {snr: th.log(vals).numpy() for snr, vals in zip(snr_dbs, hm)}
    fig, ax = plt.subplots()
    # sns.heatmap(log_hm, ax=ax)
    sns.lineplot(data=data, ax=ax)
    ax.set(ylim=(-4, 0))
    ax.set_ylabel("log MSE")
    ax.set_xlabel("Sample size")
    # ax.invert_yaxis()
    plt.show()


def plot_heatmap_greedy_unregularised_multiple_graphs(
    graph_constructor,
    bandwidth,
    SNRs,
    sampling_types=["a", "d", "e", "puy"],
    max_samples=None,
    num_graphs=4,
    normalization="sym",
    graph_type="",
    output_folder=Path(),
    verbosity=5,
    legend=True,
    bandlimit_noise=False,
):
    dfs = []
    if max_samples is None:
        max_samples = bandwidth

    def sampling_type_to_str(typ):
        if typ == "a":
            return "MMSE"
        elif typ == "e":
            return "WMSE"
        elif typ == "d":
            return "Confidence Ellipsoid"
        elif typ == "puy":
            return "Weighted Random"
        else:
            return "???"

    for sampling_type in sampling_types:

        def fn(graph, num_nodes):
            return heatmap_greedy_unregularised_experimentH(
                dataset,
                num_nodes,
                bandwidth,
                SNRs,
                sampling_type=sampling_type,
                U=None,
                max_samples=max_samples,
                num_signals=200,
                normalization=normalization,
                bandlimit_noise=bandlimit_noise,
            )

        graphs = [graph_constructor() for _ in range(num_graphs)]
        if num_graphs == 1:
            hms = [fn(*g) for g in graphs]
        else:
            hms = Parallel(n_jobs=-1, verbose=verbosity)(
                delayed(fn)(graph, num_nodes) for graph, num_nodes in graphs
            )
        for i, hm in enumerate(hms):
            for snr, mse in zip(SNRs, hm):
                snrdb = 10 * np.log10(snr)
                df = pd.DataFrame(
                    {
                        "Sampling Criterion": sampling_type_to_str(sampling_type),
                        "Graph": i,
                        "SNR (dB)": "{:.1f}".format(snrdb),
                        "Sample Size": range(1, 1 + max_samples),
                        "log MSE": np.log(mse),
                    }
                )
                dfs.append(df)
    total_df = pd.concat(dfs).reset_index()

    total_df = total_df.replace("Confidence Ellipsoid", "Conf. Ellips.")
    total_df = total_df.replace("Weighted Random", "W. Random")
    plt.rcParams["font.weight"] = "normal"
    plt.rcParams["axes.labelweight"] = "normal"

    fig, ax = plt.subplots(figsize=(3.5, 2.5))  # width x height in inches
    # sns.heatmap(log_hm, ax=ax)
    # sns.lineplot(data=data, ax=ax)
    sns.set_palette("deep")
    if len(SNRs) > 1:
        g = sns.lineplot(
            data=total_df,
            x="Sample Size",
            y="log MSE",
            hue="SNR (dB)",
            style="Sampling Criterion",
            markers=True,
            # ci="sd",
            errorbar=("pi", 90),
            legend=legend,
        )
    else:
        g = sns.lineplot(
            data=total_df,
            x="Sample Size",
            y="log MSE",
            hue="Sampling Criterion",
            markers=True,
            # ci="sd",
            errorbar=("pi", 90),
            legend=legend,
        )
    g.legend_.set_title(None)
    plt.legend(fontsize=(8.0))
    # plt.ylim(0, 6)

    num_nodes = graph_constructor()[1]
    # ax.set_title(
    #     f"Signal Reconstruction of noisy signals on a {num_nodes} node {graph_type} Graph under greedy sampling (bandwidth = {bandwidth})"
    # )
    snrstr = "_".join(map(lambda x: str(10 * np.log10(x)), SNRs))
    noisestr = "bl" if bandlimit_noise else "fb"
    print(f"SNR dbs: {snrstr}")
    filename_csv = (
        output_folder
        / f"{graph_type}_{num_nodes}_bandwidth_{bandwidth}_SNRdbs_{snrstr}_samps_{max_samples}_{noisestr}_MSE_LS.csv"
    )

    print(str(filename_csv))
    total_df.to_csv(
        filename_csv,
        index=False,
    )
    fig.savefig(filename_csv.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(filename_csv.with_suffix(".svg"), bbox_inches="tight")
    # ax.invert_yaxis()
    # plt.show()
    plt.close()


def plot_heatmap_greedy_unregularised_multiple_graphs_real(
    dataset,
    bandwidth,
    SNRs,
    sampling_types=["a", "d", "e", "puy"],
    max_samples=None,
    num_graphs=4,
    normalization="sym",
    graph_type="",
    output_folder=Path(),
    verbosity=5,
    legend=True,
    bandlimit_noise=False,
    bandlimit_signal=True,
):
    dfs = []
    if max_samples is None:
        max_samples = bandwidth

    def sampling_type_to_str(typ):
        if typ == "a":
            return "MMSE"
        elif typ == "e":
            return "WMSE"
        elif typ == "d":
            return "Confidence Ellipsoid"
        elif typ == "puy":
            return "Weighted Random"
        else:
            return "???"

    for sampling_type in sampling_types:

        def fn(dataset):
            return heatmap_greedy_unregularised_experimentH_real(
                dataset,
                bandwidth,
                SNRs,
                sampling_type=sampling_type,
                U=None,
                max_samples=max_samples,
                num_signals=200,
                normalization=normalization,
                bandlimit_noise=bandlimit_noise,
                bandlimit_signal=bandlimit_signal,
            )

        graphs = [dataset]
        if len(graphs) == 1:
            hms = [fn(g) for g in graphs]
        else:
            hms = Parallel(n_jobs=-1, verbose=verbosity)(
                delayed(fn)(graph, num_nodes) for graph, num_nodes in graphs
            )
        for i, hm in enumerate(hms):
            for snr, mse in zip(SNRs, hm):
                snrdb = 10 * np.log10(snr)
                df = pd.DataFrame(
                    {
                        "Sampling Criterion": sampling_type_to_str(sampling_type),
                        "Graph": i,
                        "SNR (dB)": "{:.1f}".format(snrdb),
                        "Sample Size": range(1, 1 + max_samples),
                        "log MSE": np.log(mse),
                    }
                )
                dfs.append(df)
    total_df = pd.concat(dfs).reset_index()

    total_df = total_df.replace("Confidence Ellipsoid", "Conf. Ellips.")
    total_df = total_df.replace("Weighted Random", "W. Random")
    plt.rcParams["font.weight"] = "normal"
    plt.rcParams["axes.labelweight"] = "normal"

    fig, ax = plt.subplots(figsize=(3.5, 2.5))  # width x height in inches
    # sns.heatmap(log_hm, ax=ax)
    # sns.lineplot(data=data, ax=ax)
    sns.set_palette("deep")
    if len(SNRs) > 1:
        g = sns.lineplot(
            data=total_df,
            x="Sample Size",
            y="log MSE",
            hue="SNR (dB)",
            style="Sampling Criterion",
            markers=True,
            # ci="sd",
            errorbar=("pi", 90),
            legend=legend,
        )
    else:
        g = sns.lineplot(
            data=total_df,
            x="Sample Size",
            y="log MSE",
            hue="Sampling Criterion",
            markers=True,
            # ci="sd",
            errorbar=("pi", 90),
            legend=legend,
        )
    g.legend_.set_title(None)
    plt.legend(fontsize=(8.0))
    # plt.ylim(0, 6)

    num_nodes = dataset.num_nodes
    # ax.set_title(
    #     f"Signal Reconstruction of noisy signals on a {num_nodes} node {graph_type} Graph under greedy sampling (bandwidth = {bandwidth})"
    # )
    snrstr = "_".join(map(lambda x: str(10 * np.log10(x)), SNRs))
    noisestr = "bl" if bandlimit_noise else "fb"
    signalstr = "blsig" if bandlimit_noise else "fbsig"
    print(f"SNR dbs: {snrstr}")
    filename_csv = (
        output_folder
        / f"{graph_type}_{num_nodes}_bandwidth_{bandwidth}_SNRdbs_{snrstr}_samps_{max_samples}_{noisestr}_{signalstr}_MSE_LS.csv"
    )

    print(str(filename_csv))
    total_df.to_csv(
        filename_csv,
        index=False,
    )
    fig.savefig(filename_csv.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(filename_csv.with_suffix(".svg"), bbox_inches="tight")
    # ax.invert_yaxis()
    # plt.show()
    plt.close()


def plot_heatmap_greedy_GLR_multiple_graphs(
    graph_constructor,
    bandwidth,
    SNRs,
    mus,
    sampling_types=["a", "e"],
    max_samples=None,
    num_graphs=4,
    normalization="sym",
    graph_type="",
    output_folder=transfer_output_path / "GLR_MSE",
    legend="auto",
):
    dfs = []
    if max_samples is None:
        max_samples = bandwidth

    def sampling_type_to_str(typ):
        if typ == "a":
            return "MMSE"
        elif typ == "e":
            return "WMSE"
        elif typ == "d":
            return "Confidence Ellipsoid"
        elif typ == "puy":
            return "Weighted Random"
        elif typ == "r":
            return "Uniform Random"
        else:
            return "???"

    for sampling_type in sampling_types:

        def fn(graph, num_nodes, mu):
            return heatmap_greedy_GLR_experimentH(
                graph,
                num_nodes,
                bandwidth,
                SNRs,
                mu=mu,
                sampling_type=sampling_type,
                U=None,
                max_samples=max_samples,
                num_signals=200,
                normalization=normalization,
            )

        for mu in mus:
            graphs = [graph_constructor() for _ in range(num_graphs)]
            if num_graphs == 1:
                hms = [fn(g, n, mu) for g, n in graphs]
            else:
                hms = Parallel(n_jobs=4, verbose=5)(
                    delayed(fn)(graph, num_nodes, mu) for graph, num_nodes in graphs
                )
            for i, hm in enumerate(hms):
                for snr, mse in zip(SNRs, hm):
                    snrdb = 10 * np.log10(snr)
                    df = pd.DataFrame(
                        {
                            "Sampling Criterion": sampling_type_to_str(sampling_type),
                            "Graph": i,
                            "mu": mu,
                            "SNR (dB)": "{:.1f}".format(snrdb),
                            "Sample Size": range(1, 1 + max_samples),
                            "log MSE": np.log(mse),
                        }
                    )
                    dfs.append(df)
    total_df = pd.concat(dfs).reset_index()
    # total_df["Sampling Criterion / SNR (db)"] = (
    #     total_df["Criterion"] + "-optimal / " + total_df["SNR (dB)"]
    # )

    # for i in range(num_graphs):
    #     print(f"##### Onto Graph {i+1} of {num_graphs}")
    #     graph, num_nodes = graph_constructor()
    #     hm = heatmap_greedy_unregularised_experimentH(
    #         graph,
    #         num_nodes,
    #         bandwidth,
    #         SNRs,
    #         sampling_type=sampling_type,
    #         U=U,
    #         max_samples=max_samples,
    #         num_signals=200,
    #     )
    #     hms.append(hm)
    #     for snr, mse in zip(SNRs, hm):
    #         snrdb = 10 * np.log10(snr)
    #         df = pd.DataFrame(
    #             {
    #                 "Graph": i,
    #                 "SNR (dB)": "{:.1f}".format(snrdb),
    #                 "Sample Size": range(1, 1 + max_samples),
    #                 "log MSE": np.log(mse),
    #             }
    #         )
    #         dfs.append(df)
    # total_df = pd.concat(dfs).reset_index()
    # snr_dbs = [f"SNR: {str(10 * np.log10(x))}dB" for x in SNRs]
    # data = {snr: th.log(vals).numpy() for snr, vals in zip(snr_dbs, hm)}
    if False:
        fig, ax = plt.subplots(figsize=(3.5, 2.5))  # width x height in inches
    else:
        fig, ax = plt.subplots()

    # sns.heatmap(log_hm, ax=ax)
    # sns.lineplot(data=data, ax=ax)
    sns.set_palette("deep")
    # snseheatmap(log_hm, ax=ax)
    # sns.lineplot(data=data, ax=ax)
    if len(SNRs) > 1:
        sns.lineplot(
            data=total_df,
            x="Sample Size",
            y="log MSE",
            hue="SNR (dB)",
            style="Sampling Criterion",
            markers=True,
            # ci="sd",
            errorbar=("pi", 90),
            legend=legend,
        )
    elif len(mus) > 1:
        sns.lineplot(
            data=total_df,
            x="Sample Size",
            y="log MSE",
            hue="Sampling Criterion",
            style="mu",
            markers=True,
            ci="sd",
            legend=legend,
        )
    else:
        sns.lineplot(
            data=total_df,
            x="Sample Size",
            y="log MSE",
            hue="Sampling Criterion",
            markers=True,
            # ci="sd",
            errorbar=("pi", 90),
            legend=legend,
        )

    num_nodes = graph_constructor()[1]
    # ax.set_title(
    #     f"Signal Reconstruction of noisy signals on a {num_nodes} node {graph_type} Graph under greedy sampling (bandwidth = {bandwidth})"
    # )
    snrstr = "_".join(map(lambda x: str(10 * np.log10(x)), SNRs))
    mustr = "_".join(map(str, mus))
    print(f"SNR dbs: {snrstr}")
    filename_csv = (
        output_folder
        / f"{graph_type}_{num_nodes}_bandwidth_{bandwidth}_SNRdbs_{snrstr}_samps_{max_samples}_mus_{mustr}_MSE_LS.csv"
    )

    print(str(filename_csv))
    total_df.to_csv(
        filename_csv,
        index=False,
    )
    fig.savefig(filename_csv.with_suffix(".png"), bbox_inches="tight")
    # ax.invert_yaxis()
    # plt.show()
    plt.close()


def plot_greedy_GLR_multiple_graphs_with_bounds_journal(
    graph_constructor,
    bandwidth,
    SNR,
    mus,
    sampling_types=["a", "e"],
    max_samples=None,
    num_graphs=4,
    graph_type="",
    output_folder=transfer_output_path / "GLR_MSE",
    legend="auto",
    debug=True,
    bandlimited_noise=False,
):
    if max_samples is None:
        max_samples = bandwidth

    sample_sizes = th.arange(1, 1 + max_samples)

    def sampling_type_to_str(typ):
        if typ == "a":
            return "MMSE"
        elif typ == "e":
            return "WMSE"
        elif typ == "d":
            return "Confidence Ellipsoid"
        elif typ == "puy":
            return "Weighted Random"
        elif typ == "r":
            return "Uniform Random"
        else:
            return "???"

    def calc_mse(graph, num_nodes, U, sampling_type, mu):
        return heatmap_greedy_GLR_experimentH_single_SNR(
            graph,
            num_nodes,
            bandwidth,
            SNR,
            mu=mu,
            sampling_type=sampling_type,
            U=U,
            max_samples=max_samples,
            num_signals=200,
            normalization=None,
            bandlimited_noise=bandlimited_noise,
        )

    def calc_all_mses(graph, num_nodes):
        U = calc_eigenbasis(graph, num_nodes, normalization=None).float()
        return th.stack(
            [
                th.stack(
                    [
                        calc_mse(graph, num_nodes, U, sampling_type, mu)
                        for mu in tqdm(mus)
                    ]
                )
                for sampling_type in tqdm(sampling_types)
            ]
        )

    def calc_mse_bound(graph, num_nodes):
        N = num_nodes
        L = calc_laplacian(graph, N, normalization=None)
        r_bl = calc_GLR_r_bl(L, bandwidth)

        def xi_2_bnd_fn_bandlimited(m):
            return r_bl * ((N / m) + m - 1)

        if bandlimited_noise:
            xi_2_bnd_fn = xi_2_bnd_fn_bandlimited

        else:
            rho_m = khatri_rao(L)
            r = rho_m[1]

            def xi_2_bnd_fn(m):
                return r * (N / m) + rho_m[m - 1]

        xi_2_bnd = th.tensor([xi_2_bnd_fn(m) for m in sample_sizes])
        # xi_1_bnd = xi_2_bnd + bandwidth
        xi_1_bnd = bandwidth + vmap(xi_2_bnd_fn_bandlimited)(sample_sizes)
        if bandlimited_noise:
            sigma_sq = 1 / SNR
        else:
            sigma_sq = bandwidth / (N * SNR)
        MSE_bound = xi_1_bnd + sigma_sq * xi_2_bnd
        return MSE_bound

    # Now we actually do the computing!
    graphs = [graph_constructor() for _ in range(num_graphs)]
    print("Calculating Bounds...")
    # MSE_bounds dim is num_graphs x num_nodes
    log_MSE_bounds = th.stack([calc_mse_bound(g, n) for g, n in tqdm(graphs)]).log()
    MSE_bound_df = pd.DataFrame(
        {
            "Sample Size": sample_sizes.repeat(len(graphs), 1).flatten(),
            "log MSE bound": log_MSE_bounds.flatten(),
        }
    )
    print("Done calculating bounds, calculating MSEs...")
    n_jobs = num_graphs if num_graphs < 16 else -1
    mses = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(calc_all_mses)(graph, num_nodes) for graph, num_nodes in graphs
    )
    mses = th.stack(mses)
    # mse dim is num_graphs x num_sampling_types x num_mus x num_nodes
    # we make it into a dataframe
    sampling_type_names = map(sampling_type_to_str)(sampling_types)
    col_iterables = [range(len(graphs)), sampling_type_names, mus, sample_sizes.numpy()]
    col_names = ["Graph", "Sampling Criterion", "μ", "Sample Size"]
    mse_df_index = pd.MultiIndex.from_product(col_iterables, names=col_names)
    mse_df = pd.DataFrame({"log MSE": mses.log().flatten()}, index=mse_df_index)
    mse_df = mse_df.reset_index(level=list(range(len(col_names))))
    total_df = mse_df.merge(MSE_bound_df, how="left", on="Sample Size")

    fig, ax = plt.subplots(figsize=(3.5, 2.5))  # width x height in inches

    sns.set_palette("deep")
    if len(mus) > 1:
        g = sns.lineplot(
            data=total_df,
            x="Sample Size",
            y="log MSE",
            hue="Sampling Criterion",
            style="μ",
            markers=True,
            errorbar=("pi", 90),
            legend=legend,
            markersize=7,
            markevery=(max_samples // 10),
            ax=ax,
        )
        g.legend_.set_title(None)
        g = sns.lineplot(
            data=MSE_bound_df,
            x="Sample Size",
            y="log MSE bound",
            # errorbar=("pi", 90),
            ax=ax,
            palette=["pink"],
        )
    else:
        raise NotImplementedError
    h, l = ax.get_legend_handles_labels()
    legend_fontsize = 7.0
    l1 = ax.legend(
        h[1 : len(sampling_types) + 1],
        l[1 : len(sampling_types) + 1],
        loc="upper left",
        # loc="best",
        fontsize=legend_fontsize,
    )
    l2 = ax.legend(
        h[len(sampling_types) + 1 :],
        l[len(sampling_types) + 1 :],
        loc="upper right",
        # loc="best",
        fontsize=legend_fontsize,
    )

    ax.add_artist(l1)  # we need this because the 2nd call to legend() erases the first

    l1.set_title(None)
    # plt.legend(fontsize=(7.0))

    num_nodes = graph_constructor()[1]
    # ax.set_title(
    #     f"Signal Reconstruction of noisy signals on a {num_nodes} node {graph_type} Graph under greedy sampling (bandwidth = {bandwidth})"
    # )
    snrstr = "_".join(map(lambda x: str(10 * np.log10(x)), [SNR]))
    mustr = "_".join(map(str, mus))
    noisestr = "bl_noise" if bandlimited_noise else "full_band"
    print(f"SNR dbs: {snrstr}")
    filename_csv = (
        output_folder
        / f"{graph_type}_{num_nodes}_bandwidth_{bandwidth}_SNRdbs_{snrstr}_samps_{max_samples}_mus_{mustr}_{noisestr}_MSE_GLR.csv"
    )

    print(str(filename_csv))
    total_df.to_csv(
        filename_csv,
        index=False,
    )
    fig.savefig(filename_csv.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(filename_csv.with_suffix(".svg"), bbox_inches="tight")
    # ax.invert_yaxis()
    # plt.show()
    plt.close()


def plot_greedy_GLR_multiple_graphs_with_bounds(
    graph_constructor,
    bandwidth,
    SNR,
    mus,
    sampling_types=["a", "e"],
    max_samples=None,
    num_graphs=4,
    graph_type="",
    output_folder=transfer_output_path / "GLR_MSE",
    legend="auto",
    debug=True,
    bandlimited_noise=False,
):
    dfs = []
    if max_samples is None:
        max_samples = bandwidth

    def sampling_type_to_str(typ):
        if typ == "a":
            return "MMSE"
        elif typ == "e":
            return "WMSE"
        elif typ == "d":
            return "Confidence Ellipsoid"
        elif typ == "puy":
            return "Weighted Random"
        elif typ == "r":
            return "Uniform Random"
        else:
            return "???"

    for sampling_type in sampling_types:

        def fn(graph, num_nodes, mu):
            # return poopy_try_z5(
            return heatmap_greedy_GLR_experimentH_single_SNR(
                graph,
                num_nodes,
                bandwidth,
                SNR,
                mu=mu,
                sampling_type=sampling_type,
                U=None,
                max_samples=max_samples,
                num_signals=200,
                normalization=None,
                bandlimited_noise=bandlimited_noise,
            )

        graphs = [graph_constructor() for _ in range(num_graphs)]
        ### Compute Xi_1 and Xi_2 bounds
        sample_sizes = th.arange(1, 1 + max_samples)
        print("Calculating Bounds...")
        MSE_bounds = []
        for g, N in tqdm(graphs):
            L = calc_laplacian(g, N, normalization=None)
            # Lpinv = connected_laplacian_pinv(L)
            # b = calc_GLR_b(L)
            # r = calc_GLR_r(L)
            # delta = optimize_delta_for_GLR_eps(Lpinv)
            # eps = calc_GLR_epsilon_multiple(delta, range(1, N), Lpinv).max()
            # xi_2_bnd = th.stack(
            #     [((1 + eps) * (N / m)) + (b * m) + (eps - 1) for m in sample_sizes]
            # )
            #
            # khatri_rao(L)[m] = rho(m)
            r_bl = calc_GLR_r_bl(L, bandwidth)

            def xi_2_bnd_fn_bandlimited(m):
                return r_bl * ((N / m) + m - 1)

            if bandlimited_noise:

                def xi_2_bnd_fn(m):
                    return xi_2_bnd_fn_bandlimited(m)

            else:
                rho_m = khatri_rao(L)
                r = rho_m[1]

                def xi_2_bnd_fn(m):
                    return r * (N / m) + rho_m[m - 1]

            # xi_2_bnd = vmap(lambda m: (r * (N / m)) + (b + 1) * m + (r - 1))(
            #     sample_sizes
            # )
            xi_2_bnd = th.tensor([xi_2_bnd_fn(m) for m in sample_sizes])
            # xi_1_bnd = xi_2_bnd + bandwidth
            xi_1_bnd = bandwidth + th.tensor(
                [xi_2_bnd_fn_bandlimited(m) for m in sample_sizes]
            )
            if bandlimited_noise:
                sigma_sq = 1 / SNR
            else:
                sigma_sq = bandwidth / (N * SNR)
            MSE_bound = xi_1_bnd + sigma_sq * xi_2_bnd
            MSE_bounds.append(MSE_bound)

        print("Done calculating bounds, calculating MSEs...")
        for mu in mus:
            if num_graphs == 1:
                mses = [fn(g, n, mu) for g, n in graphs]
            else:
                n_jobs = num_graphs if num_graphs < 16 else -1
                mses = Parallel(n_jobs=n_jobs, verbose=5)(
                    delayed(fn)(graph, num_nodes, mu) for graph, num_nodes in graphs
                )
                # mses = [fn(g, n, mu) for g, n in tqdm(graphs)]
            print("Collating dataframe")
            for i, (mse, bnd) in enumerate(zip(tqdm(mses), MSE_bounds)):
                # snrdb = 10 * np.log10(snr)
                mse = mse.numpy()
                bnd = bnd.numpy()
                acc = [
                    {
                        "Sampling Criterion": sampling_type_to_str(sampling_type),
                        "Graph": i,
                        "Signal Number": j,
                        "μ": mu,
                        "SNR": SNR,
                        "Sample Size": sample_sizes.numpy(),
                        "log MSE": np.log(mse_per_signal),
                        "log MSE bound": np.log(bnd),
                    }
                    for j, mse_per_signal in enumerate(mse.T)
                ]
                df = pd.DataFrame(acc)
                print("exploding df - expanding out")
                explode_tick = time.perf_counter()
                df = df.explode(["Sample Size", "log MSE", "log MSE bound"])
                explode_tock = time.perf_counter()
                print(f"explosiion done! Took {explode_tock-explode_tick}s.")
                dfs.append(df)
    total_df = pd.concat(dfs).reset_index()

    ### DEBUG
    #### Some analytic calculations of MSE
    if debug:
        tmp_analytic_dfs = []
        for i, (g, N) in enumerate(graphs):
            L = calc_laplacian(g, N, normalization=None)
            L_eigs = th.linalg.eigvalsh(L)
            L_eigs[0] = 0
            # z = 1 / 1 + mu * lambda_i
            for mu in mus:
                z = 1.0 / (1 + (mu * L_eigs))
                xi_2_analytic = z.square().sum()
                xi_1_analytic = (1 - z[1 : N // 10]).square().sum()
                sigma_sq = bandwidth / (N * SNR)
                anal_df = pd.DataFrame(
                    {
                        "Graph": i,
                        "μ": mu,
                        "Sample Size": np.arange(1, 1 + max_samples),
                        "log analytic MSE": np.log(
                            (xi_1_analytic + sigma_sq * xi_2_analytic).item()
                        ),
                    }
                )
                tmp_analytic_dfs.append(anal_df)
        analytic_MSE_df = pd.concat(tmp_analytic_dfs).reset_index()
    ### END DEBUG

    if True:
        fig, ax = plt.subplots(figsize=(3.5, 2.5))  # width x height in inches
    else:
        fig, ax = plt.subplots()

    # sns.heatmap(log_hm, ax=ax)
    # sns.lineplot(data=data, ax=ax)
    sns.set_palette("deep")
    # snseheatmap(log_hm, ax=ax)
    # sns.lineplot(data=data, ax=ax)
    if len(mus) > 1:
        breakpoint()
        g = sns.lineplot(
            data=total_df,
            x="Sample Size",
            y="log MSE",
            hue="Sampling Criterion",
            style="μ",
            markers=True,
            errorbar=("pi", 90),
            legend=legend,
            markersize=7,
            markevery=(max_samples // 10),
            ax=ax,
        )
        g.legend_.set_title(None)
        g = sns.lineplot(
            data=total_df,
            x="Sample Size",
            y="log MSE bound",
            errorbar=("pi", 90),
            ax=ax,
            palette=["pink"],
        )
        if debug:
            g = sns.lineplot(
                data=analytic_MSE_df,
                x="Sample Size",
                y="log analytic MSE",
                hue="μ",
                errorbar=("pi", 90),
                ax=ax,
            )
    else:
        g = sns.lineplot(
            data=total_df,
            x="Sample Size",
            y="log MSE",
            hue="Sampling Criterion",
            markers=True,
            markersize=7,
            markevery=(max_samples // 10),
            # ci="sd",
            errorbar=("pi", 90),
            legend=legend,
            ax=ax,
        )
    h, l = ax.get_legend_handles_labels()
    legend_fontsize = 7.0
    l1 = ax.legend(
        h[1 : len(sampling_types) + 1],
        l[1 : len(sampling_types) + 1],
        loc="upper left",
        # loc="best",
        fontsize=legend_fontsize,
    )
    l2 = ax.legend(
        h[len(sampling_types) + 1 :],
        l[len(sampling_types) + 1 :],
        loc="upper right",
        # loc="best",
        fontsize=legend_fontsize,
    )

    ax.add_artist(l1)  # we need this because the 2nd call to legend() erases the first

    l1.set_title(None)
    # plt.legend(fontsize=(7.0))

    num_nodes = graph_constructor()[1]
    # ax.set_title(
    #     f"Signal Reconstruction of noisy signals on a {num_nodes} node {graph_type} Graph under greedy sampling (bandwidth = {bandwidth})"
    # )
    snrstr = "_".join(map(lambda x: str(10 * np.log10(x)), [SNR]))
    mustr = "_".join(map(str, mus))
    noisestr = "bl_noise" if bandlimited_noise else "full_band"
    print(f"SNR dbs: {snrstr}")
    filename_csv = (
        output_folder
        / f"{graph_type}_{num_nodes}_bandwidth_{bandwidth}_SNRdbs_{snrstr}_samps_{max_samples}_mus_{mustr}_{noisestr}_MSE_GLR.csv"
    )

    print(str(filename_csv))
    total_df.to_csv(
        filename_csv,
        index=False,
    )
    fig.savefig(filename_csv.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(filename_csv.with_suffix(".svg"), bbox_inches="tight")
    # ax.invert_yaxis()
    # plt.show()
    plt.close()


def plot_greedy_GLR_multiple_graphs_with_bounds_real(
    dataset,
    bandwidth,
    SNR,
    mus,
    sampling_types=["a", "e"],
    max_samples=None,
    graph_type="",
    output_folder=transfer_output_path / "GLR_MSE",
    legend="auto",
    debug=True,
    bandlimited_noise=False,
    bandlimited_signal=True,
):
    dfs = []
    if max_samples is None:
        max_samples = bandwidth

    def sampling_type_to_str(typ):
        if typ == "a":
            return "MMSE"
        elif typ == "e":
            return "WMSE"
        elif typ == "d":
            return "Confidence Ellipsoid"
        elif typ == "puy":
            return "Weighted Random"
        elif typ == "r":
            return "Uniform Random"
        else:
            return "???"

    for sampling_type in sampling_types:

        def fn(dataset, mu):
            return heatmap_greedy_GLR_experimentH_single_SNR_real(
                dataset,
                bandwidth,
                SNR,
                mu=mu,
                sampling_type=sampling_type,
                U=None,
                max_samples=max_samples,
                num_signals=1000,
                normalization=None,
                bandlimited_noise=bandlimited_noise,
                bandlimited_signal=bandlimited_signal,
            )

        def calc_mse_bound(graph, num_nodes):
            N = num_nodes
            L = calc_laplacian(graph, N, normalization=None)
            r_bl = calc_GLR_r_bl(L, bandwidth)

            def xi_2_bnd_fn_bandlimited(m):
                return r_bl * ((N / m) + m - 1)

            if bandlimited_noise:
                xi_2_bnd_fn = xi_2_bnd_fn_bandlimited

            else:
                rho_m = khatri_rao(L)
                r = rho_m[1]

                def xi_2_bnd_fn(m):
                    return r * (N / m) + rho_m[m - 1]

            xi_2_bnd = th.tensor([xi_2_bnd_fn(m) for m in sample_sizes])
            # xi_1_bnd = xi_2_bnd + bandwidth
            xi_1_bnd = bandwidth + vmap(xi_2_bnd_fn_bandlimited)(sample_sizes)
            if bandlimited_noise:
                sigma_sq = 1 / SNR
            else:
                sigma_sq = bandwidth / (N * SNR)
            MSE_bound = xi_1_bnd + sigma_sq * xi_2_bnd
            return MSE_bound

        graphs = [dataset]
        num_graphs = len(graphs)
        ### Compute Xi_1 and Xi_2 bounds
        sample_sizes = th.arange(1, 1 + max_samples)
        print("Calculating Bounds...")
        MSE_bounds = [
            calc_mse_bound(dataset.edge_index, dataset.num_nodes) for dataset in graphs
        ]
        print("Done calculating bounds, calculating MSEs...")
        for mu in mus:
            # if num_graphs == 1:
            mses = [fn(dataset, mu) for dataset in graphs]
            # else:
            #     n_jobs = num_graphs if num_graphs < 16 else -1
            #     mses = Parallel(n_jobs=n_jobs, verbose=5)(
            #         delayed(fn)(graph, num_nodes, mu) for graph, num_nodes in graphs
            #     )
            for i, (mse, bnd) in enumerate(zip(mses, MSE_bounds)):
                # snrdb = 10 * np.log10(snr)
                df = pd.DataFrame(
                    {
                        "Sampling Criterion": sampling_type_to_str(sampling_type),
                        "Graph": i,
                        "μ": mu,
                        "SNR": SNR,
                        "Sample Size": sample_sizes.numpy(),
                        "log MSE": np.log(mse),
                        "log MSE bound": np.log(bnd),
                    }
                )
                dfs.append(df)
    total_df = pd.concat(dfs).reset_index()

    fig, ax = plt.subplots(figsize=(3.5, 2.5))  # width x height in inches

    sns.set_palette("deep")
    # snseheatmap(log_hm, ax=ax)
    # sns.lineplot(data=data, ax=ax)
    if len(mus) > 1:
        g = sns.lineplot(
            data=total_df,
            x="Sample Size",
            y="log MSE",
            hue="Sampling Criterion",
            style="μ",
            markers=True,
            errorbar=("pi", 90),
            legend=legend,
            markersize=7,
            markevery=(max_samples // 10),
            ax=ax,
        )
        g.legend_.set_title(None)
        g = sns.lineplot(
            data=total_df,
            x="Sample Size",
            y="log MSE bound",
            errorbar=("pi", 90),
            ax=ax,
            palette=["pink"],
        )
        if debug:
            g = sns.lineplot(
                data=analytic_MSE_df,
                x="Sample Size",
                y="log analytic MSE",
                hue="μ",
                errorbar=("pi", 90),
                ax=ax,
            )
    else:
        g = sns.lineplot(
            data=total_df,
            x="Sample Size",
            y="log MSE",
            hue="Sampling Criterion",
            markers=True,
            markersize=7,
            markevery=(max_samples // 10),
            # ci="sd",
            errorbar=("pi", 90),
            legend=legend,
            ax=ax,
        )
    h, l = ax.get_legend_handles_labels()
    legend_fontsize = 7.0
    l1 = ax.legend(
        h[1 : len(sampling_types) + 1],
        l[1 : len(sampling_types) + 1],
        loc="upper left",
        # loc="best",
        fontsize=legend_fontsize,
    )
    l2 = ax.legend(
        h[len(sampling_types) + 1 :],
        l[len(sampling_types) + 1 :],
        loc="upper right",
        # loc="best",
        fontsize=legend_fontsize,
    )

    ax.add_artist(l1)  # we need this because the 2nd call to legend() erases the first

    l1.set_title(None)
    # plt.legend(fontsize=(7.0))

    num_nodes = dataset.num_nodes
    # ax.set_title(
    #     f"Signal Reconstruction of noisy signals on a {num_nodes} node {graph_type} Graph under greedy sampling (bandwidth = {bandwidth})"
    # )
    snrstr = "_".join(map(lambda x: str(10 * np.log10(x)), [SNR]))
    mustr = "_".join(map(str, mus))
    noisestr = "bl_noise" if bandlimited_noise else "full_band"
    signalstr = "bl_signal" if bandlimited_noise else "raw_signal"
    print(f"SNR dbs: {snrstr}")
    filename_csv = (
        output_folder
        / f"{graph_type}_{num_nodes}_bandwidth_{bandwidth}_SNRdbs_{snrstr}_samps_{max_samples}_mus_{mustr}_{noisestr}_{signal_str}_MSE_GLR.csv"
    )

    print(str(filename_csv))
    total_df.to_csv(
        filename_csv,
        index=False,
    )
    fig.savefig(filename_csv.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(filename_csv.with_suffix(".svg"), bbox_inches="tight")
    # ax.invert_yaxis()
    # plt.show()
    plt.close()


def render_MSE_LS(num_nodes=500, bl_noise=False):
    if bl_noise:
        snrs = [10 ** (-1), 1, 10**10]
    else:
        snrs = [10 ** (-1), 10**2, 10**10]
    # snrs = [0.1]
    # snrs = [10**]
    constructors_and_names = [
        {
            "con": lambda: (connected_erdos_renyi_graph(num_nodes, 0.8)),
            "name": "ER_0pt8",
        },
        # {
        #     "con": lambda: clean_graph(erdos_renyi_graph(num_nodes, 0.8)),
        #     "name": "ER_0pt8",
        # },
        # {
        #     "con": lambda: clean_graph(barabasi_albert_graph(num_nodes, 3)),
        #     "name": "BA_3",
        # },
        # # {"con": lambda: clean_graph(circle(num_nodes)), "name": "Ring"},
        # {
        #     "con": lambda: sbm_constructor(
        #         th.tensor(num_nodes // 10).repeat(10), 0.1, 0.7
        #     ),
        #     "name": "SBM_50x10n_0pt1intra_0pt7inter",
        # },
    ]
    output_path = transfer_output_path / "LS_MSE"

    if not output_path.is_dir():
        output_path.mkdir(parents=True)

    for snr in snrs:
        print(" ================== ")
        print(f" SNR: {snr}")
        print(" ================== ")
        tick = time.perf_counter()
        for d in constructors_and_names:
            plot_heatmap_greedy_unregularised_multiple_graphs(
                graph_constructor=d["con"],
                bandwidth=num_nodes // 10,
                SNRs=[snr],
                max_samples=2 * (num_nodes // 10),
                num_graphs=10,
                graph_type=d["name"],
                output_folder=output_path,
                normalization=None,  # to match with GLR in journal version
                verbosity=0,
                legend=True,
                bandlimit_noise=bl_noise,
            )
        tock = time.perf_counter()
        print("+++++++")
        print(f"took {tock-tick}s")
        print("+++++++")


def render_MSE_LS_real(bl_noise=False, bl_signal=True):
    if bl_noise:
        snrs = [10 ** (-1), 1, 10**10]
    else:
        snrs = [10 ** (-1), 10**2, 10**10]
    # snrs = [0.1]
    # snrs = [10**]
    constructors_and_names = [
        {
            "dataset": datasets.fmri_subsample(),
            "name": "fmri_subsample_500",
            "bandwidth": datasets.fmri_subsample().num_nodes // 10,
        },
        {
            "dataset": datasets.weather(),
            "name": "weather",
            "bandwidth": 8,
        },
    ]
    output_path = transfer_output_path / "LS_MSE_real"

    if not output_path.is_dir():
        output_path.mkdir(parents=True)

    for snr in snrs:
        print(" ================== ")
        print(f" SNR: {snr}")
        print(" ================== ")
        tick = time.perf_counter()
        for d in constructors_and_names:
            num_nodes = d["dataset"].num_nodes
            plot_heatmap_greedy_unregularised_multiple_graphs_real(
                dataset=d["dataset"],
                bandwidth=d["bandwidth"],
                SNRs=[snr],
                max_samples=2 * d["bandwidth"],
                num_graphs=10,
                graph_type=d["name"],
                output_folder=output_path,
                normalization=None,  # to match with GLR in journal version
                verbosity=0,
                legend=True,
                bandlimit_noise=bl_noise,
                bandlimit_signal=bl_signal,
            )
        tock = time.perf_counter()
        print("+++++++")
        print(f"took {tock-tick}s")
        print("+++++++")


def render_MSE_GLR(num_nodes=100):
    # snrs = [10**0, 10**2, 10**5]
    # snrs = [0.1]
    # snrs = [10**-5, 1, 10**10]
    snrs = [10**-5]
    constructors_and_names = [
        {
            "con": lambda: clean_graph(erdos_renyi_graph(num_nodes, 0.8)),
            "name": "ER_0pt8",
        },
        {
            "con": lambda: clean_graph(barabasi_albert_graph(num_nodes, 3)),
            "name": "BA_3",
        },
        {"con": lambda: clean_graph(circle(num_nodes)), "name": "Ring"},
        {
            "con": lambda: sbm_constructor(
                th.tensor(num_nodes // 10).repeat(10), 0.1, 0.7
            ),
            "name": "SBM_50x10n_0pt1intra_0pt7inter",
        },
    ]
    for snr in snrs:
        print(" ================== ")
        print(f" SNR: {snr}")
        print(" ================== ")
        tick = time.perf_counter()
        for d in constructors_and_names:
            plot_heatmap_greedy_GLR_multiple_graphs(
                graph_constructor=d["con"],
                bandwidth=num_nodes // 10,
                SNRs=[snr],
                mus=[10**-4, 0.01, 1],
                max_samples=2 * (num_nodes // 10),
                graph_type=d["name"],
                # legend=False,
            )
        tock = time.perf_counter()
        print("+++++++")
        print(f"took {tock-tick}s")
        print("+++++++")


# Currently bounds are only proven for normalization=None, so we assume that.
def render_MSE_GLR_with_bounds(num_nodes=100, bandlimited_noise=False):
    # snrs = [10**0, 10**2, 10**5]
    # snrs = [0.1]
    # snrs = [10**-5, 1, 10**10]
    # snrs = [10**-2, 1, 10**10]
    # snrs = [10**-1]
    # snrs = [10 ** (-1), 10**2, 10**10]
    snrs = [0.1, 0.5, 10**10]
    # snrs = [10**10]
    # snrs = [10**-2, 0.5, 10**10]
    if bandlimited_noise:
        snrs = [10**-2, 0.5, 10**10]
    else:
        # snrs = [0.1, 0.5, 10**10]
        snrs = [0.5, 10**10]
    snrs = [0.1]
    constructors_and_names = [
        {
            "con": lambda: connected_erdos_renyi_graph(num_nodes, 0.8),
            "name": "ER_0pt8",
        },
        # {
        #     "con": lambda: clean_graph(erdos_renyi_graph(num_nodes, 0.8)),
        #     "name": "ER_0pt8",
        # },
        {
            "con": lambda: clean_graph(barabasi_albert_graph(num_nodes, 3)),
            "name": "BA_3",
        },
        # {"con": lambda: clean_graph(circle(num_nodes)), "name": "Ring"},
        {
            "con": lambda: sbm_constructor(
                th.tensor(num_nodes // 10).repeat(10), 0.1, 0.7
            ),
            "name": "SBM_50x10n_0pt1intra_0pt7inter",
        },
    ]

    output_path = transfer_output_path / "GLR_MSE"

    if not output_path.is_dir():
        output_path.mkdir(parents=True)

    for snr in snrs:
        print(" ================== ")
        print(f" SNR: {snr}")
        print(" ================== ")
        tick = time.perf_counter()
        for d in constructors_and_names:
            try:
                plot_greedy_GLR_multiple_graphs_with_bounds_journal(
                    graph_constructor=d["con"],
                    bandwidth=num_nodes // 10,
                    SNR=snr,
                    mus=[10**-i for i in [4, 2, 0]],
                    # max_samples=2 * (num_nodes // 10),
                    max_samples=num_nodes,
                    graph_type=d["name"],
                    sampling_types=["a", "e", "r"],
                    num_graphs=10,
                    # legend=False,
                    output_folder=output_path,
                    debug=False,
                    bandlimited_noise=bandlimited_noise,
                )
            except:
                pass
        tock = time.perf_counter()
        print("+++++++")
        print(f"took {tock-tick}s")
        print("+++++++")


def render_MSE_GLR_with_bounds_real(bandlimited_noise=False, bandlimited_signal=True):
    snrs = [0.1, 0.5, 10**10]
    if bandlimited_noise:
        snrs = [10**-2, 0.5, 10**10]
    else:
        snrs = [0.1, 0.5, 10**10]
    # snrs = [0.1]

    constructors_and_names = [
        {
            "dataset": datasets.fmri_subsample(),
            "name": "fmri_subsample_500",
            "bandwidth": datasets.fmri_subsample(500).num_nodes // 10,
        },
        {
            "dataset": datasets.weather(),
            "name": "weather",
            "bandwidth": 8,
        },
    ]
    output_path = transfer_output_path / "GLR_MSE_real"

    if not output_path.is_dir():
        output_path.mkdir(parents=True)

    for snr in snrs:
        print(" ================== ")
        print(f" SNR: {snr}")
        print(" ================== ")
        tick = time.perf_counter()
        for d in constructors_and_names:
            num_nodes = d["dataset"].num_nodes
            plot_greedy_GLR_multiple_graphs_with_bounds_real(
                dataset=d["dataset"],
                bandwidth=d["bandwidth"],
                SNR=snr,
                mus=[10**-i for i in [4, 2, 0]],
                # max_samples=2 * (num_nodes // 10),
                max_samples=num_nodes,
                graph_type=d["name"],
                sampling_types=["a", "e", "r"],
                # legend=False,
                output_folder=output_path,
                debug=False,
                bandlimited_noise=bandlimited_noise,
                bandlimited_signal=bandlimited_signal,
            )
        tock = time.perf_counter()
        print("+++++++")
        print(f"took {tock-tick}s")
        print("+++++++")


def render_thresholds_GLR(n=1000, bandlimited_noise=False):
    constructors_and_names = [
        {"con": lambda: clean_graph(erdos_renyi_graph(n, 0.8)), "name": "ER_0pt8"},
        # {"con": lambda: clean_graph(erdos_renyi_graph(n, 0.2)), "name": "ER_0pt2"},
        {"con": lambda: clean_graph(barabasi_albert_graph(n, 3)), "name": "BA_3"},
        # {"con": lambda: clean_graph(barabasi_albert_graph(n, 5)), "name": "BA_5"},
        # {"con": lambda: clean_graph(circle(n)), "name": "Ring"},
        {
            "con": lambda: sbm_constructor(th.tensor(n // 10).repeat(10), 0.1, 0.7),
            "name": "SBM_50x10n_0pt1intra_0pt7inter",
        },
    ]
    output_path = transfer_output_path / "GLR_threshold"

    if not output_path.is_dir():
        output_path.mkdir(parents=True)

    for d in constructors_and_names:
        print(" ================== ")
        print(f" Graph: {d['name']}")
        print(" ================== ")
        tick = time.perf_counter()
        plot_greedy_thresholds_GLR_multiple_graphs(
            graph_constructor=d["con"],
            bandwidth=n // 10,
            mus=[10 ** (i / 10) for i in range(-60, 10)],
            num_graphs=10,
            graph_name=d["name"],
            normalization=None,
            output_folder=output_path,
            # output_folder="/Users/baskaran/programming/python/pytorch-geometric/compressed_sensing/tmp/GLR_threshold",
            bl_noise=bandlimited_noise,
        )
        tock = time.perf_counter()
        print("+++++++")
        print(f"took {tock-tick}s")
        print("+++++++")


def render_thresholds_GLR_multiple_sizes(ns=[1000], bandlimited_noise=False):
    constructors_and_names = [
        {"confn": lambda n: clean_graph(erdos_renyi_graph(n, 0.8)), "name": "ER_0pt8"},
        # {"confn": lambda n: (connected_erdos_renyi_graph(n, 0.3)), "name": "ER_0pt3"},
        # {"con": lambda: clean_graph(erdos_renyi_graph(n, 0.2)), "name": "ER_0pt2"},
        {"confn": lambda n: clean_graph(barabasi_albert_graph(n, 3)), "name": "BA_3"},
        # {"con": lambda: clean_graph(barabasi_albert_graph(n, 5)), "name": "BA_5"},
        # {"con": lambda: clean_graph(circle(n)), "name": "Ring"},
        {
            "confn": lambda n: sbm_constructor(th.tensor(n // 10).repeat(10), 0.1, 0.7),
            "name": "SBM_50x10n_0pt1intra_0pt7inter",
        },
    ]
    output_path = transfer_output_path / "GLR_threshold"

    if not output_path.is_dir():
        output_path.mkdir(parents=True)

    for d in constructors_and_names:
        print(" ================== ")
        print(f" Graph: {d['name']}")
        print(" ================== ")
        tick = time.perf_counter()
        plot_greedy_thresholds_GLR_multiple_graphs_multiple_sizes(
            graph_constructor=d["confn"],
            graph_sizes=ns,
            bandwidth_divisor=10,
            mus=[10 ** (i / 10) for i in range(-60, 10)],
            num_graphs=10,
            graph_name=d["name"],
            normalization=None,
            output_folder=output_path,
            # output_folder="/Users/baskaran/programming/python/pytorch-geometric/compressed_sensing/tmp/GLR_threshold",
            bl_noise=bandlimited_noise,
        )
        tock = time.perf_counter()
        print("+++++++")
        print(f"took {tock-tick}s")
        print("+++++++")


def multiple_render_thresholds_GLR_single_plot(n=1000):
    # constructors_and_names = [
    #     {"con": lambda: clean_graph(erdos_renyi_graph(n, 0.9)), "name": "p=0.9"},
    #     {"con": lambda: clean_graph(erdos_renyi_graph(n, 0.7)), "name": "p=0.7"},
    #     {"con": lambda: clean_graph(erdos_renyi_graph(n, 0.5)), "name": "p=0.5"},
    #     {"con": lambda: clean_graph(erdos_renyi_graph(n, 0.2)), "name": "p=0.2"},
    #     # {"con": lambda: clean_graph(barabasi_albert_graph(n, 3)), "name": "BA_3"},
    #     # {"con": lambda: clean_graph(barabasi_albert_graph(n, 5)), "name": "BA_5"},
    #     # {"con": lambda: clean_graph(circle(n)), "name": "Ring"},
    #     # {
    #     #     "con": lambda: sbm_constructor(th.tensor(n // 10).repeat(10), 0.1, 0.7),
    #     #     "name": "SBM_50x10n_0pt1intra_0pt7inter",
    #     # },
    # ]
    # constructors_and_names = [
    #     {"con": lambda: clean_graph(erdos_renyi_graph(n, p)), "name": f"p={p}"}
    #     for p in [0.9, 0.7, 0.5, 0.2]
    # ]
    constructors_and_names = [
        {
            "con": lambda: sbm_constructor(th.tensor(n // 10).repeat(10), 0.1, q),
            "name": f"p=0.1_q={q}",
        }
        for q in [0.9, 0.7, 0.5, 0.2]
    ]
    output_path = transfer_output_path / "GLR_threshold"

    if not output_path.is_dir():
        output_path.mkdir(parents=True)
    plot_greedy_thresholds_GLR_multiple_graphs_single_plot(
        constructors_and_names=constructors_and_names,
        bandwidth=n // 10,
        mus=[10 ** (i / 10) for i in range(-60, 10)],
        num_graphs=16,
        graph_name="Erdos-Renyi",
        normalization=None,
        output_folder=output_path,
        # output_folder="/Users/baskaran/programming/python/pytorch-geometric/compressed_sensing/tmp/GLR_threshold",
    )


def render_thresholds_LS(num_nodes=100):
    constructors_and_names = [
        # {
        #     "con": lambda: clean_graph(grid(floor(sqrt(num_nodes)))),
        #     "name": "Grid",
        # },
        {
            "con": lambda: (connected_erdos_renyi_graph(num_nodes, 0.8)),
            "name": "ER_0pt3",
        },
        # {
        #     "con": lambda: clean_graph(erdos_renyi_graph(num_nodes, 0.8)),
        #     "name": "ER_0pt8",
        # },
        # {
        #     "con": lambda: clean_graph(barabasi_albert_graph(num_nodes, 3)),
        #     "name": "BA_3",
        # },
        # # # {"con": lambda: clean_graph(circle(num_nodes)), "name": "Ring"},
        # {
        #     "con": lambda: sbm_constructor(
        #         th.tensor(num_nodes // 10).repeat(10), 0.1, 0.7
        #     ),
        #     "name": "SBM_50x10n_0pt1intra_0pt7inter",
        # },
    ]
    # constructors_and_names = [
    #     {
    #         "con": lambda: clean_graph(random_k_regular_graph(num_nodes, i)),
    #         "name": f"random_{i}_regular",
    #     }
    #     for i in [2, 3, 4, 5]
    # ]
    output_path = transfer_output_path / "LS_threshold"

    if not output_path.is_dir():
        output_path.mkdir(parents=True)

    for d in constructors_and_names:
        print(" ================== ")
        print(f" Graph: {d['name']}")
        print(" ================== ")
        tick = time.perf_counter()
        plot_greedy_thresholds_LS_multiple_graphs(
            graph_constructor=d["con"],
            bandwidth=num_nodes // 10,
            num_graphs=10,
            graph_name=d["name"],
            normalization=None,
            decibels=False,
            max_samples=2 * (num_nodes // 10),
            output_folder=output_path,
        )
        tock = time.perf_counter()
        print("+++++++")
        print(f"took {tock-tick}s")
        print("+++++++")


def render_thresholds_LS_real():
    constructors_and_names = [
        {
            "dataset": datasets.fmri_subsample(500),
            "name": "fmri_subsample_500",
            "bandwidth": datasets.fmri_subsample(500).num_nodes // 10,
        },
        {
            "dataset": datasets.weather(),
            "name": "weather",
            "bandwidth": 8,
        },
    ]
    output_path = transfer_output_path / "LS_threshold_real"

    if not output_path.is_dir():
        output_path.mkdir(parents=True)

    for d in constructors_and_names:
        print(" ================== ")
        print(f" Graph: {d['name']}")
        print(" ================== ")
        tick = time.perf_counter()
        plot_greedy_thresholds_LS_multiple_graphs_real(
            dataset=d["dataset"],
            bandwidth=d["bandwidth"],
            graph_name=d["name"],
            normalization=None,
            decibels=False,
            max_samples=2 * d["bandwidth"],
            output_folder=output_path,
        )
        tock = time.perf_counter()
        print("+++++++")
        print(f"took {tock-tick}s")
        print("+++++++")


# to be used with laplacian^2
def smallest_eval_vs_det_bounds(A_sqrt, sample_size, num_repeats):
    num_nodes = A_sqrt.shape[0]
    if math.comb(num_nodes, sample_size) < num_repeats:
        M_indices = th.tensor(
            list(itertools.combinations(range(num_nodes), sample_size)), dtype=th.long
        )
        princs_sqrt = A_sqrt[M_indices]
    else:
        distribution = uniform_sampling_distn(A_sqrt)
        Ms = batch_sample_nodes_with_distn(distribution, sample_size, num_repeats)["M"]
        # as everything is linear, we only need calculate the expectation of
        # pinv(MU_k) @ MU_k
        princs_sqrt = th.bmm(Ms, A_sqrt.unsqueeze(0).expand(num_repeats, -1, -1))
    princs = th.bmm(princs_sqrt, princs_sqrt.transpose(2, 1))
    # Looking at principal submatrices of L^2, by cauchy interlacing all dets are nonneg
    logdets = th.linalg.slogdet(princs).logabsdet
    logtraces = vmap(th.trace)(princs).log()
    # lbs = (((sample_size - 1) / traces) ** (sample_size - 1)) * dets
    log_lbs = (
        ((sample_size - 1) * th.tensor(sample_size - 1).log())
        + logdets
        - ((sample_size - 1) * logtraces)
    )
    lbs = log_lbs.exp()
    e_vals = th.linalg.eigvalsh(princs)
    smallest_evals = e_vals.min(dim=1).values
    return lbs, smallest_evals


def sample_analytic_error(U, bandwidth, sample_size, num_repeats=100):
    U_k = restrict_eigenbasis(U, bandwidth)
    num_nodes = U.shape[1]
    # if it's easier to sample exactly, do so
    if math.comb(num_nodes, sample_size) < num_repeats:
        # th.combinations for th.arange(11), r=11, with_replacement=False hangs the computer
        # M_indices = th.combinations(
        #     th.arange(num_nodes), r=sample_size, with_replacement=False
        # )
        M_indices = th.tensor(
            list(itertools.combinations(range(num_nodes), sample_size)), dtype=th.long
        )
        MU_ks = U_k[M_indices]
    else:
        distribution = uniform_sampling_distn(U_k)
        Ms = batch_sample_nodes_with_distn(distribution, sample_size, num_repeats)["M"]
        # as everything is linear, we only need calculate the expectation of
        # pinv(MU_k) @ MU_k
        MU_ks = th.bmm(Ms, U_k.unsqueeze(0).expand(num_repeats, -1, -1))
    # These two should be equivalent - are they?
    # numpy is more stable
    # almost all of the time is spent here
    pinv = th.from_numpy(np.linalg.pinv(MU_ks.numpy()))
    Ps = th.bmm(pinv, MU_ks)
    # Ps = th.bmm(th.linalg.pinv(MU_ks), MU_ks)
    # Ps = th.linalg.lstsq(
    #     MU_ks, MU_ks, driver="gelsd"
    # ).solution  # the projector pinv(A) @ A
    EP = Ps.mean(dim=0)
    I = th.eye(EP.shape[-1])
    Ecov = U_k @ (I - EP) @ U_k.T
    # look it's the same:
    # Ecov2 = th.stack([U_k @ (I - P) @ U_k.T for P in Ps]).mean(dim=0)
    # print((Ecov - Ecov2).abs().max())
    err = th.diag(Ecov)
    err[err.abs() < 1e-10] = 0.0
    return err


# holding bandwidth constant, return multiple reconstruction errors.
# eigenbasis is U
# probably better to batch over bandwidth
def batch_reconstruction_errors_with_eigenbasis(
    eigenbasis, signal, bandwidths, sample_sizes, num_repeats
):
    result = []
    projs = batch_calc_proj(eigenbasis, bandwidths)
    for proj, bandwidth in tqdm(zip(projs, bandwidths)):
        U_k = restrict_eigenbasis(eigenbasis, bandwidth)
        # normalise this!
        raw_bandlimited_signal = proj @ signal
        bandlimited_signal = normalize(raw_bandlimited_signal, dim=0)
        p_star = optimal_sampling_distn(proj)
        errors = []
        for sample_size in sample_sizes:
            this_errs = []
            for _ in range(num_repeats):
                sample_matrices = sample_nodes_with_distn(p_star, sample_size)
                sample_signal = sample_matrices["M"] @ bandlimited_signal
                A, b = standard_decoder_first_half(
                    sample_matrices["P_Omega_inv_sqrt"],
                    sample_matrices["M"],
                    U_k,
                    sample_signal,
                )
                reconstructed_signal = standard_decoder_second_half(A, b, U_k)
                err = th.norm(bandlimited_signal - reconstructed_signal, p=2).item()
                this_errs.append(err)
            this_errs = th.tensor(this_errs)
            errors.append(this_errs.mean().item())
        result.append(errors)
    return np.array(result)


def classic_reconstruction_error_with_eigenbasis(
    eigenbasis, signal, bandwidth, sample_size
):
    U_k = restrict_eigenbasis(eigenbasis, bandwidth)
    proj = calc_proj(eigenbasis, bandwidth)
    bandlimited_signal = proj @ signal
    sample_matrices = {
        "M": th.randn(sample_size, eigenbasis.shape[0]),
        "P_Omega_inv_sqrt": th.eye(sample_size),
    }
    sample_signal = sample_matrices["M"] @ bandlimited_signal
    reconstructed_signal = standard_decoder(
        sample_matrices["P_Omega_inv_sqrt"], sample_matrices["M"], U_k, sample_signal
    )
    return th.norm(bandlimited_signal - reconstructed_signal, p=2)


# this only really makes sense for a circle graph
# sample in an evenly spaced way, and calculate the max error
# over signals. signals :: num_nodes x batch
def worst_regular_reconstruction_error_with_eigenbasis(
    eigenbasis, signals, bandwidth, sample_size
):
    num_nodes = eigenbasis.shape[0]
    U_k = restrict_eigenbasis(eigenbasis, bandwidth)
    proj = calc_proj(eigenbasis, bandwidth)
    # normalise this!
    raw_bandlimited_signals = proj @ signals
    bandlimited_signals = normalize(raw_bandlimited_signals, dim=0)
    # sample regularly
    omega = (th.arange(sample_size) * (float(num_nodes) / float(sample_size))).int()
    M = th.sparse_csr_tensor(
        crow_indices=th.arange(
            sample_size + 1, dtype=th.int32
        ),  # we have m non-zero elements
        col_indices=omega.int(),
        values=th.ones(sample_size),
        size=(sample_size, num_nodes),
    )
    sample_signals = M @ bandlimited_signals
    reconstructed_signals = standard_decoder_multiple_signals(
        th.eye(sample_size), M, U_k, sample_signals
    )
    return th.norm(bandlimited_signals - reconstructed_signals, p=2, dim=0).max()


# this time with uniform sampling!
def nodewise_reconstruction_error_with_eigenbasis_mult_signals(
    eigenbasis, signals, bandwidth, sample_size
):
    U_k = restrict_eigenbasis(eigenbasis, bandwidth)
    proj = calc_proj(eigenbasis, bandwidth)
    # normalise this!
    raw_bandlimited_signals = proj @ signals
    bandlimited_signals = raw_bandlimited_signals
    # bandlimited_signals = normalize(raw_bandlimited_signals, dim=0)
    p_star = uniform_sampling_distn(proj)
    sample_matrices = sample_nodes_with_distn(p_star, sample_size)
    M = sample_matrices["M"]
    sample_signals = M @ bandlimited_signals
    reconstructed_signals = standard_decoder_multiple_signals(
        th.eye(sample_size), M, U_k, sample_signals
    )
    return bandlimited_signals - reconstructed_signals, M, U_k


# we don't renormalise and we don't
def nodewise_reconstruction_error_with_M_mult_signals(
    eigenbasis, signals, bandwidth, M, pois=None
):
    U_k = restrict_eigenbasis(eigenbasis, bandwidth)
    proj = calc_proj(eigenbasis, bandwidth)
    # normalise this!
    raw_bandlimited_signals = proj @ signals
    bandlimited_signals = raw_bandlimited_signals
    # bandlimited_signals = normalize(raw_bandlimited_signals, dim=0)
    sample_signals = M @ bandlimited_signals
    if pois is None:
        sample_size = M.shape[0]
        pois = th.eye(sample_size)
    reconstructed_signals = standard_decoder_multiple_signals(
        pois, M, U_k, sample_signals
    )
    return bandlimited_signals - reconstructed_signals, U_k


def noisy_reconstruction_error_with_M_mult_signals(
    eigenbasis, num_signals, bandwidth, M, pois=None, noise_err=0.0001, snr=None
):
    num_nodes = eigenbasis.shape[0]
    U_k = restrict_eigenbasis(eigenbasis, bandwidth)
    # proj = calc_proj(eigenbasis, bandwidth)
    # equivalent in distribution to proj @ noise signals
    raw_bandlimited_signals = U_k @ noise_signal(bandwidth, num_signals)
    # bandlimited_signals = raw_bandlimited_signals
    # bandlimited_signals = normalize(raw_bandlimited_signals, dim=0)
    bandlimited_signals = raw_bandlimited_signals
    # note that if noise_err = \sigma, noise has variance multiplied by
    # \sigma^2
    if snr:
        noise_err = th.sqrt(th.tensor(bandwidth / (num_nodes * snr)))
    noise = noise_err * noise_signal(num_nodes, num_signals)
    noisy_bandlimited_signals = bandlimited_signals + noise
    sample_signals = M @ noisy_bandlimited_signals
    if pois is None:
        sample_size = M.shape[0]
        pois = th.eye(sample_size)
    reconstructed_signals = standard_decoder_multiple_signals(
        pois, M, U_k, sample_signals
    )
    # analytic cals:
    svdvals = th.linalg.svdvals(M @ U_k)
    corank = (bandwidth - svdvals.shape[0]) + th.sum(svdvals.abs() < 1e-8)
    high_freq_coeff = th.sum(svdvals[svdvals.abs() > 1e-8].reciprocal() ** 2)
    analytic_var = corank + (noise_err**2) * high_freq_coeff
    print(f"corank:{corank}, hfcoeff:{high_freq_coeff}")
    print(
        f"signal var: raw:{(bandlimited_signals ** 2).sum(dim=0).mean()}, noise: {(noise ** 2).sum(dim=0).mean()}"
    )
    print(
        f"err var:{((bandlimited_signals - reconstructed_signals) ** 2).sum(dim=0).mean()}, predicted err var: {analytic_var}"
    )
    # return ((bandlimited_signals - reconstructed_signals) ** 2).sum(
    #     dim=0
    # ).mean() - analytic_var
    return (
        ((bandlimited_signals - reconstructed_signals).double() ** 2).sum(dim=0).mean()
    )


def noisy_reconstruction_error_with_omega_mult_signals(
    eigenbasis, num_signals, bandwidth, omega, proj, pois=None
):
    num_nodes = eigenbasis.shape[0]
    U_k = restrict_eigenbasis(eigenbasis, bandwidth)
    # proj = calc_proj(eigenbasis, bandwidth)
    # equivalent in distribution to proj @ noise signals
    raw_bandlimited_signals = U_k @ noise_signal(bandwidth, num_signals)
    # bandlimited_signals = raw_bandlimited_signals
    bandlimited_signals = normalize(raw_bandlimited_signals, dim=0)
    noisy_bandlimited_signals = bandlimited_signals + 0.001 * noise_signal(
        num_nodes, num_signals
    )
    sample_signals = noisy_bandlimited_signals[omega]
    if pois is None:
        sample_size = omega.shape[0]
        pois = th.eye(sample_size)
    reconstructed_signals = iterative_decoder(omega, proj, sample_signals)
    return th.norm(bandlimited_signals - reconstructed_signals, p=2)


def noisy_reconstruction_error_with_omega_mult_signals_normalise(
    eigenbasis, num_signals, bandwidth, omega, proj
):
    num_nodes = eigenbasis.shape[0]
    U_k = restrict_eigenbasis(eigenbasis, bandwidth).contiguous()
    # proj = calc_proj(eigenbasis, bandwidth)
    # equivalent in distribution to proj @ noise signals
    raw_bandlimited_signals = U_k @ noise_signal(bandwidth, num_signals)
    # bandlimited_signals = raw_bandlimited_signals
    bandlimited_signals = normalize(raw_bandlimited_signals, dim=0)
    noisy_bandlimited_signals = bandlimited_signals + 0.001 * noise_signal(
        num_nodes, num_signals
    )
    sample_signals = noisy_bandlimited_signals[omega]
    if pois is None:
        sample_size = omega.shape[0]
        pois = th.eye(sample_size)
    reconstructed_signals = iterative_decoder(omega, proj, sample_signals)
    return th.norm(bandlimited_signals - reconstructed_signals, p=2)


@settings(deadline=timedelta(seconds=10), max_examples=100, verbosity=Verbosity.verbose)
@given(edges_num_nodes_bandwidth_omega=gen_undirected_graph_and_sample(10))
def test_analytic_reconstruction(edges_num_nodes_bandwidth_omega):
    graph_edges, num_nodes, bandwidth, omega = edges_num_nodes_bandwidth_omega
    sample_size = omega.shape[0]
    M = th.sparse_csr_tensor(
        crow_indices=th.arange(
            sample_size + 1, dtype=th.int32
        ),  # we have m non-zero elements
        col_indices=omega.int(),
        values=th.ones(sample_size),
        size=(sample_size, num_nodes),
    )
    noise = noise_signal(num_nodes, 1000000)
    U = calc_eigenbasis(graph_edges)
    errs, U_k = nodewise_reconstruction_error_with_M_mult_signals(
        U, noise, bandwidth, M
    )
    verrs = errs.var(dim=1, unbiased=True)
    verrs[verrs < 1e-6] = 0.0
    calc_errs = analytic_squared_errors(U_k, M)
    assert (
        calc_errs - verrs
    ).abs().max() < 0.01  # errors are like 0.001 most of the time


@typechecked
def mk_adjacency_matrix(
    graph: TensorType[2, "num_edges"], num_nodes: int
) -> TensorType["num_nodes", "num_nodes", float, th.sparse_coo]:
    ones = th.ones_like(graph[0])
    return (
        th.sparse_coo_tensor(graph, ones, size=(num_nodes, num_nodes))
        .float()
        .coalesce()
    )


# for each edge, how many triangles are there?
def triangle_edges(graph_edges):
    num_nodes = graph_edges.max().item() + 1
    adj = mk_adjacency_matrix(graph_edges, num_nodes)
    # for every node n, construct a matrix where (i,j)=1
    # if i->n->j is a V. Sum these across all nodes
    # which gives Vs which would be triangles with a base (i,j).
    # the multiplication then zeros where the base edge doesn't exist.
    # =======
    # This blows up for graphs of size > 1000
    # vs = vmap(lambda nrow: nrow.outer(nrow))(adj.to_dense()).sum(dim=0)
    # triangles = adj.to_dense() * vs  # elementwise
    # =======
    # annoyingly, the following is GPU only:
    # sparse_vs = sum(torch.sparse.sampled_addmm(adj, nrow.unsqueeze(1), nrow.unsqueeze(0)) for nrow in adj.to_dense())
    # =======
    triangles = th.sparse.sum(
        th.stack([nrow.outer(nrow).sparse_mask(adj) for nrow in adj.to_dense()]), dim=0
    )

    return triangles  # to do: extract edges in same order as edge_index


def coo_to_graph(matrix, has_feature=False):
    num_edges = matrix.indices().shape[1]
    graph = Data(
        edge_index=matrix.indices(),
        num_nodes=matrix.shape[0],
        edge_attr=matrix.values().reshape(num_edges, -1) if has_feature else None,
    )
    return graph


@settings(deadline=timedelta(seconds=5), max_examples=100, verbosity=Verbosity.verbose)
@given(edge_num_nodes_pair=gen_undirected_graph(1000))
def test_triangle_edges(edge_num_nodes_pair):
    graph_edges, num_nodes = edge_num_nodes_pair
    graph_edges = remove_isolated_nodes(graph_edges)[0]
    graph_edges = remove_self_loops(graph_edges)[0]
    graph_edges = (
        Data(edge_index=graph_edges, num_nodes=num_nodes).coalesce().edge_index
    )
    assume(graph_edges.numel() > 0)
    t_e = triangle_edges(graph_edges)
    assert t_e.to_dense().eq(t_e.to_dense().T).all().item()


@settings(
    deadline=timedelta(seconds=10), max_examples=1000, verbosity=Verbosity.verbose
)
@given(edge_num_nodes_pair=gen_undirected_graph(1000))
def test_triangles(edge_num_nodes_pair):
    graph_edges, num_nodes = edge_num_nodes_pair
    graph_edges = remove_isolated_nodes(graph_edges)[0]
    graph_edges = remove_self_loops(graph_edges)[0]
    graph_edges = (
        Data(edge_index=graph_edges, num_nodes=num_nodes).coalesce().edge_index
    )
    assume(graph_edges.numel() > 0)
    t_e = triangle_edges(graph_edges)
    tot_triangles_per_node = th.sparse.sum(t_e, dim=0).to_dense()
    adj = mk_adjacency_matrix(graph_edges, graph_edges.max().item() + 1)
    double_triangles = triangle_nodes_double(adj)
    assert ((double_triangles) == tot_triangles_per_node).all().item()


# for each node, how many triangles are there involving this node?
# returns double the number of triangles.
@typechecked
def triangle_nodes_double(
    adj: TensorType["num_nodes", "num_nodes", float, th.sparse_coo]
) -> TensorType["num_nodes"]:
    mm = torch.sparse.mm
    adj_cubed = mm(adj, mm(adj, adj)).to_dense()
    return th.diag(adj_cubed)


class EdgeAverager(MessagePassing):
    def __init__(self):
        super().__init__(aggr="mean", node_dim=-1)

    def forward(self, edge_index, edge_weight):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        row, col = edge_index
        deg = degree(col)
        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=deg, edge_weight=edge_weight.float())

    def message(self, x_j, edge_weight):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return edge_weight.view(1, -1)


class FormanEdge(MessagePassing):
    def __init__(self, gamma=1.0):
        super().__init__(aggr="mean", node_dim=-1)
        self.gamma = gamma

    def forward(self, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        pyg_graph = coo_to_graph(triangle_edges(edge_index), has_feature=True)
        edge_index = pyg_graph.edge_index
        edge_triangles = pyg_graph.edge_attr
        row, col = edge_index
        deg = degree(col)
        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=deg, edge_triangles=edge_triangles.float())

    def edge_updater(self, edge_index, triangles):
        return triangles

    def message(self, x_i, x_j, edge_triangles):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return 4 - x_i - x_j + (3 * self.gamma * edge_triangles.squeeze())


class MyEdgeConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr="mean", node_dim=-1)

    def forward(self, edge_index):
        deg = degree(edge_index[1])
        # edge_updater_type: (x: Tensor)
        edge_attr = self.edge_updater(edge_index, deg=deg)

        # propagate_type: (edge_attr: Tensor)
        return self.propagate(
            edge_index, edge_attr=edge_attr, size=(deg.size(0), deg.size(0))
        )

    def edge_update(self, deg_j, deg_i):
        return deg_j + deg_i

    def message(self, edge_attr):
        return 4 - edge_attr


def forman_edge(
    input_graph_edges: TensorType[2, "num_edges"],
    gamma: float = 1.0,
    num_nodes: Optional[int] = None,
) -> TensorType["num_edges"]:
    # clean inputs
    if input_graph_edges.shape[-1] == 0:
        raise IndexError
    pyg_graph = Data(edge_index=input_graph_edges).coalesce()
    if num_nodes is None:
        num_nodes = pyg_graph.num_nodes
    graph, _ = remove_self_loops(pyg_graph.edge_index)
    # generate helpful things
    node_degrees = degree(graph[0]).float()
    adj = mk_adjacency_matrix(graph, num_nodes)


# we remove self-loops and assume no isolated nodes
# we also make the graph undirected
# this probably needs some sort of testing
def fast_forman_node(
    input_graph_edges: TensorType[2, "num_edges"],
    gamma: float = 1.0,
    num_nodes: Optional[int] = None,
) -> TensorType["number_of_nodes"]:
    # clean inputs
    if input_graph_edges.shape[-1] == 0:
        raise IndexError
    pyg_graph = Data(edge_index=input_graph_edges).coalesce()
    if num_nodes is None:
        num_nodes = pyg_graph.num_nodes
    graph, _ = remove_self_loops(pyg_graph.edge_index)
    graph = to_undirected(graph)
    node_degrees = degree(graph[0]).float()
    adj = mk_adjacency_matrix(graph, num_nodes)
    sum_triangles = triangle_nodes_double(adj)
    sum_neighbour_degrees = adj @ node_degrees
    curvature = (
        4
        - node_degrees
        - (sum_neighbour_degrees / node_degrees)
        + (3 * gamma * sum_triangles / node_degrees)
    )
    # print(f"double_triangles: {sum_triangles}")

    return curvature


def cluster(cluster_size=20, num_clusters=2, p=0.8):
    cluster_sizes = []
    G, cluster_sz = clean_graph(erdos_renyi_graph(cluster_size, p))
    cluster_sizes.append(cluster_sz)
    for _ in range(num_clusters - 1):
        new_cluster, cluster_sz = clean_graph(erdos_renyi_graph(cluster_size, p))
        cluster_sizes.append(cluster_sz)
        G = add_bridge(G, new_cluster)
    return clean_graph(G)


def cluster_sampling(
    num_clusters: int = 3,
    cluster_size: int = 20,
    p: float = 0.8,
    num_connections: int = 1,
    num_samples=100,
):
    # construct graph and cluster sizes
    cluster_sizes = []
    G, cluster_sz = clean_graph(erdos_renyi_graph(cluster_size, p))
    cluster_sizes.append(cluster_sz)
    for _ in range(num_clusters - 1):
        new_cluster, cluster_sz = clean_graph(erdos_renyi_graph(cluster_size, p))
        cluster_sizes.append(cluster_sz)
        G = add_bridge(G, new_cluster)
    G, num_nodes = clean_graph(G)
    cum_cluster_sizes = th.cumsum(th.tensor(cluster_sizes), dim=0)
    # now calculate errs
    bandwidths = range(1, num_nodes)
    U = calc_eigenbasis(G, num_nodes, eps=1e-8)
    uniform_distn = uniform_sampling_distn(U)
    bandwidth = num_clusters * 3
    sample_size = bandwidth
    sample_matrices = [
        sample_nodes_with_distn(uniform_distn, sample_size) for _ in range(num_samples)
    ]
    entropys = []
    errs = []
    ranks = []
    for d in tqdm(sample_matrices):
        M = d["M"]
        omega = d["omega"]
        # entropy calc
        cluster_bins = th.zeros(num_clusters)
        for i in omega:
            i_bin = sum(cum_cluster_sizes <= i)
            cluster_bins[i_bin] += 1.0
        cluster_bins = cluster_bins / th.sum(cluster_bins)
        entropy = sum([-1 * p * th.log(p + 0.0001) for p in cluster_bins])
        # nodewise_err, _ = nodewise_reconstruction_error_with_M_mult_signals(
        #     U, noise_signal(num_nodes, 100), bandwidth, M
        # )
        # err = th.norm(nodewise_err)
        noise_err = 0.00
        err = noisy_reconstruction_error_with_M_mult_signals(
            U, 100, bandwidth, M, noise_err=noise_err
        )
        MU_k = (M @ restrict_eigenbasis(U, bandwidth)).double()
        # muk_evals = th.linalg.eigvalsh(MU_k @ MU_k.T)
        # err = th.sum(1 / muk_evals[muk_evals.abs() > 1e-9])
        rank = th.linalg.matrix_rank(MU_k)
        # print(f"err: {err}, muk_evals = {muk_evals}, rank= {rank}")
        entropys.append(entropy)
        errs.append(err)
        ranks.append(rank)
    entropys = th.tensor(entropys)
    errs = th.tensor(errs)
    ranks = th.tensor(ranks)
    # return (entropys, errs, ranks)
    df = pd.DataFrame({"entropy": entropys, "err": errs, "rank": ranks})
    g = sns.relplot(x="entropy", y="err", hue="rank", data=df, palette="deep")
    g.fig.suptitle(
        f"{num_clusters} clusters, bandwidth = {bandwidth}, sample_size={sample_size}, non-bandlimited noise with var {noise_err}, spearman_corr: {spearmanr(entropys, errs).correlation}"
        # f"{num_clusters} clusters, bandwidth = {bandwidth}, sample_size={sample_size}, error only from noise (closed form calculation), spearman_corr: {spearmanr(entropys, errs).correlation}"
    )
    g.fig.subplots_adjust(top=0.9)
    plt.show()
    return df


def big_main():
    tree_depth = 8
    graph_fns = [
        lambda: (binary_tree(tree_depth), (2**tree_depth) - 1),
        lambda: clean_graph(erdos_renyi_graph(1000, 0.8)),
        lambda: bridge_constructors(
            lambda: (binary_tree(tree_depth), (2**tree_depth) - 1),
            lambda: clean_graph(erdos_renyi_graph(700, 0.8)),
        ),
        lambda: bridge_constructors(
            lambda: clean_graph(erdos_renyi_graph(700, 0.8)),
            lambda: (grid(tree_depth), tree_depth * tree_depth),
        ),
    ]

    val_titles = [
        f"Binary Tree reconstruction errors (Depth={tree_depth})",
        "E-R reconstruction errors (p=0.8, N=1000)",
        f"Binary Tree (Depth={tree_depth}), bridged with E-R (p=0.8, N=700)",
        f"E-R bridged with Tree (p=0.8, N=700), (Depth={tree_depth}), ",
    ]
    """
    depths = [6, 7, 8, 9]
    graph_fns = [lambda: (binary_tree(depth), (2 ** depth) - 1) for depth in depths]
    val_titles = [
        f"Binary Tree reconstruction errors (Depth={depth})" for depth in depths
    ]
    """

    val_array = [
        reconstruction_errs_for_plotting(
            fg, parallel=True, max_bandwidth=255, max_sample_size=255
        )
        for fg in graph_fns
    ]
    assert len(val_titles) == len(val_array)
    vals_log_scale = True
    if vals_log_scale == True:
        vals = [np.log(0.0001 + vals) for vals in val_array]
    val_array = [vals / vals.max(axis=1, keepdims=True) for vals in val_array]

    fig1, axes = plt.subplots(ncols=2, nrows=ceil(len(val_array) / 2))
    fig1.suptitle(
        "Hard-to-reconstruct graphs (plotted with y=x, y=xlogx and paper bounds), Baraniuk (4.6) construction"
        # "Trees, cliques and a combo"
    )
    for ax, vals_normalised, title in zip(axes.ravel(), val_array, val_titles):
        sns.heatmap(vals_normalised.T, ax=ax)
        # plot y=x, the minimum sample size you can get perfect reconstruction at
        sns.lineplot(
            [0, vals_normalised.shape[0]],
            [0, vals_normalised.shape[0]],
            ax=ax,
            dashes=True,
            palette="g",
        )
        xs = range(vals_normalised.shape[0] + 1)
        # plot y = x log x, the theoretical bound
        delta = 0.9
        delta_coeff = 3 / (delta * delta)
        epsilon = 0.5
        sns.lineplot(
            xs,
            [0 if x == 0 else delta_coeff * x * log(2 * x / epsilon) for x in xs],
            ax=ax,
            palette="g",
        )
        sns.lineplot(
            xs,
            [0 if x == 0 else x * log(x) for x in xs],
            ax=ax,
            palette="white",
        )
        ax.set_xlabel("Bandwidth")
        ax.set_ylabel("Sample Size")
        ax.set_title(title)
        ax.invert_yaxis()

    plt.tight_layout()
    plt.pause(0.1)
    plt.show(block=True)


def plot_graph_and_samples(graph_edges, bandwidth, sample_size, scheme="uniform"):
    num_nodes = graph_edges.max().item() + 1
    U = calc_eigenbasis(graph_edges, num_nodes)
    proj = calc_proj(U, bandwidth)
    if scheme == "uniform":
        dist = uniform_sampling_distn(proj)
    elif scheme == "guy":
        dist = optimal_sampling_distn(proj)
    else:
        raise NotImplementedError()
    M = sample_nodes_with_distn(dist, sample_size)["M"]
    sampled_indices = M.col_indices()
    plot_graph_and_indices(graph_edges, sampled_indices)
    return sampled_indices


def plot_graph_and_indices(graph_edges, sampled_indices, layout="dot"):
    import matplotlib.pyplot as plt
    import networkx as nx
    import pydot
    from networkx.drawing.nx_pydot import graphviz_layout

    graph = Data(edge_index=graph_edges)
    signal = np.zeros(graph.num_nodes, dtype=bool)
    signal[sampled_indices] = True
    T = to_networkx(graph, to_undirected=True)
    pos = get_layout(T, prog=layout)
    # colours = ["blue" if s else "black" for s in signal]
    # nx.draw_networkx(T, pos, node_color=colours, with_labels=False, node_size=50)

    nx.draw(
        T,
        pos,
        node_color=signal,
        node_size=2700,
        cmap=plt.cm.Pastel2_r,
        vmin=-0.0,
        vmax=1,
        with_labels=True,
        font_size=28,
    )
    # nx.draw(
    #     T, pos, node_color=signal, node_size=20, cmap=plt.cm.Blues, vmin=-0.0, vmax=1
    # )
    plt.show()


def get_layout(T, prog="dot"):
    pos = graphviz_layout(T, prog=prog)
    # for some reason graphviz_layout is broken:
    if type(list(pos.keys())[0]) == str:
        pos = {i: pos[str(i)] for i in range(len(pos))}
    return pos


def plot_whisker_eigenvalues(
    graph, num_nodes, bandwidth, num_boxes=10, max_samples=None
):
    if max_samples is None:
        max_samples = bandwidth
    print("calculating....")
    U_k = restrict_eigenbasis(calc_eigenbasis(graph, num_nodes), bandwidth)
    sample_set = greedy_a_samples_only_fast(
        graph, num_nodes, bandwidth, max_samples=max_samples
    )
    spans = th.arange(0, max_samples, max_samples // num_boxes)
    if spans[-1] != max_samples:
        spans = th.hstack([spans, th.tensor([max_samples])])
    dfs = []
    for i in spans:
        MU_k = U_k[sample_set[: i + 1]]
        log_svds = th.log(th.linalg.svdvals(MU_k))
        log_evals = (2 * log_svds).numpy()
        dfs.append(
            pd.DataFrame(data={"Sample Size": i.item(), "Log Eigenvalues": log_evals})
        )
    total_df = pd.concat(dfs).reset_index()
    fig, ax = plt.subplots()
    sns.boxplot(
        data=total_df, x="Sample Size", y="Log Eigenvalues", whis=1000000.0, ax=ax
    )
    fig.suptitle(
        f"Eigenvalues for MU_k(MU_k)^T at different sample sizes for greedy a-optimal sampling on a Barabasi-Albert Graph with {num_nodes} nodes and bandwidth = {bandwidth}"
    )

    plt.show()


def plot_graph_and_signal_with_curvature(graph_edges, signal, prog="dot", **kwargs):
    graph = Data(
        edge_index=graph_edges,
    )
    T = to_networkx(graph, to_undirected=True)
    orc = OllivierRicci(T, alpha=0.5, verbose="INFO")
    orc.compute_ricci_curvature()
    edge_curvature = list(nx.get_edge_attributes(orc.G, "ricciCurvature").values())
    pos = graphviz_layout(orc.G, prog=prog)
    # nx.draw_networkx(T, pos, node_color=colours, with_labels=False, node_size=50)
    nx.draw(
        orc.G,
        pos,
        node_color=signal,
        node_size=50,
        cmap=plt.cm.Spectral,
        edge_color=edge_curvature,
        edge_cmap=plt.cm.hot,
        **kwargs,
    )


def plot_graph_and_signal(graph_edges, signal, prog="dot", **kwargs):
    graph = Data(
        edge_index=graph_edges,
    )
    T = to_networkx(graph, to_undirected=True)
    pos = get_layout(T, prog=prog)
    # nx.draw_networkx(T, pos, node_color=colours, with_labels=False, node_size=50)
    nx.draw(
        T,
        pos,
        node_color=signal,
        node_size=50,
        cmap=plt.cm.Spectral,
        # edge_color=edge_curvature,
        # edge_cmap=plt.cm.hot,
        **kwargs,
    )


def plot_clean_noisy_rec_err(graph_edges, bandwidth=0, prog="dot"):
    """For a graph, construct the plot with the following:
    clean signal         | noisy signal
    ---------------------+-------------
    reconstructed signal | error

    Do this for each sample size, and animate it.
    """
    graph = Data(
        edge_index=graph_edges,
    )
    T = to_networkx(graph, to_undirected=True)
    pos = get_layout(T, prog=prog)

    num_nodes = graph.num_nodes
    bandwidth = min(bandwidth, num_nodes)
    U = calc_eigenbasis(graph_edges, num_nodes, normalization=None, eps=1e-8)

    signal = noise_signal(num_nodes, 1)
    proj = calc_proj(U, bandwidth)
    bandlimited_signal = proj @ signal
    noisy_signal = bandlimited_signal + 0.1 * noise_signal(num_nodes, 1)

    # we want to take increasing subsamples of this permutation as nodes to sample
    # to make the animation make sense
    permutation = th.randperm(num_nodes)
    reconstructed_signals = []
    reconstructed_signals_from_clean = []
    omegas = []
    for sample_size in range(num_nodes + 1):
        omega = permutation[:sample_size]
        M = th.sparse_csr_tensor(
            crow_indices=th.arange(
                sample_size + 1, dtype=th.int32
            ),  # we have m non-zero elements
            col_indices=omega.int(),
            values=th.ones(sample_size),
            size=(sample_size, num_nodes),
        )
        reconstructed_signal = standard_decoder(
            th.eye(sample_size),
            M,
            restrict_eigenbasis(U, bandwidth),
            noisy_signal[omega].reshape(sample_size),
        )
        reconstructed_signal_from_clean = standard_decoder(
            th.eye(sample_size),
            M,
            restrict_eigenbasis(U, bandwidth),
            bandlimited_signal[omega].reshape(sample_size),
        )
        reconstructed_signals.append(reconstructed_signal)
        reconstructed_signals_from_clean.append(reconstructed_signal_from_clean)
        omegas.append(omega)

    fig, axes = plt.subplots(3, 2)
    axes[0][0].set_title(f"clean bandlimited signal (bandwidth = {bandwidth})")
    nx.draw(
        T,
        pos,
        node_color=bandlimited_signal,
        node_size=80,
        cmap=plt.cm.Spectral,
        ax=axes[0][0],
    )
    axes[0][1].set_title("signal corrupted by flat spectrum noise")
    nx.draw(
        T,
        pos,
        node_color=noisy_signal,
        node_size=80,
        cmap=plt.cm.Spectral,
        ax=axes[0][1],
    )

    def animation_update(frame, num_nodes, T, pos, axes):
        sample_size = frame
        rec_signal = reconstructed_signals[sample_size]
        err = bandlimited_signal.reshape(num_nodes) - rec_signal.reshape(num_nodes)
        omega = omegas[sample_size]
        omega_mask = th.zeros(num_nodes, dtype=bool)
        omega_mask[omega] = True
        vmax = noisy_signal.max().abs().item()
        vmin = noisy_signal.min().abs().item()
        # error calcs
        rec_signal_from_clean = reconstructed_signals_from_clean[sample_size]
        error_from_lack_of_info = (
            (
                rec_signal_from_clean.reshape(num_nodes)
                - bandlimited_signal.reshape(num_nodes)
            )
            .abs()
            .sum()
        )
        error_from_noise = err.abs().sum() - error_from_lack_of_info

        ax = axes[1][0]
        ax.clear()
        ax.set_title(f"reconstructed signal (# vertices sampled = {sample_size})")
        nx.draw(
            T,
            pos,
            node_color=rec_signal,
            node_size=80,
            cmap=plt.cm.Spectral,
            ax=ax,
        )

        ax = axes[1][1]
        ax.clear()
        ax.set_title(f"|reconstruction error| vs clean signal")
        nx.draw(
            T,
            pos,
            node_color=err.abs(),
            node_size=80,
            cmap=plt.cm.Blues,
            ax=ax,
            vmin=0.0,
            vmax=vmax,
        )

        ax = axes[2][0]
        ax.clear()
        ax.set_title(f"Sampled vertices")
        nx.draw(
            T,
            pos,
            node_color=omega_mask.int(),
            node_size=80,
            cmap=plt.cm.Blues,
            ax=ax,
            vmin=0.0,
            vmax=1.0,
        )

        ax = axes[2][1]
        ax.clear()
        ax.set_title(f"Sources of Error")
        sns.barplot(
            x=["Lack of Info", "Flat Spectrum Noise"],
            y=[error_from_lack_of_info.item(), error_from_noise.item()],
            ax=ax,
        )
        ax.set_ylim(0, num_nodes)

    # sample_size = 1
    # axes[1][0].set_title(f"reconstructed signal (# vertices sampled = {sample_size})")
    # nx.draw(
    #     T,
    #     pos,
    #     node_color=reconstructed_signals[sample_size],
    #     node_size=50,
    #     cmap=plt.cm.Spectral,
    #     ax=axes[1][0],
    # )
    ani = FuncAnimation(
        fig,
        animation_update,
        frames=num_nodes + 1,
        fargs=(num_nodes, T, pos, axes),
        interval=1000,
        repeat_delay=1000,
    )

    plt.show()


def plot_graph_and_noise(graph_edges, **kwargs):
    _graph = Data(
        edge_index=graph_edges,
    )
    num_nodes = _graph.num_nodes
    bandwidth = min(20, num_nodes)
    U = calc_eigenbasis(graph_edges, num_nodes)
    signal = noise_signal(_graph.num_nodes, 1)
    proj = calc_proj(U, bandwidth)
    bandlimited_signal = proj @ signal
    plot_graph_and_signal(graph_edges, signal)
    plt.show()


def plot_graph_and_eigenbasis(
    graph_edges: th.Tensor, which_eigenvector: int, normalization="sym", **kwargs
):
    _graph = Data(
        edge_index=graph_edges,
    )
    num_nodes = _graph.num_nodes
    # bandwidth = min(20, num_nodes)
    U = calc_eigenbasis(graph_edges, num_nodes, normalization=normalization)
    eigenvec = U[:, which_eigenvector]
    plot_graph_and_signal(graph_edges, eigenvec)
    plt.show()


def plot_graph_sampling(
    graph_edges,
    bandwidth,
    sample_size,
    num_nodes=None,
    prog="dot",
    sampling="uniform",
    **kwargs,
):
    # setup - normalise inputs
    graph = Data(
        edge_index=graph_edges,
    )
    if num_nodes is None:
        num_nodes = graph.num_nodes
    bandwidth = min(bandwidth, num_nodes)
    # get proj
    U = calc_eigenbasis(graph_edges, num_nodes)
    U_k = restrict_eigenbasis(U, bandwidth)
    proj = calc_proj(U, bandwidth)
    graph_degrees = degree(graph_edges[0])
    if sampling == "puy":
        p_star = optimal_sampling_distn(proj)
    elif sampling == "hubs":
        graph_degrees_sq = graph_degrees**2
        p_star = graph_degrees_sq / graph_degrees_sq.sum()
    elif sampling == "leaves":
        r_graph_degrees = th.reciprocal(graph_degrees + 0.01)
        p_star = r_graph_degrees / r_graph_degrees.sum()
    elif sampling == "uniform":
        p_star = uniform_sampling_distn(proj)
    else:
        print("I don't recognise that form of sampling, falling back to uniform")
        p_star = uniform_sampling_distn(proj)
    plot_graph_sampling_inner(
        graph_edges, p_star, bandwidth, sample_size, num_nodes=num_nodes, prog=prog
    )


def plot_graph_sampling_inner(
    graph_edges,
    p_star,
    bandwidth,
    sample_size,
    num_nodes=None,
    prog="dot",
    **kwargs,
):
    # setup - normalise inputs
    _graph = Data(
        edge_index=graph_edges,
    )
    if num_nodes is None:
        num_nodes = _graph.num_nodes
    bandwidth = min(bandwidth, num_nodes)
    # bandlimit signal
    U = calc_eigenbasis(graph_edges, num_nodes)
    U_k = restrict_eigenbasis(U, bandwidth)
    signal = noise_signal(_graph.num_nodes, 1)
    proj = calc_proj(U, bandwidth)
    raw_bandlimited_signal = proj @ signal
    bandlimited_signal = th.nn.functional.normalize(raw_bandlimited_signal, dim=0)
    # if sampling == "puy":
    #     p_star = optimal_sampling_distn(proj)
    # else:
    #     p_star = uniform_sampling_distn(proj)
    sample_matrices = sample_nodes_with_distn(p_star, sample_size)
    sample_signal = sample_matrices["M"] @ bandlimited_signal
    reconstructed_signal = standard_decoder(
        sample_matrices["P_Omega_inv_sqrt"], sample_matrices["M"], U_k, sample_signal
    )
    error = bandlimited_signal - reconstructed_signal
    # add zeros back into sample signal
    flat_bandlimited_signal = bandlimited_signal.flatten()
    sample_signal_with_zeros = flat_bandlimited_signal * th.zeros_like(
        flat_bandlimited_signal
    ).scatter(
        dim=0,
        index=sample_matrices["omega"],
        src=th.ones_like(flat_bandlimited_signal),
    )
    print(f"original signal:\n{bandlimited_signal.flatten()}")
    # print(f"omega:{sample_matrices['omega'].sort().values}")
    # print(f"M:{sample_matrices['M'].to_dense()}")
    # print(f"short sampled signal:\n{sample_signal}")
    print(f"sampled signal:\n{sample_signal_with_zeros.flatten()}")
    print(f"reconstructed signal:\n{reconstructed_signal.flatten()}")
    # calculate colour scaling
    vmax = max(
        bandlimited_signal.abs().max().item(), reconstructed_signal.abs().max().item()
    )
    vmin = -1.0 * vmax
    kwargs = {"vmax": vmax, "vmin": vmin, "prog": prog}
    # plot
    fig, axes = plt.subplots(2, 2)
    axes[0][0].set_title(f"original signal (k={bandwidth})")
    plot_graph_and_signal(graph_edges, bandlimited_signal, ax=axes[0][0], **kwargs)
    axes[0][1].set_title(f"sampled signal (m={sample_size})")
    plot_graph_and_signal(
        graph_edges, sample_signal_with_zeros, ax=axes[0][1], **kwargs
    )
    axes[1][1].set_title("reconstructed signal")
    plot_graph_and_signal(graph_edges, reconstructed_signal, ax=axes[1][1], **kwargs)
    axes[1][0].set_title(f"error ({th.norm(error, p=2)})")
    plot_graph_and_signal(graph_edges, error, ax=axes[1][0], **kwargs)
    plt.show()


def plot_graph_and_reconstruction(graph_edges, bandwidth=20, num_nodes=None):
    # fig1, axes = plt.subplots(ncols=2, nrows=2)
    graph = Data(
        edge_index=graph_edges,
    )
    if num_nodes is None:
        num_nodes = graph.num_nodes
    noise = noise_signal(num_nodes, num_feats=1)
    U = calc_eigenbasis(graph_edges, num_nodes)
    # U_k = restrict_eigenbasis(U, bandwidth)
    proj = calc_proj(U, bandwidth)
    bandlimited_signal = proj @ noise
    plot_graph_and_signal(graph_edges, bandlimited_signal)


def node_curvature_vs_reconstruction():
    pass


def am_gm_check(U, bandwidth, sample_size, num_repeats):
    U_k = restrict_eigenbasis(U, bandwidth)
    num_nodes = U.shape[1]
    # if it's easier to sample exactly, do so
    if math.comb(num_nodes, sample_size) < num_repeats:
        # th.combinations for th.arange(11), r=11, with_replacement=False hangs the computer
        # M_indices = th.combinations(
        #     th.arange(num_nodes), r=sample_size, with_replacement=False
        # )
        M_indices = th.tensor(
            list(itertools.combinations(range(num_nodes), sample_size)), dtype=th.long
        )
        MU_ks = U_k[M_indices]
    else:
        distribution = uniform_sampling_distn(U_k)
        Ms = batch_sample_nodes_with_distn(distribution, sample_size, num_repeats)["M"]
        # as everything is linear, we only need calculate the expectation of
        # pinv(MU_k) @ MU_k
        MU_ks = th.bmm(Ms, U_k.unsqueeze(0).expand(num_repeats, -1, -1))
    print(MU_ks.shape)
    principal_submatrices = th.bmm(th.transpose(MU_ks, 1, 2), MU_ks)
    print(principal_submatrices.shape)
    sq_sums = (principal_submatrices**2).sum(dim=(1, 2))
    traces = vmap(th.trace)(principal_submatrices)
    results = sq_sums / (traces**2)
    ub = (1 / num_nodes) + (2 / (num_nodes**3))
    print(f"how many definitely have det > 0: {(results <= ub).sum()}")
    # logdets = th.logdet(principal_submatrices)
    # evals = th.linalg.eigvalsh(principal_submatrices)
    svdvals = th.linalg.svdvals(MU_ks.double())
    evals = svdvals**2
    evals_vars = evals.var(unbiased=False, dim=1)
    # results = traces / (bandwidth * evals_vars.exp())
    print(MU_ks.shape)
    results = (evals).sum(dim=1)
    ranks = (svdvals > 1e-6).sum(dim=1)
    return results, ranks


def magic_words():
    import networkx as nx
    from GraphRicciCurvature.OllivierRicci import OllivierRicci
    from GraphRicciCurvature.FormanRicci import FormanRicci

    print("\n- Import an example NetworkX karate club graph")
    G = nx.karate_club_graph()

    print("\n===== Compute the Ollivier-Ricci curvature of the given graph G =====")
    # compute the Ollivier-Ricci curvature of the given graph G
    orc = OllivierRicci(G, alpha=0.5, verbose="INFO")
    orc.compute_ricci_curvature()
    print(
        "Karate Club Graph: The Ollivier-Ricci curvature of edge (0,1) is %f"
        % orc.G[0][1]["ricciCurvature"]
    )

    print("\n===== Compute the Forman-Ricci curvature of the given graph G =====")
    frc = FormanRicci(G)
    frc.compute_ricci_curvature()


def blah():
    nrows = 8
    for nrows in range(1, 5):
        acc = 0
        for i_s in itertools.combinations(range(nrows), nrows // 2):
            # A = Matrix([Symbol(x) for x in "abcdefghijklmnopqrstuvwxyz!@£$%^&*()"[:nrows * nrows]]).reshape(nrows,nrows)
            A = Matrix(
                [Symbol(f"A[{i} {j}]") for i in range(nrows) for j in range(nrows)]
            ).reshape(nrows, nrows)
            for j in range(nrows):
                for i in i_s:
                    A[i, j] = 0
            big = A * A * A
            # print(big)
            acc += trace(big)


if __name__ == "__main__2":
    from cProfile import run

    print("prep done, starting...")
    errs = []
    graph, num_nodes = clean_graph(binary_tree(6))
    U = calc_eigenbasis(graph, num_nodes)
    node_curvature = fast_forman_node(graph, gamma=1.0, num_nodes=num_nodes)
    node_degree = degree(graph[0])
    node_id = th.arange(num_nodes)
    for _ in range(1):
        # for aves in aves_graphs:
        # graph, num_nodes = clean_graph(erdos_renyi_graph(1000, 0.1))
        # graph, num_nodes = clean_graph(aves.edge_index)
        # U = calc_eigenbasis(graph, num_nodes)
        # node_curvature = fast_forman_node(graph, gamma=1.0, num_nodes=num_nodes)
        # node_degree = degree(graph[0])
        for i in tqdm(range(11, num_nodes)):
            node_errors = sample_analytic_error(
                U, bandwidth=i, sample_size=i, num_repeats=3
            )
            # node_errors = (
            #     nodewise_reconstruction_error_with_eigenbasis(
            #         U, noise_signal(num_nodes, 1).squeeze(), bandwidth=i, sample_size=i
            #     ).squeeze()
            #     ** 2
            # )
            errs.append(th.vstack([node_curvature, node_degree, node_id, node_errors]))
    errs = th.cat(errs, dim=1)
    # consider histogram or looking at local signal variation vs
    # curvature and reconstruction error
    data = pd.DataFrame(
        {
            "curvature": errs[0].numpy(),
            "degree": errs[1].numpy(),
            "id": errs[2].numpy(),
            "errs": errs[3].numpy(),
        }
    )
    print(data)
    # sns.scatterplot(data=data, x="curvature", y="errs")
    print(
        data.groupby("curvature")
        .agg(
            {
                "id": ["count"],
                "degree": ["mean"],
                "curvature": ["mean"],
                "errs": ["mean", "std"],
            }
        )
        .sort_values(by=("errs", "mean"))
    )
    # sns.lmplot(data=data, x="curvature", y="errs")
    # plt.show()
    #
    # graph, num_nodes = clean_graph(aves.edge_index)
    #
    graph, num_nodes = clean_graph(
        add_bridge(
            clean_graph(erdos_renyi_graph(20, 0.8))[0],
            clean_graph(erdos_renyi_graph(20, 0.8))[0],
        )
    )
    # tree_depth = 7
    # num_nodes = (2**tree_depth) - 1
    # graph = binary_tree(tree_depth)
    #
    # graph_side = 13
    # num_nodes = graph_side * graph_side
    # graph = grid(graph_side)

    # divisors of 2520
    # 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 18, 20, 21, 24, 28, 30, 35, 36, 40, 42, 45, 56, 60, 63, 70, 72, 84, 90, 105, 120, 126, 140, 168, 180, 210, 252, 280, 315, 360, 420, 504, 630, 840, 1260, 2520
    # num_nodes = 2520
    # graph = circle(num_nodes)
    U = calc_eigenbasis(graph, num_nodes, eps=1e-8)
    print(f"num nodes: {num_nodes}")
    np.set_printoptions(linewidth=160, precision=4)
    print(
        pd.DataFrame(
            ([x.numpy() for x in y] for y in find_bad_combos(U, 3, 2)),
            columns=["nodes", "eigenvectors", "corank"],
        )
    )

    def calc_my_errs0(U, num_nodes: int):
        return th.tensor(
            [
                [
                    sample_total_analytic_error(
                        U,
                        bandwidth,
                        sample_size,
                        num_repeats=4000,
                        exact_rank_calc=False,
                    )
                    for sample_size in th.arange(1, num_nodes)  # num_nodes)
                ]
                for bandwidth in (th.arange(1, num_nodes))  # num_nodes))
            ]
        )
        # res = th.zeros(num_nodes - 1, num_nodes - 1)
        # for sample_size, bandwidth in tqdm(
        #     th.cartesian_prod(th.arange(1, num_nodes), th.arange(1, num_nodes))
        # ):
        #     res[sample_size - 1, bandwidth - 1] = sample_total_analytic_error(
        #         U, bandwidth, sample_size, num_repeats=4000000
        #     )
        # return res.T

    def calc_ranks(U, num_nodes: int):
        return th.tensor(
            [
                [
                    get_mean_corank(U, bandwidth, sample_size, num_repeats=100000)
                    for sample_size in range(1, num_nodes)
                ]
                for bandwidth in tqdm(range(1, num_nodes))
            ]
        )

    def calc_high_freq(U, num_nodes: int):
        return th.tensor(
            [
                [
                    high_freq_noise_decoded(
                        U, bandwidth, sample_size, num_repeats=100000, agg=th.mean
                    )
                    for sample_size in range(1, num_nodes)
                ]
                for bandwidth in tqdm(range(1, num_nodes))
            ]
        )

    def calc_my_errs1(U, num_nodes: int, optimal: bool = True):
        def inner(bandwidth, sample_size):
            proj = calc_proj(U, bandwidth)
            p_star = (
                optimal_sampling_distn(proj)
                if optimal
                else uniform_sampling_distn(proj)
            )
            res = []
            for _ in range(4):
                sample_matrices = sample_nodes_with_distn(p_star, sample_size)
                M = sample_matrices["M"]
                omega = sample_matrices["omega"]
                res.append(
                    noisy_reconstruction_error_with_M_mult_signals(
                        U, 100, bandwidth, M, pois=None, noise_err=0.000
                    )
                    #     noisy_reconstruction_error_with_omega_mult_signals(
                    #         U, 100, bandwidth, omega, proj
                    #     )
                )
            return th.mean(th.tensor(res))

        return th.tensor(
            [
                [inner(bandwidth, sample_size) for sample_size in range(1, num_nodes)]
                for bandwidth in tqdm(range(1, num_nodes))
            ]
        )

    p = 0.8
    big_errs_list = []
    for _ in tqdm(range(1)):
        graph, num_nodes = clean_graph(
            add_bridge(
                clean_graph(erdos_renyi_graph(5, p))[0],
                clean_graph(erdos_renyi_graph(5, p))[0],
            )
        )
        # graph, num_nodes = clean_graph(binary_tree(6))
        # graph, num_nodes = clean_graph(erdos_renyi_graph(8, p))
        U = calc_eigenbasis(graph, num_nodes, eps=1e-8)
        this_errs = calc_my_errs1(U, num_nodes, optimal=False)
        #
        # this_errs = calc_high_freq(U, num_nodes)

        # this_ranks = calc_ranks(U, num_nodes)
        # this_ranks = th.vstack([th.zeros_like(this_ranks[0]), this_ranks])
        # this_ranks = this_ranks[1:] - this_ranks[:-1]
        # this_errs = all_exact_total_analytic_error(U, exact_rank_calc=False)
        big_errs_list.append(this_errs)
    errs = th.stack(big_errs_list).mean(dim=0).numpy()
    print(errs)

    # run("errs = calc_my_errs0(U, num_nodes).numpy()", sort="cumtime")
    # errs = errs.clip(max=5)
    errs = np.log(errs + 0.01)

    fig, ax = plt.subplots()
    sns.heatmap(errs.T, ax=ax)  # , vmin=0, vmax=25)

    ax.set_xlabel("Bandwidth")
    ax.set_ylabel("Sample Size")
    ax.invert_yaxis()
    # rank_submatrices(U, 20, 20, num_repeats=40000)

    # fig2, ax2 = plt.subplots()
    # sns.heatmap(this_ranks.T, ax=ax2)  # , vmin=0, vmax=25)
    # ax2.set_xlabel("Bandwidth")
    # ax2.set_ylabel("Sample Size")
    # ax2.invert_yaxis()

    plt.tight_layout()
    plt.pause(0.1)
    plt.show(block=True)

    bandwidth = 10
    sample_size = 10
    M_indices = index_combinations(num_nodes, sample_size)
    snrs = th.linspace(1, 100, 100).reciprocal()
    errs_inner = []
    errs_avg = []
    for snr in snrs:
        for omega in M_indices:
            m = sample_size
            M = th.sparse_csr_tensor(
                crow_indices=th.arange(
                    m + 1, dtype=th.int32
                ),  # we have m non-zero elements
                col_indices=omega.int(),
                values=th.ones(m),
                size=(m, num_nodes),
            )
            err = noisy_reconstruction_error_with_M_mult_signals(
                U, 100, bandwidth, M, snr=snr
            )
            errs_inner.append(err)
        errs_avg.append(th.tensor(errs_inner).numpy().mean())
    print(errs_avg)

    fig3, ax3 = plt.subplots()
    sns.regplot(x=snrs.reciprocal().numpy(), y=errs_avg, ax=ax3)
    ax3.set_xlabel("1/SNR")
    ax3.set_ylabel("Reconstruction Error Variance")
    fig3.suptitle(
        f"Noisy reconstruction on two bridged clusters (bandwidth = {bandwidth}, sample size = {sample_size})"
    )
    plt.show()


### Figure out when adding a node increases \xi_1


def analytic_GLR_xi_1(L, U_k, sample, mu=0.01):
    L1 = (mu * L).clone()
    for s in sample:
        L1[s, s] += 1.0
    MTMU_k = th.zeros_like(U_k)
    MTMU_k[sample] = U_k[sample]
    return (U_k - th.linalg.solve(L1, MTMU_k)).square().sum().item()


def analytic_GLR_xi_1_slow(L, U_k, sample, mu=0.01):
    M = construct_sample_matrix(sample, L.shape[0]).to_dense().type(U_k.dtype)
    MTM = M.T @ M
    L1 = (mu * L) + MTM
    MTMU_k = M.T @ (M @ U_k)
    return (U_k - th.linalg.solve(L1, MTMU_k)).square().sum().item()


def analytic_GLR_xi_1_fast(L, U_k_scaled, sample, mu=0.01):
    L1 = mu * L
    L1[sample, sample] += 1
    return ((th.linalg.solve(L1, U_k_scaled)).square().sum() * mu * mu).item()


def analytic_GLR_xi_2(L, _, sample, mu=0.01):
    L1 = (mu * L).clone()
    for s in sample:
        L1[s, s] += 1.0
    return L1.inverse()[sample].square().sum().item()


def analytic_GLR_xi_2_breakdown(L, _, sample, mu=0.01):
    L1 = (mu * L).clone()
    for s in sample:
        L1[s, s] += 1.0
    # X = (M^TM + mu * L)^-1 M^T
    total_inv = L1.inverse()[sample]
    # ZZT_plus is the ones
    ZZT_plus = th.ones_like(total_inv) / len(sample)
    total_inv_without_ones = total_inv - ZZT_plus
    # We want to break down the 2-norm of X
    full_two_norm_wo_ones = total_inv_without_ones.square().sum()
    # The part which isn't the principal submatrix of (M^TM + mu * L)^-1
    total_inv_off_rect_contrib = (
        full_two_norm_wo_ones - total_inv_without_ones[:, sample].square().sum()
    )
    # We now look at the part corresponding to the princopal submatrix M(M^TM + mu * L)^-1 M^T
    # We cpnvert it to the 'fancy' basis our bound uses, and look at the off-diagonal elements (that we ignore)
    # and the on-diagonal elements that we do approximate.
    total_inv_main_square = total_inv_without_ones[:, sample]
    total_inv_main_diag = total_inv_main_square.diag().square().sum()
    total_inv_off_diag = total_inv_main_square.square().sum() - total_inv_main_diag
    # return {
    #     "main_diag": total_inv_main_diag.item(),
    #     "off_diag_main_square": total_inv_off_diag.item(),
    #     "off_square": total_inv_off_rect_contrib.item(),
    #     "ones": ZZT_plus.square().sum().item(),
    # }
    return {
        "main_square": total_inv_main_square.square().sum().item(),
        "off_square": total_inv_off_rect_contrib.item(),
    }


def analytic_GLR_delta_1_all(L, U_k_scaled, sample, mu=0.01):
    A = mu * L
    A[sample, sample] += 1
    Ainv = A.inverse()
    Ainvsq_diag = Ainv.square().sum(dim=1)
    # almost ABBA but not quite
    AinvBBTAinv = (lambda X: X @ X.T)(Ainv @ U_k_scaled)
    AinvsqBBTAinv = Ainv @ AinvBBTAinv

    delta_1s = Ainvsq_diag * th.diag(AinvBBTAinv) / ((1 + th.diag(Ainv)) ** 2) - (
        (2 * th.diag(AinvsqBBTAinv)) / (1 + th.diag(Ainv))
    )
    return delta_1s * (mu**2)


def MTM_plus_mu_L(L, sample, mu):
    A = mu * L
    # A[sample, sample] += 1
    # The following two lines do A[sample, sample] +=1 in a
    # vmappable way
    sample_ones = th.ones_like(sample, dtype=L.dtype)
    A = A.index_put((sample, sample), sample_ones, accumulate=True)
    return A


# computed only the parts needed to check if delta_1s is +ve
def analytic_GLR_delta_1_is_positive(L, U_k_scaled, sample, mu=0.01):
    A = MTM_plus_mu_L(L, sample, mu)
    Ainv = A.inverse()
    AinvBBTAinv = (lambda X: X @ X.T)(Ainv @ U_k_scaled)
    Ainvsq_diag = Ainv.square().sum(dim=1)  # =  th.diag(Ainv @ Ainv)
    # AinvsqBBTAinv_diag = th.diag(Ainv @ AinvBBTAinv)
    AinvsqBBTAinv_diag = (Ainv * AinvBBTAinv.T).sum(dim=1)
    LHS = Ainvsq_diag * th.diag(AinvBBTAinv)
    RHS = 2 * AinvsqBBTAinv_diag * (1 + th.diag(Ainv))
    result = (LHS - RHS) > -1e-10
    result[sample] = False
    return result


# returns a N-length tensor, which is true delta_1 would be positive
# at sample values it returns False.
def analytic_GLR_ABBA_is_negative(L, U_k_scaled, sample, mu=0.01):
    A = MTM_plus_mu_L(L, sample, mu=mu)
    Ainv = A.inverse()
    AinvBBTAinv = (lambda X: X @ X.T)(Ainv @ U_k_scaled)
    # AinvsqBBTAinv_diag = th.diag(Ainv @ AinvBBTAinv)
    AinvsqBBTAinv_diag = (Ainv * AinvBBTAinv.T).sum(dim=1)
    result = AinvsqBBTAinv_diag < -1e-10
    result[sample] = False
    return result


def analytic_GLR_delta_2_is_positive(L, _, sample, mu=0.01):
    M = construct_sample_matrix(sample, L.shape[0])
    A = mu * L
    A[sample, sample] += 1
    Ainv = A.inverse()
    Ainvsq_diag = Ainv.square().sum(dim=1)
    AinvBBTAinv = (lambda X: X.T @ X)(M @ Ainv)
    AinvsqBBTAinv = Ainv @ AinvBBTAinv
    LHS = Ainvsq_diag * th.diag(AinvBBTAinv)
    RHS = 2 * th.diag(AinvsqBBTAinv) * (1 + th.diag(Ainv))
    result = (LHS - RHS) > 1e-10
    result[sample] = False
    return result


def does_GLR_increase(graph, num_nodes, bandwidth, sample_size, mu=0.01):
    assert sample_size > 0, "Sample Size needs to be strictly positive"
    L = calc_laplacian(graph, num_nodes, normalization=None).float()
    U = calc_eigenbasis(graph, num_nodes, eps=1e-11, normalization=None).float()
    # U_k = restrict_eigenbasis(U, bandwidth)
    U_k_scaled = restrict_eigenbasis(L @ U, bandwidth)
    # U_k_scaled[:,0] = 0

    total_result = {}
    for sample in index_combinations(num_nodes, sample_size - 1):
        curr_xi_1 = analytic_GLR_xi_1_fast(L, U_k_scaled, sample, mu=mu)
        other_nodes = set(range(num_nodes)) - set(sample.tolist())
        inner_result = []
        for o in other_nodes:
            new_sample = th.hstack((sample, th.tensor([o])))
            new_xi_1 = analytic_GLR_xi_1_fast(L, U_k_scaled, new_sample, mu=mu)
            if new_xi_1 > curr_xi_1:
                # inner_result.append((o, (curr_xi_1, new_xi_1)))
                inner_result.append(o)
        if inner_result:
            total_result[(tuple(sample.tolist()))] = inner_result

    return total_result


def does_GLR_increase_fast(
    graph, num_nodes, bandwidth, sample_size, mu=0.01, abba_approx=True
):
    assert sample_size > 0, "Sample Size needs to be strictly positive"
    L = calc_laplacian(graph, num_nodes, normalization=None).float()
    U = calc_eigenbasis(graph, num_nodes, eps=1e-11, normalization=None).float()
    # U_k = restrict_eigenbasis(U, bandwidth)
    U_k_scaled = restrict_eigenbasis(L @ U, bandwidth)
    # U_k_scaled[:,0] = 0

    total_result = {}

    combs = index_combinations(num_nodes, sample_size - 1)
    if abba_approx:
        delta_pos_fn = analytic_GLR_ABBA_is_negative
    else:
        delta_pos_fn = analytic_GLR_delta_1_is_positive

    all_delta_1_signs = vmap(delta_pos_fn, in_dims=(None, None, 0, None))(
        L, U_k_scaled, combs, mu=mu
    )
    for sample, delta_1_signs in zip(combs, all_delta_1_signs):
        # if abba_approx:
        #     delta_1_signs = analytic_GLR_ABBA_is_negative(L, U_k_scaled, sample, mu=mu)
        # else:
        #     delta_1_signs = analytic_GLR_delta_1_is_positive(
        #         L, U_k_scaled, sample, mu=mu
        #     )
        indices = set(th.where(delta_1_signs)[0].tolist())
        sample_set = set(sample.tolist())
        inner_result = indices - sample_set
        if inner_result:
            total_result[tuple(sample_set)] = inner_result

    return total_result


def does_GLR_increase_multiple_mu(
    graph, num_nodes, bandwidth, sample_size, mus=[0.01, 1, 1000]
):
    assert sample_size > 0, "Sample Size needs to be strictly positive"
    L = calc_laplacian(graph, num_nodes, normalization=None).float()
    U = calc_eigenbasis(graph, num_nodes, eps=1e-11, normalization=None).float()
    # U_k = restrict_eigenbasis(U, bandwidth)
    U_k_scaled = restrict_eigenbasis(L @ U, bandwidth)

    total_result = {}
    for sample in index_combinations(num_nodes, sample_size - 1):
        curr_xi_1 = th.tensor(
            [analytic_GLR_xi_1_fast(L, U_k_scaled, sample, mu=mu) for mu in mus]
        )
        other_nodes = set(range(sample_size)) - set(sample.tolist())
        inner_result = []
        for o in other_nodes:
            new_sample = th.hstack((sample, th.tensor([o])))
            new_xi_1 = th.tensor(
                [analytic_GLR_xi_1_fast(L, U_k_scaled, new_sample, mu=mu) for mu in mus]
            )
            if (new_xi_1 > curr_xi_1).all():
                # inner_result.append((o, (curr_xi_1, new_xi_1)))
                inner_result.append(o)
        if inner_result:
            total_result[(tuple(sample.tolist()))] = inner_result

    return total_result


def pyg_is_connected(g, num_nodes):
    if num_nodes <= 10:
        adj = to_dense_adj(g, max_num_nodes=num_nodes)
        adj_pow = th.linalg.matrix_power(adj, num_nodes)
        return (adj_pow.abs() > 0).all()
    else:
        nxG = to_networkx(Data(edge_index=g, num_nodes=num_nodes))
        nxG = nxG.to_undirected()
        return nx.is_connected(nxG)


def find_GLR_increase(num_nodes, bandwidth, sample_size, num_attempts=1000, mu=0.01):
    results = {}
    # big_results checks if any subsets have the property
    # " all added nodes are bad "
    big_results = []
    for _ in tqdm(range(num_attempts)):
        g, _ = clean_graph(erdos_renyi_graph(num_nodes, 0.8))
        g = to_undirected(g)
        g_tuple = tuple(map(tuple, g.T.tolist()))
        if pyg_is_connected(g, num_nodes):
            inner_result = does_GLR_increase_fast(
                g, num_nodes, bandwidth, sample_size, mu=mu, abba_approx=False
            )
            if inner_result:
                results[g_tuple] = inner_result
    return results


def find_GLR_increase_multiple_mu(
    num_nodes, bandwidth, sample_size, num_attempts=1000, mus=[0.01, 1, 100]
):
    results = {}
    # big_results checks if any subsets have the property
    # " all added nodes are bad "
    big_results = []
    for _ in tqdm(range(num_attempts)):
        g, _ = clean_graph(erdos_renyi_graph(num_nodes, 0.8))
        g = to_undirected(g)
        g_tuple = tuple(map(tuple, g.T.tolist()))
        nxG = to_networkx(Data(edge_index=g, num_nodes=num_nodes)).to_undirected()
        if nx.is_connected(nxG):
            inner_result = does_GLR_increase_multiple_mu(
                g, num_nodes, bandwidth, sample_size, mus=mus
            )
            if inner_result:
                results[g_tuple] = inner_result
    return results


# lower bound for the minimum eigenvalue of M^T@M+mu*L
def GLR_min_eig_lb(mu, algebraic_connectivity, m_over_N, torch=False):
    gamma = 1 + (mu * algebraic_connectivity)
    mysqrt = th.sqrt if torch else sqrt
    return 0.5 * (
        gamma - mysqrt((gamma**2) - (4 * mu * algebraic_connectivity * m_over_N))
    )


def GLR_xi_2_ub(mu, alg_connectivity, m, N, torch=False):
    min_eig = GLR_min_eig_lb(mu, alg_connectivity, m / N, torch=torch)
    return (1 / min_eig**2) + ((m - 1) / (mu * alg_connectivity))


# Believe this is an LB, gotta prove it
def GLR_xi_2_lb_maybe(mu, lambda_max, m, N):
    # not a lower bound!!
    # return ((N / m) ** 2) + ((m - 1) / (1 + (mu * lambda_max)))
    full_matrix_frob_lb = ((N / m) ** 2) + ((m - 1) / (1 + (mu * lambda_max)))
    return full_matrix_frob_lb * m / N


# Believe this is an LB, gotta prove it
def GLR_xi_2_lb_maybe_with_eigs(mu, L_eigs, m, N):
    # not a lower bound!!
    # Note that also this is a terrible approximation for the full matrix frobenius??
    full_matrix_frob_lb = ((N / m) ** 2) + (
        1.0 / (1 + (mu * L_eigs[1:]))
    ).square().sum()
    return full_matrix_frob_lb * m / N


def GLR_xi_2_lb_maybe_with_eigs_last_m(mu, L_eigs, m, N):
    # This should actually be a lower bound
    full_matrix_frob_lb = (N / m) + (
        1.0 / (1 + (mu * L_eigs[N - (m - 1) : N]))
    ).square().sum()
    return full_matrix_frob_lb


def GLR_xi_2_lb_maybe_with_taylor(mu, L, m, N):
    # not a lower bound!!
    # approximates the above via using the taylor approximation
    # (1+x)^-2 = 1 - 2x + 3x^2 + o(x^3) for small x
    # so summing across the laplacian eigs gives
    # N - 2 mu  tr(L) + 3 mu tr(L^2)
    # so N/m + m/N * (sum(1/1+mu*L_eig[i]))
    # is approximately N/m + (m/N) * (N - 2 tr(L) + 3 tr(L^2)
    eig_inverse_sum_approx = N - (2 * mu * L.trace()) + (3 * mu * mu * (L @ L).trace())
    # sometimes mu is too big; need mu < 1/lambda_max(L)
    # if mu < 1 / (2 * max_degree(L)) we're fine
    eig_inverse_sum_approx = max(eig_inverse_sum_approx, N)
    return (N / m) + ((m / N) * eig_inverse_sum_approx)


def GLR_xi_2_lb(mu, L, sample_set):
    degrees = L.diag()[sample_set]
    lb = 1.0 / ((mu * mu * degrees) + ((1 + (mu * degrees)) ** 2))
    return lb.sum()


def GLR_xi_2_lb_with_fancy_basis(mu, L, sample_set):
    N = L.shape[0]
    m = len(sample_set)
    fancy_basis = th.stack(
        [
            th.hstack([th.zeros(m - (i + 1)), th.tensor(i), -1 * th.ones(i)]).type(
                L.dtype
            )
            / sqrt(i * i + 1)
            for i in range(1, m)
        ]
    )  # m-1 elements
    muL = mu * L[sample_set][:, sample_set]
    muL_diag = th.diag(th.diag(muL))
    return (N / m) + vmap(
        lambda x_i: 1 - ((x_i @ (muL_diag @ x_i)) / (1 + (x_i @ (muL @ x_i))))
    )(fancy_basis).square().sum()


def GLR_xi_2_ub_with_eigs(mu, L_eigs, m, N):
    alg_connectivity = L_eigs[1]
    min_eig = GLR_min_eig_lb(mu, alg_connectivity, m / N, torch=True)
    return (1 / min_eig**2) + (1.0 / (L_eigs[1:m] * mu)).square().sum()


def GLR_xi_2_ub_with_eigs_again(mu, L_eigs, m, N):
    return (N / m) + (
        ((1 / m) + (m / N))
        * ((L_eigs.max() / (1 + (mu * L_eigs.max()))) ** 2)
        / (L_eigs[1:]).square()
    ).sum()
    # return (N / m) + (m / ((N * mu) * mu * (L_eigs[1] * (N - 1)).square())).sum()


def GLR_xi_2_ub_with_eigs_again_again(mu, L_eigs, m, N):
    # bound1 = 1 - (1 / (1 + (2 * mu * L_eigs.max())))
    # scaled_pinv_trace = L_eigs[1:].reciprocal().sum() / mu
    # extra_N_over_m = (3 * scaled_pinv_trace) / (1 + (2 * scaled_pinv_trace * m / N))
    extra_N_over_m = 1.5 * (N / m)
    return (
        (N / m)
        + extra_N_over_m
        + (
            (m / N)
            * ((L_eigs.max() / (1 + (mu * L_eigs.max()))) ** 2)
            / (L_eigs[1:]).square()
        ).sum()
    )
    # return (N / m) + (m / ((N * mu) * mu * (L_eigs[1] * (N - 1)).square())).sum()


def GLR_xi_2_ub_with_eigs_sectioned(mu, L_eigs, m, N):
    tr_pinv = L_eigs[1:].reciprocal().sum()
    err_from_Z = N / m
    frob_pinv = L_eigs[1:].reciprocal().square().sum()
    # top corner bound is only valid if mu < 2/lambda_N
    top_corner = m * ((tr_pinv / ((mu * N) + tr_pinv)) ** 2)
    off_diag_error = (
        (m / N)
        * (L_eigs.max() / (1 + mu * L_eigs.max()))
        * (frob_pinv - ((tr_pinv**2) / N))
    )
    error_from_P = 1 + (N / m)
    return err_from_Z + error_from_P + top_corner + off_diag_error


# This bound is not even a function of mu?!
def GLR_xi_2_ub_with_eigs_sectioned_weaker(mu, L_eigs, m, N):
    tr_pinv = L_eigs[1:].reciprocal().sum()
    err_from_Z = N / m
    frob_pinv = L_eigs[1:].reciprocal().square().sum()
    # top corner bound is only valid if mu < 2/lambda_N
    top_corner = m - 1
    off_diag_error = (
        (m / N)
        # * (L_eigs.max() / (1 + mu * L_eigs.max()))
        * L_eigs.max()
        * (frob_pinv - ((tr_pinv**2) / N))
    )
    error_from_P = 1 + (N / m)
    return err_from_Z + error_from_P + top_corner + off_diag_error


# This bound is not even a function of mu?!
def GLR_xi_2_ub_with_eigs_sectioned_synchro(mu, L_eigs, m, N):
    return 2.5 * (N / m) + (
        m * (1 + (0.25 * (L_eigs.max() / L_eigs[1:].min() - 1).square()))
    )


def GLR_MSE_ub(mu, L_eigs, k, m, N, SNR):
    xi_2 = GLR_xi_2_ub_with_eigs_sectioned_weaker(mu, L_eigs, m, N)
    full_set_noise_sensitivity = (1.0 / (1.0 + mu * L_eigs)).square().sum()
    full_set_bandlimited_error = (1.0 - (1.0 / (1.0 + mu * L_eigs[:k]))).square().sum()
    sigma_sq = k / (N * SNR)

    return 2


def GLR_xi_2_neumann_sample(mu, L, sample, num_terms=20):
    N = L.shape[0]
    m = len(sample)
    sample_C = th.ones(N, dtype=L.dtype)
    sample_C[sample] = 0
    I_minus_A = th.diag(sample_C) - mu * L
    I_minus_A_pow_tmp = th.eye(N, dtype=L.dtype)
    acc = th.zeros(N, N)
    for _ in range(num_terms):
        acc += I_minus_A_pow_tmp
        I_minus_A_pow_tmp = I_minus_A_pow_tmp @ I_minus_A
    return acc[sample].square().sum()


def GLR_compare_lb(mu, N):
    # g = barabasi_albert_graph(N,3)
    g, _ = clean_graph(erdos_renyi_graph(N, 0.7))
    L = calc_laplacian(g, N, normalization=None)
    alg = th.linalg.eigvalsh(L)[1]
    ubs = []
    actuals = []

    def direct_calc_ub(m):
        M = construct_sample_matrix(th.arange(m), N).to_dense()
        Pi_M = M.T @ M
        Pi_L = th.eye(N) - (th.outer(th.ones(N), th.ones(N)) / N)
        return th.linalg.eigvalsh(Pi_M + (mu * alg * Pi_L))[0]

    direct_ubs = []
    for i in tqdm(range(1, N)):
        L_mod = L.clone()
        L_mod[th.arange(i), th.arange(i)] += 1
        actuals.append(th.linalg.eigvalsh(L_mod)[0])
        ubs.append(GLR_min_eig_lb(mu, alg, float(i) / N))
        direct_ubs.append(direct_calc_ub(i))
    return th.stack((th.stack(actuals), th.stack(ubs), th.stack(direct_ubs))).T


def GLR_plot_xi_2_bounds(g, N, mu, max_sample_size, min_sample_size=1, graph_name=None):
    sample_sizes = th.arange(min_sample_size, max_sample_size)
    L = calc_laplacian(g, N, normalization=None)
    eigvals = th.linalg.eigvalsh(L)
    alg_conn = eigvals[1]
    lambda_max_L = eigvals[-1]
    # Calculate bounds
    # lbs = vmap(lambda m: GLR_xi_2_lb(mu, alg_conn, lambda_max_L, m, N))(sample_sizes)
    # ubs = vmap(lambda m: GLR_xi_2_ub(mu, alg_conn, m, N, torch=True))(sample_sizes)
    ubs = th.hstack(
        ([GLR_xi_2_ub_with_eigs_again(mu, eigvals, m, N) for m in sample_sizes.numpy()])
    )
    ubs2 = th.hstack(
        (
            [
                GLR_xi_2_ub_with_eigs_again_again(mu, eigvals, m, N)
                for m in sample_sizes.numpy()
            ]
        )
    )
    ubs_sectioned = th.hstack(
        (
            [
                GLR_xi_2_ub_with_eigs_sectioned(mu, eigvals, m, N)
                for m in sample_sizes.numpy()
            ]
        )
    )
    ubs_sectioned_weaker = th.hstack(
        (
            [
                GLR_xi_2_ub_with_eigs_sectioned_weaker(mu, eigvals, m, N)
                for m in sample_sizes.numpy()
            ]
        )
    )
    ubs_sectioned_synchro = th.hstack(
        (
            [
                GLR_xi_2_ub_with_eigs_sectioned_synchro(mu, eigvals, m, N)
                for m in sample_sizes.numpy()
            ]
        )
    )
    boop_lbs = th.stack(
        [
            N / m + (1 + mu * eigvals[N - (m - 1) :]).reciprocal().square().sum()
            for m in sample_sizes
        ]
    )
    maybe_lbs = th.hstack(
        [GLR_xi_2_lb_maybe_with_eigs(mu, eigvals, m, N) for m in sample_sizes]
    )
    maybe_lbs_largest_eigs = th.hstack(
        [GLR_xi_2_lb_maybe_with_eigs_last_m(mu, eigvals, m, N) for m in sample_sizes]
    )
    # maybe_lbs_without_eigs = vmap(lambda m: GLR_xi_2_lb_maybe(mu, eigvals.max(), m, N))(
    #     sample_sizes
    # )
    maybe_lbs_via_taylor = vmap(lambda m: GLR_xi_2_lb_maybe_with_taylor(mu, L, m, N))(
        sample_sizes
    )

    # Sample actul values
    rand_samp = th.randperm(N)[:max_sample_size]
    rand_samp_err = th.tensor(
        ([analytic_GLR_xi_2(L, None, rand_samp[:m], mu=mu) for m in sample_sizes])
    )
    # rand_sample_lbs = th.tensor(
    #     ([GLR_xi_2_lb(mu, L, rand_samp[:m]) for m in sample_sizes])
    # )
    # rand_sample_lbs_better = th.tensor(
    #     [GLR_xi_2_lb_with_fancy_basis(mu, L, rand_samp[:m]) for m in sample_sizes]
    # )
    if False:
        optimal_samp = greedy_glr_samples_only_fast(
            g,
            N,
            bandwidth=1,
            max_samples=max_sample_size,
            mu=mu,
            SNR=1e-10,
            normalization=None,
        )
        optimal_samp_err = th.tensor(
            [analytic_GLR_xi_2(L, None, optimal_samp[:m], mu=mu) for m in sample_sizes]
        )
        # optimal_sample_lbs = th.tensor(
        #     [GLR_xi_2_lb(mu, L, optimal_samp[:m]) for m in sample_sizes]
        # )
        optimal_sample_lbs_better = th.tensor(
            [
                GLR_xi_2_lb_with_fancy_basis(mu, L, optimal_samp[:m])
                for m in sample_sizes
            ]
        )
        # Neumann approx is bad
        # optimal_neumann_approx = th.hstack(
        #     [
        #         GLR_xi_2_neumann_sample(mu, L, optimal_samp[: m + 1], num_terms=200)
        #         for m in sample_sizes.numpy()
        #     ]
        # )
    dfs = []
    for name, vals in {
        # "upper_bound": ubs,
        # "upper_bound2": ubs2,
        # "upper_bound_fancier": ubs_sectioned,
        # "upper_bound_sectioned_weaker": ubs_sectioned_weaker,
        # "upper_bound_sectioned_synchro": ubs_sectioned_synchro,
        "boop_lb": boop_lbs,
        "global_lower_bound_approx": maybe_lbs,
        # "global_lower_bound_approx_largest_eigs": maybe_lbs_largest_eigs,
        # "global_maybe_lower_bound_approx_without_eigs": maybe_lbs_without_eigs,
        # "global_maybe_lower_bound_approx_via_taylor": maybe_lbs_via_taylor,
        "rand_samp_err": rand_samp_err,
        # "rand_sample_lbs": rand_sample_lbs,
        # "optimal_sample_lbs": optimal_sample_lbs,
        # "optimal_neumann_approx": optimal_neumann_approx,
        #
        # "optimal_samp_err": optimal_samp_err,
        # "optimal_sample_lbs_better": optimal_sample_lbs_better,
    }.items():
        dfs.append(
            pd.DataFrame(
                {
                    "type": name,
                    "xi_2": vals,
                    "Sample_Size": sample_sizes,
                    "is_err": name[-3:] == "err",
                }
            )
        )
    dfs = pd.concat(dfs).reset_index()

    fig, ax = plt.subplots()
    # sns.lineplot(x=sample_sizes, y=ubs, ax=ax)
    # sns.lineplot(x=sample_sizes, y=rand_sample_lbs, ax=ax)
    # sns.lineplot(x=sample_sizes, y=rand_samp_err, ax=ax, markers=True, legend="full")
    sns.lineplot(
        data=dfs, x="Sample_Size", y="xi_2", hue="type", markers="is_err", ax=ax
    )

    # ax.set_yscale("log")
    if graph_name is not None:
        ax.set_title(f"Xi_2 for {graph_name}: N={N}, mu={mu}")
    plt.show()


def GLR_plot_xi_2_bound_gap(
    g, N, mu, max_sample_size, min_sample_size=1, graph_name=None
):
    sample_sizes = th.arange(min_sample_size, max_sample_size)
    L = calc_laplacian(g, N, normalization=None)
    # Sample actul values
    optimal_samp = greedy_glr_samples_only_fast(
        g,
        N,
        bandwidth=1,
        max_samples=max_sample_size,
        mu=mu,
        SNR=1e-10,
        normalization=None,
    )
    # optimal_samp_err = th.tensor(
    #     [analytic_GLR_xi_2(L, None, optimal_samp[:m], mu=mu) for m in sample_sizes]
    # )
    optimal_samp_err_df = pd.DataFrame(
        [
            analytic_GLR_xi_2_breakdown(L, None, optimal_samp[:m], mu=mu)
            for m in sample_sizes
        ]
    )
    optimal_samp_err_df["sample_size"] = sample_sizes
    optimal_samp_err_df["lb"] = [
        (GLR_xi_2_lb_with_fancy_basis(mu, L, optimal_samp[:m]) - (N / m)).item()
        for m in sample_sizes
    ]
    optimal_samp_err_df["main_sq_loss"] = (
        optimal_samp_err_df["main_square"] - optimal_samp_err_df.lb
    )

    # optimal_samp_err_df
    # "main_diag", "off_diag_main_square", "off_square"
    #
    dfm = optimal_samp_err_df.drop(columns=["lb", "main_square"]).melt(
        "sample_size", var_name="err_type", value_name="xi_2_err_vals"
    )
    print(dfm)
    fig, ax = plt.subplots()
    # sns.lineplot(x=sample_sizes, y=ubs, ax=ax)
    # sns.lineplot(x=sample_sizes, y=rand_sample_lbs, ax=ax)
    # sns.lineplot(x=sample_sizes, y=rand_samp_err, ax=ax, markers=True, legend="full")
    sns.lineplot(data=dfm, x="sample_size", y="xi_2_err_vals", hue="err_type", ax=ax)

    # ax.set_yscale("log")
    if graph_name is not None:
        ax.set_title(f"Xi_2 for {graph_name}: N={N}, mu={mu}")
    plt.show()


def GLR_xi_2_eigen_breakdown(g, N, mu, max_sample_size):
    print("Calculating optimal sampling...")
    optimal_samp = greedy_glr_samples_only_fast(
        g,
        N,
        bandwidth=1,
        max_samples=max_sample_size,
        mu=mu,
        SNR=1e-10,
        normalization=None,
    )
    L = calc_laplacian(g, N, normalization=None)
    mu_L = mu * L.clone()
    res = []
    print("Interpreting the addition of each sample...")
    for i in tqdm(range(len(optimal_samp))):
        s = optimal_samp[i]
        samples_so_far = optimal_samp[: i + 1]
        mu_L[s, s] += 1
        evals, evecs = th.linalg.eigh(mu_L)
        res.append(
            th.stack(
                [
                    th.tensor(
                        (
                            eval,
                            eval * th.outer(evec, evec)[samples_so_far].square().sum(),
                        )
                    )
                    for eval, evec in zip(evals, evecs.T)
                ]
            )
        )
    return res


def connected_laplacian_pinv(L, normalization=None):
    if normalization is not None:
        evals, evecs = th.linalg.eigh(L)
        evals = evals[1:]
        evecs = evecs.T[1:]
        return vmap(lambda l, v: th.outer(v, v) / l)(evals, evecs).sum(dim=0)
    else:
        eps = th.rand_like(L[0, 0])
        n = L.shape[0]
        proj = th.eye(n).to(L.dtype) - th.ones_like(L) / n
        return (L + eps * th.ones_like(L)).inverse() @ proj


def connected_laplacian_pinv_arbitrary(L, normalization=None):
    if normalization is not None:
        evals, evecs = th.linalg.eigh(L)
        evals = evals[1:]
        evecs = evecs.T[1:]
        return vmap(lambda l, v: th.outer(v, v) / l)(evals, evecs).sum(dim=0)
    else:
        eps = 0.001
        n = L.shape[0]
        proj = th.eye(n).to(L.dtype) - th.ones_like(L) / n
        return (L + eps * th.ones_like(L)).inverse() @ proj


def GLR_calc_pseudoinverse(L, mu, sample, L_pinv=None):
    m = len(sample)
    N = L.shape[0]
    M = construct_sample_matrix(sample, N).to_dense().type(L.dtype)
    Y = M.T
    # This is inaccurate because of floating point. We need to explicitly calculate the pinv.
    if L_pinv is None:
        L_pinv = connected_laplacian_pinv(L)
    m_averager = th.eye(m, dtype=L.dtype) - (th.ones(m, m, dtype=L.dtype) / m)
    mu_F = (mu * th.eye(m, dtype=L.dtype)) + (
        m_averager @ L_pinv[sample][:, sample] @ m_averager
    )
    mu_F_approx = th.diag(mu + th.diag(L_pinv)[sample])
    # mu_F_approx = (mu * th.eye(m, dtype=L.dtype)) + (
    #     (m_averager @ (th.diag(th.diag(L_pinv[sample][:, sample]))) @ m_averager)
    # )
    Xplus_E_Xplus = (
        th.eye(N, dtype=L.dtype) - L_pinv @ M.T @ m_averager @ mu_F.inverse() @ M
    ) @ (L_pinv / mu)
    # Seems fine to approximate the inner left L_pinv by a diagonal but not the right one
    Xplus_E_Xplus_approx = (
        th.eye(N, dtype=L.dtype)
        - th.diag(th.diag(L_pinv)) @ M.T @ m_averager @ mu_F_approx.inverse() @ M
    ) @ (L_pinv / mu)
    Z_plus = th.ones(m, N, dtype=L.dtype) / m
    I_minus_YZplus = th.eye(N, dtype=L.dtype) - (Y @ Z_plus)
    ZZT_plus = th.ones(N, N, dtype=L.dtype) / m
    result_pseudoinverse = ZZT_plus + (
        I_minus_YZplus.T @ Xplus_E_Xplus @ I_minus_YZplus
    )
    result_pseudoinverse_approx = ZZT_plus + (
        I_minus_YZplus.T @ Xplus_E_Xplus_approx @ I_minus_YZplus
    )
    return result_pseudoinverse_approx


def analytic_GLR_xi_2_unsampled(L, _, sample, mu=0.01):
    L1 = (mu * L).clone()
    for s in sample:
        L1[s, s] += 1.0
    print(th.linalg.eigvalsh(L1).min())
    return L1.inverse().square().sum().item() * float(len(sample)) / float(L.shape[0])


def GLR_xi_1_lb(mu, U_k, eigvals, sample):
    _, k = U_k.shape
    mu_lambda = mu * eigvals
    norms = U_k[sample].square().sum(dim=0)
    return vmap(lambda mu_l, norm: (mu_l**2) / ((mu_l**2) + norm * (1 + 2 * mu_l)))(
        mu_lambda[1:k], norms[1:]
    ).sum()


# This uses kantorovich's inequality to give an upper bound
# for the expected error under uniform random sampling
# It only uses U_k to extract N,k
def GLR_xi_1_ub(mu, U_k, eigvals, sample_size):
    N, k = U_k.shape
    m = sample_size
    mu_lambda = mu * eigvals
    mu_alg = mu_lambda[1]
    # bounds for eigenvalues of (MTM+ mu * L).
    eig_lb = 0.5 * ((1 + mu_alg) - sqrt(((1 + mu_alg) ** 2) - (4 * mu_alg * m / N)))
    eig_ub = 1 + mu_lambda.max()
    ub_scale = 0.25 * (2 + ((eig_lb / eig_ub) ** 2) + ((eig_ub / eig_lb) ** 2))
    return vmap(lambda mu_l: ub_scale * (mu_l**2) / ((mu_l**2) + 2 * mu_l))(
        mu_lambda[1:k]
    ).sum()


def GLR_xi_1_ub_simpler(mu, U_k, eigvals, sample_size):
    N, k = U_k.shape
    m = sample_size
    mu_lambda = mu * eigvals
    mu_alg = mu_lambda[1]
    # bounds for eigenvalues of (MTM+ mu * L).
    eig_lb = 0.5 * ((1 + mu_alg) - sqrt(((1 + mu_alg) ** 2) - (4 * mu_alg * m / N)))
    # print(mu_alg)
    # print(eig_lb)
    return (eig_lb).reciprocal().square() * mu_lambda[1:k].square().reciprocal().sum()


def GLR_plot_xi_1_bounds(
    g, N, mu, bandwidth, max_sample_size, min_sample_size=1, graph_name=None
):
    sample_sizes = th.arange(min_sample_size, max_sample_size)
    L = calc_laplacian(g, N, normalization=None)
    eigvals, U = th.linalg.eigh(L)
    U_k = restrict_eigenbasis(U, bandwidth)
    # Calculate bounds
    average_ub = th.tensor(
        [GLR_xi_1_ub_simpler(mu, U_k, eigvals, m) for m in sample_sizes]
    )
    # Sample actul values
    rand_samp = th.randperm(N)[:max_sample_size]
    rand_samp_err = th.tensor(
        [analytic_GLR_xi_1(L, U_k, rand_samp[:m], mu=mu) for m in sample_sizes]
    )
    rand_sample_lbs = th.tensor(
        ([GLR_xi_1_lb(mu, U_k, eigvals, rand_samp[:m]) for m in sample_sizes])
    )
    if True:
        optimal_samp = greedy_glr_samples_only_fast(
            g,
            N,
            bandwidth=bandwidth,
            max_samples=max_sample_size,
            mu=mu,
            SNR=1e7,
            normalization=None,
        )
        optimal_samp_err = th.tensor(
            [analytic_GLR_xi_1(L, U_k, optimal_samp[:m], mu=mu) for m in sample_sizes]
        )
        optimal_sample_lbs = th.tensor(
            [GLR_xi_1_lb(mu, U_k, eigvals, optimal_samp[:m]) for m in sample_sizes]
        )
        # Neumann approx is bad
        # optimal_neumann_approx = th.hstack(
        #     [
        #         GLR_xi_2_neumann_sample(mu, L, optimal_samp[: m + 1], num_terms=200)
        #         for m in sample_sizes.numpy()
        #     ]
        # )
    dfs = []
    for name, vals in {
        # "upper_bound": average_ub,
        "rand_samp_err": rand_samp_err,
        "rand_sample_lbs": rand_sample_lbs,
        "optimal_samp_err": optimal_samp_err,
        "optimal_sample_lbs": optimal_sample_lbs,
        # "optimal_neumann_approx": optimal_neumann_approx,
    }.items():
        dfs.append(
            pd.DataFrame(
                {
                    "type": name,
                    "xi_1": vals,
                    "Sample_Size": sample_sizes,
                    "is_err": name[-3:] == "err",
                }
            )
        )
    dfs = pd.concat(dfs).reset_index()

    fig, ax = plt.subplots()
    # sns.lineplot(x=sample_sizes, y=ubs, ax=ax)
    # sns.lineplot(x=sample_sizes, y=rand_sample_lbs, ax=ax)
    # sns.lineplot(x=sample_sizes, y=rand_samp_err, ax=ax, markers=True, legend="full")
    sns.lineplot(
        data=dfs, x="Sample_Size", y="xi_1", hue="type", markers="is_err", ax=ax
    )

    # ax.set_yscale("log")
    if graph_name is not None:
        ax.set_title(f"Xi_1 for {graph_name}: N={N}, mu={mu}")
    plt.show()


def GLR_complete_graph_thm_coeffs(N):
    lambda_N = N
    b = 1 + ((N - 1) / (N**2))
    z = b + sqrt(8 * b * N)
    m = ceil(sqrt(2 * N / b))
    mu_ub = (sqrt(N / z) - 1) / lambda_N
    # A little computational:
    k = N // 10
    mu = 0.01 / sqrt(N)
    # print(b)
    # print(z)
    # print((1 + ((N - 1) / ((1 + mu * N) ** 2))))
    # print(f"mubd = {(sqrt((N-1)/(z-1)) - 1)/N}")
    tau_numerator = ((1 + ((N - 1) / ((1 + mu * N) ** 2))) - z) / N
    # print(tau_numerator)
    tau_denominator = (k + z - ((k - 1) * (((mu * N) / (1 + mu * N)) ** 2))) / k
    tau_GLR = tau_numerator / tau_denominator
    # print(10 * np.log10(tau_GLR))
    return {"m": m, "mu_ub": mu_ub, "tau_glr": tau_GLR}


# Calculates B(m_opt)
def calc_GLR_Bmopt(L):
    N = L.shape[0]
    rho_m = khatri_rao(L)
    r = rho_m[1]

    def xi_2_bnd_fn(m):
        return r * (N / m) + rho_m[m - 1]

    # This is B(m_opt)
    B_min = min([xi_2_bnd_fn(m) for m in th.arange(1, N - 1)])
    return B_min


def calc_GLR_Bmopt_and_sample_size(L):
    N = L.shape[0]
    rho_m = khatri_rao(L)
    r = rho_m[1]

    def xi_2_bnd_fn(m):
        return r * (N / m) + rho_m[m - 1]

    # This is B(m_opt)
    errs = [xi_2_bnd_fn(m) for m in th.arange(1, N - 1)]
    B_min = min(errs)
    samp_size = np.argmin(errs) + 1
    return (B_min, samp_size)


def calc_GLR_threshold(L, L_eigs, Bmopt, k, mu, debug=False, bl_noise=False):
    N = L.shape[0]
    # Calculate B(m_opt)
    # Calculate Coefficients
    # xi_1 is not a function of the noise type
    xi_1_N_avg = (1.0 - (1 / (1 + mu * L_eigs[:k]))).square().mean()
    xi_1_S_avg = 1 + (Bmopt / k)
    if bl_noise:
        xi_2_N_avg = (1 + mu * L_eigs[:k]).reciprocal().square().mean()
        xi_2_S_avg = Bmopt / k
    else:
        xi_2_N_avg = (1 + mu * L_eigs).reciprocal().square().mean()
        xi_2_S_avg = Bmopt / N
    # Delta_2/-Delta_1
    return (xi_2_N_avg - xi_2_S_avg) / (xi_1_S_avg - xi_1_N_avg)


def calc_GLR_threshold_journal(
    L, L_eigs, Bmopt, Bmopt_bl, k, mu, debug=False, bl_noise=False
):
    N = L.shape[0]
    # Calculate B(m_opt)
    # Calculate Coefficients
    # xi_1 is not a function of the noise type
    xi_1_N_avg = (1.0 - (1 / (1 + mu * L_eigs[:k]))).square().mean()
    xi_1_S_avg = 1 + (Bmopt_bl / k)
    if bl_noise:
        xi_2_N_avg = (1 + mu * L_eigs[:k]).reciprocal().square().mean()
        xi_2_S_avg = Bmopt_bl / k
    else:
        xi_2_N_avg = (1 + mu * L_eigs).reciprocal().square().mean()
        xi_2_S_avg = Bmopt / N
    # Delta_2/-Delta_1
    return (xi_2_N_avg - xi_2_S_avg) / (xi_1_S_avg - xi_1_N_avg)


# def calc_GLR_threshold2(L, Lpinv, L_eigs, k, mu, debug=False, bl_noise=False):
#     # Establish Constants
#     def ndiag(M):
#         return M.diag().diag()

#     lambda_N = L_eigs.max()
#     lambda_2 = L_eigs.sort(descending=False).values[1]
#     # L_pinv_eigs = L_eigs.reciprocal()
#     # L_pinv_eigs[0] = 0
#     N = L_eigs.shape[0]
#     r = ((lambda_N + lambda_2) ** 2) / (4 * lambda_2 * lambda_N)
#     # r = calc_GLR_r(L)
#     # b = calc_GLR_b(L)
#     b = (r / N) * th.trace(
#         ndiag(L @ L) @ (ndiag(Lpinv @ Lpinv) - ndiag(Lpinv).square())
#     )
#     # if bl_noise:
#     #     b = (k / N) + lambda_N.square() * (
#     #         L_pinv_eigs.square().mean() - L_pinv_eigs.mean().square()
#     #     )
#     # else:
#     #     b = 1 + lambda_N.square() * (
#     #         L_pinv_eigs.square().mean() - L_pinv_eigs.mean().square()
#     #     )
#     # Write down coefficients
#     if bl_noise:
#         # z = b + sqrt(4.4 * b * N)
#         # z = b + sqrt(4.4 * b * N)
#         z = (b + r) + 2 * sqrt(r * (b + 1) * N)
#     else:
#         z = (b + r) + 2 * sqrt(r * (b + 1) * N)
#         # z = b + sqrt(8 * b * N)
#     # xi_1 is not a function of the noise type
#     xi_1_N_avg = (1.0 - (1 / (1 + mu * L_eigs[:k]))).square().mean()
#     xi_1_S_avg = 1 + (z / k)
#     if bl_noise:
#         xi_2_N_avg = (1 + mu * L_eigs[:k]).reciprocal().square().mean()
#         xi_2_S_avg = z / k
#     else:
#         xi_2_N_avg = (1 + mu * L_eigs).reciprocal().square().mean()
#         xi_2_S_avg = z / N
#     # debug output
#     if debug:
#         print(f"b/N:{b/N}, b: {b}, z:{z}")
#         b_thresh = (
#             (sqrt(2 + (k / N)) - sqrt(2)) ** 2
#             if bl_noise
#             else ((sqrt(3) - sqrt(2)) ** 2)
#         )
#         print(f"is b/N small enough? {(b/N) < b_thresh }")
#         print(f"xi_1_N: {xi_1_N_avg}, xi_2_N: {xi_2_N_avg}")
#         mu_ub = (sqrt(N / z) - 1) / lambda_N
#         print(f"mu_ub:{mu_ub}, within bounds?:{mu_ub>mu}")

#     return (xi_2_N_avg - xi_2_S_avg) / (xi_1_S_avg - xi_1_N_avg)


# calculates r coefficient in GLR upper bound
def calc_GLR_r(L):
    L_eigs = th.linalg.eigvalsh(L)
    lambda_N = L_eigs.max()
    lambda_2 = L_eigs.sort(descending=False).values[1]
    r = ((lambda_N + lambda_2) ** 2) / (4 * lambda_2 * lambda_N)
    return r


def calc_GLR_r_bl(L, k):
    L_eigs = th.linalg.eigvalsh(L)
    lambda_k = L_eigs[k - 1]
    lambda_2 = L_eigs.sort(descending=False).values[1]
    r_bl = ((lambda_k + lambda_2) ** 2) / (4 * lambda_2 * lambda_k)
    return r_bl


# calculates b coefficient in GLR upper bound
def calc_GLR_b(L, vmap=False):
    L_eigs = th.linalg.eigvalsh(L)
    N = L_eigs.shape[0]
    if vmap:
        Lpinv = connected_laplacian_pinv_arbitrary(L)
    else:
        Lpinv = connected_laplacian_pinv(L)

    def ndiag(X):
        return X.diag().diag()

    r = calc_GLR_r(L)
    b = r / N * (ndiag(L @ L) @ (ndiag(Lpinv @ Lpinv) - ndiag(Lpinv).square())).trace()
    return b


def plot_GLR_b_distn(N, num_graphs=100):
    # graph_constructor = lambda: erdos_renyi_graph(N, 0.8)
    # graph_constructor = lambda: barabasi_albert_graph(N, 3)
    constructors_and_names = [
        {"con": lambda: clean_graph(erdos_renyi_graph(N, 0.8)), "name": "ER_0pt8"},
        {"con": lambda: clean_graph(barabasi_albert_graph(N, 3)), "name": "BA_3"},
        {"con": lambda: clean_graph(barabasi_albert_graph(N, 5)), "name": "BA_5"},
        # {"con": lambda: clean_graph(circle(n)), "name": "Ring"},
        # {
        #     "con": lambda: sbm_constructor(th.tensor(n // 10).repeat(10), 0.1, 0.7),
        #     "name": "SBM_50x10n_0pt1intra_0pt7inter",
        # },
    ]

    def calc_b_distn(con):
        laplacians = []
        while len(laplacians) < num_graphs:
            g, n = con()
            if n == N:
                L = calc_laplacian(g, n, normalization=None)
                laplacians.append(L)
        laplacians = th.stack(laplacians)
        return vmap(lambda L: calc_GLR_b(L, vmap=True))(laplacians)

    df = pd.concat(
        [
            pd.DataFrame(data={"graph": d["name"], "b": calc_b_distn(d["con"]).numpy()})
            for d in constructors_and_names
        ]
    )
    if True:
        return df
    else:
        fig, ax = plt.subplots()
        sns.histplot(data=df, x="b", hue="graph", ax=ax)
        # sns.kdeplot(data=df, x="b", hue="graph", log_scale=True, ax=ax)
        # ax.set_yscale("log")
        ax.set_title(f"Distribution of b for different graph classes with N = {N}")
        plt.show()


# calculates E[ sum(a[S]) * sum(X[S][:,S]) | |S| = m  ]
# where X has zero row and column-sums
def average_product_sum_Lpinv(samp_size, a, X):
    n = X.shape[0]
    z = list(
        itertools.accumulate(
            [(samp_size - i) / (n - i) for i in range(samp_size)], operator.mul
        )
    )
    z += [0, 0, 0, 0, 0, 0]
    c = (z[1] - z[2]) * np.trace(X)
    res = (z[0] - 3 * z[1] + 2 * z[2]) * np.diag(X) * a
    return np.sum(res) + np.sum(a) * c


# calculates E[ sum(a[S]) * (sum(X[S][:,S]) ** 2) | |S| = m  ]
# where X has zero row and column-sums
def average_product_sq_sum_Lpinv(samp_size, a, X):
    # constructs sum(a) * (sum(X) ** 2) with appropriate indicator coeffs
    n = X.shape[0]
    z = list(
        itertools.accumulate(
            [(samp_size - i) / (n - i) for i in range(samp_size)], operator.mul
        )
    )
    z += [0, 0, 0, 0, 0, 0]
    ones = np.ones_like(X)[0]
    # res = [0 for _ in nset]
    # build coeffs then multiply and sum them
    # consts:
    c = 0
    c += (z[2] - 2 * z[3] + z[4]) * (np.trace(X) ** 2)
    c += (z[1] - 7 * z[2] + 12 * z[3] - 6 * z[4]) * (np.sum(np.diag(X) ** 2))
    c += (2 * z[2] - 4 * z[3] + 2 * z[4]) * np.sum(X**2)
    c += 2 * z[4] * np.sum(X * np.diag(X))

    res = (z[0] - 15 * z[1] + 50 * z[2] - 60 * z[3] + 24 * z[4]) * (np.diag(X) ** 2)
    poly = 2 * z[1] - 8 * z[2] + 10 * z[3] - 4 * z[4]
    res += poly * (np.diag(X) * np.trace(X))
    res += 2 * poly * (X @ np.diag(X))
    res += 2 * poly * ((X**2) @ ones)
    # I think the following is always zero for L^+
    # res[i] += 2 * z[4] * np.sum(X * X[i])
    return np.sum(a * res) + np.sum(a) * c


def kantorCk(X, k):
    v = th.linalg.eigvalsh(X)
    num = 0
    denom = 0
    for i in range(k):
        num += (v[i + 1] + v[-(i + 1)]) ** 2
        denom += 4 * v[i + 1] * v[-(i + 1)]
    return num / denom


# For (6) in https://link.springer.com/chapter/10.1007/978-1-4615-4603-0_2
def kantorCk6(X, k):
    v = th.linalg.eigvalsh(X)
    acc = 0
    for i in range(k):
        num = (v[i + 1] + v[-(i + 1)]) ** 2
        denom = 4 * v[i + 1] * v[-(i + 1)]
        acc += num / denom
    return acc


def kantorCk6eig(X, k, eigs):
    v = eigs
    acc = 0
    # we actively ignore eig0
    for i in range(k):
        num = (v[i + 1] + v[-(i + 1)]) ** 2
        denom = 4 * v[i + 1] * v[-(i + 1)]
        acc += num / denom
    return acc


# calculates rho(m)
def khatri_rao_old(L):
    N = L.shape[0]
    l = th.linalg.eigvalsh(L)
    # construct lambda_hat, which repeats the N/2th eig
    l[0] = l[N // 2]
    l = l.sort(descending=False).values
    lhalf = l[: N // 2]
    lflip = reversed(l[N // 2 :])
    numerators = (lhalf + lflip) ** 2
    denominators = 4 * lhalf * lflip
    ratios = numerators / denominators
    ls = th.zeros(N + 1)
    # could be vectorised
    for i in range(1, N):
        if 2 * i > N:
            ls[i] = (2 * i - N) + ratios[: N - i].sum()
        else:
            ls[i] = ratios[:i].sum()
    ls[-1] = N
    return ls


# Returns a vector corresponding to the khatri-rao bounds
# for tr(PLPL^{+}) for a rank s projection P and rank N-1 Laplacian L
# for sample sizes of 0 to N-1
def khatri_rao(L):
    N = L.shape[0]
    l = th.linalg.eigvalsh(L)
    # Non-zero eigs:
    l = l[1:]
    omegas = vmap(lambda a, b: (a + b) ** 2 / (4 * a * b))(l, l.flip(0))
    # could be vectorised
    ls = th.zeros(N, dtype=L.dtype)
    # ls[0] = 0
    # s and k in the formula are s=k=i:
    t = N - 1
    for i in range(1, N):
        if t >= 2 * i:
            ls[i] = omegas[:i].sum()
        else:
            ls[i] = (i + i - t) + omegas[: t - i].sum()
    # This last value shouldn't be used
    # ls[-1] = N
    return ls


def koopy(reps=100):
    def koopy_inner():
        # goo, noo = clean_graph(barabasi_albert_graph(500, 3))
        goo, noo = sbm_constructor(th.tensor(500 // 10).repeat(10), 0.1, 0.7)
        # goo, noo = clean_graph(erdos_renyi_graph(500, 0.8))
        L = calc_laplacian(goo, noo, normalization=None)
        # Lpinv = connected_laplacian_pinv(L)
        # L_eigs = th.linalg.eigvalsh(L)
        b_old = calc_GLR_b(L)
        r_old = calc_GLR_r(L)
        # b = b_old * kantorCk(L, 11) / r_old
        b = b_old
        r = r_old
        # b = (kantorCk6(L, 22) - 22) / 22
        # b = r - 1
        # r = 1 + (Lpinv.trace() / noo) * (L_eigs.max() - L_eigs[1]) / 2
        # r = 1 + (sqrt(r_old - 1) * Lpinv.diag().mean()) * ((L_eigs.max() - L_eigs[1])/2)
        # r = s.kantorCk(L, 1)
        z = b + r + 2 * sqrt(r * (b + 1) * noo)
        LHS = (b + 1) / noo
        RHS = (sqrt(r + 0.1) - sqrt(r)) ** 2
        # print("======")
        # print(f"b:{b}, b_old:{b_old}, r:{r}, rold:{r_old} z:{z}")
        # print(f"{LHS}, {RHS}")
        # print(f"Bound satisfied? {LHS < RHS}")
        # print("=======")
        # return (kantorCk6(L, 22) + r * 22) < 500
        return LHS < RHS

    # koop = Parallel(n_jobs=16, verbose=5, backend="threading")(
    #     delayed(koopy_inner)() for _ in range(reps)
    # )
    koop = [koopy_inner() for _ in tqdm(range(reps))]
    prob_true = th.stack(koop).float().mean()
    print(f"Prob True: {prob_true}")
    return prob_true


def koopy2(reps=100):
    def koopy_inner():
        goo, noo = clean_graph(barabasi_albert_graph(500, 3))
        # goo, noo = sbm_constructor(th.tensor(500 // 10).repeat(10), 0.1, 0.7)
        # goo, noo = clean_graph(erdos_renyi_graph(500, 0.8))
        L = calc_laplacian(goo, noo, normalization=None)
        r = calc_GLR_r(L)
        m = math.ceil(sqrt(noo * r))
        return kantorCk6(L, m - 1) + r * (500 / m)
        # return (kantorCk6(L, 50) + r * (500 / 51)) < 500

    # koop = Parallel(n_jobs=16, verbose=5, backend="threading")(
    #     delayed(koopy_inner)() for _ in range(reps)
    # )
    koop = [koopy_inner() for _ in tqdm(range(reps))]
    prob_true = th.stack([k < 500 for k in koop]).float().mean()
    print(f"Prob True: {prob_true}")
    print(f"range: {th.stack([min(koop), max(koop)])}")

    return prob_true


def koopy3(reps=100):
    def koopy_inner():
        goo, noo = clean_graph(barabasi_albert_graph(500, 3))
        # goo, noo = sbm_constructor(th.tensor(500 // 10).repeat(10), 0.1, 0.7)
        # goo, noo = clean_graph(erdos_renyi_graph(500, 0.8))
        L = calc_laplacian(goo, noo, normalization=None)
        Bmopt = calc_GLR_Bmopt(L)
        return Bmopt

    def koopy_inner_catch():
        x = None
        while x is None:
            try:
                x = koopy_inner()
            except:
                pass
        return x

    # koop = Parallel(n_jobs=16, verbose=5, backend="threading")(
    #     delayed(koopy_inner)() for _ in range(reps)
    # )
    koop = [koopy_inner_catch() for _ in tqdm(range(reps))]

    koop_good = [k for k in koop if k < 500]
    prob_true = th.stack([k < 500 for k in koop]).float().mean()
    prob_true_bl = th.stack([k < 50 for k in koop]).float().mean()
    print(f"Prob True: {prob_true}")
    print(f"Prob True BL: {prob_true_bl}")
    print(f"range: {th.stack([min(koop), max(koop)])}")

    return prob_true


def check_bound_GLR(trials=100):
    res = []
    res2 = []
    resbl = []
    res2bl = []
    for _ in tqdm(range(trials)):
        try:
            g, N = clean_graph(barabasi_albert_graph(1000, 3))
            # g, N = sbm_constructor(th.tensor(1000 // 10).repeat(10), 0.1, 0.7)
            L = calc_laplacian(g, N, normalization=None)
            rho_m = khatri_rao(L)
            r = rho_m[1]
            r2 = rho_m[2] - r

            def xi_2_bnd_fn(m):
                # return r * ((N / m) + m - 1)
                # return (r * (N / m)) + (r2 * (m - 2)) + 1
                # return r * (N / m) + rho_m[m - 1]
                return r * (N / m) + rho_m[m - 1] + (1 - r)

            xi_2_bnd = th.tensor([xi_2_bnd_fn(m) for m in range(1, N)])
            res.append(xi_2_bnd.min() < (N))
            resbl.append(xi_2_bnd.min() < (N / 10))
            # res.append(xi_2_bnd.min())
            mopt_test = th.sqrt((r / r2) * th.tensor(N)).int().item()
            res2.append(xi_2_bnd_fn(mopt_test) < (N))
            res2bl.append(xi_2_bnd_fn(mopt_test) < (N / 10))
        except:
            pass
    print((th.tensor(res).sum() / trials, th.tensor(res2).sum() / trials))
    print((th.tensor(resbl).sum() / trials, th.tensor(res2bl).sum() / trials))


def check_bound_GLR(trials=100):
    res = []
    res2 = []
    resbl = []
    res2bl = []
    for _ in tqdm(range(trials)):
        try:
            # g, N = clean_graph(barabasi_albert_graph(1000, 3))
            g, N = sbm_constructor(th.tensor(1000 // 10).repeat(10), 0.1, 0.7)
            L = calc_laplacian(g, N, normalization=None)
            rho_m = khatri_rao(L)
            r = rho_m[1]
            r2 = rho_m[2] - r

            def xi_2_bnd_fn(m):
                # return r * ((N / m) + m - 1)
                return (r * (N / m)) + (r2 * (m - 2)) + 1
                # return r * (N / m) + rho_m[m - 1]
                # return r * (N / m) + rho_m[m - 1] + (1 - r)

            xi_2_bnd = th.tensor([xi_2_bnd_fn(m) for m in range(1, N)])
            res.append(xi_2_bnd.min() < (N))
            resbl.append(xi_2_bnd.min() < (N / 10))
            # res.append(xi_2_bnd.min())
            mopt_test = th.sqrt((r / r2) * th.tensor(N)).int().item()
            res2.append(xi_2_bnd_fn(mopt_test) < (N))
            res2bl.append(xi_2_bnd_fn(mopt_test) < (N / 10))
        except:
            pass
    print((th.tensor(res).sum() / trials, th.tensor(res2).sum() / trials))
    print((th.tensor(resbl).sum() / trials, th.tensor(res2bl).sum() / trials))


def main_fn():
    # render_thresholds_GLR(3000, bandlimited_noise=True)
    # render_thresholds_GLR(100, bandlimited_noise=False)
    # render_thresholds_GLR_multiple_sizes(
    #     [500, 1000, 2000, 3000], bandlimited_noise=False
    # )
    render_thresholds_GLR_multiple_sizes(
        [500, 1000, 2000, 3000], bandlimited_noise=False
    )
    render_thresholds_GLR_multiple_sizes(
        [500, 1000, 2000, 3000], bandlimited_noise=True
    )
    # render_thresholds_LS(num_nodes=100)
    # # render_thresholds_LS(num_nodes=2000)
    # # render_thresholds_LS(num_nodes=3000)
    # # render_thresholds_LS(num_nodes=500)
    # render_thresholds_LS(num_nodes=500)
    # render_MSE_LS(200)
    # render_MSE_LS(500, True)
    #
    # render_thresholds_LS_real()
    # render_MSE_LS_real(bl_noise=False, bl_signal=True)
    # render_MSE_GLR_with_bounds_real(bandlimited_noise=False, bandlimited_signal=True)
    # render_MSE_GLR_with_bounds(500, bandlimited_noise=False)
    # render_MSE_GLR_with_bounds(500, bandlimited_noise=True)


# -----------------------
# Compress Detection stuff!
# ------------------------
#


def analytic_plot(U_k, num_repeats=1000):
    N, k = U_k.shape
    res = []
    projbl = U_k @ U_k.T
    for sample_size in tqdm(range(k)):
        sets = th.stack([th.randperm(N)[:sample_size] for _ in range(num_repeats)])
        # vmap makes this slower??
        # loss = th.stack(
        #     [analytic_detection_loss_cached(U_k, projbl_diag, s).sum() for s in sets]
        # )
        loss = vmap(lambda s: analytic_detection_loss_cached(U_k, projbl, s).sum())(
            sets
        )
        res.append(loss.mean())
    res = th.stack(res)
    return res


def analytic_detection_loss_cached(U_k, projbl, S):
    # inner_proj = U_k[S, :].pinverse() @ U_k[S, :]
    # numer = (U_k @ U_k.T).diag()
    numer = projbl.diag()
    denom = (U_k @ U_k[S, :].pinverse() @ projbl[S, :]).diag()
    angle = th.sqrt((numer / denom) - 1)
    loss = 0.5 - (th.arctan(1 / angle)) / math.pi
    loss[S] = 0
    return loss


def analytic_detection_loss(U_k, S):
    inner_proj = U_k[S, :].pinverse() @ U_k[S, :]
    numer = (U_k @ U_k.T).diag()
    denom = (U_k @ inner_proj @ U_k.T).diag()
    angle = th.sqrt((numer / denom) - 1)

    def arccot(o):
        return th.arctan(1 / o)

    loss = 0.5 - arccot(angle) / math.pi
    loss[S] = 0
    return loss


if __name__ == "__main__":
    main_fn()


# def blah(n, k, p):
#     wsnx = nx.generators.random_graphs.watts_strogatz_graph(n=n, k=k, p=p)
#     g1, n1 = (from_networkx(wsnx).edge_index, 1000)
#     L1 = s.calc_laplacian(g1, n1, normalization=None)
#     a, b = th.linalg.eigvalsh(L1)[[1, -1]]
#     return b / a
