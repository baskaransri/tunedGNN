#!/usr/bin/env python3
# General imports
import torch as th
from einops import rearrange

# Graph imports
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
import networkx as nx
from torch_geometric.data import Data

# Testing imports
from hypothesis import (
    given,
    assume,
    settings,
    Verbosity,
    strategies as hstrat,
)
import hypothesis.extra.numpy as hen

from typing import Optional, Tuple, Callable, Dict
import torchtyping
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked


## Utility for Graph Constructors
def pyg_is_connected(g, num_nodes):
    if num_nodes <= 10:
        adj = to_dense_adj(g, max_num_nodes=num_nodes)
        adj_pow = th.linalg.matrix_power(adj, num_nodes)
        return (adj_pow.abs() > 0).all()
    else:
        nxG = to_networkx(Data(edge_index=g, num_nodes=num_nodes))
        nxG = nxG.to_undirected()
        return nx.is_connected(nxG)


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
