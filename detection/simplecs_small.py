
import torch as th
from torch import linalg, vmap
from torch.linalg import eigh, eigvals, lstsq, matrix_rank
from torch.nn.functional import normalize, relu, one_hot
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


## Calculate the laplacian eigenbasis
# Take n=bandwidth columns
def restrict_eigenbasis(eigenbasis, bandwidth):
    return eigenbasis[:, :bandwidth]


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


def calc_proj(U, k):
    U_k = restrict_eigenbasis(U, k)
    return U_k @ U_k.T
