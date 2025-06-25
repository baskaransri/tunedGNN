# import simplecs as s
#
import pandas as pd

import torch as th
from torch_geometric.utils import to_dense_adj
from torch_geometric.loader import NeighborLoader
from torch.nn.functional import relu
import torch.nn as nn
from torch_geometric.datasets import Planetoid, Reddit, Flickr
from ogb.nodeproppred import NodePropPredDataset

from torch_geometric.data import Data

import classification_sampling as cl

from tqdm import tqdm, trange
import gc

# dataset = Planetoid(root="/tmp/Citeseer", name="Cora")[0]
# dataset = Planetoid(root="/tmp/Citeseer", name="Citeseer")[0]
# Reddit is too big for my machine!
# dataset = Reddit(root="/tmp/Reddit")[0]
dataset = Flickr(root="/tmp/Reddit")[0]

# dataset = NodePropPredDataset(root="/tmp/ogbn-arxiv", name="ogbn-arxiv")[0][0]
# dataset = NodePropPredDataset(root="/tmp/ogbn-proteins", name="ogbn-proteins")[0][0]
# dataset = Data(
#     edge_index=th.from_numpy(dataset["edge_index"]),
#     x=th.from_numpy(dataset["node_feat"]),
#     num_nodes=dataset["num_nodes"],
# )


#
def check1():
    A = to_dense_adj(dataset.edge_index)[0]
    d = ()
    return A


def gcn_check(num_nodes=500):

    data, model = cl.mk_tasks_sgc(
        feat_dim=64,
        hidden_dim=32,
        signal_type="lap_pinv",
        num_conv_layers=2,
        num_nodes=num_nodes,
    )
    return gcn_nloader_check(data, model)


def sgc_check(num_nodes=500):

    data, model = cl.mk_tasks_sgc(
        feat_dim=64,
        hidden_dim=32,
        signal_type="lap_pinv",
        num_conv_layers=2,
        num_nodes=num_nodes,
    )
    return sgc_nloader_check(data, model)


def sgc_nloader_check(data, model):
    criterion = nn.CrossEntropyLoss()

    train_idx = th.randperm(data.num_nodes)[: data.num_nodes // 5]

    print("Running full graph convolutions...")
    data["Ax"] = cl.weightless_conv(model, data.x, data.edge_index)
    gc.collect()
    print("First done.")
    data["AAx"] = cl.weightless_conv(model, data.Ax, data.edge_index)
    gc.collect()
    print("Second done!")

    train_loader = NeighborLoader(
        data,
        input_nodes=train_idx,
        num_neighbors=[5, 5, 5],
        batch_size=100,
        # num_neighbors=[data.num_nodes] * 100,
        # batch_size=data.num_nodes,
        # num_workers=2,
        # pin_memory=True,
    )
    train_len = len(train_loader)
    res = []
    print("Running batches and comparisons")
    for batch_idx, batch in tqdm(enumerate(train_loader), leave=False, total=train_len):
        split_size = batch.input_id.shape[0]
        global_input_nodes = train_idx[batch.input_id]

        full_out = data.AAx[global_input_nodes]
        precomp_out = cl.weightless_conv(model, data.Ax, batch.edge_index)[:split_size]
        mini_out_l1 = cl.weightless_conv(model, data.x, batch.edge_index)
        mini_out = cl.weightless_conv(model, mini_out_l1, batch.edge_index)[:split_size]
        a = {"mini vs full": th.norm(full_out - mini_out).item()}
        b = {"precomp vs full": th.norm(full_out - precomp_out).item()}
        res.append(a | b)
    df = pd.DataFrame(res)
    df["Mini worse"] = df["mini vs full"] > df["precomp vs full"]
    return df


def gcn_nloader_check(data, model):
    criterion = nn.CrossEntropyLoss()

    train_idx = th.randperm(data.num_nodes)[: data.num_nodes // 5]

    data["Ax"] = cl.weightless_conv(model, data.x, data.edge_index)
    data["AAx"] = cl.weightless_conv(model, relu(data.Ax), data.edge_index)

    train_loader = NeighborLoader(
        data,
        input_nodes=train_idx,
        num_neighbors=[5, 5, 5],
        batch_size=100,
        # num_neighbors=[data.num_nodes] * 100,
        # batch_size=data.num_nodes,
        # num_workers=2,
        # pin_memory=True,
    )
    train_len = len(train_loader)
    res = []
    for batch_idx, batch in tqdm(enumerate(train_loader), leave=False, total=train_len):
        split_size = batch.input_id.shape[0]
        global_input_nodes = train_idx[batch.input_id]

        full_out = data.AAx[global_input_nodes]
        precomp_out = cl.weightless_conv(model, relu(data.Ax), batch.edge_index)[
            :split_size
        ]
        mini_out_l1 = cl.weightless_conv(model, data.x, batch.edge_index)
        mini_out = cl.weightless_conv(model, relu(mini_out_l1), batch.edge_index)[
            :split_size
        ]
        a = {"mini vs full": th.norm(full_out - mini_out).item()}
        b = {"precomp vs full": th.norm(full_out - precomp_out).item()}
        res.append(a | b)
    df = pd.DataFrame(res)
    df["Mini worse"] = df["mini vs full"] > df["precomp vs full"]
    return df


def check(n, samp, p=0.3):
    # g1, n1 = s.connected_erdos_renyi_graph(n, p)
    # g1, n1 = s.clean_graph(s.barabasi_albert_graph(n,5))
    g1 = dataset.edge_index
    # g1 = th.from_numpy(dataset['edge_index'])

    A = to_dense_adj(g1)[0]
    Asub = A[:samp][:, :samp]
    A = th.eye(A.shape[0]) + A
    Asub = th.eye(Asub.shape[0]) + Asub
    d = A @ th.ones(A.shape[0])
    dsub = Asub @ th.ones(Asub.shape[0])
    Dsqrtinv = d.rsqrt().diag()
    DAD = Dsqrtinv @ A @ Dsqrtinv
    DADsub = dsub.rsqrt().diag() @ Asub @ dsub.rsqrt().diag()

    truth = (DAD @ DAD)[:samp][:, :samp]
    full_approx = DADsub @ DADsub
    part_approx = DADsub @ DAD[:samp][:, :samp]
    miniapprox = DAD[:samp][:, :samp] @ DAD[:samp][:, :samp]
    # errs:
    full_err = (full_approx - truth).norm()
    part_err = (part_approx - truth).norm()
    miniapprox = (miniapprox - truth).norm()

    print("full, part, mini(nobatch):")
    print((full_err, part_err, miniapprox))

    # X = th.randn(A.shape[0],10)
    # X = DAD @ DAD @ DAD @ X
    X = dataset.x
    truth = (DAD @ relu(DAD @ X))[:samp]
    full_approx = DADsub @ relu(DADsub @ X[:samp])
    part_approx = DADsub @ relu(DAD @ X)[:samp]

    full_err = (full_approx - truth).norm()
    part_err = (part_approx - truth).norm()

    print("full_err, part_err with weights:")
    print((full_err, part_err))
    breakpoint()


if __name__ == "__main__":
    for _ in range(3):
        print("++++++++++++++++++++")
        check(1000, 100)
        print("++++++++++++++++++++")
