import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import (
    to_undirected,
    remove_self_loops,
    add_self_loops,
    sort_edge_index,
    index_to_mask,
    mask_to_index,
)

from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import torchmetrics as tm
from torch_geometric.nn.models import GCN
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities import grad_norm

from torch_geometric.nn.conv.gcn_conv import gcn_norm

from tqdm import tqdm
import pandas as pd

from lg_parse import parse_method, parser_add_main_args
import sys

from logger import *
from dataset import load_dataset
from data_utils import eval_acc, eval_rocauc, load_fixed_splits
from eval import *


def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def aggregate_grad_layers(batch_norm_dict):
    d = batch_norm_dict

    gstr = "grad_2.0_norm"
    res = {}
    res["norm_total"] = d[gstr + "_total"]
    # List layer numbers
    constr = gstr + "/convs."
    layer_nums = [
        k.split(constr)[1].split(".")[0] for k in d.keys() if (k.startswith(constr))
    ]
    layer_nums = list(set(map(int, layer_nums)))
    for l in layer_nums:
        relevant_keys = [k for k in d.keys() if k.startswith(constr + str(l))]
        layer_norm = sqrt(sum([d[k] ** 2 for k in relevant_keys]))
        res["l" + str(l) + "_total"] = layer_norm
    return res

### Parse args ###
parser = argparse.ArgumentParser(
    description="Training Pipeline for Node Classification"
)
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

### Load and preprocess data ###
dataset = load_dataset(args.data_dir, args.dataset)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)
dataset.label = dataset.label.to(device)

#split_idx_lst = [dataset.load_fixed_splits() for _ in range(args.runs)]

### Moar accuracy
#dataset.graph['node_feat'] = dataset.graph['node_feat'].double()

### Basic information of datasets ###
n = dataset.graph["num_nodes"]
e = dataset.graph["edge_index"].shape[1]
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph["node_feat"].shape[1]

print(
    f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}"
)

dataset.graph["edge_index"] = to_undirected(dataset.graph["edge_index"])
dataset.graph["edge_index"], _ = remove_self_loops(dataset.graph["edge_index"])
dataset.graph["edge_index"], _ = add_self_loops(
    dataset.graph["edge_index"], num_nodes=n
)

dataset.graph["edge_index"], dataset.graph["node_feat"] = dataset.graph[
    "edge_index"
].to(device), dataset.graph["node_feat"].to(device)
data = Data(
    x=dataset.graph["node_feat"],
    y=dataset.label.squeeze(1),
    edge_index=dataset.graph["edge_index"],
)
data = data.to(device)

### Load method ###
model = parse_method(args, n, c, d, device)
#model = model.double()
# model = GCN(-1, args.hidden_channels, args.local_layers, c).to(device)


criterion = nn.CrossEntropyLoss()
eval_func = eval_acc
eval_obj = tm.Accuracy(task="multiclass", num_classes=c).to(device)
logger = Logger(args.runs, args)

cv_coeff = 0.1

model.train()
print("MODEL:", model)


pd_logs = []
### Training loop ###
for run in range(args.runs):
    split_idx = dataset.load_fixed_splits() if hasattr(dataset, 'load_fixed_splits') else dataset.get_idx_split() 
    #split_idx = split_idx_lst[run]
    train_idx = split_idx["train"].to(device)
    model.reset_parameters()
    optimizer = torch.optim.Adam(
        model.parameters(), weight_decay=args.weight_decay, lr=args.lr
    )
    best_val = float("-inf")
    best_test = float("-inf")
    if args.save_model:
        save_model(args, model, optimizer, run)

    # we poison the test labels; this uses more GPU memory from duplication
    # but more easily lets us see our code is correct.
    not_train_mask = ~index_to_mask(split_idx["train"], size=data.num_nodes)
    not_valid_mask = ~index_to_mask(split_idx["valid"], size=data.num_nodes)
    train_data = data.clone()
    valid_data = data.clone()
    train_data.y[not_train_mask] = 0
    valid_data.y[not_valid_mask] = 0

    #precompute first mult:
    normalised_edge_index, normalised_edge_weight = gcn_norm(
            train_data.edge_index, None, train_data.x.size(model.local_convs[0].node_dim),
            model.local_convs[0].improved, model.local_convs[0].add_self_loops, model.local_convs[0].flow, train_data.x.dtype)

    train_data.DADx = model.local_convs[0].propagate(normalised_edge_index, 
                                x=train_data.x,
                                edge_weight=normalised_edge_weight).detach()
    del normalised_edge_index
    del normalised_edge_weight
    torch.cuda.empty_cache()

    train_data.to(device)

    train_loader = NeighborLoader(
        train_data,
        input_nodes=split_idx["train"],
        #num_neighbors=[5, 5, 5],
        #batch_size=4000,
        num_neighbors=[data.num_nodes] * 100,
        batch_size=data.num_nodes,
        num_workers=2,
        pin_memory=True,
    )
    """
    valid_loader = NeighborLoader(
        valid_data,
        input_nodes=split_idx["valid"],
        num_neighbors=[data.num_nodes] * 100,
        batch_size=data.num_nodes,
        num_workers=2,
    )
    test_loader = NeighborLoader(
        data,
        input_nodes=split_idx["test"],
        num_neighbors=[data.num_nodes] * 100,
        batch_size=data.num_nodes,
        num_workers=2,
    )
    """


    train_acc = tm.Accuracy(task="multiclass", num_classes=c).to(device)
    valid_acc = train_acc.clone()
    test_acc = train_acc.clone()

    #for tqdm visualisation:
    train_len = len(train_loader)

    for epoch in range(args.epochs):

        model.train()

        # The following are true:
        # torch.all(sort_edge_index(batch.n_id.to(device)[batch.edge_index]) == sort_edge_index(data.edge_index))
        # torch.all(data.x[batch.n_id] == batch.x)
        # So we expect that
        # out1[batch.n_id] == out
        # Also:
        # torch.all(batch.x[:split_size] == data.x[split_idx["train"][batch.input_id]]) and
        # torch.all(batch.x[:split_size] == data.x[train_loader.input_nodes[batch.input_id]])
        # that is: input_ids are indexed into the seed nodes.
        for batch_idx, batch in tqdm(enumerate(train_loader), leave=False, total=train_len):
            # Some properties for indexing:
            split_size = batch.input_id.shape[0]
            global_input_nodes = split_idx['train'][batch.input_id]
            #we have that batch.x[:split_size] == train_data.x[global_input_nodes]

            if True:
                #We now calculate the norms of three sets of gradients:
                # 1) minibatch loss - fullbatch loss gradient norms
                # 2) cv minibatch loss - fullbatch loss gradient norms
                # 3) fullbatch gradient norms (for normalization)
                # We log these and then finally step our optimizer forward.

                #(1)
                optimizer.zero_grad()

                out = model(batch.x, batch.edge_index)
                minibatch_loss = criterion(out[:split_size], batch.y[:split_size])

                full_out = model(train_data.x, train_data.edge_index)
                fullbatch_loss = criterion(full_out[global_input_nodes], train_data.y[global_input_nodes])

                loss = minibatch_loss - fullbatch_loss
                print(f"fb vs minibatch loss: {loss}")
                print(f"loss dtype: {loss.dtype}")
                loss.backward()


                norms = grad_norm(model, norm_type=2)
                dbatch_minus_full = aggregate_grad_layers(norms)

                #(2)
                optimizer.zero_grad()

                outp = model.precomputed_forward(batch.DADx, batch.edge_index)
                minibatch_with_precomp_loss = criterion(outp[:split_size], batch.y[:split_size])

                full_out = model(train_data.x, train_data.edge_index)
                fullbatch_loss = criterion(full_out[global_input_nodes], train_data.y[global_input_nodes])

                loss = minibatch_with_precomp_loss - fullbatch_loss
                loss.backward()

                norms = grad_norm(model, norm_type=2)
                dprecomp_minus_full = aggregate_grad_layers(norms)


                #(3)
                optimizer.zero_grad()
                full_out = model(train_data.x, train_data.edge_index)
                out = full_out
                fullbatch_loss = criterion(full_out[global_input_nodes], train_data.y[global_input_nodes])

                loss = fullbatch_loss
                loss.backward()
                
                train_acc.update(out[global_input_nodes], train_data.y[global_input_nodes])

                norms = grad_norm(model, norm_type=2)
                dfull = aggregate_grad_layers(norms)

                #log
                #compute stats:
                dlmc_rel = {k: (dbatch_minus_full[k] / dfull[k]).item() for k in dfull.keys()}
                dlmc_rel_precomp = {k: (dprecomp_minus_full[k] / dfull[k]).item() for k in dfull.keys()}
                #relabel:
                dlmc_rel = {f"lmc_rel_err/{k}": v for k, v in dlmc_rel.items()}
                dlmc_rel_precomp = {f"lmc_rel_err_precomp/{k}": v for k, v in dlmc_rel_precomp.items()}
                gen_dict = {"batch_idx": batch_idx, "epoch": epoch, "step": batch_idx + epoch * train_len}
                pd_logs.append( dlmc_rel | dlmc_rel_precomp | gen_dict )

            else:
                optimizer.zero_grad()

                #out = model.precomputed_forward(batch.DADx, batch.edge_index)
                #loss = criterion(out[:split_size], batch.y[:split_size])

                #out = model(batch.x, batch.edge_index)
                #loss = criterion(out[:split_size], batch.y[:split_size])

                out = model(train_data.x, train_data.edge_index)
                loss = criterion(out[global_input_nodes], train_data.y[global_input_nodes])

                #loss = minibatch_loss

                loss.backward()
                train_acc.update(out[global_input_nodes], train_data.y[global_input_nodes])
                

            #Now step        
            #train_acc.update(out[:split_size], batch.y[:split_size])
            optimizer.step()

        """
        for valid_batch in valid_loader:
            valid_out = model(valid_batch.x, valid_batch.edge_index)
            valid_split_size = valid_batch.input_id.shape[0]
            valid_acc.update(
                valid_out[:valid_split_size], valid_batch.y[:valid_split_size]
            )
            valid_loss = criterion(
                valid_out[:valid_split_size], valid_batch.y[:valid_split_size]
            ).detach()
            del valid_batch

        for test_batch in test_loader:
            test_out = model(test_batch.x, test_batch.edge_index)
            test_split_size = test_batch.input_id.shape[0]
            test_acc.update(
                test_out[:test_split_size], test_batch.y[:test_split_size]
            )
            del test_batch
        """

        out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])
        valid_loss = criterion(out[split_idx['valid']], dataset.label[split_idx['valid']].squeeze(1))
        valid_acc.update(out[split_idx['valid']], dataset.label[split_idx['valid']].squeeze(1))
        test_acc.update(out[split_idx['test']], dataset.label[split_idx['test']].squeeze(1))


        result = (
            train_acc.compute().detach(),
            valid_acc.compute().detach(),
            test_acc.compute().detach(),
            valid_loss.detach(),
            (),
        )

        logger.add_result(run, result[:-1])

        if result[1] > best_val:
            best_val = result[1]
            best_test = result[2]
            if args.save_model:
                save_model(args, model, optimizer, run)

        train_acc.reset()
        valid_acc.reset()
        test_acc.reset()
        if epoch % args.display_step == 0:
            print(
                f"Epoch: {epoch:02d}, "
                f"Loss: {loss:.4f}, "
                f"Train: {100 * result[0]:.2f}%, "
                f"Valid: {100 * result[1]:.2f}%, "
                f"Test: {100 * result[2]:.2f}%, "
                f"Best Valid: {100 * best_val:.2f}%, "
                f"Best Test: {100 * best_test:.2f}%"
            )
    logger.print_statistics(run)

    results = logger.print_statistics()
    ### Save results ###
    save_result(args, results)

pd.DataFrame(pd_logs).to_csv('lightning_logs/metrics.csv')
