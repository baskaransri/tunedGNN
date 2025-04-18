import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops, sort_edge_index

from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import torchmetrics as tm
from torch_geometric.nn.models import GCN


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

### Parse args ###
parser = argparse.ArgumentParser(description='Training Pipeline for Node Classification')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

### Load and preprocess data ###
dataset = load_dataset(args.data_dir, args.dataset)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)
dataset.label = dataset.label.to(device)

split_idx_lst = [dataset.load_fixed_splits() for _ in range(args.runs)]

### Basic information of datasets ###
n = dataset.graph['num_nodes']
e = dataset.graph['edge_index'].shape[1]
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

print(f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")

dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)

dataset.graph['edge_index'], dataset.graph['node_feat'] = dataset.graph['edge_index'].to(device), dataset.graph['node_feat'].to(device)
data = Data(x = dataset.graph['node_feat'], y = dataset.label.squeeze(1), edge_index = dataset.graph['edge_index'])
data = data.to(device)

### Load method ###
model = parse_method(args, n, c, d, device)
#model = GCN(-1, args.hidden_channels, args.local_layers, c).to(device)


criterion = nn.CrossEntropyLoss()
eval_func = eval_acc
eval_obj = tm.Accuracy(task="multiclass", num_classes=c).to(device)
logger = Logger(args.runs, args)

model.train()
print('MODEL:', model)

### Training loop ###
for run in range(args.runs):
    split_idx = split_idx_lst[run]
    train_idx = split_idx['train'].to(device)
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(),weight_decay=args.weight_decay, lr=args.lr)
    best_val = float('-inf')
    best_test = float('-inf')
    if args.save_model:
        save_model(args, model, optimizer, run)

    train_loader = NeighborLoader(
            data,
            input_nodes = split_idx['train'],
            num_neighbors = [data.num_nodes] * 100,
            batch_size = data.num_nodes,
            num_workers = 2,
            pin_memory = True
            )
    valid_loader = NeighborLoader(
            data,
            input_nodes = split_idx['valid'],
            num_neighbors = [data.num_nodes] * 100,
            batch_size = data.num_nodes,
            num_workers = 2,
            )
    test_loader = NeighborLoader(
            data,
            input_nodes = split_idx['test'],
            num_neighbors = [data.num_nodes] * 100,
            batch_size = data.num_nodes,
            num_workers = 2,
            )

    train_acc = tm.Accuracy(task="multiclass", num_classes=c).to(device)
    valid_acc = train_acc.clone()
    test_acc  = train_acc.clone()

    prev_t_acc = None
    for epoch in range(args.epochs):

        model.train()
        optimizer.zero_grad()

        #The following are true:
        # torch.all(sort_edge_index(batch.n_id.to(device)[batch.edge_index]) == sort_edge_index(data.edge_index))
        # torch.all(data.x[batch.n_id] == batch.x)
        # So we expect that
        # out1[batch.n_id] == out

        for batch, valid_batch, test_batch in zip(train_loader, valid_loader, test_loader):
            #batch = next(iter(train_loader))
            train_batch = batch
            out = model(batch.x, batch.edge_index)
            split_size = train_idx.shape[0]
            loss = criterion(out[:split_size], batch.y[:split_size])

            t_acc = train_acc(out[:split_size], batch.y[:split_size])
            if prev_t_acc is None:
                print(f"tm.update result diff: Need at least one epoch")
            else:
                print(f"tm.update result diff: {prev_t_acc - t_acc}")
            loss.backward()
            optimizer.step()

            result_old  = evaluate(model, dataset, split_idx, eval_func, criterion, args)
            result      = evaluate_dl(model, train_batch, valid_batch, test_batch,   split_idx, eval_obj, criterion, args)
            print(f"result diff: {(torch.tensor(result[:-1]) - torch.tensor(result_old[:-1])).norm()}")
            prev_t_acc = result[0]

            logger.add_result(run, result[:-1])

            if result[1] > best_val:
                best_val = result[1]
                best_test = result[2]
                if args.save_model:
                    save_model(args, model, optimizer, run)

        for valid_batch in valid_loader:
            valid_out = model(valid_batch.x, valid_batch.edge_index)
            valid_split_size = valid_batch.input_id.shape[0]
            valid_acc.update(
                    valid_out[:valid_split_size], 
                    valid_batch.y[:valid_split_size])
            valid_loss = criterion(
                    valid_out[:valid_split_size], valid_batch.y[:valid_split_size])

        for test_batch in test_loader:
            test_out = model(test_batch.x, test_batch.edge_index)
            test_split_size = test_batch.input_id.shape[0]
            test_acc.update(
                    test_out[:test_split_size], 
                    test_batch.y[:test_split_size])

        print(f"tm.update valid result diff: {result[1] - valid_acc.compute()}")
        print(f"tm.update test result diff: {result[2] - test_acc.compute()}")
            

        train_acc.reset()
        valid_acc.reset()
        test_acc.reset()
        if epoch % args.display_step == 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * result[0]:.2f}%, '
                  f'Valid: {100 * result[1]:.2f}%, '
                  f'Test: {100 * result[2]:.2f}%, '
                  f'Best Valid: {100 * best_val:.2f}%, '
                  f'Best Test: {100 * best_test:.2f}%')
    logger.print_statistics(run)

results = logger.print_statistics()
### Save results ###
save_result(args, results)

