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

# from torch_geometric.nn.models import GCN
import lightning as L
from lightning.pytorch.utilities import grad_norm
from lightning.pytorch.utilities.combined_loader import CombinedLoader


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

split_idx_lst = [dataset.load_fixed_splits() for _ in range(args.runs)]

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


criterion = nn.CrossEntropyLoss()
eval_func = eval_acc


class LightningGCN(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model = parse_method(args, n, c, d, device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = tm.Accuracy(task="multiclass", num_classes=c)
        self.valid_acc = self.train_acc.clone()
        self.test_acc = self.train_acc.clone()

    def forward_on_split(self, batch):
        out = self.model(batch.x, batch.edge_index)
        split_size = batch.input_id.shape[0]
        return (out[:split_size], batch.y[:split_size])

    def training_step(self, batch):
        out, labels = self.forward_on_split(batch)
        loss = self.loss_fn(out, labels)
        self.train_acc(out, labels)
        self.log("train_loss", loss, prog_bar=True)
        self.log(
            "train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch):
        out, labels = self.forward_on_split(batch)
        self.valid_acc(out, labels)
        self.log(
            "valid_acc", self.valid_acc, on_step=False, on_epoch=True, prog_bar=True
        )

    def test_step(self, batch):
        out, labels = self.forward_on_split(batch)
        self.test_acc(out, labels)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), weight_decay=args.weight_decay, lr=args.lr
        )
        return optimizer


### Training loop ###
for run in range(args.runs):
    split_idx = split_idx_lst[run]
    train_idx = split_idx["train"].to(device)

    # we poison the test labels; this uses more GPU memory from duplication
    # but more easily lets us see our code is correct.
    not_train_mask = ~index_to_mask(split_idx["train"], size=data.num_nodes)
    not_valid_mask = ~index_to_mask(split_idx["valid"], size=data.num_nodes)
    train_data = data.clone()
    valid_data = data.clone()
    train_data.y[not_train_mask] = 0
    valid_data.y[not_valid_mask] = 0

    train_loader_minibatch = NeighborLoader(
        train_data,
        input_nodes=split_idx["train"],
        num_neighbors=[5, 5, 5],
        batch_size=4000,
        num_workers=2,
        pin_memory=True,
    )
    train_loader_fullbatch = NeighborLoader(
        train_data,
        input_nodes=split_idx["train"],
        num_neighbors=[data.num_nodes] * 100,
        batch_size=data.num_nodes,
        num_workers=2,
    )
    train_loader_combo = CombinedLoader(
        {"partial": train_loader_minibatch, "full": train_loader_fullbatch},
        mode="max_size_cycle",
    )
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

    lightning_model = LightningGCN()
    trainer = L.Trainer(max_epochs=args.epochs)
    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_loader_combo,
        val_dataloaders=valid_loader,
    )
    print("Testing...")
    L.Trainer(accelerator="cpu").test(lightning_model, dataloaders=test_loader)
    x = evaluate(
        lightning_model.model.to("cuda"), dataset, split_idx, eval_func, criterion, args
    )
    print(f"Train acc: {x[0]}")
    print(f"Valid acc: {x[1]}")
    print(f"Test acc: {x[2]}")
