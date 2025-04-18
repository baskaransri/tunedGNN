import torch
import torch.nn.functional as F
import torchmetrics as tm

@torch.no_grad()
def evaluate_tm(model, dataset, split_idx, eval_obj, criterion, args,result=None):
    if result is not None:
        out = result
    else:
        model.eval()
        out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])
    train_acc = eval_obj(
        out[split_idx['train']], dataset.label[split_idx['train']].squeeze(1))
    valid_acc = eval_obj(
        out[split_idx['valid']], dataset.label[split_idx['valid']].squeeze(1))
    test_acc = eval_obj(
        out[split_idx['test']], dataset.label[split_idx['test']].squeeze(1))

    if args.dataset in ('questions'):
        if dataset.label.shape[1] == 1:
            true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
        else:
            true_label = dataset.label
        valid_loss = criterion(out[split_idx['valid']], true_label.squeeze(1)[
            split_idx['valid']].to(torch.float))
    else:
        #out = F.log_softmax(out, dim=1)
        valid_loss = criterion(
            out[split_idx['valid']], dataset.label.squeeze(1)[split_idx['valid']])

    return train_acc, valid_acc, test_acc, valid_loss, out

@torch.no_grad()
def evaluate_dl(model, train_batch, valid_batch, test_batch, split_idx, eval_obj, criterion, args):
    model.eval()
    train_out = model(train_batch.x, train_batch.edge_index)
    valid_out = model(valid_batch.x, valid_batch.edge_index)
    test_out = model(test_batch.x, test_batch.edge_index)

    train_split_size = train_batch.input_id.shape[0]
    train_acc = eval_obj(
            train_out[:train_split_size], 
            train_batch.y[:train_split_size])
    valid_split_size = valid_batch.input_id.shape[0]
    valid_acc = eval_obj(
            valid_out[:valid_split_size], 
            valid_batch.y[:valid_split_size])
    test_split_size = test_batch.input_id.shape[0]
    test_acc = eval_obj(
            test_out[:test_split_size], 
            test_batch.y[:test_split_size])

    valid_loss = criterion(
            valid_out[:valid_split_size], valid_batch.y[:valid_split_size])

    return train_acc, valid_acc, test_acc, valid_loss, ()

@torch.no_grad()
def evaluate(model, dataset, split_idx, eval_func, criterion, args, result=None):
    if result is not None:
        out = result
    else:
        model.eval()
        out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])

    train_acc = eval_func(
        dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        dataset.label[split_idx['test']], out[split_idx['test']])

    if args.dataset in ('questions'):
        if dataset.label.shape[1] == 1:
            true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
        else:
            true_label = dataset.label
        valid_loss = criterion(out[split_idx['valid']], true_label.squeeze(1)[
            split_idx['valid']].to(torch.float))
    else:
        #out = F.log_softmax(out, dim=1)
        valid_loss = criterion(
            out[split_idx['valid']], dataset.label.squeeze(1)[split_idx['valid']])

    return train_acc, valid_acc, test_acc, valid_loss, out

@torch.no_grad()
def evaluate_cpu(model, dataset, split_idx, eval_func, criterion, args, device, result=None):
    if result is not None:
        out = result
    else:
        model.eval()

    model.to(torch.device("cpu"))
    dataset.label = dataset.label.to(torch.device("cpu"))
    edge_index, x = dataset.graph['edge_index'], dataset.graph['node_feat']
    out = model(x, edge_index)

    train_acc = eval_func(
        dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        dataset.label[split_idx['test']], out[split_idx['test']])
    if args.dataset in ('questions'):
        if dataset.label.shape[1] == 1:
            true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
        else:
            true_label = dataset.label
        valid_loss = criterion(out[split_idx['valid']], true_label.squeeze(1)[
            split_idx['valid']].to(torch.float))
    else:
        #out = F.log_softmax(out, dim=1)
        valid_loss = criterion(
            out[split_idx['valid']], dataset.label.squeeze(1)[split_idx['valid']])

    return train_acc, valid_acc, test_acc, valid_loss, out
