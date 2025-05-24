#!/usr/bin/env python3

from math import pi
from networkx import barabasi_albert_graph
import torch as th
from torch import vmap

from tqdm import tqdm, trange
from joblib import Parallel, delayed, parallel_config

import cvxopt as cvx
import numpy as np
from cvxopt.solvers import coneqp

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import graph_construct as G
import simplecs as s
from csbm import csbm

from torch_geometric.data import Data

from torch.nn import Linear, ReLU, Softmax, Sigmoid, Parameter
from torch_geometric.nn import Sequential, GCNConv


path = Path("detect_pics/gcn").resolve()
if not path.is_dir():
    path.mkdir(parents=True)

"""
Here we always use the normalized laplacian.

We have the following steps:
1) Construct task: (Graph, Features, Labels, Regression Outputs)
    This will be stored in a pytorch geometric Data object
    Data(edge_index, x, y_label, y_cts, noise_levels). They may be batched,
    where the batch dimension will be first (unlike rest of GSP code).
    The batch dimension is labelled by noise_level.
2) Construct classification method (e.g. train GNN)
3) Construct regression method (e.g. train GNN)
4) a) Sample on graph
   b) Reconstruct Features
   c) Use methods to construct predicted labels & regression
   d) Return errors
5 (optional)) Return analytic errors
6) Plot

"""


# num_tasks = num_signals
# we use normalized laplacian
def mk_tasks_bl(
    graph_type="ER",
    num_nodes=300,
    num_tasks_per_noise_level=30,
    noise_levels=[0],
    signal_type="bl",
    # raw_signal_dist="normal",
    raw_signal_dist="sqrt",
):
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

    # make features & tasks
    bandwidth = n // 10
    U_k = s.restrict_eigenbasis(U, bandwidth)
    if signal_type == "bl":
        proj = s.calc_proj(U, bandwidth)
    elif signal_type == "lap":
        proj = U @ sqrteigs.diag() @ U.T
    else:
        raise ValueError("signal_type isn't bl or lap")

    # For this setup, we use the clean signal as the continous labels
    # y_cts has shape (num noise lvls, num tasks per noise lvl, num nodes)
    num_noise_levels = len(noise_levels)
    noise_levels = th.tensor(noise_levels)
    # y_cts = (proj @ th.randn(n, num_tasks_per_noise_level, num_noise_levels)).T
    if raw_signal_dist in ["normal", "gaussian"]:
        raw_signals = th.randn(num_noise_levels, num_tasks_per_noise_level, n)
    elif raw_signal_dist == "uniform":
        raw_signals = th.empty(num_noise_levels, num_tasks_per_noise_level, n).uniform_(
            -1, 1
        )
    elif raw_signal_dist == "sqrt":
        raw_signals = th.empty(num_noise_levels, num_tasks_per_noise_level, n).uniform_(
            -1, 1
        )
        raw_signals = raw_signals.abs().sqrt() * raw_signals.sign()

    y_cts = vmap(vmap(lambda v: proj @ v))(raw_signals)
    # y_cts = y_cts.refine_names("noise_levels", "tasks", "nodes")
    y_label = th.sign(y_cts)

    # noise is a tensor with shape (total_num_tasks, n)
    noise = vmap(
        lambda sigma: sigma * th.randn(num_tasks_per_noise_level, n),
        randomness="different",
    )(noise_levels)
    x = y_cts + noise
    return Data(
        edge_index=g,
        x=x,
        y_label=y_label,
        y_cts=y_cts,
        noise_levels=noise_levels,
        U_k=U_k,
        num_nodes=n,
        class_fn=th.sign,
    )


# We assume 1 noise level
def mk_tasks_gcn(
    feat_dim=128,
    hidden_dim=64,
    **kwargs,
):
    kwargs["num_tasks_per_noise_level"] = feat_dim
    data = mk_tasks_bl(**kwargs)
    model = Sequential(
        "x, edge_index",
        [
            (GCNConv(feat_dim, hidden_dim), "x, edge_index -> x"),
            ReLU(inplace=True),
            (GCNConv(hidden_dim, hidden_dim), "x, edge_index -> x"),
            ReLU(inplace=True),
            Linear(hidden_dim, 1),
        ],
    )

    intercept = model(data.y_cts[0].T, data.edge_index).squeeze().median().item()
    model.module_4.bias = Parameter(model.module_4.bias - intercept)
    data.y_label = model(data.y_cts[0].T, data.edge_index).sign()

    return data, model


def mk_tasks_csbm(
    num_nodes=300,
    feat_dim=1,
    num_tasks_per_noise_level=None,
    noise_levels=[0],
):

    data = csbm(num_nodes, feat_dim, num_nodes // 2, 0.7, 0.1, 100000000)

    U = s.calc_eigenbasis(
        data.edge_index, data.num_nodes, double=False, normalization="sym"
    )
    bandwidth = num_nodes // 10
    U_k = s.restrict_eigenbasis(U, bandwidth)

    data.x = data.x.T.unsqueeze(0)
    data.x_clean = data.x_clean.T.unsqueeze(0)
    if noise_levels == [0]:
        data.x = data.x_clean
    data.y_label = data.y.unsqueeze(0).unsqueeze(0)
    data.y_cts = data.x_clean
    data.U_k = U_k
    data.num_nodes = num_nodes
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


def sample_and_reconstruct_LS(data, sample_sizes, num_sample_sets):
    def inner(sample_size):
        sample_set = th.randperm(data.num_nodes)[:sample_size]
        # pois = th.eye(sample_size)
        sampled_x = data.x[:, :, sample_set]
        x_rec = th.stack(
            [
                # s.standard_decoder_multiple_signals_no_M(
                #     pois, sample_set, data.U_k, sigs.T
                # ).T
                rec_LS_mult_signals(sample_set, data.U_k, sigs.T).T
                for sigs in sampled_x
            ]
        )
        return x_rec

    return th.stack(
        [
            th.stack([inner(sample_size) for _ in range(num_sample_sets)])
            for sample_size in tqdm(sample_sizes)
        ]
    )
    # th.stack(res).refine_names("samples",noise_levels", "tasks", "nodes")


def sample_and_reconstruct_LS_vmap(data, sample_sizes, num_sample_sets):

    def inner(sample_set):
        sampled_x = data.x[:, :, sample_set]
        x_rec = th.stack(
            [rec_LS_mult_signals(sample_set, data.U_k, sigs.T).T for sigs in sampled_x]
        )
        return x_rec

    def inner_mult(sample_size):
        sample_sets = th.stack(
            [th.randperm(data.num_nodes)[:sample_size] for _ in range(num_sample_sets)]
        )
        return vmap(inner, randomness="different")(sample_sets)

    return th.stack([inner_mult(sample_size) for sample_size in tqdm(sample_sizes)])
    # th.stack(res).refine_names("samples",noise_levels", "tasks", "nodes")


def sample_and_reconstruct_LS_vmap_single(x, U_k, sample_size, num_sample_sets):
    num_nodes = U_k.shape[0]

    def inner(sample_set):
        sampled_x = x[:, :, sample_set]
        x_rec = th.stack(
            [rec_LS_mult_signals(sample_set, U_k, sigs.T).T for sigs in sampled_x]
        )
        return x_rec

    sample_sets = th.stack(
        [th.randperm(num_nodes)[:sample_size] for _ in range(num_sample_sets)]
    )
    return vmap(inner, randomness="different")(sample_sets)


def sample_and_reconstruct_dirichlet(data, sample_sizes, num_sample_sets):
    L = s.calc_laplacian(
        data.edge_index,
        num_nodes=data.num_nodes,
        normalization="sym",
    )

    def rec_dirichlet(x_obs, sample_set):
        M = (
            s.construct_sample_matrix(sample_set, data.num_nodes)
            .double()
            .to_dense()
            .numpy()
        )
        # M = cvx.spmatrix(1.0, np.arange(num_samples), np.array(sample_set), size=(num_samples, num_nodes))
        args = {
            "P": cvx.matrix(L.numpy()),
            "q": cvx.matrix(np.zeros(data.num_nodes)),
            "A": cvx.matrix(M),
            "b": cvx.matrix(x_obs.numpy().astype(np.double)),
        }
        # Note that the solution, x, to this has the property that
        # Lx is all zeros except at the sample, where it has the value
        # -1 * y, which is the negation of the slack variable.
        x_all = th.from_numpy(np.array(coneqp(**args)["x"])).squeeze(-1).float()
        if len(sample_set) == 10:
            breakpoint()
        return x_all

    def inner(sample_size):
        sample_set = th.randperm(data.num_nodes)[:sample_size]
        return th.stack(
            [
                th.stack([rec_dirichlet(sig, sample_set) for sig in sigs])
                for sigs in data.x[:, :, sample_set]
            ]
        )

    def inner_all(s):
        return th.stack([inner(s) for _ in range(num_sample_sets)])

    res = Parallel(n_jobs=-1, verbose=5)(delayed(inner_all)(s) for s in sample_sizes)
    # res = [inner_all(s) for s in sample_sizes]
    return th.stack(res)


def sample_and_reconstruct_dirichlet_vmap(data, sample_sizes, num_sample_sets):
    L = s.calc_laplacian(
        data.edge_index,
        num_nodes=data.num_nodes,
        normalization="sym",
    ).float()

    # x_obs has shape [sample_size, batch]
    # return [num_nodes, batch]
    # note that sample_set may not be ordered
    def rec_dirichlet(x_obs, sample_set):
        sample_mask_c = th.ones(data.num_nodes, dtype=bool)
        sample_mask_c[sample_set] = False

        C = L[sample_mask_c][:, sample_mask_c]
        BT = L[sample_mask_c][:, sample_set]
        x_u = -1 * th.linalg.lstsq(C, BT @ x_obs, driver="gelsd").solution

        result_shape = list(x_obs.shape)
        result_shape[0] = data.num_nodes
        result = th.zeros(result_shape)
        result[sample_set] = x_obs
        result[sample_mask_c] = x_u
        return result

    def inner(sample_size):
        sample_set = th.randperm(data.num_nodes)[:sample_size]
        return th.stack(
            [rec_dirichlet(sigs.T, sample_set).T for sigs in data.x[:, :, sample_set]]
        )

    def inner_all(s):
        return th.stack([inner(s) for _ in range(num_sample_sets)])

    res = Parallel(n_jobs=-1, verbose=5)(delayed(inner_all)(s) for s in sample_sizes)
    return th.stack(res)
    # return th.stack(
    #     [
    #         th.stack([inner(sample_size) for _ in range(num_sample_sets)])
    #         for sample_size in tqdm(sample_sizes)
    #     ]
    # )


def eval_errs(data, x_rec, sample_sizes, noise_levels):
    y_label_pred = th.sign(x_rec)
    y_cts_pred = x_rec
    class_error = vmap(vmap(lambda labs: 0.5 * th.abs(labs - data.y_label)))(
        y_label_pred
    )
    rec_error = vmap(vmap(lambda regr: th.square(regr - data.y_cts)))(y_cts_pred)

    # Aggregate errs:
    # sum over nodes
    class_error = class_error.sum(dim=-1)  # .mean(dim=3)
    rec_error = rec_error.sum(dim=-1)  # .norm(dim=-1).square()

    # We now have
    idx = pd.MultiIndex.from_product(
        [
            sample_sizes,
            np.arange(class_error.shape[1]),
            noise_levels,
            np.arange(class_error.shape[3]),
        ],
        # [np.arange(length) for length in class_error.shape[1:]],
        names=("Sample Size", "Sample idx", "Noise Level", "Noise Level idx"),
    )

    error_df = pd.DataFrame(
        {"class_error": class_error.flatten(), "rec_error": rec_error.flatten()},
        index=idx,
    )
    error_df = error_df.reset_index(level=[0, 1, 2, 3])
    return error_df


def eval_errs_gcn(gcn, data, x_rec, sample_sizes, noise_levels):
    fff = lambda x: gcn(x, data.edge_index).squeeze().sign()
    print("Running GCN on reconstructed data...")
    y_label_pred = th.stack([vmap(vmap(fff))(x) for x in tqdm(x_rec.transpose(-1, -2))])
    # [fff(x) for x in x_rec[0, 0]]
    # y_label_pred = data.model(data.y_cts[0].T, data.edge_index).sign()
    y_cts_pred = x_rec
    class_error = (y_label_pred - data.y_label.T.squeeze()).abs() * 0.5
    # sum over hidden dim
    rec_error = (y_cts_pred - data.y_cts).sum(dim=-2)

    # Aggregate errs:
    # sum over nodes
    class_error = class_error.sum(dim=-1).detach()  # .mean(dim=3)
    rec_error = rec_error.sum(dim=-1).detach()  # .norm(dim=-1).square()

    # We now have
    idx = pd.MultiIndex.from_product(
        [
            sample_sizes,
            np.arange(class_error.shape[1]),
            noise_levels,
            # np.arange(class_error.shape[3]),
        ],
        # [np.arange(length) for length in class_error.shape[1:]],
        names=("Sample Size", "Sample idx", "Noise Level"),
        # , "Noise Level idx"),
    )

    error_df = pd.DataFrame(
        {"class_error": class_error.flatten(), "rec_error": rec_error.flatten()},
        index=idx,
    )
    error_df = error_df.reset_index(level=[0, 1, 2])
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


def plot_errors(
    error_df, title="", filename="", bandlimit=None, indep_var="Sample Size"
):
    fig, axes = plt.subplots(1, 3, figsize=(24, 6), constrained_layout=True)

    # Plot 1: rec_error vs class_error
    sns.scatterplot(
        data=error_df,
        x="rec_error",
        y="class_error",
        hue=indep_var,
        palette="coolwarm",
        ax=axes[0],
    )
    sns.lineplot(
        data=error_df.groupby(indep_var).mean(),
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
        x=indep_var,
        y="rec_error",
        estimator="mean",
        errorbar=("ci", 95),
        linewidth=2,
        color="steelblue",
        ax=axes[1],
    )
    axes[1].set_title(f"Reconstruction Error vs. {indep_var}")

    # Plot 3: lineplot with CI for class_error vs sample size
    sns.lineplot(
        data=error_df,
        x=indep_var,
        y="class_error",
        estimator="mean",
        errorbar=("ci", 95),
        linewidth=2,
        color="indianred",
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


def simple_pipeline(
    num_graphs=4,
    graph_type="BA",
    signal_type="bl",
    rec_type="feat_prop",
    num_nodes=300,
    num_sigs=30,
    num_sample_sets=200,
    max_sample_size=None,
    bandlimit=None,
):
    if max_sample_size is None:
        max_sample_size = num_nodes

    def per_graph(i):
        # make graph
        tqdm.write("Making Graph...")
        noise_levels = [0]
        if graph_type == "CSBM":
            data = mk_tasks_csbm(
                num_nodes=num_nodes,
                feat_dim=1,
                num_tasks_per_noise_level=None,
                noise_levels=noise_levels,
            )
        else:
            data = mk_tasks_bl(
                graph_type=graph_type,
                num_nodes=num_nodes,
                num_tasks_per_noise_level=num_sigs,
                noise_levels=noise_levels,
                signal_type=signal_type,
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
        err_df = eval_errs(data, x_rec, sample_sizes.numpy(), noise_levels)
        err_df["graph_id"] = i
        return err_df

    total_df = pd.concat([per_graph(i) for i in trange(num_graphs)])

    plot_errors(
        total_df,
        title=f"{signal_type} signals + {rec_type} reconstruction + no noise",
        filename=path / f"{graph_type}_{signal_type}_{rec_type}_0_noise",
        bandlimit=bandlimit,
    )


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

    plot_errors(
        total_df,
        title=f"{signal_type} signals + {rec_type} reconstruction + no noise",
        filename=path / f"{graph_type}_{signal_type}_{rec_type}_0_noise",
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


def main_fn():
    num_nodes = 300
    for gtype in ["ER", "SBM", "BA"]:
        # for gtype in ["CSBM"]:
        # bandwidth_vs_laplacian(num_nodes, graph_type=gtype)
        stypes = ["default"] if gtype == "CSBM" else ["lap", "bl"]
        for stype in stypes:
            for rtype in ["LS", "feat_prop"]:
                print(
                    f"================== {gtype}, {stype}, {rtype} ================== "
                )
                bandlimit = (
                    None if (stype, rtype) == ("lap", "feat_prop") else num_nodes // 10
                )
                # simple_pipeline(
                #     num_graphs=4,
                #     signal_type=stype,
                #     rec_type=rtype,
                #     graph_type=gtype,
                #     num_nodes=num_nodes,
                #     bandlimit=bandlimit,
                #     max_sample_size=500,
                # )

                gcn_pipeline_untrained(
                    num_graphs=4,
                    signal_type=stype,
                    rec_type=rtype,
                    graph_type=gtype,
                    num_nodes=num_nodes,
                    bandlimit=bandlimit,
                    max_sample_size=num_nodes,
                )


if __name__ == "__main__":
    main_fn()
